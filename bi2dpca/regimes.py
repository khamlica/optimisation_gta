"""Identification des régimes de fonctionnement du GTA.

Aucun label de régime n'est fourni dans les données : on les apprend par
clustering (GMM) sur les variables de conduite stables `[HP, (MP), BP]`,
lissées dans le temps. EE est volontairement exclue (règle de la référence :
ne pas définir les régimes à partir de la variable surveillée).

Les labels sont ensuite lissés (vote majoritaire glissant) pour éviter le
papillotement, puis les pas situés à un changement de régime sont marqués
`transition` : ils seront exclus du fenêtrage, de l'entraînement et de l'alerte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config
from .config import Params
from .preprocessing import PreprocessResult


@dataclass
class RegimeResult:
    """Sortie de l'identification des régimes.

    Attributes
    ----------
    regime:
        Série entière (label de régime) alignée sur l'index ; ``-1`` aux pas
        non exploitables ou non labellisables.
    transition:
        Série booléenne : ``True`` aux pas appartenant à une transition de régime.
    n_regimes:
        Nombre de régimes retenus.
    model:
        GMM entraîné (réutilisable pour affecter un régime en ligne).
    scaler:
        Standardiseur des variables de régime (cohérent avec ``model``).
    regime_vars:
        Variables utilisées pour le clustering.
    bic:
        BIC du modèle retenu.
    """

    regime: pd.Series
    transition: pd.Series
    n_regimes: int
    model: GaussianMixture
    scaler: StandardScaler
    regime_vars: list[str]
    bic: float


def _regime_feature_frame(
    pre: PreprocessResult, params: Params
) -> tuple[pd.DataFrame, list[str]]:
    """Construit les features de régime : variables de conduite lissées.

    Utilise l'intersection entre ``REGIME_VARS`` et les variables présentes
    (donc `[HP, BP]` pour JFC1, `[HP, MP, BP]` pour JFC3), médiane glissante
    centrée pour atténuer le bruit court terme.
    """
    regime_vars = [v for v in config.REGIME_VARS if v in pre.variables]
    feats = (
        pre.df[regime_vars]
        .rolling(window=params.regime_smooth_window, min_periods=1, center=True)
        .median()
    )
    return feats, regime_vars


def _select_gmm(
    X: np.ndarray, params: Params
) -> tuple[GaussianMixture, int, float]:
    """Sélectionne le nombre de composantes par BIC sur la grille configurée."""
    best_model: GaussianMixture | None = None
    best_bic = np.inf
    best_k = 0
    for k in params.regime_n_components_grid:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=params.regime_random_state,
            n_init=2,
            reg_covar=1e-5,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic, best_model, best_k = bic, gmm, k
    assert best_model is not None
    return best_model, best_k, float(best_bic)


def _smooth_labels(labels: pd.Series, window: int) -> pd.Series:
    """Lissage majoritaire glissant des labels (anti-papillotement).

    À chaque pas, on retient le label le plus fréquent sur une fenêtre centrée.
    Les valeurs ``-1`` (non labellisé) sont ignorées dans le vote.
    """
    if window <= 1:
        return labels.copy()

    arr = labels.to_numpy()
    n = len(arr)
    half = window // 2
    out = arr.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = arr[lo:hi]
        seg = seg[seg >= 0]
        if seg.size == 0:
            continue
        vals, counts = np.unique(seg, return_counts=True)
        out[i] = int(vals[np.argmax(counts)])
    return pd.Series(out, index=labels.index)


def _transition_mask(regime: pd.Series, t: int) -> pd.Series:
    """Marque comme transition tout pas proche d'un changement de label.

    Une fenêtre 2D de longueur ``t`` ne doit jamais chevaucher deux régimes :
    on marque donc les ``t-1`` pas qui suivent un changement (la fenêtre se
    terminant sur ces pas serait à cheval), ainsi que les pas non labellisés.
    """
    arr = regime.to_numpy()
    n = len(arr)
    trans = np.zeros(n, dtype=bool)
    trans |= arr < 0
    change_points = np.flatnonzero(np.diff(arr) != 0) + 1  # premiers pas après un changement
    span = max(1, t - 1)
    for cp in change_points:
        trans[cp : min(n, cp + span)] = True
    return pd.Series(trans, index=regime.index)


def identify_regimes(
    pre: PreprocessResult, params: Params = config.DEFAULT_PARAMS
) -> RegimeResult:
    """Apprend les régimes par GMM et produit labels + masque de transition.

    Le clustering n'est entraîné que sur les pas exploitables ; les autres
    reçoivent le label ``-1`` puis sont traités comme transitions.
    """
    feats, regime_vars = _regime_feature_frame(pre, params)

    fit_mask = pre.exploitable & feats.notna().all(axis=1)
    X_fit = feats.loc[fit_mask].to_numpy()

    scaler = StandardScaler().fit(X_fit)
    model, n_regimes, bic = _select_gmm(scaler.transform(X_fit), params)

    # Affectation de tous les pas labellisables (features non NaN),
    # -1 ailleurs (non exploitable / NaN).
    regime = pd.Series(-1, index=pre.df.index, dtype=int)
    score_mask = feats.notna().all(axis=1)
    X_all = scaler.transform(feats.loc[score_mask].to_numpy())
    regime.loc[score_mask] = model.predict(X_all)
    # On force -1 sur les pas non exploitables.
    regime.loc[~pre.exploitable] = -1

    regime = _smooth_labels(regime, params.regime_label_smooth_window)
    transition = _transition_mask(regime, params.t)
    transition.name = "transition"
    regime.name = "regime"

    return RegimeResult(
        regime=regime,
        transition=transition,
        n_regimes=n_regimes,
        model=model,
        scaler=scaler,
        regime_vars=regime_vars,
        bic=bic,
    )
