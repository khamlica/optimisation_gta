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
from .preprocessing import PreprocessResult, stop_mask


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


def _elbow_k(grid: list[int], bics: dict[int, float], frac: float) -> int:
    """Plus petit ``k`` au « coude » de la courbe BIC (parcimonie).

    On ajoute des régimes tant que le gain marginal de BIC reste >= ``frac`` fois
    le gain du tout premier ajout ; dès qu'il passe en dessous, on s'arrête. Si
    le BIC ne s'améliore jamais (premier gain <= 0), on prend le plus petit k.
    """
    gains = {grid[i]: bics[grid[i - 1]] - bics[grid[i]] for i in range(1, len(grid))}
    first_gain = gains[grid[1]]
    if first_gain <= 0:
        return grid[0]
    threshold = frac * first_gain
    chosen = grid[0]
    for i in range(1, len(grid)):
        if gains[grid[i]] >= threshold:
            chosen = grid[i]
        else:
            break
    return chosen


def _select_gmm(
    X: np.ndarray, params: Params
) -> tuple[GaussianMixture, int, float]:
    """Sélectionne le nombre de régimes par BIC décorrélé + coude de parcimonie.

    Chaque GMM est ajusté sur **toutes** les données ``X`` ; le BIC servant à la
    sélection est en revanche évalué sur un sous-échantillon **décorrélé**
    (1 point sur ``regime_bic_thin``) pour neutraliser l'autocorrélation qui,
    sinon, pousse systématiquement vers le maximum de la grille.
    """
    grid = sorted(params.regime_n_components_grid)
    thin = max(1, params.regime_bic_thin)
    # Garde-fou : assez de points décorrélés pour des BIC stables ; sinon, BIC
    # sur toutes les données (cas de jeux courts).
    X_sel = X[::thin] if X.shape[0] // thin >= 2 * grid[-1] else X

    models: dict[int, GaussianMixture] = {}
    bics: dict[int, float] = {}
    for k in grid:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=params.regime_random_state,
            n_init=2,
            reg_covar=1e-5,
        )
        gmm.fit(X)
        models[k] = gmm
        bics[k] = float(gmm.bic(X_sel))

    best_k = _elbow_k(grid, bics, params.regime_bic_elbow_frac)
    return models[best_k], best_k, bics[best_k]


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


def _transition_mask(regime: pd.Series, span: int) -> pd.Series:
    """Marque comme transition les ``span`` premiers pas de chaque nouveau régime.

    Le non-chevauchement des fenêtres 2D entre deux régimes est déjà garanti par
    ``enumerate_windows`` (qui s'arrête à chaque changement de label) : cette
    marge ne sert donc qu'à écarter la **rampe de stabilisation** du nouveau
    régime, dont la dynamique n'est pas encore représentative de l'état établi.
    Les pas non labellisés (``-1`` : non exploitables ou à l'arrêt) sont aussi
    marqués transition.
    """
    arr = regime.to_numpy()
    n = len(arr)
    trans = np.zeros(n, dtype=bool)
    trans |= arr < 0
    change_points = np.flatnonzero(np.diff(arr) != 0) + 1  # premiers pas après un changement
    span = max(1, span)
    for cp in change_points:
        trans[cp : min(n, cp + span)] = True
    return pd.Series(trans, index=regime.index)


def assign_regimes(
    pre: PreprocessResult,
    model: GaussianMixture,
    scaler: StandardScaler,
    regime_vars: list[str],
    params: Params = config.DEFAULT_PARAMS,
) -> tuple[pd.Series, pd.Series]:
    """Affecte des labels de régime à partir d'un GMM **déjà entraîné**.

    Utilisé en ligne (et en interne par ``identify_regimes``) pour garantir une
    affectation identique entre offline et online. Renvoie ``(regime, transition)``.
    """
    feats = (
        pre.df[regime_vars]
        .rolling(window=params.regime_smooth_window, min_periods=1, center=True)
        .median()
    )

    regime = pd.Series(-1, index=pre.df.index, dtype=int)
    score_mask = feats.notna().all(axis=1)
    if score_mask.any():
        X_all = scaler.transform(feats.loc[score_mask].to_numpy())
        regime.loc[score_mask] = model.predict(X_all)

    # Non labellisables : pas non exploitables OU à l'arrêt. L'arrêt est exclu
    # des régimes (sinon il forme un faux régime « machine éteinte »).
    not_labelable = (~pre.exploitable) | stop_mask(pre, params)
    regime.loc[not_labelable] = -1
    regime = _smooth_labels(regime, params.regime_label_smooth_window)
    # Re-forcer -1 après lissage : le vote majoritaire ne doit jamais
    # « inventer » un régime sur un pas non exploitable ou à l'arrêt.
    regime.loc[not_labelable] = -1
    regime.name = "regime"

    transition = _transition_mask(regime, params.transition_steps)
    transition.name = "transition"
    return regime, transition


def identify_regimes(
    pre: PreprocessResult, params: Params = config.DEFAULT_PARAMS
) -> RegimeResult:
    """Apprend les régimes par GMM et produit labels + masque de transition.

    Le clustering n'est entraîné que sur les pas exploitables ; l'affectation
    finale (y compris lissage et transitions) est déléguée à ``assign_regimes``.
    """
    feats, regime_vars = _regime_feature_frame(pre, params)

    # Fit GMM uniquement sur les pas exploitables, HORS arrêt : l'arrêt n'est
    # pas un régime de fonctionnement et fausserait les clusters/populations.
    fit_mask = pre.exploitable & (~stop_mask(pre, params)) & feats.notna().all(axis=1)
    X_fit = feats.loc[fit_mask].to_numpy()

    # Garde-fou : refuser d'apprendre des régimes sur trop peu de points.
    max_k = max(params.regime_n_components_grid)
    min_required = max(params.regime_min_fit_points, max_k)
    if X_fit.shape[0] < min_required:
        raise ValueError(
            f"Trop peu de points exploitables pour le clustering des régimes : "
            f"{X_fit.shape[0]} < {min_required} requis "
            f"(max composantes={max_k}, regime_min_fit_points={params.regime_min_fit_points}). "
            f"Vérifier le préfiltrage / la quantité de données."
        )

    scaler = StandardScaler().fit(X_fit)
    model, n_regimes, bic = _select_gmm(scaler.transform(X_fit), params)

    regime, transition = assign_regimes(pre, model, scaler, regime_vars, params)

    return RegimeResult(
        regime=regime,
        transition=transition,
        n_regimes=n_regimes,
        model=model,
        scaler=scaler,
        regime_vars=regime_vars,
        bic=bic,
    )
