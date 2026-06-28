"""Cross-check énergétique (V2) — indicateur global ``EE = f(HP, MP, BP)``.

Indépendant des régimes Bi2DPCA. On apprend la relation entre l'électricité
produite et les débits vapeur (HP, MP si présent, BP) sur une période de
référence **ancienne et saine**, puis on suit le **résidu** ``EE_réel −
EE_attendu`` dans le temps :

* résidu négatif persistant → on produit **moins** d'EE qu'attendu = perte de
  rendement (dégradation) ;
* résidu positif → **plus** d'EE qu'attendu = gain (p.ex. après maintenance).

C'est le pendant **interprétable et non masquable** du score spatial : une
dérive de rendement qui s'auto-masquerait dans un nouveau régime Bi2DPCA reste
ici visible, car le modèle énergétique est global. C'est précisément le « biais
persistant » qu'un filtre de Kalman mesurait.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .config import Params
from .preprocessing import PreprocessResult, stop_mask

# Débits vapeur candidats comme entrées du bilan (EE est la sortie surveillée).
STEAM_INPUTS: tuple[str, ...] = ("HP", "MP", "BP")


@dataclass
class EnergyModel:
    """Modèle linéaire ``EE ≈ Σ coef·entrée + intercept`` + bande de référence."""

    inputs: list[str]
    coef: np.ndarray       # poids des entrées puis intercept (taille len+1)
    ref_std: float         # écart-type du résidu (%) sur la référence
    ref_end: str | None    # fin de la référence (date) ou None (repli fraction)
    n_ref: int


def _operating(pre: PreprocessResult, params: Params):
    """Masque des points exploitables hors arrêt avec toutes les entrées + EE."""
    inputs = [v for v in STEAM_INPUTS if v in pre.variables]
    op = pre.exploitable & (~stop_mask(pre, params))
    op &= pre.df[inputs + ["EE"]].notna().all(axis=1)
    return inputs, op


def fit_energy_model(
    pre: PreprocessResult,
    params: Params = config.DEFAULT_PARAMS,
    ref_end: str | None = None,
) -> EnergyModel | None:
    """Ajuste ``EE = f(entrées)`` par moindres carrés sur la référence saine.

    ``ref_end`` : date (incluse) de fin de référence ; si ``None`` ou si la
    référence est trop courte, on retombe sur les ``energetic_ref_frac``
    premiers points opérationnels.
    """
    if "EE" not in pre.variables:
        return None
    inputs, op = _operating(pre, params)
    sub = pre.df[op]
    if sub.empty:
        return None

    ref = sub.loc[:ref_end] if ref_end is not None else sub.iloc[:0]
    min_ref = max(10, 2 * (len(inputs) + 1))
    if len(ref) < min_ref:  # repli : pas (assez) de référence datée
        ref = sub.iloc[: max(min_ref, int(len(sub) * params.energetic_ref_frac))]
        ref_end = None
    if len(ref) < min_ref:
        return None

    X1 = np.column_stack([ref[inputs].to_numpy(), np.ones(len(ref))])
    y = ref["EE"].to_numpy()
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    with np.errstate(divide="ignore", invalid="ignore"):
        resid_pct = 100.0 * (y - pred) / pred
    resid_pct = resid_pct[np.isfinite(resid_pct)]
    ref_std = float(np.std(resid_pct)) if resid_pct.size else float("nan")
    return EnergyModel(
        inputs=inputs, coef=coef, ref_std=ref_std,
        ref_end=str(ref_end) if ref_end else None, n_ref=len(ref),
    )


def residual_frame(
    pre: PreprocessResult, model: EnergyModel, params: Params = config.DEFAULT_PARAMS
) -> pd.DataFrame:
    """Résidu énergétique sur tous les points opérationnels (index temporel)."""
    _, op = _operating(pre, params)
    sub = pre.df[op]
    X1 = np.column_stack([sub[model.inputs].to_numpy(), np.ones(len(sub))])
    pred = X1 @ model.coef
    ee = sub["EE"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        resid_pct = 100.0 * (ee - pred) / pred
    return pd.DataFrame(
        {"EE": ee, "EE_pred": pred, "resid": ee - pred, "resid_pct": resid_pct},
        index=sub.index,
    )
