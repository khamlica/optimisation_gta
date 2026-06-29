"""Traceur de dynamique locale INVARIANT AU NIVEAU (couche 2a).

Troisième couche du triptyque, distincte du Bi2DPCA (écart à la baseline du
régime) et du cross-check énergétique (performance statique). On retire ce qui
relève du niveau / du bilan énergétique, puis on suit ce qui reste : volatilité
court-terme, autocorrélation lag-1 et couplages d'incréments entre variables.

But : répondre à « reste-t-il un signal *dynamique local* après retrait du
niveau ? ». Pour JFC1/JFC3, on s'attend à ce que ce soit **plat** (la dérive y
est statique). Sur d'autres GTA, une excursion révélerait la vraie zone de
pertinence d'un Bi2DPCA appliqué sur résidus/différences (couche 2b).

⚠️ INDICATIF UNIQUEMENT. La « zone de référence » (médiane ± k·IQR sur la
période saine) est un repère visuel, **pas une limite de contrôle** : on n'a ni
modèle statistique de dépendance de ces features, ni étude ARL/FAR sous
autocorrélation. Aucun statut warning/alert n'en est dérivé, rien n'est branché
sur la décision V1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, energetic
from .config import Params
from .energetic import EnergyModel
from .preprocessing import PreprocessResult


@dataclass
class DynamicReference:
    """Zone de référence (repère visuel) par feature, sur la période saine."""

    features: list[str]
    band: dict[str, dict[str, float]]  # feature -> {med, lo, hi}
    ref_end: str | None
    roll_window: int
    n_ref: int


def _roll_mad(s: pd.Series, w: int, mp: int) -> pd.Series:
    """MAD glissante (dispersion robuste) d'une série."""
    return s.rolling(w, min_periods=mp).apply(
        lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))), raw=True
    )


def dynamic_feature_frame(
    pre: PreprocessResult,
    energy_model: EnergyModel,
    params: Params = config.DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Features dynamiques invariantes au niveau sur les points opérationnels.

    Différences premières + résidu énergétique, puis fenêtres glissantes :
    volatilité (MAD), autocorrélation lag-1 du résidu, couplages d'incréments.
    """
    rf = energetic.residual_frame(pre, energy_model, params)
    if rf.empty:
        return pd.DataFrame()
    df = pre.df.loc[rf.index]  # mêmes points opérationnels

    # Différences entre points opérationnels consécutifs (level-invariant).
    d_hp = df["HP"].diff() if "HP" in df else None
    d_bp = df["BP"].diff() if "BP" in df else None
    d_ee = df["EE"].diff() if "EE" in df else None
    resid = rf["resid"]
    d_resid = resid.diff()

    w = params.dynamic_roll_window
    mp = max(8, w // 4)
    out = pd.DataFrame(index=rf.index)
    if d_hp is not None:
        out["vol_dHP"] = _roll_mad(d_hp, w, mp)
    if d_bp is not None:
        out["vol_dBP"] = _roll_mad(d_bp, w, mp)
    if d_ee is not None:
        out["vol_dEE"] = _roll_mad(d_ee, w, mp)
    out["vol_resid"] = _roll_mad(d_resid, w, mp)
    out["ac1_resid"] = resid.rolling(w, min_periods=mp).apply(
        lambda x: pd.Series(x).autocorr(1), raw=False
    )
    if d_hp is not None and d_ee is not None:
        out["coup_dHP_dEE"] = d_hp.rolling(w, min_periods=mp).corr(d_ee)
    if d_bp is not None and d_ee is not None:
        out["coup_dBP_dEE"] = d_bp.rolling(w, min_periods=mp).corr(d_ee)
    if d_hp is not None:
        out["coup_dHP_dresid"] = d_hp.rolling(w, min_periods=mp).corr(d_resid)
    return out


def fit_dynamic_reference(
    pre: PreprocessResult,
    energy_model: EnergyModel,
    params: Params = config.DEFAULT_PARAMS,
    ref_end: str | None = None,
) -> tuple[DynamicReference, pd.DataFrame] | None:
    """Calcule les features + la zone de référence (médiane ± k·IQR) sur le sain.

    Renvoie ``(référence, features)`` ; ``None`` si features indisponibles.
    """
    feats = dynamic_feature_frame(pre, energy_model, params)
    if feats.empty:
        return None
    ref = feats.loc[:ref_end] if ref_end is not None else feats.iloc[:0]
    if len(ref.dropna(how="all")) < max(20, params.dynamic_roll_window):
        ref = feats.iloc[: max(1, int(len(feats) * params.energetic_ref_frac))]
        ref_end = None

    k = params.dynamic_band_k
    band: dict[str, dict[str, float]] = {}
    for c in feats.columns:
        s = ref[c].dropna()
        if s.empty:
            band[c] = {"med": float("nan"), "lo": float("nan"), "hi": float("nan")}
            continue
        med = float(s.median())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        band[c] = {"med": med, "lo": med - k * iqr, "hi": med + k * iqr}

    reference = DynamicReference(
        features=list(feats.columns),
        band=band,
        ref_end=str(ref_end) if ref_end else None,
        roll_window=params.dynamic_roll_window,
        n_ref=int(len(ref)),
    )
    return reference, feats
