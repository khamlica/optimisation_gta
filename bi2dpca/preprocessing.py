"""Préfiltrage grossier des données GTA, avant clustering des régimes.

L'objectif est de produire un masque ``exploitable`` (une valeur booléenne par
pas de temps) qui écarte les points manifestement inutilisables — trous longs,
capteurs bloqués, valeurs hors plages physiques — sans encore définir le jeu
« sain » final (qui sera affiné par régime dans ``healthy.py``).

Le pas de temps est rendu régulier (réindexation au pas nominal) afin que la
construction des fenêtres 2D repose sur une grille temporelle stricte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .config import Params
from .io_data import GtaData


@dataclass
class PreprocessResult:
    """Sortie du préfiltrage.

    Attributes
    ----------
    df:
        DataFrame réindexé sur la grille régulière (pas = ``dt_minutes``).
        Les variables canoniques peuvent contenir des ``NaN`` aux pas insérés.
    exploitable:
        Série booléenne alignée sur ``df.index`` : ``True`` = point utilisable.
    variables:
        Variables canoniques présentes.
    ranges:
        Bornes physiques effectivement utilisées par variable.
    report:
        Compteurs de diagnostic (doublons, trous, hors-plage, bloqués).
    """

    df: pd.DataFrame
    exploitable: pd.Series
    variables: list[str]
    ranges: dict[str, tuple[float, float]]
    report: dict[str, int]


def _regular_grid(df: pd.DataFrame, dt_minutes: int) -> pd.DataFrame:
    """Trie, déduplique et réindexe sur une grille temporelle régulière.

    Les doublons d'horodatage sont agrégés par moyenne ; les pas manquants
    sont insérés (valeurs ``NaN``) pour obtenir un index strictement régulier.
    """
    df = df.sort_index()
    # Déduplication : moyenne des éventuels doublons exacts d'horodatage.
    if df.index.has_duplicates:
        df = df.groupby(level=0).mean()
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{dt_minutes}min",
    )
    return df.reindex(full_index)


def _auto_ranges(
    df: pd.DataFrame, variables: list[str]
) -> dict[str, tuple[float, float]]:
    """Bornes physiques dérivées des données par quantiles robustes élargis.

    On part de l'intervalle interquartile et on l'élargit largement (k=5·IQR)
    afin de ne couper que les valeurs grossièrement aberrantes ; ce n'est pas
    un nettoyage fin (réservé au jeu sain par régime).
    """
    ranges: dict[str, tuple[float, float]] = {}
    for v in variables:
        s = df[v].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo = float(q1 - 5.0 * iqr)
        hi = float(q3 + 5.0 * iqr)
        if not np.isfinite(iqr) or iqr <= 0:  # variable quasi constante
            lo, hi = float(s.min()), float(s.max())
        ranges[v] = (lo, hi)
    return ranges


def _out_of_range_mask(
    df: pd.DataFrame,
    variables: list[str],
    ranges: dict[str, tuple[float, float]],
) -> pd.Series:
    """``True`` là où au moins une variable sort de ses bornes physiques."""
    out = pd.Series(False, index=df.index)
    for v in variables:
        lo, hi = ranges[v]
        out |= (df[v] < lo) | (df[v] > hi)
    return out


def _long_gap_mask(df: pd.DataFrame, max_gap_steps: int) -> pd.Series:
    """``True`` sur les pas appartenant à un trou de plus de ``max_gap_steps``.

    Un « trou » est une suite consécutive de pas où une variable au moins est
    manquante. Les trous courts (<= max_gap_steps) restent exploitables et
    seront éventuellement interpolés/écartés au niveau fenêtre.
    """
    missing = df.isna().any(axis=1).to_numpy()
    is_long = np.zeros(len(missing), dtype=bool)
    i = 0
    n = len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            if (j - i) > max_gap_steps:
                is_long[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(is_long, index=df.index)


def _stuck_sensor_mask(
    df: pd.DataFrame,
    variables: list[str],
    window: int,
    min_std: float,
) -> pd.Series:
    """``True`` là où une variable est « bloquée » (variance glissante ~nulle).

    Détecte un capteur figé sur une valeur constante pendant ``window`` pas.
    """
    stuck = pd.Series(False, index=df.index)
    for v in variables:
        rolling_std = df[v].rolling(window=window, min_periods=window).std()
        stuck |= rolling_std.fillna(np.inf) < min_std
    return stuck


def preprocess(data: GtaData, params: Params = config.DEFAULT_PARAMS) -> PreprocessResult:
    """Applique le préfiltrage grossier et construit le masque ``exploitable``.

    Étapes (cf. ref/model_reference.md, « Préfiltrage grossier ») :
    tri + déduplication, grille régulière, bornes physiques, trous longs,
    capteurs bloqués. Les arrêts/démarrages/transitions de régime ne sont pas
    traités ici : ils relèvent de l'étape régimes (transitions) et du jeu sain.
    """
    variables = data.variables
    df = _regular_grid(data.df, params.dt_minutes)

    ranges = params.physical_ranges or _auto_ranges(df, variables)

    missing = df.isna().any(axis=1)
    out_of_range = _out_of_range_mask(df, variables, ranges)
    long_gap = _long_gap_mask(df, params.long_gap_steps)
    stuck = _stuck_sensor_mask(df, variables, params.stuck_window, params.stuck_min_std)

    exploitable = ~(missing | out_of_range | long_gap | stuck)
    exploitable.name = "exploitable"

    report = {
        "n_points": int(len(df)),
        "n_missing": int(missing.sum()),
        "n_out_of_range": int(out_of_range.sum()),
        "n_long_gap": int(long_gap.sum()),
        "n_stuck": int(stuck.sum()),
        "n_exploitable": int(exploitable.sum()),
    }

    return PreprocessResult(
        df=df,
        exploitable=exploitable,
        variables=variables,
        ranges=ranges,
        report=report,
    )
