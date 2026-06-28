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
    filled: pd.Series


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


def _fill_short_gaps(
    df: pd.DataFrame,
    variables: list[str],
    max_steps: int,
    jump_k: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Comble par interpolation linéaire les trous **intérieurs** de longueur
    <= ``max_steps``, uniquement si le saut aux deux bords reste petit.

    Interpolation linéaire (et non médiane/moyenne) : elle préserve la pente
    locale et n'introduit ni plateau artificiel ni dérive. Le garde-fou de saut
    évite de « ponter » une vraie discontinuité (où le trou cacherait un
    évènement). Renvoie ``(df_comblé, filled)`` où ``filled`` marque les points
    effectivement reconstruits.
    """
    filled = pd.Series(False, index=df.index)
    if max_steps <= 0:
        return df, filled

    out = df.copy()
    missing = df.isna().any(axis=1).to_numpy()
    # Échelle « pas-type » par variable (médiane des |différences premières|).
    scale = {
        v: max(float(df[v].diff().abs().median()), 1e-9) for v in variables
    }
    arr = {v: out[v].to_numpy() for v in variables}
    n = len(df)
    i = 0
    while i < n:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j < n and missing[j]:
            j += 1
        L = j - i  # longueur du trou [i, j)
        # Trou intérieur, court, avec des bords présents des deux côtés.
        if 0 < L <= max_steps and i - 1 >= 0 and j < n:
            ok = True
            for v in variables:
                before, after = arr[v][i - 1], arr[v][j]
                if not (np.isfinite(before) and np.isfinite(after)):
                    ok = False
                    break
                if abs(after - before) > jump_k * scale[v] * (L + 1):
                    ok = False  # saut trop grand -> vraie discontinuité
                    break
            if ok:
                for v in variables:
                    before, after = arr[v][i - 1], arr[v][j]
                    for k in range(1, L + 1):
                        out.iloc[i - 1 + k, out.columns.get_loc(v)] = (
                            before + (after - before) * k / (L + 1)
                        )
                filled.iloc[i:j] = True
        i = j
    return out, filled


def preprocess(data: GtaData, params: Params = config.DEFAULT_PARAMS) -> PreprocessResult:
    """Applique le préfiltrage grossier et construit le masque ``exploitable``.

    Étapes (cf. ref/model_reference.md, « Préfiltrage grossier ») :
    tri + déduplication, grille régulière, bornes physiques, trous longs,
    capteurs bloqués. Les arrêts/démarrages/transitions de régime ne sont pas
    traités ici : ils relèvent de l'étape régimes (transitions) et du jeu sain.
    """
    variables = data.variables
    df = _regular_grid(data.df, params.dt_minutes)

    # Comblage des trous courts (interpolation linéaire bornée) AVANT les masques :
    # un point isolé manquant ne doit pas coûter ~t fenêtres.
    df, filled = _fill_short_gaps(
        df, variables, params.max_fill_steps, params.fill_max_jump_k
    )
    filled.name = "filled"

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
        "n_filled": int(filled.sum()),
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
        filled=filled,
    )


def stop_mask(
    pre: PreprocessResult, params: Params = config.DEFAULT_PARAMS
) -> pd.Series:
    """``True`` aux pas où la machine est à l'arrêt (charge quasi nulle).

    Le seuil est relatif à la médiane opérationnelle de chaque variable de
    charge (par défaut ``EE``), estimée sur les pas exploitables non nuls.

    L'arrêt est un état « machine non surveillée » : il est exclu du fenêtrage,
    du jeu sain ET du clustering des régimes (sinon il forme un faux régime).
    """
    stop = pd.Series(False, index=pre.df.index)
    for v in params.stop_vars:
        if v not in pre.variables:
            continue
        s = pre.df[v]
        operating = s[pre.exploitable & (s > 0)]
        if operating.empty:
            continue
        thr = params.stop_frac * float(operating.median())
        stop |= s < thr  # points valides sous le seuil

        # Trous (données manquantes) ENCADRÉS par de l'arrêt : la machine était
        # éteinte pendant le trou de logging -> arrêt, pas « trou ». On exige les
        # deux bords sous le seuil pour rester conservateur.
        arr = s.to_numpy()
        miss = np.isnan(arr)
        ext = np.zeros(len(arr), dtype=bool)
        n = len(arr)
        i = 0
        while i < n:
            if miss[i]:
                j = i
                while j < n and miss[j]:
                    j += 1
                before = arr[i - 1] if i - 1 >= 0 else np.nan
                after = arr[j] if j < n else np.nan
                if (np.isfinite(before) and before < thr) and (
                    np.isfinite(after) and after < thr
                ):
                    ext[i:j] = True
                i = j
            else:
                i += 1
        stop |= pd.Series(ext, index=pre.df.index)
    stop.name = "stop"
    return stop
