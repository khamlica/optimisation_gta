"""Construction du jeu sain et split temporel par régime.

Aucun label sain n'étant disponible, on part d'un **jeu sain candidat** = les
fenêtres surveillables (déjà débarrassées des arrêts et transitions par
``windows.monitorable_mask``), puis on le **nettoie par quantiles robustes** sur
les scores de reconstruction du régime (médiane + k·IQR). Le nettoyage itératif
final est piloté par ``model.py`` ; ce module fournit les briques : split
temporel strict et masque de rejet robuste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .config import Params


@dataclass
class TemporalSplit:
    """Découpage chronologique des fenêtres d'un régime (sans mélange aléatoire)."""

    train: np.ndarray  # positions de départ (triées)
    calib: np.ndarray
    test: np.ndarray


def time_split(starts: np.ndarray, params: Params = config.DEFAULT_PARAMS) -> TemporalSplit:
    """Split temporel 70/15/15 (par défaut) sur des fenêtres triées dans le temps.

    Les fenêtres les plus anciennes vont au train, les suivantes à la
    calibration, le reste au test (où l'on conserve d'éventuels événements).
    """
    starts = np.sort(starts)
    n = starts.size
    n_train = int(np.floor(n * params.train_frac))
    n_calib = int(np.floor(n * params.calib_frac))
    train = starts[:n_train]
    calib = starts[n_train : n_train + n_calib]
    test = starts[n_train + n_calib :]
    return TemporalSplit(train=train, calib=calib, test=test)


def robust_keep_mask(scores: np.ndarray, k: float) -> np.ndarray:
    """Masque ``True`` = fenêtre conservée (score <= médiane + k·IQR).

    Critère robuste unilatéral : on ne rejette que les scores anormalement
    **hauts** (anomalies de reconstruction), pas les scores bas.
    """
    if scores.size == 0:
        return np.zeros(0, dtype=bool)
    med = float(np.median(scores))
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = float(q3 - q1)
    upper = med + k * iqr
    if iqr <= 0:  # scores quasi constants : rien à rejeter
        return np.ones_like(scores, dtype=bool)
    return scores <= upper
