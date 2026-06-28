"""Construction des fenêtres 2D `A ∈ R^(t×m)`.

Une fenêtre est un bloc de ``t`` pas consécutifs de la grille régulière, sur les
``m`` variables canoniques. Une fenêtre n'est retenue que si **tous** ses pas
sont surveillables (exploitables, hors transition, hors arrêt) et appartiennent
au **même** régime — règle absolue de la référence (« ne jamais comparer des
fenêtres de régimes différents »).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .config import Params
from .preprocessing import PreprocessResult, stop_mask
from .regimes import RegimeResult

# ``stop_mask`` est défini dans ``preprocessing`` (état de la donnée, sans
# dépendance aux régimes) et ré-exporté ici pour compatibilité ascendante.
__all__ = ["stop_mask", "monitorable_mask", "enumerate_windows", "extract_windows"]


@dataclass
class WindowIndex:
    """Index des fenêtres valides : positions de départ et régime associé."""

    starts: np.ndarray  # positions entières de début dans la grille régulière
    regime: np.ndarray  # label de régime de chaque fenêtre
    t: int

    def __len__(self) -> int:
        return int(self.starts.size)

    def for_regime(self, regime_id: int) -> np.ndarray:
        """Positions de départ des fenêtres d'un régime, triées chronologiquement."""
        sel = self.starts[self.regime == regime_id]
        return np.sort(sel)


def monitorable_mask(
    pre: PreprocessResult,
    reg: RegimeResult,
    params: Params = config.DEFAULT_PARAMS,
) -> pd.Series:
    """Pas réellement surveillables : exploitable & hors transition & hors arrêt."""
    mask = pre.exploitable & (~reg.transition) & (~stop_mask(pre, params))
    mask &= reg.regime >= 0
    mask.name = "monitorable"
    return mask


def enumerate_windows(
    regime: pd.Series,
    monitorable: pd.Series,
    t: int,
    stride: int,
) -> WindowIndex:
    """Énumère les fenêtres valides en respectant l'overlap (stride).

    On parcourt les segments consécutifs de pas surveillables appartenant au
    même régime, et on y place des fenêtres de longueur ``t`` tous les
    ``stride`` pas. Toute fenêtre à cheval sur un trou, une transition ou un
    changement de régime est ainsi automatiquement exclue.
    """
    reg = regime.to_numpy()
    mon = monitorable.to_numpy()
    n = len(mon)
    starts: list[int] = []
    regs: list[int] = []

    i = 0
    while i < n:
        if not mon[i]:
            i += 1
            continue
        r = reg[i]
        j = i
        while j < n and mon[j] and reg[j] == r:
            j += 1
        # Segment valide [i, j) du régime r : fenêtres glissantes de longueur t.
        last_start = j - t
        s = i
        while s <= last_start:
            starts.append(s)
            regs.append(int(r))
            s += stride
        i = j

    return WindowIndex(
        starts=np.asarray(starts, dtype=int),
        regime=np.asarray(regs, dtype=int),
        t=t,
    )


def extract_windows(values: np.ndarray, starts: np.ndarray, t: int) -> np.ndarray:
    """Empile les fenêtres ``(N, t, m)`` à partir d'une matrice ``(n, m)``.

    ``values`` peut être brut ou déjà standardisé ; l'extraction est identique.
    """
    if starts.size == 0:
        m = values.shape[1]
        return np.empty((0, t, m), dtype=float)
    idx = starts[:, None] + np.arange(t)[None, :]  # (N, t)
    return values[idx]  # (N, t, m)
