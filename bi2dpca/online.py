"""Scoring en ligne et logique de décision.

Reproduit le pseudocode de ref/model_reference.md : une fenêtre est d'abord
classée `transition` ou `unknown_regime` le cas échéant ; sinon elle est
standardisée avec les stats de son régime, scorée, comparée aux seuils, et la
décision `normal` / `warning` / `alert` est prise via une logique de
**persistance** (ratio de dépassement sur un horizon glissant).

EE n'est jamais prédite : la décision repose uniquement sur Q_time / Q_space
(et T2_* si activés). Tout cross-check énergétique éventuel relève d'une V2
séparée et non bloquante.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import config
from .config import Params
from .model import RegimeModel


def build_reason_codes(
    scores: dict[str, float],
    model: RegimeModel,
) -> list[str]:
    """Tags expliquant un dépassement (un par indice actif au-dessus du seuil)."""
    codes: list[str] = []
    for idx in model.active_indices:
        if scores[idx] > model.thresholds[idx]:
            codes.append(f"{idx}_high")
    return codes


def window_exceeds(scores: dict[str, float], model: RegimeModel) -> bool:
    """``True`` si au moins un indice actif dépasse son seuil de régime."""
    return any(scores[idx] > model.thresholds[idx] for idx in model.active_indices)


class PersistenceState:
    """Mémorise les dépassements récents pour la règle d'alerte.

    On conserve une fenêtre glissante des derniers ``window_flag`` (un par pas
    de scoring) et l'on déclenche ``alert`` quand la proportion de dépassements
    sur l'horizon de persistance atteint ``exceed_ratio``.
    """

    def __init__(self, params: Params = config.DEFAULT_PARAMS, stride_minutes: int | None = None):
        self.params = params
        stride_minutes = stride_minutes if stride_minutes is not None else params.dt_minutes
        self.n_persist = max(1, math.ceil(params.persistence_minutes / stride_minutes))
        self._buf: deque[bool] = deque(maxlen=self.n_persist)
        self._regime: int | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._regime = None

    def update(self, exceed: bool, regime: int) -> None:
        """Ajoute un dépassement ; vide l'historique si le régime change."""
        if regime != self._regime:
            self._buf.clear()
            self._regime = regime
        self._buf.append(bool(exceed))

    def exceed_ratio(self) -> float:
        """Proportion de dépassements sur l'historique courant."""
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)

    def alert_ready(self) -> bool:
        """``True`` si l'historique est plein et le ratio atteint le seuil."""
        return (
            len(self._buf) >= self.n_persist
            and self.exceed_ratio() >= self.params.exceed_ratio
        )


@dataclass
class WindowResult:
    """Résultat d'évaluation d'une fenêtre en ligne."""

    status: str                 # normal / warning / alert / transition / unknown_regime
    regime: int
    scores: dict[str, float]
    window_flag: bool
    reason_codes: list[str]


def score_online(
    A_raw: np.ndarray,
    models: dict[int, RegimeModel],
    current_regime: int,
    is_transition: bool,
    persistence: PersistenceState,
    insufficient_regimes: frozenset[int] | set[int] | None = None,
) -> WindowResult:
    """Évalue une fenêtre brute ``A_raw (t×m)`` pour son régime courant.

    Parameters
    ----------
    A_raw:
        Fenêtre 2D non standardisée (mêmes variables/ordre que le modèle).
    models:
        Modèles entraînés par régime.
    current_regime:
        Régime affecté à la fenêtre (issu du clustering en ligne).
    is_transition:
        ``True`` si la fenêtre chevauche un changement de régime / un arrêt.
    persistence:
        État de persistance (mis à jour ici).
    insufficient_regimes:
        Régimes connus mais exclus du modèle (sous-peuplés) : étiquetés
        ``insufficient_data`` plutôt que ``unknown_regime``.
    """
    insufficient_regimes = insufficient_regimes or set()

    if current_regime not in models:
        if current_regime in insufficient_regimes:
            return WindowResult(
                "insufficient_data", current_regime, {}, False, ["insufficient_data"]
            )
        return WindowResult("unknown_regime", current_regime, {}, False, ["unknown_regime"])

    if is_transition:
        return WindowResult("transition", current_regime, {}, False, ["transition_regime"])

    model = models[current_regime]
    A_std = ((A_raw - model.mean[None, :]) / model.std[None, :])
    scores = {
        k: float(v)
        for k, v in zip(
            ("Q_time", "Q_space", "T2_time", "T2_space"),
            _score_single(A_std, model),
        )
    }

    exceed = window_exceeds(scores, model)
    persistence.update(exceed=exceed, regime=current_regime)

    if exceed and persistence.alert_ready():
        status = "alert"
    elif exceed:
        status = "warning"
    else:
        status = "normal"

    return WindowResult(
        status=status,
        regime=current_regime,
        scores=scores,
        window_flag=bool(exceed),
        reason_codes=build_reason_codes(scores, model),
    )


def _score_single(A_std: np.ndarray, model: RegimeModel) -> tuple[float, float, float, float]:
    """Scores d'une fenêtre standardisée (réutilise la forme vectorisée)."""
    sc = model.score(A_std[None, :, :])
    return (
        float(sc["Q_time"][0]),
        float(sc["Q_space"][0]),
        float(sc["T2_time"][0]),
        float(sc["T2_space"][0]),
    )
