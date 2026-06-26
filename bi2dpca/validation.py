"""Validation et diagnostics du détecteur.

Trois niveaux (cf. ref/model_reference.md) :

- **seuils sur données saines** : FAR par régime et global ;
- **détection sur événements** : dérive injectée (délai de détection), pic isolé ;
- **robustesse** : finitude des scores en présence de petits trous.

Fournit aussi les figures d'audit (séries Q/T² avec seuils, régime, statut).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # rendu headless (pas d'affichage interactif)
import matplotlib.pyplot as plt  # noqa: E402

from . import config, windows
from .config import Params
from .model import RegimeModel
from .online import PersistenceState, score_online


# --------------------------------------------------------------------------- #
# Rejeu d'une séquence de fenêtres en ligne
# --------------------------------------------------------------------------- #
def score_sequence(
    vals: np.ndarray,
    wi: windows.WindowIndex,
    timestamps: pd.DatetimeIndex,
    transition_flags: np.ndarray | None,
    models: dict[int, RegimeModel],
    params: Params = config.DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Rejoue toutes les fenêtres dans l'ordre chronologique et renvoie un journal.

    Une seule ``PersistenceState`` est utilisée : elle se vide à chaque
    changement de régime (sémantique de la référence : l'alerte exige un
    dépassement *soutenu dans un même régime*).
    """
    order = np.argsort(wi.starts)
    starts = wi.starts[order]
    regs = wi.regime[order]
    trans = (
        transition_flags[order]
        if transition_flags is not None
        else np.zeros(starts.size, dtype=bool)
    )

    ps = PersistenceState(params)
    rows = []
    for s, r, tr in zip(starts, regs, trans):
        A = vals[s : s + wi.t]
        res = score_online(A, models, int(r), bool(tr), ps)
        rows.append(
            {
                "t_end": timestamps[s + wi.t - 1],
                "regime": int(r),
                "status": res.status,
                "window_flag": res.window_flag,
                **{f"score_{k}": v for k, v in res.scores.items()},
                "reason_codes": ",".join(res.reason_codes),
            }
        )
    return pd.DataFrame(rows).set_index("t_end")


# --------------------------------------------------------------------------- #
# Niveau 1 : FAR sur données saines
# --------------------------------------------------------------------------- #
def far_on_windows(
    vals: np.ndarray,
    starts_by_regime: dict[int, np.ndarray],
    models: dict[int, RegimeModel],
    t: int,
) -> dict:
    """FAR (fraction de fenêtres en dépassement) par régime et global.

    Évalue chaque fenêtre indépendamment (sans persistance) : c'est le bruit de
    fond instantané, à comparer au FAR cible.
    """
    per_regime = {}
    n_total = 0
    n_exc_total = 0
    for r, starts in starts_by_regime.items():
        if r not in models or starts.size == 0:
            continue
        model = models[r]
        W = windows.extract_windows(vals, starts, t)
        sc = model.score(model.standardize(W))
        exc = np.zeros(W.shape[0], dtype=bool)
        for idx in model.active_indices:
            exc |= sc[idx] > model.thresholds[idx]
        per_regime[r] = {
            "n": int(W.shape[0]),
            "n_exceed": int(exc.sum()),
            "far": float(exc.mean()),
        }
        n_total += W.shape[0]
        n_exc_total += int(exc.sum())
    return {
        "per_regime": per_regime,
        "far_global": (n_exc_total / n_total) if n_total else float("nan"),
        "n_total": n_total,
    }


# --------------------------------------------------------------------------- #
# Niveau 2 : dérive injectée et pic isolé
# --------------------------------------------------------------------------- #
def inject_linear_drift(
    base_windows: np.ndarray,
    var_index: int,
    slope_per_window: float,
    start_window: int = 0,
) -> np.ndarray:
    """Ajoute une rampe linéaire sur une variable, croissante de fenêtre en fenêtre.

    ``slope_per_window`` est exprimé dans l'unité brute de la variable ; la
    dérive est constante à l'intérieur d'une fenêtre et croît d'une fenêtre à
    l'autre à partir de ``start_window``.
    """
    out = base_windows.copy()
    for i in range(start_window, out.shape[0]):
        out[i, :, var_index] += slope_per_window * (i - start_window + 1)
    return out


def detection_delay(
    base_windows: np.ndarray,
    model: RegimeModel,
    var_index: int,
    slope_per_window: float,
    params: Params = config.DEFAULT_PARAMS,
    stride_minutes: int | None = None,
) -> dict:
    """Mesure le délai jusqu'au premier ``alert`` après injection d'une dérive.

    Rejoue une séquence de fenêtres d'un **même** régime (donc persistance
    continue), dérive injectée dès la première fenêtre.
    """
    drifted = inject_linear_drift(base_windows, var_index, slope_per_window, start_window=0)
    ps = PersistenceState(params, stride_minutes=stride_minutes)
    models = {model.regime_id: model}
    stride_min = stride_minutes if stride_minutes is not None else params.dt_minutes

    first_warning = None
    first_alert = None
    for i in range(drifted.shape[0]):
        res = score_online(drifted[i], models, model.regime_id, False, ps)
        if res.status == "warning" and first_warning is None:
            first_warning = i
        if res.status == "alert":
            first_alert = i
            break
    return {
        "first_warning_window": first_warning,
        "first_alert_window": first_alert,
        "delay_minutes": None if first_alert is None else first_alert * stride_min,
        "n_windows": int(drifted.shape[0]),
    }


def isolated_spike_status(
    base_windows: np.ndarray,
    model: RegimeModel,
    var_index: int,
    spike_magnitude: float,
    spike_at: int,
    params: Params = config.DEFAULT_PARAMS,
) -> dict:
    """Vérifie qu'un pic isolé ne produit pas d'``alert`` durable.

    Renvoie l'ensemble des statuts observés ; un pic unique doit donner au plus
    un ``warning`` (jamais un ``alert``, qui exige la persistance).
    """
    seq = base_windows.copy()
    seq[spike_at, :, var_index] += spike_magnitude
    ps = PersistenceState(params)
    models = {model.regime_id: model}
    statuses = []
    for i in range(seq.shape[0]):
        statuses.append(score_online(seq[i], models, model.regime_id, False, ps).status)
    return {
        "statuses": statuses,
        "has_alert": "alert" in statuses,
        "has_warning": "warning" in statuses,
    }


# --------------------------------------------------------------------------- #
# Niveau 3 : robustesse numérique
# --------------------------------------------------------------------------- #
def scores_are_finite(
    vals: np.ndarray,
    starts_by_regime: dict[int, np.ndarray],
    models: dict[int, RegimeModel],
    t: int,
) -> bool:
    """``True`` si tous les scores de toutes les fenêtres sont finis."""
    for r, starts in starts_by_regime.items():
        if r not in models or starts.size == 0:
            continue
        model = models[r]
        sc = model.score(model.standardize(windows.extract_windows(vals, starts, t)))
        for idx in model.active_indices:
            if not np.isfinite(sc[idx]).all():
                return False
    return True


# --------------------------------------------------------------------------- #
# Figures d'audit
# --------------------------------------------------------------------------- #
_STATUS_COLOR = {
    "normal": "#2ca02c",
    "warning": "#ff7f0e",
    "alert": "#d62728",
    "transition": "#7f7f7f",
    "unknown_regime": "#9467bd",
}


def plot_monitoring(scored: pd.DataFrame, out_path: str) -> str:
    """Trace Q_time / Q_space, le régime et le statut au fil du temps.

    Les scores sont tracés sans ligne de seuil unique (le seuil dépend du
    régime) ; le statut résume le verdict par fenêtre.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(scored.index, scored["score_Q_time"], lw=0.8, color="#1f77b4")
    axes[0].set_ylabel("Q_time")
    axes[1].plot(scored.index, scored["score_Q_space"], lw=0.8, color="#1f77b4")
    axes[1].set_ylabel("Q_space")

    axes[2].plot(scored.index, scored["regime"], drawstyle="steps-post", lw=0.8, color="#8c564b")
    axes[2].set_ylabel("régime")

    status_codes = {s: i for i, s in enumerate(_STATUS_COLOR)}
    colors = scored["status"].map(_STATUS_COLOR).fillna("#000000")
    axes[3].scatter(
        scored.index,
        scored["status"].map(status_codes),
        c=colors,
        s=6,
    )
    axes[3].set_yticks(list(status_codes.values()))
    axes[3].set_yticklabels(list(status_codes.keys()))
    axes[3].set_ylabel("statut")

    axes[-1].set_xlabel("temps (fin de fenêtre)")
    fig.suptitle("Monitoring Bi2DPCA — Q_time / Q_space / régime / statut")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_far_vs_quantile(
    cal_scores: dict[str, np.ndarray],
    out_path: str,
    quantiles: np.ndarray | None = None,
) -> str:
    """Courbe FAR vs quantile de seuil (calibrage), un trait par indice."""
    if quantiles is None:
        quantiles = np.linspace(0.90, 0.999, 40)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, samples in cal_scores.items():
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        far = [float((samples > np.quantile(samples, q)).mean()) for q in quantiles]
        ax.plot(quantiles, far, label=idx)
    ax.set_xlabel("quantile de seuil")
    ax.set_ylabel("FAR (sur l'échantillon de calibration)")
    ax.set_title("FAR vs quantile de seuil")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
