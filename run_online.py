"""Rejeu online du détecteur de dérive Bi2DPCA pour un GTA.

Charge le bundle entraîné par ``run_offline.py``, ré-affecte les régimes avec le
**même** GMM, rejoue en continu toutes les fenêtres surveillables dans l'ordre
chronologique, et produit le journal des statuts + une figure de monitoring.
Lance aussi un test de dérive injectée et un test de pic isolé.

Usage :
    python run_online.py --gta JFC1 [--data data/Data_Energie_JFC1.csv]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from bi2dpca import (
    config,
    healthy,
    io_data,
    preprocessing,
    regimes,
    validation,
    windows,
)


def load_bundle(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "bundle.pkl"), "rb") as f:
        return pickle.load(f)


def replay_gta(gta_id: str, data_path: str, out_dir: str) -> dict:
    """Rejoue le monitoring online sur l'ensemble des fenêtres surveillables."""
    bundle = load_bundle(out_dir)
    params: config.Params = bundle["params"]
    models = bundle["models"]

    data = io_data.load_gta(data_path, gta_id)
    pre = preprocessing.preprocess(data, params)

    # Ré-affectation des régimes avec le GMM entraîné offline (pas de ré-fit).
    regime, transition = regimes.assign_regimes(
        pre, bundle["regime_model"], bundle["regime_scaler"], bundle["regime_vars"], params
    )
    mon = (
        pre.exploitable
        & (~transition)
        & (~windows.stop_mask(pre, params))
        & (regime >= 0)
    )
    wi = windows.enumerate_windows(regime, mon, params.t, params.stride)
    vals = pre.df[data.variables].to_numpy()

    # Drapeaux de transition par fenêtre (par construction toutes valides, donc
    # False ; conservé pour cohérence d'API).
    scored = validation.score_sequence(vals, wi, pre.df.index, None, models, params)
    scored.to_csv(os.path.join(out_dir, "online_status.csv"))
    fig = validation.plot_monitoring(scored, os.path.join(out_dir, "monitoring_online.png"))

    status_counts = scored["status"].value_counts().to_dict()

    # --- Test dérive injectée + pic isolé sur un régime bien fourni ---
    drift_report = {}
    spike_report = {}
    if models:
        # Régime avec le plus de fenêtres test propres.
        r_best = max(models, key=lambda r: int((wi.regime == r).sum()))
        starts = wi.for_regime(r_best)
        sp = healthy.time_split(starts, params)
        m_best = models[r_best]
        # Base SAINE : fenêtres d'entraînement filtrées aux fenêtres réellement
        # sous tous les seuils (on retire celles rejetées par le nettoyage
        # robuste), sinon les tests synthétiques seraient faussés.
        train_w = windows.extract_windows(vals, sp.train, params.t)
        sc_tr = m_best.score(m_best.standardize(train_w))
        healthy_mask = np.ones(train_w.shape[0], dtype=bool)
        for idx in m_best.active_indices:
            healthy_mask &= sc_tr[idx] <= m_best.thresholds[idx]
        base = train_w[healthy_mask]
        if base.shape[0] >= params.t:
            ee_idx = data.variables.index("EE")
            ee_std = float(base[:, :, ee_idx].std()) or 1.0
            drift_report = validation.detection_delay(
                base, models[r_best], ee_idx, slope_per_window=0.5 * ee_std, params=params
            )
            drift_report["regime"] = r_best
            spike_report = validation.isolated_spike_status(
                base, models[r_best], ee_idx, spike_magnitude=8.0 * ee_std,
                spike_at=base.shape[0] // 2, params=params,
            )
            spike_report = {
                "regime": r_best,
                "has_alert": spike_report["has_alert"],
                "has_warning": spike_report["has_warning"],
            }

    report = {
        "gta_id": gta_id,
        "n_windows_replayed": len(wi),
        "status_counts": status_counts,
        "figure": fig,
        "drift_injected": drift_report,
        "isolated_spike": spike_report,
    }
    with open(os.path.join(out_dir, "online_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Rejeu online Bi2DPCA par régime.")
    ap.add_argument("--gta", default="JFC1")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_path = args.data or f"data/Data_Energie_{args.gta}.csv"
    out_dir = args.out or os.path.join("artifacts", args.gta)
    replay_gta(args.gta, data_path, out_dir)


if __name__ == "__main__":
    main()
