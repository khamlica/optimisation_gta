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
import pandas as pd

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
    """Rejoue le monitoring online sur toute la grille (fenêtres classifiées)."""
    bundle = load_bundle(out_dir)
    params: config.Params = bundle["params"]
    models = bundle["models"]
    insufficient = set(bundle.get("regimes_insufficient", []))
    exclude_vars = tuple(bundle.get("exclude_vars", []))

    data = io_data.load_gta(data_path, gta_id, exclude_vars=exclude_vars)
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

    # Rejeu sur toute la grille : chaque fenêtre est classifiée (scored vs
    # transition / insufficient_data / unknown_regime).
    scored = validation.replay_grid(
        vals, regime, mon, pre.df.index, models, insufficient, params
    )
    scored.to_csv(os.path.join(out_dir, "online_status.csv"))
    fig = validation.plot_monitoring(scored, os.path.join(out_dir, "monitoring_online.png"))

    counts = validation.status_summary(scored)

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
        "n_grid_windows": int(len(scored)),
        "n_scorable_windows": len(wi),
        **counts,
        "figure": fig,
        "drift_injected": drift_report,
        "isolated_spike": spike_report,
    }
    with open(os.path.join(out_dir, "online_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_summary(
    gtas: list[str],
    base_dir: str,
    out_csv: str,
    before_map: dict[str, float] | None = None,
) -> str:
    """Assemble le rapport comparatif par GTA (offline metrics + online counts).

    ``before_map`` (gta -> FAR calib de l'exécution précédente) permet d'afficher
    le FAR calib **avant/après** correction de la calibration.
    """
    before_map = before_map or {}
    rows = []
    for gta in gtas:
        gdir = os.path.join(base_dir, gta)
        metrics_path = os.path.join(gdir, "metrics.json")
        report_path = os.path.join(gdir, "online_report.json")
        if not (os.path.exists(metrics_path) and os.path.exists(report_path)):
            print(f"[summary] {gta}: artefacts manquants, ignoré")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        with open(report_path) as f:
            rep = json.load(f)
        rows.append(
            {
                "gta_id": gta,
                "exploitable_pct": m.get("pct_exploitable"),
                "monitorable_pct": m.get("pct_monitorable"),
                "n_regimes": m.get("n_regimes"),
                "n_modeled": len(m.get("regimes_modeled", [])),
                "n_insufficient_regimes": len(m.get("regimes_insufficient", [])),
                "n_windows": m.get("n_windows"),
                "FAR_calib_before": before_map.get(gta),
                "FAR_calib_after": m.get("far_calib_global"),
                "n_normal": rep.get("n_normal", 0),
                "n_warning": rep.get("n_warning", 0),
                "n_alert": rep.get("n_alert", 0),
                "n_transition": rep.get("n_transition", 0),
                "n_unknown_regime": rep.get("n_unknown_regime", 0),
                "n_insufficient_data": rep.get("n_insufficient_data", 0),
                "n_scored_total": rep.get("n_scored_total", 0),
                "n_non_scored_total": rep.get("n_non_scored_total", 0),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n===== RAPPORT COMPARATIF ({out_csv}) =====")
    print(df.to_string(index=False))
    return out_csv


def _read_before_map(path: str) -> dict[str, float]:
    """FAR calib de l'exécution précédente, depuis un summary existant."""
    if not os.path.exists(path):
        return {}
    prev = pd.read_csv(path)
    # Préférer le FAR d'origine (avant correction) s'il est déjà mémorisé.
    col = next(
        (
            c
            for c in ("FAR_calib_before", "FAR_calib_global", "FAR_calib_after")
            if c in prev.columns
        ),
        None,
    )
    if col is None or "gta_id" not in prev.columns:
        return {}
    return {str(g): v for g, v in zip(prev["gta_id"], prev[col])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rejeu online Bi2DPCA par régime.")
    ap.add_argument("--gta", default="JFC1", help="Identifiant GTA ou 'all'")
    ap.add_argument("--data", default=None, help="Ignoré si --gta all")
    ap.add_argument("--out", default=None, help="Ignoré si --gta all")
    ap.add_argument(
        "--summary",
        default="summary_all_gta.csv",
        help="Chemin du rapport comparatif (mode all)",
    )
    args = ap.parse_args()

    gtas = sorted(config.GTA_CONFIGS) if args.gta == "all" else [args.gta]
    # Capturer le FAR calib précédent AVANT d'écraser le summary (pour avant/après).
    before_map = _read_before_map(args.summary) if args.gta == "all" else {}
    for gta in gtas:
        data_path = (
            args.data
            if (args.data and args.gta != "all")
            else io_data.resolve_data_path(gta)
        )
        out_dir = (
            args.out
            if (args.out and args.gta != "all")
            else os.path.join("artifacts", gta)
        )
        print(f"\n===== ONLINE {gta}  (source: {data_path}) =====")
        try:
            replay_gta(gta, data_path, out_dir)
        except Exception as exc:  # noqa: BLE001 - on poursuit les autres GTA
            print(f"[ERREUR] {gta}: {exc}")

    if args.gta == "all":
        build_summary(gtas, "artifacts", args.summary, before_map=before_map)


if __name__ == "__main__":
    main()
