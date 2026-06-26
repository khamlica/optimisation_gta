"""Entraînement offline du détecteur de dérive Bi2DPCA pour un GTA.

Enchaîne : chargement -> préfiltrage -> régimes -> fenêtres 2D -> entraînement
C2DPCA-R2DPCA par régime -> seuils -> sérialisation du bundle + métriques FAR
+ figures d'audit.

Usage :
    python run_offline.py --gta JFC1 [--data data/Data_Energie_JFC1.csv]
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
    model as model_mod,
    preprocessing,
    regimes,
    validation,
    windows,
)


def train_gta(gta_id: str, data_path: str, params: config.Params, out_dir: str) -> dict:
    """Entraîne tous les régimes d'un GTA et écrit le bundle + diagnostics."""
    os.makedirs(out_dir, exist_ok=True)

    data = io_data.load_gta(data_path, gta_id)
    pre = preprocessing.preprocess(data, params)
    reg = regimes.identify_regimes(pre, params)
    mon = windows.monitorable_mask(pre, reg, params)
    wi = windows.enumerate_windows(reg.regime, mon, params.t, params.stride)
    vals = pre.df[data.variables].to_numpy()

    models: dict[int, model_mod.RegimeModel] = {}
    splits: dict[int, healthy.TemporalSplit] = {}
    for r in sorted({int(x) for x in wi.regime}):
        starts = wi.for_regime(r)
        sp = healthy.time_split(starts, params)
        splits[r] = sp
        if sp.train.size < params.regime_n_components_grid[0] + 1:
            print(f"[skip] régime {r}: trop peu de fenêtres train ({sp.train.size})")
            continue
        models[r] = model_mod.train_regime_model(
            r,
            data.variables,
            windows.extract_windows(vals, sp.train, params.t),
            windows.extract_windows(vals, sp.calib, params.t),
            params,
        )

    # Diagnostics FAR sur calib (sain) et test.
    far_calib = validation.far_on_windows(
        vals, {r: splits[r].calib for r in models}, models, params.t
    )
    far_test = validation.far_on_windows(
        vals, {r: splits[r].test for r in models}, models, params.t
    )
    finite_ok = validation.scores_are_finite(
        vals, {r: splits[r].test for r in models}, models, params.t
    )

    # Sérialisation du bundle (tout ce qu'il faut pour le scoring online).
    bundle = {
        "gta_id": gta_id,
        "variables": data.variables,
        "params": params,
        "regime_model": reg.model,
        "regime_scaler": reg.scaler,
        "regime_vars": reg.regime_vars,
        "n_regimes": reg.n_regimes,
        "models": models,
    }
    bundle_path = os.path.join(out_dir, "bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    # Tableau de seuils + tailles, lisible.
    rows = []
    for r, m in models.items():
        rows.append(
            {
                "regime": r,
                "d": int(m.V.shape[1]),
                "p": int(m.U.shape[1]),
                "n_train_clean": m.n_train,
                **{f"thr_{k}": round(v, 4) for k, v in m.thresholds.items()},
                "far_calib": round(far_calib["per_regime"].get(r, {}).get("far", float("nan")), 4),
                "far_test": round(far_test["per_regime"].get(r, {}).get("far", float("nan")), 4),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "regime_summary.csv"), index=False)

    metrics = {
        "gta_id": gta_id,
        "n_points": pre.report["n_points"],
        "pct_exploitable": round(100 * pre.report["n_exploitable"] / pre.report["n_points"], 2),
        "pct_monitorable": round(100 * float(mon.mean()), 2),
        "n_windows": len(wi),
        "n_regimes": reg.n_regimes,
        "regimes_modeled": sorted(models),
        "far_calib_global": round(far_calib["far_global"], 4),
        "far_test_global": round(far_test["far_global"], 4),
        "scores_finite_test": bool(finite_ok),
        "bundle": bundle_path,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Figure FAR vs quantile sur la calib agrégée.
    cal_scores: dict[str, list] = {}
    for r, m in models.items():
        W = windows.extract_windows(vals, splits[r].calib, params.t)
        if W.size == 0:
            continue
        sc = m.score(m.standardize(W))
        for idx in m.active_indices:
            cal_scores.setdefault(idx, []).append(sc[idx])
    cal_scores = {k: np.concatenate(v) for k, v in cal_scores.items() if v}
    if cal_scores:
        validation.plot_far_vs_quantile(
            cal_scores, os.path.join(out_dir, "far_vs_quantile.png")
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n", summary.to_string(index=False))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement offline Bi2DPCA par régime.")
    ap.add_argument("--gta", default="JFC1", help="Identifiant GTA (ex. JFC1)")
    ap.add_argument("--data", default=None, help="Chemin CSV (défaut: data/Data_Energie_<GTA>.csv)")
    ap.add_argument("--out", default=None, help="Répertoire de sortie (défaut: artifacts/<GTA>)")
    args = ap.parse_args()

    data_path = args.data or f"data/Data_Energie_{args.gta}.csv"
    out_dir = args.out or os.path.join("artifacts", args.gta)
    train_gta(args.gta, data_path, config.DEFAULT_PARAMS, out_dir)


if __name__ == "__main__":
    main()
