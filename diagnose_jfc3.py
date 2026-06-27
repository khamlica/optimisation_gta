"""Diagnostic data de JFC3 et test de sensibilité MP (sans modifier Bi2DPCA).

Produit :
- artifacts/JFC3/diagnostic_mp.json : taux NaN / zéros / quasi-zéros / capteur
  bloqué par variable, et impact exact de MP sur exploitable_pct / monitorable_pct ;
- artifacts/JFC3_noMP/ : pipeline complet JFC3 en écartant MP ([HP, BP, EE]) ;
- jfc3_mp_comparison.csv : comparaison JFC3 avec MP vs sans MP.

MP n'est PAS supprimé du code : c'est un diagnostic réversible.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from bi2dpca import config, io_data, preprocessing, regimes, windows
import run_offline
import run_online

GTA = "JFC3"
NEAR_ZERO_EPS = 1e-6


def per_variable_diagnostics(df: pd.DataFrame, params: config.Params) -> dict:
    """Taux NaN / zéros / quasi-zéros / capteur bloqué, par variable."""
    out = {}
    for v in df.columns:
        s = df[v]
        n = len(s)
        rolling_std = s.rolling(window=params.stuck_window, min_periods=params.stuck_window).std()
        stuck = (rolling_std.fillna(np.inf) < params.stuck_min_std)
        out[v] = {
            "nan_pct": round(100 * float(s.isna().mean()), 2),
            "zero_pct": round(100 * float((s.abs() < 1e-9).mean()), 2),
            "near_zero_pct": round(100 * float((s.abs() < NEAR_ZERO_EPS).mean()), 2),
            "stuck_pct": round(100 * float(stuck.mean()), 2),
            "min": float(np.nanmin(s.to_numpy())),
            "max": float(np.nanmax(s.to_numpy())),
            "mean": float(np.nanmean(s.to_numpy())),
        }
    return out


def exploitable_monitorable(exclude_vars: tuple[str, ...], params: config.Params) -> dict:
    """exploitable_pct / monitorable_pct pour une variante (avec ou sans MP)."""
    data = io_data.load_gta(io_data.resolve_data_path(GTA), GTA, exclude_vars=exclude_vars)
    pre = preprocessing.preprocess(data, params)
    expl = 100 * pre.report["n_exploitable"] / pre.report["n_points"]
    res = {
        "variables": data.variables,
        "exploitable_pct": round(expl, 2),
        "preprocess_report": pre.report,
    }
    try:
        reg = regimes.identify_regimes(pre, params)
        mon = windows.monitorable_mask(pre, reg, params)
        res["monitorable_pct"] = round(100 * float(mon.mean()), 2)
        res["n_regimes"] = reg.n_regimes
    except Exception as exc:  # noqa: BLE001
        res["monitorable_pct"] = 0.0
        res["n_regimes"] = None
        res["regime_error"] = str(exc)
    return res


def _read_artifacts(out_dir: str) -> dict:
    """metrics.json + online_report.json d'une variante déjà exécutée."""
    with open(os.path.join(out_dir, "metrics.json")) as f:
        m = json.load(f)
    with open(os.path.join(out_dir, "online_report.json")) as f:
        rep = json.load(f)
    return {
        "exploitable_pct": m.get("pct_exploitable"),
        "monitorable_pct": m.get("pct_monitorable"),
        "n_windows": m.get("n_windows"),
        "n_modeled": len(m.get("regimes_modeled", [])),
        "FAR_calib": m.get("far_calib_global"),
        "n_normal": rep.get("n_normal", 0),
        "n_warning": rep.get("n_warning", 0),
        "n_alert": rep.get("n_alert", 0),
        "n_transition": rep.get("n_transition", 0),
        "n_unknown_regime": rep.get("n_unknown_regime", 0),
        "n_insufficient_data": rep.get("n_insufficient_data", 0),
    }


def main() -> None:
    params = config.DEFAULT_PARAMS
    data_path = io_data.resolve_data_path(GTA)

    # --- 1) Diagnostics par variable (sur données brutes régulières) ---
    raw = io_data.load_gta(data_path, GTA)
    pre_full = preprocessing.preprocess(raw, params)
    var_diag = per_variable_diagnostics(pre_full.df, params)

    # --- 2) Impact exact de MP : avec MP vs sans MP (préfiltrage) ---
    with_mp = exploitable_monitorable(exclude_vars=(), params=params)
    without_mp = exploitable_monitorable(exclude_vars=("MP",), params=params)
    mp_impact = {
        "exploitable_pct_with_MP": with_mp["exploitable_pct"],
        "exploitable_pct_without_MP": without_mp["exploitable_pct"],
        "exploitable_gain_pts": round(
            without_mp["exploitable_pct"] - with_mp["exploitable_pct"], 2
        ),
        "monitorable_pct_with_MP": with_mp["monitorable_pct"],
        "monitorable_pct_without_MP": without_mp["monitorable_pct"],
        "monitorable_gain_pts": round(
            without_mp["monitorable_pct"] - with_mp["monitorable_pct"], 2
        ),
    }

    diagnostic = {
        "gta_id": GTA,
        "source": data_path,
        "per_variable": var_diag,
        "preprocess_report_with_MP": pre_full.report,
        "mp_impact": mp_impact,
    }
    os.makedirs(os.path.join("artifacts", GTA), exist_ok=True)
    diag_path = os.path.join("artifacts", GTA, "diagnostic_mp.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostic, f, indent=2, ensure_ascii=False)
    print("=== Diagnostic par variable (JFC3) ===")
    print(json.dumps(var_diag, indent=2, ensure_ascii=False))
    print("\n=== Impact MP ===")
    print(json.dumps(mp_impact, indent=2, ensure_ascii=False))

    # --- 3) Pipeline complet JFC3 SANS MP ---
    out_nomp = os.path.join("artifacts", f"{GTA}_noMP")
    print(f"\n===== PIPELINE {GTA} SANS MP -> {out_nomp} =====")
    run_offline.train_gta(GTA, data_path, params, out_nomp, exclude_vars=("MP",))
    run_online.replay_gta(GTA, data_path, out_nomp)

    # --- 4) Comparaison avec MP vs sans MP ---
    with_mp_art = _read_artifacts(os.path.join("artifacts", GTA))
    without_mp_art = _read_artifacts(out_nomp)
    comp = pd.DataFrame(
        [{"variante": "avec_MP", **with_mp_art}, {"variante": "sans_MP", **without_mp_art}]
    )
    comp_path = "jfc3_mp_comparison.csv"
    comp.to_csv(comp_path, index=False)
    print(f"\n===== COMPARAISON JFC3 avec/sans MP ({comp_path}) =====")
    print(comp.to_string(index=False))
    print(f"\nDiagnostic écrit : {diag_path}")


if __name__ == "__main__":
    main()
