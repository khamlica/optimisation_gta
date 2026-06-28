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
    dynamic,
    energetic,
    healthy,
    io_data,
    model as model_mod,
    preprocessing,
    regimes,
    validation,
    windows,
)


def train_gta(
    gta_id: str,
    data_path: str,
    params: config.Params,
    out_dir: str,
    exclude_vars: tuple[str, ...] = (),
) -> dict:
    """Entraîne tous les régimes d'un GTA et écrit le bundle + diagnostics.

    ``exclude_vars`` permet d'écarter une variable (ex. MP) pour un test de
    sensibilité, sans modifier la méthode.
    """
    os.makedirs(out_dir, exist_ok=True)

    data = io_data.load_gta(data_path, gta_id, exclude_vars=exclude_vars)
    pre = preprocessing.preprocess(data, params)
    reg = regimes.identify_regimes(pre, params)
    mon = windows.monitorable_mask(pre, reg, params)
    wi = windows.enumerate_windows(reg.regime, mon, params.t, params.stride)
    vals = pre.df[data.variables].to_numpy()

    models: dict[int, model_mod.RegimeModel] = {}
    splits: dict[int, healthy.TemporalSplit] = {}
    regime_status: dict[int, str] = {}
    for r in sorted({int(x) for x in wi.regime}):
        starts = wi.for_regime(r)
        sp = healthy.time_split(starts, params)
        splits[r] = sp
        # Gate sur les effectifs : un régime sous-peuplé donne des seuils
        # instables -> exclu du modèle et marqué insufficient_data.
        if (
            sp.train.size < params.min_train_windows_per_regime
            or sp.calib.size < params.min_calib_windows_per_regime
        ):
            regime_status[r] = "insufficient_data"
            print(
                f"[insufficient_data] régime {r}: "
                f"train={sp.train.size} (<{params.min_train_windows_per_regime}) "
                f"calib={sp.calib.size} (<{params.min_calib_windows_per_regime})"
            )
            continue
        models[r] = model_mod.train_regime_model(
            r,
            data.variables,
            windows.extract_windows(vals, sp.train, params.t),
            windows.extract_windows(vals, sp.calib, params.t),
            params,
        )
        regime_status[r] = "modeled"

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
    regimes_insufficient = sorted(r for r, s in regime_status.items() if s == "insufficient_data")
    exclude_reason = config.exclude_reason_for(gta_id) if exclude_vars else ""
    bundle = {
        "gta_id": gta_id,
        "variables": data.variables,
        "exclude_vars": list(exclude_vars),
        "exclude_reason": exclude_reason,
        "params": params,
        "regime_model": reg.model,
        "regime_scaler": reg.scaler,
        "regime_vars": reg.regime_vars,
        "n_regimes": reg.n_regimes,
        "regimes_insufficient": regimes_insufficient,
        "models": models,
    }
    bundle_path = os.path.join(out_dir, "bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    # Tableau de seuils + tailles + statut, lisible. Inclut les régimes exclus.
    rows = []
    for r in sorted(splits):
        sp = splits[r]
        status = regime_status.get(r, "insufficient_data")
        if status == "modeled":
            m = models[r]
            rows.append(
                {
                    "regime": r,
                    "status": status,
                    "n_train": int(sp.train.size),
                    "n_calib": int(sp.calib.size),
                    "d": int(m.V.shape[1]),
                    "p": int(m.U.shape[1]),
                    "n_train_clean": m.n_train,
                    **{f"thr_{k}": round(v, 4) for k, v in m.thresholds.items()},
                    "far_calib": round(far_calib["per_regime"].get(r, {}).get("far", float("nan")), 4),
                    "far_test": round(far_test["per_regime"].get(r, {}).get("far", float("nan")), 4),
                }
            )
        else:
            rows.append(
                {
                    "regime": r,
                    "status": status,
                    "n_train": int(sp.train.size),
                    "n_calib": int(sp.calib.size),
                    "d": None,
                    "p": None,
                    "n_train_clean": None,
                    "thr_Q_time": float("nan"),
                    "thr_Q_space": float("nan"),
                    "far_calib": float("nan"),
                    "far_test": float("nan"),
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
        "variables": data.variables,
        "exclude_vars": list(exclude_vars),
        "exclude_reason": exclude_reason,
        "regimes_modeled": sorted(models),
        "regimes_insufficient": regimes_insufficient,
        "far_calib_global": round(far_calib["far_global"], 4) if far_calib["n_total"] else None,
        "far_test_global": round(far_test["far_global"], 4) if far_test["n_total"] else None,
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

    # Cross-check énergétique (V2) : EE = f(HP, MP, BP) sur réf saine -> résidu.
    ref_end = config.energetic_ref_end(gta_id)
    em = energetic.fit_energy_model(pre, params, ref_end=ref_end)
    if em is not None:
        rf = energetic.residual_frame(pre, em, params)
        rf.to_csv(os.path.join(out_dir, "energetic_residual.csv"))
        with open(os.path.join(out_dir, "energetic.json"), "w") as f:
            json.dump(
                {
                    "inputs": em.inputs,
                    "coef": {k: float(c) for k, c in zip(em.inputs + ["const"], em.coef)},
                    "ref_end": em.ref_end,
                    "ref_std_pct": round(em.ref_std, 3),
                    "n_ref": em.n_ref,
                    "band_k": params.energetic_band_k,
                },
                f, indent=2, ensure_ascii=False,
            )

        # Traceur dynamique invariant au niveau (couche 2a) : INDICATIF seulement.
        dyn = dynamic.fit_dynamic_reference(pre, em, params, ref_end=ref_end)
        if dyn is not None:
            dref, feats = dyn
            feats.to_csv(os.path.join(out_dir, "dynamic_features.csv"))
            with open(os.path.join(out_dir, "dynamic.json"), "w") as f:
                json.dump(
                    {
                        "features": dref.features,
                        "band": dref.band,
                        "ref_end": dref.ref_end,
                        "roll_window": dref.roll_window,
                        "n_ref": dref.n_ref,
                        "band_k": params.dynamic_band_k,
                    },
                    f, indent=2, ensure_ascii=False,
                )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\n", summary.to_string(index=False))
    return metrics


def _parse_manual_exclude(raw: str | None) -> tuple[str, ...] | None:
    """Interprète --exclude : None=config par GTA, 'none'=rien, sinon liste."""
    if raw is None:
        return None
    if raw.strip().lower() == "none":
        return ()
    return tuple(v.strip() for v in raw.split(",") if v.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement offline Bi2DPCA par régime.")
    ap.add_argument(
        "--gta",
        default="JFC1",
        help="Identifiant GTA (ex. JFC1) ou 'all' pour les 4 GTA",
    )
    ap.add_argument(
        "--data",
        default=None,
        help="Chemin CSV (défaut: résolu automatiquement par GTA). Ignoré si --gta all.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Répertoire de sortie (défaut: artifacts/<GTA>). Ignoré si --gta all.",
    )
    ap.add_argument(
        "--exclude",
        default=None,
        help=(
            "Surcharge manuelle des variables à écarter (ex. MP). Si absent, "
            "applique config.GTA_EXCLUDE_VARS. Utiliser 'none' pour ne rien écarter."
        ),
    )
    args = ap.parse_args()

    manual_exclude = _parse_manual_exclude(args.exclude)
    gtas = sorted(config.GTA_CONFIGS) if args.gta == "all" else [args.gta]
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
        exclude_vars = manual_exclude if manual_exclude is not None else config.exclude_vars_for(gta)
        print(f"\n===== OFFLINE {gta}  (source: {data_path}, exclude={exclude_vars}) =====")
        try:
            train_gta(gta, data_path, config.DEFAULT_PARAMS, out_dir, exclude_vars=exclude_vars)
        except Exception as exc:  # noqa: BLE001 - on poursuit les autres GTA
            print(f"[ERREUR] {gta}: {exc}")


if __name__ == "__main__":
    main()
