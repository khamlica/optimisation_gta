"""Distributions et variance des variables sur les données prétraitées.

Produit, par GTA :
- ``hist_variables.png`` : distribution de chaque variable (points opérationnels) ;
- ``hist_par_regime.png`` : distribution de chaque variable par régime ;
- ``cv_variables.png`` : coefficient de variation (σ/μ) par variable, MP inclus →
  un canal quasi constant (MP) ressort comme dégénéré, ce qui justifie son exclusion.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from _data import (
    gtas_to_run,
    load_all_variables,
    load_preprocessed,
    operational_frame,
    regime_color,
    savefig,
)


def plot_hist_variables(d: dict):
    of = operational_frame(d)
    vs = d["variables"]
    fig, axes = plt.subplots(1, len(vs), figsize=(4.0 * len(vs), 3.4), squeeze=False)
    for k, v in enumerate(vs):
        ax = axes[0][k]
        ax.hist(of[v].values, bins=60, color="#1f77b4", alpha=0.85)
        ax.set_title(v, fontsize=10)
        ax.set_xlabel(v)
    axes[0][0].set_ylabel("effectif")
    fig.suptitle(f"{d['gta']} — distribution des variables (points opérationnels)", fontsize=11)
    return savefig(fig, d["gta"], "hist_variables.png")


def plot_hist_par_regime(d: dict):
    of = operational_frame(d)
    vs = d["variables"]
    fig, axes = plt.subplots(1, len(vs), figsize=(4.0 * len(vs), 3.4), squeeze=False)
    for k, v in enumerate(vs):
        ax = axes[0][k]
        for r in d["modeled_regimes"]:
            sub = of[of["regime"] == r][v]
            if len(sub) < 5:
                continue
            ax.hist(sub.values, bins=40, density=True, alpha=0.45,
                    color=regime_color(r), label=f"régime {r}")
        ax.set_title(v, fontsize=10)
        ax.set_xlabel(v)
    axes[0][0].set_ylabel("densité")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{d['gta']} — distribution par régime", fontsize=11)
    return savefig(fig, d["gta"], "hist_par_regime.png")


def plot_cv(d: dict):
    allv = load_all_variables(d["gta"])  # MP inclus
    cv = (allv.std() / allv.mean().abs() * 100).sort_values()
    kept = set(d["variables"])
    colors = ["#2ca02c" if v in kept else "#d62728" for v in cv.index]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(cv.index, cv.values, color=colors)
    for i, val in enumerate(cv.values):
        ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("coefficient de variation σ/μ (%)")
    ax.set_title(f"{d['gta']} — variabilité par variable\n"
                 "(vert = retenu, rouge = exclu du modèle)", fontsize=10)
    return savefig(fig, d["gta"], "cv_variables.png")


def main(gtas: list[str]) -> None:
    for g in gtas:
        d = load_preprocessed(g)
        for fn in (plot_hist_variables, plot_hist_par_regime, plot_cv):
            print(f"  {fn(d)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Distributions et variance (données prétraitées).")
    ap.add_argument("--gta", default="all", help="Identifiant GTA ou 'all'")
    main(gtas_to_run(ap.parse_args().gta))
