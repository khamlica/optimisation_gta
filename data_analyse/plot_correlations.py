"""Graphes de corrélation entre variables sur les données prétraitées.

Produit, par GTA :
- ``corr_global.png`` : corrélation HP/BP/EE sur tous les points opérationnels ;
- ``corr_intra_regime.png`` : une heatmap par régime → la corrélation chute
  une fois la charge (niveau) retirée, révélant la vraie structure résiduelle ;
- ``scatter_pairs.png`` : nuages de points par paire, colorés par régime.
"""

from __future__ import annotations

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np

from _data import (
    gtas_to_run,
    load_preprocessed,
    operational_frame,
    regime_color,
    savefig,
)


def _heatmap(ax, corr, title: str):
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    return im


def plot_global(d: dict):
    of = operational_frame(d)
    corr = of[d["variables"]].corr()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = _heatmap(ax, corr, f"{d['gta']} — corrélation globale (n={len(of)})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return savefig(fig, d["gta"], "corr_global.png")


def plot_intra_regime(d: dict):
    of = operational_frame(d)
    regs = d["modeled_regimes"]
    ncol = min(len(regs), 3)
    nrow = int(np.ceil(len(regs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow), squeeze=False)
    im = None
    for k, r in enumerate(regs):
        ax = axes[k // ncol][k % ncol]
        sub = of[of["regime"] == r][d["variables"]]
        if len(sub) < 5:
            ax.set_visible(False)
            continue
        im = _heatmap(ax, sub.corr(), f"régime {r} (n={len(sub)})")
    for k in range(len(regs), nrow * ncol):
        axes[k // ncol][k % ncol].set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.suptitle(f"{d['gta']} — corrélation INTRA-régime (charge retirée)", fontsize=11)
    return savefig(fig, d["gta"], "corr_intra_regime.png")


def plot_scatter_pairs(d: dict):
    of = operational_frame(d)
    pairs = list(itertools.combinations(d["variables"], 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4.0), squeeze=False)
    for k, (a, b) in enumerate(pairs):
        ax = axes[0][k]
        for r in d["modeled_regimes"]:
            sub = of[of["regime"] == r]
            ax.scatter(sub[a], sub[b], s=4, alpha=0.3, color=regime_color(r), label=f"régime {r}")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} – {b}", fontsize=10)
    axes[0][0].legend(markerscale=2, fontsize=8, loc="best")
    fig.suptitle(f"{d['gta']} — nuages colorés par régime", fontsize=11)
    return savefig(fig, d["gta"], "scatter_pairs.png")


def main(gtas: list[str]) -> None:
    for g in gtas:
        d = load_preprocessed(g)
        for fn in (plot_global, plot_intra_regime, plot_scatter_pairs):
            print(f"  {fn(d)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graphes de corrélation (données prétraitées).")
    ap.add_argument("--gta", default="all", help="Identifiant GTA ou 'all'")
    main(gtas_to_run(ap.parse_args().gta))
