"""Graphes des régimes (du modèle entraîné) sur les données prétraitées.

Produit, par GTA :
- ``regimes_space.png`` : séparation des régimes dans l'espace des variables de
  clustering (HP–BP) ;
- ``regimes_timeline.png`` : régime au cours du temps + arrêts ;
- ``regimes_population.png`` : effectif et durée moyenne de séjour par régime.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from _data import (
    gtas_to_run,
    load_preprocessed,
    operational_frame,
    regime_color,
    savefig,
)


def plot_space(d: dict):
    of = operational_frame(d)
    rv = d["regime_vars"]
    xv, yv = (rv + d["variables"])[:2]  # variables de clustering, sinon repli
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for r in d["modeled_regimes"]:
        sub = of[of["regime"] == r]
        ax.scatter(sub[xv], sub[yv], s=5, alpha=0.35, color=regime_color(r), label=f"régime {r}")
    ax.set_xlabel(xv)
    ax.set_ylabel(yv)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"{d['gta']} — séparation des régimes ({xv} vs {yv})", fontsize=11)
    return savefig(fig, d["gta"], "regimes_space.png")


def plot_timeline(d: dict):
    reg = d["regime"]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for r in d["modeled_regimes"]:
        m = reg == r
        ax.scatter(reg.index[m], reg[m], s=4, color=regime_color(r), label=f"régime {r}")
    stop = d["stop"]
    if stop.any():
        ax.scatter(reg.index[stop], [-1] * int(stop.sum()), s=4, color="#999999", label="arrêt")
    ax.set_yticks(sorted(d["modeled_regimes"]) + [-1])
    ax.set_ylabel("régime")
    ax.legend(markerscale=2, fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title(f"{d['gta']} — régime au cours du temps", fontsize=11)
    return savefig(fig, d["gta"], "regimes_timeline.png")


def plot_population(d: dict):
    reg = d["regime"].dropna().astype(int)
    counts = reg[reg >= 0].value_counts().sort_index()

    # Durée moyenne de séjour : longueur moyenne des plages consécutives, ×dt.
    grp = (reg != reg.shift()).cumsum()
    runs = pd.DataFrame({"regime": reg.groupby(grp).first().values,
                         "len": reg.groupby(grp).size().values})
    dt = d["params"].dt_minutes
    dwell = runs[runs.regime >= 0].groupby("regime")["len"].mean() * dt / 60.0  # heures

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    colors = [regime_color(r) for r in counts.index]
    a1.bar([str(r) for r in counts.index], counts.values, color=colors)
    a1.set_title("effectif (points)")
    a1.set_xlabel("régime")
    a2.bar([str(r) for r in dwell.index], dwell.values,
           color=[regime_color(r) for r in dwell.index])
    a2.set_title("durée moyenne de séjour (h)")
    a2.set_xlabel("régime")
    fig.suptitle(f"{d['gta']} — population et persistance des régimes", fontsize=11)
    return savefig(fig, d["gta"], "regimes_population.png")


def main(gtas: list[str]) -> None:
    for g in gtas:
        d = load_preprocessed(g)
        for fn in (plot_space, plot_timeline, plot_population):
            print(f"  {fn(d)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Graphes des régimes (données prétraitées).")
    ap.add_argument("--gta", default="all", help="Identifiant GTA ou 'all'")
    main(gtas_to_run(ap.parse_args().gta))
