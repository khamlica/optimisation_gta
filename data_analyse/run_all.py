"""Génère toutes les figures d'analyse des données prétraitées.

Usage :  python data_analyse/run_all.py [--gta JFC3 | all]
Sorties :  data_analyse/figures/<GTA>/*.png
"""

from __future__ import annotations

import argparse

import plot_correlations
import plot_distributions
import plot_regimes
from _data import gtas_to_run


def main() -> None:
    ap = argparse.ArgumentParser(description="Toutes les figures d'analyse (données prétraitées).")
    ap.add_argument("--gta", default="all", help="Identifiant GTA ou 'all'")
    gtas = gtas_to_run(ap.parse_args().gta)
    print(f"GTA : {', '.join(gtas)}")
    for mod in (plot_correlations, plot_regimes, plot_distributions):
        print(f"\n=== {mod.__name__} ===")
        mod.main(gtas)


if __name__ == "__main__":
    main()
