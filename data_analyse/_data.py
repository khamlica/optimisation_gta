"""Socle partagé des scripts d'analyse graphique des données PRÉTRAITÉES.

Pour un GTA, charge le DataFrame prétraité (séries nettoyées HP/BP/EE) et les
régimes du **modèle entraîné** (lus depuis ``artifacts/<GTA>/bundle.pkl``, donc
cohérents avec le scoring), sans dépendre de Streamlit. Réutilise le pipeline
``bi2dpca`` exactement comme ``dashboard/readers.py`` mais en standalone.

Sorties graphiques : PNG dans ``data_analyse/figures/<GTA>/``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless (génération de fichiers, pas d'affichage)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Le repo racine doit être importable pour `bi2dpca` (ce module vit dans un
# sous-dossier, contrairement aux scripts standalone de la racine).
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from bi2dpca import config, io_data, preprocessing, regimes  # noqa: E402

ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIG_DIR = Path(__file__).resolve().parent / "figures"

# Palette catégorielle des régimes — copie de dashboard/charts.py:REGIME_PALETTE
# (gardée locale pour que data_analyse reste indépendant du dashboard).
REGIME_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def regime_color(r: int) -> str:
    return REGIME_COLORS[int(r) % len(REGIME_COLORS)]


def gtas_to_run(arg: str) -> list[str]:
    """Liste de GTA : un identifiant précis, ou tous (scan des bundles)."""
    if arg and arg.lower() != "all":
        return [arg]
    # On ne garde que les vrais GTA (config.GTA_CONFIGS) : on écarte les dossiers
    # de diagnostic type ``JFC3_noMP`` / ``JFC3_withMP``.
    found = {p.parent.name for p in ARTIFACTS_DIR.glob("*/bundle.pkl")}
    return sorted(found & set(config.GTA_CONFIGS))


def _load_bundle(gta: str) -> dict:
    with open(ARTIFACTS_DIR / gta / "bundle.pkl", "rb") as f:
        return pickle.load(f)


def load_preprocessed(gta: str) -> dict:
    """DataFrame prétraité + régimes/arrêts du modèle entraîné, pour un GTA."""
    b = _load_bundle(gta)
    params = b["params"]
    exclude = tuple(b.get("exclude_vars", []))
    data = io_data.load_gta(io_data.resolve_data_path(gta), gta, exclude_vars=exclude)
    pre = preprocessing.preprocess(data, params)
    regime, transition = regimes.assign_regimes(
        pre, b["regime_model"], b["regime_scaler"], b["regime_vars"], params
    )
    stop = preprocessing.stop_mask(pre, params)
    return {
        "gta": gta,
        "df": pre.df[data.variables],
        "variables": list(data.variables),
        "regime_vars": list(b["regime_vars"]),
        "regime": regime,
        "transition": transition,
        "stop": stop,
        "exploitable": pre.exploitable,
        "modeled_regimes": sorted(int(r) for r in b.get("models", {})),
        "params": params,
    }


def operational_mask(d: dict) -> pd.Series:
    """Points réellement exploités par le modèle (hors arrêt/trou/transition)."""
    return d["exploitable"] & (~d["stop"]) & (~d["transition"]) & (d["regime"] >= 0)


def operational_frame(d: dict) -> pd.DataFrame:
    """``df`` + colonne ``regime`` restreints aux points opérationnels (sans NaN)."""
    m = operational_mask(d)
    out = d["df"].loc[m].copy()
    out["regime"] = d["regime"].loc[m].astype(int)
    return out.dropna()


def load_all_variables(gta: str) -> pd.DataFrame:
    """Toutes les variables canoniques (MP inclus), points exploitables non-arrêt.

    Sert au graphe de coefficient de variation : montrer qu'un canal quasi
    constant (MP) a une variabilité bien plus faible → justifie son exclusion.
    """
    b = _load_bundle(gta)
    params = b["params"]
    data = io_data.load_gta(io_data.resolve_data_path(gta), gta, exclude_vars=())
    pre = preprocessing.preprocess(data, params)
    stop = preprocessing.stop_mask(pre, params)
    mask = pre.exploitable & (~stop)
    return pre.df[data.variables].loc[mask].dropna()


def savefig(fig, gta: str, name: str) -> Path:
    out = FIG_DIR / gta
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
