"""Couche de lecture des artefacts — contrat de données interne stable.

Principe (brief) : **CSV/JSON-first, bundle-second**. Les pages ne touchent
jamais aux chemins de fichiers : elles consomment ces lectures normalisées.
Tous les lecteurs tabulaires/JSON sont mis en cache via ``st.cache_data`` ;
``bundle.pkl`` (objets lourds, read-only) passe par ``st.cache_resource``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SUMMARY_CSV = BASE_DIR / "summary_all_gta.csv"
JFC3_COMPARISON_CSV = BASE_DIR / "jfc3_mp_comparison.csv"


# --------------------------------------------------------------------------- #
# Découverte des GTA
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def list_gtas() -> list[str]:
    """GTA disponibles (depuis le summary, sinon depuis les dossiers artifacts)."""
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        if "gta_id" in df.columns:
            return [str(g) for g in df["gta_id"].tolist()]
    if ARTIFACTS_DIR.exists():
        return sorted(
            p.name
            for p in ARTIFACTS_DIR.iterdir()
            if p.is_dir() and (p / "metrics.json").exists()
        )
    return []


# --------------------------------------------------------------------------- #
# Contrat de données
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=5)
def global_summary() -> pd.DataFrame:
    """Comparaison inter-GTA (summary_all_gta.csv) ou DataFrame vide."""
    if not SUMMARY_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SUMMARY_CSV)


@st.cache_data(show_spinner=False, ttl=5)
def online_series(gta: str) -> pd.DataFrame:
    """Série temporelle online d'un GTA (online_status.csv), index temporel."""
    path = ARTIFACTS_DIR / gta / "online_status.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "t_end" in df.columns:
        df["t_end"] = pd.to_datetime(df["t_end"], errors="coerce")
        df = df.dropna(subset=["t_end"]).set_index("t_end").sort_index()
    return df


@st.cache_data(show_spinner=False)
def online_report(gta: str) -> dict:
    return _read_json(ARTIFACTS_DIR / gta / "online_report.json")


@st.cache_data(show_spinner=False, ttl=5)
def energetic_residual(gta: str) -> pd.DataFrame:
    """Résidu énergétique EE_réel − EE_attendu (energetic_residual.csv)."""
    path = ARTIFACTS_DIR / gta / "energetic_residual.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=[0], index_col=0)


@st.cache_data(show_spinner=False)
def energetic_meta(gta: str) -> dict:
    """Coefficients / bande / référence du cross-check énergétique."""
    return _read_json(ARTIFACTS_DIR / gta / "energetic.json")


@st.cache_data(show_spinner=False)
def metrics(gta: str) -> dict:
    return _read_json(ARTIFACTS_DIR / gta / "metrics.json")


@st.cache_data(show_spinner=False)
def regime_metrics(gta: str) -> pd.DataFrame:
    path = ARTIFACTS_DIR / gta / "regime_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def data_quality(gta: str) -> dict:
    """diagnostic_mp.json si présent (cas JFC3), sinon dict vide."""
    return _read_json(ARTIFACTS_DIR / gta / "diagnostic_mp.json")


@st.cache_data(show_spinner=False)
def jfc3_comparison() -> pd.DataFrame:
    return pd.read_csv(JFC3_COMPARISON_CSV) if JFC3_COMPARISON_CSV.exists() else pd.DataFrame()


def figure_path(gta: str, name: str) -> Path | None:
    """Chemin d'une figure d'artefact si elle existe."""
    p = ARTIFACTS_DIR / gta / name
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# Bundle (complément read-only) : seuils, variables, t, exclusions
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def bundle(gta: str) -> dict | None:
    path = ARTIFACTS_DIR / gta / "bundle.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def regime_thresholds(gta: str) -> dict[int, dict[str, float]]:
    """Seuils par régime (depuis le bundle), pour tracer les lignes de contrôle."""
    b = bundle(gta)
    if not b:
        return {}
    return {int(r): dict(m.thresholds) for r, m in b.get("models", {}).items()}


# --------------------------------------------------------------------------- #
# Fenêtres brutes (pour le comparatif normal vs alerte) — complément pipeline
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def processed_frame(gta: str) -> dict:
    """DataFrame prétraité + t + variables, pour reconstruire des fenêtres 2D.

    Utilise le pipeline en lecture seule (préfiltrage), cohérent avec les
    exclusions de variables enregistrées dans le bundle.
    """
    b = bundle(gta)
    if not b:
        return {}
    try:
        from bi2dpca import io_data, preprocessing

        params = b["params"]
        exclude = tuple(b.get("exclude_vars", []))
        data = io_data.load_gta(io_data.resolve_data_path(gta), gta, exclude_vars=exclude)
        pre = preprocessing.preprocess(data, params)
        return {
            "df": pre.df[data.variables],
            "variables": data.variables,
            "t": params.t,
        }
    except Exception:  # noqa: BLE001 - lecture optionnelle, jamais bloquante
        return {}


@st.cache_data(show_spinner="Décomposition des non-observables…")
def window_causes(gta: str) -> pd.DataFrame:
    """Cause de (non-)scorabilité de chaque fenêtre de la grille, fidèle à l'online.

    Reconstruit les masques via le bundle (même modèle de régime que l'online) et
    classe chaque fenêtre par cause : ``scorable`` ou l'un des motifs de rejet.
    Renvoie un DataFrame indexé par ``t_end`` avec une colonne ``cause``.
    """
    b = bundle(gta)
    if not b:
        return pd.DataFrame()
    try:
        from bi2dpca import io_data, preprocessing
        from bi2dpca import regimes as R
        from bi2dpca import windows as W

        params = b["params"]
        t = params.t
        exclude = tuple(b.get("exclude_vars", []))
        data = io_data.load_gta(io_data.resolve_data_path(gta), gta, exclude_vars=exclude)
        pre = preprocessing.preprocess(data, params)
        regime, transition = R.assign_regimes(
            pre, b["regime_model"], b["regime_scaler"], b["regime_vars"], params
        )
        stop = W.stop_mask(pre, params)
        reg = regime.to_numpy()
        expl = pre.exploitable.to_numpy()
        stopa = stop.to_numpy()
        mon = (pre.exploitable & (~transition) & (~stop) & (regime >= 0)).to_numpy()
        idx = pre.df.index
        models = b.get("models", {})
        insuff = set(b.get("regimes_insufficient", []))
        stride = max(1, params.online_stride_steps)

        n = len(reg)
        rows: list[tuple] = []
        for s in range(0, n - t + 1, stride):
            sl_reg = reg[s : s + t]
            sl_mon = mon[s : s + t]
            r0 = int(sl_reg[0])
            end = idx[s + t - 1]
            if sl_mon.all() and r0 >= 0 and (sl_reg == r0).all():
                if r0 in models:
                    cause = "scorable"
                elif r0 in insuff:
                    cause = "insufficient_data"
                else:
                    cause = "unknown_regime"
            elif stopa[s : s + t].any():
                cause = "arrêt"
            elif (~expl[s : s + t]).any():
                cause = "trou / non-exploitable"
            elif (sl_reg != r0).any() or (sl_reg < 0).any():
                cause = "frontière régime"
            else:
                cause = "marge transition"
            rows.append((end, cause))
        return pd.DataFrame(rows, columns=["t_end", "cause"]).set_index("t_end")
    except Exception:  # noqa: BLE001 - décomposition optionnelle, jamais bloquante
        return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}
