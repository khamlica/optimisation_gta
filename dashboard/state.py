"""État partagé et constantes globales du dashboard.

Le code couleur des statuts est **fixe et global** à toute l'application : on ne
fond jamais ``transition`` et ``alert`` dans la même teinte (règle du brief).
"""

from __future__ import annotations

import streamlit as st

# Ordre canonique des statuts (scorés puis non-scorés).
STATUS_ORDER = [
    "normal",
    "warning",
    "alert",
    "transition",
    "insufficient_data",
    "unknown_regime",
]

SCORED_STATUSES = ("normal", "warning", "alert")
NON_SCORED_STATUSES = ("transition", "insufficient_data", "unknown_regime")

# Code couleur global (vert/orange/rouge/gris/violet/bleu-gris).
STATUS_COLORS = {
    "normal": "#2ca02c",
    "warning": "#ff7f0e",
    "alert": "#d62728",
    "transition": "#7f7f7f",
    "insufficient_data": "#9467bd",
    "unknown_regime": "#5b6e8c",
}

STATUS_LABELS = {
    "normal": "Normal",
    "warning": "Warning",
    "alert": "Alert",
    "transition": "Transition (non scoré)",
    "insufficient_data": "Insufficient data (non scoré)",
    "unknown_regime": "Unknown regime (non scoré)",
}


def init_state() -> None:
    """Initialise les clés d'état transverses si absentes."""
    defaults = {
        "gta": None,            # GTA courant
        "mode": "Replay",       # Replay | Live
        "live_every": 20,       # cadence Live (s)
        "regime_filter": "Tous",
        "status_filter": "Tous",
        "date_range": None,
        "selected_event": None,  # dict de l'alerte/évènement sélectionné
        "compare_normal": True,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go_to(page_key: str, **state) -> None:
    """Mémorise un drill-down dans l'état et bascule de page."""
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["_target_page"] = page_key
