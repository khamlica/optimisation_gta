"""Point d'entrée du dashboard Bi2DPCA GTA (cockpit + audit).

App multipage Streamlit pilotée par ``st.navigation`` ; l'état transverse vit
dans ``st.session_state`` ; la sidebar héberge les contrôles globaux (GTA, mode
Replay/Live). Lecture seule sur les artefacts du pipeline.

Lancer :  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard import readers, state

st.set_page_config(page_title="Bi2DPCA GTA — Monitoring & Audit", layout="wide")
state.init_state()


def _global_sidebar() -> None:
    """Contrôles persistants communs à toutes les pages."""
    with st.sidebar:
        st.title("Bi2DPCA GTA")
        gtas = readers.list_gtas()
        if gtas:
            current = st.session_state.get("gta") or gtas[0]
            idx = gtas.index(current) if current in gtas else 0
            st.session_state["gta"] = st.selectbox("GTA", gtas, index=idx)
        else:
            st.warning("Aucun artefact trouvé.")

        st.session_state["mode"] = st.radio(
            "Mode", ["Replay", "Live"], horizontal=True,
            index=0 if st.session_state.get("mode", "Replay") == "Replay" else 1,
        )
        if st.session_state["mode"] == "Live":
            st.session_state["live_every"] = st.slider(
                "Cadence (s)", min_value=10, max_value=60,
                value=int(st.session_state.get("live_every", 20)), step=5,
            )
        st.divider()


_global_sidebar()

pages = [
    st.Page("dashboard/pages/overview.py", title="Vue globale", icon="🏠", default=True),
    st.Page("dashboard/pages/monitoring.py", title="Monitoring temporel", icon="📈"),
    st.Page("dashboard/pages/alert_diagnostic.py", title="Diagnostic alerte", icon="🔎"),
    st.Page("dashboard/pages/regimes.py", title="Régimes", icon="🧭"),
    st.Page("dashboard/pages/model_validation.py", title="Validation modèle", icon="✅"),
    st.Page("dashboard/pages/data_quality.py", title="Qualité données", icon="🩺"),
    st.Page("dashboard/pages/method.py", title="Méthode", icon="📚"),
]

st.navigation(pages).run()
