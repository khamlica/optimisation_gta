"""Vue globale — cockpit décisionnel inter-GTA.

Répond en < 10 s à : « Quel GTA pose problème, et est-ce procédé, couverture
ou modèle ? » Cartes KPI + barres empilées de statuts + table comparable.
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, readers, state

MONITORING_PAGE = "dashboard/pages/monitoring.py"


def _dominant_recent(gta: str, n: int = 96) -> str:
    """Statut dominant sur les dernières fenêtres (priorité à la sévérité)."""
    s = readers.online_series(gta)
    if s.empty:
        return "—"
    recent = s["status"].tail(n)
    for sev in ("alert", "warning"):
        if (recent == sev).any():
            return sev
    return recent.mode().iloc[0] if not recent.mode().empty else "—"


def render() -> None:
    st.title("Vue globale — cockpit GTA")
    summary = readers.global_summary()
    if summary.empty:
        st.warning(
            "`summary_all_gta.csv` introuvable. Lancer `run_online.py --gta all`."
        )
        return

    st.caption(
        "Cliquez une ligne du tableau pour ouvrir le monitoring temporel du GTA."
    )

    # --- Cartes KPI par GTA ---
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        gta = str(row["gta_id"])
        dominant = _dominant_recent(gta)
        with col:
            st.metric(
                label=f"⚙️ {gta}",
                value=state.STATUS_LABELS.get(dominant, dominant),
                delta=f"{int(row.get('n_alert', 0) or 0)} alert",
                delta_color="inverse",
            )
            far_b = row.get("FAR_calib_before")
            far_a = row.get("FAR_calib_after")
            st.caption(
                f"FAR calib {('%.1f%%' % (100*far_b)) if far_b == far_b else '—'}"
                f" → {('%.1f%%' % (100*far_a)) if far_a == far_a else '—'}"
            )
            st.caption(
                f"modèles {int(row.get('n_modeled', 0) or 0)} · "
                f"insuff. {int(row.get('n_insufficient_regimes', 0) or 0)}"
            )
            excl = str(row.get("exclude_vars", "") or "")
            if excl and excl != "nan":
                st.caption(f"⚠️ exclu : {excl}")

    st.divider()

    # --- Répartition des statuts (barres empilées) ---
    st.subheader("Répartition des statuts par GTA")
    st.plotly_chart(charts.status_stacked_bar(summary), width='stretch')
    st.caption(
        "Les zones non scorées (transition / insufficient_data / unknown_regime) "
        "signifient *non observable*, pas *normal*."
    )

    # --- Tableau comparatif sélectionnable ---
    st.subheader("Comparatif inter-GTA")
    event = st.dataframe(
        summary,
        width='stretch',
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="overview_table",
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        gta = str(summary.iloc[rows[0]]["gta_id"])
        st.session_state["gta"] = gta
        if st.button(f"➡️ Ouvrir le monitoring de {gta}", type="primary"):
            st.switch_page(MONITORING_PAGE)


render()
