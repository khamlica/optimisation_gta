"""Diagnostic d'alerte — « pourquoi cette alerte existe-t-elle ? ».

Panneau de contexte (scores, seuils, reason_codes, persistance) + comparatif
d'une fenêtre alertée vs une fenêtre normale du **même régime**. Le dashboard
n'invente pas de causalité : il montre que la structure dynamique change.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, readers, state


def _select_event(gta: str, series: pd.DataFrame) -> pd.Timestamp | None:
    """Choix de la fenêtre à diagnostiquer (depuis le drill-down ou une table)."""
    ev = st.session_state.get("selected_event")
    default_ts = None
    if ev and ev.get("gta") == gta:
        default_ts = pd.to_datetime(ev.get("t_end"), errors="coerce")

    flagged = series[series["status"].isin(("alert", "warning"))]
    st.markdown("#### Évènements (alert / warning)")
    if flagged.empty:
        st.info("Aucun évènement alert/warning sur ce GTA.")
        return default_ts
    table = flagged[["regime", "status", "score_Q_time", "score_Q_space", "reason_codes"]]
    event = st.dataframe(
        table, width='stretch', on_select="rerun",
        selection_mode="single-row", key=f"events_{gta}",
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        return flagged.index[rows[0]]
    return default_ts


def _normal_window(series: pd.DataFrame, regime: int, before: pd.Timestamp) -> pd.Timestamp | None:
    """Une fenêtre normale du même régime, la plus proche avant l'alerte."""
    cand = series[(series["regime"] == regime) & (series["status"] == "normal")]
    cand = cand[cand.index < before]
    if cand.empty:
        cand = series[(series["regime"] == regime) & (series["status"] == "normal")]
    return cand.index[-1] if not cand.empty else None


def render() -> None:
    gta = st.session_state.get("gta") or (readers.list_gtas() or [None])[0]
    if not gta:
        st.warning("Aucun GTA disponible.")
        return
    st.title(f"Diagnostic d'alerte — {gta}")

    series = readers.online_series(gta)
    if series.empty:
        st.warning("Pas de série online.")
        return

    ts = _select_event(gta, series)
    if ts is None or ts not in series.index:
        st.info("Sélectionnez une fenêtre (table ci-dessus ou depuis le monitoring).")
        return

    row = series.loc[[ts]].iloc[0]
    regime = int(row["regime"]) if pd.notna(row["regime"]) else None
    thr = readers.regime_thresholds(gta).get(regime, {})

    # --- Panneau de contexte ---
    st.subheader("Contexte de la fenêtre")
    c1, c2, c3 = st.columns(3)
    c1.metric("Horodatage", str(ts))
    c1.metric("Régime", regime if regime is not None else "—")
    c2.metric("Statut", state.STATUS_LABELS.get(row["status"], row["status"]))
    c2.metric("reason_codes", row.get("reason_codes", "") or "—")
    q_time, q_space = row.get("score_Q_time"), row.get("score_Q_space")
    c3.metric("Q_time", f"{q_time:.2f}" if pd.notna(q_time) else "—",
              delta=f"seuil {thr.get('Q_time', float('nan')):.2f}" if thr else None)
    c3.metric("Q_space", f"{q_space:.2f}" if pd.notna(q_space) else "—",
              delta=f"seuil {thr.get('Q_space', float('nan')):.2f}" if thr else None)

    # --- Comparatif fenêtre alertée vs normale (même régime) ---
    st.subheader("Comparatif fenêtre alertée vs normale (même régime)")
    pf = readers.processed_frame(gta)
    if not pf:
        st.info("Données brutes indisponibles pour reconstruire les fenêtres.")
        return
    normal_ts = _normal_window(series, regime, ts) if regime is not None else None
    st.caption(
        f"Fenêtre alertée : {ts}"
        + (f" · fenêtre normale de référence : {normal_ts}" if normal_ts is not None else
           " · aucune fenêtre normale de référence trouvée")
    )
    fig = charts.window_comparison(pf["df"], pf["variables"], pf["t"], ts, normal_ts)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Lecture : une divergence de forme/amplitude entre la fenêtre normale "
        "(pointillés verts) et la fenêtre alertée (rouge) appuie une rupture de "
        "structure ; ce n'est pas une preuve de cause."
    )


render()
