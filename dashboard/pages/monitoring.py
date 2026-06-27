"""Monitoring temporel — page centrale.

Quatre couches alignées (statut, régime, Q_time, Q_space) sur une période
filtrable, click-to-drill vers le diagnostic, zones non scorées mises en avant.
Mode **Replay** (audit interactif) et mode **Live** (timeline auto-rafraîchie).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, readers, state

ALERT_PAGE = "dashboard/pages/alert_diagnostic.py"


def _pick_gta() -> str | None:
    gtas = readers.list_gtas()
    if not gtas:
        st.warning("Aucun artefact. Lancer le pipeline (`run_offline` + `run_online`).")
        return None
    current = st.session_state.get("gta") or gtas[0]
    if current not in gtas:
        current = gtas[0]
    st.session_state["gta"] = current
    return current


def _filters(series: pd.DataFrame) -> pd.DataFrame:
    """Filtres période / régime / statut (sidebar) — mode Replay."""
    if series.empty:
        return series
    with st.sidebar:
        st.markdown("### Filtres monitoring")
        tmin, tmax = series.index.min().to_pydatetime(), series.index.max().to_pydatetime()
        if tmin < tmax:
            rng = st.slider("Période", min_value=tmin, max_value=tmax, value=(tmin, tmax))
            series = series.loc[(series.index >= rng[0]) & (series.index <= rng[1])]
        regimes = sorted(int(r) for r in series["regime"].dropna().unique())
        sel_reg = st.multiselect("Régimes", regimes, default=regimes)
        if sel_reg:
            series = series[series["regime"].isin(sel_reg)]
        statuses = [s for s in state.STATUS_ORDER if s in set(series["status"])]
        sel_stat = st.multiselect("Statuts", statuses, default=statuses)
        if sel_stat:
            series = series[series["status"].isin(sel_stat)]
    return series


def _coverage_banner(series: pd.DataFrame) -> None:
    total = len(series)
    if total == 0:
        st.info("Aucune fenêtre sur la sélection.")
        return
    scored = int(series["status"].isin(state.SCORED_STATUSES).sum())
    non_scored = total - scored
    n_alert = int((series["status"] == "alert").sum())
    n_warn = int((series["status"] == "warning").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fenêtres", total)
    c2.metric("Scorées", scored, delta=f"{100*scored/total:.0f}%")
    c3.metric("Non observables", non_scored, delta=f"{100*non_scored/total:.0f}%", delta_color="inverse")
    c4.metric("Alert / Warning", f"{n_alert} / {n_warn}", delta_color="off")


def _timeline(gta: str, *, interactive: bool) -> None:
    series = readers.online_series(gta)
    if series.empty:
        st.warning(f"Pas de série online pour {gta} (`online_status.csv`).")
        return
    if interactive:
        series = _filters(series)
    _coverage_banner(series)

    fig = charts.monitoring_figure(series)
    if not interactive:
        st.plotly_chart(fig, width='stretch', key=f"mon_live_{gta}")
        return

    sel = st.plotly_chart(
        fig, width='stretch', on_select="rerun", key=f"mon_{gta}"
    )
    points = sel.selection.points if sel and sel.selection else []
    if points:
        ts = pd.to_datetime(points[0].get("x"), errors="coerce")
        if ts is not None and ts in series.index:
            row = series.loc[[ts]].iloc[0]
            st.session_state["selected_event"] = {
                "gta": gta, "t_end": str(ts),
                "regime": int(row["regime"]) if pd.notna(row["regime"]) else None,
                "status": row["status"],
            }
            st.success(f"Fenêtre sélectionnée : {ts} — statut {row['status']}")
            if st.button("🔎 Diagnostiquer cette fenêtre", type="primary"):
                st.switch_page(ALERT_PAGE)

    # Export snapshot du contexte d'analyse (série filtrée).
    st.download_button(
        "⬇️ Export CSV (série filtrée)",
        data=series.to_csv().encode("utf-8"),
        file_name=f"{gta}_monitoring_filtre.csv",
        mime="text/csv",
    )


def render() -> None:
    gta = _pick_gta()
    if gta is None:
        return
    mode = st.session_state.get("mode", "Replay")
    st.title(f"Monitoring temporel — {gta}  ·  {mode}")

    if mode == "Live":
        every = int(st.session_state.get("live_every", 20))
        st.caption(f"Rafraîchissement auto toutes les {every}s (timeline non filtrée).")

        @st.fragment(run_every=every)
        def _live_block() -> None:
            _timeline(gta, interactive=False)

        _live_block()
    else:
        st.subheader("Timeline statut / régime / Q_time / Q_space")
        _timeline(gta, interactive=True)
        with st.expander("Figure offline de référence (image)"):
            p = readers.figure_path(gta, "monitoring_online.png")
            st.image(str(p), width='stretch') if p else st.info(
                "Figure `monitoring_online.png` absente."
            )


render()
