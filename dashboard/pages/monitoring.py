"""Monitoring temporel — page centrale.

Quatre couches alignées (statut, régime, Q_time, Q_space) avec seuils par régime
et dépassements surlignés, plus une **vue physique interprétable** (rendement
EE/HP ou variables brutes) sur la même période. Click-to-drill vers le diagnostic.
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


def _filters(series: pd.DataFrame) -> tuple[pd.DataFrame, tuple | None]:
    """Filtres période / régime / statut (sidebar). Renvoie (série filtrée, période)."""
    if series.empty:
        return series, None
    rng = None
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
    return series, rng


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


def _non_observable_breakdown(gta: str, period: tuple | None) -> None:
    """Option : décomposition des fenêtres non observables par cause."""
    with st.expander("➕ Décomposer les non-observables par cause (optionnel)"):
        wc = readers.window_causes(gta)
        if wc.empty:
            st.info("Décomposition indisponible (bundle absent).")
            return
        if period is not None:
            wc = wc.loc[(wc.index >= period[0]) & (wc.index <= period[1])]
        counts = wc["cause"].value_counts()
        non_obs_order = [
            "arrêt", "frontière régime", "trou / non-exploitable",
            "marge transition", "insufficient_data", "unknown_regime",
        ]
        rows = [(c, int(counts.get(c, 0))) for c in non_obs_order if counts.get(c, 0)]
        tot = int(len(wc))
        non_obs = sum(v for _, v in rows)
        if non_obs == 0:
            st.success("Aucune fenêtre non observable sur la sélection.")
            return
        tab = pd.DataFrame(rows, columns=["cause", "fenêtres"]).set_index("cause")
        tab["% des non-obs."] = (100 * tab["fenêtres"] / non_obs).round(1)
        tab["% du total"] = (100 * tab["fenêtres"] / tot).round(1)
        st.caption(
            f"{non_obs} fenêtres non observables sur {tot} "
            f"({100*non_obs/tot:.0f} %). La marge de transition n'en est qu'une "
            "petite part : l'essentiel vient de l'arrêt et des frontières de régime "
            "(chaque point non surveillable invalide ~t fenêtres)."
        )
        st.dataframe(tab, width="stretch")
        st.bar_chart(tab["fenêtres"])


def _thresholds(gta: str) -> dict[int, dict[str, float]]:
    """Seuils Q_time / Q_space par régime modélisé (depuis regime_summary.csv)."""
    rm = readers.regime_metrics(gta)
    if rm.empty or "thr_Q_time" not in rm.columns:
        return {}
    out: dict[int, dict[str, float]] = {}
    for _, r in rm.iterrows():
        if r.get("status") != "modeled":
            continue
        out[int(r["regime"])] = {
            "Q_time": float(r["thr_Q_time"]),
            "Q_space": float(r["thr_Q_space"]),
        }
    return out


def _ratio_baseline(df: pd.DataFrame) -> tuple[float, float, float] | None:
    """Référence saine du rendement EE/HP : médiane/IQR sur le 1er tiers (temps)."""
    if not {"EE", "HP"}.issubset(df.columns):
        return None
    op = df[df["HP"] > 1.0]
    if len(op) < 30:
        return None
    early = op.iloc[: max(30, len(op) // 3)]
    r = (early["EE"] / early["HP"]).dropna()
    if r.empty:
        return None
    return float(r.median()), float(r.quantile(0.25)), float(r.quantile(0.75))


def _physical_view(gta: str, period: tuple | None) -> None:
    """Vue physique alignée sur la période : rendement EE/HP ou variables brutes."""
    pf = readers.processed_frame(gta)
    if not pf or pf.get("df") is None or pf["df"].empty:
        st.info("Données brutes indisponibles pour la vue physique.")
        return
    df = pf["df"]
    full_df = df
    if period is not None:
        df = df.loc[(df.index >= period[0]) & (df.index <= period[1])]

    mode = st.radio(
        "Vue", ["Rendement EE/HP", "Variables brutes"], horizontal=True,
        help="Rendement EE/HP : montre un changement de point de fonctionnement "
             "(ex. gain post-maintenance) même quand le régime ne change pas.",
    )
    regime = readers.online_series(gta)["regime"]
    baseline = _ratio_baseline(full_df) if mode == "Rendement EE/HP" else None
    st.plotly_chart(
        charts.physical_figure(df, regime, mode, baseline=baseline),
        width="stretch", key=f"phys_{gta}",
    )
    if mode == "Rendement EE/HP":
        st.caption(
            "Le rendement peut sortir de la bande de référence (saine) alors que "
            "le régime reste identique : le détecteur signale un **changement**, "
            "pas une dégradation — l'interprétation (bon/mauvais) reste humaine."
        )


def render() -> None:
    gta = _pick_gta()
    if gta is None:
        return
    st.title(f"Monitoring temporel — {gta}")

    series = readers.online_series(gta)
    if series.empty:
        st.warning(f"Pas de série online pour {gta} (`online_status.csv`).")
        return

    series_f, period = _filters(series)
    _coverage_banner(series_f)
    _non_observable_breakdown(gta, period)

    st.subheader("Timeline statut / régime / Q_time / Q_space")
    fig = charts.monitoring_figure(series_f, _thresholds(gta))
    sel = st.plotly_chart(fig, width="stretch", on_select="rerun", key=f"mon_{gta}")

    points = sel.selection.points if sel and sel.selection else []
    if points:
        ts = pd.to_datetime(points[0].get("x"), errors="coerce")
        if ts is not None and ts in series_f.index:
            row = series_f.loc[[ts]].iloc[0]
            st.session_state["selected_event"] = {
                "gta": gta, "t_end": str(ts),
                "regime": int(row["regime"]) if pd.notna(row["regime"]) else None,
                "status": row["status"],
            }
            st.success(f"Fenêtre sélectionnée : {ts} — statut {row['status']}")
            if st.button("🔎 Diagnostiquer cette fenêtre", type="primary"):
                st.switch_page(ALERT_PAGE)

    st.download_button(
        "⬇️ Export CSV (série filtrée)",
        data=series_f.to_csv().encode("utf-8"),
        file_name=f"{gta}_monitoring_filtre.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Vue physique / interprétable")
    _physical_view(gta, period)

    with st.expander("Figure offline de référence (image)"):
        p = readers.figure_path(gta, "monitoring_online.png")
        st.image(str(p), width="stretch") if p else st.info(
            "Figure `monitoring_online.png` absente."
        )


render()
