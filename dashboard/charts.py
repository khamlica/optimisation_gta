"""Graphiques Plotly réutilisables (interactifs, code couleur global).

Tous les visuels partagent ``state.STATUS_COLORS`` ; les figures de monitoring
sont des subplots à axe x partagé pour aligner statut / régime / Q_time / Q_space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import state


def _status_y(series: pd.DataFrame) -> pd.Series:
    order = {s: i for i, s in enumerate(state.STATUS_ORDER)}
    return series["status"].map(order)


def monitoring_figure(series: pd.DataFrame, thresholds: dict | None = None) -> go.Figure:
    """4 couches alignées : statut, régime, Q_time, Q_space (axe x partagé)."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.22, 0.16, 0.31, 0.31],
        subplot_titles=("Statut", "Régime", "Q_time", "Q_space"),
    )

    # --- Statut : un point coloré par fenêtre ---
    for status in state.STATUS_ORDER:
        sub = series[series["status"] == status]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergl(
                x=sub.index,
                y=[state.STATUS_LABELS[status]] * len(sub),
                mode="markers",
                marker=dict(color=state.STATUS_COLORS[status], size=6),
                name=state.STATUS_LABELS[status],
                legendgroup=status,
                hovertemplate="%{x}<br>" + state.STATUS_LABELS[status] + "<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # --- Régime : marche colorée par statut sous-jacent ---
    if "regime" in series.columns:
        fig.add_trace(
            go.Scattergl(
                x=series.index,
                y=series["regime"],
                mode="lines",
                line=dict(color="#8c564b", width=1, shape="hv"),
                name="régime",
                showlegend=False,
                hovertemplate="%{x}<br>régime %{y}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # --- Q_time / Q_space ---
    for row, col_name in ((3, "score_Q_time"), (4, "score_Q_space")):
        if col_name not in series.columns:
            continue
        fig.add_trace(
            go.Scattergl(
                x=series.index,
                y=series[col_name],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                name=col_name,
                showlegend=False,
                hovertemplate="%{x}<br>" + col_name + "=%{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    fig.update_layout(
        height=720,
        margin=dict(l=60, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def status_stacked_bar(summary: pd.DataFrame) -> go.Figure:
    """Barres horizontales empilées des statuts par GTA."""
    fig = go.Figure()
    cols = {
        "n_normal": "normal",
        "n_warning": "warning",
        "n_alert": "alert",
        "n_transition": "transition",
        "n_insufficient_data": "insufficient_data",
        "n_unknown_regime": "unknown_regime",
    }
    for col, status in cols.items():
        if col not in summary.columns:
            continue
        fig.add_trace(
            go.Bar(
                y=summary["gta_id"],
                x=summary[col],
                name=state.STATUS_LABELS[status],
                orientation="h",
                marker_color=state.STATUS_COLORS[status],
            )
        )
    fig.update_layout(
        barmode="stack",
        height=80 + 60 * max(1, len(summary)),
        margin=dict(l=60, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="nombre de fenêtres",
    )
    return fig


def regime_bars(reg: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Barres par régime (effectifs, FAR…), colorées par statut du régime."""
    fig = go.Figure()
    colors = [
        state.STATUS_COLORS["insufficient_data"] if s == "insufficient_data" else "#1f77b4"
        for s in reg.get("status", ["modeled"] * len(reg))
    ]
    fig.add_trace(
        go.Bar(
            x=reg["regime"].astype(str),
            y=reg[value_col],
            marker_color=colors,
            text=reg[value_col],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis_title="régime",
        yaxis_title=value_col,
    )
    return fig


def window_comparison(
    df: pd.DataFrame,
    variables: list[str],
    t: int,
    alert_end: pd.Timestamp,
    normal_end: pd.Timestamp | None,
) -> go.Figure:
    """Compare une fenêtre alertée vs une fenêtre normale (même régime).

    Une trace par variable ; la fenêtre normale en pointillés, l'alertée pleine.
    """
    fig = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=variables,
    )

    def _slice(end_ts):
        if end_ts is None or end_ts not in df.index:
            return None
        pos = df.index.get_loc(end_ts)
        if isinstance(pos, slice):
            pos = pos.stop - 1
        start = max(0, pos - t + 1)
        return df.iloc[start : pos + 1]

    w_alert = _slice(alert_end)
    w_norm = _slice(normal_end)

    for i, v in enumerate(variables, start=1):
        if w_norm is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(w_norm))),
                    y=w_norm[v].to_numpy(),
                    mode="lines",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                    name="normale" if i == 1 else None,
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )
        if w_alert is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(w_alert))),
                    y=w_alert[v].to_numpy(),
                    mode="lines",
                    line=dict(color="#d62728", width=2),
                    name="alertée" if i == 1 else None,
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=180 * len(variables),
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="pas dans la fenêtre", row=len(variables), col=1)
    return fig
