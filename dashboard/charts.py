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

# Palette catégorielle des régimes : une couleur stable par numéro de régime,
# pour qu'une même teinte/niveau dans le temps signale « même régime ».
REGIME_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _regime_color(r: int) -> str:
    return REGIME_PALETTE[int(r) % len(REGIME_PALETTE)]


def _status_y(series: pd.DataFrame) -> pd.Series:
    order = {s: i for i, s in enumerate(state.STATUS_ORDER)}
    return series["status"].map(order)


def _threshold_step(series: pd.DataFrame, thresholds: dict, key: str) -> pd.Series:
    """Seuil applicable à chaque fenêtre selon son régime (NaN si non modélisé)."""
    return series["regime"].map(
        lambda r: thresholds.get(int(r), {}).get(key, np.nan) if pd.notna(r) else np.nan
    )


def monitoring_figure(series: pd.DataFrame, thresholds: dict | None = None) -> go.Figure:
    """4 couches alignées : statut, régime, Q_time, Q_space (axe x partagé).

    ``thresholds`` : ``{regime: {"Q_time": x, "Q_space": y}}`` — trace les lignes
    de contrôle par régime et surligne les dépassements.
    """
    thresholds = thresholds or {}
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.22, 0.16, 0.31, 0.31],
        subplot_titles=(
            "Statut",
            "Régime (même couleur = même régime)",
            "Q_time (— seuil régime)",
            "Q_space (— seuil régime)",
        ),
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

    # --- Régime : marqueurs colorés par numéro de régime (catégoriel) ---
    if "regime" in series.columns:
        for r in sorted(int(v) for v in series["regime"].dropna().unique()):
            sub = series[series["regime"] == r]
            fig.add_trace(
                go.Scattergl(
                    x=sub.index,
                    y=sub["regime"],
                    mode="markers",
                    marker=dict(color=_regime_color(r), size=6),
                    name=f"régime {r}",
                    legendgroup=f"reg{r}",
                    hovertemplate="%{x}<br>régime %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    # --- Q_time / Q_space : score + seuil régime + dépassements ---
    for row, score_col, thr_key in (
        (3, "score_Q_time", "Q_time"),
        (4, "score_Q_space", "Q_space"),
    ):
        if score_col not in series.columns:
            continue
        fig.add_trace(
            go.Scattergl(
                x=series.index,
                y=series[score_col],
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                name=score_col,
                showlegend=False,
                hovertemplate="%{x}<br>" + score_col + "=%{y:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        if thresholds:
            thr = _threshold_step(series, thresholds, thr_key)
            fig.add_trace(
                go.Scattergl(
                    x=series.index, y=thr, mode="lines",
                    line=dict(color="#d62728", width=1, dash="dot", shape="hv"),
                    name="seuil", showlegend=(row == 3), legendgroup="seuil",
                    hovertemplate="seuil=%{y:.2f}<extra></extra>",
                ),
                row=row, col=1,
            )
            exceed = series[score_col] > thr
            if exceed.any():
                ex = series[exceed]
                fig.add_trace(
                    go.Scattergl(
                        x=ex.index, y=ex[score_col], mode="markers",
                        marker=dict(color="#d62728", size=4),
                        name="dépassement", showlegend=(row == 3),
                        legendgroup="exceed",
                        hovertemplate="%{x}<br>dépassement " + score_col
                        + "=%{y:.2f}<extra></extra>",
                    ),
                    row=row, col=1,
                )

    fig.update_layout(
        height=760,
        margin=dict(l=60, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="régime", row=2, col=1, dtick=1)
    return fig


def physical_figure(
    df: pd.DataFrame,
    regime: pd.Series,
    mode: str,
    *,
    baseline: tuple[float, float, float] | None = None,
) -> go.Figure:
    """Vue physique/interprétable alignée sur la timeline.

    ``mode`` :
    - ``"Rendement EE/HP"`` : ratio EE/HP coloré par régime + médiane glissante 1 j ;
      ``baseline`` = (médiane, q25, q75) de référence tracée en bande.
    - ``"Variables brutes"`` : HP, BP, EE en sous-graphes.
    """
    reg = regime.reindex(df.index)

    if mode == "Variables brutes":
        cols = [c for c in ("HP", "BP", "EE") if c in df.columns]
        fig = make_subplots(
            rows=len(cols), cols=1, shared_xaxes=True,
            vertical_spacing=0.04, subplot_titles=cols,
        )
        for i, c in enumerate(cols, start=1):
            fig.add_trace(
                go.Scattergl(
                    x=df.index, y=df[c], mode="lines",
                    line=dict(color="#1f77b4", width=1),
                    name=c, showlegend=False,
                    hovertemplate="%{x}<br>" + c + "=%{y:.2f}<extra></extra>",
                ),
                row=i, col=1,
            )
        fig.update_layout(
            height=160 * len(cols), margin=dict(l=60, r=20, t=40, b=30),
            hovermode="x unified",
        )
        return fig

    # --- Rendement EE/HP ---
    ratio = pd.Series(np.nan, index=df.index)
    if {"EE", "HP"}.issubset(df.columns):
        ok = df["HP"] > 1.0  # éviter la division près de l'arrêt
        ratio[ok] = df.loc[ok, "EE"] / df.loc[ok, "HP"]

    fig = go.Figure()
    if baseline is not None:
        med, q25, q75 = baseline
        fig.add_hrect(y0=q25, y1=q75, fillcolor="#2ca02c", opacity=0.12, line_width=0)
        fig.add_hline(y=med, line=dict(color="#2ca02c", width=1, dash="dash"),
                      annotation_text="référence (sain)", annotation_position="right")

    for r in sorted(int(v) for v in reg.dropna().unique()):
        m = (reg == r) & ratio.notna()
        if not m.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=ratio.index[m], y=ratio[m], mode="markers",
                marker=dict(color=_regime_color(r), size=4),
                name=f"régime {r}", legendgroup=f"reg{r}",
                hovertemplate="%{x}<br>EE/HP=%{y:.3f}<br>régime " + str(r)
                + "<extra></extra>",
            )
        )
    # Médiane glissante 1 jour (96 pas) pour la tendance.
    roll = ratio.rolling(96, min_periods=12, center=True).median()
    fig.add_trace(
        go.Scattergl(
            x=roll.index, y=roll, mode="lines",
            line=dict(color="#000000", width=1.5),
            name="médiane glissante 1 j",
            hovertemplate="%{x}<br>médiane EE/HP=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=380, margin=dict(l=60, r=20, t=30, b=30),
        yaxis_title="Rendement EE/HP",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


def energetic_figure(resid: pd.DataFrame, band: float, ref_end: str | None = None) -> go.Figure:
    """Résidu énergétique EE (%) dans le temps + bande de contrôle saine.

    Points dans la bande en gris, hors bande en rouge ; médiane glissante 1 jour
    pour la tendance ; trait vertical = fin de la période de référence.
    """
    fig = go.Figure()
    r = resid["resid_pct"]
    if band and np.isfinite(band):
        fig.add_hrect(y0=-band, y1=band, fillcolor="#2ca02c", opacity=0.10, line_width=0)
    fig.add_hline(y=0, line=dict(color="#2ca02c", width=1, dash="dash"))

    inb = r.abs() <= band if band and np.isfinite(band) else pd.Series(True, index=r.index)
    fig.add_trace(go.Scattergl(
        x=r.index[inb], y=r[inb], mode="markers",
        marker=dict(size=3, color="#7f7f7f"), name="dans la bande",
        hovertemplate="%{x}<br>résidu %{y:+.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=r.index[~inb], y=r[~inb], mode="markers",
        marker=dict(size=3, color="#d62728"), name="hors bande",
        hovertemplate="%{x}<br>résidu %{y:+.1f}%<extra></extra>",
    ))
    roll = r.rolling(96, min_periods=12, center=True).median()
    fig.add_trace(go.Scattergl(
        x=roll.index, y=roll, mode="lines",
        line=dict(color="black", width=1.6), name="médiane glissante 1 j",
        hovertemplate="%{x}<br>tendance %{y:+.1f}%<extra></extra>",
    ))
    if ref_end:
        try:
            fig.add_vline(x=pd.Timestamp(ref_end), line=dict(color="#1f77b4", width=1, dash="dot"),
                          annotation_text="fin référence", annotation_position="top left")
        except Exception:  # noqa: BLE001 - annotation non bloquante
            pass
    fig.update_layout(
        height=340, margin=dict(l=60, r=20, t=30, b=30),
        yaxis_title="résidu EE (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


_DYN_LABELS = {
    "vol_dHP": "Volatilité ΔHP", "vol_dBP": "Volatilité ΔBP",
    "vol_dEE": "Volatilité ΔEE", "vol_resid": "Volatilité résidu EE",
    "ac1_resid": "Autocorr lag-1 résidu", "coup_dHP_dEE": "Couplage ΔHP–ΔEE",
    "coup_dBP_dEE": "Couplage ΔBP–ΔEE", "coup_dHP_dresid": "Couplage ΔHP–Δrésidu",
}


def dynamic_figure(feats: pd.DataFrame, meta: dict) -> go.Figure:
    """Traceur dynamique invariant au niveau : une feature par sous-graphe.

    INDICATIF. La zone verte est une **référence visuelle** (médiane ± k·IQR sur
    le sain), PAS une limite de contrôle ; les points hors zone sont juste
    marqués (« hors zone de référence »), aucune alerte n'en découle.
    """
    band = meta.get("band", {})
    cols = [c for c in feats.columns if c in band]
    n = len(cols)
    titles = [_DYN_LABELS.get(c, c) for c in cols]
    fig = make_subplots(rows=max(1, n), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=titles)
    for i, c in enumerate(cols, start=1):
        b = band[c]
        s = feats[c]
        if np.isfinite(b.get("lo", np.nan)) and np.isfinite(b.get("hi", np.nan)):
            fig.add_hrect(y0=b["lo"], y1=b["hi"], fillcolor="#2ca02c", opacity=0.10,
                          line_width=0, row=i, col=1)
            fig.add_hline(y=b["med"], line=dict(color="#2ca02c", width=1, dash="dash"),
                          row=i, col=1)
            out = (s < b["lo"]) | (s > b["hi"])
        else:
            out = pd.Series(False, index=s.index)
        fig.add_trace(go.Scattergl(
            x=s.index, y=s, mode="lines", line=dict(color="#1f77b4", width=1),
            name=_DYN_LABELS.get(c, c), showlegend=False,
            hovertemplate="%{x}<br>" + _DYN_LABELS.get(c, c) + "=%{y:.3f}<extra></extra>",
        ), row=i, col=1)
        if out.any():
            fig.add_trace(go.Scattergl(
                x=s.index[out], y=s[out], mode="markers",
                marker=dict(color="#7f7f7f", size=3), name="hors zone de référence",
                showlegend=(i == 1),
                hovertemplate="%{x}<br>hors zone : %{y:.3f}<extra></extra>",
            ), row=i, col=1)
    fig.update_layout(
        height=140 * max(1, n), margin=dict(l=60, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
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
