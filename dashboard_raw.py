"""Dashboard INDÉPENDANT et simpliste : visualisation des données brutes GTA
avec mise en couleur des périodes exclues avant modélisation.

Objectif : vérifier *à l'œil* que le prétraitement écarte les bonnes périodes
(arrêts, trous, hors-bornes, capteurs bloqués, transitions de régime) et pas
des périodes saines. C'est un outil d'audit du PRÉTRAITEMENT, pas du modèle.

Il est volontairement séparé de l'app multipage (`app.py`) : il ne lit aucun
artefact, il recalcule tout depuis les CSV bruts en réutilisant exactement les
fonctions de masquage du pipeline (`preprocessing`, `regimes`, `windows`).

Lancer :  streamlit run dashboard_raw.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from bi2dpca import config, io_data
from bi2dpca.preprocessing import (
    _long_gap_mask,
    _out_of_range_mask,
    _stuck_sensor_mask,
    preprocess,
)
from bi2dpca.regimes import identify_regimes
from bi2dpca.windows import stop_mask

# --------------------------------------------------------------------------- #
# Catégories d'audit du prétraitement
# --------------------------------------------------------------------------- #
# Ordre = priorité d'affectation (la dernière clé écrase les précédentes quand
# un même pas coche plusieurs motifs). On veut que le motif le plus « bas
# niveau » (donnée absente) l'emporte sur l'étiquette de transition.
CAT_KEPT = "Gardé (surveillé)"
CAT_TRANSITION = "Transition / non labellisé"
CAT_STOP = "Arrêt machine"
CAT_STUCK = "Capteur bloqué"
CAT_RANGE = "Hors bornes physiques"
CAT_GAP = "Trou / valeur manquante"

COLORS = {
    CAT_KEPT: "#2ca02c",
    CAT_TRANSITION: "#ff7f0e",
    CAT_STOP: "#7f7f7f",
    CAT_STUCK: "#9467bd",
    CAT_RANGE: "#d62728",
    CAT_GAP: "#1f77b4",
}
ORDER = [CAT_KEPT, CAT_TRANSITION, CAT_STOP, CAT_STUCK, CAT_RANGE, CAT_GAP]

# Palette qualitative pour les régimes (coloration alternative).
NON_LABEL = "Non labellisé"
REGIME_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _ranges_with_k(df: pd.DataFrame, variables: list[str], k: float):
    """Bornes physiques Q1−k·IQR / Q3+k·IQR (même formule que ``_auto_ranges``,
    mais avec un facteur ``k`` réglable depuis l'interface)."""
    ranges: dict[str, tuple[float, float]] = {}
    for v in variables:
        s = df[v].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = float(q1 - k * iqr), float(q3 + k * iqr)
        if not np.isfinite(iqr) or iqr <= 0:  # variable quasi constante
            lo, hi = float(s.min()), float(s.max())
        ranges[v] = (lo, hi)
    return ranges


@st.cache_data(show_spinner="Calcul du prétraitement…")
def compute(
    gta_id: str,
    *,
    iqr_k: float,
    long_gap_steps: int,
    stuck_window: int,
    stuck_min_std: float,
    stop_frac: float,
    transition_steps: int,
    regime_smooth_window: int,
    regime_label_smooth_window: int,
    max_regimes: int,
):
    """Charge les données brutes et recalcule tous les masques du prétraitement.

    Tous les hyperparamètres de prétraitement sont injectés ici : changer l'un
    d'eux invalide le cache et relance entièrement la classification.

    Renvoie le DataFrame brut affichable (toutes variables, grille régulière),
    la série de catégories par pas, le report de préfiltrage et les bornes.
    """
    path = io_data.resolve_data_path(gta_id)

    # 1) Données brutes COMPLÈTES (pour l'affichage : on montre tout, même MP).
    raw_all = io_data.load_gta(path, gta_id, exclude_vars=())

    # 2) Pipeline FIDÈLE au modèle (avec les exclusions de variables décidées,
    #    ex. MP pour JFC3) -> masques cohérents avec ce que le modèle voit.
    excl = config.exclude_vars_for(gta_id)
    data = io_data.load_gta(path, gta_id, exclude_vars=excl)

    # Bornes physiques calculées avec le facteur IQR choisi (les quantiles sont
    # identiques sur données brutes ou réindexées : la grille n'ajoute que des NaN).
    ranges = _ranges_with_k(data.df, data.variables, iqr_k)
    params = replace(
        config.DEFAULT_PARAMS,
        long_gap_steps=long_gap_steps,
        stuck_window=stuck_window,
        stuck_min_std=stuck_min_std,
        stop_frac=stop_frac,
        transition_steps=transition_steps,
        regime_smooth_window=regime_smooth_window,
        regime_label_smooth_window=regime_label_smooth_window,
        regime_n_components_grid=tuple(range(2, max_regimes + 1)),
        physical_ranges=ranges,
    )
    pre = preprocess(data, params)

    grid = pre.df.index
    df_disp = raw_all.df.reindex(grid)  # mêmes pas que la grille régulière

    # Masques élémentaires (recalculés via les fonctions du pipeline).
    missing = pre.df.isna().any(axis=1)
    out_of_range = _out_of_range_mask(pre.df, pre.variables, pre.ranges)
    long_gap = _long_gap_mask(pre.df, params.long_gap_steps)
    stuck = _stuck_sensor_mask(
        pre.df, pre.variables, params.stuck_window, params.stuck_min_std
    )

    reg = identify_regimes(pre, params)
    stop = stop_mask(pre, params)

    # Affectation des catégories par priorité croissante.
    cat = pd.Series(CAT_KEPT, index=grid)
    cat[reg.transition.to_numpy()] = CAT_TRANSITION
    cat[stop.to_numpy()] = CAT_STOP
    cat[stuck.to_numpy()] = CAT_STUCK
    cat[out_of_range.to_numpy()] = CAT_RANGE
    cat[(missing | long_gap).to_numpy()] = CAT_GAP

    info = {
        "report": pre.report,
        "ranges": pre.ranges,
        "n_regimes": int(reg.n_regimes),
        "excluded_vars": excl,
        "exclude_reason": config.exclude_reason_for(gta_id),
        "disp_vars": raw_all.variables,
        "regime": reg.regime,
        "dt_minutes": params.dt_minutes,
    }
    return df_disp, cat, info


def regime_classes(regime: pd.Series):
    """Construit la série de classes « régime » + palette + ordre, dans le même
    format que les catégories de statut (réutilisable par ``build_figure``)."""
    cat = pd.Series(NON_LABEL, index=regime.index)
    colors: dict[str, str] = {}
    order: list[str] = []
    present = sorted(int(v) for v in pd.unique(regime.to_numpy()) if v >= 0)
    for i, v in enumerate(present):
        name = f"Régime {v}"
        cat[(regime == v).to_numpy()] = name
        colors[name] = REGIME_PALETTE[i % len(REGIME_PALETTE)]
        order.append(name)
    order.append(NON_LABEL)
    colors[NON_LABEL] = "#cccccc"
    return cat, colors, order


def build_figure(df_disp, disp_vars, ribbons, curve_cat, curve_colors, curve_order):
    """Construit la figure : N rubans de classes empilés + une courbe par variable.

    ``ribbons`` est une liste de ``(cat, colors, order, titre)`` : chaque entrée
    devient un ruban horizontal (ex. Statut, puis Régime). Les courbes de
    variables sont colorées par ``curve_cat`` (ici le statut, pour voir les
    transitions directement sur le signal).
    """
    n = len(disp_vars)
    nr = len(ribbons)
    rib_h = 0.09
    titles = [t for (_, _, _, t) in ribbons] + disp_vars
    fig = make_subplots(
        rows=nr + n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[rib_h] * nr + [(1.0 - rib_h * nr) / n] * n,
        subplot_titles=titles,
    )

    x = df_disp.index
    seen_legend: set[str] = set()

    # --- Rubans empilés : une bande verticale colorée par classe ------------ #
    for ri, (cat, colors, order, _title) in enumerate(ribbons, start=1):
        for c in order:
            m = (cat == c).to_numpy()
            if not m.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=x[m],
                    y=np.zeros(m.sum()),
                    mode="markers",
                    marker=dict(symbol="line-ns", size=18, color=colors[c],
                                line=dict(width=2, color=colors[c])),
                    name=c,
                    legendgroup=c,
                    showlegend=c not in seen_legend,
                    hovertemplate=f"{c}<br>%{{x}}<extra></extra>",
                ),
                row=ri,
                col=1,
            )
            seen_legend.add(c)
        fig.update_yaxes(visible=False, row=ri, col=1)

    # --- Une courbe par variable : ligne grise + points colorés (statut) ---- #
    for k, v in enumerate(disp_vars):
        r = k + 1 + nr
        y = df_disp[v]
        # Ligne de fond grise (continuité visuelle).
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="rgba(150,150,150,0.4)", width=1),
                name="série", showlegend=False, hoverinfo="skip",
            ),
            row=r, col=1,
        )
        # Points colorés par statut (les pas sans valeur — trous — n'ont pas de
        # marqueur ici mais restent visibles dans le ruban).
        for c in curve_order:
            m = ((curve_cat == c) & y.notna()).to_numpy()
            if not m.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=x[m], y=y[m], mode="markers",
                    marker=dict(size=4, color=curve_colors[c]),
                    name=c, legendgroup=c,
                    showlegend=c not in seen_legend,
                    hovertemplate=f"{v}=%{{y:.2f}}<br>{c}<br>%{{x}}<extra></extra>",
                ),
                row=r, col=1,
            )
            seen_legend.add(c)
        fig.update_yaxes(title_text=v, row=r, col=1)

    fig.update_layout(
        height=120 + 90 * nr + 220 * n,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Audit prétraitement GTA", layout="wide")
st.title("🩺 Audit du prétraitement — données brutes & périodes exclues")
st.caption(
    "Dashboard indépendant : recalcule tout depuis les CSV bruts. "
    "On vérifie que les périodes écartées avant modélisation correspondent "
    "bien à des arrêts / trous / aberrations / transitions, et pas à des "
    "périodes saines."
)

D = config.DEFAULT_PARAMS  # valeurs par défaut, pour les widgets

gtas = list(config.GTA_CONFIGS)
gta = st.sidebar.selectbox("GTA", gtas, index=0)

st.sidebar.markdown("### Paramètres de prétraitement")
st.sidebar.caption(
    "Modifier un paramètre relance la classification et actualise le graphe. "
    "Les réglages régime (lissage, nb régimes) recalculent le clustering "
    "(quelques secondes)."
)

with st.sidebar.expander("Bornes & trous", expanded=True):
    iqr_k = st.slider(
        "Facteur IQR (hors bornes)", 1.0, 10.0, value=5.0, step=0.5,
        help="Bornes physiques = Q1 − k·IQR  …  Q3 + k·IQR. Plus k est petit, "
             "plus on coupe de points en « hors bornes ».",
    )
    long_gap_steps = st.slider(
        "Trou long (pas)", 1, 48, value=int(D.long_gap_steps), step=1,
        help="Un trou de plus de N pas consécutifs manquants est marqué « trou ».",
    )

with st.sidebar.expander("Capteur bloqué", expanded=False):
    stuck_window = st.slider(
        "Fenêtre (pas)", 2, 48, value=int(D.stuck_window), step=1,
        help="Nombre de pas sur lesquels on mesure la variance glissante.",
    )
    stuck_min_std = st.select_slider(
        "Seuil σ (variance min.)",
        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=float(D.stuck_min_std),
        format_func=lambda x: f"{x:.0e}",
        help="En-dessous de cet écart-type glissant, le capteur est jugé bloqué.",
    )

with st.sidebar.expander("Arrêt machine", expanded=True):
    stop_frac = st.slider(
        "Fraction d'arrêt", 0.0, 0.50, value=float(D.stop_frac), step=0.01,
        help="Arrêt si la charge (EE) < fraction × médiane opérationnelle.",
    )

with st.sidebar.expander("Régimes & transition", expanded=True):
    transition_steps = st.slider(
        "Marge de transition (pas)", 1, 16, value=int(D.transition_steps), step=1,
        help="Pas exclus APRÈS chaque changement de régime (rampe de "
             "stabilisation). Découplé de la taille de fenêtre. 2 pas = 30 min.",
    )
    st.caption(f"→ marge = {transition_steps} pas "
               f"({transition_steps * D.dt_minutes} min)")
    regime_smooth_window = st.slider(
        "Lissage variables régime (pas)", 1, 48,
        value=int(D.regime_smooth_window), step=1,
        help="Médiane glissante sur HP/MP/BP avant clustering.",
    )
    regime_label_smooth_window = st.slider(
        "Lissage labels régime (pas)", 1, 48,
        value=int(D.regime_label_smooth_window), step=1,
        help="Vote majoritaire glissant sur les labels (anti-papillotement).",
    )
    max_regimes = st.slider(
        "Nb max de régimes (grille GMM)", 2, 8, value=5, step=1,
        help="Le BIC choisit le meilleur nombre dans 2..N.",
    )

try:
    df_disp, cat, info = compute(
        gta,
        iqr_k=iqr_k,
        long_gap_steps=long_gap_steps,
        stuck_window=stuck_window,
        stuck_min_std=stuck_min_std,
        stop_frac=stop_frac,
        transition_steps=transition_steps,
        regime_smooth_window=regime_smooth_window,
        regime_label_smooth_window=regime_label_smooth_window,
        max_regimes=max_regimes,
    )
except ValueError as exc:
    st.error(f"Calcul impossible avec ces paramètres : {exc}")
    st.stop()

# --- Bandeau de synthèse ---------------------------------------------------- #
rep = info["report"]
n = rep["n_points"]
counts = cat.value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pas (grille 15 min)", f"{n:,}".replace(",", " "))
kept = int(counts.get(CAT_KEPT, 0))
c2.metric("Gardé (surveillé)", f"{kept:,}".replace(",", " "), f"{100*kept/n:.1f} %")
excl_pts = n - kept
c3.metric("Exclu avant modélisation", f"{excl_pts:,}".replace(",", " "),
          f"{100*excl_pts/n:.1f} %", delta_color="inverse")
c4.metric("Régimes détectés", info["n_regimes"])

if info["excluded_vars"]:
    st.info(
        f"Variable(s) exclue(s) du modèle pour {gta} : "
        f"**{', '.join(info['excluded_vars'])}** — {info['exclude_reason']}. "
        "Elles restent affichées ci-dessous à titre indicatif."
    )

# --- Classes de régime (ruban additionnel) ---------------------------------- #
reg_cat, reg_colors, reg_order = regime_classes(info["regime"])

# --- Répartition : statut + régime ------------------------------------------ #
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Répartition par statut**")
    tab_s = (
        pd.DataFrame({"pas": [int(counts.get(c, 0)) for c in ORDER]}, index=ORDER)
        .assign(part=lambda d: (100 * d["pas"] / n).round(2).astype(str) + " %")
    )
    st.dataframe(tab_s, width='stretch')
with col_b:
    st.markdown("**Répartition par régime**")
    reg_counts = reg_cat.value_counts()
    tab_r = (
        pd.DataFrame(
            {"pas": [int(reg_counts.get(c, 0)) for c in reg_order]}, index=reg_order
        )
        .assign(part=lambda d: (100 * d["pas"] / n).round(2).astype(str) + " %")
    )
    st.dataframe(tab_r, width='stretch')

# --- Figure principale : ruban Statut + ruban Régime + courbes --------------- #
# Les courbes sont colorées par statut (pour repérer les transitions sur le
# signal) ; le ruban Régime donne, en parallèle, le régime de chaque pas.
st.plotly_chart(
    build_figure(
        df_disp,
        info["disp_vars"],
        ribbons=[
            (cat, COLORS, ORDER, "Statut (audit prétraitement)"),
            (reg_cat, reg_colors, reg_order, "Régimes (clustering GMM)"),
        ],
        curve_cat=cat,
        curve_colors=COLORS,
        curve_order=ORDER,
    ),
    width='stretch',
)

# --- Détails techniques ----------------------------------------------------- #
with st.expander("Détails du préfiltrage (compteurs bruts & bornes)"):
    st.json(rep)
    st.write(f"Bornes physiques utilisées (Q1 − {iqr_k}·IQR, Q3 + {iqr_k}·IQR) :")
    st.json({k: [round(a, 2), round(b, 2)] for k, (a, b) in info["ranges"].items()})
