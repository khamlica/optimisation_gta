"""Validation modèle — crédibilité opérationnelle, vue cross-GTA.

Agrège FAR avant/après, effectifs de régimes, couverture, et les tests de
dérive injectée / pic isolé. Répond à : « le modèle est-il stable, et où sont
ses limites ? » — sans conclure trop vite.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import readers


def render() -> None:
    st.title("Validation modèle — crédibilité opérationnelle")

    summary = readers.global_summary()
    if summary.empty:
        st.warning("`summary_all_gta.csv` introuvable.")
        return

    st.subheader("Synthèse cross-GTA")
    cols = [
        c
        for c in (
            "gta_id", "exclude_vars", "FAR_calib_before", "FAR_calib_after",
            "n_modeled", "n_insufficient_regimes", "n_unknown_regime",
            "n_scored_total", "n_non_scored_total", "n_normal", "n_warning", "n_alert",
        )
        if c in summary.columns
    ]
    st.dataframe(summary[cols], width='stretch', hide_index=True)

    # --- Tests de dérive injectée / pic isolé par GTA ---
    st.subheader("Tests synthétiques (dérive injectée, pic isolé)")
    rows = []
    for gta in summary["gta_id"]:
        rep = readers.online_report(str(gta))
        drift = rep.get("drift_injected", {}) or {}
        spike = rep.get("isolated_spike", {}) or {}
        rows.append(
            {
                "gta_id": gta,
                "dérive_régime": drift.get("regime"),
                "délai_détection_min": drift.get("delay_minutes"),
                "1er_alert_window": drift.get("first_alert_window"),
                "pic_isolé_alert": spike.get("has_alert"),
                "pic_isolé_warning": spike.get("has_warning"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption(
        "Attendu : la dérive injectée déclenche un `alert` avec un délai fini ; "
        "le pic isolé donne au plus un `warning` (jamais d'`alert`)."
    )

    # --- Points d'attention connus ---
    st.subheader("Points d'attention (à examiner, sans surinterpréter)")
    st.markdown(
        "- **FAR JFC3** légèrement au-dessus de la cible (1 seul régime modélisé "
        "après exclusion MP).\n"
        "- **Régime 2 JFC1** : FAR test élevé malgré une calibration saine → "
        "dérive réelle probable, à confirmer dans *Diagnostic alerte*.\n"
        "- **Poids des zones non scorées** (`insufficient_data`, `transition`) : "
        "un GTA peu scoré est peu observable, pas « sain ».\n"
        "- **Effets de couverture** : comparer `n_scored_total` vs `n_non_scored_total`."
    )

    # --- Figures FAR vs quantile (référence) ---
    st.subheader("FAR vs quantile de seuil (référence visuelle)")
    cols2 = st.columns(min(4, len(summary)))
    for col, gta in zip(cols2, summary["gta_id"]):
        p = readers.figure_path(str(gta), "far_vs_quantile.png")
        with col:
            st.markdown(f"**{gta}**")
            if p:
                st.image(str(p), width='stretch')
            else:
                st.caption("—")


render()
