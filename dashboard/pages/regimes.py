"""Régimes — séparer problème procédé et problème de modélisation.

Rend visibles : régime sous-peuplé, seuil instable, FAR test élevé malgré FAR
calib sain, exclusion du régime, variables exclues côté GTA (ex. régime 2 JFC1).
"""

from __future__ import annotations

import streamlit as st

from dashboard import charts, readers, state


def render() -> None:
    gta = st.session_state.get("gta") or (readers.list_gtas() or [None])[0]
    if not gta:
        st.warning("Aucun GTA disponible.")
        return
    st.title(f"Régimes — {gta}")

    reg = readers.regime_metrics(gta)
    m = readers.metrics(gta)
    excl = m.get("exclude_vars", [])
    if excl:
        st.warning(f"Variables exclues côté GTA : {excl} — {m.get('exclude_reason', '')}")

    if reg.empty:
        st.info("`regime_summary.csv` absent pour ce GTA.")
        return

    st.subheader("Tableau par régime")
    st.dataframe(reg, width='stretch', hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if "n_train" in reg.columns:
            st.plotly_chart(
                charts.regime_bars(reg, "n_train", "Effectif train par régime"),
                width='stretch',
            )
    with c2:
        if "far_test" in reg.columns:
            st.plotly_chart(
                charts.regime_bars(reg, "far_test", "FAR test par régime"),
                width='stretch',
            )

    # Lecture guidée des cas d'attention.
    st.subheader("Points d'attention détectés")
    notes = []
    for _, r in reg.iterrows():
        rid = r.get("regime")
        if r.get("status") == "insufficient_data":
            notes.append(
                f"• Régime {rid} : **insufficient_data** "
                f"(train={r.get('n_train')}, calib={r.get('n_calib')}) → exclu."
            )
            continue
        far_c, far_t = r.get("far_calib"), r.get("far_test")
        if far_c == far_c and far_t == far_t and far_c < 0.03 and far_t > 0.20:
            notes.append(
                f"• Régime {rid} : **FAR test élevé** ({100*far_t:.0f}%) malgré FAR "
                f"calib sain ({100*far_c:.1f}%) → dérive réelle probable ou régime hétérogène."
            )
    if notes:
        st.markdown("\n\n".join(notes))
    else:
        st.success("Aucun point d'attention particulier sur ce GTA.")


render()
