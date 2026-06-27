"""Qualité données — ne pas confondre dérive procédé et dégradation capteur.

Centrée sur `diagnostic_mp.json` (cas JFC3) et la comparaison avec/sans la
variable exclue : NaN, zéros, quasi-zéros, capteur bloqué, impact sur la
couverture.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import readers


def render() -> None:
    st.title("Qualité des données — capteurs & exclusions")

    gtas = readers.list_gtas()
    if not gtas:
        st.warning("Aucun artefact.")
        return
    gta = st.session_state.get("gta") or gtas[0]
    gta = st.selectbox("GTA", gtas, index=gtas.index(gta) if gta in gtas else 0)

    st.markdown(
        "**Message clé : une dérive procédé et une dégradation capteur ne sont "
        "pas interchangeables.** Cette page aide à ne pas les confondre."
    )

    dq = readers.data_quality(gta)
    m = readers.metrics(gta)
    excl = m.get("exclude_vars", [])
    if excl:
        st.warning(f"Variables exclues : {excl} — {m.get('exclude_reason', '')}")

    if not dq:
        st.info(
            f"Pas de `diagnostic_mp.json` pour {gta}. "
            "Ce diagnostic est généré par `diagnose_jfc3.py` (cas JFC3)."
        )
        return

    # --- Qualité par variable ---
    st.subheader("Qualité capteur par variable")
    per_var = dq.get("per_variable", {})
    if per_var:
        df = pd.DataFrame(per_var).T.reset_index().rename(columns={"index": "variable"})
        st.dataframe(df, width='stretch', hide_index=True)
        st.caption(
            "Un canal avec zéros/quasi-zéros et `stuck_pct` élevés est dégénéré "
            "(ex. MP sur JFC3 : ~82% de zéros, ~76% bloqué)."
        )

    # --- Impact de l'exclusion ---
    st.subheader("Impact de l'exclusion sur la couverture")
    impact = dq.get("mp_impact", {})
    if impact:
        c1, c2 = st.columns(2)
        c1.metric(
            "exploitable %",
            f"{impact.get('exploitable_pct_without_MP', '—')}",
            delta=f"+{impact.get('exploitable_gain_pts', 0)} pts vs avec MP",
        )
        c2.metric(
            "monitorable %",
            f"{impact.get('monitorable_pct_without_MP', '—')}",
            delta=f"+{impact.get('monitorable_gain_pts', 0)} pts vs avec MP",
        )

    # --- Comparaison avec / sans variable ---
    comp = readers.jfc3_comparison()
    if not comp.empty:
        st.subheader("Pipeline complet : avec vs sans la variable")
        st.dataframe(comp, width='stretch', hide_index=True)


render()
