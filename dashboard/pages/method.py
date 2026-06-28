"""Méthode — documentation embarquée (Flux / Glossaire / Statuts / Sources)."""

from __future__ import annotations

import streamlit as st

from dashboard import readers, state

_FLOW = """
```mermaid
flowchart LR
    A[online_status.csv / flux live] --> B[Assignation régime]
    B --> C{Transition ?}
    C -- Oui --> T[transition · non scoré]
    C -- Non --> D{Régime insuffisant ?}
    D -- Oui --> I[insufficient_data · non scoré]
    D -- Non --> E[Scores Q_time / Q_space]
    E --> F{Seuil dépassé ?}
    F -- Non --> N[normal]
    F -- Oui --> G[Persistance / warnings cumulés]
    G --> H{Persistance atteinte ?}
    H -- Non --> W[warning]
    H -- Oui --> AL[alert]
```
"""


def render() -> None:
    st.title("Méthode — Bi2DPCA dynamique par régime")

    tab_flux, tab_gloss, tab_status, tab_src = st.tabs(
        ["Flux", "Glossaire", "Statuts", "Sources de données"]
    )

    with tab_flux:
        st.markdown(
            "Pipeline : **chargement → prétraitement → régimes → fenêtres 2D → "
            "apprentissage C2DPCA-R2DPCA par régime → seuils → scoring online "
            "avec persistance.**"
        )
        st.info(
            "`EE` est une variable **surveillée, jamais prédite**. À 15 min, les "
            "indices `Q_time`/`Q_space` mesurent surtout un **écart à la baseline "
            "multivariable du régime** (mélange structure croisée + niveau de "
            "fonctionnement), **pas** une micro-dynamique locale pure : un "
            "déplacement de point de fonctionnement (rendement) les fait réagir."
        )
        st.markdown(
            "**Triptyque de surveillance** (rôles séparés) :\n"
            "1. **Performance statique** — cross-check énergétique `EE=f(HP,BP,MP)` "
            "(robuste, non masquable) ;\n"
            "2. **Dynamique locale** — traceur invariant au niveau (volatilité, "
            "autocorr, couplages d'incréments) — *indicatif* ;\n"
            "3. **Écart à la baseline du régime** — Bi2DPCA `Q_time`/`Q_space` "
            "(sensible, précoce, mais sensible au déplacement de point de "
            "fonctionnement)."
        )
        try:
            st.markdown(_FLOW)
        except Exception:  # noqa: BLE001 - mermaid non rendu : repli texte
            st.code(_FLOW, language="text")

    with tab_gloss:
        st.markdown(
            "- **C2DPCA** : réduction de la dimension *temporelle* (autocorrélation).\n"
            "- **R2DPCA** : réduction de la dimension *variables* (structure croisée).\n"
            "- **Q_time / Q_space** : erreurs de reconstruction (anomalie).\n"
            "- **T2_time / T2_space** : énergie dans les sous-espaces retenus (optionnel).\n"
            "- **CPV** : variance cumulée pour choisir `d` (temporel) et `p` (spatial).\n"
            "- **Persistance** : un `alert` exige un dépassement soutenu dans un même régime.\n"
            "- **FAR** : taux de fausses alarmes (calibré par régime)."
        )

    with tab_status:
        st.markdown("Code couleur **fixe** dans toute l'app :")
        for s in state.STATUS_ORDER:
            scored = "scoré" if s in state.SCORED_STATUSES else "**non scoré**"
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{state.STATUS_COLORS[s]};border-radius:3px;"
                f"margin-right:8px'></span>{state.STATUS_LABELS[s]} — {scored}",
                unsafe_allow_html=True,
            )
        st.caption(
            "`transition`, `insufficient_data`, `unknown_regime` sont **non scorés** : "
            "le GTA n'est pas observable sur ces fenêtres (≠ normal)."
        )

    with tab_src:
        st.markdown("Le dashboard est **lecture seule** sur ces artefacts :")
        st.markdown(
            "- `summary_all_gta.csv` — comparaison inter-GTA\n"
            "- `artifacts/<GTA>/online_status.csv` — série temporelle\n"
            "- `artifacts/<GTA>/online_report.json` — comptages, tests\n"
            "- `artifacts/<GTA>/metrics.json` — métriques offline, exclusions\n"
            "- `artifacts/<GTA>/regime_summary.csv` — diagnostic par régime\n"
            "- `artifacts/<GTA>/diagnostic_mp.json` — qualité capteur (JFC3)\n"
            "- `artifacts/<GTA>/bundle.pkl` — *complément* (seuils, variables, t)"
        )
        st.caption(f"Racine des artefacts : `{readers.ARTIFACTS_DIR}`")


render()
