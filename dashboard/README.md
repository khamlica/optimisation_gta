# Dashboard Bi2DPCA GTA — cockpit & audit

Application Streamlit **multipage**, en **lecture seule** sur les artefacts du
pipeline (`artifacts/<GTA>/…` et `summary_all_gta.csv`). Elle ne recalcule rien
du modèle : elle **visualise** et **explique** les résultats.

## Lancement

```bash
/home/khamlica/venv/bin/streamlit run app.py
```

(depuis le dossier `modele_gta/`). Pré-requis : avoir généré les artefacts via

```bash
python run_offline.py --gta all
python run_online.py  --gta all
python diagnose_jfc3.py   # pour la page Qualité données (cas JFC3)
```

## Architecture

```
app.py                      # entrée : st.navigation + sidebar globale (GTA, mode)
dashboard/
  readers.py                # contrat de données (CSV/JSON-first, bundle-second), caches
  state.py                  # session_state + code couleur global des statuts
  charts.py                 # graphiques Plotly réutilisables
  pages/
    overview.py             # Vue globale (cockpit) — défaut
    monitoring.py           # Monitoring temporel (Replay + Live)
    alert_diagnostic.py     # Pourquoi cette alerte ? (comparatif fenêtre)
    regimes.py              # Diagnostic par régime
    model_validation.py     # Crédibilité du modèle (cross-GTA)
    data_quality.py         # Qualité capteur / exclusions (cas JFC3)
    method.py               # Documentation embarquée
```

## Principes de lecture

- **Overview first, zoom & filter, details on demand** : Vue globale → Monitoring
  → Diagnostic alerte.
- **Code couleur fixe** : `normal` vert, `warning` orange, `alert` rouge,
  `transition` gris, `insufficient_data` violet, `unknown_regime` bleu-gris.
  On ne fond jamais `transition` et `alert`.
- **Non scoré ≠ normal** : `transition` / `insufficient_data` / `unknown_regime`
  signifient *non observable*.
- **`EE` est surveillée, jamais prédite** ; les alertes viennent d'une rupture
  de structure (`Q_time`, `Q_space`).

## Modes

- **Replay** : exploration historique, filtres, click-to-drill, export CSV.
- **Live** : timeline auto-rafraîchie via `@st.fragment(run_every=…)` (les
  lecteurs ont un TTL court). Sobre par défaut (10–60 s).

## Sources consommées

`summary_all_gta.csv`, `online_status.csv`, `online_report.json`, `metrics.json`,
`regime_summary.csv`, `diagnostic_mp.json`, figures `*.png`, et `bundle.pkl`
(complément : seuils, variables, `t`).
