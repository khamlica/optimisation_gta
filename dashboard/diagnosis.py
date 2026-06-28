"""Diagnostic synthétique opérateur — couche de décision au-dessus du triptyque.

Produit, par GTA et « as-of » récent, **un label déterministe**, une
**confiance 0–1** (solidité des indices, pas une probabilité de panne), un
**canal dominant**, un **badge de gouvernance** séparé (régime tardif non
approuvé), et des textes métier courts (raisons + action).

Principes (cf. note de synthèse) :
- l'énergétique est la lecture primaire des gains/pertes de performance ;
- le traceur dynamique ne pèse que s'il existe **indépendamment du rendement** ;
- le Bi2DPCA sert de **confirmation / précocité / écart de baseline** ;
- un régime ``observed_unapproved`` est un **modificateur de confiance et de
  wording**, jamais un verdict de panne par défaut : il n'écrase pas un signal
  énergétique robuste ni une pré-alerte sur régime approuvé.

Aucune nouvelle logique d'alerte officielle : on consomme les artefacts du
pipeline (lecture seule), avec repli sur ``inconclusive`` si une entrée manque.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dashboard import readers

SCORED = ("normal", "warning", "alert")


@dataclass
class DiagnosisConfig:
    recent_window_days: int = 14
    energy_warn_pct: float = 3.0
    energy_alarm_pct: float = 5.0
    energy_step_pct: float = 4.0
    energy_persist_frac: float = 0.30
    energy_run_days: int = 2
    dynamic_exc_min_frac: float = 0.15
    dynamic_min_feature_families: int = 2
    dynamic_run_days: int = 1
    bi_abn_frac_thr: float = 0.30
    bi_alert_frac_thr: float = 0.15
    bi_persist_days_thr: int = 3
    min_recent_points: int = 96 * 3   # ~3 jours utiles
    min_quality_recent: float = 0.50
    unapproved_recent_share: float = 0.50


@dataclass
class DiagnosticResult:
    gta: str
    label: str
    confidence: float
    dominant_channel: str
    reason: list[str]
    recommended_action: str
    regime_badge: str | None
    evidence: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Wording, couleurs, canal
# --------------------------------------------------------------------------- #
LABEL_TITLE = {
    "normal": "Normal",
    "performance_loss": "Perte de performance",
    "performance_gain": "Gain de performance",
    "baseline_shift": "Déplacement de baseline",
    "dynamic_instability_suspected": "Instabilité dynamique suspectée",
    "mixed_event": "Évènement mixte",
    "observed_unapproved_regime": "Mode observé non approuvé",
    "inconclusive": "Non concluant",
}
LABEL_COLOR = {
    "normal": "#2ca02c",
    "performance_loss": "#d62728",
    "dynamic_instability_suspected": "#d62728",
    "mixed_event": "#8c1a1a",
    "performance_gain": "#1f77b4",
    "baseline_shift": "#1f77b4",
    "observed_unapproved_regime": "#ff7f0e",
    "inconclusive": "#7f7f7f",
}
LABEL_CHANNEL = {
    "performance_loss": "Énergétique",
    "performance_gain": "Énergétique",
    "baseline_shift": "Énergétique / Bi2DPCA",
    "dynamic_instability_suspected": "Dynamique",
    "mixed_event": "Énergétique + Dynamique",
    "observed_unapproved_regime": "Gouvernance",
    "normal": "—",
    "inconclusive": "Données",
}
_REASON = {
    "normal": "Les trois canaux restent dans leur zone de référence récente.",
    "performance_loss": "Résidu énergétique négatif et persistant, sans signature dynamique locale marquée.",
    "performance_gain": "Résidu énergétique positif et persistant, sans instabilité locale.",
    "baseline_shift": "Changement durable de la baseline (nouveau normal / maintenance / recalage possible).",
    "dynamic_instability_suspected": "Excursions dynamiques persistantes après retrait du niveau, sans dérive énergétique forte.",
    "mixed_event": "Dérive énergétique ET excursions dynamiques coexistent.",
    "observed_unapproved_regime": "Mode récemment observé actif, baseline non validée, autres canaux faibles.",
    "inconclusive": "Signaux trop faibles/contradictoires ou données récentes insuffisantes.",
}
_ACTION = {
    "normal": "Poursuivre la surveillance normale.",
    "performance_loss": "Vérifier rendement turbine, pertes vapeur et instrumentation EE. Prioriser si Bi2DPCA aussi anormal.",
    "performance_gain": "Vérifier si le gain est physique et durable. Surveiller avant un éventuel rebaselining.",
    "baseline_shift": "Vérifier maintenance, étalonnage et contexte d'exploitation. Envisager une nouvelle baseline si confirmé.",
    "dynamic_instability_suspected": "Examiner régulation, oscillations, transitoires et capteurs rapides.",
    "mixed_event": "Investigation prioritaire : performance + dynamique.",
    "observed_unapproved_regime": "Ne pas conclure à une panne sur ce seul motif. Surveiller, rechercher un évènement d'exploitation.",
    "inconclusive": "Revue experte et contrôle des données.",
}


# --------------------------------------------------------------------------- #
# Agrégats
# --------------------------------------------------------------------------- #
def _longest_day_run(daily_flag: pd.Series) -> int:
    best = run = 0
    for v in daily_flag.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def _feature_family(col: str) -> str:
    return col.split("_", 1)[0]  # vol / ac1 / coup


def _aggregate(gta: str, cfg: DiagnosisConfig, as_of: pd.Timestamp | None) -> dict:
    st = readers.online_series(gta)
    er = readers.energetic_residual(gta)
    feats = readers.dynamic_features(gta)
    dmeta = readers.dynamic_meta(gta)
    rs = readers.regime_metrics(gta)
    mt = readers.metrics(gta)

    F: dict = {"available": not st.empty}
    if st.empty:
        return F
    as_of = as_of or st.index.max()
    W = pd.Timedelta(days=cfg.recent_window_days)
    lo, lo2 = as_of - W, as_of - 2 * W

    # --- Énergie ---
    e_med = e_prev = e_step = e_persist = 0.0
    e_run = 0
    if not er.empty and "resid_pct" in er.columns:
        e = er["resid_pct"].dropna()
        e_rec = e[(e.index > lo) & (e.index <= as_of)]
        e_pre = e[(e.index > lo2) & (e.index <= lo)]
        if len(e_rec):
            e_med = float(e_rec.median())
            sign = np.sign(e_med) if e_med != 0 else 1
            e_persist = float(((e_rec.abs() >= cfg.energy_warn_pct) & (np.sign(e_rec) == sign)).mean())
            day_exc = (e_rec.abs() >= cfg.energy_warn_pct) & (np.sign(e_rec) == sign)
            e_run = _longest_day_run(day_exc.groupby(day_exc.index.normalize()).mean() >= 0.5)
        if len(e_pre):
            e_prev = float(e_pre.median())
        e_step = abs(e_med - e_prev) if (len(e_rec) and len(e_pre)) else 0.0

    # --- Dynamique (familles hors zone de référence) ---
    d_exc_features = d_run = 0
    d_exc_frac = 0.0
    band = (dmeta or {}).get("band", {})
    if not feats.empty and band:
        fr = feats[(feats.index > lo) & (feats.index <= as_of)]
        if len(fr):
            out_any = pd.Series(False, index=fr.index)
            fam_out: dict[str, pd.Series] = {}
            for c in fr.columns:
                b = band.get(c)
                if not b or not (np.isfinite(b["lo"]) and np.isfinite(b["hi"])):
                    continue
                o = (fr[c] < b["lo"]) | (fr[c] > b["hi"])
                out_any |= o.fillna(False)
                fam = _feature_family(c)
                fam_out[fam] = fam_out.get(fam, pd.Series(False, index=fr.index)) | o.fillna(False)
            d_exc_frac = float(out_any.mean())
            d_exc_features = sum(
                1 for s in fam_out.values() if float(s.mean()) >= cfg.dynamic_exc_min_frac
            )
            d_run = _longest_day_run(out_any.groupby(out_any.index.normalize()).mean() >= cfg.dynamic_exc_min_frac)

    # --- Bi2DPCA (statuts online récents) ---
    sc = st[(st.index > lo) & (st.index <= as_of)]
    scored = sc[sc["status"].isin(SCORED)]
    n_recent_valid = int(len(scored))
    quality_recent = float(sc["status"].isin(SCORED).mean()) if len(sc) else 0.0
    b_abn = float(scored["status"].isin(("warning", "alert")).mean()) if len(scored) else 0.0
    b_alert = float((scored["status"] == "alert").mean()) if len(scored) else 0.0
    cur_reg = int(scored["regime"].mode().iloc[0]) if len(scored) else -1
    cur_status = "unknown"
    if not rs.empty and (rs["regime"] == cur_reg).any():
        cur_status = str(rs.loc[rs["regime"] == cur_reg, "status"].iloc[0])
    # jours d'anomalie consécutifs (pour rule 8)
    abn_day = sc["status"].isin(("warning", "alert"))
    bi_persist_days = _longest_day_run(abn_day.groupby(sc.index.normalize()).mean() >= 0.30) if len(sc) else 0

    # --- Gouvernance / pré-signal ---
    u_active = cur_status == "observed_unapproved"
    unapproved = set((mt or {}).get("regimes_unapproved", []))
    u_recent_share = float(scored["regime"].isin(unapproved).mean()) if (len(scored) and unapproved) else 0.0
    # pré-signal AVANT la fenêtre récente : énergie déjà hors warn, ou Bi2 anormal sur régimes approuvés
    ew_energy = abs(e_prev) >= cfg.energy_warn_pct
    sc_pre = st[(st.index > lo2) & (st.index <= lo)]
    sc_pre_appr = sc_pre[~sc_pre["regime"].isin(unapproved)] if unapproved else sc_pre
    ew_bi = float(sc_pre_appr["status"].isin(("warning", "alert")).mean()) >= 0.10 if len(sc_pre_appr) else False
    early_warning = bool(ew_energy or ew_bi)

    pct_expl = float((mt or {}).get("pct_exploitable", 100.0)) / 100.0

    F.update(dict(
        as_of=as_of, e_med=e_med, e_prev=e_prev, e_step=e_step,
        e_persist_frac=e_persist, e_run_days=e_run,
        d_exc_features=d_exc_features, d_exc_frac=d_exc_frac, d_run_days=d_run,
        b_abn_frac=b_abn, b_alert_frac=b_alert, bi_persist_days=bi_persist_days,
        current_regime=cur_reg, current_regime_status=cur_status,
        U_active=u_active, U_recent_share=u_recent_share, early_warning=early_warning,
        quality_recent=quality_recent, pct_exploitable=pct_expl,
        n_recent_valid=n_recent_valid,
    ))
    return F


# --------------------------------------------------------------------------- #
# Décision + confiance
# --------------------------------------------------------------------------- #
def decide_label(F: dict, cfg: DiagnosisConfig) -> str:
    if not F.get("available"):
        return "inconclusive"
    if F["n_recent_valid"] < cfg.min_recent_points or F["quality_recent"] < cfg.min_quality_recent:
        return "inconclusive"

    e = F["e_med"]
    persist_ok = F["e_persist_frac"] >= cfg.energy_persist_frac or F["e_run_days"] >= cfg.energy_run_days
    energy_strong_neg = e <= -cfg.energy_alarm_pct and persist_ok
    energy_strong_pos = e >= cfg.energy_alarm_pct and persist_ok
    energy_step = F["e_step"] >= cfg.energy_step_pct
    dynamic_strong = (
        F["d_exc_features"] >= cfg.dynamic_min_feature_families
        and F["d_run_days"] >= cfg.dynamic_run_days
    )
    bi_strong = F["b_abn_frac"] >= cfg.bi_abn_frac_thr or F["b_alert_frac"] >= cfg.bi_alert_frac_thr
    weak_other = abs(e) < cfg.energy_warn_pct and not dynamic_strong

    if F["U_active"] and weak_other and not F["early_warning"]:
        return "observed_unapproved_regime"
    if energy_strong_neg and not dynamic_strong:
        return "performance_loss"
    if energy_strong_pos and energy_step and not dynamic_strong:
        return "baseline_shift"
    if energy_strong_pos and not dynamic_strong:
        return "performance_gain"
    if dynamic_strong and abs(e) < cfg.energy_warn_pct:
        return "dynamic_instability_suspected"
    if dynamic_strong and (energy_strong_neg or energy_strong_pos):
        return "mixed_event"
    if bi_strong and abs(e) < cfg.energy_warn_pct and not dynamic_strong:
        return "baseline_shift" if F["bi_persist_days"] >= cfg.bi_persist_days_thr else "inconclusive"
    if abs(e) < cfg.energy_warn_pct and not dynamic_strong and not bi_strong:
        return "normal"
    return "inconclusive"


def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def _confidence(label: str, F: dict, cfg: DiagnosisConfig) -> float:
    e = F["e_med"]
    S_e = _clip((abs(e) - cfg.energy_warn_pct) / (cfg.energy_alarm_pct - cfg.energy_warn_pct))
    P_e = _clip(max(F["e_run_days"] / cfg.energy_run_days, F["e_persist_frac"] / cfg.energy_persist_frac))
    S_d = 0.5 * _clip((F["d_exc_features"] - 1) / 2) + 0.5 * _clip(F["d_exc_frac"] / 0.20)
    S_b = max(_clip(F["b_alert_frac"] / cfg.bi_alert_frac_thr), _clip(F["b_abn_frac"] / cfg.bi_abn_frac_thr))
    Q = (
        min(1.0, np.sqrt(F["n_recent_valid"] / (96 * 7)))
        * _clip(F["quality_recent"]) * _clip(F["pct_exploitable"], 0.60, 1.0)
    )
    step_evidence = _clip(F["e_step"] / cfg.energy_step_pct)
    tardy = 0.75 if (F["U_active"] and not F["early_warning"]) else 1.0

    if label in ("performance_loss", "performance_gain"):
        C = Q * tardy * (0.40 * S_e + 0.25 * P_e + 0.20 * max(S_b, 0.40) + 0.15 * (1 - S_d))
    elif label == "baseline_shift":
        C = Q * (0.30 * S_e + 0.20 * P_e + 0.25 * S_b + 0.25 * step_evidence) * (1 - 0.20 * S_d)
    elif label == "dynamic_instability_suspected":
        C = Q * (0.40 * S_d + 0.25 * _clip(F["d_run_days"] / 2) + 0.20 * (1 - S_e) + 0.15 * max(S_b, 0.30))
    elif label == "mixed_event":
        C = Q * (0.30 * S_e + 0.20 * P_e + 0.25 * S_d + 0.15 * _clip(F["d_run_days"] / 2) + 0.10 * S_b)
    elif label == "observed_unapproved_regime":
        C = min(0.75, Q * (0.35 * float(F["U_active"]) + 0.25 * (1 - S_e) + 0.20 * (1 - S_d) + 0.20 * (1 - float(F["early_warning"]))))
    elif label == "normal":
        C = Q * (0.45 * (1 - S_e) + 0.30 * (1 - S_d) + 0.25 * (1 - S_b))
    else:  # inconclusive
        C = 0.30 * Q
    return round(_clip(C), 2)


def _reason_lines(label: str, F: dict) -> list[str]:
    lines = [_REASON.get(label, "")]
    if F.get("available"):
        lines.append(f"Résidu énergétique médian : {F['e_med']:+.1f} % EE (persistance {100*F['e_persist_frac']:.0f}%, {F['e_run_days']} j).")
        lines.append(f"Dynamique locale : {'excursions' if F['d_exc_features']>=1 else 'stable'} ({F['d_exc_features']} familles hors zone).")
        lines.append(f"Bi2DPCA récent : {'anormal' if (F['b_abn_frac']>=0.30 or F['b_alert_frac']>=0.15) else 'calme'} ({100*F['b_abn_frac']:.0f}% warn/alert).")
    return [x for x in lines if x]


def compute_diagnostic(
    gta: str, as_of: pd.Timestamp | None = None, cfg: DiagnosisConfig = DiagnosisConfig()
) -> DiagnosticResult:
    F = _aggregate(gta, cfg, as_of)
    label = decide_label(F, cfg)
    conf = _confidence(label, F, cfg)
    badge = None
    if F.get("U_active") and label != "observed_unapproved_regime":
        badge = f"baseline non approuvée (R{F.get('current_regime')})"
    return DiagnosticResult(
        gta=gta, label=label, confidence=conf,
        dominant_channel=LABEL_CHANNEL.get(label, "—"),
        reason=_reason_lines(label, F),
        recommended_action=_ACTION.get(label, ""),
        regime_badge=badge,
        evidence={k: float(v) for k, v in F.items() if isinstance(v, (int, float))},
    )
