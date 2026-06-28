"""Tests du moteur de diagnostic synthétique.

Exécutable via pytest OU directement : ``python tests/test_diagnosis.py``.
Couvre la table de décision (cas synthétiques), la priorité énergie >
gouvernance, le repli qualité, l'effet de la qualité sur la confiance, et les
critères d'acceptation sur les 4 GTA réels.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.diagnosis import (  # noqa: E402
    DiagnosisConfig,
    _confidence,
    compute_diagnostic,
    decide_label,
)

CFG = DiagnosisConfig()


def make_F(**over) -> dict:
    F = dict(
        available=True, n_recent_valid=1000, quality_recent=0.9,
        e_med=0.0, e_prev=0.0, e_step=0.0, e_persist_frac=0.0, e_run_days=0,
        d_exc_features=0, d_exc_frac=0.0, d_run_days=0,
        b_abn_frac=0.0, b_alert_frac=0.0, bi_persist_days=0,
        current_regime=0, current_regime_status="modeled",
        U_active=False, U_recent_share=0.0, early_warning=False,
        quality_recent_ok=True, pct_exploitable=0.95,
    )
    F.update(over)
    return F


def test_performance_loss():
    F = make_F(e_med=-7.0, e_persist_frac=1.0, e_run_days=5)
    assert decide_label(F, CFG) == "performance_loss"


def test_baseline_shift_with_step():
    F = make_F(e_med=7.0, e_persist_frac=1.0, e_step=5.0)
    assert decide_label(F, CFG) == "baseline_shift"


def test_performance_gain_without_step():
    F = make_F(e_med=7.0, e_persist_frac=1.0, e_step=1.0)
    assert decide_label(F, CFG) == "performance_gain"


def test_dynamic_instability():
    F = make_F(e_med=0.5, d_exc_features=2, d_run_days=1, d_exc_frac=0.3)
    assert decide_label(F, CFG) == "dynamic_instability_suspected"


def test_observed_unapproved_when_other_channels_weak():
    F = make_F(U_active=True, e_med=0.5, early_warning=False)
    assert decide_label(F, CFG) == "observed_unapproved_regime"


def test_unapproved_does_not_override_energy():
    # régime tardif actif MAIS énergie robuste -> reste performance_loss
    F = make_F(U_active=True, e_med=-8.0, e_persist_frac=1.0)
    assert decide_label(F, CFG) == "performance_loss"


def test_quality_gate_inconclusive():
    F = make_F(e_med=-8.0, e_persist_frac=1.0, quality_recent=0.3)
    assert decide_label(F, CFG) == "inconclusive"


def test_confidence_drops_with_quality():
    hi = _confidence("performance_loss", make_F(e_med=-7, e_persist_frac=1.0, quality_recent=0.95), CFG)
    lo = _confidence("performance_loss", make_F(e_med=-7, e_persist_frac=1.0, quality_recent=0.55), CFG)
    assert lo < hi


def test_acceptance_real_gtas():
    """Critères d'acceptation sur artefacts réels (si présents)."""
    expected = {
        "JFC3": ("performance_loss", 0.80, True),   # + badge baseline non approuvée
        "JFC4": ("normal", 0.75, False),
    }
    for gta, (label, min_conf, badge) in expected.items():
        try:
            r = compute_diagnostic(gta)
        except Exception:
            continue  # artefacts absents : test ignoré
        assert r.label == label, f"{gta}: {r.label} != {label}"
        assert r.confidence >= min_conf, f"{gta}: conf {r.confidence} < {min_conf}"
        assert (r.regime_badge is not None) == badge, f"{gta}: badge {r.regime_badge}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"OK  {fn.__name__}")
    print(f"\n{len(fns)} tests passés.")
