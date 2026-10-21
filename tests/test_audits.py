import numpy as np
import pytest
from dataclasses import asdict

from ai_planner.audits import accepted_flow_summary, _route_from_definition


def test_negative_monte_carlo_balance_is_preserved_not_clamped():
    summary = accepted_flow_summary(np.array([2.0, 2.0]), np.array([1.0, 1.0]),
                                    beta=0.25, j_max=3.08)
    assert summary["L_bal_signed"] == -1.0
    assert summary["L_bal_standard_error"] == 0.0
    assert summary["sqrt_abar_L_bal_over_2_point_estimate"] is None
    assert summary["balance_bound_status"].startswith("undefined")
    assert summary["L_ratio"] == 1.0
    assert summary["epsilon_a"] == 0.0


def test_refresh_mc_bound_uses_log_minorization_and_marks_vacuity():
    summary = accepted_flow_summary(np.array([-10.0, -10.0]), np.array([0.0, 0.0]),
                                    beta=0.25, j_max=3.08)
    assert summary["L_bal_signed"] == 10.0
    for row in summary["refresh"]:
        assert row["log_kappa"] == pytest.approx(np.log(row["rho"]) - 3.08 / 0.25)
        assert row["stationary_bias_bound_point_estimate"] == pytest.approx(
            (1 - row["rho"]) * summary["epsilon_a"] / np.exp(row["log_kappa"]))
        assert row["vacuous"]


def test_route_restore_uses_saved_snapshot_without_reexecuting_simulator(monkeypatch):
    import ai_planner.routes as routes
    import json
    target = routes.make_route_targets()["test"][2]
    saved = json.loads(json.dumps(asdict(target)))
    monkeypatch.setattr(routes, "simulate", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Restoring audit context must not simulate history")))
    restored = _route_from_definition(saved)
    assert restored.target_id == target.target_id
    assert restored.context == target.context
