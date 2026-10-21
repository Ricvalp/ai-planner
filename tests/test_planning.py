import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ai_planner.planning import plan
from ai_planner.proposals import PCNProposal, RWProposal
from ai_planner.routes import make_route_targets
from ai_planner.targets import make_analytic_target


@pytest.mark.parametrize("mode,rho", [("prior", 0.0), ("exact", 0.0), ("learned", 0.0), ("learned", 0.35)])
def test_planner_respects_budget_and_validates_only_evaluated_incumbents(mode, rho):
    target = make_analytic_target()
    ratio = lambda z, zp, ctx: jnp.array(0.0)
    result = plan(target, PCNProposal(), ratio, jax.random.PRNGKey(9), 13, mode=mode, rho=rho)
    assert result["counters"]["exact_cost_evaluations"] == 13
    assert result["counters"]["final_validation_evaluations"] == 1
    assert len(result["pool_cost"]) == 12
    assert result["cost"] == pytest.approx(result["pool_cost"].min())
    assert result["counters"]["simulator_steps"] == 0
    assert result["attempts"] <= 9 * 13
    np.testing.assert_allclose(result["candidate"],
                               target.decoder.decode(target.context, result["z"]))


@pytest.mark.parametrize("mode", ["prior", "exact", "learned", "cem"])
def test_route_planner_counts_final_validation_and_has_physical_metrics(mode):
    target = make_route_targets()["test"][2]
    ratio = lambda z, zp, ctx: jnp.array(0.0)
    result = plan(target, RWProposal(0.5), ratio, jax.random.PRNGKey(3), 9, mode=mode, rho=0.2 if mode == "learned" else 0)
    assert result["counters"]["exact_cost_evaluations"] == 9
    assert result["counters"]["simulator_steps"] == 9 * 180
    assert result["counters"]["final_validation_evaluations"] == 1
    assert len(result["pool_cost"]) == 8
    assert {"goal_error", "collided", "success"}.issubset(result["metrics"])
    if not result["fallback"]:
        assert not result["metrics"]["collided"]
    assert result["candidate"].shape == (20, 2)
