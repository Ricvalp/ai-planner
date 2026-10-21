from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import norm, multivariate_normal
from ai_planner.targets import make_analytic_target, quadrature
from ai_planner.decoders import AnalyticDecoder, IdentityDecoder, compact, inverse_compact
from ai_planner.samplers import (normal_log_prob, initialize, run_chain, make_step, rebind,
                                save_chain, load_chain, counter_dict)
from ai_planner.proposals import PCNProposal, RWProposal


def test_density_decoder_and_target_scipy():
    target = make_analytic_target("rotated")
    z = jnp.array([-.3, 1.2]); y = compact(z)
    np.testing.assert_allclose(y, 2*norm.cdf(z)-1, atol=1e-14)
    np.testing.assert_allclose(inverse_compact(y), z, atol=1e-14)
    np.testing.assert_allclose(normal_log_prob(z), norm.logpdf(z).sum(), atol=1e-14)
    c = target.context
    w = c.eps_floor + (1-c.eps_floor)*sum(p*np.exp(-.5*(np.asarray(y)-m)@np.linalg.inv(s)@(np.asarray(y)-m))
                                         for p, m, s in zip(c.weights, c.centers, c.covariances))
    np.testing.assert_allclose(target.evaluate(z), -c.beta0*np.log(w), atol=1e-14)
    assert 0 <= float(target.evaluate(z)) <= target.j_max


def test_pcn_general_hastings_and_constant_acceptance():
    p = PCNProposal(.6)
    z, zp = jnp.array([1., -.4]), jnp.array([-.7, .6])
    ctx = jnp.zeros(2)
    np.testing.assert_allclose(p.log_prob(zp,z,ctx), multivariate_normal.logpdf(zp, mean=.8*z, cov=.36*np.eye(2)))
    correction = normal_log_prob(zp)-normal_log_prob(z)+p.log_prob(z,zp,ctx)-p.log_prob(zp,z,ctx)
    np.testing.assert_allclose(correction, 0., atol=2e-14)
    t = make_analytic_target("constant")
    chain, tr, counts = run_chain(t, p, initialize(t,jax.random.PRNGKey(2)),64)
    assert tr["accepted"].all()
    np.testing.assert_allclose(tr["exact_log_ratio"], 0., atol=1e-13)
    assert counts["exact_cost_evaluations"] == 65
    np.testing.assert_allclose(quadrature(t,16)["density"], .25)


def test_learned_only_no_oracle_then_refresh_and_rejection(tmp_path):
    t = make_analytic_target()
    p = PCNProposal(.65)
    c = initialize(t,jax.random.PRNGKey(8), evaluate=True)
    d = lambda z,zp,x: jnp.array(0.)
    c, tr, counts = run_chain(t,p,c,5,d,mode="learned")
    assert counts["exact_cost_evaluations"] == 1
    assert not c.state.valid and np.isnan(c.state.cost)
    c, tr, counts = run_chain(t,p,c,1,d,mode="learned",rho=1.)
    assert counts["refresh_current_evaluations"] == 1
    assert counts["refresh_proposed_evaluations"] == 1
    assert counts["exact_cost_evaluations"] == 3
    reject = lambda z,zp,x: jnp.array(-1e10)
    before = c.state
    c, tr, counts = run_chain(t,p,c,3,reject,mode="learned")
    assert not tr["accepted"].any()
    for field in ("z","candidate","cost","valid"):
        np.testing.assert_array_equal(getattr(c.state,field), getattr(before,field))
    path=tmp_path/"chain.npz"; save_chain(path,c); loaded=load_chain(path)
    a,ta,_=run_chain(t,p,c,8); b,tb,_=run_chain(t,p,loaded,8)
    np.testing.assert_array_equal(ta["z"],tb["z"])
    changed=replace(t,beta=.7)
    assert changed.target_id != t.target_id
    assert not rebind(c,changed).state.valid
    assert replace(t,decoder=AnalyticDecoder(version="changed")).target_id != t.target_id
    assert replace(t,cost_version="changed").target_id != t.target_id
    assert replace(t,context=replace(t.context,eps_floor=.02)).target_id != t.target_id


def test_runtime_oracle_skip_callback():
    calls=[]
    class Instrumented:
        noise_dim=2; context=None; context_features=jnp.zeros(1); beta=1.; target_id="instrumented"
        decoder=IdentityDecoder(); simulator_steps_per_cost=3
        def cost(self,a):
            jax.debug.callback(lambda _:calls.append(1), a[0])
            return jnp.sum(a*a)/(1+jnp.sum(a*a))
    t=Instrumented(); p=PCNProposal(.6)
    c=initialize(t,jax.random.PRNGKey(1))
    c,tr,count=run_chain(t,p,c,10,lambda z,zp,x:jnp.array(0.),mode="learned")
    jax.effects_barrier()
    assert len(calls)==0 and count["exact_cost_evaluations"]==0
    c,tr,count=run_chain(t,p,c,1,lambda z,zp,x:jnp.array(0.),mode="learned",rho=1.)
    jax.effects_barrier()
    assert len(calls)==2 and count["simulator_steps"]==6


def test_dimension_independent_and_ignored_coordinates():
    class Target:
        noise_dim=6; context=None; context_features=jnp.zeros(1); beta=1.; target_id="six"
        decoder=IdentityDecoder(6); simulator_steps_per_cost=0
        def cost(self,a): return jnp.array(0.)
        def features(self,a): return a[:2]
    t=Target()
    c,tr,_=run_chain(t,PCNProposal(.5),initialize(t,jax.random.PRNGKey(3)),8)
    assert tr["z"].shape==(8,6)
    a=jnp.zeros(6); b=a.at[5].set(100.)
    np.testing.assert_array_equal(t.features(a),t.features(b))


def test_tiny_proposals_keep_repeated_states_and_move_little():
    t=make_analytic_target()
    c=initialize(t,jax.random.PRNGKey(13),z=[-.7,0.])
    c,tr,_=run_chain(t,RWProposal(.001),c,256)
    assert tr["accepted"].mean()>.9
    assert np.mean(np.sum(np.diff(tr["z"],axis=0)**2,axis=-1))<1e-4
    assert tr["z"].shape[0]==256
    rejected=~tr["accepted"]
    np.testing.assert_array_equal(tr["z"][rejected],tr["before_z"][rejected])
