"""Bounded experiment orchestration; all saved visual data are actual evaluations."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import platform
import time
import importlib.metadata
import numpy as np
import jax
import jax.numpy as jnp
from .contexts import canonical
from .targets import analytic_splits, quadrature
from .decoders import compact, IdentityDecoder
from .proposals import PCNProposal, RWProposal
from .samplers import run_chains, rejection_sample, COUNTER_NAMES
from .training import TrainingConfig, train, load_checkpoint, calibrate
from .diagnostics import effective_sample_size, finite_grid_audit


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if hasattr(x, "shape"):
            return clean(np.asarray(x).tolist())
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        return x
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False))


def region_labels(target, y):
    if hasattr(target.context, "centers"):
        centers = np.asarray(target.context.centers)
        return np.argmin(np.sum((y[...,None,:]-centers)**2,axis=-1),axis=-1)
    return (y[...,0]>0).astype(int)


def sampling_summary(target, trace, grid, burn):
    yy = np.asarray(compact(trace["z"][:,burn:]))
    labels = region_labels(target, yy)
    counts = np.bincount(labels.ravel(), minlength=len(grid["region_masses"]))
    masses = counts/counts.sum()
    n = yy.shape[0]*yy.shape[1]
    bins = 8
    samplehist = np.histogram2d(*yy.reshape(-1,2).T,bins=bins,range=[[-1,1],[-1,1]])[0]/n
    reference = np.histogram2d(*grid["y"].reshape(-1,2).T,bins=bins,range=[[-1,1],[-1,1]],
                              weights=grid["masses"].ravel())[0]
    perchain = np.stack([np.bincount(r,minlength=len(masses))/len(r) for r in labels])
    # Conservative ESS for a region indicator, with stuck chains reporting ESS=0.
    ess = sum(effective_sample_size((r==0).astype(float)) for r in labels)
    movement_start=max(1,burn)
    candidate=trace["candidate"]
    # Reuse stored decoded objects: these feature computations make no oracle calls.
    feature=jax.jit(jax.vmap(target.features))
    now=np.asarray(feature(jnp.asarray(candidate[:,movement_start:].reshape((-1,)+candidate.shape[2:]))))
    old=np.asarray(feature(jnp.asarray(candidate[:,movement_start-1:-1].reshape((-1,)+candidate.shape[2:]))))
    movement=1-np.exp(-np.sum((now-old)**2,axis=-1)/.5)
    return dict(n_samples=n,burn_in_per_chain=burn,n_chains=len(yy),
                region_masses=masses,reference_region_masses=grid["region_masses"],
                region_mass_standard_error=perchain.std(axis=0,ddof=1)/np.sqrt(len(yy)) if len(yy)>1 else None,
                region_tv=float(.5*np.abs(masses-grid["region_masses"]).sum()),
                coarse_tv=float(.5*np.abs(samplehist-reference).sum()),coarse_histogram_bins=bins,
                acceptance_rate=float(trace["accepted"][:,burn:].mean()),ess=float(ess),
                coordinate_ess=[effective_sample_size(yy[...,i]) for i in range(yy.shape[-1])],
                mode_switches=int(np.sum(np.diff(labels,axis=1)!=0)),
                accepted_movement=float(movement.mean()),
                accepted_movement_space="target.features(decoded candidate), scale 0.5",
                provenance="finite frozen-kernel chains; repeated rejected states retained",
                uncertainty="across independent chain starts; does not eliminate finite-chain bias")


def _audit_replay(target, proposal, ratio, trace, limit=64):
    z=trace["before_z"][0,:limit]; zp=trace["proposed_z"][0,:limit]
    costfn=jax.jit(jax.vmap(target.evaluate))
    c=costfn(jnp.asarray(z)); cp=costfn(jnp.asarray(zp))
    from .acceptance import exact_log_ratio
    ell=np.asarray(exact_log_ratio(z,zp,c,cp,target.beta,proposal,target.context_features))
    refresh=trace["branches"][0,:limit]==2
    ell=np.where(refresh,-np.asarray(cp-c)/target.beta,ell)
    trace["audit_exact_log_ratio"]=np.full(trace["accepted"].shape,np.nan)
    trace["audit_exact_log_ratio"][0,:limit]=ell
    trace["audit_current_cost"]=np.full(trace["accepted"].shape,np.nan)
    trace["audit_current_cost"][0,:limit]=np.asarray(c)
    trace["audit_proposed_cost"]=np.full(trace["accepted"].shape,np.nan)
    trace["audit_proposed_cost"][0,:limit]=np.asarray(cp)
    if ratio is not None:
        trace["learned_log_ratio"][0,:limit]=np.asarray(ratio(jnp.asarray(z),jnp.asarray(zp),target.context_features))
    return {"audit_evaluations":2*len(z),"decoder_calls":2*len(z),
            "exact_cost_evaluations":2*len(z),"simulator_steps":2*len(z)*target.simulator_steps_per_cost}


def _route_frames(target, trace, path, limit=48):
    from .routes import execute_selected
    z=jnp.asarray(trace["proposed_z"][0,:limit])
    def f(zz):
        command=target.decoder.decode(target.context,zz)
        rollout=target.rollout(command)
        metrics=target.metrics(command,rollout)
        return command,rollout.positions,metrics
    commands,positions,metrics=jax.jit(jax.vmap(f))(z)
    context=target.context
    np.savez_compressed(path,z=np.asarray(z),commands=np.asarray(commands),positions=np.asarray(positions),
                        cost=np.asarray(metrics.cost),success=np.asarray(metrics.success),
                        accepted=trace["accepted"][0,:limit],
                        old_plan=context.selected_old_plan,retained_suffix=context.retained_suffix,
                        completed_tail=context.completed_tail,cached_plan=context.cached_plan,
                        executed_prefix=np.asarray(context.selected_old_plan)[:context.executed_prefix],
                        executed_positions=context.initial_positions,
                        obstacle_centers=np.asarray(context.obstacles)[:,:2],
                        obstacle_radii=np.asarray(context.obstacles)[:,2]*np.asarray(context.obstacles)[:,3],
                        start=context.snapshot.position,goal=context.goal,context_id=target.target_id)
    return dict(audit_evaluations=len(z),exact_cost_evaluations=len(z),decoder_calls=len(z),
                simulator_steps=len(z)*target.simulator_steps_per_cost)


def _sum_counters(rows):
    return {k:sum(int(r.get(k,0)) for r in rows) for k in COUNTER_NAMES}


def run_experiment(kind, config, output, checkpoint=None, resume=None, exact_only=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for d in ("traces","grids","routes","planning","checkpoints"):
        (output/d).mkdir(exist_ok=True)
    write_json(output/"config.json",config)
    started=time.perf_counter()
    seed=int(config.get("seed",17))
    support_counts=[]
    if kind=="analytic":
        training,validation,tests=analytic_splits(config.get("beta",1.),config.get("one_context",False))
        context_creation_steps=0
    else:
        from .routes import make_route_targets,support_grid
        split=make_route_targets(config.get("beta",.25))
        training,validation,tests=split["train"],split["validation"],split["test"]
        context_creation_steps=sum(t.context.executed_prefix*t.simulator.substeps for group in split.values() for t in group)
        for t in tests:
            support=support_grid(t,11)
            np.savez_compressed(output/"routes"/f"support__{t.name}.npz",**support)
            support_counts.append(dict(audit_evaluations=121,exact_cost_evaluations=121,
                                       decoder_calls=121,simulator_steps=support["simulator_steps"]))
    if config.get("test_contexts"):
        tests=[t for t in tests if t.name in config["test_contexts"]]
    metadata=dict(kind=kind,seed=seed,python=platform.python_version(),platform=platform.platform(),
                  packages={k:importlib.metadata.version(k) for k in ("jax","jaxlib","flax","optax","numpy","scipy")},
                  devices=[str(d) for d in jax.devices()],
                  contexts=[dict(name=t.name,beta=t.beta,target_id=t.target_id,j_max=t.j_max,
                                 decoder=t.decoder.version,split="test",definition=canonical(t)) for t in tests],
                  training_context_ids=[t.target_id for t in training],validation_context_ids=[t.target_id for t in validation],
                  warning="One development seed. Engineered frozen decoders; no DSBM or real robotics claim.")
    write_json(output/"metadata.json",metadata)
    result=None; ablation=None
    if not exact_only:
        if checkpoint:
            result=load_checkpoint(checkpoint)
            if result.metadata["temperature"] != tests[0].beta:
                raise ValueError("Checkpoint temperature mismatch; changed-temperature runs must use exact acceptance")
            if result.metadata["target_ids"] != [t.target_id for t in training]:
                raise ValueError("Checkpoint decoder/target family mismatch")
        else:
            cfg=dict(config.get("training",{})); cfg["seed"]=seed
            if "hidden_dims" in cfg: cfg["hidden_dims"]=tuple(cfg["hidden_dims"])
            result=train(training,TrainingConfig(**cfg),output/"checkpoints",validation,resume=resume)
        if config.get("no_movement_ablation",True):
            cfg=dict(config.get("training",{})); cfg.update(seed=seed,proposal_steps=0)
            if "hidden_dims" in cfg: cfg["hidden_dims"]=tuple(cfg["hidden_dims"])
            ablation=train(training,TrainingConfig(**cfg),output/"ablation_no_movement",validation)
        write_json(output/"training_history.json",result.history)
        calibration=result.calibration
        if not calibration:
            calibration_counts=dict.fromkeys(COUNTER_NAMES,0)
            calibration,_=calibrate(validation,result.proposal,result.ratio,jax.random.PRNGKey(seed+556),
                                    n=128,counters=calibration_counts)
            result.metadata["calibration_counters"]=calibration_counts
        np.savez_compressed(output/"calibration.npz",
                            exact_log_ratio=np.concatenate([v["exact"] for v in calibration.values()]),
                            learned_log_ratio=np.concatenate([v["predicted"] for v in calibration.values()]))
    methods=[
        ("prior",PCNProposal(1.),None,"prior",0.),
        ("rw",RWProposal(.35),None,"exact",0.),
        ("pcn",PCNProposal(.65),None,"exact",0.),
        ("high_locality",RWProposal(.012),None,"exact",0.),
    ]
    if result:
        methods.extend([("learned_exact",result.proposal,result.ratio,"exact",0.),
                        ("learned",result.proposal,result.ratio,"learned",0.)])
        for rho in config.get("refresh_rates",[.1,.3]):
            methods.append((f"refresh_{rho:g}",result.proposal,result.ratio,"learned",rho))
        injected=lambda z,zp,c:result.ratio(z,zp,c)+1.5*(zp[...,0]-z[...,0])
        methods.append(("injected_error",result.proposal,injected,"learned",0.))
        if ablation:
            methods.append(("no_movement",ablation.proposal,ablation.ratio,"learned",0.))
    metrics=[]; audit_counts=list(support_counts); ref_checks=[]; planning_rows=[]; sampling_counts=[]
    grids={}
    ngrid=int(config.get("grid_size",64)); burn=int(config.get("burn_in",256))
    steps=int(config.get("steps",1280)); nchains=int(config.get("chains",4))
    for ti,target in enumerate(tests):
        print(f"[{kind}] reference {target.name}",flush=True)
        grid=quadrature(target,ngrid); fine=quadrature(target,ngrid*2)
        grid["audit_evaluations"]=grid["exact_cost_evaluations"]
        fine["audit_evaluations"]=fine["exact_cost_evaluations"]
        grids[target.name]=grid
        np.savez_compressed(output/"grids"/f"{target.name}.npz",**grid)
        np.savez_compressed(output/"grids"/f"{target.name}__fine.npz",**fine)
        audit_counts.extend([grid,fine])
        nref=int(config.get("reference_samples",1024))
        ref,refcounts,_=rejection_sample(target,nref,jax.random.PRNGKey(seed+100+ti))
        refcounts["audit_evaluations"]=refcounts["exact_cost_evaluations"]; audit_counts.append(refcounts)
        yy=np.asarray(compact(ref)); labels=region_labels(target,yy)
        observed=np.bincount(labels,minlength=len(grid["region_masses"]))/nref
        se=np.sqrt(fine["region_masses"]*(1-fine["region_masses"])/nref)
        check=dict(context=target.name,n=nref,normalizer=grid["normalizer"],
                   refined_normalizer=fine["normalizer"],
                   region_mass_grid_difference=float(np.max(np.abs(grid["region_masses"]-fine["region_masses"]))),
                   reference_region_masses=fine["region_masses"],rejection_region_masses=observed,
                   rejection_standard_error=se,
                   rejection_max_standardized_error=float(np.max(np.abs(observed-fine["region_masses"])/np.maximum(se,1e-12))),
                   counters=refcounts,provenance="independent exact rejection; Gaussian prior draws")
        ref_checks.append(check)
        np.savez_compressed(output/"traces"/f"{target.name}__rejection_reference.npz",z=ref[None],y=yy[None])
        for mi,(name,proposal,ratio,mode,rho) in enumerate(methods):
            print(f"[{kind}] {target.name}: {name}",flush=True)
            key=jax.random.PRNGKey(seed+1000+ti*100+mi)
            trace,counts=run_chains(target,proposal,key,steps,nchains,ratio,mode,rho)
            sampling_counts.append(counts)
            summary=sampling_summary(target,trace,grid,burn)
            summary.update(context=target.name,method=name,beta=target.beta,rho=rho,**counts)
            replay_counts=_audit_replay(target,proposal,ratio,trace)
            audit_counts.append(replay_counts)
            np.savez_compressed(output/"traces"/f"{target.name}__{name}.npz",**trace)
            metrics.append(summary)
            if kind=="routes" and name in ("learned_exact","refresh_0.1","pcn"):
                audit_counts.append(_route_frames(target,trace,output/"routes"/f"{target.name}__{name}.npz"))
        write_json(output/"results.json",{"metrics":metrics})
    print(f"[{kind}] finite grid audit",flush=True)
    audit=finite_grid_audit(tests[0],result.proposal if result else PCNProposal(.65),
                            result.ratio if result else None,grid_size=11)
    audit_counts.append(dict(audit_evaluations=121,exact_cost_evaluations=121,decoder_calls=121,
                             simulator_steps=121*tests[0].simulator_steps_per_cost))
    write_json(output/"finite_grid.json",{tests[0].name:audit})
    write_json(output/"reference_checks.json",ref_checks)
    if kind=="analytic" and config.get("mechanisms",True):
        from .mechanisms import run_mechanisms
        mechanism=run_mechanisms(output,tests,seed=seed)
        metrics.extend(mechanism["metrics"])
        metadata["contexts"].extend(mechanism["contexts"])
        sampling_counts.append(mechanism["counters"]["sampling"])
        audit_counts.append(mechanism["counters"]["audit"])
        write_json(output/"metadata.json",metadata)
    if config.get("planning",True):
        from .planning import plan
        for ti,target in enumerate(tests):
            # Constant-cost target has no meaningful best-plan comparison.
            if target.name=="constant": continue
            for budget in config.get("planning_budgets",[16,48]):
                for replicate in range(int(config.get("planning_replicates",2))):
                    for name,proposal,ratio,mode,rho in methods:
                        if name in ("high_locality","injected_error","no_movement"): continue
                        p=plan(target,proposal,ratio,jax.random.PRNGKey(seed+90000+ti*1000+budget*10+replicate),
                               budget,mode=mode,rho=rho)
                        row={k:v for k,v in p.items() if k not in ("candidate","z","key","attempt_log") and not k.startswith("pool")}
                        row.update(p["metrics"]); row.update(p["counters"])
                        if "success" in p["metrics"]: row["planner_success"]=int(p["metrics"]["success"])
                        row.update(context=target.name,method=name,budget=budget,replicate=replicate)
                        planning_rows.append(row)
                        if kind=="routes" and replicate==0 and budget==max(config.get("planning_budgets",[16,48])):
                            _save_execution(target,p,output/"planning"/f"{target.name}__{name}.npz")
                    if kind=="routes":
                        p=plan(target,PCNProposal(1.),None,jax.random.PRNGKey(seed+91000+ti*1000+budget*10+replicate),
                               budget,mode="cem")
                        row={k:v for k,v in p.items() if k not in ("candidate","z","key","attempt_log") and not k.startswith("pool")}
                        row.update(p["metrics"]); row.update(p["counters"])
                        row["planner_success"]=int(p["metrics"]["success"])
                        row.update(context=target.name,method="cem",budget=budget,replicate=replicate)
                        planning_rows.append(row)
            print(f"[{kind}] planning {target.name} complete",flush=True)
            write_json(output/"planning_results.json",planning_rows)
    accounting=dict(sampling=_sum_counters(sampling_counts),audit=_sum_counters(audit_counts),
                    training=result.counters if result and not checkpoint else {},
                    source_checkpoint_training=result.counters if result and checkpoint else {},
                    training_calibration=result.metadata.get("calibration_counters",{}) if result else {},
                    ablation_training=ablation.counters if ablation else {},
                    ablation_calibration=ablation.metadata.get("calibration_counters",{}) if ablation else {},
                    planning=_sum_counters([r.get("counters",r) for r in planning_rows]),
                    physical_context_creation_steps=context_creation_steps,
                    physical_execution_steps=len(list((output/"planning").glob("*.npz")))*18 if kind=="routes" else 0,
                    total_elapsed_seconds=time.perf_counter()-started,
                    note="Audit costs excluded from deployment budgets but charged separately. JIT compilation separately timed.")
    write_json(output/"accounting.json",accounting)
    write_json(output/"results.json",dict(metrics=metrics,planning=planning_rows))
    scalar_keys=sorted({k for row in metrics for k,v in row.items() if isinstance(v,(int,float,str))})
    with (output/"metrics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=scalar_keys,extrasaction="ignore"); writer.writeheader(); writer.writerows(metrics)
    from .visualization import render
    render(output)
    return output


def _save_execution(target,plan,path):
    from .routes import execute_selected
    next_target, rollout=execute_selected(target,plan["candidate"])
    np.savez_compressed(path,selected_commands=plan["candidate"],z=plan["z"],
                        executed_positions=rollout.positions,
                        executed_commands=np.asarray(plan["candidate"])[:target.context.executed_prefix],
                        next_cached_plan=next_target.context.cached_plan,
                        retained_suffix=next_target.context.retained_suffix,
                        completed_tail=next_target.context.completed_tail,
                        old_plan=target.context.cached_plan,
                        next_position=next_target.context.snapshot.position,
                        next_velocity=next_target.context.snapshot.velocity,
                        physical_execution_steps=target.context.executed_prefix*target.simulator.substeps)
