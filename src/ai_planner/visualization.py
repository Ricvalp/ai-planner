"""Render saved measurements without decoder, simulator, or training calls.

Matplotlib writes publication-friendly static figures; a standalone HTML
canvas replay embeds the same arrays and never needs a server or CDN.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


COLORS = ["#087f8c", "#e58f34", "#7964b4", "#cb5a5e", "#4878a8", "#5d9160"]


def _read_json(path, default):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _metrics(run):
    data = _read_json(run / "results.json", [])
    if isinstance(data, dict):
        return data.get("metrics", data.get("results", []))
    return data


def _value(row, *names, default=np.nan):
    for name in names:
        if name in row and row[name] is not None:
            try:
                return float(row[name])
            except (TypeError, ValueError):
                pass
        for nested in ("counters", "sampling", "metrics", "planning"):
            if isinstance(row.get(nested), dict) and name in row[nested]:
                try:
                    return float(row[nested][name])
                except (TypeError, ValueError):
                    pass
    return default


def _context(row):
    return str(row.get("context", row.get("context_name", row.get("context_id", "context"))))


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "semibold", "axes.labelcolor": "#334155",
        "text.color": "#172c3d", "axes.edgecolor": "#cbd5e1",
        "xtick.color": "#526477", "ytick.color": "#526477",
        "figure.facecolor": "white", "axes.facecolor": "#fbfcfe",
        "savefig.facecolor": "white", "grid.color": "#e2e8f0",
        "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    })


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return name


def _field_map(ax, grid, field, title, cmap="viridis"):
    image = ax.pcolormesh(grid["y"][..., 0], grid["y"][..., 1],
                         grid[field], shading="nearest", cmap=cmap, rasterized=True)
    coordinate = str(grid.get("coordinate_label", "descriptor / coefficient"))
    ax.set(xlabel=f"{coordinate} 1", ylabel=f"{coordinate} 2",
           xlim=(-1, 1), ylim=(-1, 1), title=title, aspect="equal")
    return image


def _trace_shape(value, vector=False):
    value = np.asarray(value)
    expected = 3 if vector else 2
    return value[None] if value.ndim == expected - 1 else value


def _parse_trace_name(path):
    parts = path.stem.split("__", 1)
    return parts[0], parts[1] if len(parts) > 1 else "chain"


def _target_and_prior(grids, out):
    context, grid = next(iter(grids.items()))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    axis = np.linspace(-3, 3, 100)
    z1, z2 = np.meshgrid(axis, axis)
    prior = np.exp(-0.5 * (z1**2 + z2**2)) / (2 * np.pi)
    image = axes[0].pcolormesh(z1, z2, prior, shading="auto", cmap="Blues", rasterized=True)
    axes[0].set(xlabel="noise z₁", ylabel="noise z₂", title="Gaussian prior φ(z)", aspect="equal")
    fig.colorbar(image, ax=axes[0], shrink=.78, label="noise density")
    image = _field_map(axes[1], grid, "cost", "Bounded evaluation cost J", "magma_r")
    fig.colorbar(image, ax=axes[1], shrink=.78, label="cost")
    image = _field_map(axes[2], grid, "density", "Normalized target p(y)")
    fig.colorbar(image, ax=axes[2], shrink=.78, label="descriptor density")
    fig.suptitle(f"{context}  ·  y = 2 Φ(z) − 1 maps the prior to a uniform square", fontsize=13)
    return _save(fig, out, "target_and_prior")


def _chain_plot(grids, traces, out, burn_in=0):
    selected = list(traces.items())[:6]
    fig, axes = plt.subplots(2, max(1, len(selected)), figsize=(3.2 * max(1, len(selected)), 6.3),
                             squeeze=False, constrained_layout=True)
    for index, ((context, method), trace) in enumerate(selected):
        z = _trace_shape(trace["z"], vector=True)
        y = 2 * ndtr(z) - 1
        grid = grids.get(context)
        top, lower = axes[:, index]
        samples = y[:, min(burn_in, y.shape[1] - 1):].reshape(-1, 2)
        top.hist2d(samples[:, 0], samples[:, 1], bins=25,
                   range=[[-1, 1], [-1, 1]], density=True, cmap="Blues", rasterized=True)
        if grid is not None:
            top.contour(grid["y"][..., 0], grid["y"][..., 1], grid["density"],
                        levels=6, colors="#7f939f", linewidths=.7)
        tail = y[0, -min(128, y.shape[1]):]
        top.plot(tail[:, 0], tail[:, 1], lw=.7, alpha=.65, color="#e58f34")
        top.scatter(y[0, 0, 0], y[0, 0, 1], c="#2b4055", s=30, marker="s", label="initial")
        top.scatter(y[0, -1, 0], y[0, -1, 1], c="#e58f34", s=35, label="final")
        top.set(title=method, xlim=(-1, 1), ylim=(-1, 1), aspect="equal",
                xlabel="descriptor 1", ylabel=context if index == 0 else "")
        if grid is not None and "regions" in grid:
            region_grid = grid["regions"]
            n = region_grid.shape[0]
            cell = np.minimum(n - 1, np.maximum(0, ((y + 1) * n / 2).astype(int)))
            regions = region_grid[cell[..., 1], cell[..., 0]]
            region_count = int(np.max(region_grid)) + 1
            for region in range(region_count):
                indicator = (regions[0] == region).astype(float)
                running = np.cumsum(indicator) / np.arange(1, len(indicator) + 1)
                line, = lower.plot(running, label=f"region {region}")
                if "region_masses" in grid:
                    lower.axhline(grid["region_masses"][region], c=line.get_color(),
                                  ls=":", lw=1, alpha=.8)
            lower.set(ylabel="first-chain cumulative mass", ylim=(-.03, 1.03))
        else:
            lower.plot(y[0, :, 0])
            lower.set(ylabel="descriptor coordinate 1", ylim=(-1, 1))
        lower.set(xlabel="MCMC refinement step")
        lower.grid(axis="y", alpha=.5)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Post-burn-in density, final 128-step trajectory · first-chain occupancy retains rejections · dotted: target masses")
    return _save(fig, out, "chain_and_region_occupancy")


def _proposal_plot(grids, traces, out):
    (context, method), trace = next(iter(traces.items()))
    z = _trace_shape(trace["z"], vector=True)[0]
    proposed = _trace_shape(trace.get("proposed_z", trace["z"]), vector=True)[0]
    if "before_z" in trace:
        before = _trace_shape(trace["before_z"], vector=True)[0]
    else:
        before = np.concatenate([z[:1], z[:-1]], axis=0)
    steps = np.unique(np.linspace(0, len(z) - 1, min(4, len(z))).astype(int))
    fig, axes = plt.subplots(1, len(steps), figsize=(3.8 * len(steps), 3.8), squeeze=False,
                             constrained_layout=True)
    for ax, step in zip(axes[0], steps):
        if context in grids:
            grid = grids[context]
            ax.contour(grid["y"][..., 0], grid["y"][..., 1], grid["density"],
                       levels=6, colors="#c4cfd5", linewidths=.7)
        points = 2 * ndtr(np.stack([before[step], proposed[step], z[step]])) - 1
        ax.annotate("", xy=points[1], xytext=points[0],
                    arrowprops={"arrowstyle": "->", "color": COLORS[1], "lw": 2})
        ax.scatter(*points[0], s=65, c=COLORS[0], label="current", zorder=4)
        ax.scatter(*points[1], s=80, c=COLORS[1], marker="x", label="proposed", zorder=5)
        accepted = bool(_trace_shape(trace.get("accepted", np.ones((1, len(z)))))[0, step])
        details = [f"step {step}: {'accept' if accepted else 'reject'}"]
        for label, field in [("exact", "exact_log_ratio"), ("learned", "learned_log_ratio")]:
            if field in trace:
                value = float(_trace_shape(trace[field])[0, step])
                if np.isfinite(value):
                    details.append(f"a_{label} = {np.exp(min(0, value)):.3f}")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title="\n".join(details),
               xlabel="descriptor 1", ylabel="descriptor 2")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{context} / {method}  ·  one proposal changes one complete candidate")
    return _save(fig, out, "proposal_acceptance_replay")


def _calibration(run, traces, out):
    calibration = run / "calibration.npz"
    if calibration.exists():
        data = _load_npz(calibration)
        exact = data.get("exact_log_ratio", data.get("exact"))
        learned = data.get("learned_log_ratio", data.get("learned"))
    else:
        pairs = [(t["exact_log_ratio"].ravel(), t["learned_log_ratio"].ravel())
                 for t in traces.values() if "exact_log_ratio" in t and "learned_log_ratio" in t]
        if not pairs:
            return None
        exact, learned = [np.concatenate(parts) for parts in zip(*pairs)]
    if exact is None or learned is None:
        return None
    exact, learned = np.ravel(exact), np.ravel(learned)
    good = np.isfinite(exact) & np.isfinite(learned)
    exact, learned = exact[good], learned[good]
    if len(exact) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4), constrained_layout=True)
    stride = max(1, len(exact) // 5000)
    axes[0].scatter(exact[::stride], learned[::stride], s=6, alpha=.22, rasterized=True)
    bounds = np.quantile(np.r_[exact, learned], [.005, .995])
    axes[0].plot(bounds, bounds, "--", c=COLORS[1], lw=1)
    axes[0].set(xlabel="exact log ratio", ylabel="learned antisymmetric log ratio",
                title=f"Raw ratio MSE = {np.mean((exact - learned)**2):.3g}")
    exact_a = np.exp(np.minimum(0, exact))
    learned_a = np.exp(np.minimum(0, learned))
    axes[1].scatter(exact_a[::stride], learned_a[::stride], s=6, alpha=.22, rasterized=True)
    axes[1].plot([0, 1], [0, 1], "--", c=COLORS[1], lw=1)
    axes[1].set(xlabel="exact acceptance", ylabel="learned acceptance", xlim=(-.03, 1.03),
                ylim=(-.03, 1.03), title=f"Mean |acceptance error| = {np.mean(abs(exact_a-learned_a)):.3g}")
    fig.suptitle(f"Calibration from {len(exact):,} recorded pairs  ·  empirical estimates, not certificates")
    return _save(fig, out, "ratio_calibration")


def _refresh_plot(metrics, out):
    selected = [row for row in metrics
                if (row.get("method") == "learned" or str(row.get("method", "")).startswith("refresh_"))
                and np.isfinite(_value(row, "rho", "refresh_rate"))]
    if not selected:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    contexts = sorted({_context(row) for row in selected})
    for context in contexts:
        rows = sorted([row for row in selected if _context(row) == context], key=lambda r: _value(r, "rho", "refresh_rate"))
        rho = [_value(r, "rho", "refresh_rate") for r in rows]
        for ax, fields in zip(axes, [("region_tv", "region_error", "region_mass_error"),
                                   ("mode_switches", "switches"), ("exact_cost_evaluations", "cost_evaluations")]):
            ax.plot(rho, [_value(row, *fields) for row in rows], "o-", label=context, ms=4)
    for ax, title, ylabel in zip(axes, ["Target-region mass error", "Exploration", "True oracle work"],
                                 ["region TV", "mode switches", "cost evaluations"]):
        ax.set(xlabel="Bernoulli exact-refresh probability ρ", ylabel=ylabel, title=title)
        ax.grid(alpha=.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen-kernel empirical trade-off  ·  finite chains need not improve monotonically")
    return _save(fig, out, "refresh_bias_tradeoff")


def _finite_plot(run, out):
    source = _read_json(run / "finite_grid.json", None)
    if source is None:
        source = _read_json(run / "audit.json", None)
    if source is None:
        return None
    if "finite_grid" in source:
        source = source["finite_grid"]
    if "variants" not in source:
        values = list(source.values()) if isinstance(source, dict) else source
        source = next((v for v in values if isinstance(v, dict) and "variants" in v), None)
    if source is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    variants = source["variants"]
    positions = np.arange(len(variants))
    for index, field in enumerate(["one_step_tv", "flow_bound", "epsilon_a"]):
        axes[0].bar(positions + (index - 1) * .23,
                    [max(1e-16, row[field]) for row in variants.values()],
                    width=.23, label=field.replace("_", " "))
    axes[0].set(xticks=positions, xticklabels=list(variants), yscale="log", ylabel="TV / bound",
                title="Exact finite accepted-flow sums")
    axes[0].legend(fontsize=8)
    for name, row in variants.items():
        refresh = [r for r in row["refresh"] if r["rho"] > 0]
        line, = axes[1].plot([r["rho"] for r in refresh],
                            [max(1e-16, r["stationary_tv"]) for r in refresh], "o-", label=f"{name}: actual")
        axes[1].plot([r["rho"] for r in refresh],
                     [max(1e-16, r["stationary_bound"]) if r["stationary_bound"] is not None else np.nan
                      for r in refresh], ":", color=line.get_color(), label=f"{name}: bound")
    axes[1].axhline(1, c="#777777", lw=1, ls="--")
    axes[1].set(xlabel="refresh probability ρ", ylabel="stationary TV / raw bound",
                yscale="log", title="Bounds above 1 are vacuous")
    axes[1].legend(fontsize=7)
    failure = source.get("identity_failure", [])
    if failure:
        axes[2].loglog([r["jump_probability"] for r in failure],
                       [r["one_step_tv"] for r in failure], "o-", label="one-step error")
        axes[2].loglog([r["jump_probability"] for r in failure],
                       [r["stationary_tv"] for r in failure], "s-", label="stationary error")
    axes[2].set(xlabel="jump probability ε", ylabel="TV", title="Almost identity can remain biased")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Separate {source.get('grid_size', '?')}×{source.get('grid_size', '?')} state problem · discrete Hastings ratios recomputed")
    return _save(fig, out, "finite_grid_bound_audit")


def _route_plot(routes, out, name="route_repair_replay"):
    selected = list(routes.items())[:3]
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.4), squeeze=False,
                             constrained_layout=True)
    for ax, ((context, method), data) in zip(axes[0], selected):
        commands = data.get("commands", np.empty((0, 2)))
        if commands.ndim == 3:
            commands = commands[-1]
        positions = data.get("positions", np.empty((0, 2)))
        if positions.ndim == 3:
            positions = positions[-1]
        for field, color, style, label, width in [
            ("old_plan", "#9ba9b5", "--", "cached old plan", 1.6),
            ("retained_suffix", "#7964b4", "-", "retained suffix", 2),
            ("completed_tail", "#e58f34", "--", "completed tail", 2.3),
        ]:
            if field in data:
                values = data[field]
                if values.ndim == 3:
                    values = values[-1]
                ax.plot(values[:, 0], values[:, 1], style, color=color, lw=width, label=label)
                if field == "completed_tail":
                    ax.scatter(values[:, 0], values[:, 1], s=60, marker="s",
                               facecolors="none", edgecolors=color, linewidths=1, zorder=7)
        if len(commands):
            ax.plot(commands[:, 0], commands[:, 1], "o-", color=COLORS[0], ms=3, lw=1.5, label="whole proposed command chunk")
        # The next cache overlaps the selected commands by construction.
        # Draw its retained points as hollow markers so both objects remain
        # visible without displacing measured coordinates.
        if "selected_commands" in data and "retained_suffix" in data:
            retained = data["retained_suffix"]
            ax.scatter(retained[:, 0], retained[:, 1], s=42, facecolors="none",
                       edgecolors="#7964b4", linewidths=1, zorder=5)
        if "selected_commands" in data and "completed_tail" in data:
            tail = data["completed_tail"]
            ax.plot(tail[:, 0], tail[:, 1], color="#e58f34", lw=3,
                    ls="--", zorder=6)
        if len(positions):
            ax.plot(positions[:, 0], positions[:, 1], "-", c="#25394f", lw=2.2, label="simulated physical rollout")
        for field in ("executed_positions", "executed_prefix"):
            if field in data:
                values = data[field]
                if values.ndim == 3:
                    values = values[-1]
                ax.plot(values[:, 0], values[:, 1], c="#cb5a5e", lw=3,
                        label="actually executed prefix" if field == "executed_positions" else "selected command prefix")
                break
        centers = data.get("obstacle_centers", np.empty((0, 2)))
        radii = data.get("obstacle_radii", np.empty(0))
        for center, radius in zip(centers, radii):
            if radius > 0:
                ax.add_patch(plt.Circle(center, radius, fc="#d3dbe2", ec="#6d7d8c", alpha=.8))
        if "goal" in data:
            ax.scatter(*data["goal"], marker="*", c="#cf9b27", s=130, zorder=6, label="goal")
        ax.set(title=f"{context}\n{method}", aspect="equal", xlabel="physical position x",
               ylabel="physical position y", xlim=(-1.3, 1.3), ylim=(-1.25, 1.25))
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("MCMC replaces a complete future chunk; physical execution follows its selected prefix")
    return _save(fig, out, name)


def _support_plot(run, routes, out):
    files = sorted((run / "routes").glob("support__*.npz"))
    if not files:
        return None
    fig, axes = plt.subplots(len(files), 2, figsize=(10, 3.8 * len(files)), squeeze=False,
                             constrained_layout=True)
    for row, path in enumerate(files):
        data = _load_npz(path)
        name = path.stem.removeprefix("support__")
        coefficients = data["coefficients"]
        costs = data["cost"]
        image = axes[row, 0].scatter(coefficients[:, 0], coefficients[:, 1], c=costs,
                                      cmap="magma_r", marker="s", s=80)
        fig.colorbar(image, ax=axes[row, 0], label="actual full-chunk cost")
        axes[row, 0].set(title=f"{name}: {int(data['success'].sum())}/{len(costs)} successful grid candidates",
                        xlabel="coefficient 1", ylabel="coefficient 2", aspect="equal")
        ax = axes[row, 1]
        for index in range(0, len(costs), max(1, len(costs) // 30)):
            commands = data["commands"][index]
            ax.plot(commands[:, 0], commands[:, 1], color="#bbc7d0", alpha=.6, lw=.8)
        for index, color, label in [(int(np.argmin(costs)), COLORS[0], "lowest-cost grid candidate"),
                                     (int(np.argmax(costs)), COLORS[3], "highest-cost grid candidate")]:
            commands, positions = data["commands"][index], data["positions"][index]
            ax.plot(commands[:, 0], commands[:, 1], "--", color=color, lw=1.5, label=f"{label}: commands")
            ax.plot(positions[:, 0], positions[:, 1], color=color, lw=2.3, label="physical rollout")
        geometry = next((v for (context, _), v in routes.items() if context == name), {})
        for center, radius in zip(geometry.get("obstacle_centers", []), geometry.get("obstacle_radii", [])):
            if radius > 0:
                ax.add_patch(plt.Circle(center, radius, fc="#cfd8df", ec="#7a8a99", alpha=.7))
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.25, 1.25), aspect="equal",
               title="Engineered decoder support, before any correction",
               xlabel="physical x", ylabel="physical y")
        ax.legend(fontsize=7)
    fig.suptitle("Support audit · all costs and physical paths were actually evaluated and charged")
    return _save(fig, out, "route_decoder_support")


def _planning_plots(metrics, run, out):
    planning = _read_json(run / "planning_results.json", _read_json(run / "planning.json", []))
    if isinstance(planning, dict):
        planning = planning.get("metrics", planning.get("results", []))
    rows = planning if planning else metrics
    results = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    method_names = sorted({str(row.get("method", "method")) for row in rows})
    any_success = False
    for method in method_names:
        subset = [row for row in rows if row.get("method") == method]
        steps = [_value(row, "simulator_steps", "physical_simulator_steps") for row in subset]
        success = [_value(row, "success_rate", "planner_success", "success") for row in subset]
        costs = [_value(row, "best_cost", "final_cost", "cost", "mean_cost") for row in subset]
        any_success |= bool(np.any(np.isfinite(success)))
        axes[0].scatter(steps, success, label=method, alpha=.8, s=38)
        axes[1].scatter(steps, costs, label=method, alpha=.8, s=38)
    if any_success:
        axes[0].set(xlabel="counted simulator substeps", ylabel="success / success fraction", ylim=(-.05, 1.05))
        axes[1].set(xlabel="counted simulator substeps", ylabel="best evaluated candidate cost")
        axes[0].legend(fontsize=7)
        fig.suptitle("Planner outcomes at bounded oracle budgets · single-run points are not uncertainty estimates")
        results.append(_save(fig, out, "success_vs_simulator_steps"))
    else:
        plt.close(fig)
    if not metrics:
        return results
    methods = list(dict.fromkeys(str(row.get("method", "?")) for row in metrics))
    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(methods) * .7), 4.7), constrained_layout=True)
    for ax, fields, title in zip(axes,
             [("acceptance_rate", "acceptance"), ("region_tv", "region_error", "coarse_tv"),
              ("accepted_movement", "mean_accepted_movement", "mode_switches")],
             ["Acceptance", "Posterior region error", "Exploration / accepted movement"]):
        values = []
        for method in methods:
            items = [_value(row, *fields) for row in metrics if row.get("method") == method]
            finite = np.asarray(items)[np.isfinite(items)]
            values.append(np.mean(finite) if len(finite) else np.nan)
        ax.bar(np.arange(len(methods)), values, color=[COLORS[i % len(COLORS)] for i in range(len(methods))])
        ax.set(xticks=np.arange(len(methods)), xticklabels=methods, title=title)
        ax.tick_params(axis="x", labelrotation=65, labelsize=8)
        ax.grid(axis="y", alpha=.35)
    fig.suptitle("Posterior sampling diagnostics · averages over saved contexts, not planner success")
    results.append(_save(fig, out, "sampler_vs_planner_summary"))
    return results


def _json_array(array):
    """Round display coordinates and encode missing audit values as JSON null.

    Full numerical precision remains in the original NPZ files. Six decimals
    exceed the screen resolution and avoid a needlessly large HTML download.
    """
    array = np.asarray(array)
    if array.dtype.kind in "fc":
        array = np.round(array, decimals=6)
        return np.where(np.isfinite(array), array, None).tolist()
    return array.tolist()


def _html_replay(run, grids, traces, routes, metrics, out):
    records = []
    metadata = _read_json(run / "metadata.json", {})
    context_metadata = {item["name"]: item for item in metadata.get("contexts", [])}
    for (context, method), trace in traces.items():
        z = _trace_shape(trace["z"], vector=True)[0]
        count = len(z)
        record = {"context": context, "method": method, "steps": count}
        metric = next((row for row in metrics if _context(row) == context and row.get("method") == method), {})
        record["beta"] = _value(metric, "beta", "temperature",
                                default=context_metadata.get(context, {}).get("beta", 1.))
        record["rho"] = _value(metric, "rho", "refresh_rate", default=0.)
        for field in ("z", "proposed_z", "before_z"):
            if field in trace:
                value = _trace_shape(trace[field], vector=True)[0]
                if field == "z":
                    record[field] = _json_array(value)
                record[field.replace("z", "y")] = _json_array(2 * ndtr(value) - 1)
        if "before_y" not in record:
            record["before_y"] = [record["y"][0]] + record["y"][:-1]
        for field in ("accepted", "exact_log_ratio", "learned_log_ratio", "cost", "branches"):
            if field in trace:
                record[field] = _json_array(_trace_shape(trace[field])[0])
        route = routes.get((context, method))
        if route is not None:
            record["route"] = {key: _json_array(value) for key, value in route.items()}
            if "commands" in route and route["commands"].ndim == 3:
                # A route audit may intentionally cover only the first 48
                # transitions; never replay its final route beside later z.
                record["steps"] = min(record["steps"], len(route["commands"]))
        records.append(record)
    # Planner replays can exist independently of posterior chains.
    for (context, method), route in routes.items():
        if (context, method) not in traces:
            commands = route.get("commands", np.zeros((1, 1, 2)))
            count = len(commands) if commands.ndim == 3 else 1
            z = route.get("z", np.zeros((count, 2)))
            if z.ndim == 1:
                z = z[None]
            y = 2 * ndtr(z) - 1
            records.append({"context": context, "method": method, "steps": count,
                            "beta": float(np.asarray(route.get("beta", context_metadata.get(context, {}).get("beta", 1.)))), "rho": 0.,
                            "z": _json_array(z), "y": _json_array(y),
                            "proposed_y": _json_array(y), "before_y": _json_array(y),
                            "route": {key: _json_array(value) for key, value in route.items()}})
    grid_data = {}
    for name, grid in grids.items():
        # Compact pixels keep the standalone artifact responsive. This is
        # display decimation only; numerical references retain the full arrays.
        stride = max(1, grid["density"].shape[0] // 70)
        grid_data[name] = {"density": _json_array(grid["density"][::stride, ::stride]),
                           "cost": _json_array(grid["cost"][::stride, ::stride]),
                           "coordinate_label": str(grid.get("coordinate_label", "descriptor / coefficient space")),
                           "region_masses": _json_array(grid.get("region_masses", []))}
    payload = json.dumps({"run": run.name, "records": records, "grids": grid_data},
                         ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    markup = _HTML_TEMPLATE.replace("__DATA__", payload).replace("__RUN__", html.escape(run.name))
    path = out / "replay.html"
    path.write_text(markup)
    return path


def render(run_dir: str | Path) -> dict:
    """Build figures and a standalone replay using only existing run artifacts."""
    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(run)
    _style()
    out = run / "figures"
    out.mkdir(exist_ok=True)
    grids = {p.stem: _load_npz(p) for p in sorted((run / "grids").glob("*.npz"))
             if not p.stem.endswith("__fine")}
    traces = {_parse_trace_name(p): _load_npz(p) for p in sorted((run / "traces").glob("*.npz"))}
    for trace in traces.values():
        if "audit_exact_log_ratio" in trace:
            exact = trace.get("exact_log_ratio", np.full_like(trace["audit_exact_log_ratio"], np.nan))
            trace["exact_log_ratio"] = np.where(np.isfinite(exact), exact, trace["audit_exact_log_ratio"])
        if "audit_current_cost" in trace:
            # These arrays are copies used only for display. The original
            # cost-validity and branch metadata remain untouched on disk.
            cost = trace.get("cost", np.full_like(trace["audit_current_cost"], np.nan))
            # audit_current_cost describes before_z, not necessarily the stored
            # post-transition state. Select proposed cost only when accepted.
            post = np.where(trace["accepted"], trace["audit_proposed_cost"], trace["audit_current_cost"])
            trace["cost"] = np.where(np.isfinite(cost), cost, post)
    routes = {_parse_trace_name(p): _load_npz(p) for p in sorted((run / "routes").glob("*.npz"))
              if not p.stem.startswith("support__")}
    for route in routes.values():
        route["execution_kind"] = np.asarray("historical prefix that created the snapshot")
        if "executed_positions" in route:
            route["historical_positions"] = route.pop("executed_positions")
        route.pop("executed_prefix", None)
    for path in sorted((run / "planning").glob("*.npz")):
        context, method = _parse_trace_name(path)
        saved = _load_npz(path)
        geometry = next((route for (name, _), route in routes.items() if name == context), {})
        route = {field: geometry[field] for field in ("obstacle_centers", "obstacle_radii", "start", "goal")
                 if field in geometry}
        route.update(saved)
        route["commands"] = saved["selected_commands"][None]
        route["z"] = np.atleast_2d(saved["z"])
        route["execution_kind"] = np.asarray("selected plan's actually executed prefix")
        routes[(context, f"planner_{method}")] = route
    metrics = _metrics(run)
    # Start the tour on an informative two-mode landscape instead of the
    # alphabetically first constant-cost correctness case.
    def context_order(name):
        return (0 if name == "two" else 1 if "invalidation" in name else
                2 if "two" in name and "beta" not in name else 3, name)
    grids = dict(sorted(grids.items(), key=lambda item: context_order(item[0])))
    method_order = {"pcn": 0, "rw": 1, "tiny_rw": 2, "high_locality": 2,
                    "learned_exact": 3, "learned": 4, "learned_refresh": 5, "refresh_0.1": 5}
    traces = dict(sorted(traces.items(), key=lambda item: (
        context_order(item[0][0]), method_order.get(item[0][1], 6), item[0][1])))
    written = []
    if grids:
        written.append(_target_and_prior(grids, out))
    if traces:
        burn_in = int(_read_json(run / "config.json", {}).get("burn_in", 0))
        written.extend([_chain_plot(grids, traces, out, burn_in), _proposal_plot(grids, traces, out)])
    written.extend([_calibration(run, traces, out), _refresh_plot(metrics, out), _finite_plot(run, out)])
    if routes:
        selected = {key: value for key, value in routes.items() if key[1] == "planner_pcn"}
        if not selected:
            selected = {key: value for key, value in routes.items() if key[1].startswith("planner_")}
        written.append(_route_plot(selected or routes, out))
        posterior = {key: value for key, value in routes.items() if key[1] == "pcn"}
        if posterior:
            written.append(_route_plot(posterior, out, "route_proposal_rollouts"))
        written.append(_support_plot(run, routes, out))
    written.extend(_planning_plots(metrics, run, out))
    replay = _html_replay(run, grids, traces, routes, metrics, out)
    manifest = {"figures": [name for name in written if name], "html": str(replay),
                "oracle_evaluations_during_render": 0,
                "note": "Only saved measurements; replay controls never train or simulate. HTML display floats are rounded to six decimals; original NPZ precision is preserved."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ai-Plan · __RUN__</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#183047;background:#f2f5f8}
body{max-width:1400px;margin:auto;padding:26px}h1{font-size:30px;margin:0 0 8px;letter-spacing:-.7px}
p{color:#546779;line-height:1.5}header{margin-bottom:20px}.tag{color:#087f8c;text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700}
.controls,.card{background:white;border:1px solid #dce4eb;border-radius:12px;padding:18px}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin-bottom:16px}
label{display:flex;flex-direction:column;gap:6px;font-size:12px;font-weight:650;color:#526477}
select,button{border:1px solid #c6d3df;background:#fff;padding:9px;border-radius:6px;color:#183047}
button{background:#087f8c;color:white;cursor:pointer}input{accent-color:#087f8c}#step{min-width:250px}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card h2{font-size:17px;margin:0 0 10px}
canvas{width:100%;display:block;aspect-ratio:1.15}#readout{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.number{background:white;border:1px solid #dce4eb;border-radius:9px;padding:15px;font-size:21px}.number span{font-size:11px;display:block;text-transform:uppercase;letter-spacing:.6px;color:#61768a;margin-bottom:7px}
.legend{font-size:12px;display:flex;gap:14px;flex-wrap:wrap;line-height:1.8}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.note{font-size:12px;line-height:1.6}#status{font-size:13px;margin-top:8px}a{color:#087f8c}
@media(max-width:800px){.panels{grid-template-columns:1fr}#readout{grid-template-columns:1fr 1fr}body{padding:12px}}
</style>
<header><div class="tag">Executable research pilot · saved measurements</div><h1>How one noise proposal changes an entire plan</h1>
<p>__RUN__ · Explore fixed-context MCMC refinement and physical plan execution. Every frame comes from saved arrays. Controls do not train networks or call a simulator.</p></header>
<div class="controls">
<label>Context<select id="context"></select></label><label>Temperature β<select id="beta"></select></label>
<label>Method<select id="method"></select></label><label>Exact refresh ρ<select id="rho"></select></label>
<label>Chain / candidate step <input id="step" type="range" min="0" value="0"></label>
<button id="play">Play</button><label>Landscape<select id="field"><option value="density">Normalized target density</option><option value="cost">Bounded cost</option></select></label>
</div><div id="status"></div>
<div id="readout"><div class="number"><span>Exact acceptance (audit)</span><b id="exact">—</b></div>
<div class="number"><span>Learned acceptance</span><b id="learned">—</b></div>
<div class="number"><span>Transition result</span><b id="decision">—</b></div>
<div class="number"><span>State's measured cost</span><b id="cost">—</b></div></div>
<div class="panels"><section class="card"><h2 id="latentTitle">Descriptor / coefficient space</h2><canvas id="latent"></canvas>
<div class="legend"><span><i class="dot" style="background:#087f8c"></i>current</span><span><i class="dot" style="background:#e58f34"></i>proposed</span><span><i class="dot" style="background:#61768a"></i>stored states, including rejections</span></div></section>
<section class="card"><h2 id="rightTitle">Running region occupancy</h2><canvas id="physical"></canvas>
<div class="legend" id="rightLegend"></div></section></div>
<p class="note">MCMC step is not physical time. The analytic decoder is frozen and engineered; no DSBM revision trajectory was trained.
Exact probabilities shown for learned transitions are recorded audit values, whose oracle work is counted separately.
An unavailable exact value is shown as “—”. Planner incumbents are not posterior samples.
Temperature choices are limited to saved context-specific runs; no samples are overlaid on a different target.</p>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id), records=DATA.records;
let record=null,timer=null;
const colors=['#087f8c','#e58f34','#7964b4','#cb5a5e','#4878a8'];
function options(id,items,preserve=true){let old=$(id).value;$(id).innerHTML='';[...new Set(items)].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v;$(id).appendChild(o)});if(preserve&&items.map(String).includes(old))$(id).value=old;}
function choose(level=0){
if(level<=0)options('context',records.map(r=>r.context));
let subset=records.filter(r=>r.context==$('context').value);
if(level<=1)options('beta',subset.map(r=>String(r.beta)));
subset=subset.filter(r=>String(r.beta)==$('beta').value);
if(level<=2)options('method',subset.map(r=>r.method));
subset=subset.filter(r=>r.method==$('method').value);
if(level<=3)options('rho',subset.map(r=>String(r.rho)));
record=subset.find(r=>String(r.rho)==$('rho').value)||subset[0];
if(!record){$('status').textContent='No compatible saved records.';return}
$('step').max=Math.max(0,record.steps-1);$('step').value=Math.min(+$('step').value,record.steps-1);draw();}
function setup(id){let c=$(id),dpr=window.devicePixelRatio||1,w=c.clientWidth,h=w/1.15;c.width=w*dpr;c.height=h*dpr;let ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);return {ctx,w,h,pad:42};}
function frame(plot,range=[-1,1,-1,1]){let {ctx,h,pad}=plot,w=plot.w;
// Preserve physical geometry and compact-square coordinates. Occupancy charts
// deliberately have a time axis and therefore use the full available width.
if(range[1]-range[0]<4){w=Math.min(w,2*pad+(h-2*pad)*(range[1]-range[0])/(range[3]-range[2]));plot.w=w;}
plot.xy=p=>[pad+(p[0]-range[0])/(range[1]-range[0])*(w-2*pad),h-pad-(p[1]-range[2])/(range[3]-range[2])*(h-2*pad)];
ctx.strokeStyle='#cbd5e1';ctx.strokeRect(pad,pad,w-2*pad,h-2*pad);ctx.fillStyle='#64748b';ctx.font='11px system-ui';
for(let i=0;i<5;i++){let v=i/4,x=pad+v*(w-2*pad),y=h-pad-v*(h-2*pad);ctx.fillText((range[0]+v*(range[1]-range[0])).toFixed(1),x-10,h-pad+18);ctx.fillText((range[2]+v*(range[3]-range[2])).toFixed(1),6,y+4);}
return plot;}
function validPoint(p){return p&&p.length>=2&&p.every(v=>v!==null&&Number.isFinite(v))}
function path(p,pts,color,width=2,dash=[]){if(!pts||!pts.length)return;let c=p.ctx;c.beginPath();c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);pts.filter(validPoint).forEach((q,i)=>{let xy=p.xy(q);if(i)c.lineTo(...xy);else c.moveTo(...xy)});c.stroke();c.setLineDash([]);}
function dot(p,point,color,r=5){if(!validPoint(point))return;let xy=p.xy(point);p.ctx.beginPath();p.ctx.arc(...xy,r,0,2*Math.PI);p.ctx.fillStyle=color;p.ctx.fill();}
function arrow(p,a,b){if(!validPoint(a)||!validPoint(b))return;path(p,[a,b],'#e58f34',2.5);let x=p.xy(a),y=p.xy(b),ang=Math.atan2(y[1]-x[1],y[0]-x[0]);p.ctx.beginPath();p.ctx.moveTo(...y);p.ctx.lineTo(y[0]-10*Math.cos(ang-.4),y[1]-10*Math.sin(ang-.4));p.ctx.lineTo(y[0]-10*Math.cos(ang+.4),y[1]-10*Math.sin(ang+.4));p.ctx.closePath();p.ctx.fillStyle='#e58f34';p.ctx.fill();}
function heatmap(p,grid){let field=grid[$('field').value],n=field.length,max=Math.max(...field.flat()),min=Math.min(...field.flat()),{ctx,w,h,pad}=p;
for(let iy=0;iy<n;iy++)for(let ix=0;ix<field[iy].length;ix++){let t=(field[iy][ix]-min)/(max-min||1);let r=Math.round(244-218*t),g=Math.round(249-124*t),b=Math.round(252-115*t);ctx.fillStyle=`rgb(${r},${g},${b})`;ctx.fillRect(pad+ix*(w-2*pad)/n,h-pad-(iy+1)*(h-2*pad)/n,(w-2*pad)/n+.5,(h-2*pad)/n+.5);}}
function at(a,s){return a&&a[Math.min(s,a.length-1)]}
function acceptance(v){return v===undefined||v===null?'—':Math.exp(Math.min(0,v)).toFixed(3)}
function number(v){return v===undefined||v===null?'—':Number(v).toFixed(3)}
function routeFrame(a,s){if(!a)return null;return Array.isArray(a[0])&&Array.isArray(a[0][0])?at(a,s):a}
function physical(s){let p=frame(setup('physical'),[-1.3,1.3,-1.25,1.25]),r=record.route;
$('rightTitle').textContent='Whole commands and realized physical rollout';
let cs=r.obstacle_centers||[],rs=r.obstacle_radii||[];
cs.forEach((q,i)=>{if(!(rs[i]>0))return;let xy=p.xy(q);p.ctx.beginPath();p.ctx.arc(...xy,rs[i]/2.6*(p.w-2*p.pad),0,2*Math.PI);p.ctx.fillStyle='#cfd8e1';p.ctx.fill();p.ctx.strokeStyle='#778899';p.ctx.stroke();});
path(p,routeFrame(r.old_plan,s),'#9ba9b5',1.8,[6,4]);path(p,routeFrame(r.retained_suffix,s),'#7964b4',2.4);
path(p,routeFrame(r.completed_tail,s),'#e58f34',3,[5,3]);path(p,routeFrame(r.commands,s),'#087f8c',2);
let commands=routeFrame(r.commands,s)||[];commands.forEach(q=>dot(p,q,'#087f8c',2));
path(p,routeFrame(r.positions,s),'#25394f',2.8);
path(p,routeFrame(r.historical_positions,s),'#8999a7',2,[3,3]);
path(p,routeFrame(r.executed_positions,s)||routeFrame(r.executed_prefix,s),'#cb5a5e',3.5);
dot(p,r.goal,'#cf9b27',7);$('rightLegend').innerHTML='<span style="color:#9ba9b5">-- old cache</span><span style="color:#7964b4">retained suffix</span><span style="color:#e58f34">-- completed tail</span><span style="color:#087f8c">whole commands</span><span style="color:#25394f">physical rollout</span>'+(r.executed_positions?'<span style="color:#cb5a5e">actually executed selected prefix</span>':'<span style="color:#8999a7">-- historical prefix that created snapshot</span>');
}
function occupancy(s){let p=frame(setup('physical'),[0,Math.max(1,record.steps-1),0,1]);
$('rightTitle').textContent='Cumulative half-square occupancy (all stored states)';
let count=[0,0],curves=[[],[]];record.y.slice(0,s+1).forEach((v,i)=>{count[v[0]>=0?1:0]++;for(let k=0;k<2;k++)curves[k].push([i,count[k]/(i+1)]);});curves.forEach((c,i)=>path(p,c,colors[i],2.4));
let g=DATA.grids[record.context];if(g){let a=g.density,total=0,right=0;a.forEach(row=>row.forEach((v,i)=>{total+=v;if(i>=row.length/2)right+=v}));path(p,[[0,1-right/total],[record.steps-1,1-right/total]],colors[0],1,[5,4]);path(p,[[0,right/total],[record.steps-1,right/total]],colors[1],1,[5,4]);}
$('rightLegend').innerHTML='<span style="color:#087f8c">left half</span><span style="color:#e58f34">right half</span><span>dashed: saved numerical reference</span>';
}
function draw(){if(!record)return;let s=+$('step').value,p=frame(setup('latent')),g=DATA.grids[record.context];if(g)heatmap(p,g);$('latentTitle').textContent=g?g.coordinate_label:'descriptor / coefficient space';
path(p,record.y.slice(0,s+1),'#61768a',1.1);arrow(p,at(record.before_y,s),at(record.proposed_y,s));dot(p,at(record.before_y,s),'#087f8c',6);dot(p,at(record.proposed_y,s),'#e58f34',5);
if(record.route)physical(s);else occupancy(s);
$('exact').textContent=acceptance(at(record.exact_log_ratio,s));$('learned').textContent=acceptance(at(record.learned_log_ratio,s));
let accepted=at(record.accepted,s);$('decision').textContent=accepted===undefined?'candidate':accepted?'accept':'reject';$('decision').style.color=accepted?'#087f8c':'#cb5a5e';
$('cost').textContent=number(at(record.cost,s));let z=at(record.z,s)||[],branch=at(record.branches,s);$('status').textContent=`${record.context} · ${record.method} · β=${record.beta} · ρ=${record.rho} · saved step ${s+1}/${record.steps} · state z=[${z.map(v=>v===null?'unavailable':v.toFixed(3)).join(', ')}]`+(branch===undefined?'':` · ${['exact MH','learned acceptance','exact prior refresh','uncorrected prior'][branch]}`);
}
['context','beta','method','rho'].forEach((id,i)=>$(id).onchange=()=>choose(i+1));
$('step').oninput=draw;$('field').onchange=draw;window.onresize=draw;
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';}else{$('play').textContent='Pause';timer=setInterval(()=>{let v=+$('step').value;$('step').value=v>=+$('step').max?0:v+1;draw()},180)}};
if(records.length)choose();else $('status').textContent='No saved traces were found in this run.';
</script></html>"""
