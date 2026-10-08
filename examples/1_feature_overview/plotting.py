"""Presentation helpers for the feature overview; no estimators live here."""

from html import escape

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from eid.compat import flatten_phi_tree
from eid.discrete import effective_information_from_tpm, source_target_tpm
from eid.spt import SPTNonnegativityError, audit_syn_value, flatten_atoms, flatten_nodes

COLORS = {
    "ei": "#1F3B73",
    "single": "#748399",
    "xi": "#168B80",
    "accent": "#C56B3C",
    "gold": "#C99835",
    "light": "#ECF2F7",
}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#B4BECA",
    "axes.labelcolor": "#25364B",
    "text.color": "#25364B",
    "xtick.color": "#53657B",
    "ytick.color": "#53657B",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "lines.linewidth": 1.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}
PROBABILITY_CMAP = LinearSegmentedColormap.from_list("eid_probability", ["#F4F7FA", COLORS["ei"]])


def panel(ax, letter, description):
    ax.set_title(f"$\\bf{{{letter}}}$  {description}", loc="left")


def outside_legend(ax, **kwargs):
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, **kwargs)


def show_table(headers, rows):
    """Render a compact notebook table without introducing pandas."""
    heading = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    display(
        HTML(
            "<style>.eid-table{border-collapse:collapse;font:13px sans-serif;color:#25364B;"
            "width:auto;max-width:100%}.eid-table th{background:#ECF2F7;text-align:left;"
            "border-bottom:2px solid #B4BECA}.eid-table td,.eid-table th{padding:8px 14px;"
            "border-bottom:1px solid #E0E7EF}.eid-table tr:nth-child(even){background:#F8FAFC}"
            '</style><table class="eid-table"><thead><tr>'
            + heading
            + "</tr></thead><tbody>"
            + body
            + "</tbody></table>"
        )
    )


def show_results(named_results):
    """Keep units, raw minima, affected counts and closure next to estimates."""
    rows = []
    for name, result in named_results.items():
        audit = result.diagnostics["spt_audit"]
        minimum = audit["minimum_candidate_syn"]
        rows.append(
            [
                name,
                f"{result.ei:.4f}",
                f"{sum(result.singleton_ei.values()):.4f}",
                f"{result.xi:.4f}",
                result.unit,
                result.diagnostics["syn_tolerance"],
                "n/a" if minimum is None else f"{minimum:.4g}",
                audit["tolerance_zero_count"],
                result.diagnostics["xi_tolerance_negative_count"],
                f"{result.tree.closure_error:.1e}",
            ]
        )
    show_table(
        [
            "System",
            "Joint EI",
            "Σ single EI",
            "Ξ",
            "Unit",
            "Syn tol.",
            "Min. candidate Syn",
            "Tol. Syn count",
            "Tol. Ξ count",
            "Closure",
        ],
        rows,
    )


def plot_decomposition(ax, named_results):
    results = list(named_results.values())
    units = {result.unit for result in results}
    if len(units) != 1:
        raise ValueError("A comparison must use the same information unit.")
    positions = np.arange(len(results))
    series = [
        ("Joint EI", [r.ei for r in results], COLORS["ei"]),
        ("Σ single EI", [sum(r.singleton_ei.values()) for r in results], COLORS["single"]),
        ("Integrated EI  Ξ", [r.xi for r in results], COLORS["xi"]),
    ]
    for offset, (label, values, color) in zip([-0.24, 0, 0.24], series):
        ax.barh(positions + offset, values, height=0.21, color=color, label=label)
    ax.set_yticks(positions, list(named_results))
    ax.invert_yaxis()
    ax.set_xlabel(f"Information ({results[0].unit})")
    ax.axvline(0, color="#B4BECA", lw=0.8)
    outside_legend(ax)


def plot_tpm(ax, matrix, row_labels=None, column_labels=None):
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap=PROBABILITY_CMAP, aspect="auto")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)), row_labels)
    if column_labels is not None:
        ax.set_xticks(range(len(column_labels)), column_labels)
    ax.set_xlabel("Future state")
    ax.set_ylabel("Intervened source state")
    if matrix.size <= 64:
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}" if 0 < value < 1 else f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 0.55 else "#25364B",
            )
    return image


def tree_share_data(root, *, syn_tolerance):
    """Scale raw node Syn by root Xi; audit before displaying any percentages.

    Tolerance-scale negatives stay signed. Percentages are undefined when the
    denominator is at or below the explicitly declared native-unit tolerance.
    """
    if not np.isfinite(syn_tolerance) or syn_tolerance < 0:
        raise ValueError("Syn tolerance must be finite and nonnegative.")
    if not np.isfinite(root.xi_value):
        raise ValueError("Root integrated EI must be finite.")
    if root.xi_value < -syn_tolerance:
        raise SPTNonnegativityError(
            f"Xi nonnegativity violation in tree display: minimum={root.xi_value:.12g}, "
            f"threshold={-syn_tolerance:.12g}, affected_count=1."
        )
    nodes = flatten_nodes(root)
    negative_count = sum(
        audit_syn_value(node.syn_value, tolerance=syn_tolerance, context=f"tree node {node.sources}")
        for node in nodes
    )
    closure = float(sum(node.syn_value for node in nodes) - root.xi_value)
    if not np.isclose(closure, 0, rtol=0, atol=1e-10):
        raise ValueError(f"Tree contribution closure failed: error={closure:.12g}.")
    percentages = {
        node.sources: (100 * node.syn_value / root.xi_value if root.xi_value > syn_tolerance else None)
        for node in nodes
    }
    return {
        "percentages": percentages,
        "total_xi": root.xi_value,
        "syn_tolerance": syn_tolerance,
        "selected_tolerance_negative_count": negative_count,
        "minimum_selected_syn": min(node.syn_value for node in nodes),
        "closure_error": closure,
    }


def plot_tree(ax, root, unit="bits", *, syn_tolerance=1e-12):
    """Draw local Syn / total Xi percentages without changing estimated values.

    Color intensity uses a fixed 0–100% display scale; signed raw percentages
    remain in the labels. Leaves retain their source names and have zero Syn.
    """
    shares = tree_share_data(root, syn_tolerance=syn_tolerance)
    positions = {}
    leaf_index = 0

    def locate(node):
        nonlocal leaf_index
        if not node.children:
            x = leaf_index
            leaf_index += 1
        else:
            x = np.mean([locate(child) for child in node.children])
        positions[node.sources] = (x, -node.depth)
        return x

    locate(root)
    nodes = flatten_nodes(root)
    max_depth = max(node.depth for node in nodes)
    share_cmap = LinearSegmentedColormap.from_list("spt_share", ["#F1F6F8", "#168B80", "#1F3B73"])
    for node in nodes:
        x, y = positions[node.sources]
        for child in node.children:
            cx, cy = positions[child.sources]
            ax.plot([x, cx], [y - 0.13, cy + 0.13], color="#A8B8C8", lw=1.5, zorder=1)
        label = "{" + ",".join(map(str, node.sources)) + "}"
        percentage = shares["percentages"][node.sources]
        facecolor, textcolor, edgecolor = "#ECF2F7", "#25364B", "#CCD5DF"
        if node.children:
            if percentage is None:
                label += "\nSyn: n/a"
                facecolor = "#F1F6F8"
            else:
                marker = "†" if node.syn_value < 0 else ""
                label += f"\nSyn: {percentage:.1f}%{marker}"
                facecolor = "#FCECE4" if percentage < 0 else share_cmap(percentage / 100)
                rgb = np.asarray(to_rgb(facecolor))
                linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = float(linear_rgb @ [0.2126, 0.7152, 0.0722])
                textcolor = "white" if luminance < 0.179 else "black"
            edgecolor = COLORS["xi"]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=textcolor,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 0.9,
            },
            zorder=2,
        )
    ax.set_xlim(-0.7, leaf_index - 0.3)
    ax.set_ylim(-max_depth - 0.5, 0.55)
    ax.axis("off")
    return dict(shares, unit=unit)


def finish(fig):
    """Use one layout engine; the notebook stores one PNG for each figure."""
    plt.show()
    plt.close(fig)


def show_boolean(xor_tpm, boolean_results):
    """Compare the XOR channel with COPY and noisy parity decompositions."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(11.8, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 1.5]}
        )
        channel = source_target_tpm(xor_tpm, 2, source_indices=[0, 1], target_indices=[0])
        im = plot_tpm(axes[0], channel, ["00", "01", "10", "11"], ["0", "1"])
        panel(axes[0], "a", "XOR: neither bit alone predicts the target")
        fig.colorbar(im, ax=axes[0], label="Transition probability", fraction=0.045, pad=0.04)
        plot_decomposition(axes[1], boolean_results)
        panel(axes[1], "b", "Joint information and its source contributions")
        finish(fig)


def show_horizons(horizons, lag_results):
    """Show forecast-horizon EI and its exact values."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.7, 3.0), constrained_layout=True)
        ax.plot(
            horizons,
            [lag_results[int(lag)].ei for lag in horizons],
            "o-",
            color=COLORS["ei"],
            label="Current bit 0 → future bit 0",
        )
        ax.set(xlabel="Forecast horizon (steps)", ylabel="EI (bits)", xticks=horizons, ylim=(-0.06, 1.08))
        outside_legend(ax)
        finish(fig)
    show_table(["Horizon", "EI (bits)", "Ξ (bits)"], [[lag, r.ei, r.xi] for lag, r in lag_results.items()])


def show_discrete_channels(copy_tpm, target_only, rectangular, macro_or, three_state_channel):
    """Display marginal and coarse-grained discrete channels."""
    show_table(
        ["Channel", "Shape", "EI (bits)", "Input intervention"],
        [
            [
                "Full COPY",
                copy_tpm.shape,
                effective_information_from_tpm(copy_tpm),
                "Uniform over 4 microstates",
            ],
            [
                "Single source → XOR",
                rectangular.shape,
                effective_information_from_tpm(rectangular),
                "Marginal of full intervention",
            ],
            [
                "OR macro COPY",
                macro_or.shape,
                effective_information_from_tpm(macro_or),
                "Uniform over 2 macro states",
            ],
            [
                "Three-state channel",
                three_state_channel.shape,
                f"{effective_information_from_tpm(three_state_channel):.4f}",
                "Uniform over 3 inputs",
            ],
        ],
    )
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(10.7, 3.5), constrained_layout=True)
        plot_tpm(axes[0], target_only, ["00", "01", "10", "11"], ["0", "1"])
        panel(axes[0], "a", "Both sources → XOR")
        plot_tpm(axes[1], rectangular, ["0", "1"], ["0", "1"])
        panel(axes[1], "b", "One source → XOR")
        im = plot_tpm(axes[2], macro_or, ["OR = 0", "OR = 1"], ["OR = 0", "OR = 1"])
        panel(axes[2], "c", "COPY after OR pooling")
        fig.colorbar(im, ax=axes[2], label="Transition probability", fraction=0.045, pad=0.04)
        finish(fig)


def show_continuous(sum_response, product_response, sum_result, product_result):
    """Display noise-free means next to estimated noisy-channel information."""
    grid_1d = np.linspace(-1, 1, 100)
    x0_grid, x1_grid = np.meshgrid(grid_1d, grid_1d)
    grid_inputs = np.column_stack([x0_grid.ravel(), x1_grid.ravel()])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(13.0, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.6]}
        )
        for ax, response, letter, description in zip(
            axes[:2],
            [sum_response, product_response],
            ["a", "b"],
            ["Additive mean response", "Product mean response"],
        ):
            image = ax.pcolormesh(
                x0_grid,
                x1_grid,
                response(grid_inputs).reshape(x0_grid.shape),
                cmap="RdBu_r",
                vmin=-2,
                vmax=2,
                shading="auto",
                rasterized=True,
            )
            ax.set(xlabel="$x_0$", ylabel="$x_1$", aspect="equal")
            panel(ax, letter, description)
        fig.colorbar(image, ax=axes[:2], label="Mean response (state units)", fraction=0.035, pad=0.025)
        plot_decomposition(axes[2], {"Additive": sum_result, "Product": product_result})
        panel(axes[2], "c", "Estimated information in noisy responses")
        finish(fig)


def show_maps(logistic_response, henon_response, henon_box, logistic_result, henon_result, seed):
    """Display logistic and Hénon responses under the declared interventions."""
    logistic_grid = np.linspace(0, 1, 300)
    henon_preview = henon_box.sample(1000, np.random.default_rng(seed))
    henon_future = henon_response(henon_preview)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(12.7, 3.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1, 1.5]}
        )
        axes[0].plot(logistic_grid, logistic_response(logistic_grid[:, None]), color=COLORS["ei"])
        axes[0].plot(logistic_grid, logistic_grid, "--", color=COLORS["single"])
        axes[0].set(xlabel="$x_t$", ylabel="$x_{t+1}$", xlim=(0, 1), ylim=(0, 1))
        panel(axes[0], "a", "Logistic map and identity reference")
        dots = axes[1].scatter(
            henon_future[:, 0],
            henon_future[:, 1],
            c=henon_preview[:, 0],
            cmap="viridis",
            s=7,
            alpha=0.75,
            rasterized=True,
        )
        axes[1].set(xlabel="First future coordinate", ylabel="Second future coordinate")
        panel(axes[1], "b", "Hénon: pushed-forward intervention")
        fig.colorbar(dots, ax=axes[1], label="Initial $x_0$", fraction=0.045, pad=0.04)
        plot_decomposition(axes[2], {"Logistic": logistic_result, "Hénon": henon_result})
        panel(axes[2], "c", "Different declared noisy channels")
        finish(fig)


def show_odes(oscillator_response, lorenz_vector_field, oscillator_result, lorenz_result):
    """Display illustrative ODE trajectories and forecast information."""
    oscillator_matrix = oscillator_response.matrix
    oscillator_times = np.linspace(0, 8, 400)
    oscillator_path = np.array([expm(oscillator_matrix * t) @ [1.0, 0.0] for t in oscillator_times])
    # This trajectory illustrates the ODE; it is not the intervention ensemble for EI.
    lorenz_trajectory = solve_ivp(
        lambda t, s: lorenz_vector_field(s[None, :])[0],
        (0, 30),
        [1.0, 1.0, 20.0],
        t_eval=np.linspace(0, 30, 4500),
        rtol=1e-8,
        atol=1e-10,
    )
    assert lorenz_trajectory.success
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 3, figsize=(12.8, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1, 1.5]}
        )
        axes[0].plot(oscillator_path[:, 0], oscillator_path[:, 1], color=COLORS["ei"])
        endpoint = oscillator_response(np.array([[1.0, 0.0]]))[0]
        axes[0].scatter(*endpoint, s=38, color=COLORS["accent"], zorder=3)
        axes[0].set(xlabel="Position", ylabel="Velocity")
        panel(axes[0], "a", "Damped oscillator")
        axes[1].plot(
            lorenz_trajectory.y[0, 500:], lorenz_trajectory.y[2, 500:], color=COLORS["xi"], lw=0.6, alpha=0.85
        )
        axes[1].set(xlabel="Lorenz $x$", ylabel="Lorenz $z$")
        panel(axes[1], "b", "Illustrative Lorenz trajectory")
        plot_decomposition(axes[2], {"Oscillator → position": oscillator_result, "Lorenz → y": lorenz_result})
        panel(axes[2], "c", "Declared forecast responses")
        finish(fig)


def show_model(training, learned_result, sum_result):
    """Show holdout validation and model intervention information."""
    show_table(
        ["Holdout samples", "RMSE", "Residual SD", "Coefficients: intercept, x0, x1"],
        [
            [
                len(training.residuals),
                f"{training.rmse:.4f}",
                f"{training.noise_std:.4f}",
                np.round(training.model.coefficients, 4),
            ]
        ],
    )
    show_results({"Fitted linear model": learned_result})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)
        axes[0].scatter(
            training.predictions, training.observed, color=COLORS["ei"], s=10, alpha=0.3, rasterized=True
        )
        axes[0].plot([-2.5, 2.5], [-2.5, 2.5], "--", color=COLORS["accent"])
        axes[0].set(xlabel="Predicted response", ylabel="Observed holdout response", aspect="equal")
        panel(axes[0], "a", "Untouched holdout samples")
        plot_decomposition(axes[1], {"Known sum": sum_result, "Fitted model": learned_result})
        panel(axes[1], "b", "Model intervention analysis")
        finish(fig)


def show_hierarchy(pair_result, table_tree, affinity):
    """Show pairwise Syn affinity and the actual selected SPT."""
    show_results({"Two XOR modules": pair_result})
    show_table(
        ["SPT contribution", "Order", "Syn (bits)"],
        [[atom.sources, len(atom.sources), atom.value] for atom in flatten_atoms(table_tree.root)],
    )
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, 2, figsize=(11.5, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.8]}
        )
        im = axes[0].imshow(affinity, cmap="Blues", vmin=0, vmax=1)
        axes[0].set(xticks=range(4), yticks=range(4), xlabel="Source", ylabel="Source")
        for i, j in np.ndindex(affinity.shape):
            axes[0].text(
                j,
                i,
                f"{affinity[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if affinity[i, j] > 0.5 else "#25364B",
            )
        panel(axes[0], "a", "Pairwise Syn affinity")
        fig.colorbar(im, ax=axes[0], label="Syn (bits)", fraction=0.045, pad=0.04)
        plot_tree(
            axes[1], table_tree.root, unit="bits", syn_tolerance=pair_result.diagnostics["syn_tolerance"]
        )
        panel(axes[1], "b", "Node Syn as a share of total Ξ")
        finish(fig)


def show_searches(exact_six, spectral_six, manual_spectral, random_tree):
    """Compare candidate counts, table completeness, and tree closure."""
    show_table(
        ["Search", "Evaluated splits", "EI queries", "EI table complete?", "Closure (bits)"],
        [
            [
                "Exact at each node",
                exact_six.tree.audit.candidate_count,
                exact_six.diagnostics["ei_query_count"],
                exact_six.diagnostics["ei_table_complete"],
                exact_six.tree.closure_error,
            ],
            [
                "Public spectral",
                spectral_six.tree.audit.candidate_count,
                spectral_six.diagnostics["ei_query_count"],
                spectral_six.diagnostics["ei_table_complete"],
                spectral_six.tree.closure_error,
            ],
            [
                "Manual spectral",
                manual_spectral.audit.candidate_count,
                "Reused table",
                "Supplied complete table",
                manual_spectral.closure_error,
            ],
            [
                "Random + local refinement",
                random_tree.audit.candidate_count,
                "Reused table",
                "Supplied complete table",
                random_tree.closure_error,
            ],
        ],
    )


def show_prior(table_tree, prior_tree, *, syn_tolerance=1e-12):
    """Compare standard and imposed partition paths."""
    show_table(
        ["Partition path", "Root split", "Root Syn (bits)", "Sum of node Syn (bits)", "Closure"],
        [
            [
                "Standard exact",
                [c.sources for c in table_tree.root.children],
                table_tree.root.syn_value,
                sum(n.syn_value for n in flatten_nodes(table_tree.root)),
                table_tree.closure_error,
            ],
            [
                "Imposed prior",
                [c.sources for c in prior_tree.root.children],
                prior_tree.root.syn_value,
                sum(n.syn_value for n in flatten_nodes(prior_tree.root)),
                prior_tree.closure_error,
            ],
        ],
    )
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
        plot_tree(ax, prior_tree.root, unit="bits", syn_tolerance=syn_tolerance)
        ax.set_title("Node Syn as a share of total Ξ", loc="left")
        finish(fig)


def show_densities(data, banana_models, gaussian_log_rmse, density_errors):
    """Display held-out density errors and the mapped Gaussian reference."""
    banana_test = data.banana_test
    show_table(
        ["Held-out density model", "Log-density RMSE (natural log units)"],
        [["Affine on Gaussian", f"{gaussian_log_rmse:.4f}"]]
        + [[name + " on banana", f"{value:.4f}"] for name, value in density_errors.items()],
    )
    reference_points = banana_models["Polynomial, degree 2"].map_to_reference(banana_test)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4), constrained_layout=True)
        axes[0].scatter(
            banana_test[:, 0], banana_test[:, 1], s=7, color=COLORS["ei"], alpha=0.25, rasterized=True
        )
        axes[0].set(
            xlabel="Observed coordinate 0", ylabel="Observed coordinate 1", xlim=(-3.5, 3.5), ylim=(-4, 7)
        )
        panel(axes[0], "a", "Curved banana density")
        axes[1].scatter(
            reference_points[:, 0],
            reference_points[:, 1],
            s=7,
            color=COLORS["xi"],
            alpha=0.25,
            rasterized=True,
        )
        axes[1].set(
            xlabel="Reference $z_0$",
            ylabel="Reference $z_1$",
            aspect="equal",
            xlim=(-3.5, 3.5),
            ylim=(-3.5, 3.5),
        )
        panel(axes[1], "b", "Polynomial map → Gaussian reference")
        axes[2].barh(
            list(density_errors),
            list(density_errors.values()),
            color=[COLORS["single"], COLORS["ei"], COLORS["xi"]],
        )
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Held-out log-density RMSE")
        panel(axes[2], "c", "Known-density validation")
        finish(fig)


def show_conditionals(gaussian_test, gaussian_marginal, conditional_draws, prefixes):
    """Display a Gaussian marginal and fixed-prefix conditional draws."""
    assert conditional_draws.ndim == 3
    assert conditional_draws.shape[0] == len(prefixes) and conditional_draws.shape[2] == 2
    query_x = np.linspace(-3, 4, 250)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3), constrained_layout=True)
        axes[0].hist(gaussian_test[:, 0], bins=35, density=True, color=COLORS["light"], edgecolor="white")
        axes[0].plot(
            query_x,
            gaussian_marginal.pdf(query_x[:, None]),
            color=COLORS["ei"],
            label="Fitted affine marginal",
        )
        axes[0].set(xlabel="Coordinate 0", ylabel="Probability density")
        panel(axes[0], "a", "Marginal of the Gaussian example")
        outside_legend(axes[0])
        for draws, color, label in zip(
            conditional_draws,
            [COLORS["ei"], COLORS["xi"], COLORS["accent"]],
            [f"Prefix {p[0]:g}".replace("-", "−") for p in prefixes],
        ):
            axes[1].hist(
                draws[:, 1], bins=25, density=True, histtype="step", color=color, lw=1.7, label=label
            )
        axes[1].set(xlabel="Banana coordinate 1", ylabel="Conditional density")
        panel(axes[1], "b", "Conditional draws from the polynomial TM")
        outside_legend(axes[1])
        finish(fig)


def show_mi(mi, mi_uniform_weight, weighted_mi, analytic_mi_bits, anchor_values, specific):
    """Display global, pointwise, and specific MI with their conventions."""
    show_table(
        ["Distribution", "MI estimate (bits)", "Effective sample size"],
        [
            ["Gaussian analytic reference", f"{analytic_mi_bits:.4f}", "n/a"],
            ["Unweighted sample", f"{mi['mi_hat']:.4f}", f"{mi['effective_sample_size']:.0f}"],
            [
                "Uniform weights",
                f"{mi_uniform_weight['mi_hat']:.4f}",
                f"{mi_uniform_weight['effective_sample_size']:.0f}",
            ],
            [
                "Nonuniform reweighting",
                f"{weighted_mi['mi_hat']:.4f}",
                f"{weighted_mi['effective_sample_size']:.0f}",
            ],
        ],
    )
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), constrained_layout=True)
        axes[0].hist(mi["pointwise_mi"], bins=55, color=COLORS["ei"], alpha=0.8)
        axes[0].axvline(0, color=COLORS["single"], lw=0.8)
        axes[0].set(xlabel="Pointwise MI (bits)", ylabel="Observation count")
        panel(axes[0], "a", "Raw log density ratios")
        axes[1].plot(anchor_values[:, 0], specific["specific_mi"], "o-", ms=3.5, color=COLORS["xi"])
        axes[1].set(xlabel="Fixed target anchor", ylabel="Specific MI (bits)")
        panel(axes[1], "b", "Conditional information by target state")
        finish(fig)


def show_pairwise(pairwise):
    """Display raw pairwise EI estimates without clipping negative values."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
        values = pairwise["pairwise_ei"]
        # Include raw numerical negatives in the color scale, if present.
        im = ax.imshow(
            values, cmap="Blues", vmin=min(0, float(values.min())), vmax=float(values.max()), aspect="auto"
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Output 0", "Output 1"],
            yticks=[0, 1, 2],
            yticklabels=["Input 0", "Input 1", "Input 2"],
        )
        for i, j in np.ndindex(values.shape):
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if values[i, j] > 0.6 * values.max() else "#25364B",
            )
        fig.colorbar(im, ax=ax, label="Pairwise EI estimate (bits)", fraction=0.05, pad=0.04)
        finish(fig)
    show_table(
        ["Input", "Output 0 EI (bits)", "Output 1 EI (bits)"],
        [[i, *[f"{v:.4f}" for v in values[i]]] for i in range(3)],
    )


def show_export(exported):
    """Display exported metadata, numerical diagnostics, and units."""
    show_table(
        ["Result field", "Example value"],
        [
            ["unit", exported["unit"]],
            ["sources → targets", f"{exported['sources']} → {exported['targets']}"],
            ["estimator", exported["metadata"]["estimator"]],
            ["response noise SD", exported["metadata"]["response_noise_std"]],
            [
                "forecast interval / integrator",
                f"{exported['metadata']['forecast_interval']} / {exported['metadata']['integrator']}",
            ],
            ["density fit/evaluation", exported["metadata"]["density_fit_and_evaluation"]],
            ["EI table complete", exported["diagnostics"]["ei_table_complete"]],
            ["minimum queried Ξ", f"{exported['diagnostics']['minimum_queried_xi']:.6f}"],
            ["SPT closure", exported["tree"]["closure_error"]],
        ],
    )


def show_tolerance_audit(audit):
    """Display the raw tolerance-scale residual and expected significant-violation failure."""
    tiny_tree = audit.tree
    audit_tolerance_bits = audit.tolerance
    show_table(
        ["Unit", "Tolerance", "Raw Syn", "Affected count", "Closure"],
        [
            [
                "bits",
                audit_tolerance_bits,
                f"{tiny_tree.root.syn_value:.3f}",
                tiny_tree.audit.tolerance_zero_count,
                tiny_tree.closure_error,
            ]
        ],
    )
    print("Expected explicit failure:", audit.expected_error)


def show_legacy(legacy):
    """Display historical compatibility and nonstandard objective/stopping examples."""
    legacy_root = legacy.legacy_root
    signed_diagnostic = legacy.signed_diagnostic
    normalized_tree = legacy.normalized_tree
    terminal_tree = legacy.terminal_tree
    show_table(
        ["Legacy option", "Observed behavior", "Interpretation"],
        [
            [
                "Phi compatibility",
                f"Total atoms = {sum(a.value for a in flatten_phi_tree(legacy_root)):.1f} bits",
                "Same canonical tree",
            ],
            [
                "SIGNED",
                f"Raw residual = {signed_diagnostic.root.syn_value:.3f} bits",
                "Invalid Syn retained for diagnostics only",
            ],
            [
                "Cross-coalition normalization",
                f"Closure = {normalized_tree.closure_error:.1e} bits",
                "Different objective, not standard SPT",
            ],
            ["Early terminal", terminal_tree.root.split_kind, "Not the standard singleton stopping rule"],
        ],
    )


def show_noise(noise_levels, noise_results):
    """Display raw information estimates across the declared observation-noise levels."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.7, 3.2), constrained_layout=True)
        ax.plot(
            noise_levels, [r.ei for r in noise_results.values()], "o-", color=COLORS["ei"], label="Joint EI"
        )
        ax.plot(
            noise_levels,
            [r.xi for r in noise_results.values()],
            "s--",
            color=COLORS["xi"],
            label="Integrated EI  Ξ",
        )
        ax.set(
            xlabel="Measurement noise SD (response units)",
            ylabel="Estimated information (nats)",
            ylim=(0, None),
        )
        outside_legend(ax)
        finish(fig)


def show_spt_gallery(named_results, *, columns=2, path=None):
    """Display diverse actual trees with the same percentage convention."""
    if not named_results:
        raise ValueError("The SPT gallery must contain at least one result.")
    show_results(named_results)
    rows = (len(named_results) + columns - 1) // columns
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(7.2 * columns, 4.4 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        notes = []
        for index, ((name, result), ax) in enumerate(zip(named_results.items(), axes.flat)):
            shares = plot_tree(
                ax,
                result.tree.root,
                unit=result.unit,
                syn_tolerance=result.diagnostics["syn_tolerance"],
            )
            panel(ax, chr(ord("a") + index), name)
            if shares["selected_tolerance_negative_count"]:
                notes.append(
                    f"{name}: {shares['selected_tolerance_negative_count']} tolerance-scale negative nodes"
                )
            if result.xi <= result.diagnostics["syn_tolerance"]:
                notes.append(f"{name}: percentages undefined at the declared tolerance")
        for ax in list(axes.flat)[len(named_results) :]:
            ax.remove()
        caption = "Node labels = 100 × local Syn / total Ξ of that tree; darker nodes carry a larger share."
        if notes:
            caption += "\n† Raw negative estimates are numerical diagnostics. " + "; ".join(notes)
        fig.supxlabel(caption, fontsize=10)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finish(fig)
