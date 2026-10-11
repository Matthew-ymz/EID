"""Visual explanations of the estimator comparisons; no estimation logic here."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from estimator_comparisons import LABELS
from IPython.display import display
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
from plotting import COLORS, STYLE, panel

METHOD_COLORS = {
    "tm_polynomial": "#1F3B73",
    "tm_affine": "#6B7C93",
    "tm_affine_lifted": "#9B6543",
    "tm_affine_readout": "#5E557D",
    "gaussian": "#777777",
    "kde": "#B58024",
    "kde_heldout": "#B58024",
    "ksg": "#168B80",
    "tm_neural": "#A23E48",
    "neural_fourier": "#9C437A",
    "nis": "#B58024",
    "local_gaussian": "#5E557D",
    "linear_gaussian": "#25364B",
}
METHOD_MARKERS = {
    "tm_polynomial": "o",
    "tm_affine": "s",
    "tm_affine_lifted": "D",
    "tm_affine_readout": "P",
    "gaussian": "v",
    "kde": "^",
    "kde_heldout": "v",
    "ksg": "D",
    "tm_neural": "X",
    "neural_fourier": "P",
    "nis": "^",
    "local_gaussian": "s",
    "linear_gaussian": "*",
}


def _save(fig, directory, name):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    display(fig)
    plt.close(fig)
    return path


def _values(case, method, field="ei"):
    return np.asarray([r[field] for r in case["records"] if r["method"] == method], dtype=float)


def _method_axis(ax, methods):
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=9)
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def channels(report, directory):
    """Response geometry and mean +/- SD EI, with all individual seed values."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3,
            2,
            figsize=(12.8, 10.7),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 1.35]},
        )
        descriptions = {
            "gaussian": ("Linear Gaussian: Y = X + ε", "Gaussian input X"),
            "product": ("Bilinear: Y = X₁X₂ + ε", "Product coordinate X₁X₂"),
            "periodic": ("Periodic: Y = sin(3X) + ε", "Uniform phase input X"),
        }
        all_values = [r["ei"] for c in report["cases"].values() for r in c["records"]]
        xlim = (min(-0.025, min(all_values) - 0.04), max(all_values) + 0.12)
        for index, (name, case) in enumerate(report["cases"].items()):
            left, right = axes[index]
            data = case["scatter"]
            x, y = np.asarray(data["coordinate"]), np.asarray(data["y"])
            left.scatter(x[:300], y[:300], s=9, color=COLORS["single"], alpha=0.38, linewidth=0)
            grid = np.linspace(x.min(), x.max(), 400)
            noiseless = np.sin(3 * grid) if name == "periodic" else grid
            left.plot(grid, noiseless, color=COLORS["ei"], lw=1.8)
            left.set(xlabel=descriptions[name][1], ylabel="Target Y")
            panel(left, chr(97 + 2 * index), descriptions[name][0])
            _method_axis(right, case["methods"])
            for position, method in enumerate(case["methods"]):
                values = _values(case, method)
                jitter = np.linspace(-0.13, 0.13, len(values))
                color = METHOD_COLORS[method]
                right.scatter(values, position + jitter, s=14, color=color, alpha=0.4, linewidth=0)
                right.errorbar(
                    values.mean(),
                    position,
                    xerr=values.std(ddof=1),
                    fmt=METHOD_MARKERS[method],
                    color=color,
                    markersize=6,
                    capsize=2,
                    elinewidth=1.3,
                )
            right.axvline(case["reference"]["ei"], color="#25364B", ls="--", lw=1.1)
            right.axvline(0, color="#D9DFE7", lw=0.8)
            right.set(xlabel="Estimated EI (nats)", xlim=xlim)
            panel(right, chr(98 + 2 * index), "Mean ± SD; small dots are paired seeds")
        fig.legend(
            [Line2D([0], [0], color=COLORS["ei"]), Line2D([0], [0], color="#25364B", ls="--")],
            ["Noiseless response", "Analytic / channel-integral EI reference"],
            loc="outside upper center",
            ncol=2,
        )
        return _save(fig, directory, "ei_estimator_channels.png")


def synergy_and_cost(report, directory):
    """Differentiate joint accuracy, residual accuracy, and computational cost."""
    case = report["cases"]["product"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.8, 5.7),
            sharey=True,
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.4, 1]},
        )
        _method_axis(axes[0], case["methods"])
        for position, method in enumerate(case["methods"]):
            for offset, field, color in [(-0.14, "ei", COLORS["ei"]), (0.14, "syn", COLORS["xi"])]:
                values = _values(case, method, field)
                axes[0].errorbar(
                    values.mean(),
                    position + offset,
                    xerr=values.std(ddof=1),
                    fmt="o",
                    color=color,
                    markersize=5,
                    capsize=2,
                    elinewidth=1.3,
                )
                axes[0].scatter(
                    values,
                    position + offset + np.linspace(-0.045, 0.045, len(values)),
                    color=color,
                    s=12,
                    alpha=0.32,
                    linewidth=0,
                )
            times = np.asarray([r["seconds"] for r in case["records"] if r["method"] == method])
            median = np.median(times)
            axes[1].errorbar(
                median,
                position,
                xerr=[[median - times.min()], [times.max() - median]],
                fmt=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                capsize=2,
                markersize=6,
            )
        axes[0].axvline(case["reference"]["ei"], color=COLORS["ei"], ls="--", lw=1.2)
        axes[0].axvline(case["reference"]["syn"], color=COLORS["xi"], ls=":", lw=1.5)
        axes[0].axvline(0, color="#D9DFE7", lw=0.8)
        axes[0].set_xlabel("Information (nats); mean ± SD")
        axes[1].set(xscale="log", xlabel="Full analysis call (s)")
        axes[1].tick_params(axis="y", left=False, labelleft=False)
        panel(axes[0], "a", "Good joint EI need not imply good Syn")
        panel(axes[1], "b", "Wall time: median and full seed range")
        fig.legend(
            [
                Line2D([0], [0], color=COLORS["ei"], marker="o", ls="none"),
                Line2D([0], [0], color=COLORS["xi"], marker="o", ls="none"),
                Line2D([0], [0], color=COLORS["ei"], ls="--"),
                Line2D([0], [0], color=COLORS["xi"], ls=":"),
            ],
            ["Joint EI estimate", "Syn estimate", "Joint EI reference", "Syn reference"],
            loc="outside upper center",
            ncol=4,
        )
        return _save(fig, directory, "ei_estimator_syn_cost.png")


def _replicate_curve(ax, records, key, method, *, reference, per_dimension=False):
    levels = sorted({r[key] for r in records})
    means = []
    for value in levels:
        estimates = np.asarray([r["ei"] for r in records if r[key] == value and r["method"] == method])
        error = abs(estimates / value - reference) if per_dimension else abs(estimates - reference)
        means.append(error.mean())
        ax.scatter(np.full(len(error), value), error, color=METHOD_COLORS[method], s=15, alpha=0.32)
    (line,) = ax.plot(
        levels,
        means,
        marker=METHOD_MARKERS[method],
        color=METHOD_COLORS[method],
        markersize=6,
        ls="--" if method == "kde_heldout" else "-",
        label=LABELS[method],
    )
    return line


def sample_and_dimension(report, directory):
    data = report["sensitivity"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
        handles = []
        for method in ("tm_polynomial", "tm_affine", "kde", "kde_heldout", "ksg"):
            handles.append(
                _replicate_curve(
                    axes[0],
                    data["sizes"],
                    "n",
                    method,
                    reference=data["reference_per_coordinate"],
                )
            )
        for method in ("tm_affine", "kde", "kde_heldout", "ksg"):
            _replicate_curve(
                axes[1],
                data["dimensions"],
                "dimension",
                method,
                reference=data["reference_per_coordinate"],
                per_dimension=True,
            )
        axes[0].set(
            xscale="log",
            xlabel="Total intervention pairs N",
            ylabel="Absolute EI error (nats)",
            ylim=(0, None),
        )
        axes[0].set_xticks(report["identity"]["config"]["sample_sizes"])
        axes[0].set_xticklabels(report["identity"]["config"]["sample_sizes"])
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[1].set(
            xlabel="Source and target dimension d",
            ylabel="Absolute EI error / d (nats per coordinate)",
            ylim=(0, None),
        )
        axes[1].set_xticks(report["identity"]["config"]["dimensions"])
        panel(axes[0], "a", "Sample budget; 1D Gaussian channel")
        panel(axes[1], "b", "Dimension; independent Gaussian channels")
        fig.legend(handles, [h.get_label() for h in handles], loc="outside upper center", ncol=3)
        return _save(fig, directory, "ei_estimator_sample_dimension.png")


def approximation_limits(report, directory):
    data = report["approximations"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)
        handles = {}
        for ax, field, reference_field, extra in [
            (axes[0], "noise", "noise_references", "nis"),
            (axes[1], "local", "local_references", "local_gaussian"),
        ]:
            ref = data[reference_field]
            (handles["Channel reference"],) = ax.plot(
                [r["scale"] for r in ref],
                [r["ei"] for r in ref],
                color="#25364B",
                ls="--",
                lw=1.5,
            )
            for method in ("tm_polynomial", "ksg", extra):
                levels = sorted({r["scale"] for r in data[field]})
                values = []
                for scale in levels:
                    samples = [r["ei"] for r in data[field] if r["scale"] == scale and r["method"] == method]
                    values.append(np.mean(samples))
                    ax.scatter([scale] * len(samples), samples, s=14, color=METHOD_COLORS[method], alpha=0.32)
                label = LABELS.get(method, "NIS approximation" if method == "nis" else "Local Gaussian")
                (handles[label],) = ax.plot(
                    levels, values, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], ms=6
                )
            ax.axhline(0, color="#BAC4D0", lw=0.8)
            ax.set(xscale="log", ylabel="Raw EI / approximation (nats)")
        axes[0].set_xlabel("Response noise SD σ; Y = 2X + ε")
        axes[1].set_xlabel("Input SD s; Y = sin(3X) + ε")
        panel(axes[0], "a", "NIS: bounded uniform input X ∈ [−1, 1]")
        panel(axes[1], "b", "Local Gaussian: X ∼ N(0, s²), noise SD 0.3")
        fig.legend(list(handles.values()), list(handles), loc="outside upper center", ncol=3)
        return _save(fig, directory, "ei_estimator_approximation_limits.png")


def applicability(report, directory):
    """Discrete example plus a categorical map of actual API contracts."""
    methods = [
        "discrete",
        "tm_polynomial",
        "tm_affine",
        "tm_affine_lifted",
        "tm_affine_readout",
        "tm_neural",
        "gaussian",
        "kde",
        "ksg",
        "linear_gaussian",
        "local_gaussian",
        "nis",
    ]
    texts = [
        ["Exact", "—", "—", "Binary", "—"],
        ["—", "Fit", "—", "Yes", "Yes"],
        ["—", "Fit", "—", "Yes", "Yes"],
        ["—", "1–2D", "—", "1–2D", "No bias"],
        ["—", "Readout", "—", "Yes", "—"],
        ["—", "Model", "—", "Yes", "—"],
        ["—", "Fit", "—", "Yes", "—"],
        ["—", "Fit", "—", "Yes", "Yes"],
        ["—", "Fit", "—", "Yes", "—"],
        ["—", "—", "Gaussian", "Yes", "—"],
        ["—", "—", "Jacobian", "Yes", "—"],
        ["—", "—", "Square", "—", "—"],
    ]
    restricted = {"Binary", "1–2D", "No bias", "Readout", "Model", "Gaussian", "Jacobian", "Square"}
    codes = np.array([[0 if t == "—" else 2 if t in restricted else 1 for t in row] for row in texts])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.8, 5.8),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1, 1.7]},
        )
        p = np.linspace(0, 0.5, 501)
        from scipy.special import xlogy

        truth = np.log(2) + xlogy(p, p) + xlogy(1 - p, 1 - p)
        axes[0].plot(p, truth, ls="--", color="#25364B", label="Analytic reference")
        rows = report["discrete"]
        axes[0].scatter(
            [r["flip_probability"] for r in rows],
            [r["ei"] for r in rows],
            color=COLORS["ei"],
            marker="o",
            s=34,
            label="Exact TPM EI = Syn",
            zorder=3,
        )
        axes[0].set(
            xlabel="Independent XOR output-flip probability",
            ylabel="EI = Syn (nats)",
            xlim=(0, 0.5),
            ylim=(0, 0.73),
        )
        panel(axes[0], "a", "Discrete XOR; known four-row TPM")
        axes[1].imshow(
            codes, cmap=ListedColormap(["#F1F3F6", "#DFEEE9", "#F2E6D8"]), vmin=0, vmax=2, aspect="auto"
        )
        for i, row in enumerate(texts):
            for j, text in enumerate(row):
                axes[1].text(j, i, text, ha="center", va="center", fontsize=8.7, color="#25364B")
        axes[1].set_yticks(range(len(methods)), methods, fontsize=9)
        axes[1].set_xticks(
            range(5),
            [
                "Finite\nTPM",
                "Paired\nsamples",
                "Declared\nchannel",
                "Subset EI\n/ SPT",
                "Held-out\nlog ratio",
            ],
            fontsize=9,
        )
        axes[1].set_xticks(np.arange(-0.5, 5), minor=True)
        axes[1].set_yticks(np.arange(-0.5, len(methods)), minor=True)
        axes[1].grid(which="minor", color="white", lw=1.8)
        axes[1].tick_params(which="both", length=0)
        for spine in axes[1].spines.values():
            spine.set_visible(False)
        panel(axes[1], "b", "Interface support; amber cells have conditions")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2)
        return _save(fig, directory, "ei_estimator_applicability.png")
