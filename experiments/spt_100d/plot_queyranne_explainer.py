"""Plot existing Queyranne evidence; no simulation, fitting, or timing reruns.

Inputs are the original 2026-10-31 JSONL records. Timings are individual
observations, not means. Root panels use logarithmic y axes, no smoothing,
and exclude data generation/model fitting. Six-source points use three stored
replicates per original reader, with deterministic vertical offsets only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "docs/log/spt_100d_stress/results"
FIG = ROOT / "fig"
BLUE, ORANGE, GRAY = "#246A9A", "#C45A2C", "#727B86"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, color="#89929C")
    ax.grid(axis="y", color="#E5E9EC", linewidth=0.65)
    ax.set_axisbelow(True)


def root_evidence():
    rows = read_rows(LOGS / "q_vs_exact.jsonl")
    if len(rows) != 8 or any(r["status"] != "complete" for r in rows):
        raise ValueError("Expected eight completed paired root comparisons.")
    if any(r["gap_nats"] != 0 or r["q_near_negative"] or r["exact_near_negative"] for r in rows):
        raise ValueError("Root evidence changed: review optimality/nonnegativity captions.")
    n = np.array([r["n"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.5), layout="constrained")
    for ax, qkey, ekey, label in zip(
        axes,
        ["q_seconds", "q_unique_ei_queries"],
        ["exact_seconds", "exact_unique_ei_queries"],
        ["根节点搜索时间（秒，对数刻度）", "实际计算的不同非空 EI 子集数（对数刻度）"],
    ):
        clean(ax)
        ax.plot(n, [r[ekey] for r in rows], "s-", color=ORANGE, ms=4.5, lw=1.6, label="穷举")
        ax.plot(n, [r[qkey] for r in rows], "o-", color=BLUE, ms=4.5, lw=1.6, label="Queyranne")
        ax.set_yscale("log")
        ax.set_xticks(n)
        ax.set_xlim(5.3, 23.4)
        ax.set_xlabel("当前节点的源变量数")
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylim(0.0004, 60)
    axes[1].set_ylim(25, 3_000_000)
    axes[0].set_title("a  同一信息模型下，搜索耗时明显分开", loc="left", fontsize=11, pad=12)
    axes[1].set_title("b  加速来自更少的 EI 计算", loc="left", fontsize=11, pad=12)
    for ax, q, e in [(axes[0], "0.0376 秒", "20.23 秒"), (axes[1], "1,689 次", "1,048,575 次")]:
        i = 0 if ax is axes[0] else 1
        qkey = ["q_seconds", "q_unique_ei_queries"][i]
        ekey = ["exact_seconds", "exact_unique_ei_queries"][i]
        for key, label, color in [(qkey, q, BLUE), (ekey, e, ORANGE)]:
            ax.annotate(label, (20, rows[-1][key]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=color, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(FIG / "queyranne_spt_speed.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def boundaries():
    rows = read_rows(ROOT / "experiments/spt_copula_queyranne/results/results.jsonl")
    readers = ["original_component_neural_TM", "exact_discrete_fitted_MLP", "original_shared_channel_density"]
    groups = [[r for r in rows if r.get("estimator") == reader] for reader in readers]
    if any(len(g) != 3 for g in groups):
        raise ValueError("Expected three stored repeats for each original reader.")
    dev = next(r for r in read_rows(LOGS / "results.jsonl")
               if r.get("event") == "case" and r["n"] == 100 and r["seed"] == 101)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.7, 4.5), layout="constrained",
                              gridspec_kw={"width_ratios": [1.12, 1]})
    for ax in (a, b):
        clean(ax)
        ax.grid(False)
        ax.grid(axis="x", color="#E5E9EC", linewidth=0.65)
    for i, group in enumerate(groups):
        for j, row in enumerate(sorted(group, key=lambda r: r["seed"])):
            y = 2 - i + (j - 1) * 0.14
            e, q = (1000 * row[k]["search"]["seconds"] for k in ("exact", "queyranne"))
            a.plot([e, q], [y, y], color="#C7CDD3", lw=0.9)
            a.scatter(e, y, marker="s", s=28, color=ORANGE, zorder=3,
                      label="穷举" if i == j == 0 else None)
            a.scatter(q, y, marker="o", s=28, color=BLUE, zorder=3,
                      label="Queyranne" if i == j == 0 else None)
    a.set_yticks([2, 1, 0], ["I  神经 TM", "II  离散精确 EI", "III  共享通道密度"])
    a.set_ylim(-0.4, 2.4)
    a.set_xlim(0, 0.23)
    a.set_xlabel("纯搜索时间（毫秒；EI 表已预计算）")
    a.set_title("a  六源小问题：查询减少，耗时仍可能增加", loc="left", fontsize=11, pad=12)
    qtime = dev["queyranne"]["search"]["seconds"]
    stime = dev["spectral"]["seconds_including_pairwise_affinity"]
    b.barh([1, 0], [qtime, stime], height=0.42, color=[BLUE, GRAY])
    b.set_yticks([1, 0], ["Queyranne", "谱候选"])
    b.set_ylim(-0.65, 1.65)
    b.set_xlim(0, 820)
    b.set_xlabel("完整树搜索时间（秒）")
    b.set_title("b  100 源开发条件：完整树仍有较大成本", loc="left", fontsize=11, pad=12)
    for y, val in [(1, qtime), (0, stime)]:
        b.annotate(f"{val:.1f} 秒", (val, y), xytext=(6, 0), textcoords="offset points", va="center")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(FIG / "queyranne_spt_limits.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    with plt.rc_context({"font.family": "Arial Unicode MS", "font.size": 10,
                         "axes.unicode_minus": False, "axes.labelcolor": "#25313C",
                         "text.color": "#25313C", "axes.edgecolor": "#89929C"}):
        root_evidence()
        boundaries()
    print("Saved fig/queyranne_spt_speed.png and fig/queyranne_spt_limits.png")
