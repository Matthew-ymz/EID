"""Small exact-discrete feasibility study; not a public SPT implementation.

Run from repository root: PYTHONPATH=src .venv/bin/python experiments/spt_agglomerative/pilot.py
All sources are independent uniform bits. The same one-step channel and target
are used for every query. Full enumeration isolates search from estimation error.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field

import numpy as np

from eid.spt import SPTConfig, build_spt

TOL_BITS = 1e-10


def entropy(p: np.ndarray) -> np.ndarray:
    logp = np.zeros_like(p)
    np.log2(p, out=logp, where=p > 0)
    return -np.sum(p * logp, axis=-1)


def members(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(mask.bit_length()) if mask & (1 << i))


@dataclass
class ExactChannel:
    probabilities: np.ndarray
    cache: dict[int, float] = field(default_factory=lambda: {0: 0.0})
    audited: dict[tuple[int, ...], float] = field(default_factory=dict)

    def __post_init__(self):
        p = np.asarray(self.probabilities, dtype=float)
        self.n = int(math.log2(len(p)))
        if 2**self.n != len(p) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("Expected a channel over the full binary source support.")
        self.probabilities = p
        self.full = len(p) - 1
        self.hy = float(entropy(p.mean(axis=0)))

    def ei(self, mask: int) -> float:
        if mask not in self.cache:
            indices = members(mask)
            ids = np.arange(len(self.probabilities))
            labels = sum(((ids >> bit) & 1) << j for j, bit in enumerate(indices))
            marginal = np.zeros((2 ** len(indices), self.probabilities.shape[1]))
            np.add.at(marginal, labels, self.probabilities)
            marginal /= 2 ** (self.n - len(indices))
            self.cache[mask] = self.hy - float(entropy(marginal).mean())
        return self.cache[mask]

    def xi(self, sources) -> float:
        ids = tuple(sources)
        return self.ei(sum(1 << i for i in ids)) - sum(self.ei(1 << i) for i in ids)

    def gain(self, groups: tuple[int, ...]) -> float:
        key = tuple(sorted(groups))
        if key not in self.audited:
            union = 0
            for group in groups:
                if union & group:
                    raise ValueError("Merge groups must be disjoint.")
                union |= group
            raw = self.ei(union) - sum(self.ei(group) for group in groups)
            if not math.isfinite(raw) or raw < -TOL_BITS:
                raise RuntimeError(
                    f"Syn violation: minimum={raw}, threshold={-TOL_BITS}, affected_count=1"
                )
            self.audited[key] = raw  # Never clip raw values.
        return self.audited[key]

    def diagnostics(self):
        values = list(self.audited.values())
        return {
            "unit": "bits", "syn_tolerance": TOL_BITS,
            "minimum_candidate_syn": min(values) if values else None,
            "tolerance_negative_count": sum(-TOL_BITS <= v < 0 for v in values),
            "significant_negative_count": sum(v < -TOL_BITS for v in values),
            "unique_ei_queries": len(self.cache) - 1,
        }


def agglomerate(channel: ExactChannel, max_arity: int) -> dict:
    """Merge by raw gain/(arity-1); no compression and no forced zero merges.

    This is a proposed heuristic, not the existing top-down SPT criterion.
    The minimal pilot rescans candidates, while gain/MI values remain cached.
    """
    groups = tuple(1 << i for i in range(channel.n))
    merges = []
    while len(groups) > 1:
        candidates = []
        for arity in range(2, min(max_arity, len(groups)) + 1):
            for parts in itertools.combinations(groups, arity):
                gain = channel.gain(parts)
                candidates.append((gain / (arity - 1), gain, parts))
        score, gain, parts = min(candidates, key=lambda c: (-c[0], len(c[2]), c[2]))
        if gain <= TOL_BITS:
            break
        union = sum(parts)
        merges.append({"groups": [members(g) for g in parts], "union": members(union),
                       "gain_bits": gain, "ranking_score": score})
        groups = tuple(sorted([g for g in groups if g not in parts] + [union]))
    total = channel.xi(range(channel.n))
    resolved = sum(m["gain_bits"] for m in merges)
    remaining = channel.ei(channel.full) - sum(channel.ei(g) for g in groups)
    channel.gain(groups) if len(groups) > 1 else None
    if abs(total - resolved - remaining) > TOL_BITS:
        raise AssertionError("Partial hierarchy failed closure with unresolved mass.")
    return {"total_xi_bits": total, "resolved_bits": resolved,
            "unresolved_bits": remaining, "groups": [members(g) for g in groups],
            "merges": merges, "closure_error_bits": resolved + remaining - total,
            "diagnostics": channel.diagnostics()}


def queyranne_split(channel: ExactChannel, parent: int) -> tuple[int, int, float, int]:
    """Pendent-pair contraction for the symmetric submodular raw-residual cut.

    Algorithm: Queyranne (1998), maximum-adjacency ordering. The ordering is
    also stated in Iwata/Konno (2026), Section 2. Contractions here are search
    operations; they are not interpreted as scientific synergy modules.
    """
    blocks = [1 << i for i in members(parent)]
    cuts = {}

    def cut(side):
        if side not in cuts:
            cuts[side] = 0.0 if side in (0, parent) else channel.gain((side, parent ^ side))
        return cuts[side]

    candidates = []
    while len(blocks) > 1:
        remaining = list(blocks)
        order, accumulated = [], 0
        while remaining:
            chosen = min(remaining, key=lambda b: (cut(accumulated | b) - cut(b), b))
            remaining.remove(chosen)
            order.append(chosen)
            accumulated |= chosen
        previous, last = order[-2:]
        candidates.append((cut(last), last))
        blocks = [b for b in blocks if b not in (previous, last)] + [previous | last]
    value, side = min(candidates)
    return side, parent ^ side, value, len(cuts)


def exact_root(channel):
    first = channel.full & -channel.full
    return min((channel.gain((s, channel.full ^ s)), s)
               for s in range(1, channel.full) if s & first)


def submodular_audit(channel):
    # Check the diminishing-returns elementary inequalities for the cut.
    full = channel.full
    cuts = {s: 0.0 if s in (0, full) else channel.gain((s, full ^ s))
            for s in range(full + 1)}
    slacks = []
    for s in range(full + 1):
        absent = [1 << i for i in range(channel.n) if not s & (1 << i)]
        for a, b in itertools.combinations(absent, 2):
            slacks.append(cuts[s | a] + cuts[s | b] - cuts[s] - cuts[s | a | b])
    minimum = min(slacks)
    if minimum < -TOL_BITS:
        raise AssertionError(f"Submodularity violated: {minimum}")
    return minimum


def deterministic_channel(n, rule):
    x = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    y = np.asarray(rule(x), dtype=int)
    return np.eye(int(y.max()) + 1)[y]


def encoded_information(z, y):
    _, zi = np.unique(z, axis=0, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((zi.max() + 1, yi.max() + 1))
    np.add.at(joint, (zi, yi), 1 / len(yi))
    return float(entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1)) - entropy(joint.ravel()))


def run():
    channels = {
        "independent_target": np.tile([0.5, 0.5], (16, 1)),
        "nested_pair_and": deterministic_channel(4, lambda x: (x[:, 0] ^ x[:, 1]) & (x[:, 2] ^ x[:, 3])),
        "parity3": deterministic_channel(3, lambda x: np.bitwise_xor.reduce(x, axis=1)),
        "parity5": deterministic_channel(5, lambda x: np.bitwise_xor.reduce(x, axis=1)),
    }
    results = {}
    for name, probabilities in channels.items():
        results[name] = {f"arity_{k}": agglomerate(ExactChannel(probabilities), k) for k in (2, 3)}
        channel = ExactChannel(probabilities)
        tree = build_spt(tuple(range(channel.n)), channel,
                         config=SPTConfig(syn_tolerance=TOL_BITS))
        _, _, q_value, _ = queyranne_split(channel, channel.full)
        expected, _ = exact_root(channel)
        assert abs(q_value - expected) <= TOL_BITS
        results[name]["standard_spt"] = {"closure_error_bits": tree.closure_error,
                                        "root_syn_bits": tree.root.syn_value,
                                        "queyranne_root_error_bits": q_value - expected}
    assert np.isclose(results["nested_pair_and"]["arity_2"]["total_xi_bits"], 0.8112781244591328)
    assert len(results["nested_pair_and"]["arity_2"]["merges"]) == 3
    assert results["parity3"]["arity_2"]["resolved_bits"] == 0
    assert results["parity3"]["arity_3"]["resolved_bits"] == 1
    assert results["parity5"]["arity_3"]["unresolved_bits"] == 1

    # Analytic PCA counterexample: covariance diag(1/4, 1), hence PC1 is X_1.
    x = (np.arange(4)[:, None] >> np.arange(2)) & 1
    y = x[:, 0] ^ x[:, 1]
    scaled = x * [1, 2]
    covariance = np.cov(scaled, rowvar=False, bias=True)
    assert np.allclose(covariance, np.diag([0.25, 1.0]))
    pca = {"original_ei_bits": encoded_information(x, y),
           "pc1_ei_bits": encoded_information(scaled[:, [1]], y),
           "xor_encoder_ei_bits": encoded_information(y[:, None], y)}
    assert pca == {"original_ei_bits": 1.0, "pc1_ei_bits": 0.0, "xor_encoder_ei_bits": 1.0}

    # Marginal prediction alone does not protect later synergy with other groups.
    x3 = (np.arange(8)[:, None] >> np.arange(3)) & 1
    y3 = np.bitwise_xor.reduce(x3, axis=1)
    contextual = {"group_ei_bits": encoded_information(x3[:, :2], y3),
                  "constant_code_ei_bits": encoded_information(np.zeros((8, 1)), y3),
                  "joint_before_bits": encoded_information(x3, y3),
                  "joint_after_bits": encoded_information(x3[:, [2]], y3)}
    assert contextual == {"group_ei_bits": 0.0, "constant_code_ei_bits": 0.0,
                          "joint_before_bits": 1.0, "joint_after_bits": 0.0}
    induced_prior = {"or_macro_probability_one": 0.75,
                     "induced_prior_ei_bits": float(entropy(np.array([0.25, 0.75]))),
                     "reuniformized_macro_ei_bits": 1.0}

    # Fixed random exact channels: stochastic dynamics, not sampled estimates.
    comparisons = []
    audit_min, negative_count = 0.0, 0
    for n in range(3, 9):
        for seed in range(6):
            rng = np.random.default_rng(1000 * n + seed)
            probabilities = rng.dirichlet(np.ones(3), size=2**n)
            candidate = ExactChannel(probabilities)
            _, _, found, calls = queyranne_split(candidate, candidate.full)
            q_queries = len(candidate.cache) - 1
            reference = ExactChannel(probabilities)
            expected, _ = exact_root(reference)
            slack = submodular_audit(reference)
            assert abs(found - expected) <= TOL_BITS
            comparisons.append({"n": n, "seed": seed, "error_bits": found - expected,
                                "queyranne_ei_queries": q_queries,
                                "exhaustive_ei_queries": len(reference.cache) - 1,
                                "cut_cache_size": calls, "minimum_submodular_slack": slack})
            for c in (candidate, reference):
                d = c.diagnostics()
                audit_min = min(audit_min, d["minimum_candidate_syn"])
                negative_count += d["tolerance_negative_count"]
    return {"protocol": {"unit": "bits", "syn_tolerance": TOL_BITS,
                         "intervention": "independent uniform binary sources",
                         "horizon": 1, "estimator": "exact discrete marginalization",
                         "merge_score": "raw gain / (arity - 1)",
                         "stop": "best raw gain <= syn_tolerance; preserve unresolved budget",
                         "random_channels": 36, "maximum_source_dimension": 8},
            "toy_cases": results, "pca_counterexample": pca,
            "contextual_compression_counterexample": contextual,
            "macro_prior_counterexample": induced_prior,
            "queyranne_comparisons": comparisons,
            "queyranne_audit": {"minimum_candidate_syn_with_zero_baseline": audit_min,
                                "tolerance_negative_count": negative_count,
                                "significant_negative_count": 0}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
