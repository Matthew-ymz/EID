"""Canonical Synergy Partition Tree construction.

Every experiment supplies the same three things: an ordered source coalition,
an oracle returning coalition Xi, and an explicit candidate-split strategy.
The builder owns recursion, split selection, nonnegativity auditing, closure,
and the common tree output.  Plotting and experiment annotations stay outside.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

Source: TypeAlias = Hashable
Coalition: TypeAlias = tuple[Source, ...]
Split: TypeAlias = tuple[Coalition, Coalition]
RAW_RESIDUAL = "raw_residual"
ALL_ORDER_CROSS_DENSITY = "all_order_cross_density"
NONNEGATIVE_TOLERANT = "nonnegative_tolerant"
SIGNED = "signed"


class XiOracle(Protocol):
    def xi(self, sources: Iterable[Source]) -> float: ...


CandidateSelector: TypeAlias = Callable[[Coalition], tuple[str, Iterable[Split]]]


class SPTNonnegativityError(RuntimeError):
    """A computed Syn was below the declared native-unit tolerance."""


@dataclass(frozen=True)
class SPTConfig:
    policy: str = NONNEGATIVE_TOLERANT
    split_objective: str = RAW_RESIDUAL
    syn_tolerance: float = 1.0e-4
    eps: float = 1.0e-5
    complete_to_singletons: bool = True
    candidate_budget: int | None = None
    local_search_top_k: int = 0

    def __post_init__(self) -> None:
        if self.policy not in (NONNEGATIVE_TOLERANT, SIGNED):
            raise ValueError(f"Unsupported SPT policy: {self.policy!r}")
        if self.split_objective not in (RAW_RESIDUAL, ALL_ORDER_CROSS_DENSITY):
            raise ValueError(f"Unsupported split objective: {self.split_objective!r}")
        if self.policy == SIGNED and self.split_objective != RAW_RESIDUAL:
            raise ValueError("All-order normalization is unavailable for signed SPT.")
        if (
            not math.isfinite(self.syn_tolerance)
            or not math.isfinite(self.eps)
            or self.syn_tolerance < 0.0
            or self.eps < 0.0
        ):
            raise ValueError("SPT tolerances must be finite and nonnegative.")
        if self.candidate_budget is not None and self.candidate_budget <= 0:
            raise ValueError("The SPT candidate budget must be positive when declared.")
        if self.local_search_top_k < 0:
            raise ValueError("The SPT local-search start count must be nonnegative.")


@dataclass
class SPTAudit:
    candidate_count: int = 0
    initial_candidate_count: int = 0
    local_candidate_count: int = 0
    local_improvement_count: int = 0
    tolerance_zero_count: int = 0
    selected_split_count: int = 0
    minimum_candidate_syn: float = float("inf")


@dataclass(frozen=True)
class SPTNode:
    sources: Coalition
    xi_value: float
    syn_value: float
    depth: int
    split_kind: str
    children: tuple[SPTNode, ...] = ()

    @property
    def size(self) -> int:
        return len(self.sources)

    @property
    def order(self) -> int:
        return len(self.sources)

    @property
    def indices(self) -> Coalition:
        return self.sources

    @property
    def phi_value(self) -> float:
        return self.xi_value

    @property
    def xi_bits(self) -> float:
        return self.xi_value

    @property
    def residual(self) -> float:
        return self.syn_value

    @property
    def syn_bits(self) -> float:
        return self.syn_value

    @property
    def search_kind(self) -> str:
        return self.split_kind

    @property
    def action(self) -> str:
        if self.children:
            return "split"
        return "terminal" if self.split_kind == "terminal" else "leaf"

    @property
    def atom_kind(self) -> str | None:
        if self.split_kind == "terminal":
            return "terminal"
        if not self.children:
            return None
        return "terminal" if all(not child.children for child in self.children) else "split_residual"


@dataclass(frozen=True)
class SPTAtom:
    sources: Coalition
    value: float
    kind: str
    depth: int


@dataclass(frozen=True)
class SPTResult:
    root: SPTNode
    audit: SPTAudit
    closure_error: float


@dataclass
class TableXiOracle:
    """Adapt one complete EI table to the canonical coalition-Xi interface."""

    ei_table: Mapping[Coalition, float]
    source_order: Coalition
    singleton_ei: Mapping[Source, float] | None = None
    _cache: dict[Coalition, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.singleton_ei is None:
            self.singleton_ei = {source: float(self.ei_table[(source,)]) for source in self.source_order}

    def canonical(self, sources: Iterable[Source]) -> Coalition:
        selected = set(sources)
        return tuple(source for source in self.source_order if source in selected)

    def xi(self, sources: Iterable[Source]) -> float:
        key = self.canonical(sources)
        if len(key) <= 1:
            return 0.0
        if key not in self._cache:
            self._cache[key] = float(
                self.ei_table[key] - sum(float(self.singleton_ei[source]) for source in key)  # type: ignore[index]
            )
        return self._cache[key]


def all_nonempty_subsets(sources: Sequence[Source]) -> list[Coalition]:
    ordered = tuple(sources)
    return [
        tuple(subset)
        for size in range(1, len(ordered) + 1)
        for subset in itertools.combinations(ordered, size)
    ]


def canonical_split(left: Iterable[Source], right: Iterable[Source], order: Coalition) -> Split:
    left_set, right_set = set(left), set(right)
    left_key = tuple(source for source in order if source in left_set)
    right_key = tuple(source for source in order if source in right_set)
    return (
        (left_key, right_key)
        if order.index(left_key[0]) <= order.index(right_key[0])
        else (right_key, left_key)
    )


def nontrivial_bipartitions(sources: Sequence[Source]) -> list[Split]:
    ordered = tuple(sources)
    if len(ordered) <= 1:
        return []
    first, rest = ordered[0], ordered[1:]
    full = set(ordered)
    splits: list[Split] = []
    for mask in range(1 << len(rest)):
        left = {first}
        left.update(rest[index] for index in range(len(rest)) if mask & (1 << index))
        if len(left) == len(ordered):
            continue
        right = full - left
        splits.append(
            (
                tuple(source for source in ordered if source in left),
                tuple(source for source in ordered if source in right),
            )
        )
    return splits


def exact_candidate_selector(sources: Coalition) -> tuple[str, Iterable[Split]]:
    return "exact", nontrivial_bipartitions(sources)


def stratified_random_candidate_selector(
    *,
    initial_budget: int,
    exact_max_size: int,
    seed: int,
) -> CandidateSelector:
    """Sample candidate splits evenly across smaller-child cardinalities."""
    if int(initial_budget) <= 0:
        raise ValueError("The initial stratified-random budget must be positive.")
    if int(exact_max_size) < 2:
        raise ValueError("The exact-search threshold must be at least two.")
    rng = np.random.default_rng(int(seed))

    def select(sources: Coalition) -> tuple[str, Iterable[Split]]:
        ordered = tuple(sources)
        size = len(ordered)
        if size <= int(exact_max_size):
            return "exact", nontrivial_bipartitions(ordered)

        maximum_small_size = size // 2
        levels = list(range(1, maximum_small_size + 1))
        capacities = {
            child_size: math.comb(size, child_size)
            // (2 if size % 2 == 0 and child_size == maximum_small_size else 1)
            for child_size in levels
        }
        target = min(int(initial_budget), sum(capacities.values()))
        quotas = {child_size: 0 for child_size in levels}
        remaining = target
        active = set(levels)
        while remaining > 0 and active:
            share = max(1, remaining // len(active))
            progress = 0
            for child_size in sorted(active):
                addition = min(share, capacities[child_size] - quotas[child_size])
                quotas[child_size] += addition
                remaining -= addition
                progress += addition
                if quotas[child_size] == capacities[child_size]:
                    active.remove(child_size)
                if remaining == 0:
                    break
            if progress == 0:
                break

        full = set(ordered)
        candidates: set[Split] = set()
        for child_size in levels:
            quota = quotas[child_size]
            if quota == 0:
                continue
            if quota == capacities[child_size]:
                for subset in itertools.combinations(ordered, child_size):
                    candidates.add(canonical_split(subset, full - set(subset), ordered))
                continue
            level_candidates: set[Split] = set()
            while len(level_candidates) < quota:
                positions = np.sort(rng.choice(size, size=child_size, replace=False))
                subset = tuple(ordered[int(position)] for position in positions)
                level_candidates.add(canonical_split(subset, full - set(subset), ordered))
            candidates.update(level_candidates)
        return "stratified-random", candidates

    return select


def cross_coalition_count(left_size: int, right_size: int) -> int:
    if int(left_size) <= 0 or int(right_size) <= 0:
        raise ValueError("Both split sides must be nonempty.")
    return ((1 << int(left_size)) - 1) * ((1 << int(right_size)) - 1)


def split_objective_value(residual: float, left_size: int, right_size: int, *, objective: str) -> float:
    if objective == RAW_RESIDUAL:
        return float(residual)
    if objective == ALL_ORDER_CROSS_DENSITY:
        return float(residual) / float(cross_coalition_count(left_size, right_size))
    raise ValueError(f"Unsupported split objective: {objective!r}")


def _audit_syn(value: float, *, tolerance: float, context: str, audit: SPTAudit) -> None:
    if not math.isfinite(value):
        raise SPTNonnegativityError(f"Non-finite Syn estimate in {context}: value={value}.")
    audit.minimum_candidate_syn = min(audit.minimum_candidate_syn, float(value))
    if value < -float(tolerance):
        raise SPTNonnegativityError(
            f"Syn nonnegativity violation in {context}: minimum={value:.12g}, "
            f"threshold={-float(tolerance):.12g}, affected_count=1."
        )
    if value < 0.0:
        audit.tolerance_zero_count += 1


def audit_syn_value(value: float, *, tolerance: float, context: str) -> bool:
    """Audit one Syn and report whether it is a tolerance-scale negative."""
    audit = SPTAudit()
    _audit_syn(float(value), tolerance=float(tolerance), context=context, audit=audit)
    return bool(audit.tolerance_zero_count)


def build_spt(
    sources: Sequence[Source],
    oracle: XiOracle,
    *,
    config: SPTConfig | None = None,
    candidate_selector: CandidateSelector = exact_candidate_selector,
    depth: int = 0,
    audit: SPTAudit | None = None,
) -> SPTResult:
    """Build one SPT with a fixed input/output contract."""
    cfg = config or SPTConfig()
    shared_audit = audit or SPTAudit()
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("SPT sources must be nonempty and unique.")

    def visit(coalition: Coalition, node_depth: int) -> SPTNode:
        block_xi = float(oracle.xi(coalition))
        if not math.isfinite(block_xi):
            raise ValueError(f"Non-finite Xi for coalition {coalition}.")
        if len(coalition) == 1:
            return SPTNode(coalition, block_xi, 0.0, node_depth, "leaf")
        if not cfg.complete_to_singletons and cfg.policy == NONNEGATIVE_TOLERANT and block_xi <= cfg.eps:
            return SPTNode(coalition, block_xi, block_xi, node_depth, "terminal")

        split_kind, raw_candidates = candidate_selector(coalition)
        candidates = list(raw_candidates)
        if not candidates:
            raise RuntimeError(f"No candidate split was generated for coalition {coalition}.")
        scored: list[tuple[float, float, float, Coalition, Coalition]] = []
        evaluated: dict[Split, tuple[float, float, float, Coalition, Coalition]] = {}

        def score(
            left: Coalition, right: Coalition, *, local: bool
        ) -> tuple[float, float, float, Coalition, Coalition]:
            split = canonical_split(left, right, coalition)
            cached = evaluated.get(split)
            if cached is not None:
                return cached
            left, right = split
            if (
                not left
                or not right
                or set(left).intersection(right)
                or set(left).union(right) != set(coalition)
            ):
                raise ValueError(f"Invalid SPT bipartition {left} | {right} for {coalition}.")
            residual = float(block_xi - oracle.xi(left) - oracle.xi(right))
            shared_audit.candidate_count += 1
            if local:
                shared_audit.local_candidate_count += 1
            else:
                shared_audit.initial_candidate_count += 1
            if cfg.policy == NONNEGATIVE_TOLERANT:
                _audit_syn(
                    residual,
                    tolerance=cfg.syn_tolerance,
                    context=f"split {left} | {right}",
                    audit=shared_audit,
                )
            objective = split_objective_value(residual, len(left), len(right), objective=cfg.split_objective)
            value = (objective, float(oracle.xi(left) + oracle.xi(right)), residual, left, right)
            evaluated[split] = value
            return value

        def rank(item: tuple[float, float, float, Coalition, Coalition]) -> tuple:
            if cfg.policy == SIGNED:
                return (-item[1], abs(item[2]), item[3], item[4])
            if cfg.split_objective == ALL_ORDER_CROSS_DENSITY:
                return (item[0], item[2], item[3], item[4])
            return (-item[1], item[2], item[3], item[4])

        for left, right in candidates:
            scored.append(score(left, right, local=False))

        if split_kind == "stratified-random" and cfg.local_search_top_k > 0:
            maximum_evaluations = cfg.candidate_budget or len(scored)
            starts = sorted(scored, key=rank)[: int(cfg.local_search_top_k)]
            full = set(coalition)
            for start in starts:
                current = start
                while len(evaluated) < maximum_evaluations:
                    _, _, _, current_left, current_right = current
                    neighbors: list[tuple[float, float, float, Coalition, Coalition]] = []
                    for source in coalition:
                        if source in current_left and len(current_left) > 1:
                            moved_left = tuple(item for item in current_left if item != source)
                            moved_right = tuple(item for item in coalition if item in full - set(moved_left))
                        elif source in current_right and len(current_right) > 1:
                            moved_right = tuple(item for item in current_right if item != source)
                            moved_left = tuple(item for item in coalition if item in full - set(moved_right))
                        else:
                            continue
                        split = canonical_split(moved_left, moved_right, coalition)
                        if split in evaluated:
                            neighbors.append(evaluated[split])
                            continue
                        if len(evaluated) >= maximum_evaluations:
                            break
                        neighbors.append(score(*split, local=True))
                    if not neighbors:
                        break
                    best_neighbor = min(neighbors, key=rank)
                    if rank(best_neighbor) >= rank(current):
                        break
                    current = best_neighbor
                    shared_audit.local_improvement_count += 1
            scored = list(evaluated.values())

        _, _, residual, left, right = min(scored, key=rank)
        shared_audit.selected_split_count += 1
        children = (visit(left, node_depth + 1), visit(right, node_depth + 1))
        return SPTNode(coalition, block_xi, float(residual), node_depth, split_kind, children)

    root = visit(tuple(sources), int(depth))
    closure = float(
        sum(node.syn_value for node in flatten_nodes(root) if node.atom_kind is not None) - root.xi_value
    )
    return SPTResult(root=root, audit=shared_audit, closure_error=closure)


def build_spt_from_ei_table(
    sources: Sequence[Source],
    ei_table: Mapping[Coalition, float],
    *,
    singleton_ei: Mapping[Source, float] | None = None,
    config: SPTConfig | None = None,
    candidate_selector: CandidateSelector = exact_candidate_selector,
    depth: int = 0,
) -> SPTResult:
    ordered = tuple(sources)
    return build_spt(
        ordered,
        TableXiOracle(ei_table, ordered, singleton_ei),
        config=config,
        candidate_selector=candidate_selector,
        depth=depth,
    )


def flatten_nodes(root: SPTNode) -> list[SPTNode]:
    nodes = [root]
    for child in root.children:
        nodes.extend(flatten_nodes(child))
    return nodes


def flatten_atoms(root: SPTNode) -> list[SPTAtom]:
    return [
        SPTAtom(node.sources, float(node.syn_value), str(node.atom_kind), int(node.depth))
        for node in flatten_nodes(root)
        if node.atom_kind is not None and float(node.syn_value) != 0.0
    ]


def spectral_candidate_selector(
    affinity: np.ndarray,
    *,
    exact_max_size: int,
) -> CandidateSelector:
    matrix = np.asarray(affinity, dtype=float)

    def select(sources: Coalition) -> tuple[str, Iterable[Split]]:
        indices = tuple(int(source) for source in sources)
        if len(indices) <= int(exact_max_size):
            return "exact", nontrivial_bipartitions(indices)
        local = matrix[np.ix_(indices, indices)]
        orders: list[tuple[int, ...]] = []
        if len(indices) >= 3 and float(local.max()) > 0.0:
            degree = local.sum(axis=1)
            inverse_root = np.zeros_like(degree)
            positive = degree > 0.0
            inverse_root[positive] = 1.0 / np.sqrt(degree[positive])
            laplacian = np.eye(len(indices)) - inverse_root[:, None] * local * inverse_root[None, :]
            _, vectors = np.linalg.eigh(laplacian)
            for column in range(1, min(5, vectors.shape[1])):
                orders.append(
                    tuple(indices[position] for position in np.argsort(vectors[:, column], kind="stable"))
                )
            distances = 1.0 - local / float(local.max())
            np.fill_diagonal(distances, 0.0)
            hierarchy = linkage(squareform(distances, checks=False), method="average", optimal_ordering=True)
            orders.append(tuple(indices[position] for position in leaves_list(hierarchy)))
        orders.append(indices)
        candidates: set[Split] = set()
        full = set(indices)
        for order in orders:
            for cut in range(1, len(order)):
                candidates.add(canonical_split(order[:cut], full - set(order[:cut]), indices))
        return "spectral-candidate", candidates

    return select


def pairwise_syn_affinity(
    oracle: XiOracle,
    node_count: int,
    *,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    affinity = np.zeros((int(node_count), int(node_count)), dtype=np.float64)
    audit = SPTAudit()
    for left in range(int(node_count)):
        for right in range(left + 1, int(node_count)):
            value = float(oracle.xi((left, right)) - oracle.xi((left,)) - oracle.xi((right,)))
            _audit_syn(
                value,
                tolerance=float(tolerance),
                context=f"pair ({left}, {right})",
                audit=audit,
            )
            affinity[left, right] = affinity[right, left] = value
    return affinity, int(audit.tolerance_zero_count)
