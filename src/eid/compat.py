"""Backward-compatible names for the canonical SPT module.

New code should import :mod:`eid.spt` directly. Existing experiments keep
their historical Phi-oriented API, but every tree now delegates to the same
``build_spt`` implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from eid.spt import (
    NONNEGATIVE_TOLERANT,
    RAW_RESIDUAL,
    SPTAtom,
    SPTConfig,
    SPTNode,
    build_spt_from_ei_table,
    flatten_atoms,
)

PhiAtom = SPTAtom
PhiTreeNode = SPTNode
HierarchyPolicy = str
SplitObjective = str


def subset_phi_raw(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    singleton_ei: Mapping[str, float] | None = None,
) -> float:
    ordered = tuple(str(name) for name in subset)
    singles = {name: float(ei_table[(name,)]) for name in ordered} if singleton_ei is None else singleton_ei
    return float(ei_table[ordered] - sum(float(singles[name]) for name in ordered))


def greedy_phi_tree(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    policy: HierarchyPolicy = NONNEGATIVE_TOLERANT,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
    split_objective: SplitObjective = RAW_RESIDUAL,
    complete_to_singletons: bool = True,
) -> PhiTreeNode:
    ordered = tuple(str(name) for name in subset)
    result = build_spt_from_ei_table(
        ordered,
        ei_table,
        singleton_ei=singleton_ei,
        config=SPTConfig(
            policy=policy,
            split_objective=split_objective,
            syn_tolerance=float(split_tolerance),
            eps=float(eps),
            complete_to_singletons=bool(complete_to_singletons),
        ),
        depth=int(depth),
    )
    return result.root


def flatten_phi_tree(tree: PhiTreeNode) -> list[PhiAtom]:
    return flatten_atoms(tree)


def greedy_phi_atoms(
    subset: Sequence[str],
    ei_table: Mapping[tuple[str, ...], float],
    *,
    policy: HierarchyPolicy = NONNEGATIVE_TOLERANT,
    eps: float = 1.0e-5,
    split_tolerance: float = 1.0e-4,
    depth: int = 0,
    singleton_ei: Mapping[str, float] | None = None,
    split_objective: SplitObjective = RAW_RESIDUAL,
    complete_to_singletons: bool = True,
) -> list[PhiAtom]:
    return flatten_phi_tree(
        greedy_phi_tree(
            subset,
            ei_table,
            policy=policy,
            eps=eps,
            split_tolerance=split_tolerance,
            depth=depth,
            singleton_ei=singleton_ei,
            split_objective=split_objective,
            complete_to_singletons=complete_to_singletons,
        )
    )
