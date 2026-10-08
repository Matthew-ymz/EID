from __future__ import annotations

import math

import numpy as np
import pytest

from eid.spt import (
    SIGNED,
    SPTConfig,
    SPTNonnegativityError,
    TableXiOracle,
    build_spt,
    build_spt_from_ei_table,
    flatten_nodes,
    spectral_candidate_selector,
    stratified_random_candidate_selector,
)


def independent_pair_table() -> dict[tuple[str, ...], float]:
    return {
        ("a",): 0.0,
        ("b",): 0.0,
        ("c",): 0.0,
        ("d",): 0.0,
        ("a", "b"): 1.0,
        ("a", "c"): 0.0,
        ("a", "d"): 0.0,
        ("b", "c"): 0.0,
        ("b", "d"): 0.0,
        ("c", "d"): 1.0,
        ("a", "b", "c"): 1.0,
        ("a", "b", "d"): 1.0,
        ("a", "c", "d"): 1.0,
        ("b", "c", "d"): 1.0,
        ("a", "b", "c", "d"): 2.0,
    }


def test_table_and_oracle_inputs_share_one_complete_tree_contract() -> None:
    sources = ("a", "b", "c", "d")
    table = independent_pair_table()
    table_result = build_spt_from_ei_table(sources, table)
    oracle_result = build_spt(sources, TableXiOracle(table, sources))

    table_nodes = flatten_nodes(table_result.root)
    oracle_nodes = flatten_nodes(oracle_result.root)
    assert [(node.sources, node.syn_value) for node in table_nodes] == [
        (node.sources, node.syn_value) for node in oracle_nodes
    ]
    assert sum(not node.children for node in table_nodes) == 4
    assert all(len(node.sources) == 1 for node in table_nodes if not node.children)
    assert math.isclose(table_result.closure_error, 0.0, abs_tol=1.0e-12)


def test_prior_constraint_changes_only_candidate_route() -> None:
    sources = ("a", "b", "c", "d")
    table = independent_pair_table()
    allowed = {
        sources: (("a", "c"), ("b", "d")),
        ("a", "c"): (("a",), ("c",)),
        ("b", "d"): (("b",), ("d",)),
    }

    def prior_selector(coalition):
        return "fixed-prior", [allowed[coalition]]

    constrained = build_spt_from_ei_table(
        sources,
        table,
        config=SPTConfig(policy=SIGNED),
        candidate_selector=prior_selector,
    )
    assert constrained.root.split_kind == "fixed-prior"
    assert [child.sources for child in constrained.root.children] == [
        ("a", "c"),
        ("b", "d"),
    ]
    assert all(len(node.sources) == 1 for node in flatten_nodes(constrained.root) if not node.children)


def test_nonnegative_route_fails_instead_of_clipping_syn() -> None:
    table = {("a",): 0.0, ("b",): 0.0, ("a", "b"): -0.2}
    with pytest.raises(SPTNonnegativityError, match="affected_count=1"):
        build_spt_from_ei_table(
            ("a", "b"),
            table,
            config=SPTConfig(syn_tolerance=0.01),
        )


def test_tolerance_scale_negative_is_retained_raw_and_signed_route_is_explicit() -> None:
    table = {("a",): 0.0, ("b",): 0.0, ("a", "b"): -0.005}
    tolerant = build_spt_from_ei_table(("a", "b"), table, config=SPTConfig(syn_tolerance=0.01))
    signed = build_spt_from_ei_table(("a", "b"), table, config=SPTConfig(policy=SIGNED))
    assert tolerant.root.syn_value == -0.005
    assert tolerant.audit.tolerance_zero_count == 1
    assert signed.root.syn_value == -0.005


def test_optional_legacy_terminal_is_still_a_closed_atom() -> None:
    table = {
        ("a",): 0.2,
        ("b",): 0.3,
        ("a", "b"): 0.5,
    }
    result = build_spt_from_ei_table(
        ("a", "b"),
        table,
        config=SPTConfig(eps=1.0e-8, complete_to_singletons=False),
    )
    assert result.root.action == "terminal"
    assert result.root.atom_kind == "terminal"
    assert result.closure_error == pytest.approx(0.0)


def test_spectral_candidates_do_not_supplement_every_singleton_split() -> None:
    sources = tuple(range(6))
    _, candidates = spectral_candidate_selector(np.zeros((6, 6), dtype=float), exact_max_size=0)(sources)
    singleton_sides = {side for split in candidates for side in split if len(side) == 1}
    assert singleton_sides == {(0,), (5,)}


def test_stratified_random_candidates_cover_sizes_reproducibly() -> None:
    sources = tuple(range(10))
    first = stratified_random_candidate_selector(initial_budget=80, exact_max_size=2, seed=7)
    second = stratified_random_candidate_selector(initial_budget=80, exact_max_size=2, seed=7)
    first_kind, first_candidates = first(sources)
    second_kind, second_candidates = second(sources)
    first_candidates = set(first_candidates)
    second_candidates = set(second_candidates)
    assert first_kind == second_kind == "stratified-random"
    assert first_candidates == second_candidates
    assert len(first_candidates) == 80
    assert {min(len(left), len(right)) for left, right in first_candidates} == set(range(1, 6))
