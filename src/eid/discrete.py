"""Finite binary-state EI under a common uniform intervention.

Extracted from EISyn/utils.py; plotting and experiment code are excluded.
Generic rectangular TPM EI is also available via effective_information_from_tpm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np


def enumerate_binary_states(n: int) -> np.ndarray:
    """Enumerate all binary states in lexicographic order."""

    if n < 0:
        raise ValueError("Number of nodes must be non-negative.")
    if n == 0:
        return np.zeros((1, 0), dtype=int)
    return np.array(list(product([0, 1], repeat=n)), dtype=int)


def build_probabilistic_boolean_tpm(
    adjacency: np.ndarray,
    node_specs: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Build a system TPM from independent Bernoulli node updates.

    `adjacency[i, j]` denotes the weight from node `i` at time `t` to node `j`
    at time `t+1`. Each node specification can provide:

    - `bias`: scalar bias term
    - `alpha`, `beta`, `gamma`: coefficients for copy / coop / parity terms
    - `copy_weights`: optional explicit weights overriding the adjacency column
    - `coop_sources`: optional list of source indices for one centered product term
    - `coop_pairs`: optional list of `(i, k)` or `(i, k, weight)`
    - `parity_sources`: optional list of source indices used in parity
    - `eta`: optional scaling on the parity term
    """

    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")
    n_nodes = matrix.shape[0]
    if len(node_specs) != n_nodes:
        raise ValueError("Need one node specification per node.")

    current_states = enumerate_binary_states(n_nodes)
    next_states = current_states
    tpm = np.zeros((len(current_states), len(next_states)), dtype=float)

    for row_index, state in enumerate(current_states):
        probabilities = np.array(
            [_node_activation_probability(state, matrix, node_specs, j) for j in range(n_nodes)],
            dtype=float,
        )
        tpm[row_index] = np.prod(
            np.where(next_states == 1, probabilities, 1.0 - probabilities),
            axis=1,
        )

    return tpm


def build_deterministic_boolean_tpm(
    n_nodes: int,
    update_rule: Callable[[tuple[int, ...]], Sequence[int]],
) -> np.ndarray:
    """Build a deterministic Boolean TPM from an explicit update rule."""

    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative.")

    states = enumerate_binary_states(n_nodes)
    state_to_index = {tuple(state.tolist()): index for index, state in enumerate(states)}
    tpm = np.zeros((len(states), len(states)), dtype=float)

    for row_index, state in enumerate(states):
        state_tuple = tuple(int(bit) for bit in state.tolist())
        next_state = tuple(int(bit) for bit in update_rule(state_tuple))
        if len(next_state) != n_nodes:
            raise ValueError("update_rule must return one bit per node.")
        if any(bit not in (0, 1) for bit in next_state):
            raise ValueError("update_rule must return binary states.")
        tpm[row_index, state_to_index[next_state]] = 1.0

    return tpm


def coarse_grain_tpm_by_state_labels(
    tpm: np.ndarray,
    *,
    input_state_labels: Sequence[int],
    output_state_labels: Sequence[int] | None = None,
) -> np.ndarray:
    """Coarse-grain a TPM by averaging rows and aggregating columns."""

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")

    input_labels = np.asarray(input_state_labels, dtype=int).reshape(-1)
    if input_labels.shape[0] != matrix.shape[0]:
        raise ValueError("input_state_labels must match the number of TPM rows.")
    output_labels = np.asarray(
        output_state_labels if output_state_labels is not None else input_state_labels,
        dtype=int,
    ).reshape(-1)
    if output_labels.shape[0] != matrix.shape[1]:
        raise ValueError("output_state_labels must match the number of TPM columns.")
    if np.any(input_labels < 0) or np.any(output_labels < 0):
        raise ValueError("State labels must be non-negative integers.")

    n_input_macro = int(input_labels.max()) + 1 if input_labels.size else 0
    macro_rows = np.zeros((n_input_macro, matrix.shape[1]), dtype=float)
    for macro_state in range(n_input_macro):
        member_rows = np.flatnonzero(input_labels == macro_state)
        if len(member_rows) == 0:
            continue
        macro_rows[macro_state] = matrix[member_rows].mean(axis=0)

    return coarse_grain_output_tpm(macro_rows, output_labels)


def coarse_grain_binary_or_tpm(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    """Coarse-grain a binary system by OR-pooling node groups into macro states."""

    normalized_groups = tuple(_normalize_indices(group, n_nodes, "groups block") for group in groups)
    covered = tuple(sorted(index for group in normalized_groups for index in group))
    if covered != tuple(range(n_nodes)):
        raise ValueError("groups must form a partition of all nodes.")
    if len(set(covered)) != len(covered):
        raise ValueError("groups must be disjoint.")

    states = enumerate_binary_states(n_nodes)
    labels = np.zeros(len(states), dtype=int)
    for row_index, state in enumerate(states):
        macro_bits = [int(np.any(state[list(group)])) for group in normalized_groups]
        labels[row_index] = int(_encode_binary_states(np.asarray(macro_bits, dtype=int).reshape(1, -1))[0])

    return coarse_grain_tpm_by_state_labels(system_tpm, input_state_labels=labels)


def effective_information_from_tpm(
    tpm: np.ndarray,
    *,
    log_base: float = 2.0,
    atol: float = 1e-12,
) -> float:
    """Compute effective information from a TPM under a maximum-entropy input.

    The TPM is assumed to be row-stochastic, where rows correspond to input states
    and columns correspond to output states. The input intervention is the uniform
    distribution over all input states, which is the maximum-entropy distribution
    for a finite discrete state space.
    """

    if not math.isfinite(log_base) or log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be finite, positive, and different from one.")
    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("TPM must be non-empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("TPM entries must be finite.")
    if np.any(matrix < -atol):
        raise ValueError("TPM entries must be non-negative.")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Each TPM row must sum to 1.")

    matrix = np.clip(matrix, 0.0, None)
    n_inputs = matrix.shape[0]
    output_marginal = matrix.mean(axis=0)

    positive_mask = matrix > 0.0
    denominator = output_marginal[np.newaxis, :]
    ratios = np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > 0.0,
    )
    log_term = np.zeros_like(matrix)
    log_term[positive_mask] = np.log(ratios[positive_mask]) / math.log(log_base)

    ei = np.sum(matrix[positive_mask] * log_term[positive_mask]) / n_inputs
    return float(ei)


def source_subset_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    source_indices: Sequence[int],
) -> np.ndarray:
    """Marginalize a full system TPM to a source subset under uniform intervention."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    indices = tuple(sorted(source_indices))
    if not indices:
        raise ValueError("source_indices must be non-empty.")
    if len(set(indices)) != len(indices):
        raise ValueError("source_indices must be unique.")
    if any(i < 0 or i >= n_nodes for i in indices):
        raise ValueError("source_indices contain out-of-range values.")

    full_states = enumerate_binary_states(n_nodes)
    subset_states = full_states[:, indices]
    subset_ids = _encode_binary_states(subset_states)
    n_subset_states = 2 ** len(indices)
    averaged_tpm = np.zeros((n_subset_states, system.shape[1]), dtype=float)

    for row, subset_id in enumerate(subset_ids):
        averaged_tpm[subset_id] += system[row]

    averaged_tpm /= 2 ** (n_nodes - len(indices))
    return averaged_tpm


def coarse_grain_output_tpm(
    tpm: np.ndarray,
    output_labels: Sequence[int],
) -> np.ndarray:
    """Aggregate TPM columns according to a discrete output labeling."""

    matrix = np.asarray(tpm, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("TPM must be a 2D array.")

    labels = np.asarray(output_labels, dtype=int).reshape(-1)
    if labels.shape[0] != matrix.shape[1]:
        raise ValueError("output_labels must match the number of TPM columns.")

    unique_labels, inverse = np.unique(labels, return_inverse=True)
    coarse_tpm = np.zeros((matrix.shape[0], len(unique_labels)), dtype=float)
    for column_index, label_index in enumerate(inverse):
        coarse_tpm[:, label_index] += matrix[:, column_index]
    return coarse_tpm


def target_subset_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    target_indices: Sequence[int],
) -> np.ndarray:
    """Marginalize a full system TPM to a target subset."""

    system = _validate_system_tpm(system_tpm, n_nodes)
    indices = _normalize_indices(target_indices, n_nodes, "target_indices")
    if not indices:
        raise ValueError("target_indices must be non-empty.")

    full_states = enumerate_binary_states(n_nodes)
    target_states = full_states[:, indices]
    target_ids = _encode_binary_states(target_states)
    return coarse_grain_output_tpm(system, target_ids)


def source_target_tpm(
    system_tpm: np.ndarray,
    n_nodes: int,
    *,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
) -> np.ndarray:
    """Return the TPM from a source subset to a target subset."""

    indices = _normalize_indices(target_indices, n_nodes, "target_indices")
    if not indices:
        raise ValueError("target_indices must be non-empty.")

    source_tpm = source_subset_tpm(system_tpm, n_nodes=n_nodes, source_indices=source_indices)
    full_states = enumerate_binary_states(n_nodes)
    target_states = full_states[:, indices]
    target_ids = _encode_binary_states(target_states)
    return coarse_grain_output_tpm(source_tpm, target_ids)


def discrete_effective_information(
    system_tpm: np.ndarray,
    *,
    n_nodes: int,
    source_indices: Sequence[int],
    target_indices: Sequence[int],
    log_base: float = 2.0,
) -> float:
    """Compute EI from a source subset to a target subset in a discrete system."""

    return effective_information_from_tpm(
        source_target_tpm(
            system_tpm,
            n_nodes=n_nodes,
            source_indices=source_indices,
            target_indices=target_indices,
        ),
        log_base=log_base,
    )


def _node_activation_probability(
    state: np.ndarray,
    adjacency: np.ndarray,
    node_specs: Sequence[Mapping[str, Any]],
    node: int,
) -> float:
    spec = node_specs[node]
    bias = float(spec.get("bias", 0.0))
    alpha = float(spec.get("alpha", 1.0))
    beta = float(spec.get("beta", 1.0))
    gamma = float(spec.get("gamma", 1.0))
    eta = float(spec.get("eta", 1.0))

    copy_weights = spec.get("copy_weights")
    if copy_weights is None:
        weights = adjacency[:, node]
    else:
        weights = np.zeros(adjacency.shape[0], dtype=float)
        if isinstance(copy_weights, Mapping):
            for source, weight in copy_weights.items():
                weights[int(source)] = float(weight)
        else:
            for entry in copy_weights:
                source, weight = entry
                weights[int(source)] = float(weight)

    centered_state = 2.0 * np.asarray(state, dtype=float) - 1.0
    copy_term = float(np.dot(weights, centered_state))

    coop_sources = tuple(int(i) for i in spec.get("coop_sources", ()))
    if coop_sources:
        coop_term = float(
            np.prod(np.asarray(state, dtype=float)[list(coop_sources)]) - 2.0 ** (-len(coop_sources))
        )
    else:
        coop_term = 0.0
        for entry in spec.get("coop_pairs", []):
            if len(entry) == 2:
                i, k = entry
                weight = 1.0
            else:
                i, k, weight = entry
            coop_term += float(weight) * (float(state[int(i)] * state[int(k)]) - 0.25)

    parity_sources = tuple(int(i) for i in spec.get("parity_sources", ()))
    if parity_sources:
        parity_bit = sum(int(state[i]) for i in parity_sources) % 2
        parity_term = eta * (2.0 * parity_bit - 1.0)
    else:
        parity_term = 0.0

    logit = bias + alpha * copy_term + beta * coop_term + gamma * parity_term
    return 1.0 / (1.0 + math.exp(-logit))


def _validate_system_tpm(system_tpm: np.ndarray, n_nodes: int, atol: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(system_tpm, dtype=float)
    expected_size = 2**n_nodes
    if matrix.shape != (expected_size, expected_size):
        raise ValueError("System TPM shape does not match n_nodes.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Each system TPM row must sum to 1.")
    if np.any(matrix < -atol):
        raise ValueError("System TPM entries must be non-negative.")
    return np.clip(matrix, 0.0, None)


def _encode_binary_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.shape[1] == 0:
        return np.zeros(states.shape[0], dtype=int)
    weights = 2 ** np.arange(states.shape[1] - 1, -1, -1)
    return states @ weights


def _normalize_indices(indices: Sequence[int], size: int, name: str) -> list[int]:
    normalized = sorted(int(index) for index in indices)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique.")
    if any(index < 0 or index >= size for index in normalized):
        raise ValueError(f"{name} contain out-of-range values.")
    return normalized
