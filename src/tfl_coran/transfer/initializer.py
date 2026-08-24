from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from tfl_coran.federated.aggregator import clone_state, mix_states

TensorState = dict[str, torch.Tensor]


def _most_similar_neighbor(
    index: int, contexts: np.ndarray, cell_ids: np.ndarray, excluded: set[int]
) -> int | None:
    candidates = np.flatnonzero(cell_ids == cell_ids[index])
    candidates = np.asarray(
        [candidate for candidate in candidates if int(candidate) != index and int(candidate) not in excluded],
        dtype=np.int64,
    )
    if candidates.size == 0:
        return None
    query = contexts[index]
    query_norm = np.linalg.norm(query)
    candidate_values = contexts[candidates]
    denominator = np.linalg.norm(candidate_values, axis=1) * query_norm
    similarity = np.full(candidates.size, -np.inf, dtype=np.float64)
    valid = denominator > 1.0e-12
    similarity[valid] = (candidate_values[valid] @ query) / denominator[valid]
    if not np.any(np.isfinite(similarity)):
        return int(candidates[0])
    return int(candidates[int(np.nanargmax(similarity))])


def transfer_initialize(
    contexts: np.ndarray,
    cell_ids: np.ndarray,
    current_models: Sequence[Mapping[str, torch.Tensor]],
    previous_models: Sequence[Mapping[str, torch.Tensor]],
    handover_indices: Sequence[int],
    new_ue_indices: Sequence[int],
    delta: float,
) -> tuple[dict[int, TensorState], dict[int, int | None]]:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must lie in [0, 1]")
    contexts = np.asarray(contexts, dtype=np.float64)
    cell_ids = np.asarray(cell_ids)
    new_set = {int(index) for index in new_ue_indices}
    affected = sorted({int(index) for index in handover_indices} | new_set)
    initialized: dict[int, TensorState] = {}
    neighbors: dict[int, int | None] = {}
    for index in affected:
        neighbor = _most_similar_neighbor(index, contexts, cell_ids, new_set)
        neighbors[index] = neighbor
        if neighbor is None:
            initialized[index] = clone_state(current_models[index])
        elif index in new_set:
            # Eq. (19) has no w_prev for a new UE: delta=0 is the only defined transfer.
            initialized[index] = clone_state(current_models[neighbor])
        else:
            initialized[index] = mix_states(
                [previous_models[index], current_models[neighbor]], [delta, 1.0 - delta]
            )
    return initialized, neighbors
