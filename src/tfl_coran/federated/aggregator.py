from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

TensorState = dict[str, torch.Tensor]


def clone_state(state: Mapping[str, torch.Tensor]) -> TensorState:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def mix_states(states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[float]) -> TensorState:
    if len(states) == 0 or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and have equal length")
    coefficients = np.asarray(weights, dtype=np.float64)
    total = float(coefficients.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a finite positive sum")
    coefficients = coefficients / total
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("all state dictionaries must have identical ordered keys")
    mixed: TensorState = {}
    for key in keys:
        reference = states[0][key]
        if not torch.is_floating_point(reference):
            mixed[key] = reference.detach().cpu().clone()
            continue
        value = torch.zeros_like(reference, device="cpu")
        for coefficient, state in zip(coefficients, states, strict=True):
            value.add_(state[key].detach().cpu(), alpha=float(coefficient))
        mixed[key] = value
    return mixed


def _subtract(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> TensorState:
    return {name: left[name].detach().cpu() - right[name].detach().cpu() for name in left}


def _add(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> TensorState:
    return {name: left[name].detach().cpu() + right[name].detach().cpu() for name in left}


@dataclass(frozen=True)
class AggregationResult:
    global_model: TensorState
    cluster_models: list[TensorState]
    personalized_models: list[TensorState]
    cluster_mass: np.ndarray


def aggregate_models(
    local_models: Sequence[Mapping[str, torch.Tensor]],
    memberships: np.ndarray,
    previous_global: Mapping[str, torch.Tensor],
    dispatch_bases: Sequence[Mapping[str, torch.Tensor]] | None = None,
    delta_reference: str = "dispatch_base",
) -> AggregationResult:
    """Implement paper Eqs. (15)-(17), with an explicit delta-base choice.

    ``paper_global`` literally subtracts the previous shared global model.
    ``dispatch_base`` (default) averages post-local absolute models, which is
    coherent when clients started from different personalized broadcasts.
    """
    gamma = np.asarray(memberships, dtype=np.float64)
    if gamma.ndim != 2 or gamma.shape[0] != len(local_models):
        raise ValueError("memberships must have shape [number_of_clients, K]")
    if np.any(gamma < -1.0e-12) or not np.all(np.isfinite(gamma)):
        raise ValueError("memberships must be finite and non-negative")
    row_sums = gamma.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("every client needs positive membership mass")
    gamma = gamma / row_sums
    n, k_count = gamma.shape
    cluster_models: list[TensorState] = []
    for cluster in range(k_count):
        weights = gamma[:, cluster]
        if float(weights.sum()) <= 1.0e-12:
            cluster_models.append(clone_state(previous_global))
            continue
        if delta_reference == "paper_global":
            deltas = [_subtract(model, previous_global) for model in local_models]
            cluster_delta = mix_states(deltas, weights)
            cluster_models.append(_add(previous_global, cluster_delta))
        elif delta_reference == "dispatch_base":
            # Algebraically equivalent to aggregating local deltas around their
            # own dispatch bases and then restoring those weighted bases.
            if dispatch_bases is not None and len(dispatch_bases) != n:
                raise ValueError("dispatch_bases must match local_models")
            cluster_models.append(mix_states(local_models, weights))
        else:
            raise ValueError("delta_reference must be 'paper_global' or 'dispatch_base'")
    cluster_mass = gamma.mean(axis=0)
    global_model = mix_states(cluster_models, cluster_mass)
    personalized = [mix_states(cluster_models, gamma[index]) for index in range(n)]
    return AggregationResult(global_model, cluster_models, personalized, cluster_mass)


def serialized_parameter_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in state.values()))
