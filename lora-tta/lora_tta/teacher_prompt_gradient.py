from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def flatten_parameter_gradients(
    parameters: Sequence[torch.Tensor],
    gradients: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    if len(parameters) != len(gradients):
        raise ValueError("parameter/gradient length mismatch")
    if not parameters:
        raise ValueError("parameters must not be empty")
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            flattened.append(torch.zeros_like(parameter).flatten())
        else:
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise ValueError("parameter/gradient shape mismatch")
            flattened.append(
                gradient.detach().to(
                    device=parameter.device,
                    dtype=torch.float32,
                ).flatten()
            )
    return torch.cat(flattened)


def gradient_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float | None:
    if int(left.numel()) != int(right.numel()):
        raise ValueError("gradient vectors must have the same length")
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    if float(denominator.item()) <= float(epsilon):
        return None
    return float(torch.dot(left, right).div(denominator).item())


def _sign_agreement(
    left: torch.Tensor,
    right: torch.Tensor,
) -> float | None:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    active = (left != 0.0) | (right != 0.0)
    if not bool(active.any()):
        return None
    return float(
        (torch.sign(left[active]) == torch.sign(right[active]))
        .float()
        .mean()
        .item()
    )


def _unit(vector: torch.Tensor, *, epsilon: float) -> torch.Tensor | None:
    vector = vector.detach().float().flatten()
    norm = torch.linalg.vector_norm(vector)
    if float(norm.item()) <= float(epsilon):
        return None
    return vector / norm


def summarize_prompt_gradient_vectors(
    gradients_by_query: Mapping[int, torch.Tensor],
    *,
    stable_gradient: torch.Tensor | None,
    epsilon: float = 1e-12,
) -> dict:
    if not gradients_by_query:
        raise ValueError("at least one candidate gradient is required")
    query_ids = sorted(int(value) for value in gradients_by_query)
    vectors = {
        int(query_id): gradients_by_query[query_id].detach().float().flatten()
        for query_id in query_ids
    }
    lengths = {int(vector.numel()) for vector in vectors.values()}
    if len(lengths) != 1:
        raise ValueError("candidate gradient vectors must have equal length")
    if stable_gradient is not None:
        stable_gradient = stable_gradient.detach().float().flatten()
        if int(stable_gradient.numel()) != next(iter(lengths)):
            raise ValueError("stable gradient length mismatch")

    units = {
        query_id: _unit(vector, epsilon=epsilon)
        for query_id, vector in vectors.items()
    }
    active_units = [unit for unit in units.values() if unit is not None]
    consensus = (
        torch.stack(active_units).mean(dim=0)
        if active_units
        else None
    )
    stable_norm = (
        float(torch.linalg.vector_norm(stable_gradient).item())
        if stable_gradient is not None
        else None
    )

    pairwise_by_query: dict[int, list[float]] = {
        query_id: [] for query_id in query_ids
    }
    pairs = []
    for left_index, left_query_id in enumerate(query_ids):
        for right_query_id in query_ids[left_index + 1 :]:
            cosine = gradient_cosine(
                vectors[left_query_id],
                vectors[right_query_id],
                epsilon=epsilon,
            )
            pairs.append(
                {
                    "left_query_id": int(left_query_id),
                    "right_query_id": int(right_query_id),
                    "cosine": cosine,
                }
            )
            if cosine is not None:
                pairwise_by_query[left_query_id].append(cosine)
                pairwise_by_query[right_query_id].append(cosine)

    candidates = {}
    for query_id in query_ids:
        vector = vectors[query_id]
        pairwise = pairwise_by_query[query_id]
        stable_cosine = (
            gradient_cosine(vector, stable_gradient, epsilon=epsilon)
            if stable_gradient is not None
            else None
        )
        consensus_cosine = (
            gradient_cosine(vector, consensus, epsilon=epsilon)
            if consensus is not None
            else None
        )
        vector_norm = float(torch.linalg.vector_norm(vector).item())
        candidates[query_id] = {
            "gradient_norm": vector_norm,
            "stable_norm_ratio": (
                vector_norm / stable_norm
                if stable_norm is not None and stable_norm > float(epsilon)
                else None
            ),
            "stable_cosine": stable_cosine,
            "stable_conflict": (
                max(-stable_cosine, 0.0)
                if stable_cosine is not None
                else None
            ),
            "stable_sign_agreement": (
                _sign_agreement(vector, stable_gradient)
                if stable_gradient is not None
                else None
            ),
            "consensus_cosine": consensus_cosine,
            "consensus_sign_agreement": (
                _sign_agreement(vector, consensus)
                if consensus is not None
                else None
            ),
            "mean_pairwise_cosine": (
                sum(pairwise) / len(pairwise) if pairwise else None
            ),
            "min_pairwise_cosine": min(pairwise) if pairwise else None,
            "positive_pair_fraction": (
                sum(value > 0.0 for value in pairwise) / len(pairwise)
                if pairwise
                else None
            ),
        }
    return {
        "stable_gradient_norm": stable_norm,
        "consensus_gradient_norm": (
            float(torch.linalg.vector_norm(consensus).item())
            if consensus is not None
            else None
        ),
        "candidates": candidates,
        "pairs": pairs,
    }
