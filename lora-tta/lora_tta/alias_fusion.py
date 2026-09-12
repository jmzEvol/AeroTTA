from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class AliasReliabilityRecord:
    query_id: int
    class_id: int
    own_anchor_count: int
    foreign_anchor_count: int
    auc: float
    separation: float
    foreign_leakage: float
    reliability: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AliasReliabilityResult:
    query_weights: torch.Tensor
    anchor_counts: tuple[int, ...]
    records: tuple[AliasReliabilityRecord, ...]


def binary_auc(positives: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
    """Compute tie-correct Mann-Whitney AUC without pairwise expansion."""
    positives = positives.detach().float().flatten()
    negatives = negatives.detach().float().flatten()
    if int(positives.numel()) == 0 or int(negatives.numel()) == 0:
        raise ValueError("binary_auc requires non-empty positive and negative samples")

    values = torch.cat([positives, negatives])
    labels = torch.cat(
        [
            torch.ones_like(positives, dtype=torch.float32),
            torch.zeros_like(negatives, dtype=torch.float32),
        ]
    )
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_labels = labels[order]
    _unique, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    positive_counts = torch.zeros(
        int(counts.numel()),
        device=values.device,
        dtype=torch.float64,
    )
    positive_counts.scatter_add_(0, inverse, sorted_labels.to(torch.float64))
    negative_counts = counts.to(torch.float64) - positive_counts
    negatives_before = negative_counts.cumsum(0) - negative_counts
    favorable_pairs = (
        positive_counts * (negatives_before + 0.5 * negative_counts)
    ).sum()
    total_pairs = float(int(positives.numel()) * int(negatives.numel()))
    return (favorable_pairs / total_pairs).to(dtype=torch.float32)


def _validate_prompt_layout(
    *,
    query_scores: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    canonical_query_ids: tuple[int, ...] | list[int],
    num_classes: int,
) -> tuple[int, int, int, int]:
    if query_scores.ndim != 4:
        raise ValueError(
            f"expected query_scores [B,Q,H,W], got {tuple(query_scores.shape)}"
        )
    batch_size, num_queries, height, width = (int(v) for v in query_scores.shape)
    if len(query_idx_list) != num_queries:
        raise ValueError("query mapping/query score length mismatch")
    if len(canonical_query_ids) != int(num_classes):
        raise ValueError("canonical_query_ids must contain one query id per class")
    for class_id, query_id_value in enumerate(canonical_query_ids):
        query_id = int(query_id_value)
        if query_id < 0 or query_id >= num_queries:
            raise ValueError(
                f"canonical query id {query_id} for class {class_id} exceeds "
                f"the query range [0, {num_queries - 1}]"
            )
        if int(query_idx_list[query_id]) != class_id:
            raise ValueError(
                f"canonical query id {query_id} does not map to class {class_id}"
            )
    mapped_classes = {int(value) for value in query_idx_list}
    expected_classes = set(range(int(num_classes)))
    if mapped_classes != expected_classes:
        raise ValueError("query mapping must cover exactly the configured classes")
    return batch_size, num_queries, height, width


def build_canonical_topk_anchors(
    *,
    query_scores: torch.Tensor,
    query_presence: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    canonical_query_ids: tuple[int, ...] | list[int],
    num_classes: int,
    prob_thd: float,
    tau_pos: float,
    rho: float,
    kmax: int,
    n_min: int,
    bg_idx: int,
) -> tuple[torch.Tensor, ...]:
    """Build disjoint canonical anchors using the existing TopK policy."""
    batch_size, num_queries, height, width = _validate_prompt_layout(
        query_scores=query_scores,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
    )
    if batch_size != 1:
        raise ValueError("alias reliability currently requires batch size one")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if int(kmax) <= 0 or int(n_min) <= 0:
        raise ValueError("kmax and n_min must be positive")

    presence = query_presence.detach().float()
    if presence.ndim == 2 and int(presence.shape[0]) == 1:
        presence = presence[0]
    if presence.ndim != 1 or int(presence.numel()) != num_queries:
        raise ValueError("query_presence must have shape [Q] or [1,Q]")

    canonical_index = torch.as_tensor(
        canonical_query_ids,
        device=query_scores.device,
        dtype=torch.long,
    )
    canonical_scores = query_scores.detach().float()[0].index_select(0, canonical_index)
    canonical_presence = presence.to(query_scores.device).index_select(0, canonical_index)
    empty = torch.empty(0, device=query_scores.device, dtype=torch.long)
    preselected: list[torch.Tensor] = [empty for _ in range(int(num_classes))]

    for class_id in range(int(num_classes)):
        is_present = float(canonical_presence[class_id].item()) > float(tau_pos)
        if class_id == int(bg_idx):
            is_present = True
        if not is_present:
            continue
        score = canonical_scores[class_id].flatten()
        candidates = (score >= float(prob_thd)).nonzero(as_tuple=False).flatten()
        candidate_count = int(candidates.numel())
        if candidate_count < int(n_min):
            continue
        topk_count = max(int(n_min), int(float(rho) * candidate_count), 1)
        topk_count = min(topk_count, int(kmax), candidate_count)
        rank = torch.topk(score[candidates], k=topk_count, largest=True).indices
        preselected[class_id] = candidates[rank]

    nonempty = [indices for indices in preselected if int(indices.numel()) > 0]
    if nonempty:
        counts = torch.bincount(
            torch.cat(nonempty),
            minlength=height * width,
        )
        preselected = [
            indices[counts[indices] == 1]
            if int(indices.numel()) > 0
            else indices
            for indices in preselected
        ]

    return tuple(
        indices.detach() if int(indices.numel()) >= int(n_min) else empty
        for indices in preselected
    )


def estimate_alias_reliability(
    *,
    query_scores: torch.Tensor,
    query_presence: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    canonical_query_ids: tuple[int, ...] | list[int],
    num_classes: int,
    prob_thd: float,
    tau_pos: float,
    rho: float,
    kmax: int,
    n_min: int,
    bg_idx: int,
) -> AliasReliabilityResult:
    """Estimate frozen, image-level alias weights from canonical anchors only."""
    _batch_size, num_queries, _height, _width = _validate_prompt_layout(
        query_scores=query_scores,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
    )
    anchors = build_canonical_topk_anchors(
        query_scores=query_scores,
        query_presence=query_presence,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
        prob_thd=prob_thd,
        tau_pos=tau_pos,
        rho=rho,
        kmax=kmax,
        n_min=n_min,
        bg_idx=bg_idx,
    )
    scores = query_scores.detach().float()[0]
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    canonical_scores = scores.index_select(
        0,
        torch.as_tensor(canonical_ids, device=scores.device, dtype=torch.long),
    )
    weights = torch.zeros(num_queries, device=scores.device, dtype=torch.float32)
    weights[list(canonical_ids)] = 1.0
    records = []

    for query_id, mapped_class in enumerate(query_idx_list):
        class_id = int(mapped_class)
        if query_id == canonical_ids[class_id]:
            continue
        own_indices = anchors[class_id]
        foreign_classes = [
            other_class
            for other_class in range(int(num_classes))
            if other_class != class_id and int(anchors[other_class].numel()) > 0
        ]
        own_count = int(own_indices.numel())
        foreign_count = int(sum(int(anchors[c].numel()) for c in foreign_classes))
        auc_value = 0.5
        separation = 0.0
        leakage = 0.0
        reliability = 0.0
        if own_count > 0 and foreign_count > 0:
            alias_flat = scores[query_id].flatten()
            own_values = alias_flat[own_indices]
            foreign_values = torch.cat(
                [alias_flat[anchors[other_class]] for other_class in foreign_classes]
            )
            foreign_reference = torch.cat(
                [
                    canonical_scores[other_class].flatten()[anchors[other_class]]
                    for other_class in foreign_classes
                ]
            )
            auc = binary_auc(own_values, foreign_values)
            separation_tensor = (2.0 * auc - 1.0).clamp(0.0, 1.0)
            leakage_tensor = (foreign_values > foreign_reference).float().mean()
            reliability_tensor = separation_tensor * (1.0 - leakage_tensor)
            auc_value = float(auc.item())
            separation = float(separation_tensor.item())
            leakage = float(leakage_tensor.item())
            reliability = float(reliability_tensor.item())
            weights[query_id] = reliability_tensor
        records.append(
            AliasReliabilityRecord(
                query_id=int(query_id),
                class_id=class_id,
                own_anchor_count=own_count,
                foreign_anchor_count=foreign_count,
                auc=auc_value,
                separation=separation,
                foreign_leakage=leakage,
                reliability=reliability,
            )
        )

    return AliasReliabilityResult(
        query_weights=weights.detach(),
        anchor_counts=tuple(int(indices.numel()) for indices in anchors),
        records=tuple(records),
    )


def class_scores_from_query_scores_with_alias_reliability(
    *,
    query_scores: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
    canonical_query_ids: tuple[int, ...] | list[int],
    query_weights: torch.Tensor,
) -> torch.Tensor:
    """Fuse aliases as weighted positive residuals over each canonical map."""
    _batch_size, num_queries, _height, _width = _validate_prompt_layout(
        query_scores=query_scores,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
    )
    weights = query_weights.detach().to(device=query_scores.device, dtype=torch.float32).flatten()
    if int(weights.numel()) != num_queries:
        raise ValueError("query_weights must contain one value per query")
    if bool((~torch.isfinite(weights)).any()) or bool(((weights < 0.0) | (weights > 1.0)).any()):
        raise ValueError("query_weights must be finite values in [0, 1]")

    scores = query_scores.float()
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    canonical_index = torch.as_tensor(
        canonical_ids,
        device=scores.device,
        dtype=torch.long,
    )
    canonical_scores = scores.index_select(1, canonical_index)
    output = canonical_scores.clone()
    for query_id, mapped_class in enumerate(query_idx_list):
        class_id = int(mapped_class)
        if query_id == canonical_ids[class_id]:
            continue
        residual = (scores[:, query_id] - canonical_scores[:, class_id]).clamp_min(0.0)
        candidate = canonical_scores[:, class_id] + weights[query_id] * residual
        output[:, class_id] = torch.maximum(output[:, class_id], candidate)
    return output


def class_scores_from_query_scores_with_frozen_alias_residual(
    *,
    query_scores: torch.Tensor,
    reference_query_scores: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
    canonical_query_ids: tuple[int, ...] | list[int],
    query_weights: torch.Tensor,
) -> torch.Tensor:
    """Add reliable teacher alias residuals to current canonical maps."""
    batch_size, num_queries, height, width = _validate_prompt_layout(
        query_scores=query_scores,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
    )
    reference_shape = _validate_prompt_layout(
        query_scores=reference_query_scores,
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
        num_classes=num_classes,
    )
    if reference_shape != (batch_size, num_queries, height, width):
        raise ValueError("reference_query_scores must match query_scores shape")

    weights = query_weights.detach().to(
        device=query_scores.device,
        dtype=torch.float32,
    ).flatten()
    if int(weights.numel()) != num_queries:
        raise ValueError("query_weights must contain one value per query")
    if bool((~torch.isfinite(weights)).any()) or bool(
        ((weights < 0.0) | (weights > 1.0)).any()
    ):
        raise ValueError("query_weights must be finite values in [0, 1]")

    scores = query_scores.float()
    reference = reference_query_scores.detach().to(
        device=scores.device,
        dtype=torch.float32,
    )
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    canonical_index = torch.as_tensor(
        canonical_ids,
        device=scores.device,
        dtype=torch.long,
    )
    current_canonical = scores.index_select(1, canonical_index)
    reference_canonical = reference.index_select(1, canonical_index)
    output = current_canonical.clone()
    for query_id, mapped_class in enumerate(query_idx_list):
        class_id = int(mapped_class)
        if query_id == canonical_ids[class_id]:
            continue
        residual = (
            reference[:, query_id] - reference_canonical[:, class_id]
        ).clamp_min(0.0)
        candidate = current_canonical[:, class_id] + weights[query_id] * residual
        output[:, class_id] = torch.maximum(output[:, class_id], candidate)
    return output.clamp(0.0, 1.0)
