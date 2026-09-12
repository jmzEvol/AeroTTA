from __future__ import annotations

from collections.abc import Sequence

import torch

from .prompt_transfer_routing import (
    CORE_TIER,
    SHOULDER_TIER,
    build_frozen_anchor_tiers,
)


def replace_teacher_query(
    *,
    canonical_query_ids: Sequence[int],
    query_idx_list: Sequence[int],
    class_id: int,
    candidate_query_id: int,
) -> tuple[int, ...]:
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    mapped_classes = tuple(int(value) for value in query_idx_list)
    class_id = int(class_id)
    candidate_query_id = int(candidate_query_id)
    if class_id < 0 or class_id >= len(canonical_ids):
        raise ValueError("class id is outside the canonical query mapping")
    if (
        candidate_query_id < 0
        or candidate_query_id >= len(mapped_classes)
    ):
        raise ValueError("candidate query id is outside the query mapping")
    if mapped_classes[candidate_query_id] != class_id:
        raise ValueError(
            f"candidate query {candidate_query_id} does not belong to "
            f"class {class_id}"
        )
    output = list(canonical_ids)
    output[class_id] = candidate_query_id
    if len(set(output)) != len(output):
        raise ValueError("candidate replacement creates duplicate query ids")
    return tuple(output)


def _canonical_scores(
    query_scores: torch.Tensor,
    canonical_query_ids: Sequence[int],
) -> torch.Tensor:
    index = torch.as_tensor(
        tuple(int(value) for value in canonical_query_ids),
        device=query_scores.device,
        dtype=torch.long,
    )
    return query_scores.detach().float()[0].index_select(0, index)


def _raw_class_scores(
    query_scores: torch.Tensor,
    query_idx_list: Sequence[int],
    *,
    num_classes: int,
) -> torch.Tensor:
    scores = query_scores.detach().float()[0]
    groups = [
        [
            query_id
            for query_id, mapped_class in enumerate(query_idx_list)
            if int(mapped_class) == class_id
        ]
        for class_id in range(int(num_classes))
    ]
    if any(not group for group in groups):
        raise ValueError("query mapping does not cover every class")
    return torch.stack(
        [scores[group].amax(dim=0) for group in groups],
        dim=0,
    )


def _optional_mean(values: torch.Tensor) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(values.float().mean().item())


def _own_tier_summary(
    *,
    mask: torch.Tensor,
    class_id: int,
    pre_scores: torch.Tensor,
    post_scores: torch.Tensor,
) -> dict:
    count = int(mask.sum().item())
    if count == 0:
        return {
            "anchor_count": 0,
            "win_retention": None,
            "score_drift_mean": None,
            "margin_drift_mean": None,
        }
    foreign_ids = [
        value for value in range(int(pre_scores.shape[0]))
        if value != int(class_id)
    ]
    pre_margin = (
        pre_scores[class_id]
        - pre_scores[foreign_ids].amax(dim=0)
    )
    post_margin = (
        post_scores[class_id]
        - post_scores[foreign_ids].amax(dim=0)
    )
    return {
        "anchor_count": count,
        "win_retention": float(
            (post_scores.argmax(dim=0)[mask] == int(class_id))
            .float()
            .mean()
            .item()
        ),
        "score_drift_mean": _optional_mean(
            post_scores[class_id][mask] - pre_scores[class_id][mask]
        ),
        "margin_drift_mean": _optional_mean(
            post_margin[mask] - pre_margin[mask]
        ),
    }


def _foreign_tier_summary(
    *,
    mask: torch.Tensor,
    anchor_classes: torch.Tensor,
    class_id: int,
    pre_scores: torch.Tensor,
    post_scores: torch.Tensor,
) -> dict:
    count = int(mask.sum().item())
    if count == 0:
        return {
            "anchor_count": 0,
            "new_takeover_rate": None,
            "relieved_takeover_rate": None,
            "margin_drift_mean": None,
            "positive_erosion_mean": None,
        }
    foreign_ids = anchor_classes[mask].long()
    pre_correct = pre_scores[:, mask].gather(
        0,
        foreign_ids.unsqueeze(0),
    )[0]
    post_correct = post_scores[:, mask].gather(
        0,
        foreign_ids.unsqueeze(0),
    )[0]
    pre_margin = pre_scores[class_id][mask] - pre_correct
    post_margin = post_scores[class_id][mask] - post_correct
    margin_drift = post_margin - pre_margin
    return {
        "anchor_count": count,
        "new_takeover_rate": float(
            ((pre_margin <= 0.0) & (post_margin > 0.0))
            .float()
            .mean()
            .item()
        ),
        "relieved_takeover_rate": float(
            ((pre_margin > 0.0) & (post_margin <= 0.0))
            .float()
            .mean()
            .item()
        ),
        "margin_drift_mean": _optional_mean(margin_drift),
        "positive_erosion_mean": _optional_mean(
            (
                post_margin.clamp_min(0.0)
                - pre_margin.clamp_min(0.0)
            ).clamp_min(0.0)
        ),
    }


def measure_teacher_trial_transfer(
    *,
    pre_query_scores: torch.Tensor,
    post_query_scores: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
    class_id: int,
    prob_thd: float,
    anchor_k: int,
) -> dict:
    """Measure one candidate update on frozen, label-free Teacher anchors."""

    if tuple(pre_query_scores.shape) != tuple(post_query_scores.shape):
        raise ValueError("pre/post query score shape mismatch")
    if pre_query_scores.ndim != 4 or int(pre_query_scores.shape[0]) != 1:
        raise ValueError("query scores must have shape [1,Q,H,W]")
    if len(query_idx_list) != int(pre_query_scores.shape[1]):
        raise ValueError("query score/mapping length mismatch")
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    class_id = int(class_id)
    if class_id < 0 or class_id >= len(canonical_ids):
        raise ValueError("class id is outside the canonical query mapping")

    anchors = build_frozen_anchor_tiers(
        query_scores=pre_query_scores,
        canonical_query_ids=canonical_ids,
        prob_thd=float(prob_thd),
        anchor_k=int(anchor_k),
    )
    anchor_classes = anchors.class_ids.to(pre_query_scores.device)
    tiers = anchors.tiers.to(pre_query_scores.device)
    pre_canonical = _canonical_scores(pre_query_scores, canonical_ids)
    post_canonical = _canonical_scores(post_query_scores, canonical_ids)
    valid_anchors = anchor_classes >= 0

    own_core_mask = (
        (anchor_classes == class_id) & (tiers == CORE_TIER)
    )
    own_shoulder_mask = (
        (anchor_classes == class_id) & (tiers == SHOULDER_TIER)
    )
    foreign_core_mask = (
        (anchor_classes >= 0)
        & (anchor_classes != class_id)
        & (tiers == CORE_TIER)
    )
    foreign_shoulder_mask = (
        (anchor_classes >= 0)
        & (anchor_classes != class_id)
        & (tiers == SHOULDER_TIER)
    )

    pre_raw = _raw_class_scores(
        pre_query_scores,
        query_idx_list,
        num_classes=len(canonical_ids),
    )
    post_raw = _raw_class_scores(
        post_query_scores,
        query_idx_list,
        num_classes=len(canonical_ids),
    )
    pre_raw_pred = pre_raw.argmax(dim=0)
    post_raw_pred = post_raw.argmax(dim=0)
    anchor_count = int(valid_anchors.sum().item())
    return {
        "anchor_count": anchor_count,
        "own_core": _own_tier_summary(
            mask=own_core_mask,
            class_id=class_id,
            pre_scores=pre_canonical,
            post_scores=post_canonical,
        ),
        "own_shoulder": _own_tier_summary(
            mask=own_shoulder_mask,
            class_id=class_id,
            pre_scores=pre_canonical,
            post_scores=post_canonical,
        ),
        "foreign_core": _foreign_tier_summary(
            mask=foreign_core_mask,
            anchor_classes=anchor_classes,
            class_id=class_id,
            pre_scores=pre_canonical,
            post_scores=post_canonical,
        ),
        "foreign_shoulder": _foreign_tier_summary(
            mask=foreign_shoulder_mask,
            anchor_classes=anchor_classes,
            class_id=class_id,
            pre_scores=pre_canonical,
            post_scores=post_canonical,
        ),
        "canonical_anchor_agreement": (
            float(
                (
                    post_canonical.argmax(dim=0)[valid_anchors]
                    == anchor_classes[valid_anchors]
                )
                .float()
                .mean()
                .item()
            )
            if anchor_count
            else None
        ),
        "raw_anchor_agreement": (
            float(
                (post_raw_pred[valid_anchors] == pre_raw_pred[valid_anchors])
                .float()
                .mean()
                .item()
            )
            if anchor_count
            else None
        ),
        "raw_class_expansion_rate": (
            float(
                (
                    (pre_raw_pred[valid_anchors] != class_id)
                    & (post_raw_pred[valid_anchors] == class_id)
                )
                .float()
                .mean()
                .item()
            )
            if anchor_count
            else None
        ),
        "raw_class_contraction_rate": (
            float(
                (
                    (pre_raw_pred[valid_anchors] == class_id)
                    & (post_raw_pred[valid_anchors] != class_id)
                )
                .float()
                .mean()
                .item()
            )
            if anchor_count
            else None
        ),
    }
