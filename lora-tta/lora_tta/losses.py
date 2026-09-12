from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import torch
import torch.nn.functional as F


@dataclass
class SelectedClassEntry:
    flat_idx: torch.Tensor
    weights: torch.Tensor
    class_weight: torch.Tensor
    class_reliability: torch.Tensor | None = None
    positive_targets: torch.Tensor | None = None
    positive_loss_scales: torch.Tensor | None = None
    negative_flat_idx: torch.Tensor | None = None
    negative_weights: torch.Tensor | None = None
    negative_targets: torch.Tensor | None = None
    mask_loss_scale: float = 1.0


class SelectedBCETerms(NamedTuple):
    positive: torch.Tensor
    negative: torch.Tensor


def _as_class_score_maps(scores: torch.Tensor, *, name: str) -> torch.Tensor:
    if scores.ndim == 4 and int(scores.shape[0]) == 1:
        scores = scores[0]
    if scores.ndim != 3:
        raise ValueError(
            f"expected {name} [C,H,W], got {tuple(scores.shape)}"
        )
    return scores.float()


def apply_decision_boundary_contraction_gate(
    selected_by_class: dict[int, SelectedClassEntry],
    *,
    student_class_scores: torch.Tensor,
    final_class_scores: torch.Tensor,
    final_prediction: torch.Tensor | None = None,
) -> tuple[dict[int, SelectedClassEntry], dict]:
    student_scores = _as_class_score_maps(
        student_class_scores,
        name="student_class_scores",
    )
    final_scores = _as_class_score_maps(
        final_class_scores,
        name="final_class_scores",
    )
    if tuple(student_scores.shape) != tuple(final_scores.shape):
        raise ValueError("student and final class score shapes must match")
    if final_prediction is None:
        final_prediction = final_scores.argmax(dim=0)
    if final_prediction.ndim == 3 and int(final_prediction.shape[0]) == 1:
        final_prediction = final_prediction[0]
    if tuple(final_prediction.shape) != tuple(final_scores.shape[-2:]):
        raise ValueError("final prediction shape must match class scores")
    final_prediction = final_prediction.to(
        device=final_scores.device,
        dtype=torch.long,
    )
    probability_floor = 1e-6
    final_logits = torch.logit(
        final_scores.clamp(probability_floor, 1.0 - probability_floor)
    )

    num_classes = int(final_scores.shape[0])
    gated_entries: dict[int, SelectedClassEntry] = {}
    per_class = {}
    total_selected = 0
    total_contracting = 0
    total_gated = 0
    scale_sum = 0.0
    for class_id, entry in selected_by_class.items():
        class_id = int(class_id)
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"selected class {class_id} is out of range")
        flat_idx = entry.flat_idx.to(
            device=student_scores.device,
            dtype=torch.long,
        )
        count = int(flat_idx.numel())
        scales = student_scores.new_ones((count,))
        contracting = torch.zeros_like(scales, dtype=torch.bool)
        support_count = 0
        requested_logit_drop_mean = 0.0
        support_scale = 1.0
        support_margin_mean = 0.0
        support_margin_q01 = 0.0
        support_margin_q05 = 0.0
        support_margin_q10 = 0.0
        at_risk_support_fraction = 0.0
        risk = False
        saturated_contracting_count = 0
        if count > 0 and entry.positive_targets is not None:
            student_values = student_scores[class_id].flatten()[flat_idx]
            targets = entry.positive_targets.to(
                device=student_scores.device,
                dtype=student_values.dtype,
            )
            if tuple(targets.shape) != tuple(student_values.shape):
                raise ValueError(
                    "positive target shape must match selected pixels"
                )
            student_logits = torch.logit(
                student_values.clamp(
                    probability_floor,
                    1.0 - probability_floor,
                )
            )
            target_logits = torch.logit(
                targets.clamp(
                    probability_floor,
                    1.0 - probability_floor,
                )
            )
            requested_logit_drop = (
                student_logits - target_logits
            ).clamp_min(0.0)
            contracting = requested_logit_drop.gt(0.0)
            if num_classes > 1 and bool(contracting.any()):
                other_ids = [
                    index for index in range(num_classes)
                    if index != class_id
                ]
                other_index = torch.tensor(
                    other_ids,
                    device=final_scores.device,
                    dtype=torch.long,
                )
                pixel_weights = entry.weights.to(
                    device=student_scores.device,
                    dtype=student_values.dtype,
                )
                contraction_weights = pixel_weights[contracting]
                class_requested_logit_drop = (
                    requested_logit_drop[contracting] * contraction_weights
                ).sum() / contraction_weights.sum().clamp_min(1e-6)
                requested_logit_drop_mean = float(
                    class_requested_logit_drop.item()
                )
                support = final_prediction == class_id
                support_count = int(support.sum().item())
                class_values = final_logits[class_id]
                runner_values = (
                    final_logits.index_select(0, other_index)
                    .amax(dim=0)
                )
                if support_count > 0:
                    support_margins = (
                        class_values[support] - runner_values[support]
                    ).clamp_min(0.0)
                    support_margin_mean = float(
                        support_margins.mean().item()
                    )
                    support_quantiles = torch.quantile(
                        support_margins,
                        support_margins.new_tensor([0.01, 0.05, 0.10]),
                    )
                    support_margin_q01 = float(support_quantiles[0].item())
                    support_margin_q05 = float(support_quantiles[1].item())
                    support_margin_q10 = float(support_quantiles[2].item())
                    at_risk_support_fraction = float(
                        support_margins.lt(class_requested_logit_drop)
                        .float()
                        .mean()
                        .item()
                    )
                    class_support_scale = (
                        support_margins
                        / class_requested_logit_drop.clamp_min(1e-6)
                    ).clamp(0.0, 1.0).mean()
                    support_scale = float(class_support_scale.item())
                    risk = bool(
                        class_requested_logit_drop
                        > support_margins.mean()
                    )
                    saturated_contracting = contracting & student_values.ge(
                        1.0 - probability_floor
                    )
                    saturated_contracting_count = int(
                        saturated_contracting.sum().item()
                    )
                    if risk:
                        scales[saturated_contracting] = 0.0
        if entry.positive_loss_scales is not None:
            existing = entry.positive_loss_scales.to(
                device=scales.device,
                dtype=scales.dtype,
            )
            if tuple(existing.shape) != tuple(scales.shape):
                raise ValueError(
                    "positive loss scale shape must match selected pixels"
                )
            scales = scales * existing
        scales = scales.detach()
        gated = scales.lt(1.0 - 1e-6)
        contracting_count = int(contracting.sum().item())
        gated_count = int(gated.sum().item())
        gated_entries[class_id] = replace(
            entry,
            positive_loss_scales=scales,
        )
        per_class[str(class_id)] = {
            "selected_pixels": count,
            "contracting_pixels": contracting_count,
            "gated_pixels": gated_count,
            "mean_scale": float(
                scales.mean().item() if count > 0 else 1.0
            ),
            "support_pixels": support_count,
            "mean_requested_logit_drop": requested_logit_drop_mean,
            "support_scale": support_scale,
            "risk": risk,
            "saturated_contracting_pixels": saturated_contracting_count,
            "support_margin_mean": support_margin_mean,
            "support_margin_q01": support_margin_q01,
            "support_margin_q05": support_margin_q05,
            "support_margin_q10": support_margin_q10,
            "at_risk_support_fraction": at_risk_support_fraction,
        }
        total_selected += count
        total_contracting += contracting_count
        total_gated += gated_count
        scale_sum += float(scales.sum().item())
    return gated_entries, {
        "mode": "decision_boundary",
        "selected_pixels": total_selected,
        "contracting_pixels": total_contracting,
        "gated_pixels": total_gated,
        "mean_scale": float(
            scale_sum / total_selected if total_selected > 0 else 1.0
        ),
        "per_class": per_class,
    }


def apply_mask_loss_scale(
    selected_by_class: dict[int, SelectedClassEntry],
    *,
    class_ids,
    scale: float,
) -> dict[int, SelectedClassEntry]:
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("mask loss scale must be in [0, 1]")
    scaled_ids = {int(class_id) for class_id in class_ids}
    return {
        int(class_id): replace(
            entry,
            mask_loss_scale=(
                float(entry.mask_loss_scale) * scale
                if int(class_id) in scaled_ids
                else float(entry.mask_loss_scale)
            ),
        )
        for class_id, entry in selected_by_class.items()
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def selected_class_soft_bce_loss(
    class_logits: torch.Tensor,
    selected_by_class: dict[int, SelectedClassEntry],
    *,
    low_score_neg_weight: float = 0.0,
) -> torch.Tensor:
    """Selected-class one-vs-rest BCE used by the clean TTA path.

    `class_logits` must be [1, C, H, W].  For each selected class we optimize
    only that class logit on its selected pixels, matching the current
    classwise-BCE TTA objective.
    """
    total, _terms = selected_class_soft_bce_loss_terms(
        class_logits,
        selected_by_class,
        low_score_neg_weight=low_score_neg_weight,
    )
    return total


def selected_class_soft_bce_loss_terms(
    class_logits: torch.Tensor,
    selected_by_class: dict[int, SelectedClassEntry],
    *,
    low_score_neg_weight: float = 0.0,
) -> tuple[torch.Tensor, SelectedBCETerms]:
    if class_logits.ndim != 4 or int(class_logits.shape[0]) != 1:
        raise ValueError(f"expected class_logits [1,C,H,W], got {tuple(class_logits.shape)}")

    num_classes = int(class_logits.shape[1])
    logits_flat = class_logits[0].reshape(num_classes, -1)
    positive_losses = []
    positive_class_weights = []
    negative_losses = []
    negative_class_weights = []

    for cls_idx, entry in selected_by_class.items():
        cls_idx = int(cls_idx)
        flat_idx = entry.flat_idx.to(device=class_logits.device, dtype=torch.long)
        if int(flat_idx.numel()) == 0:
            continue

        cls_logits = logits_flat[cls_idx, flat_idx]
        target = entry.positive_targets
        if target is None:
            target = torch.ones_like(cls_logits)
        else:
            target = target.to(device=class_logits.device, dtype=cls_logits.dtype)
        per_pixel = F.binary_cross_entropy_with_logits(
            cls_logits.float(),
            target.float(),
            reduction="none",
        ).to(dtype=class_logits.dtype)
        weights = entry.weights.to(device=class_logits.device, dtype=per_pixel.dtype)
        class_weight = entry.class_weight.to(device=class_logits.device, dtype=per_pixel.dtype)
        class_reliability = entry.class_reliability
        if class_reliability is None:
            class_reliability = class_weight.new_tensor(1.0)
        else:
            class_reliability = class_reliability.to(
                device=class_logits.device,
                dtype=per_pixel.dtype,
            )
        mask_loss_scale = class_weight.new_tensor(
            float(entry.mask_loss_scale)
        )
        effective_class_weight = (
            class_weight * class_reliability * mask_loss_scale
        )
        positive_loss_scales = entry.positive_loss_scales
        if positive_loss_scales is None:
            positive_loss_scales = torch.ones_like(per_pixel)
        else:
            positive_loss_scales = positive_loss_scales.to(
                device=class_logits.device,
                dtype=per_pixel.dtype,
            )
        positive_loss = (
            per_pixel * weights * positive_loss_scales
        ).sum() / weights.sum().clamp_min(1e-6)
        positive_losses.append(positive_loss * effective_class_weight)
        positive_class_weights.append(class_weight)

        if low_score_neg_weight > 0.0 and entry.negative_flat_idx is not None:
            neg_idx = entry.negative_flat_idx.to(device=class_logits.device, dtype=torch.long)
            if int(neg_idx.numel()) > 0:
                neg_logits = logits_flat[cls_idx, neg_idx]
                neg_target = entry.negative_targets
                if neg_target is None:
                    neg_target = torch.zeros_like(neg_logits)
                else:
                    neg_target = neg_target.to(device=class_logits.device, dtype=neg_logits.dtype)
                neg_per = F.binary_cross_entropy_with_logits(
                    neg_logits.float(),
                    neg_target.float(),
                    reduction="none",
                ).to(dtype=class_logits.dtype)
                neg_weights = entry.negative_weights
                if neg_weights is None:
                    neg_weights = torch.ones_like(neg_per)
                else:
                    neg_weights = neg_weights.to(device=class_logits.device, dtype=neg_per.dtype)
                negative_losses.append(
                    weighted_mean(neg_per, neg_weights) * effective_class_weight
                )
                negative_class_weights.append(class_weight)

    if not positive_losses:
        zero = class_logits.sum() * 0.0
        return zero, SelectedBCETerms(positive=zero, negative=zero)
    selected_loss = (
        torch.stack(positive_losses).sum()
        / torch.stack(positive_class_weights).sum().clamp_min(1e-6)
    )
    neg_loss = class_logits.sum() * 0.0
    if negative_losses:
        neg_loss = (
            torch.stack(negative_losses).sum()
            / torch.stack(negative_class_weights).sum().clamp_min(1e-6)
        )
    total = selected_loss + float(low_score_neg_weight) * neg_loss
    return total, SelectedBCETerms(
        positive=selected_loss,
        negative=neg_loss,
    )


def selected_query_soft_bce_loss(
    query_logits: torch.Tensor,
    selected_by_query: dict[int, SelectedClassEntry],
    *,
    query_ids: tuple[int, ...],
    query_idx_list: list[int] | tuple[int, ...],
    low_score_neg_weight: float = 0.0,
) -> torch.Tensor:
    """Apply each prompt's own BCE, then balance prompts within semantic classes."""
    if query_logits.ndim != 4 or int(query_logits.shape[0]) != 1:
        raise ValueError(
            f"expected query_logits [1,Q,H,W], got {tuple(query_logits.shape)}"
        )
    if len(query_ids) != int(query_logits.shape[1]):
        raise ValueError("query id/logit channel length mismatch")

    channel_by_query_id = {
        int(query_id): channel for channel, query_id in enumerate(query_ids)
    }
    logits_flat = query_logits[0].reshape(int(query_logits.shape[1]), -1)
    losses_by_class: dict[int, list[torch.Tensor]] = {}
    weights_by_class: dict[int, list[torch.Tensor]] = {}

    for query_id, entry in selected_by_query.items():
        query_id = int(query_id)
        if query_id not in channel_by_query_id:
            raise ValueError(f"selected query {query_id} is missing from query_ids")
        if query_id < 0 or query_id >= len(query_idx_list):
            raise ValueError(f"selected query {query_id} has no class mapping")

        flat_idx = entry.flat_idx.to(device=query_logits.device, dtype=torch.long)
        if int(flat_idx.numel()) == 0:
            continue
        channel = channel_by_query_id[query_id]
        prompt_logits = logits_flat[channel, flat_idx]
        target = entry.positive_targets
        if target is None:
            target = torch.ones_like(prompt_logits)
        else:
            target = target.to(device=query_logits.device, dtype=prompt_logits.dtype)
        per_pixel = F.binary_cross_entropy_with_logits(
            prompt_logits.float(),
            target.float(),
            reduction="none",
        ).to(dtype=query_logits.dtype)
        pixel_weights = entry.weights.to(
            device=query_logits.device,
            dtype=per_pixel.dtype,
        )
        positive_loss_scales = entry.positive_loss_scales
        if positive_loss_scales is None:
            positive_loss_scales = torch.ones_like(per_pixel)
        else:
            positive_loss_scales = positive_loss_scales.to(
                device=query_logits.device,
                dtype=per_pixel.dtype,
            )
        prompt_loss = (
            per_pixel * pixel_weights * positive_loss_scales
        ).sum() / pixel_weights.sum().clamp_min(1e-6)

        if low_score_neg_weight > 0.0 and entry.negative_flat_idx is not None:
            neg_idx = entry.negative_flat_idx.to(
                device=query_logits.device,
                dtype=torch.long,
            )
            if int(neg_idx.numel()) > 0:
                neg_logits = logits_flat[channel, neg_idx]
                neg_target = entry.negative_targets
                if neg_target is None:
                    neg_target = torch.zeros_like(neg_logits)
                else:
                    neg_target = neg_target.to(
                        device=query_logits.device,
                        dtype=neg_logits.dtype,
                    )
                neg_per_pixel = F.binary_cross_entropy_with_logits(
                    neg_logits.float(),
                    neg_target.float(),
                    reduction="none",
                ).to(dtype=query_logits.dtype)
                neg_weights = entry.negative_weights
                if neg_weights is None:
                    neg_weights = torch.ones_like(neg_per_pixel)
                else:
                    neg_weights = neg_weights.to(
                        device=query_logits.device,
                        dtype=neg_per_pixel.dtype,
                    )
                prompt_loss = prompt_loss + float(low_score_neg_weight) * weighted_mean(
                    neg_per_pixel,
                    neg_weights,
                )

        class_id = int(query_idx_list[query_id])
        class_weight = entry.class_weight.to(
            device=query_logits.device,
            dtype=prompt_loss.dtype,
        )
        class_reliability = entry.class_reliability
        if class_reliability is None:
            class_reliability = class_weight.new_tensor(1.0)
        else:
            class_reliability = class_reliability.to(
                device=query_logits.device,
                dtype=prompt_loss.dtype,
            )
        mask_loss_scale = class_weight.new_tensor(
            float(entry.mask_loss_scale)
        )
        losses_by_class.setdefault(class_id, []).append(
            prompt_loss
            * class_weight
            * class_reliability
            * mask_loss_scale
        )
        weights_by_class.setdefault(class_id, []).append(class_weight)

    if not losses_by_class:
        return query_logits.sum() * 0.0

    class_losses = []
    for class_id in sorted(losses_by_class):
        class_losses.append(
            torch.stack(losses_by_class[class_id]).sum()
            / torch.stack(weights_by_class[class_id]).sum().clamp_min(1e-6)
        )
    return torch.stack(class_losses).mean()
