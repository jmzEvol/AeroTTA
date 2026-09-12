from __future__ import annotations

from collections.abc import Mapping

import torch


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: torch.Tensor) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(values.float().mean().item())


def _prediction_view_stats(
    *,
    selected_classes: torch.Tensor,
    selected_flat_idx: torch.Tensor,
    pre_prediction: torch.Tensor,
    post_prediction: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
) -> dict:
    pre_values = pre_prediction.reshape(-1)[selected_flat_idx].long()
    post_values = post_prediction.reshape(-1)[selected_flat_idx].long()
    gt_values = gt.reshape(-1)[selected_flat_idx].long()

    valid_gt = (gt_values >= 0) & (gt_values < int(num_classes))
    pre_selected = pre_values == selected_classes
    post_selected = post_values == selected_classes
    retained = pre_selected & post_selected
    flip_away = pre_selected & ~post_selected
    flip_into = ~pre_selected & post_selected
    prediction_changed = pre_values != post_values
    correct_anchor = pre_selected & valid_gt & (gt_values == selected_classes)
    wrong_anchor = pre_selected & valid_gt & (gt_values != selected_classes)
    harmful_flip = correct_anchor & ~post_selected
    corrective_flip = wrong_anchor & (post_values == gt_values)

    pre_selected_count = int(pre_selected.sum().item())
    correct_anchor_count = int(correct_anchor.sum().item())
    wrong_anchor_count = int(wrong_anchor.sum().item())
    flip_away_count = int(flip_away.sum().item())
    harmful_flip_count = int(harmful_flip.sum().item())
    corrective_flip_count = int(corrective_flip.sum().item())

    return {
        "pre_selected_winner_pixels": pre_selected_count,
        "post_selected_winner_pixels": int(post_selected.sum().item()),
        "retained_pixels": int(retained.sum().item()),
        "retention_rate": _rate(int(retained.sum().item()), pre_selected_count),
        "flip_away_pixels": flip_away_count,
        "flip_away_rate": _rate(flip_away_count, pre_selected_count),
        "flip_into_pixels": int(flip_into.sum().item()),
        "prediction_changed_pixels": int(prediction_changed.sum().item()),
        "prediction_changed_rate": _rate(
            int(prediction_changed.sum().item()),
            int(selected_classes.numel()),
        ),
        "correct_anchor_pixels": correct_anchor_count,
        "harmful_flip_pixels": harmful_flip_count,
        "harmful_flip_rate": _rate(
            harmful_flip_count,
            correct_anchor_count,
        ),
        "wrong_anchor_pixels": wrong_anchor_count,
        "corrective_flip_pixels": corrective_flip_count,
        "corrective_flip_rate": _rate(
            corrective_flip_count,
            wrong_anchor_count,
        ),
    }


def _score_drift_stats(
    *,
    selected_classes: torch.Tensor,
    selected_flat_idx: torch.Tensor,
    pre_class_scores: torch.Tensor,
    post_class_scores: torch.Tensor,
) -> dict:
    pre_flat = pre_class_scores.reshape(pre_class_scores.shape[0], -1).float()
    post_flat = post_class_scores.reshape(post_class_scores.shape[0], -1).float()
    pre_selected = pre_flat[selected_classes, selected_flat_idx]
    post_selected = post_flat[selected_classes, selected_flat_idx]

    if int(pre_class_scores.shape[0]) > 1:
        columns = torch.arange(
            int(selected_classes.numel()),
            device=selected_classes.device,
        )
        pre_competitors = pre_flat[:, selected_flat_idx].clone()
        post_competitors = post_flat[:, selected_flat_idx].clone()
        pre_competitors[selected_classes, columns] = -torch.inf
        post_competitors[selected_classes, columns] = -torch.inf
        pre_margin = pre_selected - pre_competitors.max(dim=0).values
        post_margin = post_selected - post_competitors.max(dim=0).values
    else:
        pre_margin = pre_selected.new_zeros(pre_selected.shape)
        post_margin = post_selected.new_zeros(post_selected.shape)

    return {
        "selected_score_pre_mean": _mean(pre_selected),
        "selected_score_post_mean": _mean(post_selected),
        "selected_score_delta_mean": _mean(post_selected - pre_selected),
        "selected_margin_pre_mean": _mean(pre_margin),
        "selected_margin_post_mean": _mean(post_margin),
        "selected_margin_delta_mean": _mean(post_margin - pre_margin),
    }


def selected_topk_flip_diagnostic(
    *,
    selected_flat_indices: Mapping[int, torch.Tensor],
    pre_class_scores: torch.Tensor,
    post_class_scores: torch.Tensor,
    prediction_views: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    gt: torch.Tensor,
) -> dict:
    """Compare frozen Teacher-selected pixels with post-TTA outputs."""

    if pre_class_scores.shape != post_class_scores.shape:
        raise ValueError("pre/post class score shapes must match")
    if pre_class_scores.ndim != 3:
        raise ValueError("class scores must have shape [classes, height, width]")
    num_classes = int(pre_class_scores.shape[0])
    spatial_pixels = int(pre_class_scores.shape[-2] * pre_class_scores.shape[-1])
    if int(gt.numel()) != spatial_pixels:
        raise ValueError("GT and class scores must have the same spatial size")

    class_parts = []
    index_parts = []
    for class_id, flat_idx in sorted(selected_flat_indices.items()):
        if not 0 <= int(class_id) < num_classes:
            raise ValueError(f"selected class {class_id} is out of range")
        indices = flat_idx.detach().to(
            device=pre_class_scores.device,
            dtype=torch.long,
        ).reshape(-1)
        if int(indices.numel()) == 0:
            continue
        if bool(((indices < 0) | (indices >= spatial_pixels)).any().item()):
            raise ValueError("selected flat index is out of range")
        index_parts.append(indices)
        class_parts.append(torch.full_like(indices, int(class_id)))

    if index_parts:
        selected_flat_idx = torch.cat(index_parts)
        selected_classes = torch.cat(class_parts)
    else:
        selected_flat_idx = torch.empty(
            0,
            device=pre_class_scores.device,
            dtype=torch.long,
        )
        selected_classes = torch.empty_like(selected_flat_idx)

    for name, (pre_prediction, post_prediction) in prediction_views.items():
        if int(pre_prediction.numel()) != spatial_pixels:
            raise ValueError(f"pre prediction view {name!r} has the wrong size")
        if int(post_prediction.numel()) != spatial_pixels:
            raise ValueError(f"post prediction view {name!r} has the wrong size")

    def build_record(class_mask: torch.Tensor) -> dict:
        class_indices = selected_flat_idx[class_mask]
        class_labels = selected_classes[class_mask]
        gt_values = gt.reshape(-1)[class_indices].long()
        valid_gt = (gt_values >= 0) & (gt_values < num_classes)
        correct_gt = valid_gt & (gt_values == class_labels)
        valid_count = int(valid_gt.sum().item())
        return {
            "selected_pixels": int(class_indices.numel()),
            "selected_valid_gt_pixels": valid_count,
            "selected_correct_gt_pixels": int(correct_gt.sum().item()),
            "selected_gt_purity": _rate(
                int(correct_gt.sum().item()),
                valid_count,
            ),
            "score_drift": _score_drift_stats(
                selected_classes=class_labels,
                selected_flat_idx=class_indices,
                pre_class_scores=pre_class_scores,
                post_class_scores=post_class_scores,
            ),
            "views": {
                name: _prediction_view_stats(
                    selected_classes=class_labels,
                    selected_flat_idx=class_indices,
                    pre_prediction=pre_prediction,
                    post_prediction=post_prediction,
                    gt=gt,
                    num_classes=num_classes,
                )
                for name, (pre_prediction, post_prediction) in prediction_views.items()
            },
        }

    all_mask = torch.ones_like(selected_classes, dtype=torch.bool)
    result = build_record(all_mask)
    result["per_class"] = {
        str(class_id): build_record(selected_classes == int(class_id))
        for class_id in sorted(selected_flat_indices)
        if bool((selected_classes == int(class_id)).any().item())
    }
    return result

