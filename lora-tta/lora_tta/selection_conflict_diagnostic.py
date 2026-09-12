from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch


DEFAULT_NEAR_TIE_THRESHOLDS = (0.01, 0.05, 0.10)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _threshold_key(value: float) -> str:
    return f"{float(value):g}"


def _validate_indices(
    indices: torch.Tensor,
    *,
    spatial_pixels: int,
    device: torch.device,
) -> torch.Tensor:
    indices = indices.detach().to(device=device, dtype=torch.long).reshape(-1)
    if bool(((indices < 0) | (indices >= int(spatial_pixels))).any().item()):
        raise ValueError("selected flat index is out of range")
    return indices


def _gt_group(
    mask: torch.Tensor,
    *,
    valid_gt: torch.Tensor,
    correct_gt: torch.Tensor,
    total_selected: int,
    prefix: str,
) -> dict:
    pixels = int(mask.sum().item())
    valid_pixels = int((mask & valid_gt).sum().item())
    correct_pixels = int((mask & correct_gt).sum().item())
    return {
        f"{prefix}_pixels": pixels,
        f"{prefix}_rate": _rate(pixels, total_selected),
        f"{prefix}_valid_gt_pixels": valid_pixels,
        f"{prefix}_correct_gt_pixels": correct_pixels,
        f"{prefix}_gt_purity": _rate(correct_pixels, valid_pixels),
    }


def _score_stats(
    selected_scores: torch.Tensor,
    max_other_scores: torch.Tensor,
) -> dict:
    if int(selected_scores.numel()) == 0:
        return {
            "selected_score_sum": 0.0,
            "selected_score_mean": None,
            "max_other_score_sum": 0.0,
            "max_other_score_mean": None,
            "score_delta_sum": 0.0,
            "score_delta_mean": None,
            "score_delta_min": None,
            "score_delta_max": None,
        }
    delta = selected_scores - max_other_scores
    return {
        "selected_score_sum": float(selected_scores.sum().item()),
        "selected_score_mean": float(selected_scores.mean().item()),
        "max_other_score_sum": float(max_other_scores.sum().item()),
        "max_other_score_mean": float(max_other_scores.mean().item()),
        "score_delta_sum": float(delta.sum().item()),
        "score_delta_mean": float(delta.mean().item()),
        "score_delta_min": float(delta.min().item()),
        "score_delta_max": float(delta.max().item()),
    }


def selection_conflict_diagnostic(
    *,
    class_scores: torch.Tensor,
    selected_flat_indices: Mapping[int, torch.Tensor],
    negative_flat_indices: Mapping[int, torch.Tensor],
    gt: torch.Tensor,
    near_tie_thresholds: Sequence[float] = DEFAULT_NEAR_TIE_THRESHOLDS,
) -> dict:
    """Diagnose hidden score ordering and cross-class positive/negative reuse."""

    if class_scores.ndim == 4 and int(class_scores.shape[0]) == 1:
        class_scores = class_scores[0]
    if class_scores.ndim != 3:
        raise ValueError("class scores must have shape [classes, height, width]")
    class_scores = class_scores.detach().float()
    num_classes = int(class_scores.shape[0])
    if num_classes < 2:
        raise ValueError("selection conflict diagnostic requires at least two classes")
    spatial_pixels = int(class_scores.shape[-2] * class_scores.shape[-1])
    if int(gt.numel()) != spatial_pixels:
        raise ValueError("GT and class scores must have the same spatial size")
    thresholds = tuple(sorted({float(value) for value in near_tie_thresholds}))
    if any(value <= 0.0 for value in thresholds):
        raise ValueError("near-tie thresholds must be positive")

    positives: dict[int, torch.Tensor] = {}
    negatives: dict[int, torch.Tensor] = {}
    for class_id, indices in selected_flat_indices.items():
        class_id = int(class_id)
        if not 0 <= class_id < num_classes:
            raise ValueError(f"selected class {class_id} is out of range")
        positives[class_id] = _validate_indices(
            indices,
            spatial_pixels=spatial_pixels,
            device=class_scores.device,
        )
    for class_id, indices in negative_flat_indices.items():
        class_id = int(class_id)
        if not 0 <= class_id < num_classes:
            raise ValueError(f"negative class {class_id} is out of range")
        negatives[class_id] = _validate_indices(
            indices,
            spatial_pixels=spatial_pixels,
            device=class_scores.device,
        )

    class_parts = []
    index_parts = []
    class_slices: dict[int, slice] = {}
    offset = 0
    for class_id in sorted(positives):
        indices = positives[class_id]
        if int(indices.numel()) == 0:
            continue
        index_parts.append(indices)
        class_parts.append(torch.full_like(indices, class_id))
        class_slices[class_id] = slice(offset, offset + int(indices.numel()))
        offset += int(indices.numel())

    if index_parts:
        selected_indices = torch.cat(index_parts)
        selected_classes = torch.cat(class_parts)
    else:
        selected_indices = torch.empty(
            0,
            device=class_scores.device,
            dtype=torch.long,
        )
        selected_classes = torch.empty_like(selected_indices)

    selected_count = int(selected_indices.numel())
    scores_flat = class_scores.reshape(num_classes, -1)
    columns = torch.arange(selected_count, device=class_scores.device)
    scores_at_selected = scores_flat[:, selected_indices]
    selected_scores = scores_at_selected[selected_classes, columns]
    competitor_scores = scores_at_selected.clone()
    competitor_scores[selected_classes, columns] = -torch.inf
    max_other_scores = competitor_scores.max(dim=0).values
    score_delta = selected_scores - max_other_scores
    selected_rank = 1 + (scores_at_selected > selected_scores.unsqueeze(0)).sum(dim=0)
    argmax_mask = selected_rank == 1
    non_argmax_mask = ~argmax_mask

    gt_values = gt.detach().to(device=class_scores.device).reshape(-1)[selected_indices].long()
    valid_gt = (gt_values >= 0) & (gt_values < num_classes)
    correct_gt = valid_gt & (gt_values == selected_classes)

    rank_histogram = {
        str(rank): int((selected_rank == rank).sum().item())
        for rank in range(1, num_classes + 1)
        if bool((selected_rank == rank).any().item())
    }
    near_tie = {}
    for threshold in thresholds:
        mask = argmax_mask & (score_delta >= 0.0) & (score_delta < threshold)
        pixels = int(mask.sum().item())
        valid_pixels = int((mask & valid_gt).sum().item())
        correct_pixels = int((mask & correct_gt).sum().item())
        near_tie[_threshold_key(threshold)] = {
            "pixels": pixels,
            "rate": _rate(pixels, selected_count),
            "valid_gt_pixels": valid_pixels,
            "correct_gt_pixels": correct_pixels,
            "gt_purity": _rate(correct_pixels, valid_pixels),
        }

    overlap_mask = torch.zeros(selected_count, device=class_scores.device, dtype=torch.bool)
    overlap_pairs: dict[str, int] = {}
    overlap_by_positive_class: dict[int, dict[str, int]] = {}
    overlap_events = 0
    for positive_class, class_slice in class_slices.items():
        positive_indices = selected_indices[class_slice]
        class_overlap = torch.zeros_like(positive_indices, dtype=torch.bool)
        by_negative_class = {}
        for negative_class, negative_indices in sorted(negatives.items()):
            if negative_class == positive_class or int(negative_indices.numel()) == 0:
                continue
            pair_overlap = torch.isin(positive_indices, negative_indices)
            pair_count = int(pair_overlap.sum().item())
            if pair_count <= 0:
                continue
            pair_key = f"{positive_class}->{negative_class}"
            overlap_pairs[pair_key] = pair_count
            by_negative_class[str(negative_class)] = pair_count
            overlap_events += pair_count
            class_overlap |= pair_overlap
        overlap_mask[class_slice] = class_overlap
        overlap_by_positive_class[positive_class] = by_negative_class

    if selected_count > 0:
        positive_counts = torch.bincount(selected_indices, minlength=spatial_pixels)
        positive_positive_overlap_pixels = int((positive_counts > 1).sum().item())
    else:
        positive_positive_overlap_pixels = 0

    def build_record(mask: torch.Tensor, class_id: int | None = None) -> dict:
        count = int(mask.sum().item())
        valid_count = int((mask & valid_gt).sum().item())
        correct_count = int((mask & correct_gt).sum().item())
        record = {
            "selected_pixels": count,
            "selected_valid_gt_pixels": valid_count,
            "selected_correct_gt_pixels": correct_count,
            "selected_gt_purity": _rate(correct_count, valid_count),
            **_gt_group(
                mask & argmax_mask,
                valid_gt=valid_gt,
                correct_gt=correct_gt,
                total_selected=count,
                prefix="argmax",
            ),
            **_gt_group(
                mask & non_argmax_mask,
                valid_gt=valid_gt,
                correct_gt=correct_gt,
                total_selected=count,
                prefix="non_argmax",
            ),
        }
        record.update(
            _score_stats(
                selected_scores[mask],
                max_other_scores[mask],
            )
        )
        record["rank_histogram"] = {
            str(rank): int((mask & (selected_rank == rank)).sum().item())
            for rank in range(1, num_classes + 1)
            if bool((mask & (selected_rank == rank)).any().item())
        }
        record["near_tie"] = {}
        for threshold in thresholds:
            tie_mask = mask & argmax_mask & (score_delta >= 0.0) & (
                score_delta < threshold
            )
            tie_pixels = int(tie_mask.sum().item())
            tie_valid = int((tie_mask & valid_gt).sum().item())
            tie_correct = int((tie_mask & correct_gt).sum().item())
            record["near_tie"][_threshold_key(threshold)] = {
                "pixels": tie_pixels,
                "rate": _rate(tie_pixels, count),
                "valid_gt_pixels": tie_valid,
                "correct_gt_pixels": tie_correct,
                "gt_purity": _rate(tie_correct, tie_valid),
            }
        class_overlap_pixels = int((mask & overlap_mask).sum().item())
        record["positive_negative_overlap_pixels"] = class_overlap_pixels
        record["positive_negative_overlap_rate"] = _rate(
            class_overlap_pixels,
            count,
        )
        if class_id is not None:
            record["positive_negative_overlap_by_negative_class"] = (
                overlap_by_positive_class.get(class_id, {})
            )
        return record

    all_mask = torch.ones(selected_count, device=class_scores.device, dtype=torch.bool)
    result = build_record(all_mask)
    result["near_tie"] = near_tie
    result["positive_positive_overlap_pixels"] = positive_positive_overlap_pixels
    result["positive_negative_overlap_events"] = overlap_events
    result["positive_negative_overlap_pairs"] = overlap_pairs
    result["per_class"] = {
        str(class_id): build_record(
            selected_classes == class_id,
            class_id=class_id,
        )
        for class_id in sorted(class_slices)
    }
    return result


def _sum_int(records: Iterable[Mapping], key: str) -> int:
    return sum(int(record.get(key, 0) or 0) for record in records)


def _merge_counts(records: Sequence[Mapping], key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for record in records:
        for name, value in (record.get(key, {}) or {}).items():
            merged[str(name)] = merged.get(str(name), 0) + int(value)
    return dict(sorted(merged.items()))


def _aggregate_group(records: Sequence[Mapping]) -> dict:
    selected = _sum_int(records, "selected_pixels")
    valid = _sum_int(records, "selected_valid_gt_pixels")
    correct = _sum_int(records, "selected_correct_gt_pixels")
    result = {
        "selected_pixels": selected,
        "selected_valid_gt_pixels": valid,
        "selected_correct_gt_pixels": correct,
        "selected_gt_purity": _rate(correct, valid),
    }
    for prefix in ("argmax", "non_argmax"):
        pixels = _sum_int(records, f"{prefix}_pixels")
        valid_pixels = _sum_int(records, f"{prefix}_valid_gt_pixels")
        correct_pixels = _sum_int(records, f"{prefix}_correct_gt_pixels")
        result.update(
            {
                f"{prefix}_pixels": pixels,
                f"{prefix}_rate": _rate(pixels, selected),
                f"{prefix}_valid_gt_pixels": valid_pixels,
                f"{prefix}_correct_gt_pixels": correct_pixels,
                f"{prefix}_gt_purity": _rate(correct_pixels, valid_pixels),
            }
        )
    result["rank_histogram"] = _merge_counts(records, "rank_histogram")
    threshold_names = sorted(
        {
            str(name)
            for record in records
            for name in (record.get("near_tie", {}) or {})
        },
        key=float,
    )
    result["near_tie"] = {}
    for name in threshold_names:
        entries = [
            record.get("near_tie", {}).get(name, {})
            for record in records
        ]
        tie_pixels = _sum_int(entries, "pixels")
        tie_valid = _sum_int(entries, "valid_gt_pixels")
        tie_correct = _sum_int(entries, "correct_gt_pixels")
        result["near_tie"][name] = {
            "pixels": tie_pixels,
            "rate": _rate(tie_pixels, selected),
            "valid_gt_pixels": tie_valid,
            "correct_gt_pixels": tie_correct,
            "gt_purity": _rate(tie_correct, tie_valid),
        }
    overlap_pixels = _sum_int(records, "positive_negative_overlap_pixels")
    result["positive_negative_overlap_pixels"] = overlap_pixels
    result["positive_negative_overlap_rate"] = _rate(overlap_pixels, selected)
    for prefix in ("selected_score", "max_other_score", "score_delta"):
        total = sum(float(record.get(f"{prefix}_sum", 0.0) or 0.0) for record in records)
        result[f"{prefix}_sum"] = total
        result[f"{prefix}_mean"] = float(total / selected) if selected > 0 else None
    delta_mins = [
        float(record["score_delta_min"])
        for record in records
        if record.get("score_delta_min") is not None
    ]
    delta_maxs = [
        float(record["score_delta_max"])
        for record in records
        if record.get("score_delta_max") is not None
    ]
    result["score_delta_min"] = min(delta_mins) if delta_mins else None
    result["score_delta_max"] = max(delta_maxs) if delta_maxs else None
    return result


def aggregate_selection_conflict_diagnostics(records: Sequence[Mapping]) -> dict:
    records = list(records)
    summary = _aggregate_group(records)
    summary["images"] = len(records)
    summary["positive_positive_overlap_pixels"] = _sum_int(
        records,
        "positive_positive_overlap_pixels",
    )
    summary["positive_negative_overlap_events"] = _sum_int(
        records,
        "positive_negative_overlap_events",
    )
    summary["positive_negative_overlap_pairs"] = _merge_counts(
        records,
        "positive_negative_overlap_pairs",
    )

    threshold_names = sorted(
        {
            str(name)
            for record in records
            for name in (record.get("near_tie", {}) or {})
        },
        key=float,
    )
    summary["near_tie"] = {}
    for name in threshold_names:
        entries = [
            record.get("near_tie", {}).get(name, {})
            for record in records
        ]
        pixels = _sum_int(entries, "pixels")
        valid = _sum_int(entries, "valid_gt_pixels")
        correct = _sum_int(entries, "correct_gt_pixels")
        summary["near_tie"][name] = {
            "pixels": pixels,
            "rate": _rate(pixels, summary["selected_pixels"]),
            "valid_gt_pixels": valid,
            "correct_gt_pixels": correct,
            "gt_purity": _rate(correct, valid),
        }

    class_ids = sorted(
        {
            str(class_id)
            for record in records
            for class_id in (record.get("per_class", {}) or {})
        },
        key=int,
    )
    summary["per_class"] = {}
    for class_id in class_ids:
        class_records = [
            record["per_class"][class_id]
            for record in records
            if class_id in (record.get("per_class", {}) or {})
        ]
        class_summary = _aggregate_group(class_records)
        class_summary["images"] = len(class_records)
        class_summary["positive_negative_overlap_by_negative_class"] = (
            _merge_counts(
                class_records,
                "positive_negative_overlap_by_negative_class",
            )
        )
        summary["per_class"][class_id] = class_summary
    return summary
