from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


_ROUTING_COUNT_KEYS = (
    "evaluated_pixels",
    "semantic_winner_pixels",
    "instance_winner_pixels",
    "tie_pixels",
)
_ROUTING_SUM_KEYS = (
    "semantic_score_sum",
    "instance_score_sum",
    "fused_score_sum",
    "teacher_target_sum",
)


def _aggregate_routing_rows(rows) -> dict[str, int | float]:
    totals: dict[str, int | float] = {
        **{key: 0 for key in _ROUTING_COUNT_KEYS},
        **{key: 0.0 for key in _ROUTING_SUM_KEYS},
    }
    for row in rows:
        row = dict(row or {})
        for key in _ROUTING_COUNT_KEYS:
            totals[key] += int(row.get(key, 0))
        for key in _ROUTING_SUM_KEYS:
            totals[key] += float(row.get(key, 0.0))
    pixel_count = int(totals["evaluated_pixels"])
    denominator = max(pixel_count, 1)
    totals.update(
        {
            "semantic_winner_ratio": float(
                totals["semantic_winner_pixels"] / denominator
            ),
            "instance_winner_ratio": float(
                totals["instance_winner_pixels"] / denominator
            ),
            "tie_ratio": float(totals["tie_pixels"] / denominator),
            "mean_semantic_score": float(
                totals["semantic_score_sum"] / denominator
            ),
            "mean_instance_score": float(
                totals["instance_score_sum"] / denominator
            ),
            "mean_fused_score": float(
                totals["fused_score_sum"] / denominator
            ),
            "mean_teacher_target": float(
                totals["teacher_target_sum"] / denominator
            ),
        }
    )
    return totals


def aggregate_student_branch_routing(images) -> dict[str, Any]:
    """Aggregate per-step routing by BCE pixel count, never image ratios."""
    records = [
        dict(record)
        for image in images
        for record in (image.get("student_branch_routing", ()) or ())
    ]
    overall = _aggregate_routing_rows(records)
    positive = _aggregate_routing_rows(
        record.get("positive", {}) for record in records
    )
    negative = _aggregate_routing_rows(
        record.get("negative", {}) for record in records
    )
    class_ids = sorted(
        {
            int(class_id)
            for record in records
            for class_id in (record.get("per_class", {}) or {})
        }
    )
    per_class = {}
    for class_id in class_ids:
        class_rows = [
            (record.get("per_class", {}) or {}).get(str(class_id), {})
            for record in records
        ]
        per_class[str(class_id)] = {
            **_aggregate_routing_rows(class_rows),
            "positive": _aggregate_routing_rows(
                row.get("positive", {}) for row in class_rows
            ),
            "negative": _aggregate_routing_rows(
                row.get("negative", {}) for row in class_rows
            ),
        }
    return {
        "enabled": bool(records),
        "steps": len(records),
        **overall,
        "prompts_without_kept_instances": sum(
            int(record.get("prompts_without_kept_instances", 0))
            for record in records
        ),
        "positive": positive,
        "negative": negative,
        "per_class": per_class,
    }


def load_result(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_iou_deltas(path: str | Path) -> list[dict[str, float | int | str]]:
    result = load_result(path)
    names = result.get("class_names") or result.get("params", {}).get("class_names")
    baseline = result["baseline_per_class_iou"]
    tta = result["tta_per_class_iou"]
    if names is None:
        names = [str(idx) for idx in range(len(baseline))]
    rows = []
    for idx, (name, base_iou, tta_iou) in enumerate(zip(names, baseline, tta)):
        rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "baseline": float(base_iou),
                "tta": float(tta_iou),
                "delta": float(tta_iou) - float(base_iou),
            }
        )
    return rows


def selection_distribution(path: str | Path) -> dict[str, Any]:
    result = load_result(path)
    names = result.get("class_names") or result.get("params", {}).get("class_names")
    if names is None:
        names = []
    pixels: Counter[int] = Counter()
    images: Counter[int] = Counter()
    for image in result.get("images", []):
        selected_per_class = image.get("selected_per_class") or {}
        for cls_key, value in selected_per_class.items():
            cls_idx = int(cls_key)
            count = int(value)
            pixels[cls_idx] += count
            if count > 0:
                images[cls_idx] += 1
    total = sum(pixels.values())
    max_cls = max([len(names) - 1, *pixels.keys()], default=-1)
    classes = []
    for cls_idx in range(max_cls + 1):
        count = int(pixels[cls_idx])
        classes.append(
            {
                "class_id": cls_idx,
                "class_name": names[cls_idx] if cls_idx < len(names) else str(cls_idx),
                "pixels": count,
                "percent": (100.0 * count / total) if total else 0.0,
                "images": int(images[cls_idx]),
            }
        )
    return {"total_selected": int(total), "classes": classes}
