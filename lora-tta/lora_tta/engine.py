from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .alias_fusion import (
    AliasReliabilityResult,
    class_scores_from_query_scores_with_frozen_alias_residual,
    class_scores_from_query_scores_with_alias_reliability,
    estimate_alias_reliability,
)
from .adapter import (
    SAM3LoRAAdapter,
    build_dataloader,
    build_dataset_from_eval_config,
    load_eval_config,
    resolve_project_path,
)
from .boundary_causal_diagnostic import (
    DEFAULT_BOUNDARY_RADII,
    normalize_boundary_radii,
)
from .correction_survival_diagnostic import (
    merge_prompt_risk_with_leave_one_out,
)
from .diagnostics import aggregate_student_branch_routing
from .config import (
    FinalHeadConfig,
    MiningConfig,
    PromptViewConfig,
    ProtectionConfig,
    TTAConfig,
)
from .losses import (
    SelectedClassEntry,
    apply_decision_boundary_contraction_gate,
    apply_mask_loss_scale,
    selected_class_soft_bce_loss,
    selected_class_soft_bce_loss_terms,
    selected_query_soft_bce_loss,
)
from .metrics import ConfusionMatrix
from .mining import (
    build_component_gate_masks,
    select_classwise_pixels,
    select_independent_query_pixels,
)
from .oracle import (
    OracleCandidate,
    apply_optimizer_lr_multiplier,
    build_oracle_candidates,
    capture_optimizer_lrs,
    compare_output_states,
    extract_output_state,
    extract_teacher_features,
    gradient_norm_features,
    rebuild_selected_targets,
    restore_optimizer_lrs,
    select_oracle_candidate,
    select_oracle_candidate_or_no_update,
)
from .prompt_mining_audit import (
    build_canonical_query_ids,
    build_first_query_ids_by_class,
    build_prompt_class_views,
)
from .protection import (
    ProtectionDecision,
    apply_protection_decision,
    decide_supported_class_retention,
    decide_supported_class_survival,
    decide_unsupported_class_birth,
    summarize_prediction_replacements,
)
from .teacher_prompt_diagnostic import evaluate_teacher_prompt_candidates
from .teacher_prompt_gradient import (
    flatten_parameter_gradients,
    summarize_prompt_gradient_vectors,
)
from .teacher_prompt_trial import (
    measure_teacher_trial_transfer,
    replace_teacher_query,
)
from .prompt_causal_diagnostic import CrossTimePromptCausalAccumulator
from .prompt_transfer_routing import PromptRoutingDiagnosticAccumulator
from .runtime import (
    all_reduce_tensor,
    build_lora_param_groups,
    cleanup_distributed,
    clear_lora_weight_cache,
    reset_optimizer_state,
    restore_lora_params,
    set_random_seed,
    setup_distributed,
    snapshot_lora_params,
)
from .scores import (
    class_scores_from_query_scores,
    class_scores_from_query_scores_with_canonical_overrides,
    predict_from_class_scores,
)
from .selected_flip_diagnostic import selected_topk_flip_diagnostic
from .selection_conflict_diagnostic import (
    DEFAULT_NEAR_TIE_THRESHOLDS,
    aggregate_selection_conflict_diagnostics,
    selection_conflict_diagnostic,
)
from .synonym_diagnostics import SynonymDiagnosticAccumulator
from .visualization import ComparisonVisualizationWriter


def _sample_id_from_batch(batch) -> str:
    data_samples = batch["data_samples"] if isinstance(batch, dict) else [item["data_samples"] for item in batch]
    sample = data_samples[0]
    return str(sample.metainfo.get("img_path", sample.metainfo.get("seg_map_path", "")))


def _save_primary_visualization(
    writer: ComparisonVisualizationWriter | None,
    *,
    batch,
    sample_id: str,
    primary_head_name: str,
    baseline_preds: dict[str, torch.Tensor],
    tta_preds: dict[str, torch.Tensor],
    gt: torch.Tensor,
    adapted: bool,
    baseline_miou: float,
    tta_miou: float,
    delta_miou: float,
) -> str | None:
    if writer is None:
        return None
    return writer.save(
        batch=batch,
        sample_id=sample_id,
        gt=gt,
        before=baseline_preds[primary_head_name],
        after=tta_preds[primary_head_name],
        adapted=adapted,
        baseline_miou=baseline_miou,
        tta_miou=tta_miou,
        delta_miou=delta_miou,
    )


def _update_synonym_diagnostic(
    accumulator: SynonymDiagnosticAccumulator | None,
    *,
    sample_id: str,
    pre_query_scores: torch.Tensor,
    post_query_scores: torch.Tensor | None,
    gt: torch.Tensor,
    adapted: bool,
) -> None:
    if accumulator is None:
        return
    accumulator.update_image(
        sample_id,
        pre_query_scores,
        pre_query_scores if post_query_scores is None else post_query_scores,
        gt,
        adapted=adapted,
    )


def _update_prompt_routing_diagnostic(
    accumulator: PromptRoutingDiagnosticAccumulator | None,
    *,
    sample_id: str,
    pre_query_scores: torch.Tensor,
    pre_reliability_scores: torch.Tensor,
    pre_query_presence: torch.Tensor,
    post_query_scores: torch.Tensor | None,
    post_reliability_scores: torch.Tensor | None,
    post_query_presence: torch.Tensor | None,
    gt: torch.Tensor,
    adapted: bool,
) -> None:
    if accumulator is None:
        return
    accumulator.update_image(
        sample_id,
        pre_query_scores=pre_query_scores,
        pre_reliability_scores=pre_reliability_scores,
        pre_query_presence=pre_query_presence,
        post_query_scores=(
            pre_query_scores
            if post_query_scores is None
            else post_query_scores
        ),
        post_reliability_scores=(
            pre_reliability_scores
            if post_reliability_scores is None
            else post_reliability_scores
        ),
        post_query_presence=(
            pre_query_presence
            if post_query_presence is None
            else post_query_presence
        ),
        gt=gt,
        adapted=adapted,
    )


def _update_cross_time_diagnostic(
    accumulator: CrossTimePromptCausalAccumulator | None,
    *,
    sample_id: str,
    pre_query_scores: torch.Tensor,
    post_query_scores: torch.Tensor | None,
    gt: torch.Tensor,
    adapted: bool,
) -> None:
    if accumulator is None:
        return
    accumulator.update_image(
        sample_id,
        pre_query_scores=pre_query_scores,
        post_query_scores=(
            pre_query_scores
            if post_query_scores is None
            else post_query_scores
        ),
        gt=gt,
        adapted=adapted,
    )


def _finalize_cross_time_diagnostic(
    accumulator: CrossTimePromptCausalAccumulator | None,
    *,
    baseline_metrics,
    tta_metrics,
    raw_head: str | None,
    canonical_head: str | None,
) -> dict | None:
    if accumulator is None:
        return None
    if raw_head is None or canonical_head is None:
        return accumulator.finalize()
    return accumulator.finalize(
        expected_f00=baseline_metrics[raw_head].matrix,
        expected_f11=tta_metrics[raw_head].matrix,
        expected_canonical_pre=(
            baseline_metrics[canonical_head].matrix
        ),
        expected_canonical_post=tta_metrics[canonical_head].matrix,
    )


def _merge_synonym_cross_time_diagnostics(
    *,
    synonym_report: dict | None,
    cross_time_report: dict | None,
) -> None:
    if synonym_report is None or cross_time_report is None:
        return
    correction = cross_time_report.get("correction_survival")
    if correction is None:
        return
    correction["prompt_risk_with_leave_one_out"] = (
        merge_prompt_risk_with_leave_one_out(
            correction,
            synonym_report,
        )
    )


def _valid_mask_for_model(gt: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
    valid = gt != 255
    if tuple(valid.shape[-2:]) == tuple(out_size):
        return valid
    resized = torch.nn.functional.interpolate(
        valid[None, None].float(),
        size=out_size,
        mode="nearest",
    )[0, 0]
    return resized > 0.5


def _summary_to_percent(summary) -> tuple[float, list[float]]:
    return (
        float(summary.miou) * 100.0,
        [float(value) * 100.0 for value in summary.per_class_iou],
    )


def _accuracy_summary_to_percent(
    summary,
) -> tuple[float, list[float], float]:
    return (
        float(summary.macc) * 100.0,
        [
            float(value) * 100.0
            for value in summary.per_class_accuracy
        ],
        float(summary.aacc) * 100.0,
    )


def _confusion_matrix_to_list(metric: ConfusionMatrix) -> list[list[int]]:
    return metric.matrix.detach().cpu().tolist()


def _round_report_metric_fields(fields: dict) -> dict:
    """Round aggregate percentage metrics like MMSeg's IoUMetric."""
    rounded = dict(fields)
    scalar_suffixes = ("_miou", "_macc", "_aacc")
    vector_suffixes = ("_per_class_iou", "_per_class_accuracy")
    for key, value in fields.items():
        if key.endswith(scalar_suffixes):
            rounded[key] = float(np.round(float(value), 2))
        elif key.endswith(vector_suffixes):
            rounded[key] = [
                float(np.round(float(item), 2)) for item in value
            ]
    return rounded


def _single_metric_fields(
    metric: ConfusionMatrix,
    *,
    prefix: str,
) -> dict:
    summary = metric.summary()
    miou, per_class_iou = _summary_to_percent(summary)
    macc, per_class_accuracy, aacc = _accuracy_summary_to_percent(
        summary
    )
    return {
        f"{prefix}_miou": miou,
        f"{prefix}_per_class_iou": per_class_iou,
        f"{prefix}_valid_classes": int(summary.valid_classes),
        f"{prefix}_macc": macc,
        f"{prefix}_per_class_accuracy": per_class_accuracy,
        f"{prefix}_valid_accuracy_classes": int(
            summary.valid_accuracy_classes
        ),
        f"{prefix}_aacc": aacc,
        f"{prefix}_confusion_matrix": _confusion_matrix_to_list(
            metric
        ),
    }


def _comparison_metric_fields(
    baseline_metric: ConfusionMatrix,
    comparison_metric: ConfusionMatrix,
    *,
    comparison_prefix: str = "tta",
    delta_prefix: str = "delta",
    include_baseline: bool = True,
) -> dict:
    baseline = _single_metric_fields(
        baseline_metric,
        prefix="baseline",
    )
    comparison = _single_metric_fields(
        comparison_metric,
        prefix=comparison_prefix,
    )
    fields = dict(baseline) if include_baseline else {}
    fields.update(comparison)
    fields.update(
        {
            f"{delta_prefix}_miou": (
                comparison[f"{comparison_prefix}_miou"]
                - baseline["baseline_miou"]
            ),
            f"{delta_prefix}_per_class_iou": [
                float(current) - float(previous)
                for previous, current in zip(
                    baseline["baseline_per_class_iou"],
                    comparison[
                        f"{comparison_prefix}_per_class_iou"
                    ],
                )
            ],
            f"{delta_prefix}_macc": (
                comparison[f"{comparison_prefix}_macc"]
                - baseline["baseline_macc"]
            ),
            f"{delta_prefix}_per_class_accuracy": [
                float(current) - float(previous)
                for previous, current in zip(
                    baseline["baseline_per_class_accuracy"],
                    comparison[
                        f"{comparison_prefix}_per_class_accuracy"
                    ],
                )
            ],
            f"{delta_prefix}_aacc": (
                comparison[f"{comparison_prefix}_aacc"]
                - baseline["baseline_aacc"]
            ),
        }
    )
    return _round_report_metric_fields(fields)


def _image_miou_fields(
    *,
    baseline_pred: torch.Tensor,
    tta_pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    device: str | torch.device,
) -> dict:
    gt = gt.to(device=device, dtype=torch.long)
    baseline_metric = ConfusionMatrix(num_classes, device=device)
    tta_metric = ConfusionMatrix(num_classes, device=device)
    baseline_metric.update(baseline_pred, gt)
    tta_metric.update(tta_pred, gt)
    baseline_summary = baseline_metric.summary()
    tta_summary = tta_metric.summary()
    baseline_miou, baseline_per_class_iou = _summary_to_percent(baseline_summary)
    tta_miou, tta_per_class_iou = _summary_to_percent(tta_summary)
    (
        baseline_macc,
        baseline_per_class_accuracy,
        baseline_aacc,
    ) = _accuracy_summary_to_percent(baseline_summary)
    (
        tta_macc,
        tta_per_class_accuracy,
        tta_aacc,
    ) = _accuracy_summary_to_percent(tta_summary)
    return {
        "baseline_miou": baseline_miou,
        "baseline_per_class_iou": baseline_per_class_iou,
        "baseline_valid_classes": int(baseline_summary.valid_classes),
        "baseline_macc": baseline_macc,
        "baseline_per_class_accuracy": baseline_per_class_accuracy,
        "baseline_valid_accuracy_classes": int(
            baseline_summary.valid_accuracy_classes
        ),
        "baseline_aacc": baseline_aacc,
        "tta_miou": tta_miou,
        "tta_per_class_iou": tta_per_class_iou,
        "tta_valid_classes": int(tta_summary.valid_classes),
        "tta_macc": tta_macc,
        "tta_per_class_accuracy": tta_per_class_accuracy,
        "tta_valid_accuracy_classes": int(
            tta_summary.valid_accuracy_classes
        ),
        "tta_aacc": tta_aacc,
        "delta_miou": tta_miou - baseline_miou,
        "delta_per_class_iou": [
            float(tta_iou) - float(baseline_iou)
            for baseline_iou, tta_iou in zip(baseline_per_class_iou, tta_per_class_iou)
        ],
        "delta_macc": tta_macc - baseline_macc,
        "delta_per_class_accuracy": [
            float(tta_accuracy) - float(baseline_accuracy)
            for baseline_accuracy, tta_accuracy in zip(
                baseline_per_class_accuracy,
                tta_per_class_accuracy,
            )
        ],
        "delta_aacc": tta_aacc - baseline_aacc,
    }


def _raw_tta_image_fields(fields: dict) -> dict:
    return {
        "raw_tta_miou": fields["tta_miou"],
        "raw_tta_per_class_iou": fields["tta_per_class_iou"],
        "raw_tta_valid_classes": fields["tta_valid_classes"],
        "raw_delta_miou": fields["delta_miou"],
        "raw_delta_per_class_iou": fields["delta_per_class_iou"],
        "raw_tta_macc": fields["tta_macc"],
        "raw_tta_per_class_accuracy": fields[
            "tta_per_class_accuracy"
        ],
        "raw_tta_valid_accuracy_classes": fields[
            "tta_valid_accuracy_classes"
        ],
        "raw_tta_aacc": fields["tta_aacc"],
        "raw_delta_macc": fields["delta_macc"],
        "raw_delta_per_class_accuracy": fields[
            "delta_per_class_accuracy"
        ],
        "raw_delta_aacc": fields["delta_aacc"],
    }


def _protect_final_head_predictions(
    *,
    config: ProtectionConfig,
    primary_head_name: str,
    selected_class_ids,
    baseline_preds: dict[str, torch.Tensor],
    raw_tta_preds: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], ProtectionDecision | None]:
    if not config.enabled:
        return raw_tta_preds, None
    decision = decide_unsupported_class_birth(
        baseline_pred=baseline_preds[primary_head_name],
        selected_class_ids=selected_class_ids,
        raw_tta_pred=raw_tta_preds[primary_head_name],
        action=config.action,
    )
    return (
        apply_protection_decision(
            baseline_predictions=baseline_preds,
            raw_tta_predictions=raw_tta_preds,
            decision=decision,
        ),
        decision,
    )


def _decide_class_survival(
    *,
    config: ProtectionConfig,
    baseline_pred: torch.Tensor,
    raw_tta_pred: torch.Tensor,
    selected_class_ids,
):
    decision_fn = (
        decide_supported_class_retention
        if config.survival_metric == "retention"
        else decide_supported_class_survival
    )
    return decision_fn(
        baseline_pred=baseline_pred,
        raw_tta_pred=raw_tta_pred,
        selected_class_ids=selected_class_ids,
        threshold=config.survival_threshold,
        min_baseline_pixels=config.survival_min_baseline_pixels,
    )


def _query_ids_for_risk_classes(
    *,
    risk_class_ids,
    loss_query_ids: tuple[int, ...],
    canonical_query_ids: tuple[int, ...] | None,
    query_idx_list: list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    risk_classes = {int(class_id) for class_id in risk_class_ids}
    loss_query_set = {int(query_id) for query_id in loss_query_ids}
    if canonical_query_ids is not None:
        invalid = [
            class_id
            for class_id in risk_classes
            if class_id < 0 or class_id >= len(canonical_query_ids)
        ]
        if invalid:
            raise ValueError(
                f"risk classes {tuple(sorted(invalid))} have no canonical query"
            )
        return tuple(
            int(canonical_query_ids[class_id])
            for class_id in sorted(risk_classes)
            if int(canonical_query_ids[class_id]) in loss_query_set
        )
    return tuple(
        int(query_id)
        for query_id in loss_query_ids
        if int(query_idx_list[int(query_id)]) in risk_classes
    )
def _format_progress(
    *,
    rank: int,
    world_size: int,
    processed: int,
    total: int,
    sample_id: str,
    adapted: bool,
    selected_pixels: int,
    selected_classes: int,
    baseline_miou: float,
    tta_miou: float,
    delta_miou: float,
    baseline_macc: float,
    tta_macc: float,
    delta_macc: float,
    alias_reliability_mean: float | None = None,
    active_aliases: int | None = None,
    total_aliases: int | None = None,
) -> str:
    sample_name = Path(sample_id).name if sample_id else "<unknown>"
    message = (
        f"[clean-tta][rank {rank}/{world_size}] "
        f"{processed}/{total} {sample_name} "
        f"adapted={int(bool(adapted))} "
        f"selected_pixels={int(selected_pixels)} "
        f"selected_classes={int(selected_classes)} "
        f"baseline_miou={float(baseline_miou):.2f} "
        f"tta_miou={float(tta_miou):.2f} "
        f"delta_miou={float(delta_miou):+.2f} "
        f"baseline_macc={float(baseline_macc):.2f} "
        f"tta_macc={float(tta_macc):.2f} "
        f"delta_macc={float(delta_macc):+.2f}"
    )
    if alias_reliability_mean is not None:
        message += f" alias_r_mean={float(alias_reliability_mean):.3f}"
    if active_aliases is not None and total_aliases is not None:
        message += f" active_aliases={int(active_aliases)}/{int(total_aliases)}"
    return message


def _selection_skip_reason(
    *,
    selected_pixels: int,
    selected_classes: int,
    mining: MiningConfig,
) -> str | None:
    if int(selected_pixels) <= 0:
        return "empty_selection"
    if int(selected_classes) < int(mining.min_selected_classes):
        return "few_selected_classes"
    return None


def _alias_fusion_image_fields(
    result: AliasReliabilityResult,
    *,
    query_words: list[str] | tuple[str, ...],
) -> dict:
    aliases = []
    for record in result.records:
        values = record.to_dict()
        values["word"] = str(query_words[record.query_id])
        aliases.append(values)
    reliabilities = [float(record.reliability) for record in result.records]
    return {
        "anchor_counts": list(result.anchor_counts),
        "mean_reliability": (
            float(sum(reliabilities) / len(reliabilities)) if reliabilities else 0.0
        ),
        "active_aliases": int(sum(value > 0.0 for value in reliabilities)),
        "total_aliases": len(reliabilities),
        "aliases": aliases,
    }


def _summarize_alias_fusion(
    images: list[dict],
    *,
    field_name: str = "alias_fusion",
) -> dict | None:
    accumulators: dict[int, dict] = {}
    anchor_sums: list[float] = []
    image_count = 0
    for image in images:
        fields = image.get(field_name)
        if not fields:
            continue
        image_count += 1
        anchor_counts = [float(value) for value in fields.get("anchor_counts", [])]
        if len(anchor_sums) < len(anchor_counts):
            anchor_sums.extend([0.0] * (len(anchor_counts) - len(anchor_sums)))
        for class_id, count in enumerate(anchor_counts):
            anchor_sums[class_id] += count
        for record in fields.get("aliases", []):
            query_id = int(record["query_id"])
            accumulator = accumulators.setdefault(
                query_id,
                {
                    "query_id": query_id,
                    "class_id": int(record["class_id"]),
                    "word": str(record.get("word", "")),
                    "count": 0,
                    "active": 0,
                    "auc": 0.0,
                    "separation": 0.0,
                    "foreign_leakage": 0.0,
                    "reliability": 0.0,
                },
            )
            accumulator["count"] += 1
            accumulator["active"] += int(float(record["reliability"]) > 0.0)
            for key in ("auc", "separation", "foreign_leakage", "reliability"):
                accumulator[key] += float(record[key])
    if image_count == 0:
        return None

    aliases = []
    for query_id in sorted(accumulators):
        values = accumulators[query_id]
        count = max(int(values.pop("count")), 1)
        active = int(values.pop("active"))
        values["active_fraction"] = float(active / count)
        for key in ("auc", "separation", "foreign_leakage", "reliability"):
            values[f"mean_{key}"] = float(values.pop(key) / count)
        aliases.append(values)
    return {
        "images": image_count,
        "mean_anchor_counts": [float(value / image_count) for value in anchor_sums],
        "aliases": aliases,
    }


def _canonical_query_index(
    canonical_query_ids: tuple[int, ...] | list[int],
    *,
    num_queries: int,
    device: torch.device,
) -> torch.Tensor:
    query_index = torch.as_tensor(canonical_query_ids, device=device, dtype=torch.long)
    if query_index.ndim != 1 or int(query_index.numel()) == 0:
        raise ValueError("canonical_query_ids must be a non-empty one-dimensional sequence")
    if int(query_index.min().item()) < 0 or int(query_index.max().item()) >= int(num_queries):
        raise ValueError(
            f"canonical query ids {query_index.detach().cpu().tolist()} exceed "
            f"the query range [0, {int(num_queries) - 1}]"
        )
    return query_index


def _class_logits_for_selected_loss(
    adapter,
    query_logits: torch.Tensor,
    target_size: tuple[int, int],
    canonical_query_ids: tuple[int, ...] | None = None,
    query_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Resize query logits before class aggregation, matching the legacy TTA loss path."""
    if tuple(query_logits.shape[-2:]) != tuple(target_size):
        query_logits = F.interpolate(
            query_logits.float(),
            size=tuple(target_size),
            mode="bilinear",
            align_corners=False,
        )
    if canonical_query_ids is not None:
        if query_ids is None:
            query_index = _canonical_query_index(
                canonical_query_ids,
                num_queries=int(query_logits.shape[1]),
                device=query_logits.device,
            )
        else:
            channel_by_query_id = {
                int(query_id): channel for channel, query_id in enumerate(query_ids)
            }
            missing = [
                int(query_id)
                for query_id in canonical_query_ids
                if int(query_id) not in channel_by_query_id
            ]
            if missing:
                raise ValueError(
                    f"canonical query ids {tuple(missing)} are missing from query subset"
                )
            query_index = torch.tensor(
                [channel_by_query_id[int(query_id)] for query_id in canonical_query_ids],
                device=query_logits.device,
                dtype=torch.long,
            )
        return query_logits.index_select(1, query_index)
    return adapter.aggregate_class_logits(query_logits)


def _selected_union_flat_indices(
    selected,
    *,
    device: torch.device,
) -> torch.Tensor:
    flat_indices = []
    for entry in selected.values():
        if int(entry.flat_idx.numel()) > 0:
            flat_indices.append(
                entry.flat_idx.to(device=device, dtype=torch.long)
            )
        if (
            entry.negative_flat_idx is not None
            and int(entry.negative_flat_idx.numel()) > 0
        ):
            flat_indices.append(
                entry.negative_flat_idx.to(
                    device=device,
                    dtype=torch.long,
                )
            )
    if not flat_indices:
        return torch.empty((0,), device=device, dtype=torch.long)
    union_flat_idx = torch.unique(
        torch.cat(flat_indices, dim=0),
        sorted=True,
    )
    return union_flat_idx


class SparseSelectedLossPlan(NamedTuple):
    union_flat_idx: torch.Tensor
    selected: dict[int, SelectedClassEntry]
    target_size: tuple[int, int]


def _build_sparse_selected_loss_plan(
    selected,
    *,
    target_size: tuple[int, int],
    device: torch.device,
) -> SparseSelectedLossPlan:
    target_size = (int(target_size[0]), int(target_size[1]))
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    union_flat_idx = _selected_union_flat_indices(
        selected,
        device=device,
    )
    if int(union_flat_idx.numel()) > 0:
        bounds = union_flat_idx[[0, -1]].detach().cpu().tolist()
        if int(bounds[0]) < 0 or int(bounds[1]) >= target_size[0] * target_size[1]:
            raise ValueError(
                f"selected index range [{int(bounds[0])}, {int(bounds[1])}] "
                f"is outside target pixel range [0, {target_size[0] * target_size[1] - 1}]"
            )
    return SparseSelectedLossPlan(
        union_flat_idx=union_flat_idx,
        selected=_remap_selected_to_sparse_indices(
            selected,
            union_flat_idx=union_flat_idx,
        ),
        target_size=target_size,
    )


def _sparse_plan_for_update(
    *,
    selected,
    target_size: tuple[int, int],
    device: torch.device,
    loss_config,
    steps: int,
) -> SparseSelectedLossPlan | None:
    if (
        not bool(getattr(loss_config, "sparse_selected_logits", False))
        or int(steps) <= 0
    ):
        return None
    return _build_sparse_selected_loss_plan(
        selected,
        target_size=target_size,
        device=device,
    )


def _sample_query_logits_at_flat_indices(
    query_logits: torch.Tensor,
    *,
    flat_indices: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    if query_logits.ndim != 4 or int(query_logits.shape[0]) != 1:
        raise ValueError(
            f"expected query_logits [1,Q,H,W], got {tuple(query_logits.shape)}"
        )
    target_height, target_width = (
        int(target_size[0]),
        int(target_size[1]),
    )
    same_size = tuple(query_logits.shape[-2:]) == (target_height, target_width)
    if int(flat_indices.numel()) == 0:
        empty_logits = query_logits if same_size else query_logits.float()
        return empty_logits[:, :, :0, :1]
    if same_size:
        return (
            query_logits
            .flatten(2)
            .index_select(2, flat_indices)
            .unsqueeze(-1)
        )

    target_y = torch.div(
        flat_indices,
        target_width,
        rounding_mode="floor",
    )
    target_x = flat_indices.remainder(target_width)
    grid_x = (
        2.0 * (target_x.float() + 0.5) / float(target_width) - 1.0
    )
    grid_y = (
        2.0 * (target_y.float() + 0.5) / float(target_height) - 1.0
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
        1,
        -1,
        1,
        2,
    )
    return F.grid_sample(
        query_logits.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _remap_selected_to_sparse_indices(
    selected,
    *,
    union_flat_idx: torch.Tensor,
) -> dict[int, SelectedClassEntry]:
    remapped = {}
    for cls_idx, entry in selected.items():
        flat_idx = entry.flat_idx.to(
            device=union_flat_idx.device,
            dtype=torch.long,
        )
        negative_flat_idx = entry.negative_flat_idx
        if negative_flat_idx is not None:
            negative_flat_idx = torch.searchsorted(
                union_flat_idx,
                negative_flat_idx.to(
                    device=union_flat_idx.device,
                    dtype=torch.long,
                ),
            )
        remapped[int(cls_idx)] = SelectedClassEntry(
            flat_idx=torch.searchsorted(union_flat_idx, flat_idx),
            weights=entry.weights,
            class_weight=entry.class_weight,
            class_reliability=entry.class_reliability,
            positive_targets=entry.positive_targets,
            positive_loss_scales=entry.positive_loss_scales,
            negative_flat_idx=negative_flat_idx,
            negative_weights=entry.negative_weights,
            negative_targets=entry.negative_targets,
            mask_loss_scale=entry.mask_loss_scale,
        )
    return remapped


def _sparse_class_logits_for_selected_loss(
    adapter,
    query_logits: torch.Tensor,
    *,
    selected,
    target_size: tuple[int, int],
    canonical_query_ids: tuple[int, ...] | None,
    query_ids: tuple[int, ...],
    sparse_plan: SparseSelectedLossPlan | None = None,
) -> tuple[torch.Tensor, dict[int, SelectedClassEntry]]:
    if sparse_plan is None:
        sparse_plan = _build_sparse_selected_loss_plan(
            selected,
            target_size=target_size,
            device=query_logits.device,
        )
    if tuple(target_size) != sparse_plan.target_size:
        raise ValueError(
            f"sparse plan target size {sparse_plan.target_size} does not match "
            f"requested target size {tuple(target_size)}"
        )
    if sparse_plan.union_flat_idx.device != query_logits.device:
        raise ValueError(
            f"sparse plan is on {sparse_plan.union_flat_idx.device}, "
            f"but query logits are on {query_logits.device}"
        )
    sampled_query_logits = _sample_query_logits_at_flat_indices(
        query_logits,
        flat_indices=sparse_plan.union_flat_idx,
        target_size=target_size,
    )
    class_logits = _class_logits_for_selected_loss(
        adapter,
        sampled_query_logits,
        target_size=tuple(sampled_query_logits.shape[-2:]),
        canonical_query_ids=canonical_query_ids,
        query_ids=query_ids,
    )
    return class_logits, sparse_plan.selected


def _presence_logits_for_selected_loss(
    adapter,
    presence_logits: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None = None,
    query_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if canonical_query_ids is not None:
        if query_ids is None:
            query_index = _canonical_query_index(
                canonical_query_ids,
                num_queries=int(presence_logits.shape[1]),
                device=presence_logits.device,
            )
        else:
            channel_by_query_id = {
                int(query_id): channel for channel, query_id in enumerate(query_ids)
            }
            missing = [
                int(query_id)
                for query_id in canonical_query_ids
                if int(query_id) not in channel_by_query_id
            ]
            if missing:
                raise ValueError(
                    f"canonical query ids {tuple(missing)} are missing from query subset"
                )
            query_index = torch.tensor(
                [channel_by_query_id[int(query_id)] for query_id in canonical_query_ids],
                device=presence_logits.device,
                dtype=torch.long,
            )
        return presence_logits.index_select(1, query_index)
    return adapter.aggregate_presence_logits(presence_logits)


def _active_replay_query_indices(
    grad_query_logits: torch.Tensor,
    grad_presence_logits: torch.Tensor | None,
) -> tuple[int, ...]:
    semantic_active = grad_query_logits.detach().ne(0).flatten(2).any(dim=2).any(dim=0)
    if grad_presence_logits is not None:
        semantic_active = semantic_active | grad_presence_logits.detach().ne(0).any(dim=0)
    return tuple(
        int(index)
        for index in torch.nonzero(semantic_active, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def _tta_step_loss(
    *,
    adapter,
    query_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    selected,
    target_size: tuple[int, int],
    loss_config,
    canonical_query_ids: tuple[int, ...] | None,
    query_ids: tuple[int, ...],
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None = None,
    sparse_plan: SparseSelectedLossPlan | None = None,
    adaptation_mode: str = "classwise",
    query_idx_list: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    selected_for_loss = selected
    if adaptation_mode == "independent_queries":
        if query_idx_list is None:
            raise ValueError("independent query loss requires query_idx_list")
        if bool(getattr(loss_config, "sparse_selected_logits", False)):
            if sparse_plan is None:
                sparse_plan = _build_sparse_selected_loss_plan(
                    selected,
                    target_size=target_size,
                    device=query_logits.device,
                )
            query_loss_logits = _sample_query_logits_at_flat_indices(
                query_logits,
                flat_indices=sparse_plan.union_flat_idx,
                target_size=target_size,
            )
            selected_for_loss = sparse_plan.selected
        else:
            query_loss_logits = query_logits
            if tuple(query_loss_logits.shape[-2:]) != tuple(target_size):
                query_loss_logits = F.interpolate(
                    query_loss_logits.float(),
                    size=tuple(target_size),
                    mode="bilinear",
                    align_corners=False,
                )
        total_loss = selected_query_soft_bce_loss(
            query_loss_logits,
            selected_for_loss,
            query_ids=query_ids,
            query_idx_list=query_idx_list,
            low_score_neg_weight=loss_config.low_score_neg_weight,
        )
    elif adaptation_mode == "classwise":
        if bool(getattr(loss_config, "sparse_selected_logits", False)):
            class_logits, selected_for_loss = (
                _sparse_class_logits_for_selected_loss(
                    adapter,
                    query_logits,
                    selected=selected,
                    target_size=target_size,
                    canonical_query_ids=canonical_query_ids,
                    query_ids=query_ids,
                    sparse_plan=sparse_plan,
                )
            )
        else:
            class_logits = _class_logits_for_selected_loss(
                adapter,
                query_logits,
                target_size=target_size,
                canonical_query_ids=canonical_query_ids,
                query_ids=query_ids,
            )
        total_loss = selected_class_soft_bce_loss(
            class_logits,
            selected_for_loss,
            low_score_neg_weight=loss_config.low_score_neg_weight,
        )
    else:
        raise ValueError(f"unsupported adaptation mode: {adaptation_mode}")
    total_loss = float(loss_config.selected_loss_weight) * total_loss
    if presence_ids is None:
        return total_loss
    if presence_targets is None:
        raise ValueError("presence targets are required when presence ids are provided")
    if presence_weights is None:
        presence_weights = torch.ones_like(presence_targets)
    else:
        presence_weights = presence_weights.to(
            device=presence_targets.device,
            dtype=presence_targets.dtype,
        )
    if presence_weights.shape != presence_targets.shape:
        raise ValueError("presence weights must match presence targets")

    if adaptation_mode == "independent_queries":
        channel_by_query_id = {
            int(query_id): channel for channel, query_id in enumerate(query_ids)
        }
        presence_query_ids = [
            int(query_id) for query_id in presence_ids.detach().cpu().tolist()
        ]
        missing = [
            query_id
            for query_id in presence_query_ids
            if query_id not in channel_by_query_id
        ]
        if missing:
            raise ValueError(f"presence queries {tuple(missing)} are missing from query_ids")
        presence_channels = torch.tensor(
            [channel_by_query_id[query_id] for query_id in presence_query_ids],
            device=presence_logits.device,
            dtype=torch.long,
        )
        per_query_presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits[0].index_select(0, presence_channels).float(),
            presence_targets,
            reduction="none",
        )
        presence_losses_by_class: dict[int, list[torch.Tensor]] = {}
        presence_weights_by_class: dict[int, list[torch.Tensor]] = {}
        for query_id, query_loss, query_weight in zip(
            presence_query_ids,
            per_query_presence_loss,
            presence_weights,
        ):
            class_id = int(query_idx_list[query_id])
            presence_losses_by_class.setdefault(class_id, []).append(query_loss)
            presence_weights_by_class.setdefault(class_id, []).append(query_weight)
        presence_loss = torch.stack(
            [
                (
                    torch.stack(presence_losses_by_class[class_id])
                    * torch.stack(presence_weights_by_class[class_id])
                ).mean()
                for class_id in sorted(presence_losses_by_class)
            ]
        ).mean()
    else:
        class_presence_logits = _presence_logits_for_selected_loss(
            adapter,
            presence_logits,
            canonical_query_ids=canonical_query_ids,
            query_ids=query_ids,
        )
        per_class_presence_loss = F.binary_cross_entropy_with_logits(
            class_presence_logits[0, presence_ids].float(),
            presence_targets,
            reduction="none",
        )
        presence_loss = (per_class_presence_loss * presence_weights).mean()
    return total_loss + float(loss_config.presence_loss_weight) * presence_loss


def _routing_score_channels(
    scores: torch.Tensor,
    *,
    query_ids: tuple[int, ...],
    canonical_query_ids: tuple[int, ...] | None,
    adaptation_mode: str,
    query_idx_list: list[int] | tuple[int, ...] | None,
) -> dict[int, torch.Tensor]:
    if scores.ndim != 3 or int(scores.shape[0]) != 1:
        raise ValueError(
            f"expected routing scores [1,Q,K], got {tuple(scores.shape)}"
        )
    if len(query_ids) != int(scores.shape[1]):
        raise ValueError("routing query id/score channel length mismatch")
    channel_by_query_id = {
        int(query_id): channel for channel, query_id in enumerate(query_ids)
    }
    if adaptation_mode == "independent_queries":
        return {
            int(query_id): scores[0, channel]
            for channel, query_id in enumerate(query_ids)
        }
    if adaptation_mode != "classwise":
        raise ValueError(f"unsupported adaptation mode: {adaptation_mode}")
    if canonical_query_ids is not None:
        missing = [
            int(query_id)
            for query_id in canonical_query_ids
            if int(query_id) not in channel_by_query_id
        ]
        if missing:
            raise ValueError(
                f"canonical query ids {tuple(missing)} are missing from query subset"
            )
        return {
            class_id: scores[0, channel_by_query_id[int(query_id)]]
            for class_id, query_id in enumerate(canonical_query_ids)
        }
    if query_idx_list is None:
        raise ValueError("classwise routing requires query_idx_list")
    channels_by_class: dict[int, list[int]] = {}
    for channel, query_id in enumerate(query_ids):
        if int(query_id) < 0 or int(query_id) >= len(query_idx_list):
            raise ValueError(f"query {query_id} has no class mapping")
        class_id = int(query_idx_list[int(query_id)])
        channels_by_class.setdefault(class_id, []).append(channel)
    return {
        class_id: scores[0, channels].amax(dim=0)
        for class_id, channels in channels_by_class.items()
    }


def _empty_branch_routing_counts() -> dict[str, int | float]:
    return {
        "evaluated_pixels": 0,
        "semantic_winner_pixels": 0,
        "instance_winner_pixels": 0,
        "tie_pixels": 0,
        "semantic_score_sum": 0.0,
        "instance_score_sum": 0.0,
        "fused_score_sum": 0.0,
        "teacher_target_sum": 0.0,
    }


def _update_branch_routing_counts(
    counts: dict[str, int | float],
    *,
    semantic: torch.Tensor,
    instance: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    semantic = semantic.detach().float()
    instance = instance.detach().float()
    targets = targets.detach().to(device=semantic.device, dtype=torch.float32)
    semantic_winner = semantic > instance + 1e-6
    instance_winner = instance > semantic + 1e-6
    tie = ~(semantic_winner | instance_winner)
    counts["evaluated_pixels"] += int(semantic.numel())
    counts["semantic_winner_pixels"] += int(semantic_winner.sum().item())
    counts["instance_winner_pixels"] += int(instance_winner.sum().item())
    counts["tie_pixels"] += int(tie.sum().item())
    counts["semantic_score_sum"] += float(semantic.double().sum().item())
    counts["instance_score_sum"] += float(instance.double().sum().item())
    counts["fused_score_sum"] += float(
        torch.maximum(semantic, instance).double().sum().item()
    )
    counts["teacher_target_sum"] += float(targets.double().sum().item())


def _student_branch_routing_step(
    *,
    student,
    selected,
    canonical_query_ids: tuple[int, ...] | None,
    query_ids: tuple[int, ...],
    adaptation_mode: str,
    query_idx_list: list[int] | tuple[int, ...] | None,
    include_negative: bool,
    step_index: int,
) -> dict:
    semantic_channels = _routing_score_channels(
        student.semantic_scores,
        query_ids=query_ids,
        canonical_query_ids=canonical_query_ids,
        adaptation_mode=adaptation_mode,
        query_idx_list=query_idx_list,
    )
    instance_channels = _routing_score_channels(
        student.instance_scores,
        query_ids=query_ids,
        canonical_query_ids=canonical_query_ids,
        adaptation_mode=adaptation_mode,
        query_idx_list=query_idx_list,
    )
    overall = _empty_branch_routing_counts()
    positive = _empty_branch_routing_counts()
    negative = _empty_branch_routing_counts()
    per_class: dict[str, dict] = {}
    for selected_id, entry in selected.items():
        selected_id = int(selected_id)
        if selected_id not in semantic_channels:
            raise ValueError(
                f"selected id {selected_id} is missing from routing channels"
            )
        semantic = semantic_channels[selected_id]
        instance = instance_channels[selected_id]
        class_overall = _empty_branch_routing_counts()
        class_positive = _empty_branch_routing_counts()
        class_negative = _empty_branch_routing_counts()

        positive_idx = entry.flat_idx.to(
            device=semantic.device,
            dtype=torch.long,
        )
        if int(positive_idx.numel()) > 0:
            positive_targets = entry.positive_targets
            if positive_targets is None:
                positive_targets = torch.ones(
                    positive_idx.shape,
                    device=semantic.device,
                    dtype=torch.float32,
                )
            for counts in (overall, positive, class_overall, class_positive):
                _update_branch_routing_counts(
                    counts,
                    semantic=semantic.index_select(0, positive_idx),
                    instance=instance.index_select(0, positive_idx),
                    targets=positive_targets,
                )

        if include_negative and entry.negative_flat_idx is not None:
            negative_idx = entry.negative_flat_idx.to(
                device=semantic.device,
                dtype=torch.long,
            )
            if int(negative_idx.numel()) > 0:
                negative_targets = entry.negative_targets
                if negative_targets is None:
                    negative_targets = torch.zeros(
                        negative_idx.shape,
                        device=semantic.device,
                        dtype=torch.float32,
                    )
                for counts in (
                    overall,
                    negative,
                    class_overall,
                    class_negative,
                ):
                    _update_branch_routing_counts(
                        counts,
                        semantic=semantic.index_select(0, negative_idx),
                        instance=instance.index_select(0, negative_idx),
                        targets=negative_targets,
                    )
        per_class[str(selected_id)] = {
            **class_overall,
            "positive": class_positive,
            "negative": class_negative,
        }

    return {
        "step": int(step_index),
        **overall,
        "prompts_without_kept_instances": int(
            student.kept_instance_counts.eq(0).sum().item()
        ),
        "positive": positive,
        "negative": negative,
        "per_class": per_class,
    }


def _run_direct_update_steps(
    *,
    adapter,
    optimizer,
    backbone_out,
    steps: int,
    loss_query_ids: tuple[int, ...],
    selected,
    target_size: tuple[int, int],
    loss_config,
    canonical_query_ids: tuple[int, ...] | None,
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None = None,
    grad_clip: float,
    first_step_outputs: tuple[torch.Tensor, torch.Tensor] | None,
    first_step_filtered_raw_context=None,
    adaptation_mode: str = "classwise",
    query_idx_list: list[int] | tuple[int, ...] | None = None,
    student_score_mode: str = "semantic_raw",
    mask_chunk: int = 1,
    branch_diagnostics: list[dict] | None = None,
    probability_reconstruction_query_ids: tuple[int, ...] = (),
) -> float:
    last_step_loss = None
    reusable_outputs = first_step_outputs
    reusable_filtered_raw_context = first_step_filtered_raw_context
    filtered_raw_student = _is_filtered_raw_student_mode(student_score_mode)
    reconstruct_all_probabilities = (
        student_score_mode == "filtered_raw_probability_reconstruction"
    )
    if filtered_raw_student and int(steps) > 0:
        if reusable_outputs is not None:
            raise ValueError(
                "filtered raw fusion student received semantic teacher outputs"
            )
        sparse_plan = _build_sparse_selected_loss_plan(
            selected,
            target_size=target_size,
            device=adapter.lora_params[0].device,
        )
    else:
        sparse_plan = _sparse_plan_for_update(
            selected=selected,
            target_size=target_size,
            device=adapter.lora_params[0].device,
            loss_config=loss_config,
            steps=steps,
        )
    for step_index in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        if filtered_raw_student:
            clear_lora_weight_cache(adapter.encoder)
            student_kwargs = {
                "query_ids": loss_query_ids,
                "flat_indices": sparse_plan.union_flat_idx,
                "target_size": target_size,
                "mask_chunk": mask_chunk,
            }
            if reconstruct_all_probabilities:
                student_kwargs[
                    "probability_reconstruction_query_ids"
                ] = loss_query_ids
            elif probability_reconstruction_query_ids:
                student_kwargs[
                    "probability_reconstruction_query_ids"
                ] = probability_reconstruction_query_ids
            if step_index == 0 and reusable_filtered_raw_context is not None:
                student_kwargs[
                    "precomputed_context"
                ] = reusable_filtered_raw_context
            student = adapter.forward_filtered_raw_student(
                backbone_out,
                **student_kwargs,
            )
            query_logits = student.query_logits
            presence_logits = student.presence_logits
            step_selected = sparse_plan.selected
            step_target_size = tuple(query_logits.shape[-2:])
            step_sparse_plan = None
            step_loss_config = replace(
                loss_config,
                sparse_selected_logits=False,
            )
            if branch_diagnostics is not None:
                branch_diagnostics.append(
                    _student_branch_routing_step(
                        student=student,
                        selected=step_selected,
                        canonical_query_ids=canonical_query_ids,
                        query_ids=loss_query_ids,
                        adaptation_mode=adaptation_mode,
                        query_idx_list=query_idx_list,
                        include_negative=(
                            float(loss_config.low_score_neg_weight) > 0.0
                        ),
                        step_index=step_index,
                    )
                )
        elif step_index == 0 and reusable_outputs is not None:
            query_logits, presence_logits = reusable_outputs
            step_selected = selected
            step_target_size = target_size
            step_sparse_plan = sparse_plan
            step_loss_config = loss_config
        else:
            clear_lora_weight_cache(adapter.encoder)
            query_logits, presence_logits = adapter.forward_queries(
                backbone_out,
                grad=True,
                out_size=None,
                query_ids=loss_query_ids,
            )
            step_selected = selected
            step_target_size = target_size
            step_sparse_plan = sparse_plan
            step_loss_config = loss_config
        total_step_loss = _tta_step_loss(
            adapter=adapter,
            query_logits=query_logits,
            presence_logits=presence_logits,
            selected=step_selected,
            target_size=step_target_size,
            loss_config=step_loss_config,
            canonical_query_ids=canonical_query_ids,
            query_ids=loss_query_ids,
            presence_ids=presence_ids,
            presence_targets=presence_targets,
            presence_weights=presence_weights,
            sparse_plan=step_sparse_plan,
            adaptation_mode=adaptation_mode,
            query_idx_list=query_idx_list,
        )
        total_step_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.lora_params, grad_clip)
        optimizer.step()
        last_step_loss = total_step_loss.detach()
        reusable_outputs = None
        reusable_filtered_raw_context = None
    if last_step_loss is None:
        return 0.0
    return float(last_step_loss.item())


def _teacher_prompt_gradient_vector(
    *,
    adapter,
    backbone_out,
    loss_query_ids: tuple[int, ...],
    selected,
    target_size: tuple[int, int],
    loss_config,
    canonical_query_ids: tuple[int, ...],
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """Return one supervision gradient without touching optimizer state."""

    clear_lora_weight_cache(adapter.encoder)
    query_logits, presence_logits = adapter.forward_queries(
        backbone_out,
        grad=True,
        out_size=None,
        query_ids=loss_query_ids,
    )
    sparse_plan = _sparse_plan_for_update(
        selected=selected,
        target_size=target_size,
        device=adapter.lora_params[0].device,
        loss_config=loss_config,
        steps=1,
    )
    total_loss = _tta_step_loss(
        adapter=adapter,
        query_logits=query_logits,
        presence_logits=presence_logits,
        selected=selected,
        target_size=target_size,
        loss_config=loss_config,
        canonical_query_ids=canonical_query_ids,
        query_ids=loss_query_ids,
        presence_ids=presence_ids,
        presence_targets=presence_targets,
        presence_weights=presence_weights,
        sparse_plan=sparse_plan,
        adaptation_mode="classwise",
    )
    gradients = torch.autograd.grad(
        total_loss,
        tuple(adapter.lora_params),
        allow_unused=True,
    )
    flat_gradient = flatten_parameter_gradients(
        adapter.lora_params,
        gradients,
    )
    loss_value = float(total_loss.detach().item())
    clear_lora_weight_cache(adapter.encoder)
    return loss_value, flat_gradient


def _canonical_class_words(path: str, *, expected_num_classes: int) -> list[str]:
    from segearthov3_segmentor import get_cls_idx

    resolved_path = resolve_project_path(path)
    if resolved_path is None:
        raise ValueError("canonical classname path is required")
    canonical_words, canonical_class_ids = get_cls_idx(resolved_path)
    if canonical_class_ids != list(range(len(canonical_words))):
        raise ValueError(
            "canonical classname path must contain exactly one prompt per class"
        )
    if len(canonical_words) != int(expected_num_classes):
        raise ValueError(
            f"canonical classname path contains {len(canonical_words)} classes, "
            f"expected {int(expected_num_classes)}"
        )
    return canonical_words


def _canonical_query_ids_for_prompt_view(
    adapter: SAM3LoRAAdapter,
    prompt: PromptViewConfig,
) -> tuple[int, ...] | None:
    if not prompt.needs_canonical_view:
        return None
    if not prompt.canonical_classname_path:
        return build_first_query_ids_by_class(
            adapter.query_idx_list,
            num_classes=adapter.num_classes,
        )
    canonical_words = _canonical_class_words(
        prompt.canonical_classname_path,
        expected_num_classes=adapter.num_classes,
    )
    return build_canonical_query_ids(
        adapter.query_words,
        adapter.query_idx_list,
        canonical_words,
    )


def _canonical_query_ids_for_path(
    adapter: SAM3LoRAAdapter,
    classname_path: str | None,
) -> tuple[int, ...]:
    if not classname_path:
        return build_first_query_ids_by_class(
            adapter.query_idx_list,
            num_classes=adapter.num_classes,
        )
    canonical_words = _canonical_class_words(
        classname_path,
        expected_num_classes=adapter.num_classes,
    )
    return build_canonical_query_ids(
        adapter.query_words,
        adapter.query_idx_list,
        canonical_words,
    )


class PromptQueryViews(NamedTuple):
    default_canonical_query_ids: tuple[int, ...] | None
    mining_canonical_query_ids: tuple[int, ...] | None
    loss_canonical_query_ids: tuple[int, ...] | None
    loss_query_ids: tuple[int, ...]


def _resolve_prompt_query_views(
    adapter: SAM3LoRAAdapter,
    prompt: PromptViewConfig,
) -> PromptQueryViews:
    default_ids = _canonical_query_ids_for_prompt_view(adapter, prompt)
    if prompt.adaptation_mode == "independent_queries":
        if default_ids is None:
            default_ids = _canonical_query_ids_for_path(
                adapter,
                prompt.canonical_classname_path,
            )
        return PromptQueryViews(
            default_canonical_query_ids=default_ids,
            mining_canonical_query_ids=default_ids,
            loss_canonical_query_ids=None,
            loss_query_ids=tuple(range(adapter.num_queries)),
        )
    mining_ids = None
    if prompt.mining_view == "canonical":
        mining_ids = _canonical_query_ids_for_path(
            adapter,
            prompt.mining_classname_path or prompt.canonical_classname_path,
        )
    loss_ids = None
    if prompt.loss_view == "canonical":
        loss_ids = _canonical_query_ids_for_path(
            adapter,
            prompt.loss_classname_path or prompt.canonical_classname_path,
        )
    return PromptQueryViews(
        default_canonical_query_ids=default_ids,
        mining_canonical_query_ids=mining_ids,
        loss_canonical_query_ids=loss_ids,
        loss_query_ids=(
            tuple(int(query_id) for query_id in loss_ids)
            if loss_ids is not None
            else tuple(range(adapter.num_queries))
        ),
    )


def _resolve_final_heads(
    prompt: PromptViewConfig,
    *,
    num_classes: int,
) -> tuple[FinalHeadConfig, ...]:
    if prompt.final_heads:
        heads = prompt.final_heads
    elif prompt.auto_canonical_final_head:
        heads = (
            FinalHeadConfig(name="synonym_full"),
            FinalHeadConfig(
                name="canonical_only",
                canonical_class_ids=tuple(range(int(num_classes))),
            ),
        )
    else:
        heads = (FinalHeadConfig(name="synonym_full"),)
    for head in heads:
        invalid_ids = [
            class_id
            for class_id in head.canonical_class_ids
            if class_id < 0 or class_id >= int(num_classes)
        ]
        if invalid_ids:
            raise ValueError(
                f"final head {head.name!r} canonical class ids {tuple(invalid_ids)} "
                f"exceed the class range [0, {int(num_classes) - 1}]"
            )
    return tuple(heads)


def _cross_time_parity_head_names(
    heads: Sequence[FinalHeadConfig],
    *,
    num_classes: int,
) -> tuple[str, str]:
    all_classes = tuple(range(int(num_classes)))
    raw = [
        head.name
        for head in heads
        if not head.canonical_class_ids
        and head.fusion_mode == "synonym_max"
    ]
    canonical = [
        head.name
        for head in heads
        if head.canonical_class_ids == all_classes
        and head.fusion_mode == "synonym_max"
    ]
    if len(raw) != 1 or len(canonical) != 1:
        raise ValueError(
            "synonym-TTA overlap requires one raw synonym head and one "
            "all-canonical head for parity"
        )
    return raw[0], canonical[0]


def _canonical_query_ids_for_final_heads(
    adapter: SAM3LoRAAdapter,
    heads: tuple[FinalHeadConfig, ...],
    *,
    default_canonical_query_ids: tuple[int, ...] | None,
) -> dict[str, tuple[int, ...] | None]:
    query_ids_by_head: dict[str, tuple[int, ...] | None] = {}
    for head in heads:
        if head.classname_path is None:
            query_ids_by_head[head.name] = default_canonical_query_ids
            continue
        canonical_words = _canonical_class_words(
            head.classname_path,
            expected_num_classes=adapter.num_classes,
        )
        query_ids_by_head[head.name] = build_canonical_query_ids(
            adapter.query_words,
            adapter.query_idx_list,
            canonical_words,
        )
    return query_ids_by_head


class AliasFusionRequirements(NamedTuple):
    needs_pre_reliability: bool
    needs_post_reliability: bool
    needs_teacher_reference: bool


def _alias_fusion_requirements(
    heads: tuple[FinalHeadConfig, ...],
) -> AliasFusionRequirements:
    modes = {head.fusion_mode for head in heads}
    return AliasFusionRequirements(
        needs_pre_reliability=bool(
            modes
            & {
                "topk_reliability",
                "post_tta_reliability",
                "frozen_alias_residual",
            }
        ),
        needs_post_reliability="post_tta_reliability" in modes,
        needs_teacher_reference="frozen_alias_residual" in modes,
    )


def _class_scores_for_final_head(
    adapter,
    *,
    query_scores: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None,
    alias_reliability: torch.Tensor | None = None,
    post_alias_reliability: torch.Tensor | None = None,
    reference_query_scores: torch.Tensor | None = None,
    head: FinalHeadConfig,
) -> torch.Tensor:
    if head.fusion_mode == "synonym_max" and not head.canonical_class_ids:
        return class_scores_from_query_scores(
            query_scores=query_scores,
            query_idx_list=adapter.query_idx_list,
            num_classes=adapter.num_classes,
        )
    if canonical_query_ids is None:
        raise ValueError(
            f"final head {head.name!r} requires canonical query ids"
        )
    if head.fusion_mode in {"topk_reliability", "post_tta_reliability"}:
        reliability = (
            post_alias_reliability
            if head.fusion_mode == "post_tta_reliability"
            else alias_reliability
        )
        if reliability is None:
            raise ValueError(
                f"final head {head.name!r} requires alias reliability weights"
            )
        class_scores = class_scores_from_query_scores_with_alias_reliability(
            query_scores=query_scores,
            query_idx_list=adapter.query_idx_list,
            num_classes=adapter.num_classes,
            canonical_query_ids=canonical_query_ids,
            query_weights=reliability,
        )
    elif head.fusion_mode == "frozen_alias_residual":
        if alias_reliability is None or reference_query_scores is None:
            raise ValueError(
                f"final head {head.name!r} requires pre-TTA reliability and "
                "reference query scores"
            )
        class_scores = class_scores_from_query_scores_with_frozen_alias_residual(
            query_scores=query_scores,
            reference_query_scores=reference_query_scores,
            query_idx_list=adapter.query_idx_list,
            num_classes=adapter.num_classes,
            canonical_query_ids=canonical_query_ids,
            query_weights=alias_reliability,
        )
    else:
        class_scores = class_scores_from_query_scores_with_canonical_overrides(
            query_scores=query_scores,
            query_idx_list=adapter.query_idx_list,
            num_classes=adapter.num_classes,
            canonical_query_ids=canonical_query_ids,
            canonical_class_ids=head.canonical_class_ids,
        )

    if head.fusion_mode != "synonym_max" and head.canonical_class_ids:
        class_scores = class_scores.clone()
        for class_id in head.canonical_class_ids:
            class_scores[:, class_id] = query_scores[
                :, canonical_query_ids[class_id]
            ].float()
    return class_scores


def _class_presence_for_final_head(
    adapter,
    *,
    presence_logits: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None,
    head: FinalHeadConfig,
) -> torch.Tensor:
    class_presence = adapter.aggregate_presence(presence_logits).float()
    if not head.canonical_class_ids:
        return class_presence
    if canonical_query_ids is None:
        raise ValueError(
            f"final head {head.name!r} requires canonical query ids"
        )
    class_presence = class_presence.clone()
    query_presence = presence_logits.float().sigmoid()
    for class_id in head.canonical_class_ids:
        class_presence[:, class_id] = query_presence[
            :, canonical_query_ids[class_id]
        ]
    return class_presence


def _output_state_for_final_head(
    adapter,
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    prediction: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None,
    alias_reliability: torch.Tensor | None,
    post_alias_reliability: torch.Tensor | None,
    reference_query_scores: torch.Tensor | None,
    head: FinalHeadConfig,
) -> dict:
    class_scores = _class_scores_for_final_head(
        adapter,
        query_scores=query_scores,
        canonical_query_ids=canonical_query_ids,
        alias_reliability=alias_reliability,
        post_alias_reliability=post_alias_reliability,
        reference_query_scores=reference_query_scores,
        head=head,
    )
    class_presence = _class_presence_for_final_head(
        adapter,
        presence_logits=presence_logits,
        canonical_query_ids=canonical_query_ids,
        head=head,
    )
    return extract_output_state(
        pred=prediction,
        class_scores=class_scores,
        class_presence=class_presence,
        num_classes=adapter.num_classes,
    )


def _predict_final_head(
    adapter,
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None,
    alias_reliability: torch.Tensor | None = None,
    post_alias_reliability: torch.Tensor | None = None,
    reference_query_scores: torch.Tensor | None = None,
    head: FinalHeadConfig,
    prob_thd: float,
    bg_idx: int,
    out_size: tuple[int, int],
) -> torch.Tensor:
    del presence_logits
    class_scores = _class_scores_for_final_head(
        adapter,
        query_scores=query_scores,
        canonical_query_ids=canonical_query_ids,
        alias_reliability=alias_reliability,
        post_alias_reliability=post_alias_reliability,
        reference_query_scores=reference_query_scores,
        head=head,
    )
    prediction_hook = getattr(adapter, "predict_class_scores", None)
    if prediction_hook is not None:
        return prediction_hook(
            class_scores,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
            out_size=out_size,
        )
    return predict_from_class_scores(
        class_scores,
        prob_thd=prob_thd,
        bg_idx=bg_idx,
        out_size=out_size,
    )


def _predict_final_heads(
    adapter,
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    canonical_query_ids: tuple[int, ...] | None,
    canonical_query_ids_by_head: dict[
        str, tuple[int, ...] | None
    ] | None = None,
    alias_reliability: torch.Tensor | None,
    post_alias_reliability: torch.Tensor | None = None,
    reference_query_scores: torch.Tensor | None = None,
    heads: tuple[FinalHeadConfig, ...],
    prob_thd: float,
    bg_idx: int,
    out_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    return {
        head.name: _predict_final_head(
            adapter,
            query_scores=query_scores,
            presence_logits=presence_logits,
            canonical_query_ids=(
                canonical_query_ids_by_head.get(head.name, canonical_query_ids)
                if canonical_query_ids_by_head is not None
                else canonical_query_ids
            ),
            alias_reliability=alias_reliability,
            post_alias_reliability=post_alias_reliability,
            reference_query_scores=reference_query_scores,
            head=head,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
            out_size=out_size,
        )
        for head in heads
    }


def _final_head_image_fields(
    *,
    heads: tuple[FinalHeadConfig, ...],
    baseline_preds: dict[str, torch.Tensor],
    tta_preds: dict[str, torch.Tensor],
    gt: torch.Tensor,
    num_classes: int,
    device: str | torch.device,
) -> tuple[dict, dict[str, dict]]:
    primary_head = heads[0]
    primary_fields = _image_miou_fields(
        baseline_pred=baseline_preds[primary_head.name],
        tta_pred=tta_preds[primary_head.name],
        gt=gt,
        num_classes=num_classes,
        device=device,
    )
    alternative_fields = {
        head.name: _image_miou_fields(
            baseline_pred=baseline_preds[head.name],
            tta_pred=tta_preds[head.name],
            gt=gt,
            num_classes=num_classes,
            device=device,
        )
        for head in heads[1:]
    }
    return primary_fields, alternative_fields


class _FinalHeadProtectionEvaluation(NamedTuple):
    predictions: dict[str, torch.Tensor]
    decision: ProtectionDecision | None
    raw_primary: dict
    raw_alternatives: dict[str, dict]
    primary: dict
    alternatives: dict[str, dict]


def _protection_record_fields(
    *,
    decision: ProtectionDecision | None,
    raw_tta_preds: dict[str, torch.Tensor],
    protected_preds: dict[str, torch.Tensor],
) -> dict | None:
    if decision is None:
        return None
    fields = decision.to_dict()
    if decision.action == "unsupported_pixel_fallback":
        fields.update(
            summarize_prediction_replacements(
                raw_tta_predictions=raw_tta_preds,
                protected_predictions=protected_preds,
            )
        )
    return fields


def _attach_raw_protection_fields(
    *,
    record: dict,
    alternatives: dict[str, dict],
    raw_primary: dict,
    raw_alternatives: dict[str, dict],
    decision: ProtectionDecision | None,
    raw_tta_preds: dict[str, torch.Tensor],
    protected_preds: dict[str, torch.Tensor],
) -> None:
    if decision is None:
        return
    record.update(_raw_tta_image_fields(raw_primary))
    record["protection"] = _protection_record_fields(
        decision=decision,
        raw_tta_preds=raw_tta_preds,
        protected_preds=protected_preds,
    )
    for head_name, head_fields in alternatives.items():
        head_fields.update(
            _raw_tta_image_fields(raw_alternatives[head_name])
        )


def _evaluate_final_head_protection(
    *,
    config: ProtectionConfig,
    primary_head_name: str,
    selected_class_ids,
    heads: tuple[FinalHeadConfig, ...],
    baseline_preds: dict[str, torch.Tensor],
    raw_tta_preds: dict[str, torch.Tensor],
    gt: torch.Tensor,
    num_classes: int,
    device: str | torch.device,
) -> _FinalHeadProtectionEvaluation:
    raw_primary, raw_alternatives = _final_head_image_fields(
        heads=heads,
        baseline_preds=baseline_preds,
        tta_preds=raw_tta_preds,
        gt=gt,
        num_classes=num_classes,
        device=device,
    )
    predictions, decision = _protect_final_head_predictions(
        config=config,
        primary_head_name=primary_head_name,
        selected_class_ids=selected_class_ids,
        baseline_preds=baseline_preds,
        raw_tta_preds=raw_tta_preds,
    )
    primary, alternatives = _final_head_image_fields(
        heads=heads,
        baseline_preds=baseline_preds,
        tta_preds=predictions,
        gt=gt,
        num_classes=num_classes,
        device=device,
    )
    return _FinalHeadProtectionEvaluation(
        predictions=predictions,
        decision=decision,
        raw_primary=raw_primary,
        raw_alternatives=raw_alternatives,
        primary=primary,
        alternatives=alternatives,
    )


class _OracleTrialResult(NamedTuple):
    candidate: OracleCandidate
    loss: float
    query_scores: torch.Tensor
    presence_logits: torch.Tensor
    predictions: dict[str, torch.Tensor]
    post_alias_reliability: torch.Tensor | None
    post_alias_fusion_fields: dict | None
    lora_snapshot: list[torch.Tensor] | None


def _observe_reference_objective(
    *,
    adapter,
    optimizer,
    backbone_out,
    initial_lora: list[torch.Tensor],
    loss_query_ids: tuple[int, ...],
    selected: dict[int, SelectedClassEntry],
    target_size: tuple[int, int],
    loss_config,
    canonical_query_ids: tuple[int, ...] | None,
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None,
) -> dict:
    restore_lora_params(adapter.lora_params, initial_lora)
    reset_optimizer_state(optimizer)
    optimizer.zero_grad(set_to_none=True)
    clear_lora_weight_cache(adapter.encoder)
    query_logits, presence_logits = adapter.forward_queries(
        backbone_out,
        grad=True,
        out_size=None,
        query_ids=loss_query_ids,
    )
    class_logits = _class_logits_for_selected_loss(
        adapter,
        query_logits,
        target_size=target_size,
        canonical_query_ids=canonical_query_ids,
        query_ids=loss_query_ids,
    )
    selected_total, selected_terms = selected_class_soft_bce_loss_terms(
        class_logits,
        selected,
        low_score_neg_weight=loss_config.low_score_neg_weight,
    )
    presence_loss = selected_total.new_zeros(())
    if presence_ids is not None:
        if presence_targets is None:
            raise ValueError(
                "presence targets are required when presence ids are provided"
            )
        if presence_weights is None:
            presence_weights = torch.ones_like(presence_targets)
        class_presence_logits = _presence_logits_for_selected_loss(
            adapter,
            presence_logits,
            canonical_query_ids=canonical_query_ids,
            query_ids=loss_query_ids,
        )
        per_class_presence_loss = F.binary_cross_entropy_with_logits(
            class_presence_logits[0, presence_ids].float(),
            presence_targets,
            reduction="none",
        )
        presence_loss = (per_class_presence_loss * presence_weights).mean()
    total = float(loss_config.selected_loss_weight) * selected_total + float(
        loss_config.presence_loss_weight
    ) * presence_loss
    total.backward()
    features = {
        "total_loss": float(total.detach().item()),
        "positive_bce": float(selected_terms.positive.detach().item()),
        "negative_bce": float(selected_terms.negative.detach().item()),
        "presence_loss": float(presence_loss.detach().item()),
        **gradient_norm_features(adapter.lora_params_by_layer),
    }
    optimizer.zero_grad(set_to_none=True)
    restore_lora_params(adapter.lora_params, initial_lora)
    reset_optimizer_state(optimizer)
    clear_lora_weight_cache(adapter.encoder)
    return features


def _iter_strength_oracle_trials(
    *,
    adapter,
    optimizer,
    backbone_out,
    initial_lora: list[torch.Tensor],
    candidates: tuple[OracleCandidate, ...],
    reference_candidate: OracleCandidate,
    target_class_scores_by_power: dict[float, torch.Tensor],
    selected: dict[int, SelectedClassEntry],
    target_size: tuple[int, int],
    loss_config,
    optim_config,
    runtime,
    loss_query_ids: tuple[int, ...],
    canonical_loss_query_ids: tuple[int, ...] | None,
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None,
    inference_presence_gate_power: float,
    inference_score_fusion_mode: str,
    canonical_query_ids: tuple[int, ...] | None,
    canonical_query_ids_by_head: dict[str, tuple[int, ...] | None],
    alias_reliability: torch.Tensor | None,
    alias_requirements,
    reference_query_scores: torch.Tensor | None,
    final_heads: tuple[FinalHeadConfig, ...],
    mining: MiningConfig,
):
    base_lrs = capture_optimizer_lrs(optimizer)
    try:
        for candidate in candidates:
            restore_lora_params(adapter.lora_params, initial_lora)
            reset_optimizer_state(optimizer)
            clear_lora_weight_cache(adapter.encoder)
            apply_optimizer_lr_multiplier(
                optimizer,
                base_lrs,
                candidate.lr_multiplier,
            )
            candidate_selected = rebuild_selected_targets(
                selected,
                target_scores=target_class_scores_by_power[
                    candidate.target_presence_power
                ],
                minimum=loss_config.positive_target_min,
                maximum=loss_config.positive_target_max,
                offset=loss_config.positive_target_offset,
            )
            loss_value = _run_direct_update_steps(
                adapter=adapter,
                optimizer=optimizer,
                backbone_out=backbone_out,
                steps=optim_config.steps,
                loss_query_ids=loss_query_ids,
                selected=candidate_selected,
                target_size=target_size,
                loss_config=loss_config,
                canonical_query_ids=canonical_loss_query_ids,
                presence_ids=presence_ids,
                presence_targets=presence_targets,
                presence_weights=presence_weights,
                grad_clip=optim_config.grad_clip,
                first_step_outputs=None,
                adaptation_mode="classwise",
                query_idx_list=adapter.query_idx_list,
                student_score_mode=loss_config.student_score_mode,
                mask_chunk=runtime.mask_chunk,
            )
            clear_lora_weight_cache(adapter.encoder)
            with torch.no_grad():
                query_scores, presence_logits = adapter.forward_fused_scores(
                    backbone_out,
                    out_size=target_size,
                    presence_gate_power=inference_presence_gate_power,
                    mask_chunk=runtime.mask_chunk,
                    score_fusion_mode=inference_score_fusion_mode,
                )
            post_alias_reliability = None
            post_alias_fusion_fields = None
            if alias_requirements.needs_post_reliability:
                if canonical_query_ids is None:
                    raise ValueError(
                        "post-TTA oracle inference requires canonical query ids"
                    )
                post_alias = estimate_alias_reliability(
                    query_scores=query_scores,
                    query_presence=presence_logits[0].float().sigmoid(),
                    query_idx_list=adapter.query_idx_list,
                    canonical_query_ids=canonical_query_ids,
                    num_classes=adapter.num_classes,
                    prob_thd=mining.prob_thd,
                    tau_pos=mining.tau_pos,
                    rho=mining.rho,
                    kmax=mining.kmax,
                    n_min=mining.n_min,
                    bg_idx=mining.bg_idx,
                )
                post_alias_reliability = post_alias.query_weights
                post_alias_fusion_fields = _alias_fusion_image_fields(
                    post_alias,
                    query_words=adapter.query_words,
                )
            predictions = _predict_final_heads(
                adapter,
                query_scores=query_scores,
                presence_logits=presence_logits,
                canonical_query_ids=canonical_query_ids,
                canonical_query_ids_by_head=canonical_query_ids_by_head,
                alias_reliability=alias_reliability,
                post_alias_reliability=post_alias_reliability,
                reference_query_scores=(
                    reference_query_scores
                    if alias_requirements.needs_teacher_reference
                    else None
                ),
                heads=final_heads,
                prob_thd=mining.prob_thd,
                bg_idx=mining.bg_idx,
                out_size=target_size,
            )
            yield _OracleTrialResult(
                candidate=candidate,
                loss=loss_value,
                query_scores=query_scores.detach(),
                presence_logits=presence_logits.detach(),
                predictions={
                    name: prediction.detach()
                    for name, prediction in predictions.items()
                },
                post_alias_reliability=(
                    post_alias_reliability.detach()
                    if post_alias_reliability is not None
                    else None
                ),
                post_alias_fusion_fields=post_alias_fusion_fields,
                lora_snapshot=(
                    snapshot_lora_params(adapter.lora_params)
                    if candidate == reference_candidate
                    else None
                ),
            )
    finally:
        restore_optimizer_lrs(optimizer, base_lrs)
        restore_lora_params(adapter.lora_params, initial_lora)
        reset_optimizer_state(optimizer)
        clear_lora_weight_cache(adapter.encoder)


class _TeacherTrialSupervision(NamedTuple):
    canonical_query_ids: tuple[int, ...]
    loss_query_ids: tuple[int, ...]
    selected: dict[int, SelectedClassEntry]
    present: torch.Tensor
    class_presence: torch.Tensor


def _teacher_trial_supervision(
    *,
    query_scores: torch.Tensor,
    target_query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    query_idx_list: tuple[int, ...] | list[int],
    mining_canonical_query_ids: tuple[int, ...],
    loss_canonical_query_ids: tuple[int, ...],
    class_id: int,
    candidate_query_id: int,
    valid: torch.Tensor,
    mining,
    loss_config,
) -> _TeacherTrialSupervision:
    candidate_mining_ids = replace_teacher_query(
        canonical_query_ids=mining_canonical_query_ids,
        query_idx_list=query_idx_list,
        class_id=class_id,
        candidate_query_id=candidate_query_id,
    )
    candidate_loss_ids = replace_teacher_query(
        canonical_query_ids=loss_canonical_query_ids,
        query_idx_list=query_idx_list,
        class_id=class_id,
        candidate_query_id=candidate_query_id,
    )
    views = build_prompt_class_views(
        query_scores=query_scores,
        presence_logits=presence_logits,
        query_idx_list=query_idx_list,
        canonical_query_ids=candidate_mining_ids,
        num_classes=len(candidate_mining_ids),
    )
    target_views = build_prompt_class_views(
        query_scores=target_query_scores,
        presence_logits=presence_logits,
        query_idx_list=query_idx_list,
        canonical_query_ids=candidate_mining_ids,
        num_classes=len(candidate_mining_ids),
    )
    component_gates = None
    if mining.component_gate:
        component_gates = build_component_gate_masks(
            class_scores=views.canonical_scores,
            class_presence=views.canonical_presence,
            valid=valid,
            mining=mining,
        )
    selected, present = select_classwise_pixels(
        class_scores=views.canonical_scores,
        class_presence=views.canonical_presence,
        raw_pred=views.canonical_scores.argmax(dim=0).long(),
        valid=valid,
        mining=mining,
        loss=loss_config,
        target_scores=target_views.canonical_scores,
        component_gate_masks=component_gates,
    )
    return _TeacherTrialSupervision(
        canonical_query_ids=candidate_mining_ids,
        loss_query_ids=candidate_loss_ids,
        selected=selected,
        present=present,
        class_presence=views.canonical_presence,
    )


def _classwise_presence_supervision(
    *,
    present: torch.Tensor,
    class_presence: torch.Tensor,
    num_classes: int,
    tau_neg: float,
    device: torch.device,
    selected: dict[int, SelectedClassEntry] | None = None,
    competition_reliability_mode: str = "none",
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    present_ids = [int(value) for value in present.detach().cpu().tolist()]
    present_set = set(present_ids)
    positive_ids = sorted(
        class_id
        for class_id in present_set
        if 0 <= class_id < int(num_classes)
    )
    presence_values = class_presence.detach().cpu().tolist()
    negative_ids = [
        class_id
        for class_id in range(int(num_classes))
        if class_id not in present_set
        and float(presence_values[class_id]) <= float(tau_neg)
    ]
    if not positive_ids and not negative_ids:
        return None, None, None
    if competition_reliability_mode == "none":
        positive_weights = [1.0] * len(positive_ids)
    elif competition_reliability_mode == "score_ratio":
        selected = selected or {}
        positive_weights = []
        for class_id in positive_ids:
            entry = selected.get(class_id)
            reliability = (
                entry.class_reliability
                if entry is not None
                else None
            )
            positive_weights.append(
                float(reliability.detach().item())
                if reliability is not None
                else 0.0
            )
    else:
        raise ValueError(
            "unknown competition_reliability_mode="
            f"{competition_reliability_mode}"
        )
    return (
        torch.tensor(
            positive_ids + negative_ids,
            device=device,
            dtype=torch.long,
        ),
        torch.tensor(
            [1.0] * len(positive_ids) + [0.0] * len(negative_ids),
            device=device,
            dtype=torch.float32,
        ),
        torch.tensor(
            positive_weights + [1.0] * len(negative_ids),
            device=device,
            dtype=torch.float32,
        ),
    )


def _filter_presence_supervision(
    presence_ids: torch.Tensor | None,
    presence_targets: torch.Tensor | None,
    presence_weights: torch.Tensor | None,
    *,
    class_id: int,
    keep_class: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if presence_ids is None:
        return None, None, None
    if presence_targets is None:
        raise ValueError("presence targets are required with presence ids")
    if presence_weights is None:
        raise ValueError("presence weights are required with presence ids")
    mask = presence_ids.eq(int(class_id))
    if not keep_class:
        mask = ~mask
    if not bool(mask.any()):
        return None, None, None
    return presence_ids[mask], presence_targets[mask], presence_weights[mask]


def _run_teacher_prompt_gradient_diagnostic(
    *,
    adapter,
    backbone_out,
    teacher_query_scores: torch.Tensor,
    teacher_target_query_scores: torch.Tensor,
    teacher_presence_logits: torch.Tensor,
    mining_canonical_query_ids: tuple[int, ...],
    loss_canonical_query_ids: tuple[int, ...],
    class_ids: tuple[int, ...],
    selected,
    present: torch.Tensor,
    class_presence: torch.Tensor,
    valid: torch.Tensor,
    mining,
    loss_config,
) -> tuple[list[dict], list[dict]]:
    device = adapter.lora_params[0].device
    all_presence_ids = None
    all_presence_targets = None
    all_presence_weights = None
    if float(loss_config.presence_loss_weight) > 0.0:
        all_presence_ids, all_presence_targets, all_presence_weights = (
            _classwise_presence_supervision(
                present=present,
                class_presence=class_presence,
                num_classes=adapter.num_classes,
                tau_neg=mining.tau_neg,
                device=device,
                selected=selected,
                competition_reliability_mode=(
                    mining.competition_reliability_mode
                ),
            )
        )

    rows: list[dict] = []
    pair_rows: list[dict] = []
    for class_id in class_ids:
        class_id = int(class_id)
        stable_selected = {
            int(selected_class): entry
            for selected_class, entry in selected.items()
            if int(selected_class) != class_id
        }
        (
            stable_presence_ids,
            stable_presence_targets,
            stable_presence_weights,
        ) = (
            _filter_presence_supervision(
                all_presence_ids,
                all_presence_targets,
                all_presence_weights,
                class_id=class_id,
                keep_class=False,
            )
        )
        stable_active = bool(stable_selected) or stable_presence_ids is not None
        stable_loss = None
        stable_gradient = None
        if stable_active:
            stable_loss, stable_gradient = _teacher_prompt_gradient_vector(
                adapter=adapter,
                backbone_out=backbone_out,
                loss_query_ids=loss_canonical_query_ids,
                selected=stable_selected,
                target_size=tuple(teacher_query_scores.shape[-2:]),
                loss_config=loss_config,
                canonical_query_ids=loss_canonical_query_ids,
                presence_ids=stable_presence_ids,
                presence_targets=stable_presence_targets,
                presence_weights=stable_presence_weights,
            )

        candidate_gradients: dict[int, torch.Tensor] = {}
        candidate_fields: dict[int, dict] = {}
        candidate_ids = [
            query_id
            for query_id, mapped_class in enumerate(adapter.query_idx_list)
            if int(mapped_class) == class_id
        ]
        for candidate_query_id in candidate_ids:
            supervision = _teacher_trial_supervision(
                query_scores=teacher_query_scores,
                target_query_scores=teacher_target_query_scores,
                presence_logits=teacher_presence_logits,
                query_idx_list=adapter.query_idx_list,
                mining_canonical_query_ids=mining_canonical_query_ids,
                loss_canonical_query_ids=loss_canonical_query_ids,
                class_id=class_id,
                candidate_query_id=candidate_query_id,
                valid=valid,
                mining=mining,
                loss_config=loss_config,
            )
            candidate_selected = {
                class_id: supervision.selected[class_id]
            } if class_id in supervision.selected else {}
            candidate_presence_ids = None
            candidate_presence_targets = None
            candidate_presence_weights = None
            if float(loss_config.presence_loss_weight) > 0.0:
                (
                    candidate_all_ids,
                    candidate_all_targets,
                    candidate_all_weights,
                ) = (
                    _classwise_presence_supervision(
                        present=supervision.present,
                        class_presence=supervision.class_presence,
                        num_classes=adapter.num_classes,
                        tau_neg=mining.tau_neg,
                        device=device,
                        selected=supervision.selected,
                        competition_reliability_mode=(
                            mining.competition_reliability_mode
                        ),
                    )
                )
                (
                    candidate_presence_ids,
                    candidate_presence_targets,
                    candidate_presence_weights,
                ) = (
                    _filter_presence_supervision(
                        candidate_all_ids,
                        candidate_all_targets,
                        candidate_all_weights,
                        class_id=class_id,
                        keep_class=True,
                    )
                )
            candidate_loss, candidate_gradient = (
                _teacher_prompt_gradient_vector(
                    adapter=adapter,
                    backbone_out=backbone_out,
                    loss_query_ids=supervision.loss_query_ids,
                    selected=candidate_selected,
                    target_size=tuple(teacher_query_scores.shape[-2:]),
                    loss_config=loss_config,
                    canonical_query_ids=supervision.loss_query_ids,
                    presence_ids=candidate_presence_ids,
                    presence_targets=candidate_presence_targets,
                    presence_weights=candidate_presence_weights,
                )
            )
            candidate_gradients[int(candidate_query_id)] = candidate_gradient
            entry = candidate_selected.get(class_id)
            candidate_fields[int(candidate_query_id)] = {
                "loss": float(candidate_loss),
                "selected_pixels": (
                    int(entry.flat_idx.numel()) if entry is not None else 0
                ),
                "present": bool(
                    (supervision.present == class_id).any().item()
                ),
                "presence_target": (
                    None
                    if candidate_presence_targets is None
                    else float(candidate_presence_targets[0].item())
                ),
            }

        summary = summarize_prompt_gradient_vectors(
            candidate_gradients,
            stable_gradient=stable_gradient,
        )
        stable_selected_pixels = int(
            sum(
                int(entry.flat_idx.numel())
                for entry in stable_selected.values()
            )
        )
        for candidate_query_id in candidate_ids:
            row = {
                "class_id": class_id,
                "query_id": int(candidate_query_id),
                "query": str(adapter.query_words[candidate_query_id]),
                "configured_teacher": (
                    int(mining_canonical_query_ids[class_id])
                    == int(candidate_query_id)
                ),
                "stable_loss": stable_loss,
                "stable_selected_pixels": stable_selected_pixels,
                **candidate_fields[int(candidate_query_id)],
                **summary["candidates"][int(candidate_query_id)],
            }
            rows.append(row)
        pair_rows.extend(
            {"class_id": class_id, **pair}
            for pair in summary["pairs"]
        )
    return rows, pair_rows


def _run_teacher_prompt_trials(
    *,
    adapter,
    optimizer,
    backbone_out,
    teacher_query_scores: torch.Tensor,
    teacher_inference_query_scores: torch.Tensor,
    teacher_target_query_scores: torch.Tensor,
    teacher_presence_logits: torch.Tensor,
    mining_canonical_query_ids: tuple[int, ...],
    loss_canonical_query_ids: tuple[int, ...],
    class_ids: tuple[int, ...],
    trial_steps: int,
    valid: torch.Tensor,
    gt: torch.Tensor,
    teacher_size: tuple[int, int],
    mining,
    loss_config,
    optim_config,
    runtime,
    inference_presence_gate_power: float,
    inference_score_fusion_mode: str,
    canonical_query_ids: tuple[int, ...] | None,
    canonical_query_ids_by_head: dict[str, tuple[int, ...] | None],
    alias_reliability: torch.Tensor | None,
    reference_query_scores: torch.Tensor | None,
    final_heads: tuple[FinalHeadConfig, ...],
    baseline_preds: dict[str, torch.Tensor],
) -> list[dict]:
    base_lora = snapshot_lora_params(adapter.lora_params)
    rows = []
    try:
        for class_id in class_ids:
            candidate_ids = [
                query_id
                for query_id, mapped_class in enumerate(adapter.query_idx_list)
                if int(mapped_class) == int(class_id)
            ]
            for candidate_query_id in candidate_ids:
                restore_lora_params(adapter.lora_params, base_lora)
                reset_optimizer_state(optimizer)
                clear_lora_weight_cache(adapter.encoder)
                supervision = _teacher_trial_supervision(
                    query_scores=teacher_query_scores,
                    target_query_scores=teacher_target_query_scores,
                    presence_logits=teacher_presence_logits,
                    query_idx_list=adapter.query_idx_list,
                    mining_canonical_query_ids=mining_canonical_query_ids,
                    loss_canonical_query_ids=loss_canonical_query_ids,
                    class_id=class_id,
                    candidate_query_id=candidate_query_id,
                    valid=valid,
                    mining=mining,
                    loss_config=loss_config,
                )
                selected_pixels = int(
                    sum(
                        int(entry.flat_idx.numel())
                        for entry in supervision.selected.values()
                    )
                )
                selected_classes = int(
                    sum(
                        int(entry.flat_idx.numel()) > 0
                        for entry in supervision.selected.values()
                    )
                )
                skip_reason = _selection_skip_reason(
                    selected_pixels=selected_pixels,
                    selected_classes=selected_classes,
                    mining=mining,
                )
                loss_value = 0.0
                adapted = skip_reason is None
                if adapted:
                    presence_ids = None
                    presence_targets = None
                    presence_weights = None
                    if float(loss_config.presence_loss_weight) > 0.0:
                        presence_ids, presence_targets, presence_weights = (
                            _classwise_presence_supervision(
                                present=supervision.present,
                                class_presence=supervision.class_presence,
                                num_classes=adapter.num_classes,
                                tau_neg=mining.tau_neg,
                                device=adapter.lora_params[0].device,
                                selected=supervision.selected,
                                competition_reliability_mode=(
                                    mining.competition_reliability_mode
                                ),
                            )
                        )
                    loss_value = _run_direct_update_steps(
                        adapter=adapter,
                        optimizer=optimizer,
                        backbone_out=backbone_out,
                        steps=trial_steps,
                        loss_query_ids=supervision.loss_query_ids,
                        selected=supervision.selected,
                        target_size=tuple(teacher_query_scores.shape[-2:]),
                        loss_config=loss_config,
                        canonical_query_ids=supervision.loss_query_ids,
                        presence_ids=presence_ids,
                        presence_targets=presence_targets,
                        presence_weights=presence_weights,
                        grad_clip=optim_config.grad_clip,
                        first_step_outputs=None,
                        adaptation_mode="classwise",
                        query_idx_list=adapter.query_idx_list,
                        student_score_mode=loss_config.student_score_mode,
                        mask_chunk=runtime.mask_chunk,
                    )
                    clear_lora_weight_cache(adapter.encoder)
                    with torch.no_grad():
                        post_query_scores, post_presence_logits = (
                            adapter.forward_fused_scores(
                                backbone_out,
                                out_size=teacher_size,
                                presence_gate_power=(
                                    inference_presence_gate_power
                                ),
                                mask_chunk=runtime.mask_chunk,
                                score_fusion_mode=(
                                    inference_score_fusion_mode
                                ),
                            )
                        )
                else:
                    post_query_scores = teacher_inference_query_scores
                    post_presence_logits = teacher_presence_logits

                transfer = measure_teacher_trial_transfer(
                    pre_query_scores=teacher_inference_query_scores,
                    post_query_scores=post_query_scores,
                    query_idx_list=adapter.query_idx_list,
                    canonical_query_ids=supervision.canonical_query_ids,
                    class_id=class_id,
                    prob_thd=mining.prob_thd,
                    anchor_k=mining.kmax,
                )
                trial_preds = _predict_final_heads(
                    adapter,
                    query_scores=post_query_scores,
                    presence_logits=post_presence_logits,
                    canonical_query_ids=canonical_query_ids,
                    canonical_query_ids_by_head=(
                        canonical_query_ids_by_head
                    ),
                    alias_reliability=alias_reliability,
                    post_alias_reliability=alias_reliability,
                    reference_query_scores=reference_query_scores,
                    heads=final_heads,
                    prob_thd=mining.prob_thd,
                    bg_idx=mining.bg_idx,
                    out_size=tuple(gt.shape[-2:]),
                )
                primary_fields, alternative_fields = (
                    _final_head_image_fields(
                        heads=final_heads,
                        baseline_preds=baseline_preds,
                        tta_preds=trial_preds,
                        gt=gt,
                        num_classes=adapter.num_classes,
                        device=adapter.lora_params[0].device,
                    )
                )
                gt_heads = {final_heads[0].name: primary_fields}
                gt_heads.update(alternative_fields)
                row = {
                    "query_id": int(candidate_query_id),
                    "query": str(adapter.query_words[candidate_query_id]),
                    "class_id": int(class_id),
                    "configured_teacher": (
                        int(mining_canonical_query_ids[class_id])
                        == int(candidate_query_id)
                    ),
                    "trial_steps": int(trial_steps),
                    "adapted": bool(adapted),
                    "skip_reason": skip_reason,
                    "selected_pixels": selected_pixels,
                    "selected_classes": selected_classes,
                    "loss": float(loss_value),
                    "transfer": transfer,
                    "gt_audit_final_heads": gt_heads,
                }
                rows.append(row)
    finally:
        restore_lora_params(adapter.lora_params, base_lora)
        reset_optimizer_state(optimizer)
        clear_lora_weight_cache(adapter.encoder)
    return rows


def _performance_fields(
    *,
    processed: int,
    elapsed_seconds: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, float]:
    elapsed_seconds = max(float(elapsed_seconds), 0.0)
    processed = max(int(processed), 0)
    return {
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            float(processed) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "seconds_per_image": (
            elapsed_seconds / float(processed) if processed > 0 else 0.0
        ),
        "peak_cuda_memory_mb": float(peak_cuda_memory_bytes) / (1024.0 * 1024.0),
    }


def _resolve_report_class_names(
    *,
    dataset,
    adapter,
    canonical_query_ids: Sequence[int] | None,
) -> list[str]:
    metainfo = dict(getattr(dataset, "metainfo", {}) or {})
    dataset_classes = list(metainfo.get("classes") or ())
    if len(dataset_classes) == int(adapter.num_classes):
        return [str(name) for name in dataset_classes]

    canonical_ids = tuple(canonical_query_ids or ())
    names: list[str] = []
    for class_id in range(int(adapter.num_classes)):
        query_id = (
            int(canonical_ids[class_id])
            if class_id < len(canonical_ids)
            else None
        )
        if query_id is None or not (0 <= query_id < len(adapter.query_words)):
            query_id = next(
                (
                    index
                    for index, mapped_class in enumerate(adapter.query_idx_list)
                    if int(mapped_class) == class_id
                ),
                None,
            )
        names.append(
            str(adapter.query_words[query_id])
            if query_id is not None
            else f"class_{class_id}"
        )
    return names


def _format_report_number(value, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{number:+.2f}" if signed else f"{number:.2f}"


def _format_report_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> list[str]:
    string_rows = [[str(value) for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in string_rows:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(value))
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: Sequence[object]) -> str:
        return "| " + " | ".join(
            str(value).ljust(width)
            for value, width in zip(row, widths, strict=True)
        ) + " |"

    return [
        separator,
        format_row(headers),
        separator,
        *(format_row(row) for row in string_rows),
        separator,
    ]


def _format_final_report(result: Mapping[str, object]) -> str:
    lines = [
        "",
        "=" * 80,
        "LoRA-TTA Final Report",
        "=" * 80,
        f"Config: {result.get('config', 'N/A')}",
        f"Split: {result.get('split', 'N/A')}",
        (
            f"Samples: processed={int(result.get('processed', 0))} "
            f"adapted={int(result.get('adapted', 0))} "
            f"skipped_empty={int(result.get('skipped_empty_pos', 0))} "
            f"skipped_few_classes={int(result.get('skipped_few_classes', 0))}"
        ),
        "",
        "Overall Metrics",
    ]
    overall_rows = []
    for label, key in (("mIoU", "miou"), ("mAcc", "macc"), ("aAcc", "aacc")):
        overall_rows.append(
            (
                label,
                _format_report_number(result.get(f"baseline_{key}")),
                _format_report_number(result.get(f"tta_{key}")),
                _format_report_number(result.get(f"delta_{key}"), signed=True),
            )
        )
    lines.extend(
        _format_report_table(
            ("Metric", "Baseline", "TTA", "Delta"),
            overall_rows,
        )
    )

    baseline_iou = list(result.get("baseline_per_class_iou", ()) or ())
    tta_iou = list(result.get("tta_per_class_iou", ()) or ())
    delta_iou = list(result.get("delta_per_class_iou", ()) or ())
    baseline_accuracy = list(
        result.get("baseline_per_class_accuracy", ()) or ()
    )
    tta_accuracy = list(result.get("tta_per_class_accuracy", ()) or ())
    delta_accuracy = list(
        result.get("delta_per_class_accuracy", ()) or ()
    )
    class_count = len(baseline_iou)
    class_names = list(result.get("class_names", ()) or ())
    if len(class_names) != class_count:
        class_names = [f"class_{class_id}" for class_id in range(class_count)]
    class_rows = []
    for class_id, class_name in enumerate(class_names):
        class_rows.append(
            (
                class_name,
                _format_report_number(baseline_iou[class_id]),
                _format_report_number(tta_iou[class_id]),
                _format_report_number(delta_iou[class_id], signed=True),
                _format_report_number(baseline_accuracy[class_id]),
                _format_report_number(tta_accuracy[class_id]),
                _format_report_number(delta_accuracy[class_id], signed=True),
            )
        )
    if class_rows:
        lines.extend(("", "Per-Class Metrics"))
        lines.extend(
            _format_report_table(
                (
                    "Class",
                    "IoU Before",
                    "IoU After",
                    "IoU Delta",
                    "Acc Before",
                    "Acc After",
                    "Acc Delta",
                ),
                class_rows,
            )
        )

    final_heads = dict(result.get("final_heads", {}) or {})
    if len(final_heads) > 1:
        head_rows = []
        for head_name, raw_metrics in final_heads.items():
            metrics = dict(raw_metrics or {})
            head_rows.append(
                (
                    head_name,
                    _format_report_number(metrics.get("baseline_miou")),
                    _format_report_number(metrics.get("tta_miou")),
                    _format_report_number(metrics.get("delta_miou"), signed=True),
                    _format_report_number(metrics.get("baseline_macc")),
                    _format_report_number(metrics.get("tta_macc")),
                    _format_report_number(metrics.get("delta_macc"), signed=True),
                    _format_report_number(metrics.get("baseline_aacc")),
                    _format_report_number(metrics.get("tta_aacc")),
                    _format_report_number(metrics.get("delta_aacc"), signed=True),
                )
            )
        lines.extend(("", "Final Inference Heads"))
        lines.extend(
            _format_report_table(
                (
                    "Head",
                    "mIoU Before",
                    "mIoU After",
                    "mIoU Delta",
                    "mAcc Before",
                    "mAcc After",
                    "mAcc Delta",
                    "aAcc Before",
                    "aAcc After",
                    "aAcc Delta",
                ),
                head_rows,
            )
        )

    lines.extend(
        (
            "",
            "Adaptation",
            (
                f"avg_loss={float(result.get('avg_loss', 0.0)):.4f} "
                f"avg_selected_pixels={float(result.get('avg_selected_pixels', 0.0)):.1f} "
                f"avg_selected_classes={float(result.get('avg_selected_classes', 0.0)):.2f}"
            ),
        )
    )
    performance = dict(result.get("performance", {}) or {})
    lines.extend(
        (
            "",
            "Runtime",
            (
                f"elapsed={float(performance.get('elapsed_seconds', 0.0)):.2f}s "
                f"seconds_per_image={float(performance.get('seconds_per_image', 0.0)):.3f} "
                f"images_per_second={float(performance.get('images_per_second', 0.0)):.3f} "
                f"peak_cuda_memory={float(performance.get('peak_cuda_memory_mb', 0.0)):.2f}MB"
            ),
            "=" * 80,
        )
    )
    return "\n".join(lines)


def _resolve_teacher_first_step_reuse(optim_config) -> bool:
    requested = bool(
        getattr(optim_config, "reuse_teacher_first_step", False)
    )
    if requested and optim_config.backward_mode != "direct":
        raise ValueError(
            "teacher first-step reuse requires direct backward mode"
        )
    return requested and int(optim_config.steps) > 0


def _is_filtered_raw_student_mode(mode: str) -> bool:
    return str(mode) in {
        "filtered_raw_fusion",
        "filtered_raw_probability_reconstruction",
    }


def _resolve_student_execution(loss_config, optim_config) -> tuple[str, bool]:
    mode = str(getattr(loss_config, "student_score_mode", "semantic_raw"))
    if (
        _is_filtered_raw_student_mode(mode)
        and optim_config.backward_mode != "direct"
    ):
        raise ValueError(
            "filtered raw fusion student requires direct backward mode"
        )
    reuse = _resolve_teacher_first_step_reuse(optim_config)
    return mode, reuse


def _validate_contraction_gate_configuration(
    loss_config,
    prompt_config,
) -> None:
    if loss_config.contraction_gate_mode == "none":
        return
    if loss_config.contraction_gate_mode != "decision_boundary":
        raise ValueError(
            f"unsupported contraction gate: {loss_config.contraction_gate_mode}"
        )
    if not _is_filtered_raw_student_mode(loss_config.student_score_mode):
        raise ValueError(
            "decision-boundary contraction gate requires "
            "filtered_raw_fusion student mode"
        )
    if loss_config.positive_target_mode != "PTST":
        raise ValueError(
            "decision-boundary contraction gate requires PTST targets"
        )
    if prompt_config.adaptation_mode != "classwise":
        raise ValueError(
            "decision-boundary contraction gate requires classwise adaptation"
        )


def _aggregate_contraction_gate_reports(reports: list[dict]) -> dict:
    selected = sum(int(report["selected_pixels"]) for report in reports)
    contracting = sum(
        int(report["contracting_pixels"])
        for report in reports
    )
    gated = sum(int(report["gated_pixels"]) for report in reports)
    scale_sum = sum(
        float(report["mean_scale"]) * int(report["selected_pixels"])
        for report in reports
    )
    per_class_accumulators: dict[str, dict[str, float]] = {}
    for report in reports:
        for class_id, values in report.get("per_class", {}).items():
            accumulator = per_class_accumulators.setdefault(
                str(class_id),
                {
                    "selected_pixels": 0,
                    "contracting_pixels": 0,
                    "gated_pixels": 0,
                    "scale_sum": 0.0,
                },
            )
            class_selected = int(values["selected_pixels"])
            accumulator["selected_pixels"] += class_selected
            accumulator["contracting_pixels"] += int(
                values["contracting_pixels"]
            )
            accumulator["gated_pixels"] += int(values["gated_pixels"])
            accumulator["scale_sum"] += (
                float(values["mean_scale"]) * class_selected
            )
    per_class = {}
    for class_id, values in per_class_accumulators.items():
        class_selected = int(values["selected_pixels"])
        class_gated = int(values["gated_pixels"])
        per_class[class_id] = {
            "selected_pixels": class_selected,
            "contracting_pixels": int(
                values["contracting_pixels"]
            ),
            "gated_pixels": class_gated,
            "gated_fraction": float(
                class_gated / class_selected if class_selected else 0.0
            ),
            "mean_scale": float(
                values["scale_sum"] / class_selected
                if class_selected else 1.0
            ),
        }
    return {
        "enabled": True,
        "mode": "decision_boundary",
        "images": len(reports),
        "selected_pixels": selected,
        "contracting_pixels": contracting,
        "contracting_fraction": float(
            contracting / selected if selected else 0.0
        ),
        "gated_pixels": gated,
        "gated_fraction": float(gated / selected if selected else 0.0),
        "mean_scale": float(scale_sum / selected if selected else 1.0),
        "per_class": per_class,
    }


def _resolve_score_fusion_views(
    mining_config,
    loss_config,
) -> tuple[str, str, str]:
    mining_mode = str(mining_config.score_fusion_mode)
    inference_mode = str(
        mining_config.resolved_inference_score_fusion_mode
    )
    target_mode = (
        mining_mode
        if loss_config.soft_target_score_fusion_mode is None
        else str(loss_config.soft_target_score_fusion_mode)
    )
    return mining_mode, inference_mode, target_mode


class CleanTTAEngine:
    """Pure-source episodic LoRA TTA engine.

    This is the clean replacement for the current classwise-BCE wrapper path.
    It does not import `eval_lora_tta_structured.pyc`.
    """

    def __init__(
        self,
        config: TTAConfig,
        *,
        adapter_cls=None,
        diagnose_synonyms: bool = False,
        diagnose_prompt_routing: bool = False,
        diagnose_cross_time_only: bool = False,
        diagnose_synonym_tta_overlap: bool = False,
        boundary_widths: tuple[int, ...] = DEFAULT_BOUNDARY_RADII,
        diagnose_teacher_prompts: bool = False,
        diagnose_teacher_prompt_gradients: bool = False,
        teacher_diagnostic_classes: tuple[int, ...] = (),
        teacher_prompt_trial_steps: int = 0,
        diagnose_selected_flips: bool = False,
        diagnose_selection_conflicts: bool = False,
    ):
        self.config = config
        self.adapter_cls = adapter_cls or SAM3LoRAAdapter
        self.diagnose_prompt_routing = bool(diagnose_prompt_routing)
        self.diagnose_synonym_tta_overlap = bool(
            diagnose_synonym_tta_overlap
        )
        self.diagnose_cross_time_only = bool(
            diagnose_cross_time_only
            or self.diagnose_synonym_tta_overlap
        )
        self.boundary_widths = normalize_boundary_radii(boundary_widths)
        self.diagnose_teacher_prompts = bool(diagnose_teacher_prompts)
        self.diagnose_teacher_prompt_gradients = bool(
            diagnose_teacher_prompt_gradients
        )
        self.teacher_diagnostic_classes = tuple(
            dict.fromkeys(int(value) for value in teacher_diagnostic_classes)
        )
        self.teacher_prompt_trial_steps = int(teacher_prompt_trial_steps)
        self.diagnose_selected_flips = bool(diagnose_selected_flips)
        self.diagnose_selection_conflicts = bool(
            diagnose_selection_conflicts
        )
        if self.teacher_prompt_trial_steps < 0:
            raise ValueError("teacher_prompt_trial_steps must be non-negative")
        if (
            self.diagnose_prompt_routing
            and self.diagnose_cross_time_only
        ):
            raise ValueError(
                "prompt routing and cross-time/overlap diagnostics are "
                "mutually exclusive"
            )
        self.diagnose_synonyms = (
            bool(diagnose_synonyms)
            or self.diagnose_prompt_routing
            or self.diagnose_cross_time_only
        )

    def run(self) -> dict:
        run_started_at = time.perf_counter()
        runtime = self.config.runtime
        mining = self.config.mining
        loss_cfg = self.config.loss
        prompt_cfg = self.config.prompt
        optim_cfg = self.config.optim
        oracle_cfg = self.config.oracle
        protection_cfg = self.config.protection
        if (
            mining.competition_reliability_mode != "none"
            and prompt_cfg.adaptation_mode != "classwise"
        ):
            raise ValueError(
                "competition reliability currently requires classwise adaptation"
            )
        if protection_cfg.survival_enabled:
            if oracle_cfg.enabled:
                raise ValueError(
                    "class survival rerun cannot be combined with the GT oracle"
                )
            if optim_cfg.backward_mode != "direct":
                raise ValueError(
                    "class survival rerun currently requires direct backward mode"
                )
            if prompt_cfg.adaptation_mode != "classwise":
                raise ValueError(
                    "class survival rerun currently requires classwise adaptation"
                )
            if int(optim_cfg.steps) <= 0:
                raise ValueError("class survival rerun requires positive TTA steps")
            if (
                protection_cfg.survival_intervention
                == "old_probability_reconstruction"
                and not _is_filtered_raw_student_mode(
                    loss_cfg.student_score_mode
                )
            ):
                raise ValueError(
                    "old probability reconstruction survival intervention "
                    "requires filtered_raw_fusion student mode"
                )
        _validate_contraction_gate_configuration(loss_cfg, prompt_cfg)
        contraction_gate_enabled = (
            loss_cfg.contraction_gate_mode == "decision_boundary"
        )
        student_score_mode, reuse_teacher_first_step = _resolve_student_execution(loss_cfg, optim_cfg)
        if contraction_gate_enabled:
            reuse_teacher_first_step = False
        oracle_candidates: tuple[OracleCandidate, ...] = ()
        oracle_reference = OracleCandidate(
            oracle_cfg.reference_lr_multiplier,
            oracle_cfg.reference_target_presence_power,
        )
        if oracle_cfg.enabled:
            if loss_cfg.positive_target_mode != "PTST":
                raise ValueError(
                    "image-adaptive oracle requires PTST targets"
                )
            if optim_cfg.backward_mode != "direct":
                raise ValueError(
                    "image-adaptive oracle requires direct backward mode"
                )
            if prompt_cfg.adaptation_mode != "classwise":
                raise ValueError(
                    "image-adaptive oracle requires classwise adaptation"
                )
            if loss_cfg.sparse_selected_logits:
                raise ValueError(
                    "image-adaptive oracle does not yet support sparse selected logits"
                )
            if oracle_reference.lr_multiplier != 1.0:
                raise ValueError(
                    "image-adaptive oracle reference learning-rate multiplier must be 1"
                )
            if (
                oracle_reference.target_presence_power
                != float(loss_cfg.soft_target_presence_gate_power)
            ):
                raise ValueError(
                    "oracle reference target power must match the configured soft target power"
                )
            oracle_candidates = build_oracle_candidates(oracle_cfg)
            reuse_teacher_first_step = False
        if self.teacher_prompt_trial_steps > 0:
            if optim_cfg.backward_mode != "direct":
                raise ValueError(
                    "teacher prompt trials currently require direct backward mode"
                )
            if reuse_teacher_first_step:
                raise ValueError(
                    "teacher prompt trials cannot reuse the first-step graph"
                )
            if prompt_cfg.adaptation_mode != "classwise":
                raise ValueError(
                    "teacher prompt trials require classwise adaptation"
                )
            if (
                prompt_cfg.mining_view != "canonical"
                or prompt_cfg.loss_view != "canonical"
            ):
                raise ValueError(
                    "teacher prompt trials require canonical mining and loss"
                )
        if self.diagnose_teacher_prompt_gradients:
            if prompt_cfg.adaptation_mode != "classwise":
                raise ValueError(
                    "teacher prompt gradient diagnostics require classwise "
                    "adaptation"
                )
            if (
                prompt_cfg.mining_view != "canonical"
                or prompt_cfg.loss_view != "canonical"
            ):
                raise ValueError(
                    "teacher prompt gradient diagnostics require canonical "
                    "mining and loss"
                )
        set_random_seed(runtime.seed)
        dist_ctx = setup_distributed(runtime.device)
        if dist_ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dist_ctx.device)
        try:
            eval_cfg = load_eval_config(runtime.eval_config)
            dataset, dataloader_cfg = build_dataset_from_eval_config(
                eval_cfg,
                split=runtime.split,
                num_workers=runtime.num_workers,
            )
            loader = build_dataloader(
                dataset,
                dataloader_cfg,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
            )
            local_total = len(loader)
            if runtime.max_samples > 0:
                local_total = min(local_total, int(runtime.max_samples))
            adapter = self.adapter_cls(
                eval_cfg=eval_cfg,
                device=dist_ctx.device,
                lora_rank=optim_cfg.lora_rank,
                lora_alpha=optim_cfg.lora_alpha,
                lora_layers=optim_cfg.lora_layers,
                lora_layer_ranks=optim_cfg.lora_layer_ranks,
                lora_adapt_key=optim_cfg.lora_adapt_key,
                source_lora_path=getattr(optim_cfg, "source_lora_path", None),
                resolution=runtime.resolution,
                query_batch_size=runtime.query_batch_size,
                full_query_batch_size=runtime.full_query_batch_size,
            )
            prompt_query_views = _resolve_prompt_query_views(adapter, prompt_cfg)
            if (
                self.diagnose_selected_flips
                and prompt_cfg.adaptation_mode != "classwise"
            ):
                raise ValueError(
                    "selected TopK flip diagnostics require classwise adaptation"
                )
            canonical_query_ids = (
                prompt_query_views.default_canonical_query_ids
            )
            synonym_diagnostic = None
            prompt_routing_diagnostic = None
            cross_time_diagnostic = None
            cross_time_raw_head = None
            cross_time_canonical_head = None
            diagnostic_canonical_query_ids = None
            if (
                self.diagnose_synonyms
                or self.diagnose_prompt_routing
                or self.diagnose_cross_time_only
            ):
                diagnostic_canonical_query_ids = (
                    canonical_query_ids
                    if canonical_query_ids is not None
                    else _canonical_query_ids_for_path(
                        adapter,
                        prompt_cfg.canonical_classname_path,
                    )
                )
            if self.diagnose_synonyms:
                synonym_diagnostic = SynonymDiagnosticAccumulator(
                    query_words=adapter.query_words,
                    query_idx_list=adapter.query_idx_list,
                    canonical_query_ids=diagnostic_canonical_query_ids,
                    prob_thd=mining.prob_thd,
                    bg_idx=mining.bg_idx,
                    device=dist_ctx.device,
                )
            if self.diagnose_prompt_routing:
                prompt_routing_diagnostic = (
                    PromptRoutingDiagnosticAccumulator(
                        query_words=adapter.query_words,
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=diagnostic_canonical_query_ids,
                        prob_thd=mining.prob_thd,
                        tau_pos=mining.tau_pos,
                        rho=mining.rho,
                        kmax=mining.kmax,
                        n_min=mining.n_min,
                        bg_idx=mining.bg_idx,
                        device=dist_ctx.device,
                    )
                )
            if self.diagnose_cross_time_only:
                cross_time_diagnostic = (
                    CrossTimePromptCausalAccumulator(
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=diagnostic_canonical_query_ids,
                        num_classes=adapter.num_classes,
                        prob_thd=mining.prob_thd,
                        bg_idx=mining.bg_idx,
                        device=dist_ctx.device,
                        image_row_limit=0,
                        query_words=adapter.query_words,
                        boundary_radii=(
                            self.boundary_widths
                            if self.diagnose_synonym_tta_overlap
                            else ()
                        ),
                    )
                )
            mining_canonical_query_ids = (
                prompt_query_views.mining_canonical_query_ids
            )
            loss_canonical_query_ids = (
                prompt_query_views.loss_canonical_query_ids
            )
            loss_query_ids = prompt_query_views.loss_query_ids
            resolved_teacher_diagnostic_classes = self.teacher_diagnostic_classes
            if (
                self.diagnose_teacher_prompts
                or self.diagnose_teacher_prompt_gradients
                or self.teacher_prompt_trial_steps > 0
            ):
                if (
                    mining_canonical_query_ids is None
                    or loss_canonical_query_ids is None
                ):
                    raise ValueError(
                        "teacher prompt diagnostics require canonical mining "
                        "and loss query ids"
                    )
                if not resolved_teacher_diagnostic_classes:
                    resolved_teacher_diagnostic_classes = tuple(
                        class_id
                        for class_id in range(adapter.num_classes)
                        if sum(
                            int(mapped) == class_id
                            for mapped in adapter.query_idx_list
                        )
                        > 1
                    )
            final_heads = _resolve_final_heads(prompt_cfg, num_classes=adapter.num_classes)
            final_head_canonical_query_ids = _canonical_query_ids_for_final_heads(
                adapter,
                final_heads,
                default_canonical_query_ids=canonical_query_ids,
            )
            if self.diagnose_synonym_tta_overlap:
                (
                    cross_time_raw_head,
                    cross_time_canonical_head,
                ) = _cross_time_parity_head_names(
                    final_heads,
                    num_classes=adapter.num_classes,
                )
                if (
                    final_head_canonical_query_ids[
                        cross_time_canonical_head
                    ]
                    != tuple(diagnostic_canonical_query_ids)
                ):
                    raise ValueError(
                        "overlap canonical parity head uses different "
                        "canonical query ids"
                    )
            primary_final_head = final_heads[0]
            visualization_writer = None
            if runtime.visualization_dir is not None:
                dataset_metainfo = dict(getattr(dataset, "metainfo", {}) or {})
                visualization_writer = ComparisonVisualizationWriter(
                    output_dir=runtime.visualization_dir,
                    classes=dataset_metainfo.get("classes"),
                    palette=dataset_metainfo.get("palette"),
                    num_classes=adapter.num_classes,
                    max_side=runtime.visualization_max_side,
                    rank=dist_ctx.rank,
                    min_delta_miou=runtime.visualization_min_delta_miou,
                    delta_in_filename=runtime.visualization_delta_in_filename,
                )
                if dist_ctx.is_main:
                    print(
                        "[clean-tta] visualizations: "
                        f"{visualization_writer.output_dir}",
                        flush=True,
                    )
            alias_requirements = _alias_fusion_requirements(final_heads)
            prompt_view_metadata = {
                "adaptation_mode": prompt_cfg.adaptation_mode,
                "baseline_inference": primary_final_head.name,
                "final_inference": primary_final_head.name,
                "primary_final_head": primary_final_head.name,
                "final_heads": [
                    {
                        "name": head.name,
                        "canonical_class_ids": list(head.canonical_class_ids),
                        "classname_path": head.classname_path,
                        "fusion_mode": head.fusion_mode,
                        "canonical_query_ids": (
                            list(final_head_canonical_query_ids[head.name])
                            if final_head_canonical_query_ids[head.name] is not None
                            else None
                        ),
                    }
                    for head in final_heads
                ],
                "present_class": (
                    "independent_queries"
                    if prompt_cfg.adaptation_mode == "independent_queries"
                    else prompt_cfg.mining_view
                ),
                "topk": (
                    "independent_queries"
                    if prompt_cfg.adaptation_mode == "independent_queries"
                    else prompt_cfg.mining_view
                ),
                "soft_target": (
                    "independent_queries"
                    if prompt_cfg.adaptation_mode == "independent_queries"
                    else prompt_cfg.mining_view
                ),
                "selected_bce": (
                    "independent_queries"
                    if prompt_cfg.adaptation_mode == "independent_queries"
                    else prompt_cfg.loss_view
                ),
                "presence_loss": (
                    "independent_queries"
                    if prompt_cfg.adaptation_mode == "independent_queries"
                    else prompt_cfg.loss_view
                ),
                "canonical_classname_path": prompt_cfg.canonical_classname_path,
                "mining_classname_path": prompt_cfg.mining_classname_path,
                "loss_classname_path": prompt_cfg.loss_classname_path,
                "canonical_query_source": (
                    None
                    if canonical_query_ids is None
                    else (
                        "file"
                        if prompt_cfg.canonical_classname_path
                        else "first_prompt_per_class"
                    )
                ),
                "canonical_query_ids": (
                    list(canonical_query_ids) if canonical_query_ids is not None else None
                ),
                "mining_query_ids": (
                    list(mining_canonical_query_ids)
                    if mining_canonical_query_ids is not None
                    else None
                ),
                "student_query_ids": list(loss_query_ids),
                "loss_query_ids": list(loss_query_ids),
                "alias_fusion_requirements": alias_requirements._asdict(),
            }
            if dist_ctx.is_main:
                print(f"[clean-tta] prompt views: {prompt_view_metadata}", flush=True)
            optimizer_groups = build_lora_param_groups(
                adapter.lora_params_by_layer,
                default_lr=optim_cfg.lr,
                layer_lrs=optim_cfg.lora_layer_lrs,
            )
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=optim_cfg.lr,
                weight_decay=optim_cfg.weight_decay,
            )
            if dist_ctx.is_main:
                for group in optimizer_groups:
                    print(
                        "[clean-tta] optimizer group: "
                        f"layers={list(group['layers'])} lr={group['lr']}",
                        flush=True,
                    )
            initial_lora = snapshot_lora_params(adapter.lora_params)
            baseline_metrics = {
                head.name: ConfusionMatrix(adapter.num_classes, device=dist_ctx.device)
                for head in final_heads
            }
            tta_metrics = {
                head.name: ConfusionMatrix(adapter.num_classes, device=dist_ctx.device)
                for head in final_heads
            }
            raw_tta_metrics = (
                {
                    head.name: ConfusionMatrix(
                        adapter.num_classes,
                        device=dist_ctx.device,
                    )
                    for head in final_heads
                }
                if protection_cfg.enabled
                else {}
            )
            oracle_candidate_metrics = {
                candidate.key: {
                    head.name: ConfusionMatrix(
                        adapter.num_classes,
                        device=dist_ctx.device,
                    )
                    for head in final_heads
                }
                for candidate in oracle_candidates
            }
            oracle_raw_candidate_metrics = (
                {
                    candidate.key: {
                        head.name: ConfusionMatrix(
                            adapter.num_classes,
                            device=dist_ctx.device,
                        )
                        for head in final_heads
                    }
                    for candidate in oracle_candidates
                }
                if oracle_cfg.enabled and protection_cfg.enabled
                else {}
            )
            oracle_best_metrics = {
                head.name: ConfusionMatrix(
                    adapter.num_classes,
                    device=dist_ctx.device,
                )
                for head in final_heads
            } if oracle_cfg.enabled else {}
            oracle_win_counts = (
                {
                    "no_update": 0,
                    **{
                        candidate.key: 0
                        for candidate in oracle_candidates
                    },
                }
                if oracle_cfg.enabled
                else {}
            )
            images = []
            processed = 0
            adapted = 0
            skipped_empty = 0
            skipped_few = 0
            total_loss = 0.0
            total_selected = 0
            total_selected_classes = 0
            guarded_images = 0
            survival_rerun_images = 0
            inference_presence_gate_power = (
                mining.resolved_inference_presence_gate_power
            )
            (
                mining_score_fusion_mode,
                inference_score_fusion_mode,
                target_score_fusion_mode,
            ) = _resolve_score_fusion_views(mining, loss_cfg)
            score_fusion_views = {
                "mining": mining_score_fusion_mode,
                "soft_target": target_score_fusion_mode,
                "final_inference": inference_score_fusion_mode,
            }
            if dist_ctx.is_main:
                print(
                    f"[clean-tta] score fusion views: {score_fusion_views}",
                    flush=True,
                )
            target_presence_gate_power = (
                loss_cfg.soft_target_presence_gate_power
                if loss_cfg.positive_target_mode == "PTST"
                else mining.presence_gate_power
            )
            if dist_ctx.device.type == "cuda":
                torch.cuda.synchronize(dist_ctx.device)
            online_loop_started_at = time.perf_counter()
            for batch in loader:
                if runtime.max_samples > 0 and processed >= runtime.max_samples:
                    break
                processed += 1
                if optim_cfg.reset_lora_between_images:
                    restore_lora_params(adapter.lora_params, initial_lora)
                    reset_optimizer_state(optimizer)
                    clear_lora_weight_cache(adapter.encoder)
                adaptation_start_lora = (
                    snapshot_lora_params(adapter.lora_params)
                    if protection_cfg.survival_enabled
                    else None
                )
                image_batch, gt_labels = adapter.batch_to_tensors(batch)
                gt = gt_labels[0].long()
                teacher_size = tuple(gt.shape[-2:])
                backbone_out = adapter.encode_image(image_batch)
                clear_lora_weight_cache(adapter.encoder)
                if runtime.baseline_only:
                    if alias_requirements.needs_pre_reliability:
                        raise ValueError(
                            "baseline-only timing does not support reliability "
                            "final heads"
                        )
                    (
                        baseline_query_scores,
                        baseline_presence_logits,
                    ) = adapter.forward_fused_scores(
                        backbone_out,
                        out_size=teacher_size,
                        presence_gate_power=inference_presence_gate_power,
                        score_fusion_mode=inference_score_fusion_mode,
                        mask_chunk=runtime.mask_chunk,
                    )
                    baseline_preds = _predict_final_heads(
                        adapter,
                        query_scores=baseline_query_scores,
                        presence_logits=baseline_presence_logits,
                        canonical_query_ids=canonical_query_ids,
                        canonical_query_ids_by_head=(
                            final_head_canonical_query_ids
                        ),
                        alias_reliability=None,
                        heads=final_heads,
                        prob_thd=mining.prob_thd,
                        bg_idx=mining.bg_idx,
                        out_size=teacher_size,
                    )
                    for head in final_heads:
                        prediction = baseline_preds[head.name]
                        baseline_metrics[head.name].update(
                            prediction,
                            gt.to(dist_ctx.device),
                        )
                        tta_metrics[head.name].update(
                            prediction,
                            gt.to(dist_ctx.device),
                        )
                    sample_id = _sample_id_from_batch(batch)
                    primary_fields, alternative_fields = (
                        _final_head_image_fields(
                            heads=final_heads,
                            baseline_preds=baseline_preds,
                            tta_preds=baseline_preds,
                            gt=gt,
                            num_classes=adapter.num_classes,
                            device=dist_ctx.device,
                        )
                    )
                    image_record = {
                        "sample_id": sample_id,
                        "adapted": False,
                        "selected_pixels": 0,
                        "selected_classes": 0,
                        "present_classes": [],
                        "selected_per_class": {},
                        "skip_reason": "baseline_only",
                        **primary_fields,
                    }
                    if alternative_fields:
                        image_record["final_heads"] = alternative_fields
                    images.append(image_record)
                    print(
                        _format_progress(
                            rank=dist_ctx.rank,
                            world_size=dist_ctx.world_size,
                            processed=processed,
                            total=local_total,
                            sample_id=sample_id,
                            adapted=False,
                            selected_pixels=0,
                            selected_classes=0,
                            baseline_miou=image_record["baseline_miou"],
                            tta_miou=image_record["tta_miou"],
                            delta_miou=image_record["delta_miou"],
                            baseline_macc=image_record["baseline_macc"],
                            tta_macc=image_record["tta_macc"],
                            delta_macc=image_record["delta_macc"],
                        ),
                        flush=True,
                    )
                    continue
                first_step_query_logits = None
                first_step_presence_logits = None
                first_step_filtered_raw_context = None
                teacher_score_components = None
                teacher_target_query_scores_by_power = {}
                if oracle_cfg.enabled:
                    requested_specs = [
                        (
                            float(mining.presence_gate_power),
                            str(mining_score_fusion_mode),
                        ),
                        (
                            float(inference_presence_gate_power),
                            str(inference_score_fusion_mode),
                        ),
                        *[
                            (float(power), str(target_score_fusion_mode))
                            for power in oracle_cfg.target_presence_powers
                        ],
                    ]
                    unique_specs = tuple(dict.fromkeys(requested_specs))
                    (
                        oracle_score_views,
                        teacher_presence_logits,
                        teacher_score_components,
                    ) = adapter.forward_oracle_score_views(
                        backbone_out,
                        out_size=teacher_size,
                        presence_gate_powers=tuple(
                            power for power, _mode in unique_specs
                        ),
                        score_fusion_modes=tuple(
                            mode for _power, mode in unique_specs
                        ),
                        mask_chunk=runtime.mask_chunk,
                    )
                    score_by_spec = dict(
                        zip(unique_specs, oracle_score_views, strict=True)
                    )
                    teacher_query_scores = score_by_spec[
                        requested_specs[0]
                    ]
                    teacher_inference_query_scores = score_by_spec[
                        requested_specs[1]
                    ]
                    teacher_target_query_scores_by_power = {
                        float(power): score_by_spec[
                            (float(power), str(target_score_fusion_mode))
                        ]
                        for power in oracle_cfg.target_presence_powers
                    }
                    teacher_target_query_scores = (
                        teacher_target_query_scores_by_power[
                            oracle_reference.target_presence_power
                        ]
                    )
                elif reuse_teacher_first_step:
                    teacher_result = adapter.forward_teacher_scores(
                        backbone_out,
                        out_size=teacher_size,
                        presence_gate_power=mining.presence_gate_power,
                        inference_presence_gate_power=(
                            inference_presence_gate_power
                        ),
                        target_presence_gate_power=target_presence_gate_power,
                        score_fusion_mode=mining_score_fusion_mode,
                        inference_score_fusion_mode=(
                            inference_score_fusion_mode
                        ),
                        target_score_fusion_mode=target_score_fusion_mode,
                        mask_chunk=runtime.mask_chunk,
                        grad_query_ids=loss_query_ids,
                        retain_filtered_raw_student=(
                            _is_filtered_raw_student_mode(student_score_mode)
                        ),
                    )
                    teacher_query_scores = teacher_result.query_scores
                    teacher_inference_query_scores = (
                        teacher_result.inference_query_scores
                    )
                    teacher_target_query_scores = (
                        teacher_result.target_query_scores
                    )
                    teacher_presence_logits = teacher_result.presence_logits
                    first_step_query_logits = (
                        teacher_result.student_query_logits
                    )
                    first_step_presence_logits = (
                        teacher_result.student_presence_logits
                    )
                    first_step_filtered_raw_context = (
                        teacher_result.filtered_raw_student_context
                    )
                    if (
                        _is_filtered_raw_student_mode(student_score_mode)
                        and first_step_filtered_raw_context is None
                    ):
                        raise RuntimeError(
                            "teacher did not retain the filtered-raw student context"
                        )
                    if teacher_result.student_query_ids != loss_query_ids:
                        raise RuntimeError(
                            "teacher student-query order does not match loss queries"
                        )
                    del teacher_result
                else:
                    if contraction_gate_enabled:
                        (
                            teacher_score_views,
                            teacher_presence_logits,
                            teacher_score_components,
                        ) = adapter.forward_oracle_score_views(
                            backbone_out,
                            out_size=teacher_size,
                            presence_gate_powers=(
                                mining.presence_gate_power,
                                inference_presence_gate_power,
                                target_presence_gate_power,
                            ),
                            score_fusion_modes=(
                                mining_score_fusion_mode,
                                inference_score_fusion_mode,
                                target_score_fusion_mode,
                            ),
                            mask_chunk=runtime.mask_chunk,
                        )
                    else:
                        (
                            teacher_score_views,
                            teacher_presence_logits,
                        ) = adapter.forward_fused_score_views(
                            backbone_out,
                            out_size=teacher_size,
                            presence_gate_powers=(
                                mining.presence_gate_power,
                                inference_presence_gate_power,
                                target_presence_gate_power,
                            ),
                            score_fusion_modes=(
                                mining_score_fusion_mode,
                                inference_score_fusion_mode,
                                target_score_fusion_mode,
                            ),
                            mask_chunk=runtime.mask_chunk,
                        )
                    (
                        teacher_query_scores,
                        teacher_inference_query_scores,
                        teacher_target_query_scores,
                    ) = teacher_score_views
                alias_fusion_result = None
                alias_reliability = None
                alias_fusion_fields = None
                if alias_requirements.needs_pre_reliability:
                    if canonical_query_ids is None:
                        raise ValueError(
                            "topk_reliability final heads require canonical query ids"
                        )
                    alias_fusion_result = estimate_alias_reliability(
                        query_scores=teacher_query_scores,
                        query_presence=teacher_presence_logits[0].float().sigmoid(),
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=canonical_query_ids,
                        num_classes=adapter.num_classes,
                        prob_thd=mining.prob_thd,
                        tau_pos=mining.tau_pos,
                        rho=mining.rho,
                        kmax=mining.kmax,
                        n_min=mining.n_min,
                        bg_idx=mining.bg_idx,
                    )
                    alias_reliability = alias_fusion_result.query_weights
                    alias_fusion_fields = _alias_fusion_image_fields(
                        alias_fusion_result,
                        query_words=adapter.query_words,
                    )
                independent_queries = (
                    prompt_cfg.adaptation_mode == "independent_queries"
                )
                target_class_scores = None
                if independent_queries:
                    if mining_canonical_query_ids is None:
                        raise ValueError(
                            "independent query mining requires canonical anchors"
                        )
                    teacher_views = build_prompt_class_views(
                        query_scores=teacher_query_scores,
                        presence_logits=teacher_presence_logits,
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=mining_canonical_query_ids,
                        num_classes=adapter.num_classes,
                    )
                    class_scores = teacher_views.canonical_scores
                    class_presence = teacher_views.canonical_presence
                    raw_pred = class_scores.argmax(dim=0).long()
                elif prompt_cfg.mining_view == "synonym":
                    class_scores, class_presence, raw_pred = adapter.class_scores_for_mining(
                        teacher_query_scores,
                        teacher_presence_logits,
                        presence_gate_power=mining.presence_gate_power,
                    )
                    (
                        target_class_scores,
                        _target_class_presence,
                        _target_raw_pred,
                    ) = adapter.class_scores_for_mining(
                        teacher_target_query_scores,
                        teacher_presence_logits,
                        presence_gate_power=target_presence_gate_power,
                    )
                else:
                    if mining_canonical_query_ids is None:
                        raise ValueError(
                            "canonical mining view requires mining query ids"
                        )
                    teacher_views = build_prompt_class_views(
                        query_scores=teacher_query_scores,
                        presence_logits=teacher_presence_logits,
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=mining_canonical_query_ids,
                        num_classes=adapter.num_classes,
                    )
                    class_scores = teacher_views.canonical_scores
                    class_presence = teacher_views.canonical_presence
                    raw_pred = class_scores.argmax(dim=0).long()
                    target_teacher_views = build_prompt_class_views(
                        query_scores=teacher_target_query_scores,
                        presence_logits=teacher_presence_logits,
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=mining_canonical_query_ids,
                        num_classes=adapter.num_classes,
                    )
                    target_class_scores = target_teacher_views.canonical_scores
                selected_flip_pre_class_scores = None
                selected_flip_pre_prediction = None
                if self.diagnose_selected_flips:
                    if prompt_cfg.mining_view == "synonym":
                        (
                            selected_flip_pre_class_scores,
                            _selected_flip_pre_presence,
                            selected_flip_pre_prediction,
                        ) = adapter.class_scores_for_mining(
                            teacher_inference_query_scores,
                            teacher_presence_logits,
                            presence_gate_power=(
                                inference_presence_gate_power
                            ),
                        )
                    else:
                        selected_flip_pre_views = build_prompt_class_views(
                            query_scores=teacher_inference_query_scores,
                            presence_logits=teacher_presence_logits,
                            query_idx_list=adapter.query_idx_list,
                            canonical_query_ids=mining_canonical_query_ids,
                            num_classes=adapter.num_classes,
                        )
                        selected_flip_pre_class_scores = (
                            selected_flip_pre_views.canonical_scores
                        )
                        selected_flip_pre_prediction = (
                            selected_flip_pre_class_scores.argmax(dim=0).long()
                        )
                oracle_target_class_scores_by_power = {}
                oracle_semantic_class_scores = None
                oracle_gated_semantic_class_scores = None
                oracle_instance_class_scores = None
                oracle_synonym_class_scores = None
                if oracle_cfg.enabled:
                    if teacher_score_components is None:
                        raise RuntimeError(
                            "oracle teacher score components were not produced"
                        )
                    oracle_gated_semantic_query_scores = (
                        teacher_score_components.semantic_scores
                        * teacher_score_components.presence_scores[
                            :, :, None, None
                        ].pow(float(mining.presence_gate_power))
                    )
                    if prompt_cfg.mining_view == "synonym":
                        for power, query_scores_for_power in (
                            teacher_target_query_scores_by_power.items()
                        ):
                            oracle_target_class_scores_by_power[power] = (
                                adapter.class_scores_for_mining(
                                    query_scores_for_power,
                                    teacher_presence_logits,
                                    presence_gate_power=power,
                                )[0]
                            )
                        oracle_semantic_class_scores = (
                            adapter.class_scores_for_mining(
                                teacher_score_components.semantic_scores,
                                teacher_presence_logits,
                                presence_gate_power=0.0,
                            )[0]
                        )
                        oracle_gated_semantic_class_scores = (
                            adapter.class_scores_for_mining(
                                oracle_gated_semantic_query_scores,
                                teacher_presence_logits,
                                presence_gate_power=0.0,
                            )[0]
                        )
                        oracle_instance_class_scores = (
                            adapter.class_scores_for_mining(
                                teacher_score_components.instance_scores,
                                teacher_presence_logits,
                                presence_gate_power=0.0,
                            )[0]
                        )
                    else:
                        if mining_canonical_query_ids is None:
                            raise ValueError(
                                "oracle canonical view requires canonical query ids"
                            )
                        for power, query_scores_for_power in (
                            teacher_target_query_scores_by_power.items()
                        ):
                            oracle_target_class_scores_by_power[power] = (
                                build_prompt_class_views(
                                    query_scores=query_scores_for_power,
                                    presence_logits=teacher_presence_logits,
                                    query_idx_list=adapter.query_idx_list,
                                    canonical_query_ids=mining_canonical_query_ids,
                                    num_classes=adapter.num_classes,
                                ).canonical_scores
                            )
                        oracle_semantic_class_scores = build_prompt_class_views(
                            query_scores=teacher_score_components.semantic_scores,
                            presence_logits=teacher_presence_logits,
                            query_idx_list=adapter.query_idx_list,
                            canonical_query_ids=mining_canonical_query_ids,
                            num_classes=adapter.num_classes,
                        ).canonical_scores
                        oracle_gated_semantic_class_scores = (
                            build_prompt_class_views(
                                query_scores=(
                                    oracle_gated_semantic_query_scores
                                ),
                                presence_logits=teacher_presence_logits,
                                query_idx_list=adapter.query_idx_list,
                                canonical_query_ids=(
                                    mining_canonical_query_ids
                                ),
                                num_classes=adapter.num_classes,
                            ).canonical_scores
                        )
                        oracle_instance_class_scores = build_prompt_class_views(
                            query_scores=teacher_score_components.instance_scores,
                            presence_logits=teacher_presence_logits,
                            query_idx_list=adapter.query_idx_list,
                            canonical_query_ids=mining_canonical_query_ids,
                            num_classes=adapter.num_classes,
                        ).canonical_scores
                    if adapter.num_queries > adapter.num_classes:
                        oracle_synonym_class_scores = (
                            adapter.class_scores_for_mining(
                                teacher_query_scores,
                                teacher_presence_logits,
                                presence_gate_power=mining.presence_gate_power,
                            )[0]
                        )
                valid_model = _valid_mask_for_model(gt, tuple(class_scores.shape[-2:])).to(dist_ctx.device)
                teacher_prompt_candidates = None
                if (
                    self.diagnose_teacher_prompts
                    or self.diagnose_teacher_prompt_gradients
                    or self.teacher_prompt_trial_steps > 0
                ):
                    teacher_prompt_candidates = (
                        evaluate_teacher_prompt_candidates(
                            query_scores=teacher_query_scores,
                            presence_logits=teacher_presence_logits,
                            query_idx_list=adapter.query_idx_list,
                            query_words=adapter.query_words,
                            canonical_query_ids=mining_canonical_query_ids,
                            class_ids=resolved_teacher_diagnostic_classes,
                            valid=valid_model,
                            gt=gt,
                            mining=mining,
                            loss=loss_cfg,
                        )
                    )
                if independent_queries:
                    selected, present = select_independent_query_pixels(
                        query_scores=teacher_query_scores,
                        target_query_scores=teacher_target_query_scores,
                        presence_logits=teacher_presence_logits,
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=mining_canonical_query_ids,
                        valid=valid_model,
                        mining=mining,
                        loss=loss_cfg,
                    )
                else:
                    component_gate_masks = None
                    if mining.component_gate:
                        component_gate_masks = build_component_gate_masks(
                            class_scores=class_scores,
                            class_presence=class_presence,
                            valid=valid_model,
                            mining=mining,
                        )
                    selected, present = select_classwise_pixels(
                        class_scores=class_scores,
                        class_presence=class_presence,
                        raw_pred=raw_pred,
                        valid=valid_model,
                        mining=mining,
                        loss=loss_cfg,
                        target_scores=target_class_scores,
                        component_gate_masks=component_gate_masks,
                    )
                contraction_gate_report = None
                if contraction_gate_enabled:
                    if teacher_score_components is None:
                        raise RuntimeError(
                            "contraction gate teacher components were not produced"
                        )
                    gate_canonical_query_ids = (
                        loss_canonical_query_ids
                        if prompt_cfg.loss_view == "canonical"
                        else None
                    )
                    semantic_class_scores = _class_logits_for_selected_loss(
                        adapter,
                        teacher_score_components.semantic_scores,
                        target_size=tuple(class_scores.shape[-2:]),
                        canonical_query_ids=gate_canonical_query_ids,
                        query_ids=tuple(range(adapter.num_queries)),
                    )[0]
                    primary_final_class_scores = _class_scores_for_final_head(
                        adapter,
                        query_scores=teacher_inference_query_scores,
                        canonical_query_ids=(
                            final_head_canonical_query_ids[
                                primary_final_head.name
                            ]
                        ),
                        alias_reliability=alias_reliability,
                        post_alias_reliability=alias_reliability,
                        reference_query_scores=(
                            teacher_inference_query_scores
                            if alias_requirements.needs_teacher_reference
                            else None
                        ),
                        head=primary_final_head,
                    )
                    gate_final_prediction = predict_from_class_scores(
                        primary_final_class_scores,
                        prob_thd=mining.prob_thd,
                        bg_idx=mining.bg_idx,
                        out_size=tuple(class_scores.shape[-2:]),
                    )
                    selected, contraction_gate_report = (
                        apply_decision_boundary_contraction_gate(
                            selected,
                            student_class_scores=semantic_class_scores,
                            final_class_scores=primary_final_class_scores,
                            final_prediction=gate_final_prediction,
                        )
                    )
                    contraction_gate_report.update(
                        {
                            "student_score_source": "raw_semantic",
                            "final_head": primary_final_head.name,
                        }
                    )
                selection_conflict_record = None
                if self.diagnose_selection_conflicts:
                    if independent_queries:
                        raise ValueError(
                            "selection conflict diagnostic requires "
                            "classwise adaptation"
                        )
                    selection_conflict_record = (
                        selection_conflict_diagnostic(
                            class_scores=class_scores,
                            selected_flat_indices={
                                int(class_id): entry.flat_idx
                                for class_id, entry in selected.items()
                            },
                            negative_flat_indices={
                                int(class_id): entry.negative_flat_idx
                                for class_id, entry in selected.items()
                                if entry.negative_flat_idx is not None
                            },
                            gt=gt,
                        )
                    )
                selected_pixels = int(sum(int(entry.flat_idx.numel()) for entry in selected.values()))
                present_ids = [int(value) for value in present.detach().cpu().tolist()]
                if independent_queries:
                    selected_query_ids = sorted(
                        int(query_id)
                        for query_id, entry in selected.items()
                        if int(entry.flat_idx.numel()) > 0
                    )
                    selected_class_ids = sorted(
                        {int(adapter.query_idx_list[query_id]) for query_id in selected_query_ids}
                    )
                    selected_classes = len(selected_class_ids)
                    present_class_ids = sorted(
                        {int(adapter.query_idx_list[query_id]) for query_id in present_ids}
                    )
                    selected_per_query = {
                        str(query_id): int(selected[query_id].flat_idx.numel())
                        for query_id in selected_query_ids
                    }
                    selected_per_class: dict[str, int] = {}
                    for query_id in selected_query_ids:
                        class_id = str(int(adapter.query_idx_list[query_id]))
                        selected_per_class[class_id] = (
                            selected_per_class.get(class_id, 0)
                            + int(selected[query_id].flat_idx.numel())
                        )
                else:
                    selected_class_ids = sorted(
                        int(class_id)
                        for class_id, entry in selected.items()
                        if int(entry.flat_idx.numel()) > 0
                    )
                    selected_classes = int(
                        sum(
                            1
                            for entry in selected.values()
                            if int(entry.flat_idx.numel()) > 0
                        )
                    )
                    present_class_ids = present_ids
                    selected_query_ids = []
                    selected_per_query = {}
                    selected_per_class = {
                        str(cls_idx): int(entry.flat_idx.numel())
                        for cls_idx, entry in selected.items()
                    }
                baseline_preds = _predict_final_heads(
                    adapter,
                    query_scores=teacher_inference_query_scores,
                    presence_logits=teacher_presence_logits,
                    canonical_query_ids=canonical_query_ids,
                    canonical_query_ids_by_head=final_head_canonical_query_ids,
                    alias_reliability=alias_reliability,
                    post_alias_reliability=alias_reliability,
                    reference_query_scores=(
                        teacher_inference_query_scores
                        if alias_requirements.needs_teacher_reference
                        else None
                    ),
                    heads=final_heads,
                    prob_thd=mining.prob_thd,
                    bg_idx=mining.bg_idx,
                    out_size=tuple(gt.shape[-2:]),
                )
                oracle_baseline_output_state = None
                if oracle_cfg.enabled:
                    oracle_baseline_output_state = (
                        _output_state_for_final_head(
                            adapter,
                            query_scores=teacher_inference_query_scores,
                            presence_logits=teacher_presence_logits,
                            prediction=baseline_preds[
                                primary_final_head.name
                            ],
                            canonical_query_ids=(
                                final_head_canonical_query_ids[
                                    primary_final_head.name
                                ]
                            ),
                            alias_reliability=alias_reliability,
                            post_alias_reliability=alias_reliability,
                            reference_query_scores=(
                                teacher_inference_query_scores
                                if alias_requirements.needs_teacher_reference
                                else None
                            ),
                            head=primary_final_head,
                        )
                    )
                teacher_prompt_gradients = None
                teacher_prompt_gradient_pairs = None
                if self.diagnose_teacher_prompt_gradients:
                    (
                        teacher_prompt_gradients,
                        teacher_prompt_gradient_pairs,
                    ) = _run_teacher_prompt_gradient_diagnostic(
                        adapter=adapter,
                        backbone_out=backbone_out,
                        teacher_query_scores=teacher_query_scores,
                        teacher_target_query_scores=(
                            teacher_target_query_scores
                        ),
                        teacher_presence_logits=teacher_presence_logits,
                        mining_canonical_query_ids=(
                            mining_canonical_query_ids
                        ),
                        loss_canonical_query_ids=(
                            loss_canonical_query_ids
                        ),
                        class_ids=resolved_teacher_diagnostic_classes,
                        selected=selected,
                        present=present,
                        class_presence=class_presence,
                        valid=valid_model,
                        mining=mining,
                        loss_config=loss_cfg,
                    )
                teacher_prompt_trials = None
                if self.teacher_prompt_trial_steps > 0:
                    teacher_prompt_trials = _run_teacher_prompt_trials(
                        adapter=adapter,
                        optimizer=optimizer,
                        backbone_out=backbone_out,
                        teacher_query_scores=teacher_query_scores,
                        teacher_inference_query_scores=(
                            teacher_inference_query_scores
                        ),
                        teacher_target_query_scores=(
                            teacher_target_query_scores
                        ),
                        teacher_presence_logits=teacher_presence_logits,
                        mining_canonical_query_ids=(
                            mining_canonical_query_ids
                        ),
                        loss_canonical_query_ids=loss_canonical_query_ids,
                        class_ids=resolved_teacher_diagnostic_classes,
                        trial_steps=self.teacher_prompt_trial_steps,
                        valid=valid_model,
                        gt=gt,
                        teacher_size=teacher_size,
                        mining=mining,
                        loss_config=loss_cfg,
                        optim_config=optim_cfg,
                        runtime=runtime,
                        inference_presence_gate_power=(
                            inference_presence_gate_power
                        ),
                        inference_score_fusion_mode=(
                            inference_score_fusion_mode
                        ),
                        canonical_query_ids=canonical_query_ids,
                        canonical_query_ids_by_head=(
                            final_head_canonical_query_ids
                        ),
                        alias_reliability=alias_reliability,
                        reference_query_scores=(
                            teacher_inference_query_scores
                            if alias_requirements.needs_teacher_reference
                            else None
                        ),
                        final_heads=final_heads,
                        baseline_preds=baseline_preds,
                    )
                for head in final_heads:
                    baseline_metrics[head.name].update(
                        baseline_preds[head.name],
                        gt.to(dist_ctx.device),
                    )
                baseline_pred = baseline_preds[primary_final_head.name]
                sample_id = _sample_id_from_batch(batch)
                image_record = {
                    "sample_id": sample_id,
                    "adapted": False,
                    "selected_pixels": selected_pixels,
                    "selected_classes": selected_classes,
                    "present_classes": present_class_ids,
                    "selected_per_class": selected_per_class,
                }
                if contraction_gate_report is not None:
                    image_record["contraction_gate"] = (
                        contraction_gate_report
                    )
                student_branch_routing = (
                    []
                    if _is_filtered_raw_student_mode(student_score_mode)
                    else None
                )
                if student_branch_routing is not None:
                    image_record["student_branch_routing"] = (
                        student_branch_routing
                    )
                if (
                    mining.competition_reliability_mode != "none"
                    and not independent_queries
                ):
                    image_record["competition_reliability_per_class"] = {
                        str(class_id): float(
                            entry.class_reliability.detach().item()
                        )
                        for class_id, entry in selected.items()
                        if entry.class_reliability is not None
                    }
                if selection_conflict_record is not None:
                    image_record["selection_conflict_diagnostic"] = (
                        selection_conflict_record
                    )
                oracle_teacher_features = None
                if oracle_cfg.enabled:
                    oracle_teacher_features = extract_teacher_features(
                        class_scores=class_scores,
                        class_presence=class_presence,
                        semantic_class_scores=oracle_semantic_class_scores,
                        gated_semantic_class_scores=(
                            oracle_gated_semantic_class_scores
                        ),
                        instance_class_scores=oracle_instance_class_scores,
                        reference_target_scores=(
                            oracle_target_class_scores_by_power[
                                oracle_reference.target_presence_power
                            ]
                        ),
                        selected=selected,
                        raw_pred=raw_pred,
                        valid=valid_model,
                        prob_thd=mining.prob_thd,
                        presence_gate_power=mining.presence_gate_power,
                        score_fusion_mode=mining_score_fusion_mode,
                        synonym_class_scores=oracle_synonym_class_scores,
                    )
                if teacher_prompt_candidates is not None:
                    image_record["teacher_prompt_candidates"] = (
                        teacher_prompt_candidates
                    )
                if teacher_prompt_trials is not None:
                    image_record["teacher_prompt_trials"] = (
                        teacher_prompt_trials
                    )
                if teacher_prompt_gradients is not None:
                    image_record["teacher_prompt_gradients"] = (
                        teacher_prompt_gradients
                    )
                    image_record["teacher_prompt_gradient_pairs"] = (
                        teacher_prompt_gradient_pairs
                    )
                if independent_queries:
                    image_record.update(
                        {
                            "selected_queries": selected_query_ids,
                            "present_queries": present_ids,
                            "selected_per_query": selected_per_query,
                        }
                    )
                if alias_fusion_fields is not None:
                    image_record["alias_fusion"] = alias_fusion_fields
                skip_reason = _selection_skip_reason(
                    selected_pixels=selected_pixels,
                    selected_classes=selected_classes,
                    mining=mining,
                )
                if oracle_cfg.enabled and skip_reason is not None:
                    if oracle_baseline_output_state is None:
                        raise RuntimeError(
                            "oracle baseline output state was not produced"
                        )
                    skipped_primary, skipped_alternatives = (
                        _final_head_image_fields(
                            heads=final_heads,
                            baseline_preds=baseline_preds,
                            tta_preds=baseline_preds,
                            gt=gt,
                            num_classes=adapter.num_classes,
                            device=dist_ctx.device,
                        )
                    )
                    no_update_drift = compare_output_states(
                        baseline_pred=baseline_preds[
                            primary_final_head.name
                        ],
                        candidate_pred=baseline_preds[
                            primary_final_head.name
                        ],
                        baseline_state=oracle_baseline_output_state,
                        candidate_state=oracle_baseline_output_state,
                    )
                    _, skipped_protection_decision = (
                        _protect_final_head_predictions(
                            config=protection_cfg,
                            primary_head_name=primary_final_head.name,
                            selected_class_ids=selected_class_ids,
                            baseline_preds=baseline_preds,
                            raw_tta_preds=baseline_preds,
                        )
                    )
                    skipped_record = {
                        "loss": 0.0,
                        **skipped_primary,
                        "final_heads": skipped_alternatives,
                        "output_drift": no_update_drift,
                    }
                    _attach_raw_protection_fields(
                        record=skipped_record,
                        alternatives=skipped_alternatives,
                        raw_primary=skipped_primary,
                        raw_alternatives=skipped_alternatives,
                        decision=skipped_protection_decision,
                        raw_tta_preds=baseline_preds,
                        protected_preds=baseline_preds,
                    )
                    skipped_candidates = {"no_update": skipped_record}
                    for candidate in oracle_candidates:
                        for head in final_heads:
                            oracle_candidate_metrics[candidate.key][
                                head.name
                            ].update(
                                baseline_preds[head.name],
                                gt.to(dist_ctx.device),
                            )
                            if protection_cfg.enabled:
                                oracle_raw_candidate_metrics[
                                    candidate.key
                                ][head.name].update(
                                    baseline_preds[head.name],
                                    gt.to(dist_ctx.device),
                                )
                        skipped_candidates[candidate.key] = dict(
                            skipped_record
                        )
                    for head in final_heads:
                        oracle_best_metrics[head.name].update(
                            baseline_preds[head.name],
                            gt.to(dist_ctx.device),
                        )
                    oracle_win_counts["no_update"] += 1
                    image_record["image_adaptive_oracle"] = {
                        "teacher_features": oracle_teacher_features,
                        "baseline_output_state": (
                            oracle_baseline_output_state
                        ),
                        "reference_candidate": oracle_reference.key,
                        "oracle_candidate": "no_update",
                        "candidates": skipped_candidates,
                    }
                if skip_reason == "empty_selection":
                    skipped_empty += 1
                    image_record["skip_reason"] = skip_reason
                    tta_preds = baseline_preds
                    for head in final_heads:
                        if protection_cfg.enabled:
                            raw_tta_metrics[head.name].update(
                                tta_preds[head.name],
                                gt.to(dist_ctx.device),
                            )
                        tta_metrics[head.name].update(
                            tta_preds[head.name],
                            gt.to(dist_ctx.device),
                        )
                    primary_fields, alternative_fields = _final_head_image_fields(
                        heads=final_heads,
                        baseline_preds=baseline_preds,
                        tta_preds=tta_preds,
                        gt=gt,
                        num_classes=adapter.num_classes,
                        device=dist_ctx.device,
                    )
                    image_record.update(primary_fields)
                    if protection_cfg.enabled:
                        image_record.update(
                            _raw_tta_image_fields(primary_fields)
                        )
                        for head_fields in alternative_fields.values():
                            head_fields.update(
                                _raw_tta_image_fields(head_fields)
                            )
                    if alternative_fields:
                        image_record["final_heads"] = alternative_fields
                    _update_synonym_diagnostic(
                        synonym_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        post_query_scores=None,
                        gt=gt,
                        adapted=False,
                    )
                    _update_cross_time_diagnostic(
                        cross_time_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        post_query_scores=None,
                        gt=gt,
                        adapted=False,
                    )
                    _update_prompt_routing_diagnostic(
                        prompt_routing_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        pre_reliability_scores=teacher_query_scores,
                        pre_query_presence=(
                            teacher_presence_logits[0].float().sigmoid()
                        ),
                        post_query_scores=None,
                        post_reliability_scores=None,
                        post_query_presence=None,
                        gt=gt,
                        adapted=False,
                    )
                    visualization_path = _save_primary_visualization(
                        visualization_writer,
                        batch=batch,
                        sample_id=sample_id,
                        primary_head_name=primary_final_head.name,
                        baseline_preds=baseline_preds,
                        tta_preds=tta_preds,
                        gt=gt,
                        adapted=False,
                        baseline_miou=float(image_record["baseline_miou"]),
                        tta_miou=float(image_record["tta_miou"]),
                        delta_miou=float(image_record["delta_miou"]),
                    )
                    if visualization_path is not None:
                        image_record["visualization_path"] = visualization_path
                    images.append(image_record)
                    print(
                        _format_progress(
                            rank=dist_ctx.rank,
                            world_size=dist_ctx.world_size,
                            processed=processed,
                            total=local_total,
                            sample_id=sample_id,
                            adapted=False,
                            selected_pixels=selected_pixels,
                            selected_classes=selected_classes,
                            baseline_miou=image_record["baseline_miou"],
                            tta_miou=image_record["tta_miou"],
                            delta_miou=image_record["delta_miou"],
                            baseline_macc=image_record["baseline_macc"],
                            tta_macc=image_record["tta_macc"],
                            delta_macc=image_record["delta_macc"],
                            alias_reliability_mean=(
                                alias_fusion_fields["mean_reliability"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                            active_aliases=(
                                alias_fusion_fields["active_aliases"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                            total_aliases=(
                                alias_fusion_fields["total_aliases"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                        ),
                        flush=True,
                    )
                    first_step_query_logits = None
                    first_step_presence_logits = None
                    first_step_filtered_raw_context = None
                    continue
                if skip_reason == "few_selected_classes":
                    skipped_few += 1
                    image_record["skip_reason"] = skip_reason
                    tta_preds = baseline_preds
                    for head in final_heads:
                        if protection_cfg.enabled:
                            raw_tta_metrics[head.name].update(
                                tta_preds[head.name],
                                gt.to(dist_ctx.device),
                            )
                        tta_metrics[head.name].update(
                            tta_preds[head.name],
                            gt.to(dist_ctx.device),
                        )
                    primary_fields, alternative_fields = _final_head_image_fields(
                        heads=final_heads,
                        baseline_preds=baseline_preds,
                        tta_preds=tta_preds,
                        gt=gt,
                        num_classes=adapter.num_classes,
                        device=dist_ctx.device,
                    )
                    image_record.update(primary_fields)
                    if protection_cfg.enabled:
                        image_record.update(
                            _raw_tta_image_fields(primary_fields)
                        )
                        for head_fields in alternative_fields.values():
                            head_fields.update(
                                _raw_tta_image_fields(head_fields)
                            )
                    if alternative_fields:
                        image_record["final_heads"] = alternative_fields
                    _update_synonym_diagnostic(
                        synonym_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        post_query_scores=None,
                        gt=gt,
                        adapted=False,
                    )
                    _update_cross_time_diagnostic(
                        cross_time_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        post_query_scores=None,
                        gt=gt,
                        adapted=False,
                    )
                    _update_prompt_routing_diagnostic(
                        prompt_routing_diagnostic,
                        sample_id=sample_id,
                        pre_query_scores=teacher_inference_query_scores,
                        pre_reliability_scores=teacher_query_scores,
                        pre_query_presence=(
                            teacher_presence_logits[0].float().sigmoid()
                        ),
                        post_query_scores=None,
                        post_reliability_scores=None,
                        post_query_presence=None,
                        gt=gt,
                        adapted=False,
                    )
                    visualization_path = _save_primary_visualization(
                        visualization_writer,
                        batch=batch,
                        sample_id=sample_id,
                        primary_head_name=primary_final_head.name,
                        baseline_preds=baseline_preds,
                        tta_preds=tta_preds,
                        gt=gt,
                        adapted=False,
                        baseline_miou=float(image_record["baseline_miou"]),
                        tta_miou=float(image_record["tta_miou"]),
                        delta_miou=float(image_record["delta_miou"]),
                    )
                    if visualization_path is not None:
                        image_record["visualization_path"] = visualization_path
                    images.append(image_record)
                    print(
                        _format_progress(
                            rank=dist_ctx.rank,
                            world_size=dist_ctx.world_size,
                            processed=processed,
                            total=local_total,
                            sample_id=sample_id,
                            adapted=False,
                            selected_pixels=selected_pixels,
                            selected_classes=selected_classes,
                            baseline_miou=image_record["baseline_miou"],
                            tta_miou=image_record["tta_miou"],
                            delta_miou=image_record["delta_miou"],
                            baseline_macc=image_record["baseline_macc"],
                            tta_macc=image_record["tta_macc"],
                            delta_macc=image_record["delta_macc"],
                            alias_reliability_mean=(
                                alias_fusion_fields["mean_reliability"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                            active_aliases=(
                                alias_fusion_fields["active_aliases"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                            total_aliases=(
                                alias_fusion_fields["total_aliases"]
                                if alias_fusion_fields is not None
                                else None
                            ),
                        ),
                        flush=True,
                    )
                    first_step_query_logits = None
                    first_step_filtered_raw_context = None
                    first_step_presence_logits = None
                    continue

                loss_value = 0.0
                selected_loss_size = tuple(class_scores.shape[-2:])
                presence_ids = None
                presence_targets = None
                presence_weights = None
                if float(loss_cfg.presence_loss_weight) > 0.0:
                    if independent_queries:
                        present_set = set(present_ids)
                        pos_ids = sorted(
                            query_id
                            for query_id in present_set
                            if 0 <= query_id < adapter.num_queries
                        )
                        query_presence_values = (
                            teacher_presence_logits[0]
                            .float()
                            .sigmoid()
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        neg_ids = [
                            query_id
                            for query_id in range(adapter.num_queries)
                            if query_id not in present_set
                            and float(query_presence_values[query_id])
                            <= float(mining.tau_neg)
                        ]
                        if pos_ids or neg_ids:
                            presence_ids = torch.tensor(
                                pos_ids + neg_ids,
                                device=dist_ctx.device,
                                dtype=torch.long,
                            )
                            presence_targets = torch.tensor(
                                [1.0] * len(pos_ids) + [0.0] * len(neg_ids),
                                device=dist_ctx.device,
                                dtype=torch.float32,
                            )
                            presence_weights = torch.ones_like(
                                presence_targets
                            )
                    else:
                        (
                            presence_ids,
                            presence_targets,
                            presence_weights,
                        ) = _classwise_presence_supervision(
                            present=present,
                            class_presence=class_presence,
                            num_classes=adapter.num_classes,
                            tau_neg=mining.tau_neg,
                            device=dist_ctx.device,
                            selected=selected,
                            competition_reliability_mode=(
                                mining.competition_reliability_mode
                            ),
                        )
                canonical_loss_query_ids = (
                    loss_canonical_query_ids
                    if (
                        not independent_queries
                        and prompt_cfg.loss_view == "canonical"
                    )
                    else None
                )
                if oracle_cfg.enabled:
                    if oracle_baseline_output_state is None:
                        raise RuntimeError(
                            "oracle baseline output state was not produced"
                        )
                    oracle_teacher_features["reference_objective"] = (
                        _observe_reference_objective(
                            adapter=adapter,
                            optimizer=optimizer,
                            backbone_out=backbone_out,
                            initial_lora=initial_lora,
                            loss_query_ids=loss_query_ids,
                            selected=selected,
                            target_size=selected_loss_size,
                            loss_config=loss_cfg,
                            canonical_query_ids=canonical_loss_query_ids,
                            presence_ids=presence_ids,
                            presence_targets=presence_targets,
                            presence_weights=presence_weights,
                        )
                    )
                    no_update_primary, no_update_alternatives = (
                        _final_head_image_fields(
                            heads=final_heads,
                            baseline_preds=baseline_preds,
                            tta_preds=baseline_preds,
                            gt=gt,
                            num_classes=adapter.num_classes,
                            device=dist_ctx.device,
                        )
                    )
                    no_update_drift = compare_output_states(
                        baseline_pred=baseline_preds[
                            primary_final_head.name
                        ],
                        candidate_pred=baseline_preds[
                            primary_final_head.name
                        ],
                        baseline_state=oracle_baseline_output_state,
                        candidate_state=oracle_baseline_output_state,
                    )
                    _, no_update_protection_decision = (
                        _protect_final_head_predictions(
                            config=protection_cfg,
                            primary_head_name=primary_final_head.name,
                            selected_class_ids=selected_class_ids,
                            baseline_preds=baseline_preds,
                            raw_tta_preds=baseline_preds,
                        )
                    )
                    oracle_candidate_scores = {}
                    no_update_record = {
                        "loss": 0.0,
                        **no_update_primary,
                        "final_heads": no_update_alternatives,
                        "output_drift": no_update_drift,
                    }
                    _attach_raw_protection_fields(
                        record=no_update_record,
                        alternatives=no_update_alternatives,
                        raw_primary=no_update_primary,
                        raw_alternatives=no_update_alternatives,
                        decision=no_update_protection_decision,
                        raw_tta_preds=baseline_preds,
                        protected_preds=baseline_preds,
                    )
                    oracle_candidate_records = {
                        "no_update": no_update_record
                    }
                    oracle_best_predictions = None
                    oracle_reference_snapshot = None
                    for trial in _iter_strength_oracle_trials(
                        adapter=adapter,
                        optimizer=optimizer,
                        backbone_out=backbone_out,
                        initial_lora=initial_lora,
                        candidates=oracle_candidates,
                        reference_candidate=oracle_reference,
                        target_class_scores_by_power=(
                            oracle_target_class_scores_by_power
                        ),
                        selected=selected,
                        target_size=selected_loss_size,
                        loss_config=loss_cfg,
                        optim_config=optim_cfg,
                        runtime=runtime,
                        loss_query_ids=loss_query_ids,
                        canonical_loss_query_ids=canonical_loss_query_ids,
                        presence_ids=presence_ids,
                        presence_targets=presence_targets,
                        presence_weights=presence_weights,
                        inference_presence_gate_power=(
                            inference_presence_gate_power
                        ),
                        inference_score_fusion_mode=(
                            inference_score_fusion_mode
                        ),
                        canonical_query_ids=canonical_query_ids,
                        canonical_query_ids_by_head=(
                            final_head_canonical_query_ids
                        ),
                        alias_reliability=alias_reliability,
                        alias_requirements=alias_requirements,
                        reference_query_scores=(
                            teacher_inference_query_scores
                        ),
                        final_heads=final_heads,
                        mining=mining,
                    ):
                        trial_evaluation = (
                            _evaluate_final_head_protection(
                                config=protection_cfg,
                                primary_head_name=(
                                    primary_final_head.name
                                ),
                                selected_class_ids=selected_class_ids,
                                heads=final_heads,
                                baseline_preds=baseline_preds,
                                raw_tta_preds=trial.predictions,
                                gt=gt,
                                num_classes=adapter.num_classes,
                                device=dist_ctx.device,
                            )
                        )
                        oracle_candidate_scores[trial.candidate] = float(
                            trial_evaluation.primary["tta_miou"]
                        )
                        trial_output_state = _output_state_for_final_head(
                            adapter,
                            query_scores=trial.query_scores,
                            presence_logits=trial.presence_logits,
                            prediction=trial.predictions[
                                primary_final_head.name
                            ],
                            canonical_query_ids=(
                                final_head_canonical_query_ids[
                                    primary_final_head.name
                                ]
                            ),
                            alias_reliability=alias_reliability,
                            post_alias_reliability=(
                                trial.post_alias_reliability
                            ),
                            reference_query_scores=(
                                teacher_inference_query_scores
                                if alias_requirements.needs_teacher_reference
                                else None
                            ),
                            head=primary_final_head,
                        )
                        output_drift = compare_output_states(
                            baseline_pred=baseline_preds[
                                primary_final_head.name
                            ],
                            candidate_pred=trial.predictions[
                                primary_final_head.name
                            ],
                            baseline_state=oracle_baseline_output_state,
                            candidate_state=trial_output_state,
                        )
                        trial_record = {
                            "loss": float(trial.loss),
                            **trial_evaluation.primary,
                            "final_heads": trial_evaluation.alternatives,
                            "output_drift": output_drift,
                        }
                        _attach_raw_protection_fields(
                            record=trial_record,
                            alternatives=trial_evaluation.alternatives,
                            raw_primary=trial_evaluation.raw_primary,
                            raw_alternatives=(
                                trial_evaluation.raw_alternatives
                            ),
                            decision=trial_evaluation.decision,
                            raw_tta_preds=trial.predictions,
                            protected_preds=(
                                trial_evaluation.predictions
                            ),
                        )
                        oracle_candidate_records[
                            trial.candidate.key
                        ] = trial_record
                        for head in final_heads:
                            oracle_candidate_metrics[trial.candidate.key][
                                head.name
                            ].update(
                                trial_evaluation.predictions[head.name],
                                gt.to(dist_ctx.device),
                            )
                            if protection_cfg.enabled:
                                oracle_raw_candidate_metrics[
                                    trial.candidate.key
                                ][head.name].update(
                                    trial.predictions[head.name],
                                    gt.to(dist_ctx.device),
                                )
                        current_best = select_oracle_candidate(
                            oracle_candidate_scores,
                            reference=oracle_reference,
                        )
                        if current_best == trial.candidate:
                            oracle_best_predictions = {
                                name: prediction.detach().clone()
                                for name, prediction in (
                                    trial_evaluation.predictions.items()
                                )
                            }
                        if trial.candidate == oracle_reference:
                            loss_value = float(trial.loss)
                            oracle_reference_snapshot = trial.lora_snapshot
                    oracle_best = select_oracle_candidate_or_no_update(
                        oracle_candidate_scores,
                        baseline_score=float(
                            no_update_primary["tta_miou"]
                        ),
                        reference=oracle_reference,
                    )
                    if (
                        oracle_best is not None
                        and oracle_best_predictions is None
                    ):
                        raise RuntimeError(
                            "oracle trials did not produce a best prediction"
                        )
                    if oracle_reference_snapshot is None:
                        raise RuntimeError(
                            "oracle trials did not produce the reference state"
                        )
                    selected_oracle_predictions = (
                        baseline_preds
                        if oracle_best is None
                        else oracle_best_predictions
                    )
                    if selected_oracle_predictions is None:
                        raise RuntimeError(
                            "oracle selection did not produce predictions"
                        )
                    for head in final_heads:
                        oracle_best_metrics[head.name].update(
                            selected_oracle_predictions[head.name],
                            gt.to(dist_ctx.device),
                        )
                    oracle_best_key = (
                        "no_update"
                        if oracle_best is None
                        else oracle_best.key
                    )
                    oracle_win_counts[oracle_best_key] += 1
                    image_record["image_adaptive_oracle"] = {
                        "teacher_features": oracle_teacher_features,
                        "baseline_output_state": (
                            oracle_baseline_output_state
                        ),
                        "reference_candidate": oracle_reference.key,
                        "oracle_candidate": oracle_best_key,
                        "candidates": oracle_candidate_records,
                    }
                    restore_lora_params(
                        adapter.lora_params,
                        oracle_reference_snapshot,
                    )
                    reset_optimizer_state(optimizer)
                    clear_lora_weight_cache(adapter.encoder)
                    first_step_filtered_raw_context = None
                    first_step_query_logits = None
                    first_step_presence_logits = None
                elif optim_cfg.backward_mode == "direct":
                    first_step_outputs = None
                    if (
                        not _is_filtered_raw_student_mode(student_score_mode)
                        and first_step_query_logits is not None
                        and first_step_presence_logits is not None
                    ):
                        first_step_outputs = (
                            first_step_query_logits,
                            first_step_presence_logits,
                        )
                    loss_value = _run_direct_update_steps(
                        adapter=adapter,
                        optimizer=optimizer,
                        backbone_out=backbone_out,
                        steps=optim_cfg.steps,
                        loss_query_ids=loss_query_ids,
                        selected=selected,
                        target_size=selected_loss_size,
                        loss_config=loss_cfg,
                        canonical_query_ids=canonical_loss_query_ids,
                        presence_ids=presence_ids,
                        presence_targets=presence_targets,
                        presence_weights=presence_weights,
                        grad_clip=optim_cfg.grad_clip,
                        first_step_outputs=first_step_outputs,
                        first_step_filtered_raw_context=(
                            first_step_filtered_raw_context
                        ),
                        adaptation_mode=prompt_cfg.adaptation_mode,
                        query_idx_list=adapter.query_idx_list,
                        student_score_mode=student_score_mode,
                        mask_chunk=runtime.mask_chunk,
                        branch_diagnostics=student_branch_routing,
                    )
                    first_step_outputs = None
                    first_step_query_logits = None
                    first_step_presence_logits = None
                    first_step_filtered_raw_context = None
                elif optim_cfg.backward_mode == "replay":
                    sparse_plan = _sparse_plan_for_update(
                        selected=selected,
                        target_size=selected_loss_size,
                        device=dist_ctx.device,
                        loss_config=loss_cfg,
                        steps=optim_cfg.steps,
                    )
                    last_step_loss = None
                    for _step in range(int(optim_cfg.steps)):
                        optimizer.zero_grad(set_to_none=True)
                        clear_lora_weight_cache(adapter.encoder)
                        with torch.no_grad():
                            query_logits_detached, presence_logits_detached = adapter.forward_queries(
                                backbone_out,
                                grad=False,
                                out_size=None,
                                query_ids=loss_query_ids,
                            )
                        query_logits_detached.requires_grad_(True)
                        need_presence_grad = presence_ids is not None
                        presence_logits_detached.requires_grad_(need_presence_grad)
                        total_step_loss = _tta_step_loss(
                            adapter=adapter,
                            query_logits=query_logits_detached,
                            presence_logits=presence_logits_detached,
                            selected=selected,
                            target_size=selected_loss_size,
                            loss_config=loss_cfg,
                            canonical_query_ids=canonical_loss_query_ids,
                            query_ids=loss_query_ids,
                            presence_ids=presence_ids,
                            presence_targets=presence_targets,
                            presence_weights=presence_weights,
                            sparse_plan=sparse_plan,
                            adaptation_mode=prompt_cfg.adaptation_mode,
                            query_idx_list=adapter.query_idx_list,
                        )
                        if need_presence_grad:
                            grad_query_logits, grad_presence_logits = torch.autograd.grad(
                                total_step_loss,
                                (query_logits_detached, presence_logits_detached),
                            )
                            grad_presence_logits = grad_presence_logits.detach()
                        else:
                            grad_query_logits = torch.autograd.grad(
                                total_step_loss,
                                query_logits_detached,
                            )[0]
                            grad_presence_logits = None
                        grad_query_logits = grad_query_logits.detach()
                        active_query_indices = _active_replay_query_indices(
                            grad_query_logits,
                            grad_presence_logits,
                        )
                        for query_index in active_query_indices:
                            query_id = loss_query_ids[query_index]
                            semantic_logits, presence_logits_one = adapter.forward_query_one(
                                backbone_out,
                                query_id=query_id,
                                out_size=None,
                            )
                            if grad_presence_logits is None:
                                semantic_logits.float().backward(
                                    grad_query_logits[:, query_index]
                                )
                            else:
                                torch.autograd.backward(
                                    tensors=(semantic_logits.float(), presence_logits_one.float()),
                                    grad_tensors=(
                                        grad_query_logits[:, query_index],
                                        grad_presence_logits[:, query_index],
                                    ),
                                )
                        torch.nn.utils.clip_grad_norm_(
                            adapter.lora_params,
                            optim_cfg.grad_clip,
                        )
                        optimizer.step()
                        last_step_loss = total_step_loss.detach()
                    if last_step_loss is not None:
                        loss_value = float(last_step_loss.item())
                else:
                    raise ValueError(
                        f"unsupported backward mode: {optim_cfg.backward_mode}"
                    )

                clear_lora_weight_cache(adapter.encoder)
                with torch.no_grad():
                    tta_query_scores, tta_presence_logits = adapter.forward_fused_scores(
                        backbone_out,
                        out_size=teacher_size,
                        presence_gate_power=inference_presence_gate_power,
                        mask_chunk=runtime.mask_chunk,
                        score_fusion_mode=inference_score_fusion_mode,
                    )
                selected_flip_post_class_scores = None
                selected_flip_post_prediction = None
                if self.diagnose_selected_flips:
                    if prompt_cfg.mining_view == "synonym":
                        (
                            selected_flip_post_class_scores,
                            _selected_flip_post_presence,
                            selected_flip_post_prediction,
                        ) = adapter.class_scores_for_mining(
                            tta_query_scores,
                            tta_presence_logits,
                            presence_gate_power=(
                                inference_presence_gate_power
                            ),
                        )
                    else:
                        selected_flip_post_views = build_prompt_class_views(
                            query_scores=tta_query_scores,
                            presence_logits=tta_presence_logits,
                            query_idx_list=adapter.query_idx_list,
                            canonical_query_ids=mining_canonical_query_ids,
                            num_classes=adapter.num_classes,
                        )
                        selected_flip_post_class_scores = (
                            selected_flip_post_views.canonical_scores
                        )
                        selected_flip_post_prediction = (
                            selected_flip_post_class_scores.argmax(dim=0).long()
                        )
                post_alias_fusion_result = None
                post_alias_reliability = None
                post_alias_fusion_fields = None
                if alias_requirements.needs_post_reliability:
                    if canonical_query_ids is None:
                        raise ValueError(
                            "post_tta_reliability final heads require canonical query ids"
                        )
                    post_alias_fusion_result = estimate_alias_reliability(
                        query_scores=tta_query_scores,
                        query_presence=tta_presence_logits[0].float().sigmoid(),
                        query_idx_list=adapter.query_idx_list,
                        canonical_query_ids=canonical_query_ids,
                        num_classes=adapter.num_classes,
                        prob_thd=mining.prob_thd,
                        tau_pos=mining.tau_pos,
                        rho=mining.rho,
                        kmax=mining.kmax,
                        n_min=mining.n_min,
                        bg_idx=mining.bg_idx,
                    )
                    post_alias_reliability = post_alias_fusion_result.query_weights
                    post_alias_fusion_fields = _alias_fusion_image_fields(
                        post_alias_fusion_result,
                        query_words=adapter.query_words,
                    )
                    image_record["post_alias_fusion"] = post_alias_fusion_fields
                raw_tta_preds = _predict_final_heads(
                    adapter,
                    query_scores=tta_query_scores,
                    presence_logits=tta_presence_logits,
                    canonical_query_ids=canonical_query_ids,
                    canonical_query_ids_by_head=final_head_canonical_query_ids,
                    alias_reliability=alias_reliability,
                    post_alias_reliability=post_alias_reliability,
                    reference_query_scores=(
                        teacher_inference_query_scores
                        if alias_requirements.needs_teacher_reference
                        else None
                    ),
                    heads=final_heads,
                    prob_thd=mining.prob_thd,
                    bg_idx=mining.bg_idx,
                    out_size=tuple(gt.shape[-2:]),
                )
                if protection_cfg.survival_enabled:
                    if adaptation_start_lora is None:
                        raise RuntimeError(
                            "class survival rerun did not capture the adaptation state"
                        )
                    initial_survival_decision = _decide_class_survival(
                        config=protection_cfg,
                        baseline_pred=baseline_preds[
                            primary_final_head.name
                        ],
                        raw_tta_pred=raw_tta_preds[
                            primary_final_head.name
                        ],
                        selected_class_ids=selected_class_ids,
                    )
                    survival_record = {
                        "initial": initial_survival_decision.to_dict(),
                        "metric": protection_cfg.survival_metric,
                        "intervention": (
                            protection_cfg.survival_intervention
                        ),
                        "mask_bce_scale": float(
                            protection_cfg.survival_mask_bce_scale
                        ),
                        "rerun": False,
                    }
                    if initial_survival_decision.triggered:
                        initial_raw_tta_preds = {
                            name: prediction.detach().clone()
                            for name, prediction in raw_tta_preds.items()
                        }
                        (
                            initial_survival_primary,
                            initial_survival_alternatives,
                        ) = _final_head_image_fields(
                            heads=final_heads,
                            baseline_preds=baseline_preds,
                            tta_preds=initial_raw_tta_preds,
                            gt=gt,
                            num_classes=adapter.num_classes,
                            device=dist_ctx.device,
                        )
                        restore_lora_params(
                            adapter.lora_params,
                            adaptation_start_lora,
                        )
                        reset_optimizer_state(optimizer)
                        clear_lora_weight_cache(adapter.encoder)
                        rescue_selected = selected
                        reconstruction_query_ids: tuple[int, ...] = ()
                        if (
                            protection_cfg.survival_intervention
                            == "mask_loss_scale"
                        ):
                            rescue_selected = apply_mask_loss_scale(
                                selected,
                                class_ids=(
                                    initial_survival_decision.collapsing_classes
                                ),
                                scale=(
                                    protection_cfg.survival_mask_bce_scale
                                ),
                            )
                        else:
                            reconstruction_query_ids = (
                                _query_ids_for_risk_classes(
                                    risk_class_ids=(
                                        initial_survival_decision.collapsing_classes
                                    ),
                                    loss_query_ids=loss_query_ids,
                                    canonical_query_ids=(
                                        canonical_loss_query_ids
                                    ),
                                    query_idx_list=adapter.query_idx_list,
                                )
                            )
                            if not reconstruction_query_ids:
                                raise RuntimeError(
                                    "survival-risk classes did not map to any "
                                    "loss query"
                                )
                        loss_value = _run_direct_update_steps(
                            adapter=adapter,
                            optimizer=optimizer,
                            backbone_out=backbone_out,
                            steps=optim_cfg.steps,
                            loss_query_ids=loss_query_ids,
                            selected=rescue_selected,
                            target_size=selected_loss_size,
                            loss_config=loss_cfg,
                            canonical_query_ids=canonical_loss_query_ids,
                            presence_ids=presence_ids,
                            presence_targets=presence_targets,
                            presence_weights=presence_weights,
                            grad_clip=optim_cfg.grad_clip,
                            first_step_outputs=None,
                            adaptation_mode=prompt_cfg.adaptation_mode,
                            query_idx_list=adapter.query_idx_list,
                            student_score_mode=student_score_mode,
                            mask_chunk=runtime.mask_chunk,
                            probability_reconstruction_query_ids=(
                                reconstruction_query_ids
                            ),
                        )
                        clear_lora_weight_cache(adapter.encoder)
                        with torch.no_grad():
                            (
                                tta_query_scores,
                                tta_presence_logits,
                            ) = adapter.forward_fused_scores(
                                backbone_out,
                                out_size=teacher_size,
                                presence_gate_power=(
                                    inference_presence_gate_power
                                ),
                                mask_chunk=runtime.mask_chunk,
                                score_fusion_mode=(
                                    inference_score_fusion_mode
                                ),
                            )
                        post_alias_reliability = None
                        post_alias_fusion_fields = None
                        if alias_requirements.needs_post_reliability:
                            if canonical_query_ids is None:
                                raise ValueError(
                                    "post-TTA survival inference requires "
                                    "canonical query ids"
                                )
                            post_alias_fusion_result = (
                                estimate_alias_reliability(
                                    query_scores=tta_query_scores,
                                    query_presence=(
                                        tta_presence_logits[0]
                                        .float()
                                        .sigmoid()
                                    ),
                                    query_idx_list=adapter.query_idx_list,
                                    canonical_query_ids=(
                                        canonical_query_ids
                                    ),
                                    num_classes=adapter.num_classes,
                                    prob_thd=mining.prob_thd,
                                    tau_pos=mining.tau_pos,
                                    rho=mining.rho,
                                    kmax=mining.kmax,
                                    n_min=mining.n_min,
                                    bg_idx=mining.bg_idx,
                                )
                            )
                            post_alias_reliability = (
                                post_alias_fusion_result.query_weights
                            )
                            post_alias_fusion_fields = (
                                _alias_fusion_image_fields(
                                    post_alias_fusion_result,
                                    query_words=adapter.query_words,
                                )
                            )
                            image_record["post_alias_fusion"] = (
                                post_alias_fusion_fields
                            )
                        raw_tta_preds = _predict_final_heads(
                            adapter,
                            query_scores=tta_query_scores,
                            presence_logits=tta_presence_logits,
                            canonical_query_ids=canonical_query_ids,
                            canonical_query_ids_by_head=(
                                final_head_canonical_query_ids
                            ),
                            alias_reliability=alias_reliability,
                            post_alias_reliability=(
                                post_alias_reliability
                            ),
                            reference_query_scores=(
                                teacher_inference_query_scores
                                if alias_requirements.needs_teacher_reference
                                else None
                            ),
                            heads=final_heads,
                            prob_thd=mining.prob_thd,
                            bg_idx=mining.bg_idx,
                            out_size=tuple(gt.shape[-2:]),
                        )
                        post_survival_decision = _decide_class_survival(
                            config=protection_cfg,
                            baseline_pred=baseline_preds[
                                primary_final_head.name
                            ],
                            raw_tta_pred=raw_tta_preds[
                                primary_final_head.name
                            ],
                            selected_class_ids=selected_class_ids,
                        )
                        survival_rerun_images += 1
                        survival_record.update(
                            {
                                "rerun": True,
                                "post_rerun": (
                                    post_survival_decision.to_dict()
                                ),
                                "initial_candidate": {
                                    **initial_survival_primary,
                                    "final_heads": (
                                        initial_survival_alternatives
                                    ),
                                },
                            }
                        )
                    image_record["class_survival"] = survival_record
                tta_preds, protection_decision = (
                    _protect_final_head_predictions(
                        config=protection_cfg,
                        primary_head_name=primary_final_head.name,
                        selected_class_ids=selected_class_ids,
                        baseline_preds=baseline_preds,
                        raw_tta_preds=raw_tta_preds,
                    )
                )
                if self.diagnose_selected_flips:
                    if (
                        selected_flip_pre_class_scores is None
                        or selected_flip_post_class_scores is None
                        or selected_flip_pre_prediction is None
                        or selected_flip_post_prediction is None
                    ):
                        raise RuntimeError(
                            "selected TopK flip score views were not produced"
                        )
                    image_record["selected_flip_diagnostic"] = (
                        selected_topk_flip_diagnostic(
                            selected_flat_indices={
                                int(class_id): entry.flat_idx
                                for class_id, entry in selected.items()
                            },
                            pre_class_scores=(
                                selected_flip_pre_class_scores
                            ),
                            post_class_scores=(
                                selected_flip_post_class_scores
                            ),
                            prediction_views={
                                "mining_argmax": (
                                    selected_flip_pre_prediction,
                                    selected_flip_post_prediction,
                                ),
                                "primary_raw": (
                                    baseline_preds[
                                        primary_final_head.name
                                    ],
                                    raw_tta_preds[
                                        primary_final_head.name
                                    ],
                                ),
                                "primary_protected": (
                                    baseline_preds[
                                        primary_final_head.name
                                    ],
                                    tta_preds[primary_final_head.name],
                                ),
                            },
                            gt=gt.to(dist_ctx.device),
                        )
                    )
                raw_primary_fields = None
                raw_alternative_fields = {}
                if protection_cfg.enabled:
                    if protection_decision is None:
                        raise RuntimeError(
                            "enabled protection did not produce a decision"
                        )
                    guarded_images += int(protection_decision.triggered)
                    for head in final_heads:
                        raw_tta_metrics[head.name].update(
                            raw_tta_preds[head.name],
                            gt.to(dist_ctx.device),
                        )
                    (
                        raw_primary_fields,
                        raw_alternative_fields,
                    ) = _final_head_image_fields(
                        heads=final_heads,
                        baseline_preds=baseline_preds,
                        tta_preds=raw_tta_preds,
                        gt=gt,
                        num_classes=adapter.num_classes,
                        device=dist_ctx.device,
                    )
                for head in final_heads:
                    tta_metrics[head.name].update(
                        tta_preds[head.name],
                        gt.to(dist_ctx.device),
                    )
                adapted += 1
                total_loss += loss_value
                total_selected += selected_pixels
                total_selected_classes += selected_classes
                primary_fields, alternative_fields = _final_head_image_fields(
                    heads=final_heads,
                    baseline_preds=baseline_preds,
                    tta_preds=tta_preds,
                    gt=gt,
                    num_classes=adapter.num_classes,
                    device=dist_ctx.device,
                )
                image_record.update(
                    {
                        "adapted": True,
                        "loss": loss_value,
                        **primary_fields,
                    }
                )
                if protection_cfg.enabled:
                    if raw_primary_fields is None:
                        raise RuntimeError("raw TTA metrics were not produced")
                    image_record.update(
                        _raw_tta_image_fields(raw_primary_fields)
                    )
                    image_record["protection"] = (
                        _protection_record_fields(
                            decision=protection_decision,
                            raw_tta_preds=raw_tta_preds,
                            protected_preds=tta_preds,
                        )
                    )
                    for head_name, head_fields in alternative_fields.items():
                        head_fields.update(
                            _raw_tta_image_fields(
                                raw_alternative_fields[head_name]
                            )
                        )
                if alternative_fields:
                    image_record["final_heads"] = alternative_fields
                _update_synonym_diagnostic(
                    synonym_diagnostic,
                    sample_id=sample_id,
                    pre_query_scores=teacher_inference_query_scores,
                    post_query_scores=tta_query_scores,
                    gt=gt,
                    adapted=True,
                )
                _update_cross_time_diagnostic(
                    cross_time_diagnostic,
                    sample_id=sample_id,
                    pre_query_scores=teacher_inference_query_scores,
                    post_query_scores=tta_query_scores,
                    gt=gt,
                    adapted=True,
                )
                _update_prompt_routing_diagnostic(
                    prompt_routing_diagnostic,
                    sample_id=sample_id,
                    pre_query_scores=teacher_inference_query_scores,
                    pre_reliability_scores=teacher_query_scores,
                    pre_query_presence=(
                        teacher_presence_logits[0].float().sigmoid()
                    ),
                    post_query_scores=tta_query_scores,
                    post_reliability_scores=tta_query_scores,
                    post_query_presence=(
                        tta_presence_logits[0].float().sigmoid()
                    ),
                    gt=gt,
                    adapted=True,
                )
                visualization_path = _save_primary_visualization(
                    visualization_writer,
                    batch=batch,
                    sample_id=sample_id,
                    primary_head_name=primary_final_head.name,
                    baseline_preds=baseline_preds,
                    tta_preds=tta_preds,
                    gt=gt,
                    adapted=True,
                    baseline_miou=float(image_record["baseline_miou"]),
                    tta_miou=float(image_record["tta_miou"]),
                    delta_miou=float(image_record["delta_miou"]),
                )
                if visualization_path is not None:
                    image_record["visualization_path"] = visualization_path
                images.append(image_record)
                print(
                    _format_progress(
                        rank=dist_ctx.rank,
                        world_size=dist_ctx.world_size,
                        processed=processed,
                        total=local_total,
                        sample_id=sample_id,
                        adapted=True,
                        selected_pixels=selected_pixels,
                        selected_classes=selected_classes,
                        baseline_miou=image_record["baseline_miou"],
                        tta_miou=image_record["tta_miou"],
                        delta_miou=image_record["delta_miou"],
                        baseline_macc=image_record["baseline_macc"],
                        tta_macc=image_record["tta_macc"],
                        delta_macc=image_record["delta_macc"],
                        alias_reliability_mean=(
                            (post_alias_fusion_fields or alias_fusion_fields)[
                                "mean_reliability"
                            ]
                            if (post_alias_fusion_fields or alias_fusion_fields) is not None
                            else None
                        ),
                        active_aliases=(
                            (post_alias_fusion_fields or alias_fusion_fields)[
                                "active_aliases"
                            ]
                            if (post_alias_fusion_fields or alias_fusion_fields) is not None
                            else None
                        ),
                        total_aliases=(
                            (post_alias_fusion_fields or alias_fusion_fields)[
                                "total_aliases"
                            ]
                            if (post_alias_fusion_fields or alias_fusion_fields) is not None
                            else None
                        ),
                    ),
                    flush=True,
                )

            if dist_ctx.device.type == "cuda":
                torch.cuda.synchronize(dist_ctx.device)
                peak_cuda_memory_bytes = torch.cuda.max_memory_allocated(
                    dist_ctx.device
                )
            else:
                peak_cuda_memory_bytes = 0
            performance_stats = torch.tensor(
                [
                    time.perf_counter() - online_loop_started_at,
                    time.perf_counter() - run_started_at,
                    peak_cuda_memory_bytes,
                ],
                device=dist_ctx.device,
                dtype=torch.float64,
            )
            count_stats = torch.tensor(
                [
                    processed,
                    adapted,
                    skipped_empty,
                    skipped_few,
                    total_loss,
                    total_selected,
                    total_selected_classes,
                    guarded_images,
                    survival_rerun_images,
                ],
                device=dist_ctx.device,
                dtype=torch.float64,
            )
            oracle_win_keys = (
                (
                    "no_update",
                    *(candidate.key for candidate in oracle_candidates),
                )
                if oracle_cfg.enabled
                else ()
            )
            oracle_win_tensor = torch.tensor(
                [oracle_win_counts[key] for key in oracle_win_keys],
                device=dist_ctx.device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                for head in final_heads:
                    all_reduce_tensor(baseline_metrics[head.name].matrix)
                    all_reduce_tensor(tta_metrics[head.name].matrix)
                    if protection_cfg.enabled:
                        all_reduce_tensor(raw_tta_metrics[head.name].matrix)
                for candidate in oracle_candidates:
                    for head in final_heads:
                        all_reduce_tensor(
                            oracle_candidate_metrics[candidate.key][
                                head.name
                            ].matrix
                        )
                        if protection_cfg.enabled:
                            all_reduce_tensor(
                                oracle_raw_candidate_metrics[
                                    candidate.key
                                ][head.name].matrix
                            )
                for head in final_heads:
                    if oracle_cfg.enabled:
                        all_reduce_tensor(
                            oracle_best_metrics[head.name].matrix
                        )
                all_reduce_tensor(count_stats)
                all_reduce_tensor(performance_stats, op=dist.ReduceOp.MAX)
                if oracle_cfg.enabled:
                    all_reduce_tensor(oracle_win_tensor)
            if synonym_diagnostic is not None:
                synonym_diagnostic.reduce_distributed()
            if cross_time_diagnostic is not None:
                cross_time_diagnostic.reduce_distributed()
            if prompt_routing_diagnostic is not None:
                prompt_routing_diagnostic.reduce_distributed()
            if dist.is_initialized():
                gathered_images = [None for _ in range(dist_ctx.world_size)]
                dist.all_gather_object(gathered_images, images)
                if dist_ctx.is_main:
                    images = [
                        image
                        for rank_images in gathered_images
                        for image in (rank_images or [])
                    ]
            final_head_results = {}
            for head in final_heads:
                final_head_results[head.name] = {
                    "canonical_class_ids": list(head.canonical_class_ids),
                    "classname_path": head.classname_path,
                    "canonical_query_ids": (
                        list(final_head_canonical_query_ids[head.name])
                        if final_head_canonical_query_ids[head.name] is not None
                        else None
                    ),
                    "fusion_mode": head.fusion_mode,
                    **_comparison_metric_fields(
                        baseline_metrics[head.name],
                        tta_metrics[head.name],
                    ),
                }
                if protection_cfg.enabled:
                    final_head_results[head.name].update(
                        _comparison_metric_fields(
                            baseline_metrics[head.name],
                            raw_tta_metrics[head.name],
                            comparison_prefix="raw_tta",
                            delta_prefix="raw_delta",
                            include_baseline=False,
                        )
                    )
            oracle_summary = None
            if oracle_cfg.enabled:
                no_update_heads = {}
                for head in final_heads:
                    no_update_heads[head.name] = (
                        _comparison_metric_fields(
                            baseline_metrics[head.name],
                            baseline_metrics[head.name],
                            include_baseline=False,
                        )
                    )
                    if protection_cfg.enabled:
                        no_update_heads[head.name].update(
                            _comparison_metric_fields(
                                baseline_metrics[head.name],
                                baseline_metrics[head.name],
                                comparison_prefix="raw_tta",
                                delta_prefix="raw_delta",
                                include_baseline=False,
                            )
                        )
                candidate_summaries = {
                    "no_update": {
                        "lr_multiplier": None,
                        "target_presence_power": None,
                        "final_heads": no_update_heads,
                    }
                }
                for candidate in oracle_candidates:
                    candidate_heads = {}
                    for head in final_heads:
                        candidate_heads[head.name] = (
                            _comparison_metric_fields(
                                baseline_metrics[head.name],
                                oracle_candidate_metrics[candidate.key][
                                    head.name
                                ],
                                include_baseline=False,
                            )
                        )
                        if protection_cfg.enabled:
                            candidate_heads[head.name].update(
                                _comparison_metric_fields(
                                    baseline_metrics[head.name],
                                    oracle_raw_candidate_metrics[
                                        candidate.key
                                    ][head.name],
                                    comparison_prefix="raw_tta",
                                    delta_prefix="raw_delta",
                                    include_baseline=False,
                                )
                            )
                    candidate_summaries[candidate.key] = {
                        "lr_multiplier": candidate.lr_multiplier,
                        "target_presence_power": (
                            candidate.target_presence_power
                        ),
                        "final_heads": candidate_heads,
                    }
                oracle_heads = {}
                for head in final_heads:
                    oracle_heads[head.name] = (
                        _comparison_metric_fields(
                            baseline_metrics[head.name],
                            oracle_best_metrics[head.name],
                            include_baseline=False,
                        )
                    )
                oracle_summary = {
                    "config": oracle_cfg.to_dict(),
                    "candidate_grid": [
                        {
                            "key": "no_update",
                            "kind": "identity",
                            "lr_multiplier": None,
                            "target_presence_power": None,
                        },
                        *[
                        {
                            "key": candidate.key,
                            "kind": "adaptation",
                            "lr_multiplier": candidate.lr_multiplier,
                            "target_presence_power": (
                                candidate.target_presence_power
                            ),
                        }
                        for candidate in oracle_candidates
                        ],
                    ],
                    "reference_candidate": oracle_reference.key,
                    "reference": candidate_summaries[
                        oracle_reference.key
                    ],
                    "oracle": {"final_heads": oracle_heads},
                    "candidates": candidate_summaries,
                    "candidate_win_counts": {
                        key: int(count)
                        for key, count in zip(
                            oracle_win_keys,
                            oracle_win_tensor.detach().cpu().tolist(),
                            strict=True,
                        )
                    },
                    "selection_uses_gt": True,
                    "teacher_features_use_gt": False,
                }
            primary_metrics = final_head_results[primary_final_head.name]
            alias_fusion_summary = _summarize_alias_fusion(images)
            post_alias_fusion_summary = _summarize_alias_fusion(
                images,
                field_name="post_alias_fusion",
            )
            student_branch_summary = (
                aggregate_student_branch_routing(images)
                if _is_filtered_raw_student_mode(student_score_mode)
                else None
            )
            (
                processed_total,
                adapted_total,
                skipped_empty_total,
                skipped_few_total,
                total_loss_all,
                total_selected_all,
                total_selected_classes_all,
                guarded_total,
                survival_rerun_total,
            ) = [float(v) for v in count_stats.detach().cpu().tolist()]
            (
                elapsed_seconds,
                total_elapsed_seconds,
                peak_cuda_memory_bytes,
            ) = [
                float(value)
                for value in performance_stats.detach().cpu().tolist()
            ]
            performance = _performance_fields(
                processed=int(processed_total),
                elapsed_seconds=elapsed_seconds,
                peak_cuda_memory_bytes=int(peak_cuda_memory_bytes),
            )
            performance.update(
                {
                    "total_elapsed_seconds": total_elapsed_seconds,
                    "backward_mode": optim_cfg.backward_mode,
                    "student_query_batch_size": runtime.query_batch_size,
                    "full_query_batch_size": runtime.full_query_batch_size,
                    "instance_mask_chunk": runtime.mask_chunk,
                    "reuse_teacher_first_step": reuse_teacher_first_step,
                    "sparse_selected_logits": loss_cfg.sparse_selected_logits,
                    "student_score_mode": student_score_mode,
                    "mode": (
                        "baseline_only" if runtime.baseline_only else "tta"
                    ),
                    "sparse_filtered_instance_sampling": (
                        _is_filtered_raw_student_mode(student_score_mode)
                    ),
                }
            )
            report_class_names = _resolve_report_class_names(
                dataset=dataset,
                adapter=adapter,
                canonical_query_ids=canonical_query_ids,
            )
            result = {
                "config": str(runtime.eval_config),
                "split": runtime.split,
                "processed": int(processed_total),
                "world_size": dist_ctx.world_size,
                "metric_scale": "percent",
                "class_names": report_class_names,
                "primary_final_head": primary_final_head.name,
                "final_heads": final_head_results,
                "baseline_miou": primary_metrics["baseline_miou"],
                "baseline_per_class_iou": primary_metrics["baseline_per_class_iou"],
                "baseline_valid_classes": primary_metrics[
                    "baseline_valid_classes"
                ],
                "baseline_macc": primary_metrics["baseline_macc"],
                "baseline_per_class_accuracy": primary_metrics[
                    "baseline_per_class_accuracy"
                ],
                "baseline_valid_accuracy_classes": primary_metrics[
                    "baseline_valid_accuracy_classes"
                ],
                "baseline_aacc": primary_metrics["baseline_aacc"],
                "baseline_confusion_matrix": primary_metrics[
                    "baseline_confusion_matrix"
                ],
                "tta_miou": primary_metrics["tta_miou"],
                "tta_per_class_iou": primary_metrics["tta_per_class_iou"],
                "tta_valid_classes": primary_metrics[
                    "tta_valid_classes"
                ],
                "tta_macc": primary_metrics["tta_macc"],
                "tta_per_class_accuracy": primary_metrics[
                    "tta_per_class_accuracy"
                ],
                "tta_valid_accuracy_classes": primary_metrics[
                    "tta_valid_accuracy_classes"
                ],
                "tta_aacc": primary_metrics["tta_aacc"],
                "tta_confusion_matrix": primary_metrics[
                    "tta_confusion_matrix"
                ],
                "delta_miou": primary_metrics["delta_miou"],
                "delta_per_class_iou": primary_metrics[
                    "delta_per_class_iou"
                ],
                "delta_macc": primary_metrics["delta_macc"],
                "delta_per_class_accuracy": primary_metrics[
                    "delta_per_class_accuracy"
                ],
                "delta_aacc": primary_metrics["delta_aacc"],
                "adapted": int(adapted_total),
                "skipped_empty_pos": int(skipped_empty_total),
                "skipped_few_pixels": 0,
                "skipped_few_classes": int(skipped_few_total),
                "avg_loss": float(total_loss_all / max(adapted_total, 1.0)),
                "avg_selected_pixels": float(total_selected_all / max(adapted_total, 1.0)),
                "avg_selected_classes": float(total_selected_classes_all / max(adapted_total, 1.0)),
                "images": images,
                "params": self.config.to_dict(),
                "prompt_views": prompt_view_metadata,
                "score_fusion_views": score_fusion_views,
                "alias_fusion_summary": alias_fusion_summary,
                "post_alias_fusion_summary": post_alias_fusion_summary,
                "effective_source_lora_path": adapter.source_lora_path,
                "performance": performance,
            }
            if contraction_gate_enabled:
                gate_reports = [
                    image["contraction_gate"]
                    for image in images
                    if "contraction_gate" in image
                ]
                result["contraction_gate"] = (
                    _aggregate_contraction_gate_reports(gate_reports)
                )
            if protection_cfg.survival_enabled:
                result["class_survival"] = {
                    "enabled": True,
                    "metric": protection_cfg.survival_metric,
                    "intervention": protection_cfg.survival_intervention,
                    "threshold": float(
                        protection_cfg.survival_threshold
                    ),
                    "mask_bce_scale": float(
                        protection_cfg.survival_mask_bce_scale
                    ),
                    "min_baseline_pixels": int(
                        protection_cfg.survival_min_baseline_pixels
                    ),
                    "rerun_images": int(survival_rerun_total),
                    "rerun_rate": float(
                        survival_rerun_total
                        / max(processed_total, 1.0)
                    ),
                    "uses_gt_for_decision": False,
                }
            if student_branch_summary is not None:
                result["student_branch_routing"] = student_branch_summary
            if self.diagnose_selection_conflicts:
                conflict_records = [
                    image["selection_conflict_diagnostic"]
                    for image in images
                    if "selection_conflict_diagnostic" in image
                ]
                result["selection_conflict_diagnostic"] = {
                    "enabled": True,
                    "selection_source": prompt_cfg.mining_view,
                    "score_comparison_presence_gate_power": float(
                        mining.presence_gate_power
                    ),
                    "near_tie_thresholds": list(
                        DEFAULT_NEAR_TIE_THRESHOLDS
                    ),
                    "uses_gt_for_purity_only": True,
                    "summary": aggregate_selection_conflict_diagnostics(
                        conflict_records
                    ),
                }
            if self.diagnose_selected_flips:
                result["selected_flip_diagnostic"] = {
                    "enabled": True,
                    "selection_source": prompt_cfg.mining_view,
                    "score_comparison_presence_gate_power": float(
                        inference_presence_gate_power
                    ),
                    "views": [
                        "mining_argmax",
                        "primary_raw",
                        "primary_protected",
                    ],
                    "uses_gt_for_purity_and_flip_semantics": True,
                }
            if oracle_summary is not None:
                result["image_adaptive_oracle"] = oracle_summary
            if protection_cfg.enabled:
                result.update(
                    {
                        "raw_tta_miou": primary_metrics[
                            "raw_tta_miou"
                        ],
                        "raw_tta_per_class_iou": primary_metrics[
                            "raw_tta_per_class_iou"
                        ],
                        "raw_delta_miou": primary_metrics[
                            "raw_delta_miou"
                        ],
                        "raw_delta_per_class_iou": primary_metrics[
                            "raw_delta_per_class_iou"
                        ],
                        "raw_tta_valid_classes": primary_metrics[
                            "raw_tta_valid_classes"
                        ],
                        "raw_tta_macc": primary_metrics[
                            "raw_tta_macc"
                        ],
                        "raw_tta_per_class_accuracy": primary_metrics[
                            "raw_tta_per_class_accuracy"
                        ],
                        "raw_tta_valid_accuracy_classes": primary_metrics[
                            "raw_tta_valid_accuracy_classes"
                        ],
                        "raw_tta_aacc": primary_metrics[
                            "raw_tta_aacc"
                        ],
                        "raw_tta_confusion_matrix": primary_metrics[
                            "raw_tta_confusion_matrix"
                        ],
                        "raw_delta_macc": primary_metrics[
                            "raw_delta_macc"
                        ],
                        "raw_delta_per_class_accuracy": primary_metrics[
                            "raw_delta_per_class_accuracy"
                        ],
                        "raw_delta_aacc": primary_metrics[
                            "raw_delta_aacc"
                        ],
                        "guarded_images": int(guarded_total),
                        "guard_rate": float(
                            guarded_total / max(processed_total, 1.0)
                        ),
                    }
                )
            if (
                self.diagnose_teacher_prompts
                or self.diagnose_teacher_prompt_gradients
                or self.teacher_prompt_trial_steps > 0
            ):
                result["teacher_prompt_diagnostic"] = {
                    "class_ids": list(resolved_teacher_diagnostic_classes),
                    "query_words": list(adapter.query_words),
                    "query_idx_list": [
                        int(value) for value in adapter.query_idx_list
                    ],
                    "canonical_query_ids": list(
                        mining_canonical_query_ids or ()
                    ),
                    "label_free_fields": [
                        "presence",
                        "present",
                        "candidate_pixels",
                        "selected_pixels",
                        "selected_score_mean",
                        "selected_score_q10",
                        "same_class_support_mean",
                        "same_class_agreement_mean",
                        "foreign_max_mean",
                        "margin_mean",
                        "margin_q10",
                        "candidate_wins_foreign_fraction",
                    ],
                    "gt_only_fields": [
                        "gt_topk_correct",
                        "gt_topk_evaluable",
                        "gt_topk_precision",
                        "gt_distribution",
                    ],
                }
            if self.diagnose_teacher_prompt_gradients:
                result["teacher_prompt_gradient_diagnostic"] = {
                    "class_ids": list(resolved_teacher_diagnostic_classes),
                    "selection_uses_gt": False,
                    "parameter_updates_applied": False,
                    "candidate_gradient": (
                        "target class semantic and presence losses only"
                    ),
                    "stable_gradient": (
                        "all configured classes except the target class"
                    ),
                    "label_free_fields": [
                        "gradient_norm",
                        "stable_norm_ratio",
                        "stable_cosine",
                        "stable_conflict",
                        "stable_sign_agreement",
                        "consensus_cosine",
                        "consensus_sign_agreement",
                        "mean_pairwise_cosine",
                        "min_pairwise_cosine",
                        "positive_pair_fraction",
                    ],
                }
            if self.teacher_prompt_trial_steps > 0:
                result["teacher_prompt_trial_diagnostic"] = {
                    "class_ids": list(resolved_teacher_diagnostic_classes),
                    "trial_steps": int(self.teacher_prompt_trial_steps),
                    "selection_uses_gt": False,
                    "label_free_fields": [
                        "transfer.own_core",
                        "transfer.own_shoulder",
                        "transfer.foreign_core",
                        "transfer.foreign_shoulder",
                        "transfer.canonical_anchor_agreement",
                        "transfer.raw_anchor_agreement",
                        "transfer.raw_class_expansion_rate",
                        "transfer.raw_class_contraction_rate",
                    ],
                    "gt_only_fields": ["gt_audit_final_heads"],
                }
            synonym_report = None
            cross_time_report = None
            if synonym_diagnostic is not None:
                synonym_report = synonym_diagnostic.finalize()
                result["synonym_diagnostic"] = synonym_report
            if cross_time_diagnostic is not None:
                cross_time_report = _finalize_cross_time_diagnostic(
                    cross_time_diagnostic,
                    baseline_metrics=baseline_metrics,
                    tta_metrics=tta_metrics,
                    raw_head=cross_time_raw_head,
                    canonical_head=cross_time_canonical_head,
                )
                result["cross_time_causal"] = cross_time_report
            _merge_synonym_cross_time_diagnostics(
                synonym_report=synonym_report,
                cross_time_report=cross_time_report,
            )
            if prompt_routing_diagnostic is not None:
                result["prompt_routing_diagnostic"] = (
                    prompt_routing_diagnostic.finalize()
                )
            if dist_ctx.is_main:
                output_path = Path(runtime.output_json)
                if not output_path.is_absolute():
                    output_path = Path.cwd() / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(_format_final_report(result), flush=True)
                print(f"[clean-tta] wrote {output_path}", flush=True)
            return result
        finally:
            cleanup_distributed()
