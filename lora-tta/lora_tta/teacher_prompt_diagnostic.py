from __future__ import annotations

from collections.abc import Sequence

import torch

from .config import LossConfig, MiningConfig
from .mining import build_component_gate_masks, select_classwise_pixels
from .prompt_mining_audit import build_prompt_class_views


def _optional_mean(values: torch.Tensor) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(values.float().mean().item())


def _optional_quantile(values: torch.Tensor, quantile: float) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(torch.quantile(values.float(), float(quantile)).item())


def _validate_inputs(
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    query_idx_list: Sequence[int],
    query_words: Sequence[str],
    canonical_query_ids: Sequence[int],
    class_ids: Sequence[int],
    valid: torch.Tensor,
    gt: torch.Tensor | None,
) -> None:
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError(
            f"expected query_scores [1,Q,H,W], got {tuple(query_scores.shape)}"
        )
    if tuple(presence_logits.shape) != tuple(query_scores.shape[:2]):
        raise ValueError("presence/query shape mismatch")
    num_queries = int(query_scores.shape[1])
    if len(query_idx_list) != num_queries or len(query_words) != num_queries:
        raise ValueError("query metadata length mismatch")
    num_classes = len(canonical_query_ids)
    if num_classes <= 0:
        raise ValueError("canonical query ids must not be empty")
    if tuple(valid.shape) != tuple(query_scores.shape[-2:]):
        raise ValueError("valid mask/query score shape mismatch")
    if gt is not None and tuple(gt.shape[-2:]) != tuple(query_scores.shape[-2:]):
        raise ValueError("GT/query score shape mismatch")
    invalid = sorted(
        {
            int(class_id)
            for class_id in class_ids
            if int(class_id) < 0 or int(class_id) >= num_classes
        }
    )
    if invalid:
        raise ValueError(f"diagnostic class ids are outside range: {invalid}")


def evaluate_teacher_prompt_candidates(
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    query_idx_list: Sequence[int],
    query_words: Sequence[str],
    canonical_query_ids: Sequence[int],
    class_ids: Sequence[int],
    valid: torch.Tensor,
    gt: torch.Tensor | None,
    mining: MiningConfig,
    loss: LossConfig,
) -> list[dict]:
    """Observe how each query would mine TopK if used as its class teacher.

    All fields except the ``gt_*`` audit fields are label-free. The function
    reuses the production classwise miner and never mutates model parameters.
    """

    _validate_inputs(
        query_scores=query_scores,
        presence_logits=presence_logits,
        query_idx_list=query_idx_list,
        query_words=query_words,
        canonical_query_ids=canonical_query_ids,
        class_ids=class_ids,
        valid=valid,
        gt=gt,
    )
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    requested_classes = tuple(dict.fromkeys(int(value) for value in class_ids))
    num_classes = len(canonical_ids)
    views = build_prompt_class_views(
        query_scores=query_scores,
        presence_logits=presence_logits,
        query_idx_list=tuple(int(value) for value in query_idx_list),
        canonical_query_ids=canonical_ids,
        num_classes=num_classes,
    )
    valid = valid.to(device=query_scores.device, dtype=torch.bool)
    gt_device = None if gt is None else gt.to(device=query_scores.device).long()
    rows: list[dict] = []

    for class_id in requested_classes:
        foreign_ids = [index for index in range(num_classes) if index != class_id]
        same_class_ids = tuple(
            int(query_id)
            for query_id in views.query_ids_by_class[class_id]
        )
        for query_id in same_class_ids:
            class_scores = views.canonical_scores.clone()
            class_scores[class_id] = views.query_scores[query_id]
            class_presence = views.canonical_presence.clone()
            class_presence[class_id] = views.query_presence[query_id]
            component_gates = None
            if mining.component_gate:
                component_gates = build_component_gate_masks(
                    class_scores=class_scores,
                    class_presence=class_presence,
                    valid=valid,
                    mining=mining,
                )
            selected, present = select_classwise_pixels(
                class_scores=class_scores,
                class_presence=class_presence,
                raw_pred=class_scores.argmax(dim=0).long(),
                valid=valid,
                mining=mining,
                loss=loss,
                target_scores=class_scores,
                component_gate_masks=component_gates,
            )
            present_set = {
                int(value) for value in present.detach().cpu().tolist()
            }
            candidate_mask = valid & (
                views.query_scores[query_id] >= float(mining.prob_thd)
            )
            candidate_pixels = (
                int(candidate_mask.sum().item())
                if class_id in present_set
                else 0
            )
            entry = selected.get(class_id)
            flat_idx = (
                torch.empty(0, device=query_scores.device, dtype=torch.long)
                if entry is None
                else entry.flat_idx.to(device=query_scores.device).long()
            )
            own_values = views.query_scores[query_id].flatten()[flat_idx]
            foreign_values = (
                class_scores[foreign_ids]
                .reshape(len(foreign_ids), -1)[:, flat_idx]
                .max(dim=0)
                .values
            )
            margins = own_values - foreign_values
            other_query_ids = [
                value for value in same_class_ids if value != query_id
            ]
            if other_query_ids and int(flat_idx.numel()) > 0:
                other_scores = views.query_scores[other_query_ids].reshape(
                    len(other_query_ids), -1
                )[:, flat_idx]
                same_class_support = other_scores.max(dim=0).values
                same_class_agreement = (
                    other_scores >= float(mining.prob_thd)
                ).float().mean(dim=0)
            else:
                same_class_support = own_values.new_empty((0,))
                same_class_agreement = own_values.new_empty((0,))

            row = {
                "query_id": int(query_id),
                "query": str(query_words[query_id]),
                "class_id": int(class_id),
                "configured_teacher": query_id == canonical_ids[class_id],
                "presence": float(views.query_presence[query_id].item()),
                "present": class_id in present_set,
                "valid_pixels": int(valid.sum().item()),
                "candidate_pixels": candidate_pixels,
                "selected_pixels": int(flat_idx.numel()),
                "selected_score_mean": _optional_mean(own_values),
                "selected_score_q10": _optional_quantile(own_values, 0.10),
                "same_class_support_mean": _optional_mean(same_class_support),
                "same_class_agreement_mean": _optional_mean(
                    same_class_agreement
                ),
                "foreign_max_mean": _optional_mean(foreign_values),
                "margin_mean": _optional_mean(margins),
                "margin_q10": _optional_quantile(margins, 0.10),
                "candidate_wins_foreign_fraction": _optional_mean(
                    (margins > 0).float()
                ),
            }
            if gt_device is not None:
                labels = gt_device.flatten()[flat_idx]
                evaluable = (labels >= 0) & (labels < num_classes)
                labels = labels[evaluable]
                correct = int((labels == class_id).sum().item())
                evaluable_count = int(labels.numel())
                row.update(
                    {
                        "gt_topk_correct": correct,
                        "gt_topk_evaluable": evaluable_count,
                        "gt_topk_precision": (
                            float(correct) / float(evaluable_count)
                            if evaluable_count
                            else None
                        ),
                        "gt_distribution": [
                            int(value)
                            for value in torch.bincount(
                                labels,
                                minlength=num_classes,
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        ],
                    }
                )
            rows.append(row)
    return rows
