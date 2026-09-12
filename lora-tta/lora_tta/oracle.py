from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .config import OracleDiagnosticConfig
from .losses import SelectedClassEntry
from .mining import as_chw_class_scores


@dataclass(frozen=True, order=True)
class OracleCandidate:
    lr_multiplier: float
    target_presence_power: float

    @property
    def key(self) -> str:
        return f"lr{self.lr_multiplier:g}_p{self.target_presence_power:g}"


def build_oracle_candidates(
    config: OracleDiagnosticConfig,
) -> tuple[OracleCandidate, ...]:
    return tuple(
        OracleCandidate(lr_multiplier, target_power)
        for lr_multiplier in config.lr_multipliers
        for target_power in config.target_presence_powers
    )


def select_oracle_candidate(
    scores: dict[OracleCandidate, float],
    *,
    reference: OracleCandidate,
) -> OracleCandidate:
    if not scores:
        raise ValueError("at least one oracle candidate score is required")

    def key(item: tuple[OracleCandidate, float]) -> tuple[float, float, float, float]:
        candidate, score = item
        distance = abs(
            candidate.lr_multiplier - reference.lr_multiplier
        ) + abs(
            candidate.target_presence_power
            - reference.target_presence_power
        )
        return (
            float(score),
            -float(distance),
            -float(candidate.lr_multiplier),
            -float(candidate.target_presence_power),
        )

    return max(scores.items(), key=key)[0]


def select_oracle_candidate_or_no_update(
    scores: dict[OracleCandidate, float],
    *,
    baseline_score: float,
    reference: OracleCandidate,
) -> OracleCandidate | None:
    """Return ``None`` when adapting does not strictly beat the baseline."""
    best = select_oracle_candidate(scores, reference=reference)
    if float(baseline_score) >= float(scores[best]):
        return None
    return best


def _as_hw_prediction(pred: torch.Tensor) -> torch.Tensor:
    pred = pred.detach()
    if pred.ndim == 3 and int(pred.shape[0]) == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"expected prediction [H,W], got {tuple(pred.shape)}")
    return pred.long()


def extract_output_state(
    *,
    pred: torch.Tensor,
    class_scores: torch.Tensor,
    class_presence: torch.Tensor,
    num_classes: int,
) -> dict:
    """Summarize a final-head output without consulting ground truth."""
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    pred = _as_hw_prediction(pred)
    class_scores = as_chw_class_scores(class_scores).detach().float()
    class_presence = class_presence.detach().float().flatten()
    if int(class_scores.shape[0]) != num_classes:
        raise ValueError(
            "class score count does not match num_classes: "
            f"{int(class_scores.shape[0])} != {num_classes}"
        )
    if int(class_presence.numel()) != num_classes:
        raise ValueError(
            "class presence count does not match num_classes: "
            f"{int(class_presence.numel())} != {num_classes}"
        )
    if tuple(class_scores.shape[-2:]) != tuple(pred.shape):
        raise ValueError(
            "class score and prediction shapes must match: "
            f"{tuple(class_scores.shape[-2:])} != {tuple(pred.shape)}"
        )
    if bool(((pred < 0) | (pred >= num_classes)).any()):
        raise ValueError("prediction contains a class outside the configured range")

    pixel_count = int(pred.numel())
    predicted_classes = sorted(
        int(value) for value in torch.unique(pred).detach().cpu().tolist()
    )
    per_class = {}
    for class_id in range(num_classes):
        area_pixels = int((pred == class_id).sum().item())
        per_class[str(class_id)] = {
            "presence": float(class_presence[class_id].item()),
            "area_pixels": area_pixels,
            "area_ratio": float(area_pixels / pixel_count),
            "max_score": float(class_scores[class_id].max().item()),
        }
    return {
        "predicted_classes": predicted_classes,
        "per_class": per_class,
    }


def compare_output_states(
    *,
    baseline_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    baseline_state: dict,
    candidate_state: dict,
) -> dict:
    """Measure candidate output drift relative to the pre-TTA final head."""
    baseline_pred = _as_hw_prediction(baseline_pred)
    candidate_pred = _as_hw_prediction(candidate_pred)
    if tuple(baseline_pred.shape) != tuple(candidate_pred.shape):
        raise ValueError("baseline and candidate prediction shapes must match")

    baseline_classes = set(int(v) for v in baseline_state["predicted_classes"])
    candidate_classes = set(int(v) for v in candidate_state["predicted_classes"])
    baseline_per_class = baseline_state["per_class"]
    candidate_per_class = candidate_state["per_class"]
    if set(baseline_per_class) != set(candidate_per_class):
        raise ValueError("baseline and candidate class state keys must match")

    presence_deltas = []
    area_ratio_deltas = []
    max_score_deltas = []
    per_class = {}
    for class_id in sorted(baseline_per_class, key=int):
        baseline = baseline_per_class[class_id]
        candidate = candidate_per_class[class_id]
        presence_delta = float(candidate["presence"] - baseline["presence"])
        area_delta_pixels = int(
            candidate["area_pixels"] - baseline["area_pixels"]
        )
        area_ratio_delta = float(
            candidate["area_ratio"] - baseline["area_ratio"]
        )
        max_score_delta = float(
            candidate["max_score"] - baseline["max_score"]
        )
        presence_deltas.append(abs(presence_delta))
        area_ratio_deltas.append(abs(area_ratio_delta))
        max_score_deltas.append(abs(max_score_delta))
        per_class[class_id] = {
            "presence": float(candidate["presence"]),
            "presence_delta": presence_delta,
            "area_pixels": int(candidate["area_pixels"]),
            "area_delta_pixels": area_delta_pixels,
            "area_ratio": float(candidate["area_ratio"]),
            "area_ratio_delta": area_ratio_delta,
            "max_score": float(candidate["max_score"]),
            "max_score_delta": max_score_delta,
        }

    return {
        "prediction_change_ratio": float(
            (baseline_pred != candidate_pred).float().mean().item()
        ),
        "predicted_classes": sorted(candidate_classes),
        "new_predicted_classes": sorted(candidate_classes - baseline_classes),
        "disappeared_predicted_classes": sorted(
            baseline_classes - candidate_classes
        ),
        "new_predicted_class_count": len(candidate_classes - baseline_classes),
        "disappeared_predicted_class_count": len(
            baseline_classes - candidate_classes
        ),
        "max_abs_presence_delta": max(presence_deltas, default=0.0),
        "mean_abs_area_ratio_delta": float(
            sum(area_ratio_deltas) / max(len(area_ratio_deltas), 1)
        ),
        "mean_abs_max_score_delta": float(
            sum(max_score_deltas) / max(len(max_score_deltas), 1)
        ),
        "per_class": per_class,
    }


def capture_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, ...]:
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def apply_optimizer_lr_multiplier(
    optimizer: torch.optim.Optimizer,
    base_lrs: tuple[float, ...],
    multiplier: float,
) -> None:
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("optimizer group count changed during oracle trial")
    multiplier = float(multiplier)
    if multiplier <= 0.0:
        raise ValueError("oracle learning-rate multiplier must be positive")
    for group, base_lr in zip(
        optimizer.param_groups,
        base_lrs,
        strict=True,
    ):
        group["lr"] = float(base_lr) * multiplier


def restore_optimizer_lrs(
    optimizer: torch.optim.Optimizer,
    base_lrs: tuple[float, ...],
) -> None:
    apply_optimizer_lr_multiplier(optimizer, base_lrs, 1.0)


def rebuild_selected_targets(
    selected: dict[int, SelectedClassEntry],
    *,
    target_scores: torch.Tensor,
    minimum: float,
    maximum: float,
    offset: float,
) -> dict[int, SelectedClassEntry]:
    target_scores = as_chw_class_scores(target_scores)
    rebuilt: dict[int, SelectedClassEntry] = {}
    for class_id, entry in selected.items():
        flat_idx = entry.flat_idx.to(
            device=target_scores.device,
            dtype=torch.long,
        )
        values = target_scores[int(class_id)].flatten()[flat_idx]
        rebuilt[int(class_id)] = replace(
            entry,
            positive_targets=(values.float() + float(offset))
            .clamp(float(minimum), float(maximum))
            .detach(),
        )
    return rebuilt


def _quantile_fields(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float().flatten()
    if int(values.numel()) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_q90": 0.0,
        }
    quantiles = torch.quantile(
        values,
        values.new_tensor([0.1, 0.5, 0.9]),
    )
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_q10": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_q90": float(quantiles[2].item()),
    }


def _component_fields(mask: torch.Tensor) -> dict[str, float | int]:
    pixel_count = int(mask.sum().item())
    if pixel_count == 0:
        return {
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "fragmentation": 0.0,
        }
    if mask.is_cuda:
        from sam3.perflib.connected_components import connected_components

        labels, component_sizes = connected_components(
            mask.to(torch.uint8).unsqueeze(0).unsqueeze(0).contiguous()
        )
    else:
        from skimage.measure import label

        labels_array, _count = label(
            mask.detach().cpu().numpy(),
            return_num=True,
        )
        labels = torch.from_numpy(labels_array)
        flat_labels = labels.flatten().long()
        counts = torch.bincount(flat_labels)
        component_sizes = counts[flat_labels].reshape_as(labels)
    foreground_labels = labels[labels > 0]
    component_count = int(torch.unique(foreground_labels).numel())
    largest_component = int(component_sizes.max().item())
    return {
        "component_count": component_count,
        "largest_component_ratio": float(largest_component / pixel_count),
        "fragmentation": float(component_count / pixel_count),
    }


_BRANCH_RATIO_FIELDS = (
    "semantic_branch_winner_ratio",
    "instance_branch_winner_ratio",
    "branch_tie_ratio",
    "semantic_branch_only_ratio",
    "instance_branch_only_ratio",
    "both_branch_high_ratio",
    "both_branch_low_ratio",
    "gated_semantic_instance_agreement",
)


def _unavailable_branch_attribution_fields() -> dict[str, bool | None]:
    return {
        "branch_attribution_available": False,
        **{field: None for field in _BRANCH_RATIO_FIELDS},
    }


def _branch_attribution_fields(
    semantic_scores: torch.Tensor,
    instance_scores: torch.Tensor,
    *,
    prob_thd: float,
) -> dict[str, bool | float]:
    semantic_scores = semantic_scores.detach().float().flatten()
    instance_scores = instance_scores.detach().float().flatten()
    if tuple(semantic_scores.shape) != tuple(instance_scores.shape):
        raise ValueError("semantic and instance attribution shapes must match")

    def ratio(mask: torch.Tensor) -> float:
        if int(mask.numel()) == 0:
            return 0.0
        return float(mask.float().mean().item())

    delta = semantic_scores - instance_scores
    semantic_winner = delta > 1e-6
    instance_winner = delta < -1e-6
    tie = ~(semantic_winner | instance_winner)
    semantic_high = semantic_scores >= float(prob_thd)
    instance_high = instance_scores >= float(prob_thd)

    return {
        "branch_attribution_available": True,
        "semantic_branch_winner_ratio": ratio(semantic_winner),
        "instance_branch_winner_ratio": ratio(instance_winner),
        "branch_tie_ratio": ratio(tie),
        "semantic_branch_only_ratio": ratio(semantic_high & ~instance_high),
        "instance_branch_only_ratio": ratio(instance_high & ~semantic_high),
        "both_branch_high_ratio": ratio(semantic_high & instance_high),
        "both_branch_low_ratio": ratio(~semantic_high & ~instance_high),
        "gated_semantic_instance_agreement": ratio(delta.abs() <= 0.100001),
    }


def extract_teacher_features(
    *,
    class_scores: torch.Tensor,
    class_presence: torch.Tensor,
    semantic_class_scores: torch.Tensor,
    instance_class_scores: torch.Tensor,
    gated_semantic_class_scores: torch.Tensor | None = None,
    reference_target_scores: torch.Tensor,
    selected: dict[int, SelectedClassEntry],
    raw_pred: torch.Tensor,
    valid: torch.Tensor,
    prob_thd: float,
    presence_gate_power: float,
    score_fusion_mode: str,
    synonym_class_scores: torch.Tensor | None = None,
) -> dict:
    class_scores = as_chw_class_scores(class_scores).detach().float()
    semantic_scores = as_chw_class_scores(semantic_class_scores).detach().float()
    instance_scores = as_chw_class_scores(instance_class_scores).detach().float()
    gated_semantic_scores = (
        None
        if gated_semantic_class_scores is None
        else as_chw_class_scores(gated_semantic_class_scores).detach().float()
    )
    reference_targets = as_chw_class_scores(reference_target_scores).detach().float()
    class_presence = class_presence.detach().float().flatten()
    valid = valid.to(device=class_scores.device, dtype=torch.bool)
    raw_pred = raw_pred.to(device=class_scores.device, dtype=torch.long)
    if tuple(raw_pred.shape) != tuple(valid.shape):
        raise ValueError("raw prediction and valid mask shapes must match")
    expected_shape = (int(class_scores.shape[0]), *tuple(valid.shape))
    for name, values in (
        ("semantic", semantic_scores),
        ("instance", instance_scores),
        ("reference target", reference_targets),
    ):
        if tuple(values.shape) != expected_shape:
            raise ValueError(f"{name} class score shape must match mining scores")
    if int(class_presence.numel()) != int(class_scores.shape[0]):
        raise ValueError("class presence length must match class scores")
    presence_gate_power = float(presence_gate_power)
    if not math.isfinite(presence_gate_power) or presence_gate_power < 0.0:
        raise ValueError("presence gate power must be finite and non-negative")
    if score_fusion_mode not in {"legacy_max", "proc_pgrf"}:
        raise ValueError(f"unknown score fusion mode: {score_fusion_mode}")
    branch_attribution_available = score_fusion_mode == "legacy_max"
    if gated_semantic_scores is None:
        gated_semantic_scores = semantic_scores * class_presence[:, None, None].pow(
            presence_gate_power
        )
    elif tuple(gated_semantic_scores.shape) != expected_shape:
        raise ValueError("gated semantic class score shape must match mining scores")

    class_ids = tuple(sorted(int(class_id) for class_id in selected))
    selected_union = torch.zeros_like(valid)
    all_scores = []
    all_margins = []
    all_target_gaps = []
    all_target_abs_gaps = []
    all_sem_inst_abs = []
    semantic_only = []
    all_gated_semantic = []
    all_instance = []
    per_class = {}
    num_classes = int(class_scores.shape[0])

    for class_id in class_ids:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"selected class id {class_id} is out of range")
        entry = selected[class_id]
        flat_idx = entry.flat_idx.to(
            device=class_scores.device,
            dtype=torch.long,
        )
        class_mask = torch.zeros_like(valid).flatten()
        class_mask[flat_idx] = True
        class_mask = class_mask.view_as(valid) & valid
        selected_union |= class_mask
        effective_idx = class_mask.flatten().nonzero(as_tuple=False).flatten()

        score_values = class_scores[class_id].flatten()[effective_idx]
        if num_classes > 1:
            other_ids = [idx for idx in range(num_classes) if idx != class_id]
            other_scores = class_scores[other_ids].reshape(
                len(other_ids),
                -1,
            )[:, effective_idx]
            margins = score_values - other_scores.max(dim=0).values
        else:
            margins = score_values
        semantic_values = semantic_scores[class_id].flatten()[effective_idx]
        instance_values = instance_scores[class_id].flatten()[effective_idx]
        gated_semantic_values = gated_semantic_scores[class_id].flatten()[
            effective_idx
        ]
        target_values = reference_targets[class_id].flatten()[effective_idx]
        target_gaps = semantic_values - target_values
        sem_inst_abs = (semantic_values - instance_values).abs()
        semantic_only_values = (
            (semantic_values >= float(prob_thd))
            & (instance_values < float(prob_thd))
        ).float()

        class_fields = {
            "selected_pixels": int(effective_idx.numel()),
            "presence": float(class_presence[class_id].item()),
            **_quantile_fields(score_values, "selected_score"),
            **_quantile_fields(margins, "selected_margin"),
            **_quantile_fields(target_gaps, "raw_target_gap"),
            "semantic_instance_agreement": float(
                (sem_inst_abs <= 0.100001).float().mean().item()
            ) if int(sem_inst_abs.numel()) else 0.0,
            "semantic_only_ratio": float(semantic_only_values.mean().item())
            if int(semantic_only_values.numel())
            else 0.0,
            **(
                _branch_attribution_fields(
                    gated_semantic_values,
                    instance_values,
                    prob_thd=prob_thd,
                )
                if branch_attribution_available
                else _unavailable_branch_attribution_fields()
            ),
            **_component_fields(class_mask),
        }
        per_class[str(class_id)] = class_fields
        all_scores.append(score_values)
        all_margins.append(margins)
        all_target_gaps.append(target_gaps)
        all_target_abs_gaps.append(target_gaps.abs())
        all_sem_inst_abs.append(sem_inst_abs)
        semantic_only.append(semantic_only_values)
        all_gated_semantic.append(gated_semantic_values)
        all_instance.append(instance_values)

    def cat_or_empty(parts: list[torch.Tensor]) -> torch.Tensor:
        if parts:
            return torch.cat(parts)
        return class_scores.new_empty((0,))

    present_score_maps = (
        class_scores[list(class_ids)]
        if class_ids
        else class_scores.new_empty((0, *valid.shape))
    )
    if int(present_score_maps.shape[0]) >= 2:
        conflict = (present_score_maps >= float(prob_thd)).sum(dim=0) >= 2
    else:
        conflict = torch.zeros_like(valid)
    valid_count = int(valid.sum().item())
    selected_count = int(selected_union.sum().item())

    synonym_agreement = None
    if synonym_class_scores is not None:
        synonym_scores = as_chw_class_scores(synonym_class_scores).to(
            device=class_scores.device,
            dtype=class_scores.dtype,
        )
        if tuple(synonym_scores.shape) != tuple(class_scores.shape):
            raise ValueError("synonym class score shape must match mining scores")
        synonym_pred = synonym_scores.argmax(dim=0)
        synonym_agreement = float(
            ((synonym_pred == raw_pred) & valid).sum().item()
            / max(valid_count, 1)
        )

    score_values = cat_or_empty(all_scores)
    margin_values = cat_or_empty(all_margins)
    target_gaps = cat_or_empty(all_target_gaps)
    target_abs_gaps = cat_or_empty(all_target_abs_gaps)
    sem_inst_abs = cat_or_empty(all_sem_inst_abs)
    semantic_only_values = cat_or_empty(semantic_only)
    gated_semantic_values = cat_or_empty(all_gated_semantic)
    instance_values = cat_or_empty(all_instance)
    return {
        "selected_pixels": selected_count,
        "selected_classes": len(class_ids),
        **_quantile_fields(score_values, "selected_score"),
        **_quantile_fields(margin_values, "selected_margin"),
        **_quantile_fields(target_gaps, "raw_target_gap"),
        "mean_abs_raw_target_gap": float(target_abs_gaps.mean().item())
        if int(target_abs_gaps.numel())
        else 0.0,
        "semantic_instance_agreement": float(
            (sem_inst_abs <= 0.100001).float().mean().item()
        ) if int(sem_inst_abs.numel()) else 0.0,
        "semantic_only_ratio": float(semantic_only_values.mean().item())
        if int(semantic_only_values.numel())
        else 0.0,
        **(
            _branch_attribution_fields(
                gated_semantic_values,
                instance_values,
                prob_thd=prob_thd,
            )
            if branch_attribution_available
            else _unavailable_branch_attribution_fields()
        ),
        "conflict_rate_valid": float(
            (conflict & valid).sum().item() / max(valid_count, 1)
        ),
        "conflict_rate_selected": float(
            (conflict & selected_union).sum().item() / max(selected_count, 1)
        ),
        "synonym_agreement_available": synonym_agreement is not None,
        "synonym_agreement": synonym_agreement,
        "per_class": per_class,
    }


def gradient_norm_features(
    params_by_layer: dict[int, list[torch.nn.Parameter]],
) -> dict:
    per_layer: dict[str, float] = {}
    for layer, params in sorted(params_by_layer.items()):
        squared_norm = 0.0
        for parameter in params:
            if parameter.grad is not None:
                squared_norm += float(
                    parameter.grad.detach().float().square().sum().item()
                )
        per_layer[str(int(layer))] = math.sqrt(squared_norm)
    return {
        "gradient_norm": math.sqrt(
            sum(value * value for value in per_layer.values())
        ),
        "gradient_norm_per_layer": per_layer,
    }
