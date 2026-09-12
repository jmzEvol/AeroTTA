from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeVar

import torch


Prediction = TypeVar("Prediction")
PROTECTION_ACTIONS = frozenset(
    {"baseline_fallback", "unsupported_pixel_fallback"}
)


@dataclass(frozen=True)
class ProtectionDecision:
    triggered: bool
    action: str
    baseline_classes: tuple[int, ...]
    selected_classes: tuple[int, ...]
    raw_tta_classes: tuple[int, ...]
    new_classes: tuple[int, ...]
    unsupported_new_classes: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "triggered": bool(self.triggered),
            "action": self.action,
            "baseline_classes": list(self.baseline_classes),
            "selected_classes": list(self.selected_classes),
            "raw_tta_classes": list(self.raw_tta_classes),
            "new_classes": list(self.new_classes),
            "unsupported_new_classes": list(
                self.unsupported_new_classes
            ),
        }


@dataclass(frozen=True)
class ClassSurvivalDecision:
    triggered: bool
    threshold: float
    min_baseline_pixels: int
    checked_classes: tuple[int, ...]
    collapsing_classes: tuple[int, ...]
    baseline_pixels: dict[int, int]
    candidate_pixels: dict[int, int]
    survival_ratios: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "triggered": bool(self.triggered),
            "threshold": float(self.threshold),
            "min_baseline_pixels": int(self.min_baseline_pixels),
            "checked_classes": list(self.checked_classes),
            "collapsing_classes": list(self.collapsing_classes),
            "baseline_pixels": {
                str(class_id): int(value)
                for class_id, value in self.baseline_pixels.items()
            },
            "candidate_pixels": {
                str(class_id): int(value)
                for class_id, value in self.candidate_pixels.items()
            },
            "survival_ratios": {
                str(class_id): float(value)
                for class_id, value in self.survival_ratios.items()
            },
        }


@dataclass(frozen=True)
class ClassRetentionDecision:
    triggered: bool
    threshold: float
    min_baseline_pixels: int
    checked_classes: tuple[int, ...]
    collapsing_classes: tuple[int, ...]
    baseline_pixels: dict[int, int]
    retained_pixels: dict[int, int]
    retention_ratios: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "enabled": True,
            "metric": "retention",
            "triggered": bool(self.triggered),
            "threshold": float(self.threshold),
            "min_baseline_pixels": int(self.min_baseline_pixels),
            "checked_classes": list(self.checked_classes),
            "collapsing_classes": list(self.collapsing_classes),
            "baseline_pixels": {
                str(class_id): int(value)
                for class_id, value in self.baseline_pixels.items()
            },
            "retained_pixels": {
                str(class_id): int(value)
                for class_id, value in self.retained_pixels.items()
            },
            "retention_ratios": {
                str(class_id): float(value)
                for class_id, value in self.retention_ratios.items()
            },
        }


def predicted_class_ids(prediction: torch.Tensor) -> tuple[int, ...]:
    prediction = prediction.detach()
    if prediction.ndim == 3 and int(prediction.shape[0]) == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(
            f"expected prediction [H,W], got {tuple(prediction.shape)}"
        )
    return tuple(
        sorted(
            int(class_id)
            for class_id in torch.unique(prediction).cpu().tolist()
        )
    )


def decide_supported_class_survival(
    *,
    baseline_pred: torch.Tensor,
    raw_tta_pred: torch.Tensor,
    selected_class_ids,
    threshold: float,
    min_baseline_pixels: int = 1,
) -> ClassSurvivalDecision:
    threshold = float(threshold)
    min_baseline_pixels = int(min_baseline_pixels)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("survival threshold must be in (0, 1]")
    if min_baseline_pixels <= 0:
        raise ValueError("minimum baseline pixels must be positive")

    baseline_pred = baseline_pred.detach()
    raw_tta_pred = raw_tta_pred.detach()
    if baseline_pred.ndim == 3 and int(baseline_pred.shape[0]) == 1:
        baseline_pred = baseline_pred[0]
    if raw_tta_pred.ndim == 3 and int(raw_tta_pred.shape[0]) == 1:
        raw_tta_pred = raw_tta_pred[0]
    if baseline_pred.ndim != 2 or raw_tta_pred.ndim != 2:
        raise ValueError("survival predictions must be [H,W]")
    if tuple(baseline_pred.shape) != tuple(raw_tta_pred.shape):
        raise ValueError("survival prediction shapes must match")

    baseline_pixels: dict[int, int] = {}
    candidate_pixels: dict[int, int] = {}
    survival_ratios: dict[int, float] = {}
    for class_id in sorted({int(value) for value in selected_class_ids}):
        baseline_count = int((baseline_pred == class_id).sum().item())
        if baseline_count < min_baseline_pixels:
            continue
        candidate_count = int((raw_tta_pred == class_id).sum().item())
        baseline_pixels[class_id] = baseline_count
        candidate_pixels[class_id] = candidate_count
        survival_ratios[class_id] = float(candidate_count / baseline_count)

    collapsing_classes = tuple(
        class_id
        for class_id, ratio in survival_ratios.items()
        if ratio < threshold
    )
    return ClassSurvivalDecision(
        triggered=bool(collapsing_classes),
        threshold=threshold,
        min_baseline_pixels=min_baseline_pixels,
        checked_classes=tuple(baseline_pixels),
        collapsing_classes=collapsing_classes,
        baseline_pixels=baseline_pixels,
        candidate_pixels=candidate_pixels,
        survival_ratios=survival_ratios,
    )


def decide_supported_class_retention(
    *,
    baseline_pred: torch.Tensor,
    raw_tta_pred: torch.Tensor,
    selected_class_ids,
    threshold: float,
    min_baseline_pixels: int = 1,
) -> ClassRetentionDecision:
    threshold = float(threshold)
    min_baseline_pixels = int(min_baseline_pixels)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("retention threshold must be in (0, 1]")
    if min_baseline_pixels <= 0:
        raise ValueError("minimum baseline pixels must be positive")

    baseline_pred = baseline_pred.detach()
    raw_tta_pred = raw_tta_pred.detach()
    if baseline_pred.ndim == 3 and int(baseline_pred.shape[0]) == 1:
        baseline_pred = baseline_pred[0]
    if raw_tta_pred.ndim == 3 and int(raw_tta_pred.shape[0]) == 1:
        raw_tta_pred = raw_tta_pred[0]
    if baseline_pred.ndim != 2 or raw_tta_pred.ndim != 2:
        raise ValueError("retention predictions must be [H,W]")
    if tuple(baseline_pred.shape) != tuple(raw_tta_pred.shape):
        raise ValueError("retention prediction shapes must match")

    baseline_pixels: dict[int, int] = {}
    retained_pixels: dict[int, int] = {}
    retention_ratios: dict[int, float] = {}
    for class_id in sorted({int(value) for value in selected_class_ids}):
        baseline_support = baseline_pred == class_id
        baseline_count = int(baseline_support.sum().item())
        if baseline_count < min_baseline_pixels:
            continue
        retained_count = int(
            (baseline_support & (raw_tta_pred == class_id)).sum().item()
        )
        baseline_pixels[class_id] = baseline_count
        retained_pixels[class_id] = retained_count
        retention_ratios[class_id] = float(
            retained_count / baseline_count
        )

    collapsing_classes = tuple(
        class_id
        for class_id, ratio in retention_ratios.items()
        if ratio < threshold
    )
    return ClassRetentionDecision(
        triggered=bool(collapsing_classes),
        threshold=threshold,
        min_baseline_pixels=min_baseline_pixels,
        checked_classes=tuple(baseline_pixels),
        collapsing_classes=collapsing_classes,
        baseline_pixels=baseline_pixels,
        retained_pixels=retained_pixels,
        retention_ratios=retention_ratios,
    )


def decide_unsupported_class_birth(
    *,
    baseline_pred: torch.Tensor,
    selected_class_ids,
    raw_tta_pred: torch.Tensor,
    action: str = "baseline_fallback",
) -> ProtectionDecision:
    if action not in PROTECTION_ACTIONS:
        raise ValueError(f"unsupported protection action: {action}")

    baseline_classes = predicted_class_ids(baseline_pred)
    raw_tta_classes = predicted_class_ids(raw_tta_pred)
    selected_classes = tuple(
        sorted({int(class_id) for class_id in selected_class_ids})
    )
    new_classes = tuple(
        sorted(set(raw_tta_classes) - set(baseline_classes))
    )
    unsupported_new_classes = tuple(
        sorted(set(new_classes) - set(selected_classes))
    )
    return ProtectionDecision(
        triggered=bool(unsupported_new_classes),
        action=action,
        baseline_classes=baseline_classes,
        selected_classes=selected_classes,
        raw_tta_classes=raw_tta_classes,
        new_classes=new_classes,
        unsupported_new_classes=unsupported_new_classes,
    )


def apply_protection_decision(
    *,
    baseline_predictions: Mapping[str, Prediction],
    raw_tta_predictions: Mapping[str, Prediction],
    decision: ProtectionDecision,
) -> dict[str, Prediction]:
    if set(baseline_predictions) != set(raw_tta_predictions):
        raise ValueError("baseline and raw TTA final head keys must match")
    if decision.action not in PROTECTION_ACTIONS:
        raise ValueError(f"unsupported protection action: {decision.action}")
    if not decision.triggered:
        return {
            head_name: raw_tta_predictions[head_name]
            for head_name in raw_tta_predictions
        }
    if decision.action == "baseline_fallback":
        return {
            head_name: baseline_predictions[head_name]
            for head_name in baseline_predictions
        }

    protected = {}
    for head_name, raw_prediction in raw_tta_predictions.items():
        baseline_prediction = baseline_predictions[head_name]
        if tuple(raw_prediction.shape) != tuple(baseline_prediction.shape):
            raise ValueError(
                "baseline and raw TTA prediction shapes must match for "
                f"head {head_name!r}"
            )
        unsupported = torch.zeros_like(raw_prediction, dtype=torch.bool)
        for class_id in decision.unsupported_new_classes:
            unsupported |= raw_prediction == int(class_id)
        if not bool(unsupported.any()):
            protected[head_name] = raw_prediction
            continue
        output = raw_prediction.clone()
        output[unsupported] = baseline_prediction[unsupported]
        protected[head_name] = output
    return protected


def summarize_prediction_replacements(
    *,
    raw_tta_predictions: Mapping[str, torch.Tensor],
    protected_predictions: Mapping[str, torch.Tensor],
) -> dict:
    if set(raw_tta_predictions) != set(protected_predictions):
        raise ValueError("raw and protected final head keys must match")

    replaced_pixels_by_head = {}
    replaced_ratio_by_head = {}
    for head_name, raw_prediction in raw_tta_predictions.items():
        protected_prediction = protected_predictions[head_name]
        if tuple(raw_prediction.shape) != tuple(protected_prediction.shape):
            raise ValueError(
                "raw and protected prediction shapes must match for "
                f"head {head_name!r}"
            )
        pixel_count = int(raw_prediction.numel())
        if pixel_count <= 0:
            raise ValueError("predictions must contain at least one pixel")
        replaced_pixels = int(
            (raw_prediction != protected_prediction).sum().item()
        )
        replaced_pixels_by_head[head_name] = replaced_pixels
        replaced_ratio_by_head[head_name] = float(
            replaced_pixels / pixel_count
        )
    return {
        "replaced_pixels_by_head": replaced_pixels_by_head,
        "replaced_ratio_by_head": replaced_ratio_by_head,
    }
