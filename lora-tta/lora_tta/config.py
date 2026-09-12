from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MiningConfig:
    sampling_mode: Literal["classwise_quota", "global_topk"] = "classwise_quota"
    tau_pos: float = 0.5
    tau_neg: float = 0.2
    prob_thd: float = 0.3
    selected_point_min_confidence: float | None = None
    presence_gate_power: float = 1.0
    inference_presence_gate_power: float | None = None
    score_fusion_mode: Literal["legacy_max", "proc_pgrf"] = "legacy_max"
    inference_score_fusion_mode: Literal[
        "legacy_max",
        "proc_pgrf",
    ] | None = None
    rho: float = 0.1
    kmax: int = 2048
    n_min: int = 64
    min_selected_classes: int = 1
    include_bg: bool = False
    bg_idx: int = 0
    pixel_weight_mode: Literal["none", "score"] = "score"
    class_weight_mode: Literal["none", "mean_margin", "mean_reliability"] = "mean_margin"
    class_weight_min: float = 0.3
    drop_overlaps: bool = True
    positive_argmax_only: bool = False
    competition_reliability_mode: Literal["none", "score_ratio"] = "none"
    target_score_source: Literal["class_fused", "raw_fused"] = "class_fused"
    bg_positive_source: Literal["prompt_score", "foreground_complement"] = "prompt_score"
    bg_complement_thd: float | None = None
    class_kmax_overrides: dict[int, int] = field(default_factory=dict)
    score_band_min: float | None = None
    score_band_max: float | None = None
    score_drop_top_frac: float = 0.0
    component_gate: bool = False
    component_fallback: Literal["none", "topk"] = "none"
    component_select_mode: Literal["topk", "whole"] = "topk"
    component_mask_thd: float = 0.5
    component_min_area: int = 4096
    component_mean_thd: float = 0.9

    def __post_init__(self) -> None:
        if self.positive_argmax_only and self.competition_reliability_mode != "none":
            raise ValueError(
                "positive_argmax_only and competition_reliability_mode "
                "cannot be enabled together"
            )
        if self.sampling_mode not in {"classwise_quota", "global_topk"}:
            raise ValueError(
                "sampling mode must be 'classwise_quota' or 'global_topk'"
            )
        if (
            self.sampling_mode == "global_topk"
            and self.component_select_mode == "whole"
        ):
            raise ValueError(
                "global_topk sampling cannot be combined with "
                "component_select_mode='whole'"
            )

    @property
    def resolved_inference_presence_gate_power(self) -> float:
        if self.inference_presence_gate_power is None:
            return float(self.presence_gate_power)
        return float(self.inference_presence_gate_power)

    @property
    def resolved_inference_score_fusion_mode(self) -> str:
        if self.inference_score_fusion_mode is None:
            return str(self.score_fusion_mode)
        return str(self.inference_score_fusion_mode)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LossConfig:
    student_score_mode: Literal[
        "semantic_raw",
        "filtered_raw_fusion",
        "filtered_raw_probability_reconstruction",
    ] = "semantic_raw"
    selected_loss_weight: float = 1.0
    contraction_gate_mode: Literal[
        "none",
        "decision_boundary",
    ] = "none"
    class_aggregation: Literal["balanced", "pooled"] = "balanced"
    presence_loss_weight: float = 0.0
    positive_target_mode: Literal["hard", "PTST"] = "hard"
    soft_target_presence_gate_power: float = 1.0
    soft_target_score_fusion_mode: Literal[
        "legacy_max",
        "proc_pgrf",
    ] | None = None
    sparse_selected_logits: bool = False
    positive_target_min: float = 0.0
    positive_target_max: float = 0.95
    positive_target_offset: float = 0.0
    low_score_neg_weight: float = 0.0
    low_score_neg_thd: float = 0.5
    low_score_neg_rho: float = 0.25
    low_score_neg_kmax: int = 512
    low_score_neg_n_min: int = 1

    def __post_init__(self) -> None:
        if self.selected_loss_weight < 0.0:
            raise ValueError("selected loss weight must be non-negative")
        if self.presence_loss_weight < 0.0:
            raise ValueError("presence loss weight must be non-negative")
        if self.student_score_mode not in {
            "semantic_raw",
            "filtered_raw_fusion",
            "filtered_raw_probability_reconstruction",
        }:
            raise ValueError(
                "student score mode must be 'semantic_raw', "
                "'filtered_raw_fusion', or "
                "'filtered_raw_probability_reconstruction'"
            )
        if self.contraction_gate_mode not in {
            "none",
            "decision_boundary",
        }:
            raise ValueError(
                "contraction gate mode must be 'none' or "
                "'decision_boundary'"
            )
        if self.class_aggregation not in {"balanced", "pooled"}:
            raise ValueError(
                "class aggregation must be 'balanced' or 'pooled'"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FinalHeadConfig:
    """One inference-only class-map policy for a shared TTA episode."""

    name: str
    canonical_class_ids: tuple[int, ...] = ()
    classname_path: str | None = None
    fusion_mode: Literal[
        "synonym_max",
        "topk_reliability",
        "post_tta_reliability",
        "frozen_alias_residual",
    ] = "synonym_max"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("final head name must not be empty")
        class_ids = tuple(int(class_id) for class_id in self.canonical_class_ids)
        if any(class_id < 0 for class_id in class_ids):
            raise ValueError("final head canonical class ids must be non-negative")
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("final head canonical class ids must not contain duplicates")
        classname_path = (
            str(self.classname_path).strip()
            if self.classname_path is not None
            else None
        )
        if classname_path == "":
            raise ValueError("final head classname path must not be empty")
        if classname_path is not None and not class_ids:
            raise ValueError(
                "final head classname path requires canonical class ids"
            )
        fusion_mode = str(self.fusion_mode).strip()
        valid_modes = {
            "synonym_max",
            "topk_reliability",
            "post_tta_reliability",
            "frozen_alias_residual",
        }
        if fusion_mode not in valid_modes:
            raise ValueError(
                "final head fusion_mode must be one of "
                f"{tuple(sorted(valid_modes))}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "canonical_class_ids", class_ids)
        object.__setattr__(self, "classname_path", classname_path)
        object.__setattr__(self, "fusion_mode", fusion_mode)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PromptViewConfig:
    """Choose which prompt view supplies TTA supervision.

    Final heads may keep synonym max, override selected classes with canonical
    prompts, or gate aliases from canonical TopK anchors.
    """

    canonical_classname_path: str | None = None
    mining_classname_path: str | None = None
    loss_classname_path: str | None = None
    adaptation_mode: Literal["classwise", "independent_queries"] = "classwise"
    mining_view: Literal["synonym", "canonical"] = "synonym"
    loss_view: Literal["synonym", "canonical"] = "synonym"
    auto_canonical_final_head: bool = False
    final_heads: tuple[FinalHeadConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.mining_classname_path and self.mining_view != "canonical":
            raise ValueError(
                "mining_classname_path requires mining_view='canonical'"
            )
        if self.loss_classname_path and self.loss_view != "canonical":
            raise ValueError(
                "loss_classname_path requires loss_view='canonical'"
            )
        final_heads = tuple(
            head
            if isinstance(head, FinalHeadConfig)
            else FinalHeadConfig(**dict(head))
            for head in self.final_heads
        )
        names = tuple(head.name for head in final_heads)
        if len(set(names)) != len(names):
            raise ValueError("final head names must be unique")
        object.__setattr__(self, "final_heads", final_heads)

    @property
    def needs_canonical_view(self) -> bool:
        uses_auto_canonical_head = (
            self.auto_canonical_final_head and not self.final_heads
        )
        return (
            self.adaptation_mode == "independent_queries"
            or self.mining_view == "canonical"
            or self.loss_view == "canonical"
            or self.mining_classname_path is not None
            or self.loss_classname_path is not None
            or uses_auto_canonical_head
            or any(head.canonical_class_ids for head in self.final_heads)
            or any(head.fusion_mode != "synonym_max" for head in self.final_heads)
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OptimConfig:
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: tuple[int, ...] | None = None
    lora_layer_ranks: dict[int, int] = field(default_factory=dict)
    lora_adapt_key: bool = False
    lora_layer_lrs: dict[int, float] = field(default_factory=dict)
    lr: float = 5e-4
    weight_decay: float = 0.0
    steps: int = 10
    grad_clip: float = 1.0
    backward_mode: Literal["direct", "replay"] = "direct"
    reuse_teacher_first_step: bool = False
    reset_lora_between_images: bool = True
    source_lora_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    eval_config: str
    split: Literal["train", "val", "test"] = "test"
    max_samples: int = 0
    num_workers: int = 0
    seed: int = 3407
    resolution: int = 1008
    mask_chunk: int = 64
    query_batch_size: int = 4
    full_query_batch_size: int = 1
    output_json: str = "work_dirs_tta/lora_tta_clean.json"
    device: str = "cuda"
    baseline_only: bool = False
    visualization_dir: str | None = None
    visualization_max_side: int = 1024
    visualization_min_delta_miou: float | None = None
    visualization_delta_in_filename: bool = False

    def __post_init__(self) -> None:
        if int(self.visualization_max_side) <= 0:
            raise ValueError("visualization_max_side must be positive")
        if self.visualization_min_delta_miou is not None and not math.isfinite(
            float(self.visualization_min_delta_miou)
        ):
            raise ValueError("visualization_min_delta_miou must be finite")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OracleDiagnosticConfig:
    enabled: bool = False
    lr_multipliers: tuple[float, ...] = (1.0,)
    target_presence_powers: tuple[float, ...] = (1.0,)
    reference_lr_multiplier: float = 1.0
    reference_target_presence_power: float = 1.0

    def __post_init__(self) -> None:
        lr_multipliers = tuple(
            sorted(set(float(value) for value in self.lr_multipliers))
        )
        target_powers = tuple(
            sorted(set(float(value) for value in self.target_presence_powers))
        )
        if not lr_multipliers or any(
            not math.isfinite(value) or value <= 0.0
            for value in lr_multipliers
        ):
            raise ValueError("oracle learning-rate multipliers must be positive")
        if not target_powers or any(
            not math.isfinite(value) or value < 0.0
            for value in target_powers
        ):
            raise ValueError("oracle target presence powers must be non-negative")
        reference = (
            float(self.reference_lr_multiplier),
            float(self.reference_target_presence_power),
        )
        if not math.isfinite(reference[0]) or reference[0] <= 0.0:
            raise ValueError("oracle reference learning-rate multiplier must be positive")
        if not math.isfinite(reference[1]) or reference[1] < 0.0:
            raise ValueError("oracle reference target presence power must be non-negative")
        if self.enabled and (
            reference[0] not in lr_multipliers
            or reference[1] not in target_powers
        ):
            raise ValueError("enabled oracle grid must contain its reference pair")
        object.__setattr__(self, "lr_multipliers", lr_multipliers)
        object.__setattr__(self, "target_presence_powers", target_powers)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProtectionConfig:
    enabled: bool = False
    action: Literal[
        "baseline_fallback",
        "unsupported_pixel_fallback",
    ] = "baseline_fallback"
    survival_enabled: bool = False
    survival_metric: Literal["area", "retention"] = "area"
    survival_intervention: Literal[
        "mask_loss_scale",
        "old_probability_reconstruction",
    ] = "mask_loss_scale"
    survival_threshold: float = 0.9
    survival_mask_bce_scale: float = 0.5
    survival_min_baseline_pixels: int = 64

    def __post_init__(self) -> None:
        if self.action not in {
            "baseline_fallback",
            "unsupported_pixel_fallback",
        }:
            raise ValueError(
                "protection action must be 'baseline_fallback' or "
                "'unsupported_pixel_fallback'"
            )
        if not 0.0 < float(self.survival_threshold) <= 1.0:
            raise ValueError("survival threshold must be in (0, 1]")
        if self.survival_metric not in {"area", "retention"}:
            raise ValueError("survival metric must be 'area' or 'retention'")
        if self.survival_intervention not in {
            "mask_loss_scale",
            "old_probability_reconstruction",
        }:
            raise ValueError(
                "survival intervention must be 'mask_loss_scale' or "
                "'old_probability_reconstruction'"
            )
        if not 0.0 <= float(self.survival_mask_bce_scale) <= 1.0:
            raise ValueError("survival mask BCE scale must be in [0, 1]")
        if int(self.survival_min_baseline_pixels) <= 0:
            raise ValueError("survival minimum baseline pixels must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TTAConfig:
    runtime: RuntimeConfig
    mining: MiningConfig = field(default_factory=MiningConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    prompt: PromptViewConfig = field(default_factory=PromptViewConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    oracle: OracleDiagnosticConfig = field(default_factory=OracleDiagnosticConfig)
    protection: ProtectionConfig = field(default_factory=ProtectionConfig)

    def to_dict(self) -> dict:
        return {
            "runtime": self.runtime.to_dict(),
            "mining": self.mining.to_dict(),
            "loss": self.loss.to_dict(),
            "prompt": self.prompt.to_dict(),
            "optim": self.optim.to_dict(),
            "oracle": self.oracle.to_dict(),
            "protection": self.protection.to_dict(),
        }
