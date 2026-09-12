from __future__ import annotations

import argparse
import math
import os
import sys

from .boundary_causal_diagnostic import (
    DEFAULT_BOUNDARY_RADII,
    normalize_boundary_radii,
)
from .config import (
    FinalHeadConfig,
    LossConfig,
    MiningConfig,
    OracleDiagnosticConfig,
    OptimConfig,
    ProtectionConfig,
    PromptViewConfig,
    RuntimeConfig,
    TTAConfig,
)
from .engine import CleanTTAEngine


def parse_lora_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "LoRA layers must be comma-separated integers"
        ) from error
    if not layers:
        raise argparse.ArgumentTypeError("at least one LoRA layer is required")
    if any(layer < 0 for layer in layers):
        raise argparse.ArgumentTypeError("LoRA layer indices must be non-negative")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError("LoRA layer indices must not contain duplicates")
    return layers


def parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(item.strip())
            for item in str(value).split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated numbers"
        ) from error
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError(
            "expected finite comma-separated numbers"
        )
    return values


def parse_lora_layer_lr(value: str) -> tuple[int, float]:
    layer_text, separator, lr_text = str(value).partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("LoRA layer learning rate must use LAYER:LR")
    try:
        layer = int(layer_text.strip())
        lr = float(lr_text.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "LoRA layer learning rate must use integer LAYER and numeric LR"
        ) from error
    if layer < 0:
        raise argparse.ArgumentTypeError("LoRA layer index must be non-negative")
    if not math.isfinite(lr) or lr <= 0:
        raise argparse.ArgumentTypeError("LoRA layer learning rate must be positive")
    return layer, lr


def parse_lora_layer_rank(value: str) -> tuple[int, int]:
    layer_text, separator, rank_text = str(value).partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("LoRA layer rank must use LAYER:RANK")
    try:
        layer = int(layer_text.strip())
        rank = int(rank_text.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "LoRA layer rank must use integer LAYER and integer RANK"
        ) from error
    if layer < 0:
        raise argparse.ArgumentTypeError("LoRA layer index must be non-negative")
    if rank <= 0:
        raise argparse.ArgumentTypeError("LoRA layer rank must be positive")
    return layer, rank


def parse_final_head(value: str) -> FinalHeadConfig:
    parts = str(value).split(":", maxsplit=2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "final head must use "
            "NAME[@FUSION_MODE]:CLASS_ID[,CLASS_ID...][:CLASSNAME_PATH] format"
        )
    name_and_mode, class_ids_text = parts[:2]
    classname_path = parts[2].strip() if len(parts) == 3 else None
    name, mode_separator, fusion_mode = name_and_mode.partition("@")
    if not name.strip():
        raise argparse.ArgumentTypeError("final head name must not be empty")
    if mode_separator and not fusion_mode.strip():
        raise argparse.ArgumentTypeError("final head fusion mode must not be empty")
    if not class_ids_text.strip():
        class_ids = ()
    else:
        try:
            class_ids = tuple(
                int(class_id.strip())
                for class_id in class_ids_text.split(",")
                if class_id.strip()
            )
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "final head class ids must be comma-separated integers"
            ) from error
        if not class_ids:
            raise argparse.ArgumentTypeError(
                "final head class ids must be comma-separated integers"
            )
    try:
        return FinalHeadConfig(
            name=name,
            canonical_class_ids=class_ids,
            classname_path=classname_path,
            fusion_mode=fusion_mode if mode_separator else "synonym_max",
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean classwise LoRA TTA entrypoint.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--mask-chunk", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--full-query-batch-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run source-only inference without mining or adaptation.",
    )
    parser.add_argument("--visualization-dir", default=None)
    parser.add_argument("--visualization-max-side", type=int, default=1024)
    parser.add_argument("--visualization-min-delta-miou", type=float, default=None)
    parser.add_argument(
        "--visualization-delta-in-filename",
        action="store_true",
    )
    parser.add_argument(
        "--diagnose-synonyms",
        action="store_true",
        help=(
            "Collect per-alias rescue, harm, score drift, and leave-one-out "
            "statistics before and after TTA."
        ),
    )
    parser.add_argument(
        "--diagnose-prompt-routing",
        action="store_true",
        help="Evaluate prompt-symmetric transfer routing heads.",
    )
    parser.add_argument(
        "--diagnose-cross-time-only",
        action="store_true",
        help=(
            "Collect lightweight cross-time causal and synonym diagnostics "
            "without prompt-routing heads."
        ),
    )
    parser.add_argument(
        "--diagnose-synonym-tta-overlap",
        action="store_true",
        help=(
            "Localize synonym shielding and alias transfer by GT "
            "boundary region."
        ),
    )
    parser.add_argument(
        "--boundary-widths",
        nargs="+",
        type=int,
        default=list(DEFAULT_BOUNDARY_RADII),
        metavar="PIXELS",
    )
    parser.add_argument(
        "--diagnose-teacher-prompts",
        action="store_true",
        help=(
            "Record observation-only TopK statistics for candidate TTA "
            "teacher prompts."
        ),
    )
    parser.add_argument(
        "--diagnose-teacher-prompt-gradients",
        action="store_true",
        help=(
            "Record label-free candidate-teacher gradient alignment "
            "diagnostics without applying candidate updates."
        ),
    )
    parser.add_argument(
        "--diagnose-selected-flips",
        action="store_true",
        help=(
            "Compare frozen Teacher TopK pixels with post-TTA mining and "
            "final-head predictions."
        ),
    )
    parser.add_argument(
        "--diagnose-selection-conflicts",
        action="store_true",
        help=(
            "Measure retained TopK score ordering and cross-class "
            "positive/negative overlap without changing adaptation."
        ),
    )
    parser.add_argument(
        "--teacher-diagnostic-class",
        action="append",
        type=int,
        default=None,
        metavar="CLASS_ID",
        help=(
            "Restrict teacher-prompt diagnostics to a semantic class; may be "
            "repeated. Defaults to every class with multiple prompts."
        ),
    )
    parser.add_argument(
        "--teacher-prompt-trial-steps",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "Try each diagnostic teacher prompt from the same LoRA state "
            "for this many update steps and record label-free transfer "
            "metrics. Zero disables candidate trials."
        ),
    )
    parser.add_argument(
        "--diagnose-image-adaptive-oracle",
        action="store_true",
        help=(
            "Evaluate isolated per-image learning-rate and soft-target "
            "presence-power candidates."
        ),
    )
    parser.add_argument(
        "--oracle-lr-multipliers",
        type=parse_float_tuple,
        default=(1.0,),
        metavar="VALUES",
    )
    parser.add_argument(
        "--oracle-target-presence-powers",
        type=parse_float_tuple,
        default=(1.0,),
        metavar="VALUES",
    )
    parser.add_argument(
        "--oracle-reference-lr-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--oracle-reference-target-presence-power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--protect-unsupported-class-birth",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--protection-action",
        choices=[
            "baseline_fallback",
            "unsupported_pixel_fallback",
        ],
        default="baseline_fallback",
    )
    parser.add_argument(
        "--protect-class-survival",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--class-survival-metric",
        choices=["area", "retention"],
        default="area",
    )
    parser.add_argument(
        "--class-survival-intervention",
        choices=[
            "mask_loss_scale",
            "old_probability_reconstruction",
        ],
        default="mask_loss_scale",
    )
    parser.add_argument(
        "--class-survival-threshold",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--class-survival-mask-bce-scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--class-survival-min-baseline-pixels",
        type=int,
        default=64,
    )

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-layers", type=parse_lora_layers, default=None)
    parser.add_argument(
        "--lora-adapt-key",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--lora-layer-rank",
        action="append",
        type=parse_lora_layer_rank,
        default=None,
        metavar="LAYER:RANK",
    )
    parser.add_argument(
        "--lora-layer-lr",
        action="append",
        type=parse_lora_layer_lr,
        default=None,
        metavar="LAYER:LR",
    )
    parser.add_argument("--source-lora-path", default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tta-steps", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--backward-mode",
        choices=["direct", "replay"],
        default="direct",
        help="Direct is faster but retains all loss-query graphs until backward.",
    )
    parser.add_argument(
        "--reuse-teacher-first-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse the differentiable canonical teacher output for direct "
            "update step 1."
        ),
    )

    parser.add_argument(
        "--sampling-mode",
        choices=["classwise_quota", "global_topk"],
        default="classwise_quota",
    )
    parser.add_argument("--tau-pos", type=float, default=0.5)
    parser.add_argument("--tau-neg", type=float, default=0.2)
    parser.add_argument("--prob-thd", type=float, default=0.3)
    parser.add_argument(
        "--selected-point-min-confidence",
        type=float,
        default=None,
        help="Final confidence floor applied after positive Top-K selection.",
    )
    parser.add_argument("--presence-gate-power", type=float, default=1.0)
    parser.add_argument(
        "--inference-presence-gate-power",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--score-fusion-mode",
        choices=["legacy_max", "proc_pgrf"],
        default="legacy_max",
        help="Prompt-level semantic/instance score fusion used for mining.",
    )
    parser.add_argument(
        "--inference-score-fusion-mode",
        choices=["legacy_max", "proc_pgrf"],
        default=None,
        help="Final inference fusion; defaults to --score-fusion-mode.",
    )
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--kmax", type=int, default=2048)
    parser.add_argument("--n-min", type=int, default=64)
    parser.add_argument("--min-selected-classes", type=int, default=1)
    parser.add_argument("--include-bg", action="store_true")
    parser.add_argument("--bg-idx", type=int, default=0)
    parser.add_argument(
        "--pixel-weight-mode",
        default="score",
        choices=["none", "score"],
    )
    parser.add_argument("--class-weight-mode", default="mean_margin", choices=["none", "mean_margin", "mean_reliability"])
    parser.add_argument("--class-weight-min", type=float, default=0.3)
    parser.add_argument(
        "--positive-argmax-only",
        action="store_true",
        help=(
            "After classwise TopK and overlap removal, keep a positive only "
            "when its class is the pixelwise class-score argmax."
        ),
    )
    parser.add_argument(
        "--competition-reliability-mode",
        choices=["none", "score_ratio"],
        default="none",
        help=(
            "Use cross-class score competition as a continuous pixel, class, "
            "and positive-presence reliability weight."
        ),
    )
    parser.add_argument("--classwise-component-gate", action="store_true")
    parser.add_argument("--classwise-component-fallback", default="none", choices=["none", "topk"])
    parser.add_argument("--classwise-component-select-mode", default="topk", choices=["topk", "whole"])
    parser.add_argument("--classwise-component-mask-thd", type=float, default=0.5)
    parser.add_argument("--classwise-component-min-area", type=int, default=4096)
    parser.add_argument("--classwise-component-mean-thd", type=float, default=0.9)

    parser.add_argument("--bce-positive-target-mode", default="hard", choices=["hard", "PTST"])
    parser.add_argument(
        "--bce-soft-target-presence-gate-power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--bce-soft-target-score-fusion-mode",
        choices=["legacy_max", "proc_pgrf"],
        default=None,
        help="Soft-target fusion; defaults to --score-fusion-mode.",
    )
    parser.add_argument(
        "--sparse-selected-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample native logits only at selected positive/negative pixels.",
    )
    parser.add_argument(
        "--student-score-mode",
        choices=[
            "semantic_raw",
            "filtered_raw_fusion",
            "filtered_raw_probability_reconstruction",
        ],
        default="semantic_raw",
    )
    parser.add_argument("--selected-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--contraction-gate-mode",
        choices=["none", "decision_boundary"],
        default="none",
    )
    parser.add_argument("--presence-loss-weight", type=float, default=0.0)
    parser.add_argument("--bce-positive-target-min", type=float, default=0.0)
    parser.add_argument("--bce-positive-target-max", type=float, default=0.95)
    parser.add_argument("--bce-low-score-neg-weight", type=float, default=0.0)
    parser.add_argument("--bce-low-score-neg-thd", type=float, default=0.5)
    parser.add_argument("--bce-low-score-neg-rho", type=float, default=0.25)
    parser.add_argument("--bce-low-score-neg-kmax", type=int, default=512)
    parser.add_argument("--bce-low-score-neg-n-min", type=int, default=1)
    parser.add_argument("--canonical-classname-path", default=None)
    parser.add_argument("--mining-classname-path", default=None)
    parser.add_argument("--loss-classname-path", default=None)
    parser.add_argument(
        "--adaptation-mode",
        choices=["classwise", "independent_queries"],
        default="classwise",
    )
    parser.add_argument(
        "--mining-prompt-view",
        choices=["synonym", "canonical"],
        default="synonym",
    )
    parser.add_argument(
        "--loss-prompt-view",
        choices=["synonym", "canonical"],
        default="synonym",
    )
    parser.add_argument(
        "--auto-canonical-final-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When no explicit final heads are given, emit synonym_full plus "
            "a canonical_only head covering every class."
        ),
    )
    parser.add_argument(
        "--final-head",
        action="append",
        type=parse_final_head,
        default=None,
        metavar="NAME[@FUSION_MODE]:CLASS_IDS[:CLASSNAME_PATH]",
        help=(
            "Inference-only head; topk_reliability gates aliases from canonical "
            "TopK anchors. Empty CLASS_IDS applies no hard canonical overrides. "
            "An optional classname path selects a separate one-prompt-per-class "
            "view for that head."
        ),
    )
    runtime_args_file = os.environ.get("LORA_TTA_ARGS_FILE")
    runtime_argv = (
        [*sys.argv[1:], f"@{runtime_args_file}"]
        if runtime_args_file
        else None
    )
    return parser.parse_args(runtime_argv)


def config_from_args(args: argparse.Namespace) -> TTAConfig:
    return TTAConfig(
        runtime=RuntimeConfig(
            eval_config=args.eval_config,
            split=args.split,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            resolution=args.resolution,
            mask_chunk=args.mask_chunk,
            query_batch_size=getattr(args, "query_batch_size", 4),
            full_query_batch_size=getattr(args, "full_query_batch_size", 1),
            output_json=args.output_json,
            device=args.device,
            baseline_only=getattr(args, "baseline_only", False),
            visualization_dir=getattr(args, "visualization_dir", None),
            visualization_max_side=getattr(
                args,
                "visualization_max_side",
                1024,
            ),
            visualization_min_delta_miou=getattr(
                args,
                "visualization_min_delta_miou",
                None,
            ),
            visualization_delta_in_filename=getattr(
                args,
                "visualization_delta_in_filename",
                False,
            ),
        ),
        mining=MiningConfig(
            sampling_mode=getattr(args, "sampling_mode", "classwise_quota"),
            tau_pos=args.tau_pos,
            tau_neg=args.tau_neg,
            prob_thd=args.prob_thd,
            selected_point_min_confidence=getattr(
                args,
                "selected_point_min_confidence",
                None,
            ),
            presence_gate_power=args.presence_gate_power,
            inference_presence_gate_power=getattr(
                args,
                "inference_presence_gate_power",
                None,
            ),
            score_fusion_mode=getattr(
                args,
                "score_fusion_mode",
                "legacy_max",
            ),
            inference_score_fusion_mode=getattr(
                args,
                "inference_score_fusion_mode",
                None,
            ),
            rho=args.rho,
            kmax=args.kmax,
            n_min=args.n_min,
            min_selected_classes=getattr(args, "min_selected_classes", 1),
            include_bg=args.include_bg,
            bg_idx=args.bg_idx,
            pixel_weight_mode=getattr(args, "pixel_weight_mode", "score"),
            class_weight_mode=args.class_weight_mode,
            class_weight_min=args.class_weight_min,
            positive_argmax_only=getattr(
                args,
                "positive_argmax_only",
                False,
            ),
            competition_reliability_mode=getattr(
                args,
                "competition_reliability_mode",
                "none",
            ),
            component_gate=args.classwise_component_gate,
            component_fallback=args.classwise_component_fallback,
            component_select_mode=args.classwise_component_select_mode,
            component_mask_thd=args.classwise_component_mask_thd,
            component_min_area=args.classwise_component_min_area,
            component_mean_thd=args.classwise_component_mean_thd,
        ),
        loss=LossConfig(
            student_score_mode=getattr(
                args,
                "student_score_mode",
                "semantic_raw",
            ),
            selected_loss_weight=getattr(args, "selected_loss_weight", 1.0),
            contraction_gate_mode=getattr(
                args,
                "contraction_gate_mode",
                "none",
            ),
            presence_loss_weight=args.presence_loss_weight,
            positive_target_mode=args.bce_positive_target_mode,
            soft_target_presence_gate_power=getattr(
                args,
                "bce_soft_target_presence_gate_power",
                1.0,
            ),
            soft_target_score_fusion_mode=getattr(
                args,
                "bce_soft_target_score_fusion_mode",
                None,
            ),
            sparse_selected_logits=getattr(
                args,
                "sparse_selected_logits",
                False,
            ),
            positive_target_min=args.bce_positive_target_min,
            positive_target_max=args.bce_positive_target_max,
            low_score_neg_weight=args.bce_low_score_neg_weight,
            low_score_neg_thd=args.bce_low_score_neg_thd,
            low_score_neg_rho=args.bce_low_score_neg_rho,
            low_score_neg_kmax=args.bce_low_score_neg_kmax,
            low_score_neg_n_min=args.bce_low_score_neg_n_min,
        ),
        prompt=PromptViewConfig(
            canonical_classname_path=getattr(args, "canonical_classname_path", None),
            mining_classname_path=getattr(args, "mining_classname_path", None),
            loss_classname_path=getattr(args, "loss_classname_path", None),
            adaptation_mode=getattr(args, "adaptation_mode", "classwise"),
            mining_view=getattr(args, "mining_prompt_view", "synonym"),
            loss_view=getattr(args, "loss_prompt_view", "synonym"),
            auto_canonical_final_head=getattr(
                args,
                "auto_canonical_final_head",
                False,
            ),
            final_heads=tuple(getattr(args, "final_head", ()) or ()),
        ),
        optim=OptimConfig(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layers=getattr(args, "lora_layers", None),
            lora_layer_ranks=dict(getattr(args, "lora_layer_rank", ()) or ()),
            lora_adapt_key=getattr(args, "lora_adapt_key", False),
            lora_layer_lrs=dict(getattr(args, "lora_layer_lr", ()) or ()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps=args.tta_steps,
            grad_clip=args.grad_clip,
            backward_mode=getattr(args, "backward_mode", "direct"),
            reuse_teacher_first_step=getattr(
                args,
                "reuse_teacher_first_step",
                False,
            ),
            source_lora_path=args.source_lora_path,
        ),
        oracle=OracleDiagnosticConfig(
            enabled=bool(
                getattr(args, "diagnose_image_adaptive_oracle", False)
            ),
            lr_multipliers=tuple(
                getattr(args, "oracle_lr_multipliers", (1.0,))
            ),
            target_presence_powers=tuple(
                getattr(
                    args,
                    "oracle_target_presence_powers",
                    (1.0,),
                )
            ),
            reference_lr_multiplier=float(
                getattr(args, "oracle_reference_lr_multiplier", 1.0)
            ),
            reference_target_presence_power=float(
                getattr(
                    args,
                    "oracle_reference_target_presence_power",
                    1.0,
                )
            ),
        ),
        protection=ProtectionConfig(
            enabled=bool(
                getattr(
                    args,
                    "protect_unsupported_class_birth",
                    False,
                )
            ),
            action=str(
                getattr(
                    args,
                    "protection_action",
                    "baseline_fallback",
                )
            ),
            survival_enabled=bool(
                getattr(args, "protect_class_survival", False)
            ),
            survival_metric=str(
                getattr(args, "class_survival_metric", "area")
            ),
            survival_intervention=str(
                getattr(
                    args,
                    "class_survival_intervention",
                    "mask_loss_scale",
                )
            ),
            survival_threshold=float(
                getattr(args, "class_survival_threshold", 0.9)
            ),
            survival_mask_bce_scale=float(
                getattr(args, "class_survival_mask_bce_scale", 0.5)
            ),
            survival_min_baseline_pixels=int(
                getattr(
                    args,
                    "class_survival_min_baseline_pixels",
                    64,
                )
            ),
        ),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    diagnose_prompt_routing = getattr(
        args,
        "diagnose_prompt_routing",
        False,
    )
    diagnose_overlap = bool(
        getattr(args, "diagnose_synonym_tta_overlap", False)
    )
    boundary_widths = normalize_boundary_radii(
        getattr(args, "boundary_widths", DEFAULT_BOUNDARY_RADII)
    )
    diagnose_cross_time_only = bool(
        getattr(args, "diagnose_cross_time_only", False)
        or diagnose_overlap
    )
    diagnose_teacher_prompts = getattr(
        args,
        "diagnose_teacher_prompts",
        False,
    )
    diagnose_teacher_prompt_gradients = getattr(
        args,
        "diagnose_teacher_prompt_gradients",
        False,
    )
    diagnose_selected_flips = bool(
        getattr(args, "diagnose_selected_flips", False)
    )
    diagnose_selection_conflicts = bool(
        getattr(args, "diagnose_selection_conflicts", False)
    )
    CleanTTAEngine(
        config,
        diagnose_synonyms=(
            getattr(args, "diagnose_synonyms", False)
            or diagnose_prompt_routing
            or diagnose_cross_time_only
        ),
        diagnose_prompt_routing=diagnose_prompt_routing,
        diagnose_cross_time_only=diagnose_cross_time_only,
        diagnose_synonym_tta_overlap=diagnose_overlap,
        boundary_widths=boundary_widths,
        diagnose_teacher_prompts=diagnose_teacher_prompts,
        diagnose_teacher_prompt_gradients=(
            diagnose_teacher_prompt_gradients
        ),
        teacher_diagnostic_classes=tuple(
            getattr(args, "teacher_diagnostic_class", None) or ()
        ),
        teacher_prompt_trial_steps=int(
            getattr(args, "teacher_prompt_trial_steps", 0)
        ),
        diagnose_selected_flips=diagnose_selected_flips,
        diagnose_selection_conflicts=diagnose_selection_conflicts,
    ).run()


if __name__ == "__main__":
    main()
