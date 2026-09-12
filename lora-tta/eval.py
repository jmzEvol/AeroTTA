from __future__ import annotations

import argparse
import ast
import json
import os
import pprint
import runpy
import shlex
import subprocess
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence


LORA_TTA_DIR = Path(__file__).resolve().parent
ROOT = LORA_TTA_DIR.parent

if str(LORA_TTA_DIR) not in sys.path:
    sys.path.insert(0, str(LORA_TTA_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_tta.config import LossConfig, MiningConfig, OptimConfig, OracleDiagnosticConfig, ProtectionConfig, PromptViewConfig, RuntimeConfig, TTAConfig
from lora_tta.boundary_causal_diagnostic import (
    DEFAULT_BOUNDARY_RADII,
    normalize_boundary_radii,
)


CONFIG_SECTION_TYPES = {
    "runtime": RuntimeConfig,
    "optim": OptimConfig,
    "mining": MiningConfig,
    "loss": LossConfig,
    "prompt": PromptViewConfig,
    "oracle": OracleDiagnosticConfig,
    "protection": ProtectionConfig,
}

LAUNCH_OVERRIDE_KEYS = {
    "dataset",
    "env",
    "entrypoint_script",
    "gpus",
    "log_file",
    "master_port",
    "nproc_per_node",
    "output_json",
    "python",
    "run_id",
    "session",
    "tag",
    "timeout_seconds",
    "tmux",
    "diagnose_teacher_prompts",
    "diagnose_teacher_prompt_gradients",
    "teacher_diagnostic_classes",
    "teacher_prompt_trial_steps",
    "work_dir",
}

DEFAULT_ACCELERATION_PROFILE = "fast"

GENERIC_TTA_DEFAULTS: dict[str, dict[str, Any]] = {
    "optim": {
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_layers": (3, 4, 5),
        "steps": 3,
    },
    "loss": {
        "presence_loss_weight": 0.1,
        "positive_target_mode": "PTST",
        "positive_target_max": 0.95,
    },
    "mining": {
        "inference_presence_gate_power": 1.0,
    },
    "prompt": {
        "mining_view": "canonical",
        "loss_view": "canonical",
        "auto_canonical_final_head": False,
    },
}

ACCELERATION_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "runtime": {
            "mask_chunk": 1,
            "query_batch_size": 8,
            "full_query_batch_size": 8,
        },
        "optim": {
            "backward_mode": "direct",
            "reuse_teacher_first_step": True,
        },
    },
    "compat": {
        "runtime": {
            "mask_chunk": 1,
            "query_batch_size": 1,
            "full_query_batch_size": 1,
        },
        "optim": {
            "backward_mode": "replay",
            "reuse_teacher_first_step": False,
        },
    },
}


class RunPlan(NamedTuple):
    config_path: Path
    profile: str
    work_dir: Path
    tmux: bool
    session: str
    output_json: str
    log_file: str | None
    config_file: str | None
    config_snapshot: Mapping[str, Any]
    shell_command: str


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping, got {type(value).__name__}")


def _load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    namespace = runpy.run_path(str(config_path))
    return {key: value for key, value in namespace.items() if not key.startswith("__")}


def _section(namespace: Mapping[str, Any], name: str) -> dict[str, Any]:
    nested = _as_dict(namespace.get("tta"))
    if name in nested:
        return _as_dict(nested[name])
    return _as_dict(namespace.get(name))


def _resolve_acceleration_profile(
    profile_name: str,
) -> dict[str, dict[str, Any]]:
    try:
        profile = ACCELERATION_PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(ACCELERATION_PROFILES))
        raise ValueError(
            f"unknown acceleration profile {profile_name!r}; choose from: {choices}"
        ) from exc

    resolved = {name: {} for name in CONFIG_SECTION_TYPES}
    for section_name, raw_values in _as_dict(profile).items():
        if section_name not in CONFIG_SECTION_TYPES:
            raise ValueError(
                f"unknown profile section {section_name!r} in {profile_name!r}"
            )
        values = _as_dict(raw_values)
        valid_keys = {
            field.name for field in fields(CONFIG_SECTION_TYPES[section_name])
        }
        unknown_keys = sorted(set(values) - valid_keys)
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ValueError(
                f"unknown profile keys for {section_name}: {joined}"
            )
        resolved[section_name].update(values)
    return resolved


def _parse_override_value(raw_value: str) -> Any:
    value = raw_value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return value


def _apply_overrides(
    sections: Mapping[str, dict[str, Any]],
    overrides: Sequence[str],
) -> None:
    valid_sections = set(CONFIG_SECTION_TYPES) | {"launch"}
    for override in overrides:
        path, separator, raw_value = str(override).partition("=")
        if not separator:
            raise ValueError(
                f"override must use SECTION.KEY=VALUE, got {override!r}"
            )
        section_name, path_separator, key = path.strip().partition(".")
        if not path_separator or not section_name or not key:
            raise ValueError(
                f"override must use SECTION.KEY=VALUE, got {override!r}"
            )
        if section_name not in valid_sections:
            raise ValueError(f"unknown override section {section_name!r}")

        if section_name == "launch":
            valid_keys = LAUNCH_OVERRIDE_KEYS
        else:
            valid_keys = {
                field.name for field in fields(CONFIG_SECTION_TYPES[section_name])
            }
        if key not in valid_keys:
            raise ValueError(f"unknown override key '{section_name}.{key}'")
        sections[section_name][key] = _parse_override_value(raw_value)


def _format_template(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    return value


def _default_dataset(runtime: Mapping[str, Any]) -> str:
    eval_config = str(runtime.get("eval_config", "tta"))
    stem = Path(eval_config).stem
    return stem[4:] if stem.startswith("cfg_") else stem


def _resolve_file_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def _load_eval_model_settings(
    eval_config: str | os.PathLike[str],
    *,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Resolve the model section needed as defaults by the TTA launcher."""

    config_path = _resolve_file_path(eval_config).resolve()
    if not config_path.is_file():
        return {}
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"cyclic eval config inheritance at {config_path}")
    seen.add(config_path)
    try:
        namespace = _load_config(config_path)
        resolved: dict[str, Any] = {}
        base_configs = namespace.get("_base_", ())
        if isinstance(base_configs, (str, os.PathLike)):
            base_configs = (base_configs,)
        for base_config in base_configs:
            base_path = Path(base_config)
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            resolved.update(
                _load_eval_model_settings(base_path, _seen=seen)
            )
        resolved.update(_as_dict(namespace.get("model")))
        return resolved
    finally:
        seen.remove(config_path)


def _append_arg(args: list[str], flag: str, value: Any) -> None:
    args.extend([flag, str(value)])


def _tta_cli_args(config: TTAConfig) -> list[str]:
    args: list[str] = []
    runtime = config.runtime
    optim = config.optim
    mining = config.mining
    loss = config.loss
    oracle = config.oracle
    protection = config.protection

    _append_arg(args, "--eval-config", runtime.eval_config)
    _append_arg(args, "--split", runtime.split)
    _append_arg(args, "--max-samples", runtime.max_samples)
    _append_arg(args, "--num-workers", runtime.num_workers)
    _append_arg(args, "--seed", runtime.seed)
    _append_arg(args, "--resolution", runtime.resolution)
    _append_arg(args, "--mask-chunk", runtime.mask_chunk)
    _append_arg(args, "--query-batch-size", runtime.query_batch_size)
    _append_arg(args, "--full-query-batch-size", runtime.full_query_batch_size)
    _append_arg(args, "--output-json", runtime.output_json)
    _append_arg(args, "--device", runtime.device)
    if runtime.baseline_only:
        args.append("--baseline-only")
    if runtime.visualization_dir is not None:
        _append_arg(
            args,
            "--visualization-dir",
            runtime.visualization_dir,
        )
        _append_arg(
            args,
            "--visualization-max-side",
            runtime.visualization_max_side,
        )
        if runtime.visualization_min_delta_miou is not None:
            _append_arg(
                args,
                "--visualization-min-delta-miou",
                runtime.visualization_min_delta_miou,
            )
        if runtime.visualization_delta_in_filename:
            args.append("--visualization-delta-in-filename")

    _append_arg(args, "--lora-rank", optim.lora_rank)
    _append_arg(args, "--lora-alpha", optim.lora_alpha)
    if optim.lora_layers is not None:
        _append_arg(args, "--lora-layers", ",".join(str(layer) for layer in optim.lora_layers))
    if optim.lora_adapt_key:
        args.append("--lora-adapt-key")
    for layer, layer_rank in sorted(optim.lora_layer_ranks.items()):
        _append_arg(args, "--lora-layer-rank", f"{layer}:{layer_rank}")
    for layer, layer_lr in sorted(optim.lora_layer_lrs.items()):
        _append_arg(args, "--lora-layer-lr", f"{layer}:{layer_lr}")
    if optim.source_lora_path:
        _append_arg(args, "--source-lora-path", optim.source_lora_path)
    _append_arg(args, "--lr", optim.lr)
    _append_arg(args, "--weight-decay", optim.weight_decay)
    _append_arg(args, "--tta-steps", optim.steps)
    _append_arg(args, "--grad-clip", optim.grad_clip)
    _append_arg(args, "--backward-mode", optim.backward_mode)
    if optim.backward_mode == "direct" and optim.reuse_teacher_first_step:
        args.append("--reuse-teacher-first-step")

    _append_arg(args, "--sampling-mode", mining.sampling_mode)
    _append_arg(args, "--tau-pos", mining.tau_pos)
    _append_arg(args, "--tau-neg", mining.tau_neg)
    _append_arg(args, "--prob-thd", mining.prob_thd)
    if mining.selected_point_min_confidence is not None:
        _append_arg(
            args,
            "--selected-point-min-confidence",
            mining.selected_point_min_confidence,
        )
    _append_arg(args, "--presence-gate-power", mining.presence_gate_power)
    _append_arg(args, "--score-fusion-mode", mining.score_fusion_mode)
    if mining.inference_presence_gate_power is not None:
        _append_arg(
            args,
            "--inference-presence-gate-power",
            mining.inference_presence_gate_power,
        )
    if mining.inference_score_fusion_mode is not None:
        _append_arg(
            args,
            "--inference-score-fusion-mode",
            mining.inference_score_fusion_mode,
        )
    _append_arg(args, "--rho", mining.rho)
    _append_arg(args, "--kmax", mining.kmax)
    _append_arg(args, "--n-min", mining.n_min)
    _append_arg(args, "--min-selected-classes", mining.min_selected_classes)
    if mining.include_bg:
        args.append("--include-bg")
    _append_arg(args, "--bg-idx", mining.bg_idx)
    _append_arg(args, "--pixel-weight-mode", mining.pixel_weight_mode)
    _append_arg(args, "--class-weight-mode", mining.class_weight_mode)
    _append_arg(args, "--class-weight-min", mining.class_weight_min)
    if mining.positive_argmax_only:
        args.append("--positive-argmax-only")
    _append_arg(
        args,
        "--competition-reliability-mode",
        mining.competition_reliability_mode,
    )
    if mining.component_gate:
        args.append("--classwise-component-gate")
    _append_arg(args, "--classwise-component-fallback", mining.component_fallback)
    _append_arg(args, "--classwise-component-select-mode", mining.component_select_mode)
    _append_arg(args, "--classwise-component-mask-thd", mining.component_mask_thd)
    _append_arg(args, "--classwise-component-min-area", mining.component_min_area)
    _append_arg(args, "--classwise-component-mean-thd", mining.component_mean_thd)

    _append_arg(args, "--bce-positive-target-mode", loss.positive_target_mode)
    _append_arg(
        args,
        "--bce-soft-target-presence-gate-power",
        loss.soft_target_presence_gate_power,
    )
    if loss.soft_target_score_fusion_mode is not None:
        _append_arg(
            args,
            "--bce-soft-target-score-fusion-mode",
            loss.soft_target_score_fusion_mode,
        )
    if loss.sparse_selected_logits:
        args.append("--sparse-selected-logits")
    _append_arg(args, "--student-score-mode", loss.student_score_mode)
    _append_arg(args, "--selected-loss-weight", loss.selected_loss_weight)
    _append_arg(
        args,
        "--contraction-gate-mode",
        loss.contraction_gate_mode,
    )
    _append_arg(args, "--presence-loss-weight", loss.presence_loss_weight)
    _append_arg(args, "--bce-positive-target-min", loss.positive_target_min)
    _append_arg(args, "--bce-positive-target-max", loss.positive_target_max)
    _append_arg(args, "--bce-low-score-neg-weight", loss.low_score_neg_weight)
    _append_arg(args, "--bce-low-score-neg-thd", loss.low_score_neg_thd)
    _append_arg(args, "--bce-low-score-neg-rho", loss.low_score_neg_rho)
    _append_arg(args, "--bce-low-score-neg-kmax", loss.low_score_neg_kmax)
    _append_arg(args, "--bce-low-score-neg-n-min", loss.low_score_neg_n_min)
    if config.prompt.canonical_classname_path:
        _append_arg(args, "--canonical-classname-path", config.prompt.canonical_classname_path)
    if config.prompt.mining_classname_path:
        _append_arg(args, "--mining-classname-path", config.prompt.mining_classname_path)
    if config.prompt.loss_classname_path:
        _append_arg(args, "--loss-classname-path", config.prompt.loss_classname_path)
    _append_arg(args, "--adaptation-mode", config.prompt.adaptation_mode)
    _append_arg(args, "--mining-prompt-view", config.prompt.mining_view)
    _append_arg(args, "--loss-prompt-view", config.prompt.loss_view)
    if (
        config.prompt.auto_canonical_final_head
        and not config.prompt.final_heads
    ):
        args.append("--auto-canonical-final-head")
    for final_head in config.prompt.final_heads:
        class_ids = ",".join(str(class_id) for class_id in final_head.canonical_class_ids)
        name = final_head.name
        if final_head.fusion_mode != "synonym_max":
            name = f"{name}@{final_head.fusion_mode}"
        serialized = f"{name}:{class_ids}"
        if final_head.classname_path is not None:
            serialized = f"{serialized}:{final_head.classname_path}"
        _append_arg(args, "--final-head", serialized)
    if oracle.enabled:
        args.append("--diagnose-image-adaptive-oracle")
        _append_arg(
            args,
            "--oracle-lr-multipliers",
            ",".join(f"{value:g}" for value in oracle.lr_multipliers),
        )
        _append_arg(
            args,
            "--oracle-target-presence-powers",
            ",".join(
                f"{value:g}" for value in oracle.target_presence_powers
            ),
        )
        _append_arg(
            args,
            "--oracle-reference-lr-multiplier",
            f"{oracle.reference_lr_multiplier:g}",
        )
        _append_arg(
            args,
            "--oracle-reference-target-presence-power",
            f"{oracle.reference_target_presence_power:g}",
        )
    if protection.enabled:
        args.append("--protect-unsupported-class-birth")
    _append_arg(args, "--protection-action", protection.action)
    if protection.survival_enabled:
        args.append("--protect-class-survival")
    _append_arg(
        args,
        "--class-survival-metric",
        protection.survival_metric,
    )
    _append_arg(
        args,
        "--class-survival-intervention",
        protection.survival_intervention,
    )
    _append_arg(
        args,
        "--class-survival-threshold",
        f"{protection.survival_threshold:g}",
    )
    _append_arg(
        args,
        "--class-survival-mask-bce-scale",
        f"{protection.survival_mask_bce_scale:g}",
    )
    _append_arg(
        args,
        "--class-survival-min-baseline-pixels",
        protection.survival_min_baseline_pixels,
    )
    return args


def _shell_env_prefix(env: Mapping[str, Any], gpus: str, timeout_seconds: int) -> str:
    merged: dict[str, Any] = dict(env)
    merged.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    merged["CUDA_VISIBLE_DEVICES"] = gpus
    merged["LORA_TTA_DIST_TIMEOUT_SECONDS"] = int(timeout_seconds)

    parts = [f"{key}={shlex.quote(str(value))}" for key, value in sorted(merged.items())]
    parts.append(f"PYTHONPATH={shlex.quote(str(LORA_TTA_DIR))}${{PYTHONPATH:+:${{PYTHONPATH}}}}")
    return " ".join(parts)


def build_run_plan(
    config_path: str | os.PathLike[str],
    *,
    profile: str = DEFAULT_ACCELERATION_PROFILE,
    overrides: Sequence[str] = (),
    run_id: str | None = None,
    tmux: bool | None = None,
    session: str | None = None,
    gpus: str | None = None,
    nproc_per_node: int | None = None,
    master_port: int | None = None,
    output_json: str | None = None,
    log_file: str | None = None,
    python: str | None = None,
    diagnose_synonyms: bool = False,
    diagnose_prompt_routing: bool = False,
    diagnose_cross_time_only: bool = False,
    diagnose_synonym_tta_overlap: bool = False,
    boundary_widths: Sequence[int] = DEFAULT_BOUNDARY_RADII,
    diagnose_teacher_prompts: bool | None = None,
    diagnose_teacher_prompt_gradients: bool | None = None,
    diagnose_selected_flips: bool = False,
    diagnose_selection_conflicts: bool = False,
    teacher_diagnostic_classes: Sequence[int] | None = None,
    teacher_prompt_trial_steps: int | None = None,
) -> RunPlan:
    config_path = Path(config_path).expanduser().resolve()
    namespace = _load_config(config_path)

    runtime_dict = _section(namespace, "runtime")
    optim_dict = _section(namespace, "optim")
    mining_dict = _section(namespace, "mining")
    loss_dict = _section(namespace, "loss")
    prompt_dict = _section(namespace, "prompt")
    oracle_dict = _section(namespace, "oracle")
    protection_dict = _section(namespace, "protection")
    launch_dict = _section(namespace, "launch")

    # Dataset-level inference thresholds are the source of truth unless a TTA
    # config intentionally overrides them. CLI --set overrides are applied
    # later and therefore retain the highest priority.
    if "prob_thd" not in mining_dict:
        eval_model = _load_eval_model_settings(runtime_dict["eval_config"])
        if "prob_thd" in eval_model:
            mining_dict["prob_thd"] = eval_model["prob_thd"]

    config_sections = {
        "runtime": runtime_dict,
        "optim": optim_dict,
        "mining": mining_dict,
        "loss": loss_dict,
        "prompt": prompt_dict,
        "oracle": oracle_dict,
        "protection": protection_dict,
    }
    profile_sections = _resolve_acceleration_profile(profile)
    for section_name, config_values in config_sections.items():
        merged = dict(GENERIC_TTA_DEFAULTS.get(section_name, {}))
        merged.update(profile_sections[section_name])
        merged.update(config_values)
        config_sections[section_name] = merged

    runtime_dict = config_sections["runtime"]
    optim_dict = config_sections["optim"]
    mining_dict = config_sections["mining"]
    loss_dict = config_sections["loss"]
    prompt_dict = config_sections["prompt"]
    oracle_dict = config_sections["oracle"]
    protection_dict = config_sections["protection"]

    sections = {
        "runtime": runtime_dict,
        "optim": optim_dict,
        "mining": mining_dict,
        "loss": loss_dict,
        "prompt": prompt_dict,
        "oracle": oracle_dict,
        "protection": protection_dict,
        "launch": launch_dict,
    }
    _apply_overrides(sections, overrides)

    run_id = run_id or str(launch_dict.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S"))
    dataset = str(launch_dict.get("dataset") or _default_dataset(runtime_dict))
    tag = str(launch_dict.get("tag") or "tta")
    resolved_nproc = int(nproc_per_node or launch_dict.get("nproc_per_node", 1))
    resolved_gpus = str(gpus or launch_dict.get("gpus", "0"))
    resolved_port = int(master_port or launch_dict.get("master_port", 29500))
    timeout_seconds = int(launch_dict.get("timeout_seconds", 43200))
    resolved_tmux = bool(launch_dict.get("tmux", False)) if tmux is None else bool(tmux)
    resolved_python = str(python or launch_dict.get("python") or sys.executable)
    resolved_boundary_widths = normalize_boundary_radii(boundary_widths)
    if diagnose_prompt_routing and diagnose_synonym_tta_overlap:
        raise ValueError(
            "prompt routing and synonym-TTA overlap diagnostics are "
            "mutually exclusive"
        )
    resolved_teacher_diagnostic = (
        bool(launch_dict.get("diagnose_teacher_prompts", False))
        if diagnose_teacher_prompts is None
        else bool(diagnose_teacher_prompts)
    )
    resolved_teacher_gradient_diagnostic = (
        bool(launch_dict.get("diagnose_teacher_prompt_gradients", False))
        if diagnose_teacher_prompt_gradients is None
        else bool(diagnose_teacher_prompt_gradients)
    )
    resolved_teacher_classes = tuple(
        int(value)
        for value in (
            launch_dict.get("teacher_diagnostic_classes", ())
            if teacher_diagnostic_classes is None
            else teacher_diagnostic_classes
        )
    )
    resolved_teacher_trial_steps = int(
        launch_dict.get("teacher_prompt_trial_steps", 0)
        if teacher_prompt_trial_steps is None
        else teacher_prompt_trial_steps
    )
    if resolved_teacher_trial_steps < 0:
        raise ValueError("teacher_prompt_trial_steps must be non-negative")
    if resolved_teacher_trial_steps > 0:
        optim_dict["reuse_teacher_first_step"] = False
    work_dir = _resolve_file_path(str(launch_dict.get("work_dir", ROOT)))

    context = dict(
        dataset=dataset,
        config_name=config_path.stem,
        tag=tag,
        run_id=run_id,
        nproc=resolved_nproc,
        nproc_per_node=resolved_nproc,
        gpus=resolved_gpus,
        master_port=resolved_port,
    )

    configured_output = (
        launch_dict.get("output_json")
        or runtime_dict.get("output_json")
    )
    configured_log = launch_dict.get("log_file")
    uses_default_layout = (
        output_json is None
        and configured_output is None
        and log_file is None
        and configured_log is None
    )
    run_dir_template = "work_dirs_tta/{config_name}/{run_id}"
    resolved_run_dir = str(_format_template(run_dir_template, context))
    output_template = (
        output_json
        or configured_output
        or f"{resolved_run_dir}/{run_id}.json"
    )
    resolved_output = str(_format_template(output_template, context))
    runtime_dict["output_json"] = resolved_output
    visualization_template = runtime_dict.get("visualization_dir")
    if visualization_template is not None:
        runtime_dict["visualization_dir"] = str(
            _format_template(visualization_template, context)
        )

    log_template = (
        log_file
        or configured_log
        or f"{resolved_run_dir}/{run_id}.log"
    )
    resolved_log = str(_format_template(log_template, context)) if log_template else None
    resolved_config_file = (
        f"{resolved_run_dir}/config.py" if uses_default_layout else None
    )
    resolved_session = str(
        session
        or _format_template(
            launch_dict.get("session", "lora_tta_{dataset}_{tag}_{run_id}"),
            context,
        )
    )

    tta_config = TTAConfig(
        runtime=RuntimeConfig(**runtime_dict),
        mining=MiningConfig(**mining_dict),
        loss=LossConfig(**loss_dict),
        prompt=PromptViewConfig(**prompt_dict),
        optim=OptimConfig(**optim_dict),
        oracle=OracleDiagnosticConfig(**oracle_dict),
        protection=ProtectionConfig(**protection_dict),
    )

    entrypoint_script = launch_dict.get("entrypoint_script")
    entrypoint_args = (
        [str(_resolve_file_path(str(entrypoint_script)))]
        if entrypoint_script
        else ["-m", "lora_tta.cli"]
    )
    initial_worker_args = _tta_cli_args(tta_config)
    cmd = [
        resolved_python,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(resolved_nproc),
        "--master_port",
        str(resolved_port),
        *entrypoint_args,
        *initial_worker_args,
    ]
    worker_arg_offset = len(cmd) - len(initial_worker_args)
    if (
        diagnose_synonyms
        or diagnose_prompt_routing
        or diagnose_cross_time_only
        or diagnose_synonym_tta_overlap
    ):
        cmd.append("--diagnose-synonyms")
    if diagnose_prompt_routing:
        cmd.append("--diagnose-prompt-routing")
    if diagnose_cross_time_only or diagnose_synonym_tta_overlap:
        cmd.append("--diagnose-cross-time-only")
    if diagnose_synonym_tta_overlap:
        cmd.append("--diagnose-synonym-tta-overlap")
        cmd.append("--boundary-widths")
        cmd.extend(str(value) for value in resolved_boundary_widths)
    if resolved_teacher_diagnostic:
        cmd.append("--diagnose-teacher-prompts")
    if resolved_teacher_gradient_diagnostic:
        cmd.append("--diagnose-teacher-prompt-gradients")
    if diagnose_selected_flips:
        cmd.append("--diagnose-selected-flips")
    if diagnose_selection_conflicts:
        cmd.append("--diagnose-selection-conflicts")
    if resolved_teacher_trial_steps > 0:
        cmd.extend(
            [
                "--teacher-prompt-trial-steps",
                str(resolved_teacher_trial_steps),
            ]
        )
    if (
        resolved_teacher_diagnostic
        or resolved_teacher_gradient_diagnostic
        or resolved_teacher_trial_steps > 0
    ):
        for class_id in resolved_teacher_classes:
            cmd.extend(["--teacher-diagnostic-class", str(class_id)])

    runtime_args_file = None
    runtime_args: tuple[str, ...] = ()
    if not entrypoint_script:
        visible_worker_args = tuple(cmd[worker_arg_offset:worker_arg_offset + 2])
        if not visible_worker_args or visible_worker_args[0] != "--eval-config":
            raise RuntimeError("worker arguments must start with --eval-config")
        runtime_args = tuple(cmd[worker_arg_offset + 2:])
        runtime_args_file = str(
            _resolve_file_path(f"{resolved_run_dir}/runtime.args")
        )
        cmd = [*cmd[:worker_arg_offset], *visible_worker_args]
    worker_env = _as_dict(launch_dict.get("env"))
    if runtime_args_file:
        worker_env["LORA_TTA_ARGS_FILE"] = runtime_args_file
    base_command = f"{_shell_env_prefix(worker_env, resolved_gpus, timeout_seconds)} {shlex.join(cmd)}"
    shell_command = base_command
    if resolved_log:
        shell_command = f"{base_command} 2>&1 | tee -a {shlex.quote(resolved_log)}"

    config_snapshot = {
        "source_config": str(config_path),
        "profile": profile,
        **tta_config.to_dict(),
        "launch": {
            "dataset": dataset,
            "tag": tag,
            "run_id": run_id,
            "work_dir": str(work_dir),
            "tmux": resolved_tmux,
            "session": resolved_session,
            "gpus": resolved_gpus,
            "nproc_per_node": resolved_nproc,
            "master_port": resolved_port,
            "timeout_seconds": timeout_seconds,
            "output_json": resolved_output,
            "log_file": resolved_log,
        },
        "_runtime_args_file": runtime_args_file,
        "_runtime_args": runtime_args,
    }

    return RunPlan(
        config_path=config_path,
        profile=profile,
        work_dir=work_dir,
        tmux=resolved_tmux,
        session=resolved_session,
        output_json=resolved_output,
        log_file=resolved_log,
        config_file=resolved_config_file,
        config_snapshot=config_snapshot,
        shell_command=shell_command,
    )


def _write_config_snapshot(plan: RunPlan) -> None:
    if not plan.config_file:
        return
    config_path = _resolve_file_path(plan.config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = plan.config_snapshot
    lines = [
        "# Auto-generated effective LoRA-TTA configuration.",
        f"source_config = {snapshot['source_config']!r}",
        f"profile = {snapshot['profile']!r}",
        "",
    ]
    for section_name in (*CONFIG_SECTION_TYPES, "launch"):
        rendered = pprint.pformat(
            snapshot[section_name],
            sort_dicts=False,
            width=100,
        )
        lines.extend((f"{section_name} = {rendered}", ""))
    config_path.write_text("\n".join(lines), encoding="utf-8")


def _write_runtime_args(plan: RunPlan) -> None:
    args_file = plan.config_snapshot.get("_runtime_args_file")
    if not args_file:
        return
    runtime_args = tuple(plan.config_snapshot.get("_runtime_args", ()))
    if any("\n" in arg or "\r" in arg for arg in runtime_args):
        raise ValueError("worker arguments must not contain line breaks")
    args_path = _resolve_file_path(str(args_file))
    args_path.parent.mkdir(parents=True, exist_ok=True)
    args_path.write_text("\n".join(runtime_args) + "\n", encoding="utf-8")


def _write_header(plan: RunPlan) -> None:
    _resolve_file_path(plan.output_json).parent.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(plan)
    _write_runtime_args(plan)
    if not plan.log_file:
        return
    log_path = _resolve_file_path(plan.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().strftime('%F %T')}] config={plan.config_path}\n")
        handle.write(f"profile={plan.profile}\n")
        handle.write(f"session={plan.session}\n")
        handle.write(f"output_json={plan.output_json}\n")
        if plan.config_file:
            handle.write(f"effective_config={plan.config_file}\n")
        handle.write(f"command={plan.shell_command}\n")


def _start_plan(plan: RunPlan) -> None:
    _write_header(plan)
    if plan.tmux:
        existing = subprocess.run(
            ["tmux", "has-session", "-t", plan.session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if existing.returncode == 0:
            raise SystemExit(f"tmux session already exists: {plan.session}")
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", plan.session, "-c", str(plan.work_dir), plan.shell_command],
            check=True,
        )
        print(f"started tmux session: {plan.session}")
        if plan.log_file:
            print(f"log: {_resolve_file_path(plan.log_file)}")
        if plan.config_file:
            print(f"config: {_resolve_file_path(plan.config_file)}")
        print(f"output: {_resolve_file_path(plan.output_json)}")
        return

    subprocess.run(
        plan.shell_command,
        cwd=str(plan.work_dir),
        shell=True,
        executable="/bin/bash",
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven LoRA-TTA launcher.")
    parser.add_argument("config", help="Python config file, e.g. tta-configs/cfg_udd5.py")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without starting it.")
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(ACCELERATION_PROFILES)),
        default=DEFAULT_ACCELERATION_PROFILE,
        help="Launcher acceleration profile; defaults to fast.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--nproc-per-node", "--nproc", dest="nproc_per_node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--log-file", "--log", dest="log_file", default=None)
    parser.add_argument(
        "--diagnose-synonyms",
        action="store_true",
        help="Embed pre/post per-alias causal diagnostics in the result JSON.",
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
        help="Localize synonym/TTA changes by GT boundary region.",
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record candidate-teacher TopK diagnostics.",
    )
    parser.add_argument(
        "--diagnose-teacher-prompt-gradients",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record candidate-teacher gradient alignment diagnostics.",
    )
    parser.add_argument(
        "--teacher-diagnostic-class",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--diagnose-selected-flips",
        action="store_true",
        help="Record post-TTA retention and flips of frozen Teacher TopK pixels.",
    )
    parser.add_argument(
        "--diagnose-selection-conflicts",
        action="store_true",
        help="Record retained TopK score ordering and positive/negative reuse.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help=(
            "Override a runtime, optim, mining, loss, prompt, or launch value. "
            "May be repeated; later values win."
        ),
    )

    tmux_group = parser.add_mutually_exclusive_group()
    tmux_group.add_argument("--tmux", dest="tmux", action="store_true", default=None)
    tmux_group.add_argument("--no-tmux", dest="tmux", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_run_plan(
        args.config,
        profile=args.profile,
        overrides=args.overrides,
        run_id=args.run_id,
        tmux=args.tmux,
        session=args.session,
        gpus=args.gpus,
        nproc_per_node=args.nproc_per_node,
        master_port=args.master_port,
        output_json=args.output_json,
        log_file=args.log_file,
        diagnose_synonyms=args.diagnose_synonyms,
        diagnose_prompt_routing=args.diagnose_prompt_routing,
        diagnose_cross_time_only=args.diagnose_cross_time_only,
        diagnose_synonym_tta_overlap=(
            args.diagnose_synonym_tta_overlap
        ),
        boundary_widths=args.boundary_widths,
        diagnose_teacher_prompts=args.diagnose_teacher_prompts,
        diagnose_teacher_prompt_gradients=(
            args.diagnose_teacher_prompt_gradients
        ),
        diagnose_selected_flips=args.diagnose_selected_flips,
        diagnose_selection_conflicts=(
            args.diagnose_selection_conflicts
        ),
        teacher_diagnostic_classes=args.teacher_diagnostic_class,
    )
    if args.dry_run:
        print(f"profile: {plan.profile}")
        print(f"tmux: {plan.tmux}")
        print(f"session: {plan.session}")
        print(f"work_dir: {plan.work_dir}")
        if plan.log_file:
            print(f"log: {_resolve_file_path(plan.log_file)}")
        if plan.config_file:
            print(f"config: {_resolve_file_path(plan.config_file)}")
        print(f"output: {_resolve_file_path(plan.output_json)}")
        print("command:")
        print(plan.shell_command)
        return
    _start_plan(plan)


if __name__ == "__main__":
    main()
