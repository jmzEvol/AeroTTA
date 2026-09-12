#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIXED_AUDIT_SETTINGS = {
    "split": "test",
    "resolution": 1008,
    "mask_chunk": 64,
    "prob_thd": 0.3,
    "tau_pos": 0.5,
    "rho": 0.1,
    "kmax": 2048,
    "n_min": 64,
    "bg_idx": 0,
    "presence_gate_power": 1.0,
    "lora_rank": 8,
    "lora_alpha": 16.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit canonical vs synonym LoveDA mining."
    )
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--canonical-classname-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--mask-chunk", type=int, default=64)
    parser.add_argument("--prob-thd", type=float, default=0.3)
    parser.add_argument("--tau-pos", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--kmax", type=int, default=2048)
    parser.add_argument("--n-min", type=int, default=64)
    parser.add_argument("--bg-idx", type=int, default=0)
    parser.add_argument("--presence-gate-power", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--expected-images", type=int, default=1669)
    parser.add_argument("--expected-synonym-miou", type=float, default=0.466608)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.0002)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_fixed_audit_settings(args: argparse.Namespace) -> None:
    for name, expected in FIXED_AUDIT_SETTINGS.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"fixed audit setting {name} must be {expected!r}, got {actual!r}"
            )


def validate_background_index(bg_idx: int, *, num_classes: int) -> None:
    if int(bg_idx) < 0 or int(bg_idx) >= int(num_classes):
        raise ValueError(
            f"background index {bg_idx} is outside [0, {int(num_classes)})"
        )


def selection_valid_mask(gt):
    import torch

    return torch.ones_like(gt, dtype=torch.bool)


def rank_stride_indices(
    length: int,
    *,
    rank: int,
    world_size: int,
) -> list[int]:
    if length < 0 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid rank-stride arguments")
    return list(range(int(rank), int(length), int(world_size)))


def effective_dataset_length(dataset_length: int, *, max_samples: int) -> int:
    if dataset_length < 0 or max_samples < 0:
        raise ValueError("dataset length and max samples must be non-negative")
    return (
        int(dataset_length)
        if int(max_samples) == 0
        else min(int(dataset_length), int(max_samples))
    )


def validate_canonical_prompts(
    query_words: list[str],
    query_idx_list: list[int],
    *,
    expected_classes: int,
) -> list[str]:
    if len(query_words) != len(query_idx_list):
        raise ValueError("canonical query word/mapping length mismatch")
    canonical_words = []
    for cls_idx in range(int(expected_classes)):
        words = [
            word
            for word, mapped in zip(query_words, query_idx_list)
            if int(mapped) == cls_idx
        ]
        if len(words) != 1:
            raise ValueError(
                "canonical prompt file must contain exactly one query per class; "
                f"class {cls_idx} has {len(words)}"
            )
        canonical_words.append(words[0])
    mapped_classes = {int(value) for value in query_idx_list}
    expected = set(range(int(expected_classes)))
    if mapped_classes != expected:
        raise ValueError(
            f"canonical prompt classes mismatch: {sorted(mapped_classes)} vs "
            f"{sorted(expected)}"
        )
    return canonical_words


def validate_sample_ids(
    sample_ids: list[str],
    *,
    expected_count: int,
) -> list[str]:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample ID in distributed audit")
    if len(sample_ids) != int(expected_count):
        raise ValueError(
            f"audit sample count mismatch: {len(sample_ids)} vs expected "
            f"{expected_count}"
        )
    return sorted(str(value) for value in sample_ids)


def gather_rank_objects(local_object, *, world_size: int):
    if int(world_size) == 1:
        return [local_object]
    import torch.distributed as dist

    gathered = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, local_object)
    return gathered


def validate_output_paths(json_path: Path, markdown_path: Path) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    if json_path.suffix != ".json" or markdown_path.suffix != ".md":
        raise ValueError("audit outputs require .json and .md suffixes")
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown paths must differ")
    if json_path.with_suffix("").resolve() != markdown_path.with_suffix("").resolve():
        raise ValueError("audit outputs require a shared filename stem")
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"refusing to overwrite unreadable non-audit artifact: {json_path}"
            ) from error
        if existing.get("audit_type") != "loveda_prompt_mining":
            raise ValueError(
                f"refusing to overwrite non-audit artifact: {json_path}"
            )
    elif markdown_path.exists():
        raise ValueError(
            "refusing to overwrite Markdown without its matching audit JSON: "
            f"{markdown_path}"
        )


def write_outputs(
    result: dict,
    markdown: str,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(".json.tmp")
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_tmp.write_text(markdown, encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)


def merge_rank_results(rank_results: list[dict]) -> dict:
    if not rank_results:
        raise ValueError("cannot merge an empty rank-result list")
    from lora_tta.prompt_mining_audit import merge_audit_accumulators

    accumulator = rank_results[0]["accumulator"]
    synonym_confusion = rank_results[0]["synonym_confusion"].clone()
    canonical_confusion = rank_results[0]["canonical_confusion"].clone()
    sample_ids = list(rank_results[0]["sample_ids"])
    for rank_result in rank_results[1:]:
        accumulator = merge_audit_accumulators(
            accumulator,
            rank_result["accumulator"],
        )
        synonym_confusion += rank_result["synonym_confusion"]
        canonical_confusion += rank_result["canonical_confusion"]
        sample_ids.extend(rank_result["sample_ids"])
    return {
        "accumulator": accumulator,
        "synonym_confusion": synonym_confusion,
        "canonical_confusion": canonical_confusion,
        "sample_ids": sample_ids,
    }


def _sample_id_from_batch(batch, dataset_index: int) -> str:
    if isinstance(batch, dict):
        data_samples = batch.get("data_samples", [])
    else:
        data_samples = [item["data_samples"] for item in batch]
    sample = data_samples[0] if data_samples else None
    if sample is not None:
        path = sample.metainfo.get("img_path") or sample.metainfo.get("seg_map_path")
        if path:
            return str(path)
    return f"dataset-index:{int(dataset_index)}"


def _metric_summary_from_matrix(matrix):
    from lora_tta.metrics import ConfusionMatrix

    metric = ConfusionMatrix(int(matrix.shape[0]), device="cpu")
    metric.matrix.copy_(matrix.detach().cpu().long())
    summary = metric.summary()
    return {
        "miou": float(summary.miou),
        "per_class_iou": [float(value) for value in summary.per_class_iou],
        "valid_classes": int(summary.valid_classes),
    }


def run(args: argparse.Namespace) -> dict | None:
    import torch
    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader, Subset

    from lora_tta.adapter import (
        SAM3LoRAAdapter,
        build_dataset_from_eval_config,
        load_eval_config,
        resolve_project_path,
    )
    from lora_tta.config import LossConfig, MiningConfig
    from lora_tta.metrics import ConfusionMatrix
    from lora_tta.prompt_mining_audit import (
        build_canonical_query_ids,
        build_prompt_class_views,
        finalize_audit,
        new_audit_accumulator,
        prediction_from_scores,
        render_markdown_report,
        run_mining_variants,
        update_audit_accumulator,
    )
    from lora_tta.runtime import (
        cleanup_distributed,
        set_random_seed,
        setup_distributed,
    )
    from segearthov3_segmentor import get_cls_idx

    validate_fixed_audit_settings(args)
    set_random_seed(args.seed)
    distributed = setup_distributed(args.device)
    try:
        json_path = Path(args.output_json)
        markdown_path = Path(args.output_markdown)
        validate_output_paths(json_path, markdown_path)

        eval_config = load_eval_config(args.eval_config)
        dataset, dataloader_config = build_dataset_from_eval_config(
            eval_config,
            split=args.split,
            num_workers=args.num_workers,
        )
        effective_length = effective_dataset_length(
            len(dataset),
            max_samples=args.max_samples,
        )
        if args.max_samples == 0 and effective_length != int(args.expected_images):
            raise ValueError(
                f"full audit dataset size mismatch: {effective_length} vs expected "
                f"{args.expected_images}"
            )
        local_indices = rank_stride_indices(
            effective_length,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        loader = DataLoader(
            Subset(dataset, local_indices),
            shuffle=False,
            collate_fn=pseudo_collate,
            **dataloader_config,
        )

        adapter = SAM3LoRAAdapter(
            eval_cfg=eval_config,
            device=distributed.device,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_layers=None,
            source_lora_path=None,
            resolution=args.resolution,
        )
        validate_background_index(args.bg_idx, num_classes=adapter.num_classes)
        canonical_path = resolve_project_path(args.canonical_classname_path)
        canonical_file_words, canonical_file_idx = get_cls_idx(canonical_path)
        canonical_words = validate_canonical_prompts(
            canonical_file_words,
            canonical_file_idx,
            expected_classes=adapter.num_classes,
        )
        canonical_query_ids = build_canonical_query_ids(
            adapter.query_words,
            adapter.query_idx_list,
            canonical_words,
        )
        class_names = [
            canonical_words[cls_idx]
            for cls_idx in range(int(adapter.num_classes))
        ]

        mining = MiningConfig(
            tau_pos=args.tau_pos,
            prob_thd=args.prob_thd,
            presence_gate_power=args.presence_gate_power,
            rho=args.rho,
            kmax=args.kmax,
            n_min=args.n_min,
            include_bg=False,
            bg_idx=args.bg_idx,
            class_weight_mode="mean_margin",
            class_weight_min=0.3,
            drop_overlaps=True,
        )
        loss = LossConfig(
            positive_target_mode="PTST",
            positive_target_min=0.0,
            positive_target_max=0.95,
        )
        accumulator = new_audit_accumulator(
            num_classes=adapter.num_classes,
            num_queries=adapter.num_queries,
        )
        synonym_metric = ConfusionMatrix(
            adapter.num_classes,
            device=distributed.device,
        )
        canonical_metric = ConfusionMatrix(
            adapter.num_classes,
            device=distributed.device,
        )
        sample_ids = []

        for local_position, batch in enumerate(loader):
            dataset_index = local_indices[local_position]
            image_batch, gt_labels = adapter.batch_to_tensors(batch)
            gt = gt_labels[0].long()
            teacher_size = tuple(gt.shape[-2:])
            backbone_output = adapter.encode_image(image_batch)
            with torch.no_grad():
                query_scores, presence_logits = adapter.forward_fused_scores(
                    backbone_output,
                    out_size=teacher_size,
                    presence_gate_power=args.presence_gate_power,
                    mask_chunk=args.mask_chunk,
                    grad=False,
                )
            views = build_prompt_class_views(
                query_scores=query_scores,
                presence_logits=presence_logits,
                query_idx_list=adapter.query_idx_list,
                canonical_query_ids=canonical_query_ids,
                num_classes=adapter.num_classes,
            )
            reference_scores, reference_presence, _reference_pred = (
                adapter.class_scores_for_mining(
                    query_scores,
                    presence_logits,
                    presence_gate_power=args.presence_gate_power,
                )
            )
            torch.testing.assert_close(views.synonym_scores, reference_scores)
            torch.testing.assert_close(views.synonym_presence, reference_presence)

            valid = selection_valid_mask(gt).to(device=distributed.device)
            variants = run_mining_variants(
                views=views,
                valid=valid,
                mining=mining,
                loss=loss,
            )
            synonym_pred = prediction_from_scores(
                views.synonym_scores,
                prob_thd=args.prob_thd,
                bg_idx=args.bg_idx,
            )
            canonical_pred = prediction_from_scores(
                views.canonical_scores,
                prob_thd=args.prob_thd,
                bg_idx=args.bg_idx,
            )
            synonym_metric.update(synonym_pred, gt)
            canonical_metric.update(canonical_pred, gt)
            sample_id = _sample_id_from_batch(batch, dataset_index)
            sample_ids.append(sample_id)
            update_audit_accumulator(
                accumulator,
                variants=variants,
                views=views,
                gt=gt,
                sample_id=sample_id,
                dataset_index=dataset_index,
                bg_idx=args.bg_idx,
                tau_pos=args.tau_pos,
            )

            processed = local_position + 1
            if processed % 5 == 0 or processed == len(local_indices):
                print(
                    f"[prompt-audit][rank {distributed.rank}/"
                    f"{distributed.world_size}] {processed}/{len(local_indices)} "
                    f"{Path(sample_id).name}",
                    flush=True,
                )

        local_result = {
            "accumulator": accumulator,
            "synonym_confusion": synonym_metric.matrix.detach().cpu(),
            "canonical_confusion": canonical_metric.matrix.detach().cpu(),
            "sample_ids": sample_ids,
        }
        rank_results = gather_rank_objects(
            local_result,
            world_size=distributed.world_size,
        )
        if not distributed.is_main:
            return None

        merged = merge_rank_results(rank_results)
        validated_ids = validate_sample_ids(
            merged["sample_ids"],
            expected_count=effective_length,
        )
        synonym_baseline = _metric_summary_from_matrix(
            merged["synonym_confusion"]
        )
        canonical_baseline = _metric_summary_from_matrix(
            merged["canonical_confusion"]
        )
        parameters = {
            "eval_config": str(args.eval_config),
            "canonical_classname_path": str(args.canonical_classname_path),
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "resolution": int(args.resolution),
            "mask_chunk": int(args.mask_chunk),
            "seed": int(args.seed),
            "prob_thd": float(args.prob_thd),
            "tau_pos": float(args.tau_pos),
            "rho": float(args.rho),
            "kmax": int(args.kmax),
            "n_min": int(args.n_min),
            "bg_idx": int(args.bg_idx),
            "include_bg": False,
            "drop_overlaps": True,
            "presence_gate_power": float(args.presence_gate_power),
            "lora_rank": int(args.lora_rank),
            "lora_alpha": float(args.lora_alpha),
            "world_size": int(distributed.world_size),
        }
        result = finalize_audit(
            merged["accumulator"],
            class_names=class_names,
            query_words=list(adapter.query_words),
            query_idx_list=[int(value) for value in adapter.query_idx_list],
            canonical_query_ids=canonical_query_ids,
            sample_ids=validated_ids,
            synonym_baseline=synonym_baseline,
            canonical_baseline=canonical_baseline,
            expected_synonym_miou=args.expected_synonym_miou,
            reproduction_tolerance=args.reproduction_tolerance,
            enforce_reproduction_guard=(
                args.max_samples == 0
                and effective_length == int(args.expected_images)
            ),
            parameters=parameters,
        )
        markdown = render_markdown_report(result)
        write_outputs(
            result,
            markdown,
            json_path=json_path,
            markdown_path=markdown_path,
        )
        print(f"[prompt-audit] wrote {json_path}", flush=True)
        print(f"[prompt-audit] wrote {markdown_path}", flush=True)
        return result
    finally:
        cleanup_distributed()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
