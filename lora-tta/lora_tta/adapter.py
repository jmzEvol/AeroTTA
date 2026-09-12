from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmseg.registry import DATASETS
from torch.utils.data import DataLoader
from torchvision.transforms import v2

SAMTTA_ROOT = Path(__file__).resolve().parents[2]
if str(SAMTTA_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMTTA_ROOT))

import custom_datasets  # noqa: F401,E402
import mmseg.datasets.transforms  # noqa: F401,E402
from segearthov3_segmentor import SegEarthOV3Segmentation, get_cls_idx  # noqa: E402
from sam3.model.data_misc import FindStage  # noqa: E402
from train_lora import aggregate_query_logits, load_batch_tensors  # noqa: E402

from .scores import (
    aggregate_gated_instance_map,
    aggregate_proc_instance_map,
    class_scores_from_query_scores,
    filtered_raw_instance_scores_at_indices,
    fuse_proc_pgrf_query_head_scores,
    predict_from_class_scores,
    sample_maps_at_flat_indices,
    upsample_score_maps,
    fuse_query_head_scores,
)


class FilteredRawStudentContext(NamedTuple):
    query_ids: tuple[int, ...]
    batch_outputs: tuple[dict, ...]


class TeacherForwardResult(NamedTuple):
    query_scores: torch.Tensor
    inference_query_scores: torch.Tensor
    target_query_scores: torch.Tensor
    presence_logits: torch.Tensor
    student_query_logits: torch.Tensor | None
    student_presence_logits: torch.Tensor | None
    student_query_ids: tuple[int, ...]
    filtered_raw_student_context: FilteredRawStudentContext | None


class QueryScoreComponents(NamedTuple):
    semantic_scores: torch.Tensor
    instance_scores: torch.Tensor
    presence_scores: torch.Tensor


class FilteredRawStudentResult(NamedTuple):
    query_logits: torch.Tensor
    presence_logits: torch.Tensor
    semantic_scores: torch.Tensor
    instance_scores: torch.Tensor
    fused_scores: torch.Tensor
    kept_instance_counts: torch.Tensor
    query_ids: tuple[int, ...]


def resolve_project_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = SAMTTA_ROOT / p
    return str(p)


def load_eval_config(path: str | Path) -> Config:
    cfg = Config.fromfile(resolve_project_path(path))
    if "test_dataloader" not in cfg:
        raise ValueError(f"{path} does not define test_dataloader")
    return cfg


def build_dataset_from_eval_config(cfg: Config, *, split: str, num_workers: int):
    init_default_scope("mmseg")
    dataloader_key = f"{split}_dataloader"
    if dataloader_key not in cfg:
        if split == "test":
            dataloader_key = "test_dataloader"
        else:
            raise ValueError(f"config does not define {dataloader_key}")
    dataloader_cfg = copy.deepcopy(cfg[dataloader_key])
    dataset_cfg = dataloader_cfg["dataset"]
    if dataset_cfg.get("data_root") is not None:
        dataset_cfg["data_root"] = resolve_project_path(dataset_cfg["data_root"])
    dataset = DATASETS.build(dataset_cfg)
    dataloader_cfg.pop("dataset", None)
    dataloader_cfg.pop("sampler", None)
    dataloader_cfg["batch_size"] = 1
    dataloader_cfg["num_workers"] = int(num_workers)
    dataloader_cfg["persistent_workers"] = bool(num_workers > 0 and dataloader_cfg.get("persistent_workers", False))
    dataloader_cfg.setdefault("pin_memory", True)
    return dataset, dataloader_cfg


def build_dataloader(dataset, dataloader_cfg: dict, *, rank: int, world_size: int):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if world_size > 1 else torch.utils.data.SequentialSampler(dataset)
    return DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=pseudo_collate,
        **dataloader_cfg,
    )


class SAM3LoRAAdapter:
    """Pure-source SAM3/LoRA adapter for the clean TTA engine."""

    def __init__(
        self,
        *,
        eval_cfg: Config,
        device: torch.device,
        lora_rank: int,
        lora_alpha: float,
        lora_layers: tuple[int, ...] | None,
        lora_layer_ranks: dict[int, int] | None = None,
        lora_adapt_key: bool = False,
        source_lora_path: str | None = None,
        resolution: int = 1008,
        query_batch_size: int = 1,
        full_query_batch_size: int = 1,
    ) -> None:
        model_cfg = copy.deepcopy(eval_cfg.get("model", {}))
        classname_path = resolve_project_path(model_cfg.get("classname_path"))
        if classname_path is None:
            raise ValueError("eval config model.classname_path is required")
        self.query_words, self.query_idx_list = get_cls_idx(classname_path)
        self.num_classes = max(self.query_idx_list) + 1
        self.num_queries = len(self.query_words)
        self.bg_idx = int(model_cfg.get("bg_idx", 0))
        self.device = device
        self.presence_gate_power = float(model_cfg.get("presence_gate_power", 1.0))
        self.confidence_threshold = float(model_cfg.get("confidence_threshold", 0.5))
        self.source_lora_path = resolve_project_path(source_lora_path) if source_lora_path else None
        self.query_batch_size = int(query_batch_size)
        if self.query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        self.full_query_batch_size = int(full_query_batch_size)
        if self.full_query_batch_size <= 0:
            raise ValueError("full_query_batch_size must be positive")

        self.segmentor = SegEarthOV3Segmentation(
            classname_path=classname_path,
            device=device,
            prob_thd=0.0,
            bg_idx=self.bg_idx,
            confidence_threshold=self.confidence_threshold,
            use_presence_score=False,
            enable_lora=True,
            lora_rank=int(lora_rank),
            lora_alpha=float(lora_alpha),
            lora_layers=list(lora_layers) if lora_layers is not None else None,
            lora_layer_ranks=dict(lora_layer_ranks or {}),
            lora_adapt_key=bool(lora_adapt_key),
            lora_path=self.source_lora_path,
        )
        self.sam3_model = self.segmentor.processor.model
        self.sam3_model.eval()
        self.encoder = self.segmentor.get_encoder()
        self.lora_params_by_layer = self.encoder.get_lora_parameters_by_layer()
        self.lora_params = [
            param
            for layer_index in sorted(self.lora_params_by_layer)
            for param in self.lora_params_by_layer[layer_index]
        ]
        self.img_transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(int(resolution), int(resolution))),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            self.text_features = self.sam3_model.backbone.forward_text(self.query_words, device=device)

    def batch_to_tensors(self, batch):
        return load_batch_tensors(batch, self.img_transform, self.device)

    def encode_image(self, image_batch: torch.Tensor):
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return self.sam3_model.backbone.forward_image(image_batch)

    def _forward_grounding_for_queries(
        self,
        backbone_out_base,
        *,
        query_ids,
        semantic_only: bool = False,
    ):
        query_ids = tuple(int(query_id) for query_id in query_ids)
        if not query_ids:
            raise ValueError("at least one query id is required")
        num_queries = len(query_ids)
        backbone_out = {**backbone_out_base, **self.text_features}
        find_stage = FindStage(
            img_ids=torch.zeros(num_queries, device=self.device, dtype=torch.long),
            text_ids=torch.tensor(query_ids, device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = self.sam3_model._get_dummy_prompt(num_prompts=num_queries)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            return self.sam3_model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                find_target=None,
                geometric_prompt=geometric_prompt,
                semantic_only=semantic_only,
            )

    def _forward_grounding_for_query(self, backbone_out_base, *, query_id: int):
        return self._forward_grounding_for_queries(
            backbone_out_base,
            query_ids=(int(query_id),),
        )

    @staticmethod
    def _presence_logits_from_output(out: dict, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        presence_logits = out.get("presence_logit_dec", None)
        if presence_logits is None:
            return like.new_zeros((int(batch_size),))
        return presence_logits.float().reshape(int(batch_size), -1)[:, 0]

    @staticmethod
    def _semantic_logits_from_output(
        out: dict,
        *,
        out_size: tuple[int, int] | None,
    ) -> torch.Tensor:
        semantic_logits = out["semantic_seg"][:, 0].float()
        if out_size is not None:
            semantic_logits = upsample_score_maps(semantic_logits, tuple(out_size))
        return semantic_logits

    def forward_queries(
        self,
        backbone_out_base,
        *,
        grad: bool,
        out_size: tuple[int, int] | None = None,
        query_ids: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_query_ids = (
            tuple(range(self.num_queries))
            if query_ids is None
            else tuple(int(query_id) for query_id in query_ids)
        )
        if not resolved_query_ids:
            raise ValueError("at least one query id is required")
        if len(set(resolved_query_ids)) != len(resolved_query_ids):
            raise ValueError("query ids must not contain duplicates")
        if min(resolved_query_ids) < 0 or max(resolved_query_ids) >= self.num_queries:
            raise ValueError(
                f"query ids {resolved_query_ids} exceed the range "
                f"[0, {self.num_queries - 1}]"
            )
        query_logits_batches = []
        presence_logits_batches = []
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            for start in range(0, len(resolved_query_ids), self.query_batch_size):
                batch_query_ids = resolved_query_ids[start : start + self.query_batch_size]
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=True,
                )
                semantic_logits = self._semantic_logits_from_output(out, out_size=out_size)
                presence_logit = self._presence_logits_from_output(
                    out,
                    batch_size=len(batch_query_ids),
                    like=semantic_logits,
                )
                query_logits_batches.append(semantic_logits)
                presence_logits_batches.append(presence_logit)
        return (
            torch.cat(query_logits_batches, dim=0).unsqueeze(0).float(),
            torch.cat(presence_logits_batches, dim=0).unsqueeze(0).float(),
        )

    def forward_filtered_raw_student(
        self,
        backbone_out_base,
        *,
        query_ids: tuple[int, ...] | list[int],
        flat_indices: torch.Tensor,
        target_size: tuple[int, int],
        mask_chunk: int,
        probability_reconstruction_query_ids: tuple[int, ...]
        | list[int] = (),
        precomputed_context: FilteredRawStudentContext | None = None,
    ) -> FilteredRawStudentResult:
        resolved_query_ids = tuple(int(query_id) for query_id in query_ids)
        reconstruction_query_ids = {
            int(query_id)
            for query_id in probability_reconstruction_query_ids
        }
        if not resolved_query_ids:
            raise ValueError("at least one query id is required")
        if len(set(resolved_query_ids)) != len(resolved_query_ids):
            raise ValueError("query ids must not contain duplicates")
        if (
            min(resolved_query_ids) < 0
            or max(resolved_query_ids) >= self.num_queries
        ):
            raise ValueError(
                f"query ids {resolved_query_ids} exceed the range "
                f"[0, {self.num_queries - 1}]"
            )
        missing_reconstruction_ids = (
            reconstruction_query_ids - set(resolved_query_ids)
        )
        if missing_reconstruction_ids:
            raise ValueError(
                "probability reconstruction query ids are missing from "
                f"the requested subset: {tuple(sorted(missing_reconstruction_ids))}"
            )

        if (
            precomputed_context is not None
            and precomputed_context.query_ids != resolved_query_ids
        ):
            raise ValueError(
                "precomputed filtered-raw query order does not match the "
                f"requested queries: {precomputed_context.query_ids} != "
                f"{resolved_query_ids}"
            )

        semantic_batches = []
        instance_batches = []
        fused_batches = []
        fused_logit_batches = []
        presence_batches = []
        consumed_precomputed_batches = 0
        kept_count_batches = []
        with torch.enable_grad():
            for start in range(
                0,
                len(resolved_query_ids),
                self.full_query_batch_size,
            ):
                batch_query_ids = resolved_query_ids[
                    start : start + self.full_query_batch_size
                ]
                if precomputed_context is None:
                    out = self._forward_grounding_for_queries(
                        backbone_out_base,
                        query_ids=batch_query_ids,
                        semantic_only=False,
                    )
                else:
                    if consumed_precomputed_batches >= len(
                        precomputed_context.batch_outputs
                    ):
                        raise ValueError(
                            "precomputed filtered-raw context has too few batches"
                        )
                    out = precomputed_context.batch_outputs[
                        consumed_precomputed_batches
                    ]
                    consumed_precomputed_batches += 1
                semantic_logits = self._semantic_logits_from_output(
                    out,
                    out_size=None,
                )
                presence_logits = self._presence_logits_from_output(
                    out,
                    batch_size=len(batch_query_ids),
                    like=semantic_logits,
                )
                sampled_semantic_logits = sample_maps_at_flat_indices(
                    semantic_logits,
                    flat_indices=flat_indices,
                    target_size=target_size,
                )
                semantic_scores = sampled_semantic_logits.sigmoid()
                pred_masks = out.get("pred_masks", None)
                pred_logits = out.get("pred_logits", None)
                instance_scores = []
                kept_counts = []
                for batch_index in range(len(batch_query_ids)):
                    score, kept_count = (
                        filtered_raw_instance_scores_at_indices(
                            mask_logits=(
                                None
                                if pred_masks is None
                                else pred_masks[
                                    batch_index : batch_index + 1
                                ]
                            ),
                            det_logits=(
                                None
                                if pred_logits is None
                                else pred_logits[
                                    batch_index : batch_index + 1
                                ]
                            ),
                            presence_score=presence_logits[
                                batch_index
                            ].sigmoid(),
                            flat_indices=flat_indices,
                            target_size=target_size,
                            confidence_threshold=self.confidence_threshold,
                            mask_chunk=mask_chunk,
                        )
                    )
                    instance_scores.append(score)
                    kept_counts.append(kept_count)
                instance_score_batch = torch.stack(instance_scores, dim=0)
                fused_score_batch = torch.maximum(
                    semantic_scores,
                    instance_score_batch,
                )
                instance_logit_batch = torch.logit(
                    instance_score_batch.clamp(1e-6, 1.0 - 1e-6)
                )
                fused_logit_batch = torch.where(
                    instance_score_batch > semantic_scores,
                    instance_logit_batch,
                    sampled_semantic_logits,
                )
                if reconstruction_query_ids:
                    old_fused_logit_batch = torch.logit(
                        fused_score_batch.clamp(1e-6, 1.0 - 1e-6)
                    )
                    reconstruct_batch = torch.tensor(
                        [
                            query_id in reconstruction_query_ids
                            for query_id in batch_query_ids
                        ],
                        device=fused_logit_batch.device,
                        dtype=torch.bool,
                    ).unsqueeze(1)
                    fused_logit_batch = torch.where(
                        reconstruct_batch,
                        old_fused_logit_batch,
                        fused_logit_batch,
                    )
                semantic_batches.append(semantic_scores)
                instance_batches.append(instance_score_batch)
                fused_batches.append(fused_score_batch)
                fused_logit_batches.append(fused_logit_batch)
                presence_batches.append(presence_logits)
                kept_count_batches.append(
                    torch.tensor(
                        kept_counts,
                        device=presence_logits.device,
                        dtype=torch.long,
                    )
                )

        if (
            precomputed_context is not None
            and consumed_precomputed_batches
            != len(precomputed_context.batch_outputs)
        ):
            raise ValueError("precomputed filtered-raw context has too many batches")

        semantic_scores = torch.cat(semantic_batches, dim=0).float()
        instance_scores = torch.cat(instance_batches, dim=0).float()
        fused_scores = torch.cat(fused_batches, dim=0).float()
        fused_logits = torch.cat(fused_logit_batches, dim=0).float()
        return FilteredRawStudentResult(
            query_logits=fused_logits.unsqueeze(0).unsqueeze(-1),
            presence_logits=torch.cat(presence_batches, dim=0)
            .unsqueeze(0)
            .float(),
            semantic_scores=semantic_scores.unsqueeze(0),
            instance_scores=instance_scores.unsqueeze(0),
            fused_scores=fused_scores.unsqueeze(0),
            kept_instance_counts=torch.cat(kept_count_batches, dim=0)
            .unsqueeze(0),
            query_ids=resolved_query_ids,
        )

    def forward_query_one(
        self,
        backbone_out_base,
        *,
        query_id: int,
        out_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._forward_grounding_for_queries(
            backbone_out_base,
            query_ids=(int(query_id),),
            semantic_only=True,
        )
        semantic_logits = self._semantic_logits_from_output(out, out_size=out_size)
        presence_logit = self._presence_logits_from_output(out, batch_size=1, like=semantic_logits)
        return semantic_logits, presence_logit

    def _forward_fused_query_one(
        self,
        backbone_out_base,
        *,
        query_id: int,
        out_size: tuple[int, int],
        presence_gate_power: float,
        mask_chunk: int,
        score_fusion_mode: str = "legacy_max",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._forward_grounding_for_queries(
            backbone_out_base,
            query_ids=(int(query_id),),
        )
        fused_scores, presence_logits = self._fused_scores_from_output(
            out,
            out_size=out_size,
            presence_gate_power=presence_gate_power,
            mask_chunk=mask_chunk,
            score_fusion_mode=score_fusion_mode,
        )
        return fused_scores[0], presence_logits

    def _fused_scores_from_output(
        self,
        out: dict,
        *,
        out_size: tuple[int, int],
        presence_gate_power: float,
        mask_chunk: int,
        score_fusion_mode: str = "legacy_max",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score_views, presence_logits = SAM3LoRAAdapter._fused_score_views_from_output(
            self,
            out,
            out_size=out_size,
            presence_gate_powers=(presence_gate_power,),
            score_fusion_modes=(score_fusion_mode,),
            mask_chunk=mask_chunk,
        )
        return score_views[0], presence_logits

    def _fused_score_views_from_output(
        self,
        out: dict,
        *,
        out_size: tuple[int, int],
        presence_gate_powers: tuple[float, ...] | list[float],
        mask_chunk: int,
        score_fusion_modes: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        score_views, presence_logits, _components = (
            SAM3LoRAAdapter._fused_score_views_with_components_from_output(
                self,
                out,
                out_size=out_size,
                presence_gate_powers=presence_gate_powers,
                mask_chunk=mask_chunk,
                score_fusion_modes=score_fusion_modes,
            )
        )
        return score_views, presence_logits

    def _fused_score_views_with_components_from_output(
        self,
        out: dict,
        *,
        out_size: tuple[int, int],
        presence_gate_powers: tuple[float, ...] | list[float],
        mask_chunk: int,
        score_fusion_modes: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        torch.Tensor,
        QueryScoreComponents,
    ]:
        resolved_powers = tuple(float(power) for power in presence_gate_powers)
        if not resolved_powers:
            raise ValueError("at least one presence gate power is required")
        resolved_modes = (
            ("legacy_max",) * len(resolved_powers)
            if score_fusion_modes is None
            else tuple(str(mode) for mode in score_fusion_modes)
        )
        if len(resolved_modes) != len(resolved_powers):
            raise ValueError(
                "score fusion modes must match presence gate powers"
            )
        invalid_modes = sorted(
            set(resolved_modes) - {"legacy_max", "proc_pgrf"}
        )
        if invalid_modes:
            raise ValueError(
                f"unsupported score fusion modes: {tuple(invalid_modes)}"
            )
        semantic_logits = self._semantic_logits_from_output(
            out,
            out_size=out_size,
        )
        batch_size = int(semantic_logits.shape[0])
        presence_logits = self._presence_logits_from_output(
            out,
            batch_size=batch_size,
            like=semantic_logits,
        )
        pred_masks = out.get("pred_masks", None)
        pred_logits = out.get("pred_logits", None)
        fused_scores_by_view = [[] for _ in resolved_powers]
        semantic_scores = []
        instance_scores = []
        needs_legacy = "legacy_max" in resolved_modes
        needs_proc = "proc_pgrf" in resolved_modes
        for batch_index in range(batch_size):
            semantic_logit = semantic_logits[batch_index]
            presence_score = presence_logits[batch_index].float().sigmoid()
            if not needs_legacy or pred_masks is None or pred_logits is None:
                inst_gated = torch.zeros_like(semantic_logit, dtype=torch.float32)
            else:
                inst_gated = aggregate_gated_instance_map(
                    mask_logits=pred_masks[batch_index : batch_index + 1],
                    det_logits=pred_logits[batch_index : batch_index + 1],
                    presence_score=presence_score,
                    out_size=tuple(out_size),
                    confidence_threshold=self.confidence_threshold,
                    mask_chunk=int(mask_chunk),
                )
            if not needs_proc or pred_masks is None or pred_logits is None:
                proc_instance_best = torch.zeros_like(
                    semantic_logit,
                    dtype=torch.float32,
                )
                proc_max_instance_score = semantic_logit.new_zeros(
                    (),
                    dtype=torch.float32,
                )
            else:
                (
                    proc_instance_best,
                    proc_max_instance_score,
                ) = aggregate_proc_instance_map(
                    mask_logits=pred_masks[batch_index : batch_index + 1],
                    det_logits=pred_logits[batch_index : batch_index + 1],
                    presence_score=presence_score,
                    out_size=tuple(out_size),
                    confidence_threshold=self.confidence_threshold,
                    mask_chunk=int(mask_chunk),
                )
            semantic_scores.append(semantic_logit.float().sigmoid())
            instance_scores.append(
                inst_gated.float()
                if needs_legacy
                else proc_instance_best.float()
            )
            for view_index, (
                presence_gate_power,
                score_fusion_mode,
            ) in enumerate(zip(resolved_powers, resolved_modes, strict=True)):
                if score_fusion_mode == "legacy_max":
                    _semantic_prob, _semantic_gated, fused = (
                        fuse_query_head_scores(
                            semantic_logits=semantic_logit,
                            presence_score=presence_score,
                            inst_gated=inst_gated,
                            presence_gate_power=presence_gate_power,
                        )
                    )
                else:
                    _semantic_prob, _max_fused, fused = (
                        fuse_proc_pgrf_query_head_scores(
                            semantic_logits=semantic_logit,
                            presence_score=presence_score,
                            instance_best=proc_instance_best,
                            max_instance_score=proc_max_instance_score,
                        )
                    )
                fused_scores_by_view[view_index].append(fused)
        score_views = tuple(
            torch.stack(fused_scores, dim=0).float()
            for fused_scores in fused_scores_by_view
        )
        return (
            score_views,
            presence_logits.float(),
            QueryScoreComponents(
                semantic_scores=torch.stack(semantic_scores, dim=0).float(),
                instance_scores=torch.stack(instance_scores, dim=0).float(),
                presence_scores=presence_logits.float().sigmoid(),
            ),
        )

    def forward_fused_scores(
        self,
        backbone_out_base,
        *,
        out_size: tuple[int, int],
        presence_gate_power: float,
        mask_chunk: int,
        score_fusion_mode: str = "legacy_max",
        grad: bool = False,
        query_ids: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_query_ids = (
            tuple(range(self.num_queries))
            if query_ids is None
            else tuple(int(query_id) for query_id in query_ids)
        )
        if not resolved_query_ids:
            raise ValueError("at least one query id is required")
        if len(set(resolved_query_ids)) != len(resolved_query_ids):
            raise ValueError("query ids must not contain duplicates")
        if min(resolved_query_ids) < 0 or max(resolved_query_ids) >= self.num_queries:
            raise ValueError(
                f"query ids {resolved_query_ids} exceed the range "
                f"[0, {self.num_queries - 1}]"
            )
        query_score_batches = []
        presence_logit_batches = []
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            for start in range(0, len(resolved_query_ids), self.full_query_batch_size):
                batch_query_ids = resolved_query_ids[start : start + self.full_query_batch_size]
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=False,
                )
                fused, presence_logit = self._fused_scores_from_output(
                    out,
                    out_size=tuple(out_size),
                    presence_gate_power=presence_gate_power,
                    mask_chunk=int(mask_chunk),
                    score_fusion_mode=score_fusion_mode,
                )
                query_score_batches.append(fused)
                presence_logit_batches.append(presence_logit)
        return (
            torch.cat(query_score_batches, dim=0).unsqueeze(0).float(),
            torch.cat(presence_logit_batches, dim=0).unsqueeze(0).float(),
        )

    def forward_fused_score_views(
        self,
        backbone_out_base,
        *,
        out_size: tuple[int, int],
        presence_gate_powers: tuple[float, ...] | list[float],
        mask_chunk: int,
        score_fusion_modes: tuple[str, ...] | list[str] | None = None,
        query_ids: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        resolved_query_ids = (
            tuple(range(self.num_queries))
            if query_ids is None
            else tuple(int(query_id) for query_id in query_ids)
        )
        if not resolved_query_ids:
            raise ValueError("at least one query id is required")
        resolved_powers = tuple(float(power) for power in presence_gate_powers)
        if not resolved_powers:
            raise ValueError("at least one presence gate power is required")
        resolved_modes = (
            ("legacy_max",) * len(resolved_powers)
            if score_fusion_modes is None
            else tuple(str(mode) for mode in score_fusion_modes)
        )
        if len(resolved_modes) != len(resolved_powers):
            raise ValueError(
                "score fusion modes must match presence gate powers"
            )
        score_batches_by_view = [[] for _ in resolved_powers]
        presence_logit_batches = []
        with torch.no_grad():
            for start in range(0, len(resolved_query_ids), self.full_query_batch_size):
                batch_query_ids = resolved_query_ids[
                    start : start + self.full_query_batch_size
                ]
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=False,
                )
                score_views, presence_logits = self._fused_score_views_from_output(
                    out,
                    out_size=tuple(out_size),
                    presence_gate_powers=resolved_powers,
                    score_fusion_modes=resolved_modes,
                    mask_chunk=int(mask_chunk),
                )
                for view_index, scores in enumerate(score_views):
                    score_batches_by_view[view_index].append(scores)
                presence_logit_batches.append(presence_logits)
        return (
            tuple(
                torch.cat(score_batches, dim=0).unsqueeze(0).float()
                for score_batches in score_batches_by_view
            ),
            torch.cat(presence_logit_batches, dim=0).unsqueeze(0).float(),
        )

    def forward_oracle_score_views(
        self,
        backbone_out_base,
        *,
        out_size: tuple[int, int],
        presence_gate_powers: tuple[float, ...] | list[float],
        mask_chunk: int,
        score_fusion_modes: tuple[str, ...] | list[str] | None = None,
        query_ids: tuple[int, ...] | list[int] | None = None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        torch.Tensor,
        QueryScoreComponents,
    ]:
        resolved_query_ids = (
            tuple(range(self.num_queries))
            if query_ids is None
            else tuple(int(query_id) for query_id in query_ids)
        )
        if not resolved_query_ids:
            raise ValueError("at least one query id is required")
        if len(set(resolved_query_ids)) != len(resolved_query_ids):
            raise ValueError("query ids must not contain duplicates")
        if min(resolved_query_ids) < 0 or max(resolved_query_ids) >= self.num_queries:
            raise ValueError(
                f"query ids {resolved_query_ids} exceed the range "
                f"[0, {self.num_queries - 1}]"
            )
        resolved_powers = tuple(float(power) for power in presence_gate_powers)
        if not resolved_powers:
            raise ValueError("at least one presence gate power is required")
        resolved_modes = (
            ("legacy_max",) * len(resolved_powers)
            if score_fusion_modes is None
            else tuple(str(mode) for mode in score_fusion_modes)
        )
        if len(resolved_modes) != len(resolved_powers):
            raise ValueError("score fusion modes must match presence gate powers")

        score_batches_by_view = [[] for _ in resolved_powers]
        presence_logit_batches = []
        semantic_batches = []
        instance_batches = []
        presence_score_batches = []
        with torch.no_grad():
            for start in range(
                0,
                len(resolved_query_ids),
                self.full_query_batch_size,
            ):
                batch_query_ids = resolved_query_ids[
                    start : start + self.full_query_batch_size
                ]
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=False,
                )
                score_views, presence_logits, components = (
                    self._fused_score_views_with_components_from_output(
                        out,
                        out_size=tuple(out_size),
                        presence_gate_powers=resolved_powers,
                        score_fusion_modes=resolved_modes,
                        mask_chunk=int(mask_chunk),
                    )
                )
                for view_index, scores in enumerate(score_views):
                    score_batches_by_view[view_index].append(scores)
                presence_logit_batches.append(presence_logits)
                semantic_batches.append(components.semantic_scores)
                instance_batches.append(components.instance_scores)
                presence_score_batches.append(components.presence_scores)
        return (
            tuple(
                torch.cat(score_batches, dim=0).unsqueeze(0).float()
                for score_batches in score_batches_by_view
            ),
            torch.cat(presence_logit_batches, dim=0).unsqueeze(0).float(),
            QueryScoreComponents(
                semantic_scores=torch.cat(semantic_batches, dim=0)
                .unsqueeze(0)
                .float(),
                instance_scores=torch.cat(instance_batches, dim=0)
                .unsqueeze(0)
                .float(),
                presence_scores=torch.cat(presence_score_batches, dim=0)
                .unsqueeze(0)
                .float(),
            ),
        )

    def forward_teacher_scores(
        self,
        backbone_out_base,
        *,
        out_size: tuple[int, int],
        presence_gate_power: float,
        inference_presence_gate_power: float | None = None,
        target_presence_gate_power: float | None = None,
        score_fusion_mode: str = "legacy_max",
        inference_score_fusion_mode: str | None = None,
        target_score_fusion_mode: str | None = None,
        mask_chunk: int,
        grad_query_ids: tuple[int, ...] | list[int],
        retain_filtered_raw_student: bool = False,
    ) -> TeacherForwardResult:
        resolved_grad_query_ids = tuple(
            int(query_id) for query_id in grad_query_ids
        )
        if not resolved_grad_query_ids:
            raise ValueError("at least one gradient query id is required")
        if len(set(resolved_grad_query_ids)) != len(resolved_grad_query_ids):
            raise ValueError("gradient query ids must not contain duplicates")
        if (
            min(resolved_grad_query_ids) < 0
            or max(resolved_grad_query_ids) >= self.num_queries
        ):
            raise ValueError(
                f"gradient query ids {resolved_grad_query_ids} exceed the range "
                f"[0, {self.num_queries - 1}]"
            )

        teacher_scores_by_query = [None] * self.num_queries
        teacher_inference_scores_by_query = [None] * self.num_queries
        teacher_target_scores_by_query = [None] * self.num_queries
        teacher_presence_by_query = [None] * self.num_queries
        student_query_batches = []
        student_presence_batches = []
        filtered_raw_student_batch_outputs = []

        def record_teacher_batch(batch_query_ids, out) -> None:
            with torch.no_grad():
                score_views, presence_logits = self._fused_score_views_from_output(
                    out,
                    out_size=tuple(out_size),
                    presence_gate_powers=(
                        presence_gate_power,
                        (
                            presence_gate_power
                            if inference_presence_gate_power is None
                            else inference_presence_gate_power
                        ),
                        (
                            presence_gate_power
                            if target_presence_gate_power is None
                            else target_presence_gate_power
                        ),
                    ),
                    score_fusion_modes=(
                        score_fusion_mode,
                        (
                            score_fusion_mode
                            if inference_score_fusion_mode is None
                            else inference_score_fusion_mode
                        ),
                        (
                            score_fusion_mode
                            if target_score_fusion_mode is None
                            else target_score_fusion_mode
                        ),
                    ),
                    mask_chunk=int(mask_chunk),
                )
                (
                    fused_scores,
                    inference_fused_scores,
                    target_fused_scores,
                ) = score_views
            for batch_index, query_id in enumerate(batch_query_ids):
                teacher_scores_by_query[query_id] = (
                    fused_scores[batch_index].detach()
                )
                teacher_inference_scores_by_query[query_id] = (
                    inference_fused_scores[batch_index].detach()
                )
                teacher_target_scores_by_query[query_id] = (
                    target_fused_scores[batch_index].detach()
                )
                teacher_presence_by_query[query_id] = (
                    presence_logits[batch_index].detach()
                )

        grad_query_id_set = set(resolved_grad_query_ids)
        inference_only_query_ids = tuple(
            query_id
            for query_id in range(self.num_queries)
            if query_id not in grad_query_id_set
        )
        consumed_inference_query_ids = 0
        with torch.enable_grad():
            for start in range(
                0,
                len(resolved_grad_query_ids),
                self.full_query_batch_size,
            ):
                student_batch_query_ids = resolved_grad_query_ids[
                    start : start + self.full_query_batch_size
                ]
                remaining_capacity = (
                    self.full_query_batch_size - len(student_batch_query_ids)
                )
                fill_query_ids = inference_only_query_ids[
                    consumed_inference_query_ids : (
                        consumed_inference_query_ids + remaining_capacity
                    )
                ]
                consumed_inference_query_ids += len(fill_query_ids)
                batch_query_ids = student_batch_query_ids + fill_query_ids
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=False,
                )
                student_out = {}
                for key in (
                    "semantic_seg",
                    "presence_logit_dec",
                    "pred_masks",
                    "pred_logits",
                ):
                    value = out.get(key, None)
                    if value is not None:
                        student_out[key] = value[: len(student_batch_query_ids)]
                if retain_filtered_raw_student:
                    filtered_raw_student_batch_outputs.append(student_out)
                else:
                    semantic_logits = self._semantic_logits_from_output(
                        student_out,
                        out_size=None,
                    )
                    presence_logits = self._presence_logits_from_output(
                        student_out,
                        batch_size=len(student_batch_query_ids),
                        like=semantic_logits,
                    )
                    student_query_batches.append(semantic_logits)
                    student_presence_batches.append(presence_logits)
                record_teacher_batch(batch_query_ids, out)

        inference_only_query_ids = inference_only_query_ids[
            consumed_inference_query_ids:
        ]
        with torch.no_grad():
            for start in range(
                0,
                len(inference_only_query_ids),
                self.full_query_batch_size,
            ):
                batch_query_ids = inference_only_query_ids[
                    start : start + self.full_query_batch_size
                ]
                out = self._forward_grounding_for_queries(
                    backbone_out_base,
                    query_ids=batch_query_ids,
                    semantic_only=False,
                )
                record_teacher_batch(batch_query_ids, out)

        if any(score is None for score in teacher_scores_by_query):
            raise RuntimeError("teacher forward did not produce every query score")
        if any(score is None for score in teacher_inference_scores_by_query):
            raise RuntimeError(
                "teacher forward did not produce every inference query score"
            )
        if any(score is None for score in teacher_target_scores_by_query):
            raise RuntimeError("teacher forward did not produce every target query score")
        if any(logit is None for logit in teacher_presence_by_query):
            raise RuntimeError("teacher forward did not produce every presence logit")

        return TeacherForwardResult(
            query_scores=torch.stack(
                teacher_scores_by_query,
                dim=0,
            ).unsqueeze(0).float().detach(),
            inference_query_scores=torch.stack(
                teacher_inference_scores_by_query,
                dim=0,
            ).unsqueeze(0).float().detach(),
            target_query_scores=torch.stack(
                teacher_target_scores_by_query,
                dim=0,
            ).unsqueeze(0).float().detach(),
            presence_logits=torch.stack(
                teacher_presence_by_query,
                dim=0,
            ).unsqueeze(0).float().detach(),
            student_query_logits=(
                None
                if retain_filtered_raw_student
                else torch.cat(
                    student_query_batches,
                    dim=0,
                ).unsqueeze(0).float()
            ),
            student_presence_logits=(
                None
                if retain_filtered_raw_student
                else torch.cat(
                    student_presence_batches,
                    dim=0,
                ).unsqueeze(0).float()
            ),
            student_query_ids=resolved_grad_query_ids,
            filtered_raw_student_context=(
                FilteredRawStudentContext(
                    query_ids=resolved_grad_query_ids,
                    batch_outputs=tuple(filtered_raw_student_batch_outputs),
                )
                if retain_filtered_raw_student
                else None
            ),
        )

    def aggregate_class_logits(self, query_logits: torch.Tensor) -> torch.Tensor:
        return aggregate_query_logits(query_logits, self.query_idx_list, self.num_classes)

    def aggregate_presence(self, presence_logits: torch.Tensor) -> torch.Tensor:
        presence_prob = torch.sigmoid(presence_logits.float())
        values = []
        for cls_idx in range(self.num_classes):
            query_ids = [i for i, mapped in enumerate(self.query_idx_list) if int(mapped) == cls_idx]
            if not query_ids:
                raise ValueError(f"class {cls_idx} has no query")
            values.append(presence_prob[:, query_ids].amax(dim=1))
        return torch.stack(values, dim=1)

    def aggregate_presence_logits(self, presence_logits: torch.Tensor) -> torch.Tensor:
        values = []
        for cls_idx in range(self.num_classes):
            query_ids = [i for i, mapped in enumerate(self.query_idx_list) if int(mapped) == cls_idx]
            if not query_ids:
                raise ValueError(f"class {cls_idx} has no query")
            values.append(presence_logits[:, query_ids].float().amax(dim=1))
        return torch.stack(values, dim=1)

    def class_scores_for_mining(
        self,
        query_scores: torch.Tensor,
        presence_logits: torch.Tensor,
        *,
        presence_gate_power: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del presence_gate_power
        class_presence = self.aggregate_presence(presence_logits)
        class_scores = class_scores_from_query_scores(
            query_scores=query_scores,
            query_idx_list=self.query_idx_list,
            num_classes=self.num_classes,
        )
        raw_pred = class_scores.argmax(dim=1)[0].long()
        return class_scores[0], class_presence[0], raw_pred

    def predict(
        self,
        query_scores: torch.Tensor,
        presence_logits: torch.Tensor,
        *,
        presence_gate_power: float,
        prob_thd: float,
        bg_idx: int,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        del presence_gate_power
        class_presence = self.aggregate_presence(presence_logits)
        class_scores = class_scores_from_query_scores(
            query_scores=query_scores,
            query_idx_list=self.query_idx_list,
            num_classes=self.num_classes,
        )
        del class_presence
        return predict_from_class_scores(
            class_scores,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
            out_size=out_size,
        )
