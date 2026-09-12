from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from .boundary_causal_diagnostic import BoundaryCausalAccumulator
from .correction_survival_diagnostic import (
    CanonicalCorrectionSurvivalAccumulator,
)
from .metrics import ConfusionMatrix
from .scores import predict_from_class_scores


_CONFUSION_NAMES = (
    "canonical_pre",
    "canonical_post",
    "f00",
    "f10",
    "f01",
    "f11",
)


@dataclass(frozen=True)
class CrossTimeClassScores:
    pre_canonical: torch.Tensor
    post_canonical: torch.Tensor
    pre_alias: torch.Tensor
    post_alias: torch.Tensor
    has_alias: torch.Tensor
    hybrid_scores: dict[str, torch.Tensor]


def _validate_prompt_mapping(
    *,
    num_queries: int,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    query_mapping = tuple(int(value) for value in query_idx_list)
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    if len(query_mapping) != int(num_queries):
        raise ValueError("query score/mapping length mismatch")
    if not canonical_ids:
        raise ValueError("canonical_query_ids must not be empty")
    if set(query_mapping) != set(range(len(canonical_ids))):
        raise ValueError(
            "query mapping must cover exactly the canonical classes"
        )
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("canonical query ids must be unique")
    for class_id, query_id in enumerate(canonical_ids):
        if query_id < 0 or query_id >= num_queries:
            raise ValueError("canonical query id is outside the score tensor")
        if query_mapping[query_id] != class_id:
            raise ValueError(
                f"canonical query {query_id} does not map to class "
                f"{class_id}"
            )
    return query_mapping, canonical_ids


def _fuse(
    canonical: torch.Tensor,
    alias: torch.Tensor,
    has_alias: torch.Tensor,
) -> torch.Tensor:
    mask = has_alias.view(1, -1, 1, 1)
    return torch.where(mask, torch.maximum(canonical, alias), canonical)


def build_cross_time_class_scores(
    *,
    pre_query_scores: torch.Tensor,
    post_query_scores: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
) -> CrossTimeClassScores:
    if tuple(pre_query_scores.shape) != tuple(post_query_scores.shape):
        raise ValueError("pre/post query score shape mismatch")
    if pre_query_scores.ndim != 4 or int(pre_query_scores.shape[0]) != 1:
        raise ValueError("query scores must have shape [1,Q,H,W]")

    query_mapping, canonical_ids = _validate_prompt_mapping(
        num_queries=int(pre_query_scores.shape[1]),
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
    )
    canonical_index = torch.as_tensor(
        canonical_ids,
        device=pre_query_scores.device,
        dtype=torch.long,
    )
    pre_scores = pre_query_scores.detach().float()
    post_scores = post_query_scores.detach().float()
    pre_canonical = pre_scores.index_select(1, canonical_index)
    post_canonical = post_scores.index_select(1, canonical_index)

    pre_alias_maps = []
    post_alias_maps = []
    has_alias_values = []
    for class_id, canonical_id in enumerate(canonical_ids):
        alias_ids = [
            query_id
            for query_id, mapped_class in enumerate(query_mapping)
            if mapped_class == class_id and query_id != canonical_id
        ]
        has_alias_values.append(bool(alias_ids))
        if alias_ids:
            pre_alias_maps.append(pre_scores[:, alias_ids].amax(dim=1))
            post_alias_maps.append(post_scores[:, alias_ids].amax(dim=1))
        else:
            shape = (
                int(pre_scores.shape[0]),
                int(pre_scores.shape[2]),
                int(pre_scores.shape[3]),
            )
            pre_alias_maps.append(
                torch.full(
                    shape,
                    float("-inf"),
                    device=pre_scores.device,
                    dtype=pre_scores.dtype,
                )
            )
            post_alias_maps.append(
                torch.full(
                    shape,
                    float("-inf"),
                    device=post_scores.device,
                    dtype=post_scores.dtype,
                )
            )

    pre_alias = torch.stack(pre_alias_maps, dim=1)
    post_alias = torch.stack(post_alias_maps, dim=1)
    has_alias = torch.as_tensor(
        has_alias_values,
        device=pre_scores.device,
        dtype=torch.bool,
    )
    hybrid_scores = {
        "f00": _fuse(pre_canonical, pre_alias, has_alias),
        "f10": _fuse(post_canonical, pre_alias, has_alias),
        "f01": _fuse(pre_canonical, post_alias, has_alias),
        "f11": _fuse(post_canonical, post_alias, has_alias),
    }
    return CrossTimeClassScores(
        pre_canonical=pre_canonical,
        post_canonical=post_canonical,
        pre_alias=pre_alias,
        post_alias=post_alias,
        has_alias=has_alias,
        hybrid_scores=hybrid_scores,
    )


def _metric_values(
    confusion: ConfusionMatrix,
) -> tuple[float, torch.Tensor]:
    summary = confusion.summary()
    return (
        float(summary.miou * 100.0),
        torch.as_tensor(
            summary.per_class_iou,
            dtype=torch.float64,
        )
        * 100.0,
    )


def _new_confusion(
    *,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> ConfusionMatrix:
    return ConfusionMatrix(
        num_classes,
        ignore_index=ignore_index,
        device=device,
    )


def _causal_arithmetic(
    values: dict[str, float | torch.Tensor],
) -> dict[str, float | torch.Tensor]:
    canonical_direct = (
        values["canonical_post"] - values["canonical_pre"]
    )
    canonical_visible = values["f10"] - values["f00"]
    alias_only_transfer = values["f01"] - values["f00"]
    interaction = (
        values["f11"]
        - values["f10"]
        - values["f01"]
        + values["f00"]
    )
    static_alias_shielding = canonical_visible - canonical_direct
    total = values["f11"] - values["f00"]
    canonical_shapley = 0.5 * (
        (values["f10"] - values["f00"])
        + (values["f11"] - values["f01"])
    )
    alias_shapley = 0.5 * (
        (values["f01"] - values["f00"])
        + (values["f11"] - values["f10"])
    )
    identity_error = total - (
        canonical_visible + alias_only_transfer + interaction
    )
    return {
        "canonical_direct": canonical_direct,
        "canonical_visible": canonical_visible,
        "static_alias_shielding": static_alias_shielding,
        "alias_only_transfer": alias_only_transfer,
        "interaction": interaction,
        "total_synonym_tta_gain": total,
        "canonical_shapley": canonical_shapley,
        "alias_shapley": alias_shapley,
        "identity_error": identity_error,
    }


class CrossTimePromptCausalAccumulator:
    def __init__(
        self,
        *,
        query_idx_list: Sequence[int],
        canonical_query_ids: Sequence[int],
        num_classes: int,
        prob_thd: float,
        bg_idx: int,
        device: torch.device,
        ignore_index: int = 255,
        image_row_limit: int = 200,
        query_words: Sequence[str] | None = None,
        boundary_radii: Sequence[int] = (),
    ) -> None:
        if int(num_classes) != len(canonical_query_ids):
            raise ValueError(
                "num_classes must match canonical_query_ids length"
            )
        if int(image_row_limit) < 0:
            raise ValueError("image_row_limit must be non-negative")
        query_mapping, canonical_ids = _validate_prompt_mapping(
            num_queries=len(query_idx_list),
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_query_ids,
        )
        self.query_idx_list = query_mapping
        self.canonical_query_ids = canonical_ids
        self.num_classes = int(num_classes)
        self.prob_thd = float(prob_thd)
        self.bg_idx = int(bg_idx)
        self.device = torch.device(device)
        self.ignore_index = int(ignore_index)
        self.image_row_limit = int(image_row_limit)
        self._processed = torch.zeros(
            2,
            device=self.device,
            dtype=torch.long,
        )
        self._confusions = {
            name: _new_confusion(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                device=self.device,
            )
            for name in _CONFUSION_NAMES
        }
        self._sample_ids: list[str] = []
        self._image_outliers: dict[str, list[dict]] = {
            "static_alias_shielding": [],
            "alias_only_transfer": [],
            "interaction": [],
        }
        self._boundary_overlap = None
        self._correction_survival = None
        if boundary_radii:
            if query_words is None:
                raise ValueError(
                    "query_words are required for boundary diagnostics"
                )
            words = tuple(str(value) for value in query_words)
            if len(words) != len(self.query_idx_list):
                raise ValueError(
                    "query_words length must match query_idx_list"
                )
            class_prompts = tuple(
                tuple(
                    words[query_id]
                    for query_id, mapped_class in enumerate(
                        self.query_idx_list
                    )
                    if mapped_class == class_id
                )
                for class_id in range(self.num_classes)
            )
            class_names = tuple(
                words[query_id]
                for query_id in self.canonical_query_ids
            )
            self._boundary_overlap = BoundaryCausalAccumulator(
                num_classes=self.num_classes,
                class_names=class_names,
                class_prompts=class_prompts,
                boundary_radii=boundary_radii,
                device=self.device,
                ignore_index=self.ignore_index,
            )
            self._correction_survival = (
                CanonicalCorrectionSurvivalAccumulator(
                    num_classes=self.num_classes,
                    class_names=class_names,
                    query_words=words,
                    query_idx_list=self.query_idx_list,
                    canonical_query_ids=self.canonical_query_ids,
                    boundary_radii=boundary_radii,
                    device=self.device,
                    ignore_index=self.ignore_index,
                )
            )

    def _predict(
        self,
        class_scores: torch.Tensor,
        *,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        return predict_from_class_scores(
            class_scores,
            prob_thd=self.prob_thd,
            bg_idx=self.bg_idx,
            out_size=tuple(gt.shape[-2:]),
        )

    def _image_metrics(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        gt: torch.Tensor,
    ) -> dict[str, float]:
        output = {}
        for name, pred in predictions.items():
            confusion = _new_confusion(
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
                device=self.device,
            )
            confusion.update(pred, gt)
            output[name] = _metric_values(confusion)[0]
        return output

    def _append_image_outlier(
        self,
        metric_name: str,
        row: dict,
    ) -> None:
        if self.image_row_limit == 0:
            return
        rows = [*self._image_outliers[metric_name], row]
        self._image_outliers[metric_name] = sorted(
            rows,
            key=lambda value: abs(
                float(value[f"{metric_name}_miou"])
            ),
            reverse=True,
        )[: self.image_row_limit]

    def update_image(
        self,
        sample_id: str,
        *,
        pre_query_scores: torch.Tensor,
        post_query_scores: torch.Tensor,
        gt: torch.Tensor,
        adapted: bool,
    ) -> None:
        views = build_cross_time_class_scores(
            pre_query_scores=pre_query_scores,
            post_query_scores=post_query_scores,
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
        )
        scores = {
            "canonical_pre": views.pre_canonical,
            "canonical_post": views.post_canonical,
            **views.hybrid_scores,
        }
        predictions = {
            name: self._predict(class_scores, gt=gt)
            for name, class_scores in scores.items()
        }
        for name, prediction in predictions.items():
            self._confusions[name].update(prediction, gt)

        if self._boundary_overlap is not None:
            self._boundary_overlap.update_image(
                sample_id,
                scores={
                    "c0": views.pre_canonical,
                    "ct": views.post_canonical,
                    "a0": views.pre_alias,
                    "at": views.post_alias,
                    "s0": views.hybrid_scores["f00"],
                    "s10": views.hybrid_scores["f10"],
                    "s01": views.hybrid_scores["f01"],
                    "st": views.hybrid_scores["f11"],
                },
                predictions={
                    "p0": predictions["canonical_pre"],
                    "s0": predictions["f00"],
                    "pt": predictions["canonical_post"],
                    "s10": predictions["f10"],
                    "s01": predictions["f01"],
                    "st": predictions["f11"],
                },
                gt=gt,
                adapted=adapted,
            )
        if self._correction_survival is not None:
            self._correction_survival.update_image(
                sample_id,
                scores={
                    "c0": views.pre_canonical,
                    "ct": views.post_canonical,
                    "a0": views.pre_alias,
                    "at": views.post_alias,
                    "s0": views.hybrid_scores["f00"],
                    "s10": views.hybrid_scores["f10"],
                    "st": views.hybrid_scores["f11"],
                },
                predictions={
                    "p0": predictions["canonical_pre"],
                    "s0": predictions["f00"],
                    "pt": predictions["canonical_post"],
                    "s10": predictions["f10"],
                    "st": predictions["f11"],
                },
                pre_query_scores=pre_query_scores,
                post_query_scores=post_query_scores,
                gt=gt,
                adapted=adapted,
            )

        if self.image_row_limit > 0:
            image_metrics = self._image_metrics(predictions, gt=gt)
            image_causal = _causal_arithmetic(image_metrics)
            row = {
                "sample_id": str(sample_id),
                "adapted": bool(adapted),
                "f00_miou": image_metrics["f00"],
                "f10_miou": image_metrics["f10"],
                "f01_miou": image_metrics["f01"],
                "f11_miou": image_metrics["f11"],
                "canonical_direct_miou": float(
                    image_causal["canonical_direct"]
                ),
                "canonical_visible_miou": float(
                    image_causal["canonical_visible"]
                ),
                "static_alias_shielding_miou": float(
                    image_causal["static_alias_shielding"]
                ),
                "alias_only_transfer_miou": float(
                    image_causal["alias_only_transfer"]
                ),
                "interaction_miou": float(image_causal["interaction"]),
            }
            for metric_name in self._image_outliers:
                self._append_image_outlier(metric_name, dict(row))

        self._processed += torch.tensor(
            [1, int(bool(adapted))],
            device=self.device,
            dtype=torch.long,
        )
        self._sample_ids.append(str(sample_id))

    def reduce_distributed(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        metadata = (
            self.query_idx_list,
            self.canonical_query_ids,
            self.num_classes,
            self.prob_thd,
            self.bg_idx,
        )
        world_size = dist.get_world_size()
        gathered_metadata = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metadata, metadata)
        if any(value != metadata for value in gathered_metadata):
            raise ValueError(
                "cross-time causal metadata differs across ranks"
            )

        dist.all_reduce(self._processed)
        for confusion in self._confusions.values():
            dist.all_reduce(confusion.matrix)

        gathered_ids = [None for _ in range(world_size)]
        gathered_outliers = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_ids, self._sample_ids)
        dist.all_gather_object(gathered_outliers, self._image_outliers)
        self._sample_ids = [
            str(sample_id)
            for rank_ids in gathered_ids
            for sample_id in (rank_ids or [])
        ]
        self._image_outliers = {
            name: []
            for name in self._image_outliers
        }
        for rank_outliers in gathered_outliers:
            for metric_name, rows in (rank_outliers or {}).items():
                if metric_name not in self._image_outliers:
                    raise ValueError(
                        "cross-time outlier fields differ across ranks"
                    )
                for row in rows:
                    self._append_image_outlier(
                        metric_name,
                        dict(row),
                    )
        if self._boundary_overlap is not None:
            self._boundary_overlap.reduce_distributed()
        if self._correction_survival is not None:
            self._correction_survival.reduce_distributed()

    def _validate_parity(
        self,
        *,
        expected_f00: torch.Tensor | None,
        expected_f11: torch.Tensor | None,
        expected_canonical_pre: torch.Tensor | None,
        expected_canonical_post: torch.Tensor | None,
    ) -> dict[str, bool]:
        expected = (
            (
                "f00",
                expected_f00,
                "f00_matches_pre_raw",
            ),
            (
                "f11",
                expected_f11,
                "f11_matches_post_raw",
            ),
            (
                "canonical_pre",
                expected_canonical_pre,
                "canonical_pre_matches",
            ),
            (
                "canonical_post",
                expected_canonical_post,
                "canonical_post_matches",
            ),
        )
        parity = {}
        for confusion_name, reference, report_name in expected:
            if reference is None:
                continue
            matches = torch.equal(
                self._confusions[confusion_name].matrix,
                reference.to(
                    device=self.device,
                    dtype=torch.long,
                ),
            )
            if not matches:
                raise ValueError(
                    f"cross-time {confusion_name} head parity failed"
                )
            parity[report_name] = True
        return parity

    def finalize(
        self,
        *,
        expected_f00: torch.Tensor | None = None,
        expected_f11: torch.Tensor | None = None,
        expected_canonical_pre: torch.Tensor | None = None,
        expected_canonical_post: torch.Tensor | None = None,
    ) -> dict:
        parity = self._validate_parity(
            expected_f00=expected_f00,
            expected_f11=expected_f11,
            expected_canonical_pre=expected_canonical_pre,
            expected_canonical_post=expected_canonical_post,
        )
        head_miou = {}
        head_per_class = {}
        heads = {}
        for name, confusion in self._confusions.items():
            miou, per_class = _metric_values(confusion)
            head_miou[name] = miou
            head_per_class[name] = per_class
            heads[name] = {
                "miou": miou,
                "per_class_iou": per_class.tolist(),
                "confusion": confusion.matrix.detach().cpu().tolist(),
            }

        causal_miou = _causal_arithmetic(head_miou)
        causal_per_class = _causal_arithmetic(head_per_class)
        causal = {}
        for name, value in causal_miou.items():
            causal[f"{name}_miou"] = float(value)
            causal[f"{name}_per_class_iou"] = (
                causal_per_class[name].detach().cpu().tolist()
            )

        sample_count = len(self._sample_ids)
        unique_count = len(set(self._sample_ids))
        report = {
            "metric_scale": "percent",
            "processed": int(self._processed[0].item()),
            "adapted": int(self._processed[1].item()),
            "unique_samples": int(unique_count),
            "duplicate_sample_ids": int(sample_count - unique_count),
            "parity": parity,
            "heads": heads,
            "causal": causal,
            "image_outliers": {
                name: list(rows)
                for name, rows in self._image_outliers.items()
            },
        }
        if self._boundary_overlap is not None:
            report["boundary_overlap"] = (
                self._boundary_overlap.finalize()
            )
        if self._correction_survival is not None:
            report["correction_survival"] = (
                self._correction_survival.finalize()
            )
        return report
