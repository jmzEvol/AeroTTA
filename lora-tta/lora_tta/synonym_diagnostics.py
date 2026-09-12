from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist


_COUNT_FIELDS = (
    "winner_pixels",
    "own_class_winner_pixels",
    "foreign_class_winner_pixels",
    "rescue_pixels",
    "harm_pixels",
    "lateral_pixels",
)
_RESIDUAL_FIELDS = (
    "own_residual_sum",
    "own_residual_count",
    "foreign_residual_sum",
    "foreign_residual_count",
)


@dataclass(frozen=True)
class AliasSnapshotEffects:
    query_id: int
    class_id: int
    winner_pixels: int
    own_class_winner_pixels: int
    foreign_class_winner_pixels: int
    rescue_pixels: int
    harm_pixels: int
    lateral_pixels: int
    steal_matrix: torch.Tensor
    own_residual_sum: float
    own_residual_count: int
    foreign_residual_sum: float
    foreign_residual_count: int
    leave_one_out_confusion: torch.Tensor


@dataclass(frozen=True)
class SnapshotSynonymEffects:
    canonical_confusion: torch.Tensor
    synonym_confusion: torch.Tensor
    aliases: dict[int, AliasSnapshotEffects]


def _confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    pred = pred.long()
    target = target.to(device=pred.device, dtype=torch.long)
    valid = (
        (target != int(ignore_index))
        & (target >= 0)
        & (target < int(num_classes))
    )
    if not bool(valid.any()):
        return torch.zeros(
            (int(num_classes), int(num_classes)),
            device=pred.device,
            dtype=torch.long,
        )
    flat = (
        target[valid] * int(num_classes)
        + pred[valid].clamp(0, int(num_classes) - 1)
    )
    return torch.bincount(
        flat,
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))


def _prediction_from_class_scores(
    class_scores: torch.Tensor,
    *,
    prob_thd: float,
    bg_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_scores, natural_pred = class_scores.max(dim=0)
    pred = natural_pred.long().clone()
    pred[max_scores < float(prob_thd)] = int(bg_idx)
    return pred, natural_pred.long(), max_scores


def _query_ids_by_class(
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    num_classes = len(canonical_query_ids)
    output = []
    for class_id, canonical_query_id in enumerate(canonical_query_ids):
        query_ids = [
            query_id
            for query_id, mapped_class in enumerate(query_idx_list)
            if int(mapped_class) == int(class_id)
        ]
        if int(canonical_query_id) not in query_ids:
            raise ValueError(
                f"canonical query {canonical_query_id} does not map to class {class_id}"
            )
        output.append(
            (
                int(canonical_query_id),
                *(
                    int(query_id)
                    for query_id in query_ids
                    if int(query_id) != int(canonical_query_id)
                ),
            )
        )
    return tuple(output)


def snapshot_synonym_effects(
    *,
    query_scores: torch.Tensor,
    gt: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
    prob_thd: float,
    bg_idx: int,
    ignore_index: int = 255,
) -> SnapshotSynonymEffects:
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError(
            f"expected query scores [1,Q,H,W], got {tuple(query_scores.shape)}"
        )
    if len(query_idx_list) != int(query_scores.shape[1]):
        raise ValueError("query score/mapping length mismatch")
    if not canonical_query_ids:
        raise ValueError("canonical query ids must not be empty")
    if tuple(gt.shape[-2:]) != tuple(query_scores.shape[-2:]):
        raise ValueError(
            "ground-truth/query score shape mismatch: "
            f"{tuple(gt.shape[-2:])} vs {tuple(query_scores.shape[-2:])}"
        )

    scores = query_scores[0].detach().float()
    gt = gt.to(device=scores.device, dtype=torch.long)
    num_classes = len(canonical_query_ids)
    query_ids_by_class = _query_ids_by_class(
        query_idx_list,
        canonical_query_ids,
    )
    canonical_index = torch.tensor(
        tuple(int(value) for value in canonical_query_ids),
        device=scores.device,
        dtype=torch.long,
    )
    canonical_scores = scores.index_select(0, canonical_index)
    synonym_scores = []
    winner_queries = []
    for query_ids in query_ids_by_class:
        local = scores[list(query_ids)]
        class_scores, winner_local = local.max(dim=0)
        global_ids = torch.tensor(
            query_ids,
            device=scores.device,
            dtype=torch.long,
        )
        synonym_scores.append(class_scores)
        winner_queries.append(global_ids[winner_local])
    synonym_scores_tensor = torch.stack(synonym_scores)
    winner_query_tensor = torch.stack(winner_queries)

    canonical_pred, _, _ = _prediction_from_class_scores(
        canonical_scores,
        prob_thd=prob_thd,
        bg_idx=bg_idx,
    )
    synonym_pred, natural_synonym_pred, synonym_max_scores = (
        _prediction_from_class_scores(
            synonym_scores_tensor,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
        )
    )
    canonical_confusion = _confusion_matrix(
        canonical_pred,
        gt,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    synonym_confusion = _confusion_matrix(
        synonym_pred,
        gt,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    valid = (
        (gt != int(ignore_index))
        & (gt >= 0)
        & (gt < int(num_classes))
    )
    confidently_attributed = synonym_max_scores >= float(prob_thd)

    aliases: dict[int, AliasSnapshotEffects] = {}
    for class_id, query_ids in enumerate(query_ids_by_class):
        canonical_query_id = int(canonical_query_ids[class_id])
        canonical_map = scores[canonical_query_id]
        for query_id in query_ids:
            query_id = int(query_id)
            if query_id == canonical_query_id:
                continue
            winning_final = (
                valid
                & confidently_attributed
                & (natural_synonym_pred == int(class_id))
                & (winner_query_tensor[class_id] == query_id)
            )
            own_winner = winning_final & (gt == int(class_id))
            foreign_winner = winning_final & (gt != int(class_id))
            rescue = (
                winning_final
                & (canonical_pred != gt)
                & (synonym_pred == gt)
            )
            harm = (
                winning_final
                & (canonical_pred == gt)
                & (synonym_pred != gt)
            )
            lateral = (
                winning_final
                & (canonical_pred != gt)
                & (synonym_pred != gt)
                & (canonical_pred != synonym_pred)
            )
            steal_matrix = _confusion_matrix(
                synonym_pred,
                gt,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            steal_matrix = steal_matrix * 0
            if bool(foreign_winner.any()):
                flat = (
                    gt[foreign_winner] * int(num_classes)
                    + synonym_pred[foreign_winner]
                )
                steal_matrix += torch.bincount(
                    flat,
                    minlength=int(num_classes) * int(num_classes),
                ).reshape(int(num_classes), int(num_classes))

            residual = (scores[query_id] - canonical_map).clamp_min(0.0)
            own_region = valid & (gt == int(class_id))
            foreign_region = valid & (gt != int(class_id))

            remaining_query_ids = tuple(
                candidate
                for candidate in query_ids
                if int(candidate) != query_id
            )
            leave_one_out_scores = synonym_scores_tensor.clone()
            leave_one_out_scores[class_id] = scores[
                list(remaining_query_ids)
            ].amax(dim=0)
            leave_one_out_pred, _, _ = _prediction_from_class_scores(
                leave_one_out_scores,
                prob_thd=prob_thd,
                bg_idx=bg_idx,
            )
            aliases[query_id] = AliasSnapshotEffects(
                query_id=query_id,
                class_id=int(class_id),
                winner_pixels=int(winning_final.sum().item()),
                own_class_winner_pixels=int(own_winner.sum().item()),
                foreign_class_winner_pixels=int(foreign_winner.sum().item()),
                rescue_pixels=int(rescue.sum().item()),
                harm_pixels=int(harm.sum().item()),
                lateral_pixels=int(lateral.sum().item()),
                steal_matrix=steal_matrix,
                own_residual_sum=float(residual[own_region].sum().item()),
                own_residual_count=int(own_region.sum().item()),
                foreign_residual_sum=float(
                    residual[foreign_region].sum().item()
                ),
                foreign_residual_count=int(foreign_region.sum().item()),
                leave_one_out_confusion=_confusion_matrix(
                    leave_one_out_pred,
                    gt,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                ),
            )

    return SnapshotSynonymEffects(
        canonical_confusion=canonical_confusion,
        synonym_confusion=synonym_confusion,
        aliases=aliases,
    )


def _metric_summary(confusion: torch.Tensor) -> tuple[float, list[float]]:
    values = confusion.detach().float()
    true_positive = torch.diag(values)
    union = values.sum(dim=1) + values.sum(dim=0) - true_positive
    valid = union > 0
    iou = torch.zeros_like(true_positive)
    iou[valid] = true_positive[valid] / union[valid].clamp_min(1.0)
    miou = iou[valid].mean() if bool(valid.any()) else iou.new_tensor(0.0)
    return (
        float(miou.item() * 100.0),
        [float(value * 100.0) for value in iou.detach().cpu().tolist()],
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


class SynonymDiagnosticAccumulator:
    def __init__(
        self,
        *,
        query_words: Sequence[str],
        query_idx_list: Sequence[int],
        canonical_query_ids: Sequence[int],
        prob_thd: float,
        bg_idx: int,
        device: torch.device,
        ignore_index: int = 255,
    ):
        self.query_words = tuple(str(value) for value in query_words)
        self.query_idx_list = tuple(int(value) for value in query_idx_list)
        self.canonical_query_ids = tuple(
            int(value) for value in canonical_query_ids
        )
        self.num_classes = len(self.canonical_query_ids)
        self.prob_thd = float(prob_thd)
        self.bg_idx = int(bg_idx)
        self.ignore_index = int(ignore_index)
        self.device = torch.device(device)
        if len(self.query_words) != len(self.query_idx_list):
            raise ValueError("query word/mapping length mismatch")

        canonical_set = set(self.canonical_query_ids)
        self.alias_query_ids = tuple(
            query_id
            for query_id in range(len(self.query_words))
            if query_id not in canonical_set
        )
        self._snapshots = {
            name: self._new_snapshot_aggregate()
            for name in ("pre", "post")
        }
        self._image_rows: list[dict] = []
        self._counts = torch.zeros(
            2,
            device=self.device,
            dtype=torch.long,
        )

    def _new_snapshot_aggregate(self) -> dict:
        matrix_shape = (self.num_classes, self.num_classes)
        return {
            "canonical_confusion": torch.zeros(
                matrix_shape,
                device=self.device,
                dtype=torch.long,
            ),
            "synonym_confusion": torch.zeros(
                matrix_shape,
                device=self.device,
                dtype=torch.long,
            ),
            "aliases": {
                query_id: {
                    "counts": torch.zeros(
                        len(_COUNT_FIELDS),
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "residuals": torch.zeros(
                        len(_RESIDUAL_FIELDS),
                        device=self.device,
                        dtype=torch.float64,
                    ),
                    "steal_matrix": torch.zeros(
                        matrix_shape,
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "leave_one_out_confusion": torch.zeros(
                        matrix_shape,
                        device=self.device,
                        dtype=torch.long,
                    ),
                }
                for query_id in self.alias_query_ids
            },
        }

    def _update_snapshot(
        self,
        name: str,
        effects: SnapshotSynonymEffects,
    ) -> None:
        target = self._snapshots[name]
        target["canonical_confusion"] += effects.canonical_confusion
        target["synonym_confusion"] += effects.synonym_confusion
        for query_id, source in effects.aliases.items():
            alias = target["aliases"][query_id]
            alias["counts"] += torch.tensor(
                [getattr(source, field) for field in _COUNT_FIELDS],
                device=self.device,
                dtype=torch.long,
            )
            alias["residuals"] += torch.tensor(
                [getattr(source, field) for field in _RESIDUAL_FIELDS],
                device=self.device,
                dtype=torch.float64,
            )
            alias["steal_matrix"] += source.steal_matrix
            alias[
                "leave_one_out_confusion"
            ] += source.leave_one_out_confusion

    def _snapshot_effects(
        self,
        query_scores: torch.Tensor,
        gt: torch.Tensor,
    ) -> SnapshotSynonymEffects:
        return snapshot_synonym_effects(
            query_scores=query_scores,
            gt=gt,
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
            prob_thd=self.prob_thd,
            bg_idx=self.bg_idx,
            ignore_index=self.ignore_index,
        )

    def update_image(
        self,
        sample_id: str,
        pre_query_scores: torch.Tensor,
        post_query_scores: torch.Tensor,
        gt: torch.Tensor,
        *,
        adapted: bool,
    ) -> None:
        pre = self._snapshot_effects(pre_query_scores, gt)
        post = self._snapshot_effects(post_query_scores, gt)
        self._update_snapshot("pre", pre)
        self._update_snapshot("post", post)
        self._counts[0] += 1
        self._counts[1] += int(bool(adapted))

        pre_canonical_miou, _ = _metric_summary(pre.canonical_confusion)
        pre_synonym_miou, _ = _metric_summary(pre.synonym_confusion)
        post_canonical_miou, _ = _metric_summary(post.canonical_confusion)
        post_synonym_miou, _ = _metric_summary(post.synonym_confusion)
        pre_advantage = pre_synonym_miou - pre_canonical_miou
        post_advantage = post_synonym_miou - post_canonical_miou
        self._image_rows.append(
            {
                "sample_id": str(sample_id),
                "adapted": bool(adapted),
                "pre_canonical_miou": pre_canonical_miou,
                "pre_synonym_miou": pre_synonym_miou,
                "post_canonical_miou": post_canonical_miou,
                "post_synonym_miou": post_synonym_miou,
                "canonical_tta_delta_miou": (
                    post_canonical_miou - pre_canonical_miou
                ),
                "synonym_tta_delta_miou": (
                    post_synonym_miou - pre_synonym_miou
                ),
                "pre_alias_advantage_miou": pre_advantage,
                "post_alias_advantage_miou": post_advantage,
                "alias_drift_miou": post_advantage - pre_advantage,
            }
        )
        if len(self._image_rows) > 200:
            ordered = sorted(
                self._image_rows,
                key=lambda row: float(row["alias_drift_miou"]),
            )
            self._image_rows = ordered[:100] + ordered[-100:]

    def reduce_distributed(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        dist.all_reduce(self._counts)
        for snapshot in self._snapshots.values():
            dist.all_reduce(snapshot["canonical_confusion"])
            dist.all_reduce(snapshot["synonym_confusion"])
            for alias in snapshot["aliases"].values():
                dist.all_reduce(alias["counts"])
                dist.all_reduce(alias["residuals"])
                dist.all_reduce(alias["steal_matrix"])
                dist.all_reduce(alias["leave_one_out_confusion"])
        gathered_rows: list[list[dict] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_rows, self._image_rows)
        self._image_rows = [
            row
            for rank_rows in gathered_rows
            for row in (rank_rows or [])
        ]

    def _snapshot_result(self, name: str) -> dict:
        snapshot = self._snapshots[name]
        canonical_miou, canonical_per_class = _metric_summary(
            snapshot["canonical_confusion"]
        )
        synonym_miou, synonym_per_class = _metric_summary(
            snapshot["synonym_confusion"]
        )
        return {
            "canonical_miou": canonical_miou,
            "synonym_miou": synonym_miou,
            "alias_advantage_miou": synonym_miou - canonical_miou,
            "canonical_per_class_iou": canonical_per_class,
            "synonym_per_class_iou": synonym_per_class,
            "alias_advantage_per_class_iou": [
                synonym - canonical
                for canonical, synonym in zip(
                    canonical_per_class,
                    synonym_per_class,
                )
            ],
            "canonical_confusion": snapshot[
                "canonical_confusion"
            ].detach().cpu().tolist(),
            "synonym_confusion": snapshot[
                "synonym_confusion"
            ].detach().cpu().tolist(),
        }

    def _alias_phase_result(
        self,
        phase: str,
        query_id: int,
        *,
        synonym_miou: float,
    ) -> tuple[dict, float]:
        aggregate = self._snapshots[phase]["aliases"][query_id]
        counts = [
            int(value)
            for value in aggregate["counts"].detach().cpu().tolist()
        ]
        count_fields = dict(zip(_COUNT_FIELDS, counts))
        residuals = [
            float(value)
            for value in aggregate["residuals"].detach().cpu().tolist()
        ]
        residual_fields = dict(zip(_RESIDUAL_FIELDS, residuals))
        own_count = int(residual_fields["own_residual_count"])
        foreign_count = int(residual_fields["foreign_residual_count"])
        leave_one_out_miou, leave_one_out_per_class = _metric_summary(
            aggregate["leave_one_out_confusion"]
        )
        output = {
            **count_fields,
            "rescue_to_harm_ratio": _ratio(
                count_fields["rescue_pixels"],
                count_fields["harm_pixels"],
            ),
            "own_residual_mean": (
                residual_fields["own_residual_sum"] / own_count
                if own_count
                else None
            ),
            "foreign_residual_mean": (
                residual_fields["foreign_residual_sum"] / foreign_count
                if foreign_count
                else None
            ),
            "steal_matrix": aggregate[
                "steal_matrix"
            ].detach().cpu().tolist(),
            "leave_one_out_miou": leave_one_out_miou,
            "leave_one_out_per_class_iou": leave_one_out_per_class,
            "leave_one_out_effect_miou": (
                synonym_miou - leave_one_out_miou
            ),
        }
        return output, synonym_miou - leave_one_out_miou

    def finalize(self, *, outlier_limit: int = 30) -> dict:
        pre = self._snapshot_result("pre")
        post = self._snapshot_result("post")
        aliases = []
        for query_id in self.alias_query_ids:
            pre_phase, pre_leave_one_out_effect = self._alias_phase_result(
                "pre",
                query_id,
                synonym_miou=pre["synonym_miou"],
            )
            post_phase, post_leave_one_out_effect = self._alias_phase_result(
                "post",
                query_id,
                synonym_miou=post["synonym_miou"],
            )
            aliases.append(
                {
                    "query_id": int(query_id),
                    "query": self.query_words[query_id],
                    "class_id": int(self.query_idx_list[query_id]),
                    "canonical_query_id": int(
                        self.canonical_query_ids[
                            self.query_idx_list[query_id]
                        ]
                    ),
                    "canonical_query": self.query_words[
                        self.canonical_query_ids[
                            self.query_idx_list[query_id]
                        ]
                    ],
                    "pre": pre_phase,
                    "post": post_phase,
                    "leave_one_out": {
                        "pre_effect_miou": pre_leave_one_out_effect,
                        "post_effect_miou": post_leave_one_out_effect,
                        "effect_drift_miou": (
                            post_leave_one_out_effect
                            - pre_leave_one_out_effect
                        ),
                    },
                }
            )
        aliases.sort(
            key=lambda row: float(
                row["leave_one_out"]["post_effect_miou"]
            )
        )

        ordered_rows = sorted(
            self._image_rows,
            key=lambda row: float(row["alias_drift_miou"]),
        )
        limit = max(int(outlier_limit), 0)
        if limit:
            outliers = ordered_rows[:limit] + ordered_rows[-limit:]
            deduplicated = {
                (row["sample_id"], row["adapted"]): row
                for row in outliers
            }
            outliers = sorted(
                deduplicated.values(),
                key=lambda row: abs(float(row["alias_drift_miou"])),
                reverse=True,
            )
        else:
            outliers = []

        return {
            "metric_scale": "percent",
            "processed": int(self._counts[0].item()),
            "adapted": int(self._counts[1].item()),
            "class_names": [
                self.query_words[query_id]
                for query_id in self.canonical_query_ids
            ],
            "pre": pre,
            "post": post,
            "canonical_tta_delta_miou": (
                post["canonical_miou"] - pre["canonical_miou"]
            ),
            "synonym_tta_delta_miou": (
                post["synonym_miou"] - pre["synonym_miou"]
            ),
            "alias_drift_miou": (
                post["alias_advantage_miou"]
                - pre["alias_advantage_miou"]
            ),
            "alias_drift_per_class_iou": [
                post_value - pre_value
                for pre_value, post_value in zip(
                    pre["alias_advantage_per_class_iou"],
                    post["alias_advantage_per_class_iou"],
                )
            ],
            "aliases": aliases,
            "image_outliers": outliers,
        }
