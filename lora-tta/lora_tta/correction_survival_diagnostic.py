from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .boundary_causal_diagnostic import (
    REGION_NAMES,
    build_class_regions,
    finalize_moment,
    normalize_boundary_radii,
)


SCHEMA_VERSION = 1
SCORE_EPSILON = 1e-6
SCOPE_NAMES = ("all", "adapted_only")
CORRECTION_FATE_NAMES = (
    "correction_pre_solved_retained",
    "correction_pre_solved_degraded",
    "correction_visible_preserved",
    "correction_dynamic_overwrite",
    "correction_dynamic_recovery",
    "correction_static_unresolved",
)
COLLAPSE_FATE_NAMES = (
    "collapse_static_retained",
    "collapse_protection_lost",
    "collapse_dynamic_protection",
    "collapse_unprotected",
)
SURVIVAL_EVENT_NAMES = (
    "canonical_correction",
    *CORRECTION_FATE_NAMES,
    "canonical_damage",
    *COLLAPSE_FATE_NAMES,
    "canonical_fp_removed",
    "fp_reintroduced",
)
SUPPRESSION_COUNT_NAMES = (
    "canonical_suppressed",
    "alias_down",
    "alias_flat",
    "alias_up",
)
SUPPRESSION_FEATURE_NAMES = (
    "canonical_delta",
    "alias_delta",
    "synonym_delta",
)
PROMPT_RISK_NAMES = (
    "dynamic_overwrite",
    "foreign_expansion",
    "true_class_withdrawal",
    "dynamic_rescue",
    "fp_reintroduction",
)
TRANSITION_COUNT_NAMES = (
    "dynamic_overwrite",
    "foreign_expansion",
    "true_class_withdrawal",
    "mixed_conflict",
)
_PREDICTION_NAMES = ("p0", "s0", "pt", "s10", "st")
_SCORE_NAMES = ("c0", "ct", "a0", "at", "s0", "s10", "st")


def build_survival_event_masks(
    gt: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    class_id: int,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    if set(predictions) != set(_PREDICTION_NAMES):
        raise ValueError(
            "predictions must contain p0, s0, pt, s10, and st"
        )
    if any(tuple(value.shape) != tuple(gt.shape) for value in predictions.values()):
        raise ValueError("prediction/gt shape mismatch")

    class_id = int(class_id)
    valid = gt != int(ignore_index)
    positive = valid & (gt == class_id)
    negative = valid & (gt != class_id)
    p0 = predictions["p0"]
    s0 = predictions["s0"]
    pt = predictions["pt"]
    s10 = predictions["s10"]
    st = predictions["st"]

    correction = positive & (p0 != class_id) & (pt == class_id)
    pre_solved = correction & (s0 == class_id)
    not_pre_solved = correction & (s0 != class_id)
    visible = not_pre_solved & (s10 == class_id)
    blocked = not_pre_solved & (s10 != class_id)

    damage = positive & (p0 == class_id) & (pt != class_id)
    static_protected = damage & (s10 == class_id)
    not_static_protected = damage & (s10 != class_id)

    fp_removed = negative & (p0 == class_id) & (pt != class_id)
    output = {
        "canonical_correction": correction,
        "correction_pre_solved_retained": pre_solved & (st == class_id),
        "correction_pre_solved_degraded": pre_solved & (st != class_id),
        "correction_visible_preserved": visible & (st == class_id),
        "correction_dynamic_overwrite": visible & (st != class_id),
        "correction_dynamic_recovery": blocked & (st == class_id),
        "correction_static_unresolved": blocked & (st != class_id),
        "canonical_damage": damage,
        "collapse_static_retained": static_protected & (st == class_id),
        "collapse_protection_lost": static_protected & (st != class_id),
        "collapse_dynamic_protection": not_static_protected & (st == class_id),
        "collapse_unprotected": not_static_protected & (st != class_id),
        "canonical_fp_removed": fp_removed,
        "fp_reintroduced": fp_removed & (st == class_id),
    }
    return {name: output[name] for name in SURVIVAL_EVENT_NAMES}


def _assert_partition(
    name: str,
    whole: torch.Tensor,
    parts: Sequence[torch.Tensor],
) -> None:
    union = torch.zeros_like(whole)
    count = 0
    for part in parts:
        union |= part
        count += int(part.sum().item())
    if not torch.equal(whole, union) or count != int(whole.sum().item()):
        raise ValueError(f"correction survival {name} partition failed")


def _resize_class_scores(
    value: torch.Tensor,
    *,
    num_classes: int,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if (
        value.ndim != 4
        or int(value.shape[0]) != 1
        or int(value.shape[1]) != int(num_classes)
    ):
        raise ValueError("class scores must have shape [1,C,H,W]")
    output = value.detach().to(device=device, dtype=torch.float32)
    if tuple(output.shape[-2:]) != tuple(size):
        output = F.interpolate(
            output,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return output


def _ratio(numerator: int, denominator: int) -> float | None:
    if int(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


class CanonicalCorrectionSurvivalAccumulator:
    def __init__(
        self,
        *,
        num_classes: int,
        class_names: Sequence[str],
        query_words: Sequence[str],
        query_idx_list: Sequence[int],
        canonical_query_ids: Sequence[int],
        boundary_radii: Sequence[int],
        device: torch.device,
        ignore_index: int = 255,
        score_epsilon: float = SCORE_EPSILON,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_names = tuple(str(value) for value in class_names)
        self.query_words = tuple(str(value) for value in query_words)
        self.query_idx_list = tuple(int(value) for value in query_idx_list)
        self.canonical_query_ids = tuple(
            int(value) for value in canonical_query_ids
        )
        self.boundary_radii = normalize_boundary_radii(boundary_radii)
        self.device = torch.device(device)
        self.ignore_index = int(ignore_index)
        self.score_epsilon = float(score_epsilon)
        if self.score_epsilon <= 0.0:
            raise ValueError("score_epsilon must be positive")
        if len(self.class_names) != self.num_classes:
            raise ValueError("class_names length must equal num_classes")
        if len(self.query_words) != len(self.query_idx_list):
            raise ValueError("query_words length must match query mapping")
        if len(self.canonical_query_ids) != self.num_classes:
            raise ValueError("canonical query count must equal num_classes")
        if set(self.query_idx_list) != set(range(self.num_classes)):
            raise ValueError("query mapping must cover every class")

        aliases = []
        for class_id, canonical_id in enumerate(self.canonical_query_ids):
            if (
                canonical_id < 0
                or canonical_id >= len(self.query_idx_list)
                or self.query_idx_list[canonical_id] != class_id
            ):
                raise ValueError("canonical query mapping is invalid")
            aliases.append(
                tuple(
                    query_id
                    for query_id, mapped in enumerate(self.query_idx_list)
                    if mapped == class_id and query_id != canonical_id
                )
            )
        self.alias_query_ids = tuple(aliases)
        self.has_alias = tuple(bool(values) for values in aliases)

        scopes = len(SCOPE_NAMES)
        radii = len(self.boundary_radii)
        classes = self.num_classes
        regions = len(REGION_NAMES)
        queries = len(self.query_words)
        self._processed = torch.zeros(
            scopes, dtype=torch.long, device=self.device
        )
        self._event_counts = torch.zeros(
            (scopes, radii, classes, len(SURVIVAL_EVENT_NAMES), regions),
            dtype=torch.long,
            device=self.device,
        )
        self._suppression_counts = torch.zeros(
            (scopes, radii, classes, regions, len(SUPPRESSION_COUNT_NAMES)),
            dtype=torch.long,
            device=self.device,
        )
        self._suppression_moment_count = torch.zeros(
            (scopes, radii, classes, regions),
            dtype=torch.long,
            device=self.device,
        )
        self._suppression_sum = torch.zeros(
            (scopes, radii, classes, regions, len(SUPPRESSION_FEATURE_NAMES)),
            dtype=torch.float64,
            device=self.device,
        )
        self._suppression_sq_sum = torch.zeros_like(self._suppression_sum)
        self._prompt_counts = torch.zeros(
            (scopes, queries, classes, len(PROMPT_RISK_NAMES)),
            dtype=torch.long,
            device=self.device,
        )
        self._prompt_delta_sum = torch.zeros(
            (scopes, queries, classes, 2),
            dtype=torch.float64,
            device=self.device,
        )
        self._transition_counts = torch.zeros(
            (scopes, classes, classes, len(TRANSITION_COUNT_NAMES)),
            dtype=torch.long,
            device=self.device,
        )
        self._transition_delta_sum = torch.zeros(
            (scopes, classes, classes, 2),
            dtype=torch.float64,
            device=self.device,
        )
        self._sample_ids: list[str] = []
        self._image_rows: list[dict] = []
        self._prompt_image_rows: list[dict] = []
        self._output_sizes: set[tuple[int, int]] = set()
        self._validation = {
            "correction_partition": True,
            "collapse_partition": True,
            "suppression_partition": True,
            "production_prediction_heads": True,
        }

    @staticmethod
    def _scope_indices(adapted: bool) -> tuple[int, ...]:
        return (0, 1) if adapted else (0,)

    def _alias_views(
        self,
        query_scores: torch.Tensor,
        *,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query_scores.ndim != 4
            or int(query_scores.shape[0]) != 1
            or int(query_scores.shape[1]) != len(self.query_words)
        ):
            raise ValueError("query scores must have shape [1,Q,H,W]")
        scores = query_scores.detach().to(
            device=self.device, dtype=torch.float32
        )
        if tuple(scores.shape[-2:]) != tuple(size):
            raise ValueError(
                "query scores must use the production prediction resolution"
            )
        alias_scores = []
        alias_winners = []
        for alias_ids in self.alias_query_ids:
            if alias_ids:
                local = scores[0, list(alias_ids)]
                values, local_winner = local.max(dim=0)
                global_ids = torch.as_tensor(
                    alias_ids, device=self.device, dtype=torch.long
                )
                alias_scores.append(values)
                alias_winners.append(global_ids[local_winner])
            else:
                alias_scores.append(
                    torch.full(
                        size,
                        float("-inf"),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                alias_winners.append(
                    torch.full(
                        size,
                        -1,
                        device=self.device,
                        dtype=torch.long,
                    )
                )
        return torch.stack(alias_scores), torch.stack(alias_winners)

    def _validate_events(
        self,
        events: Mapping[str, torch.Tensor],
    ) -> None:
        _assert_partition(
            "canonical correction",
            events["canonical_correction"],
            tuple(events[name] for name in CORRECTION_FATE_NAMES),
        )
        _assert_partition(
            "canonical damage",
            events["canonical_damage"],
            tuple(events[name] for name in COLLAPSE_FATE_NAMES),
        )

    def _add_prompt_event(
        self,
        *,
        scope_indices: Sequence[int],
        query_id: int,
        gt_class: int,
        event_name: str,
        mask: torch.Tensor,
        true_delta: torch.Tensor | None = None,
        wrong_delta: torch.Tensor | None = None,
        prompt_rows: dict[tuple[int, int], dict],
    ) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        event_index = PROMPT_RISK_NAMES.index(event_name)
        true_sum = (
            float(true_delta[mask].double().sum().item())
            if true_delta is not None
            else 0.0
        )
        wrong_sum = (
            float(wrong_delta[mask].double().sum().item())
            if wrong_delta is not None
            else 0.0
        )
        for scope_index in scope_indices:
            self._prompt_counts[
                scope_index, query_id, gt_class, event_index
            ] += count
            if event_name == "dynamic_overwrite":
                self._prompt_delta_sum[
                    scope_index, query_id, gt_class, 0
                ] += true_sum
                self._prompt_delta_sum[
                    scope_index, query_id, gt_class, 1
                ] += wrong_sum
        key = (int(query_id), int(gt_class))
        row = prompt_rows.setdefault(
            key,
            {
                "query_id": int(query_id),
                "query": self.query_words[query_id],
                "class_id": int(self.query_idx_list[query_id]),
                "gt_class_id": int(gt_class),
                "gt_class_name": self.class_names[gt_class],
                "counts": {name: 0 for name in PROMPT_RISK_NAMES},
                "dynamic_overwrite_true_delta_sum": 0.0,
                "dynamic_overwrite_wrong_delta_sum": 0.0,
            },
        )
        row["counts"][event_name] += count
        if event_name == "dynamic_overwrite":
            row["dynamic_overwrite_true_delta_sum"] += true_sum
            row["dynamic_overwrite_wrong_delta_sum"] += wrong_sum

    def update_image(
        self,
        sample_id: str,
        *,
        scores: Mapping[str, torch.Tensor],
        predictions: Mapping[str, torch.Tensor],
        pre_query_scores: torch.Tensor,
        post_query_scores: torch.Tensor,
        gt: torch.Tensor,
        adapted: bool,
    ) -> None:
        if set(scores) != set(_SCORE_NAMES):
            raise ValueError(
                "scores must contain c0, ct, a0, at, s0, s10, and st"
            )
        if set(predictions) != set(_PREDICTION_NAMES):
            raise ValueError(
                "predictions must contain p0, s0, pt, s10, and st"
            )
        gt = gt.detach().to(device=self.device, dtype=torch.long)
        if gt.ndim != 2:
            raise ValueError("gt must have shape [H,W]")
        size = (int(gt.shape[0]), int(gt.shape[1]))
        self._output_sizes.add(size)
        normalized_predictions = {
            name: value.detach().to(device=self.device, dtype=torch.long)
            for name, value in predictions.items()
        }
        if any(
            tuple(value.shape) != size
            for value in normalized_predictions.values()
        ):
            raise ValueError("production prediction/gt shape mismatch")
        resized = {
            name: _resize_class_scores(
                scores[name],
                num_classes=self.num_classes,
                size=size,
                device=self.device,
            )
            for name in ("c0", "ct", "s0", "s10", "st")
        }
        pre_alias, pre_winner = self._alias_views(
            pre_query_scores, size=size
        )
        post_alias, post_winner = self._alias_views(
            post_query_scores, size=size
        )
        scope_indices = self._scope_indices(bool(adapted))
        prompt_rows: dict[tuple[int, int], dict] = {}
        image_rows = []
        events_by_class = {}

        for class_id in range(self.num_classes):
            events = build_survival_event_masks(
                gt,
                normalized_predictions,
                class_id=class_id,
                ignore_index=self.ignore_index,
            )
            self._validate_events(events)
            events_by_class[class_id] = events
            regions_by_radius = [
                build_class_regions(
                    gt,
                    class_id=class_id,
                    radius=radius,
                    ignore_index=self.ignore_index,
                )
                for radius in self.boundary_radii
            ]

            canonical_delta = resized["ct"][0, class_id] - resized["c0"][0, class_id]
            synonym_delta = resized["st"][0, class_id] - resized["s0"][0, class_id]
            if self.has_alias[class_id]:
                alias_delta = post_alias[class_id] - pre_alias[class_id]
                suppressed = canonical_delta < -self.score_epsilon
                directions = (
                    alias_delta < -self.score_epsilon,
                    alias_delta.abs() <= self.score_epsilon,
                    alias_delta > self.score_epsilon,
                )
                _assert_partition(
                    "alias direction",
                    suppressed,
                    tuple(suppressed & value for value in directions),
                )
            else:
                alias_delta = None
                suppressed = None
                directions = ()

            for radius_index, (radius, regions) in enumerate(
                zip(self.boundary_radii, regions_by_radius)
            ):
                event_row = {}
                suppression_row = None
                for event_index, event_name in enumerate(SURVIVAL_EVENT_NAMES):
                    event_row[event_name] = {}
                    for region_index, (region_name, region_mask) in enumerate(
                        regions.as_dict().items()
                    ):
                        count = int((events[event_name] & region_mask).sum().item())
                        event_row[event_name][region_name] = count
                        for scope_index in scope_indices:
                            self._event_counts[
                                scope_index,
                                radius_index,
                                class_id,
                                event_index,
                                region_index,
                            ] += count

                if self.has_alias[class_id]:
                    suppression_row = {}
                    for region_index, (region_name, region_mask) in enumerate(
                        regions.as_dict().items()
                    ):
                        condition = suppressed & region_mask
                        count_values = (
                            int(condition.sum().item()),
                            *(
                                int((condition & direction).sum().item())
                                for direction in directions
                            ),
                        )
                        if sum(count_values[1:]) != count_values[0]:
                            raise ValueError("suppression direction partition failed")
                        features = (canonical_delta, alias_delta, synonym_delta)
                        sums = [
                            float(value[condition].double().sum().item())
                            for value in features
                        ]
                        squared = [
                            float(value[condition].double().square().sum().item())
                            for value in features
                        ]
                        for scope_index in scope_indices:
                            self._suppression_counts[
                                scope_index, radius_index, class_id, region_index
                            ] += torch.as_tensor(
                                count_values,
                                device=self.device,
                                dtype=torch.long,
                            )
                            self._suppression_moment_count[
                                scope_index, radius_index, class_id, region_index
                            ] += count_values[0]
                            self._suppression_sum[
                                scope_index, radius_index, class_id, region_index
                            ] += torch.as_tensor(
                                sums, device=self.device, dtype=torch.float64
                            )
                            self._suppression_sq_sum[
                                scope_index, radius_index, class_id, region_index
                            ] += torch.as_tensor(
                                squared, device=self.device, dtype=torch.float64
                            )
                        suppression_row[region_name] = {
                            "counts": dict(zip(SUPPRESSION_COUNT_NAMES, count_values)),
                            "feature_sums": dict(zip(SUPPRESSION_FEATURE_NAMES, sums)),
                        }
                image_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "adapted": bool(adapted),
                        "class_id": int(class_id),
                        "class_name": self.class_names[class_id],
                        "radius": int(radius),
                        "events": event_row,
                        "suppression_coupling": suppression_row,
                    }
                )

        s10 = resized["s10"][0]
        st_scores = resized["st"][0]
        ct = resized["ct"][0]
        final_prediction = normalized_predictions["st"]
        for gt_class in range(self.num_classes):
            events = events_by_class[gt_class]
            overwrite = events["correction_dynamic_overwrite"]
            for wrong_class in range(self.num_classes):
                if wrong_class == gt_class:
                    continue
                pair = overwrite & (final_prediction == wrong_class)
                count = int(pair.sum().item())
                if count == 0:
                    continue
                true_delta = st_scores[gt_class] - s10[gt_class]
                wrong_delta = st_scores[wrong_class] - s10[wrong_class]
                expansion = pair & (wrong_delta > self.score_epsilon)
                withdrawal = pair & (true_delta < -self.score_epsilon)
                mixed = expansion & withdrawal
                transition_values = (
                    count,
                    int(expansion.sum().item()),
                    int(withdrawal.sum().item()),
                    int(mixed.sum().item()),
                )
                true_sum = float(true_delta[pair].double().sum().item())
                wrong_sum = float(wrong_delta[pair].double().sum().item())
                for scope_index in scope_indices:
                    self._transition_counts[
                        scope_index, gt_class, wrong_class
                    ] += torch.as_tensor(
                        transition_values,
                        device=self.device,
                        dtype=torch.long,
                    )
                    self._transition_delta_sum[
                        scope_index, gt_class, wrong_class
                    ] += torch.tensor(
                        [true_sum, wrong_sum],
                        device=self.device,
                        dtype=torch.float64,
                    )

                wrong_control = pair & (
                    post_alias[wrong_class]
                    > ct[wrong_class] + self.score_epsilon
                )
                for query_id in self.alias_query_ids[wrong_class]:
                    query_mask = wrong_control & (post_winner[wrong_class] == query_id)
                    self._add_prompt_event(
                        scope_indices=scope_indices,
                        query_id=query_id,
                        gt_class=gt_class,
                        event_name="dynamic_overwrite",
                        mask=query_mask,
                        true_delta=true_delta,
                        wrong_delta=wrong_delta,
                        prompt_rows=prompt_rows,
                    )
                    self._add_prompt_event(
                        scope_indices=scope_indices,
                        query_id=query_id,
                        gt_class=gt_class,
                        event_name="foreign_expansion",
                        mask=query_mask & expansion,
                        prompt_rows=prompt_rows,
                    )

                true_control = withdrawal & (
                    pre_alias[gt_class]
                    > ct[gt_class] + self.score_epsilon
                )
                for query_id in self.alias_query_ids[gt_class]:
                    self._add_prompt_event(
                        scope_indices=scope_indices,
                        query_id=query_id,
                        gt_class=gt_class,
                        event_name="true_class_withdrawal",
                        mask=true_control & (pre_winner[gt_class] == query_id),
                        prompt_rows=prompt_rows,
                    )

            recovery = events["correction_dynamic_recovery"] & (
                post_alias[gt_class] > ct[gt_class] + self.score_epsilon
            )
            for query_id in self.alias_query_ids[gt_class]:
                self._add_prompt_event(
                    scope_indices=scope_indices,
                    query_id=query_id,
                    gt_class=gt_class,
                    event_name="dynamic_rescue",
                    mask=recovery & (post_winner[gt_class] == query_id),
                    prompt_rows=prompt_rows,
                )

        for target_class in range(self.num_classes):
            reintroduced = events_by_class[target_class]["fp_reintroduced"] & (
                post_alias[target_class]
                > ct[target_class] + self.score_epsilon
            )
            for gt_class in range(self.num_classes):
                if gt_class == target_class:
                    continue
                gt_mask = reintroduced & (gt == gt_class)
                for query_id in self.alias_query_ids[target_class]:
                    self._add_prompt_event(
                        scope_indices=scope_indices,
                        query_id=query_id,
                        gt_class=gt_class,
                        event_name="fp_reintroduction",
                        mask=gt_mask & (post_winner[target_class] == query_id),
                        prompt_rows=prompt_rows,
                    )

        self._processed[0] += 1
        if adapted:
            self._processed[1] += 1
        self._sample_ids.append(str(sample_id))
        self._image_rows.extend(image_rows)
        self._prompt_image_rows.extend(
            {"sample_id": str(sample_id), "adapted": bool(adapted), **row}
            for row in prompt_rows.values()
        )

    def _metadata(self) -> tuple:
        return (
            SCHEMA_VERSION,
            self.num_classes,
            self.class_names,
            self.query_words,
            self.query_idx_list,
            self.canonical_query_ids,
            self.alias_query_ids,
            self.boundary_radii,
            self.ignore_index,
            self.score_epsilon,
            tuple(sorted(self._output_sizes)),
        )

    def reduce_distributed(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        metadata = self._metadata()
        world_size = dist.get_world_size()
        gathered_metadata = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metadata, metadata)
        if any(value != metadata for value in gathered_metadata):
            raise ValueError("correction survival metadata differs across ranks")
        for tensor in (
            self._processed,
            self._event_counts,
            self._suppression_counts,
            self._suppression_moment_count,
            self._suppression_sum,
            self._suppression_sq_sum,
            self._prompt_counts,
            self._prompt_delta_sum,
            self._transition_counts,
            self._transition_delta_sum,
        ):
            dist.all_reduce(tensor)
        gathered_ids = [None for _ in range(world_size)]
        gathered_rows = [None for _ in range(world_size)]
        gathered_prompt_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_ids, self._sample_ids)
        dist.all_gather_object(gathered_rows, self._image_rows)
        dist.all_gather_object(gathered_prompt_rows, self._prompt_image_rows)
        self._sample_ids = [
            str(value)
            for values in gathered_ids
            for value in (values or [])
        ]
        self._image_rows = [
            dict(value)
            for values in gathered_rows
            for value in (values or [])
        ]
        self._prompt_image_rows = [
            dict(value)
            for values in gathered_prompt_rows
            for value in (values or [])
        ]

    def _event_report(self, scope_index: int) -> dict:
        output = {}
        for event_index, event_name in enumerate(SURVIVAL_EVENT_NAMES):
            output[event_name] = {}
            for class_id in range(self.num_classes):
                output[event_name][str(class_id)] = {}
                for radius_index, radius in enumerate(self.boundary_radii):
                    output[event_name][str(class_id)][str(radius)] = {
                        region_name: int(
                            self._event_counts[
                                scope_index,
                                radius_index,
                                class_id,
                                event_index,
                                region_index,
                            ].item()
                        )
                        for region_index, region_name in enumerate(REGION_NAMES)
                    }
        return output

    def _fate_ratio_report(self, event_report: Mapping) -> dict:
        output = {}
        for class_id in range(self.num_classes):
            class_key = str(class_id)
            output[class_key] = {}
            for radius in self.boundary_radii:
                radius_key = str(radius)
                output[class_key][radius_key] = {}
                for region in REGION_NAMES:
                    def value(name: str) -> int:
                        return int(event_report[name][class_key][radius_key][region])

                    correction = value("canonical_correction")
                    pre_solved = value(
                        "correction_pre_solved_retained"
                    ) + value("correction_pre_solved_degraded")
                    visible = value(
                        "correction_visible_preserved"
                    ) + value("correction_dynamic_overwrite")
                    blocked = value(
                        "correction_dynamic_recovery"
                    ) + value("correction_static_unresolved")
                    damage = value("canonical_damage")
                    protected = value(
                        "collapse_static_retained"
                    ) + value("collapse_dynamic_protection")
                    fp_removed = value("canonical_fp_removed")
                    output[class_key][radius_key][region] = {
                        "pre_solved_rate": _ratio(pre_solved, correction),
                        "static_shielding_rate": _ratio(blocked, visible + blocked),
                        "dynamic_overwrite_rate": _ratio(
                            value("correction_dynamic_overwrite"),
                            visible,
                        ),
                        "dynamic_recovery_rate": _ratio(
                            value("correction_dynamic_recovery"),
                            blocked,
                        ),
                        "collapse_protection_rate": _ratio(protected, damage),
                        "fp_reintroduction_rate": _ratio(value("fp_reintroduced"), fp_removed),
                    }
        return output

    def _suppression_report(self, scope_index: int) -> dict:
        output = {}
        for class_id in range(self.num_classes):
            if not self.has_alias[class_id]:
                continue
            class_key = str(class_id)
            output[class_key] = {}
            for radius_index, radius in enumerate(self.boundary_radii):
                radius_key = str(radius)
                output[class_key][radius_key] = {}
                for region_index, region_name in enumerate(REGION_NAMES):
                    counts = self._suppression_counts[
                        scope_index, radius_index, class_id, region_index
                    ]
                    count = int(
                        self._suppression_moment_count[
                            scope_index, radius_index, class_id, region_index
                        ].item()
                    )
                    moments = {}
                    for feature_index, feature_name in enumerate(SUPPRESSION_FEATURE_NAMES):
                        moments[feature_name] = finalize_moment(
                            count,
                            float(
                                self._suppression_sum[
                                    scope_index,
                                    radius_index,
                                    class_id,
                                    region_index,
                                    feature_index,
                                ].item()
                            ),
                            float(
                                self._suppression_sq_sum[
                                    scope_index,
                                    radius_index,
                                    class_id,
                                    region_index,
                                    feature_index,
                                ].item()
                            ),
                        )
                    output[class_key][radius_key][region_name] = {
                        "counts": {
                            name: int(counts[index].item())
                            for index, name in enumerate(SUPPRESSION_COUNT_NAMES)
                        },
                        "ratios": {
                            name: _ratio(int(counts[index].item()), int(counts[0].item()))
                            for index, name in enumerate(SUPPRESSION_COUNT_NAMES[1:], start=1)
                        },
                        "moments": moments,
                    }
        return output

    def _prompt_report(self, scope_index: int) -> list[dict]:
        output = []
        for query_id, class_id in enumerate(self.query_idx_list):
            if query_id == self.canonical_query_ids[class_id]:
                continue
            by_gt = []
            totals = [0 for _ in PROMPT_RISK_NAMES]
            delta_totals = [0.0, 0.0]
            for gt_class in range(self.num_classes):
                counts = [
                    int(value)
                    for value in self._prompt_counts[
                        scope_index, query_id, gt_class
                    ].detach().cpu().tolist()
                ]
                deltas = [
                    float(value)
                    for value in self._prompt_delta_sum[
                        scope_index, query_id, gt_class
                    ].detach().cpu().tolist()
                ]
                totals = [left + right for left, right in zip(totals, counts)]
                delta_totals = [left + right for left, right in zip(delta_totals, deltas)]
                if any(counts):
                    by_gt.append(
                        {
                            "gt_class_id": int(gt_class),
                            "gt_class_name": self.class_names[gt_class],
                            **dict(zip(PROMPT_RISK_NAMES, counts)),
                        }
                    )
            dynamic_count = totals[PROMPT_RISK_NAMES.index("dynamic_overwrite")]
            rescue_count = totals[PROMPT_RISK_NAMES.index("dynamic_rescue")]
            output.append(
                {
                    "query_id": int(query_id),
                    "query": self.query_words[query_id],
                    "class_id": int(class_id),
                    "class_name": self.class_names[class_id],
                    **dict(zip(PROMPT_RISK_NAMES, totals)),
                    "overwrite_to_rescue_ratio": _ratio(dynamic_count, rescue_count),
                    "dynamic_overwrite_true_delta_mean": _ratio(delta_totals[0], dynamic_count),
                    "dynamic_overwrite_wrong_delta_mean": _ratio(delta_totals[1], dynamic_count),
                    "by_gt_class": by_gt,
                }
            )
        return output

    def _transition_report(self, scope_index: int) -> list[dict]:
        output = []
        for gt_class in range(self.num_classes):
            for wrong_class in range(self.num_classes):
                if gt_class == wrong_class:
                    continue
                counts = [
                    int(value)
                    for value in self._transition_counts[
                        scope_index, gt_class, wrong_class
                    ].detach().cpu().tolist()
                ]
                if not any(counts):
                    continue
                deltas = [
                    float(value)
                    for value in self._transition_delta_sum[
                        scope_index, gt_class, wrong_class
                    ].detach().cpu().tolist()
                ]
                total = counts[0]
                output.append(
                    {
                        "gt_class_id": gt_class,
                        "gt_class_name": self.class_names[gt_class],
                        "wrong_class_id": wrong_class,
                        "wrong_class_name": self.class_names[wrong_class],
                        **dict(zip(TRANSITION_COUNT_NAMES, counts)),
                        "true_score_delta_mean": _ratio(deltas[0], total),
                        "wrong_score_delta_mean": _ratio(deltas[1], total),
                    }
                )
        return output

    def finalize(self) -> dict:
        scopes = {}
        for scope_index, scope_name in enumerate(SCOPE_NAMES):
            event_report = self._event_report(scope_index)
            scopes[scope_name] = {
                "processed": int(self._processed[scope_index].item()),
                "event_counts": event_report,
                "fate_ratios": self._fate_ratio_report(event_report),
                "suppression_coupling": self._suppression_report(scope_index),
                "prompt_risk": self._prompt_report(scope_index),
                "transition_mechanisms": self._transition_report(scope_index),
            }
        sample_count = len(self._sample_ids)
        return {
            "schema_version": SCHEMA_VERSION,
            "prediction_semantics": "production predict_from_class_scores heads",
            "score_epsilon": self.score_epsilon,
            "processed": int(self._processed[0].item()),
            "adapted": int(self._processed[1].item()),
            "unique_samples": len(set(self._sample_ids)),
            "duplicate_sample_ids": sample_count - len(set(self._sample_ids)),
            "class_names": list(self.class_names),
            "query_words": list(self.query_words),
            "query_idx_list": list(self.query_idx_list),
            "canonical_query_ids": list(self.canonical_query_ids),
            "boundary_radii": list(self.boundary_radii),
            "output_sizes": [list(value) for value in sorted(self._output_sizes)],
            "scopes": scopes,
            "image_rows": sorted(
                self._image_rows,
                key=lambda row: (
                    row["sample_id"], row["class_id"], row["radius"]
                ),
            ),
            "prompt_image_rows": sorted(
                self._prompt_image_rows,
                key=lambda row: (
                    row["sample_id"], row["query_id"], row["gt_class_id"]
                ),
            ),
            "validation": dict(self._validation),
            "dense_outputs_serialized": False,
        }


def merge_prompt_risk_with_leave_one_out(
    correction_report: Mapping,
    synonym_report: Mapping,
) -> list[dict]:
    scopes = correction_report.get("scopes", {})
    all_rows = {
        int(row["query_id"]): row
        for row in scopes.get("all", {}).get("prompt_risk", [])
    }
    adapted_rows = {
        int(row["query_id"]): row
        for row in scopes.get("adapted_only", {}).get("prompt_risk", [])
    }
    aliases = {
        int(row["query_id"]): row
        for row in synonym_report.get("aliases", [])
    }
    output = []
    for query_id in sorted(set(all_rows) | set(adapted_rows) | set(aliases)):
        alias = aliases.get(query_id, {})
        query = (
            all_rows.get(query_id, {}).get("query")
            or adapted_rows.get(query_id, {}).get("query")
            or alias.get("query")
            or str(query_id)
        )
        output.append(
            {
                "query_id": query_id,
                "query": query,
                "dynamic_all": all_rows.get(query_id),
                "dynamic_adapted_only": adapted_rows.get(query_id),
                "leave_one_out": alias.get("leave_one_out"),
            }
        )
    return output
