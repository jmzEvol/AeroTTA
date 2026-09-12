from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


SCHEMA_VERSION = 1
DEFAULT_BOUNDARY_RADII = (3, 5, 9)
REGION_NAMES = (
    "inner_boundary",
    "interior",
    "outer_boundary",
    "far_exterior",
)
EVENT_NAMES = (
    "canonical_rescue",
    "synonym_rescue",
    "fn_pre_solved",
    "already_solved",
    "newly_visible",
    "frozen_blocked",
    "canonical_fp_removed",
    "synonym_fp_removed",
    "fp_pre_solved",
    "fp_correction_retained",
    "alias_reintroduced_fp",
    "pre_alias_harm",
    "pre_alias_rescue",
    "pre_alias_lateral",
    "pre_alias_unchanged",
    "post_alias_harm",
    "post_alias_rescue",
    "post_alias_lateral",
    "post_alias_unchanged",
)
ABSOLUTE_FEATURE_NAMES = (
    "c0_score",
    "ct_score",
    "s0_score",
    "st_score",
)
SIGNED_FEATURE_NAMES = (
    "c0_margin",
    "ct_margin",
    "s0_margin",
    "st_margin",
    "canonical_delta",
    "pre_alias_residual",
    "post_alias_residual",
    "synonym_delta",
)
FEATURE_NAMES = ABSOLUTE_FEATURE_NAMES + SIGNED_FEATURE_NAMES
_SCORE_NAMES = ("c0", "ct", "a0", "at", "s0", "s10", "s01", "st")
_PREDICTION_NAMES = ("p0", "s0", "pt", "s10", "s01", "st")
_ABSOLUTE_HISTOGRAM = (0.0, 1.0, 100)
_SIGNED_HISTOGRAM = (-1.0, 1.0, 200)
_RATIO_FIELDS = {
    "fn_pre_solved_ratio": ("fn_pre_solved", "canonical_rescue"),
    "fp_pre_solved_ratio": (
        "fp_pre_solved",
        "canonical_fp_removed",
    ),
    "fp_reintroduced_share": (
        "alias_reintroduced_fp",
        "canonical_fp_removed",
    ),
}


@dataclass(frozen=True)
class BoundaryRegions:
    inner_boundary: torch.Tensor
    interior: torch.Tensor
    outer_boundary: torch.Tensor
    far_exterior: torch.Tensor

    def __iter__(self):
        yield self.inner_boundary
        yield self.interior
        yield self.outer_boundary
        yield self.far_exterior

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "inner_boundary": self.inner_boundary,
            "interior": self.interior,
            "outer_boundary": self.outer_boundary,
            "far_exterior": self.far_exterior,
        }


def normalize_boundary_radii(values: Sequence[int]) -> tuple[int, ...]:
    radii = tuple(int(value) for value in values)
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("boundary radii must be positive integers")
    if tuple(sorted(set(radii))) != radii:
        raise ValueError(
            "boundary radii must be strictly increasing and unique"
        )
    return radii


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    padded = F.pad(
        mask[None, None].float(),
        (radius, radius, radius, radius),
        value=0.0,
    )
    return (
        F.max_pool2d(padded, 2 * radius + 1, stride=1)[0, 0]
        > 0.5
    )


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    padded = F.pad(
        mask[None, None].float(),
        (radius, radius, radius, radius),
        value=0.0,
    )
    return (
        -F.max_pool2d(-padded, 2 * radius + 1, stride=1)[0, 0]
        > 0.5
    )


def build_class_regions(
    gt: torch.Tensor,
    class_id: int,
    radius: int,
    ignore_index: int = 255,
) -> BoundaryRegions:
    if gt.ndim != 2:
        raise ValueError("gt must have shape [H,W]")
    radius = normalize_boundary_radii((radius,))[0]
    valid = gt != int(ignore_index)
    positive = valid & (gt == int(class_id))
    eroded = _erode(positive, radius)
    dilated = _dilate(positive, radius)
    return BoundaryRegions(
        inner_boundary=valid & positive & ~eroded,
        interior=valid & eroded,
        outer_boundary=valid & ~positive & dilated,
        far_exterior=valid & ~dilated,
    )


def build_event_masks(
    gt: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
    class_id: int,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    required = ("p0", "s0", "pt", "s10", "s01", "st")
    if set(predictions) != set(required):
        raise ValueError(
            "predictions must contain p0, s0, pt, s10, s01, and st"
        )
    if any(
        tuple(value.shape) != tuple(gt.shape)
        for value in predictions.values()
    ):
        raise ValueError("prediction/gt shape mismatch")

    class_id = int(class_id)
    valid = gt != int(ignore_index)
    positive = valid & (gt == class_id)
    negative = valid & (gt != class_id)
    p0 = predictions["p0"]
    s0 = predictions["s0"]
    pt = predictions["pt"]
    s10 = predictions["s10"]
    s01 = predictions["s01"]
    st = predictions["st"]

    canonical_rescue = positive & (p0 != class_id) & (pt == class_id)
    canonical_fp_removed = (
        negative & (p0 == class_id) & (pt != class_id)
    )
    events = {
        "canonical_rescue": canonical_rescue,
        "synonym_rescue": (
            positive & (p0 != class_id) & (s0 == class_id)
        ),
        "fn_pre_solved": canonical_rescue & (s0 == class_id),
        "already_solved": canonical_rescue & (s0 == class_id),
        "newly_visible": (
            canonical_rescue & (s0 != class_id) & (s10 == class_id)
        ),
        "frozen_blocked": (
            canonical_rescue & (s0 != class_id) & (s10 != class_id)
        ),
        "canonical_fp_removed": canonical_fp_removed,
        "synonym_fp_removed": (
            negative & (p0 == class_id) & (s0 != class_id)
        ),
        "fp_pre_solved": canonical_fp_removed & (s0 != class_id),
        "fp_correction_retained": (
            canonical_fp_removed & (st != class_id)
        ),
        "alias_reintroduced_fp": (
            canonical_fp_removed & (st == class_id)
        ),
    }
    for prefix, before, after in (
        ("pre", s0, s01),
        ("post", s10, st),
    ):
        before_correct = before == gt
        after_correct = after == gt
        events[f"{prefix}_alias_harm"] = (
            positive & before_correct & ~after_correct
        )
        events[f"{prefix}_alias_rescue"] = (
            positive & ~before_correct & after_correct
        )
        events[f"{prefix}_alias_lateral"] = (
            positive
            & ~before_correct
            & ~after_correct
            & (before != after)
        )
        events[f"{prefix}_alias_unchanged"] = (
            positive & (before == after)
        )
    return {name: events[name] for name in EVENT_NAMES}


def class_margin(scores: torch.Tensor, class_id: int) -> torch.Tensor:
    if (
        scores.ndim != 4
        or int(scores.shape[0]) != 1
        or int(scores.shape[1]) < 2
    ):
        raise ValueError(
            "scores must have shape [1,C,H,W] with C >= 2"
        )
    class_id = int(class_id)
    if class_id < 0 or class_id >= int(scores.shape[1]):
        raise ValueError("class_id is outside the score tensor")
    competitors = torch.cat(
        (scores[:, :class_id], scores[:, class_id + 1 :]),
        dim=1,
    )
    return (
        scores[0, class_id].float()
        - competitors.amax(dim=1)[0].float()
    )


def fixed_histogram(
    values: torch.Tensor,
    *,
    low: float,
    high: float,
    bins: int,
) -> torch.Tensor:
    if int(bins) <= 0 or float(high) <= float(low):
        raise ValueError("histogram range and bins must be positive")
    values = values.detach().double().reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("histogram values must be finite")
    output = torch.zeros(
        int(bins) + 2,
        dtype=torch.long,
        device=values.device,
    )
    output[0] = (values < float(low)).sum()
    output[-1] = (values > float(high)).sum()
    regular = values[
        (values >= float(low)) & (values <= float(high))
    ]
    if int(regular.numel()) > 0:
        indices = torch.floor(
            (regular - float(low))
            * int(bins)
            / (float(high) - float(low))
        ).long()
        indices = indices.clamp(0, int(bins) - 1) + 1
        output.scatter_add_(0, indices, torch.ones_like(indices))
    return output


def finalize_moment(
    count: int,
    total: float,
    squared_total: float,
) -> dict | None:
    if int(count) == 0:
        return None
    mean = float(total) / int(count)
    variance = max(
        float(squared_total) / int(count) - mean * mean,
        0.0,
    )
    return {
        "count": int(count),
        "sum": float(total),
        "squared_sum": float(squared_total),
        "mean": float(mean),
        "std": float(variance**0.5),
    }


def _resize_score_tensor(
    scores: torch.Tensor,
    *,
    num_classes: int,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if (
        scores.ndim != 4
        or int(scores.shape[0]) != 1
        or int(scores.shape[1]) != int(num_classes)
    ):
        raise ValueError(
            "class scores must have shape [1,num_classes,H,W]"
        )
    scores = scores.detach().to(device=device, dtype=torch.float32)
    if tuple(scores.shape[-2:]) == tuple(size):
        return scores
    return F.interpolate(
        scores,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def _resize_single_score_map(
    score_map: torch.Tensor,
    *,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    score_map = score_map.detach().to(device=device, dtype=torch.float32)
    if score_map.ndim != 2:
        raise ValueError("score map must have shape [H,W]")
    if tuple(score_map.shape) == tuple(size):
        return score_map
    return F.interpolate(
        score_map[None, None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


class BoundaryCausalAccumulator:
    def __init__(
        self,
        *,
        num_classes: int,
        class_names: Sequence[str],
        class_prompts: Sequence[Sequence[str]],
        boundary_radii: Sequence[int],
        device: torch.device,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = int(num_classes)
        self.class_names = tuple(str(value) for value in class_names)
        self.class_prompts = tuple(
            tuple(str(prompt) for prompt in prompts)
            for prompts in class_prompts
        )
        self.boundary_radii = normalize_boundary_radii(boundary_radii)
        self.device = torch.device(device)
        self.ignore_index = int(ignore_index)
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names length must equal num_classes"
            )
        if len(self.class_prompts) != self.num_classes:
            raise ValueError(
                "class_prompts length must equal num_classes"
            )
        if any(not prompts for prompts in self.class_prompts):
            raise ValueError("every class must have at least one prompt")
        self.has_alias = tuple(
            len(prompts) > 1 for prompts in self.class_prompts
        )

        radii = len(self.boundary_radii)
        classes = self.num_classes
        events = len(EVENT_NAMES)
        regions = len(REGION_NAMES)
        features = len(FEATURE_NAMES)
        prefix = (radii, classes, events, regions)
        self._processed = torch.zeros(
            2,
            dtype=torch.long,
            device=self.device,
        )
        self._region_counts = torch.zeros(
            (radii, classes, regions),
            dtype=torch.long,
            device=self.device,
        )
        self._event_counts = torch.zeros(
            prefix,
            dtype=torch.long,
            device=self.device,
        )
        self._moment_count = torch.zeros(
            (*prefix, features),
            dtype=torch.long,
            device=self.device,
        )
        self._moment_sum = torch.zeros(
            (*prefix, features),
            dtype=torch.float64,
            device=self.device,
        )
        self._moment_sq_sum = torch.zeros_like(self._moment_sum)
        self._absolute_hist = torch.zeros(
            (
                *prefix,
                len(ABSOLUTE_FEATURE_NAMES),
                _ABSOLUTE_HISTOGRAM[2] + 2,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._signed_hist = torch.zeros(
            (
                *prefix,
                len(SIGNED_FEATURE_NAMES),
                _SIGNED_HISTOGRAM[2] + 2,
            ),
            dtype=torch.long,
            device=self.device,
        )
        self._sample_ids: list[str] = []
        self._image_rows: list[dict] = []
        self._output_sizes: set[tuple[int, int]] = set()
        self._validation = {
            "positive_region_partition": True,
            "negative_region_partition": True,
            "region_monotonicity": True,
            "fn_partition": True,
            "fp_partition": True,
            "pre_alias_partition": True,
            "post_alias_partition": True,
        }

    @staticmethod
    def _assert_partition(
        name: str,
        whole: torch.Tensor,
        parts: Sequence[torch.Tensor],
    ) -> None:
        union = torch.zeros_like(whole)
        part_count = 0
        for part in parts:
            union |= part
            part_count += int(part.sum().item())
        if not torch.equal(whole, union) or part_count != int(
            whole.sum().item()
        ):
            raise ValueError(f"boundary diagnostic {name} failed")

    def _validate_regions(
        self,
        *,
        gt: torch.Tensor,
        class_id: int,
        all_regions: Sequence[BoundaryRegions],
    ) -> None:
        valid = gt != self.ignore_index
        positive = valid & (gt == int(class_id))
        negative = valid & (gt != int(class_id))
        for regions in all_regions:
            self._assert_partition(
                "positive region partition",
                positive,
                (regions.inner_boundary, regions.interior),
            )
            self._assert_partition(
                "negative region partition",
                negative,
                (regions.outer_boundary, regions.far_exterior),
            )
        for smaller, larger in zip(all_regions, all_regions[1:]):
            invalid = (
                bool(
                    (
                        smaller.inner_boundary
                        & ~larger.inner_boundary
                    ).any()
                )
                or bool((larger.interior & ~smaller.interior).any())
                or bool(
                    (
                        smaller.outer_boundary
                        & ~larger.outer_boundary
                    ).any()
                )
                or bool(
                    (
                        larger.far_exterior
                        & ~smaller.far_exterior
                    ).any()
                )
            )
            if invalid:
                raise ValueError(
                    "boundary diagnostic region monotonicity failed"
                )

    def _validate_events(
        self,
        *,
        gt: torch.Tensor,
        class_id: int,
        events: Mapping[str, torch.Tensor],
    ) -> None:
        self._assert_partition(
            "FN partition",
            events["canonical_rescue"],
            (
                events["already_solved"],
                events["newly_visible"],
                events["frozen_blocked"],
            ),
        )
        self._assert_partition(
            "FP partition",
            events["canonical_fp_removed"],
            (
                events["fp_correction_retained"],
                events["alias_reintroduced_fp"],
            ),
        )
        positive = (gt != self.ignore_index) & (gt == int(class_id))
        for prefix in ("pre", "post"):
            self._assert_partition(
                f"{prefix} alias partition",
                positive,
                (
                    events[f"{prefix}_alias_harm"],
                    events[f"{prefix}_alias_rescue"],
                    events[f"{prefix}_alias_lateral"],
                    events[f"{prefix}_alias_unchanged"],
                ),
            )

    def _feature_maps(
        self,
        *,
        resized: Mapping[str, torch.Tensor],
        alias_maps: Mapping[str, torch.Tensor],
        class_id: int,
    ) -> dict[str, torch.Tensor]:
        c0 = resized["c0"][0, class_id]
        ct = resized["ct"][0, class_id]
        s0 = resized["s0"][0, class_id]
        st = resized["st"][0, class_id]
        output = {
            "c0_score": c0,
            "ct_score": ct,
            "s0_score": s0,
            "st_score": st,
            "c0_margin": class_margin(resized["c0"], class_id),
            "ct_margin": class_margin(resized["ct"], class_id),
            "s0_margin": class_margin(resized["s0"], class_id),
            "st_margin": class_margin(resized["st"], class_id),
            "canonical_delta": ct - c0,
            "synonym_delta": st - s0,
        }
        if self.has_alias[class_id]:
            output["pre_alias_residual"] = alias_maps["a0"] - c0
            output["post_alias_residual"] = alias_maps["at"] - ct
        return output

    def _update_score_statistics(
        self,
        *,
        radius_index: int,
        class_id: int,
        event_index: int,
        region_index: int,
        mask: torch.Tensor,
        feature_maps: Mapping[str, torch.Tensor],
    ) -> None:
        count = int(mask.sum().item())
        if count == 0:
            return
        for feature_index, feature_name in enumerate(FEATURE_NAMES):
            feature_map = feature_maps.get(feature_name)
            if feature_map is None:
                continue
            values = feature_map[mask]
            if not bool(torch.isfinite(values).all()):
                raise ValueError(
                    f"nonfinite boundary feature {feature_name}"
                )
            position = (
                radius_index,
                class_id,
                event_index,
                region_index,
                feature_index,
            )
            values64 = values.double()
            self._moment_count[position] += count
            self._moment_sum[position] += values64.sum()
            self._moment_sq_sum[position] += values64.square().sum()
            if feature_name in ABSOLUTE_FEATURE_NAMES:
                histogram = fixed_histogram(
                    values,
                    low=_ABSOLUTE_HISTOGRAM[0],
                    high=_ABSOLUTE_HISTOGRAM[1],
                    bins=_ABSOLUTE_HISTOGRAM[2],
                )
                histogram_index = ABSOLUTE_FEATURE_NAMES.index(
                    feature_name
                )
                self._absolute_hist[
                    radius_index,
                    class_id,
                    event_index,
                    region_index,
                    histogram_index,
                ] += histogram
            else:
                histogram = fixed_histogram(
                    values,
                    low=_SIGNED_HISTOGRAM[0],
                    high=_SIGNED_HISTOGRAM[1],
                    bins=_SIGNED_HISTOGRAM[2],
                )
                histogram_index = SIGNED_FEATURE_NAMES.index(
                    feature_name
                )
                self._signed_hist[
                    radius_index,
                    class_id,
                    event_index,
                    region_index,
                    histogram_index,
                ] += histogram

    def update_image(
        self,
        sample_id: str,
        *,
        scores: Mapping[str, torch.Tensor],
        predictions: Mapping[str, torch.Tensor],
        gt: torch.Tensor,
        adapted: bool,
    ) -> None:
        if set(scores) != set(_SCORE_NAMES):
            raise ValueError(
                "scores must contain c0, ct, a0, at, s0, s10, s01, and st"
            )
        if set(predictions) != set(_PREDICTION_NAMES):
            raise ValueError(
                "predictions must contain p0, s0, pt, s10, s01, and st"
            )
        gt = gt.detach().to(device=self.device, dtype=torch.long)
        if gt.ndim != 2:
            raise ValueError("gt must have shape [H,W]")
        output_size = (int(gt.shape[0]), int(gt.shape[1]))
        self._output_sizes.add(output_size)
        for score in scores.values():
            if (
                score.ndim != 4
                or int(score.shape[0]) != 1
                or int(score.shape[1]) != self.num_classes
            ):
                raise ValueError(
                    "class scores must have shape [1,num_classes,H,W]"
                )

        resized = {
            name: _resize_score_tensor(
                scores[name],
                num_classes=self.num_classes,
                size=output_size,
                device=self.device,
            )
            for name in ("c0", "ct", "s0", "st")
        }
        if not all(
            bool(torch.isfinite(value).all())
            for value in resized.values()
        ):
            raise ValueError("canonical and synonym scores must be finite")
        normalized_predictions = {
            name: value.detach().to(device=self.device, dtype=torch.long)
            for name, value in predictions.items()
        }

        for class_id in range(self.num_classes):
            alias_maps = {}
            if self.has_alias[class_id]:
                for name in ("a0", "at"):
                    alias_maps[name] = _resize_single_score_map(
                        scores[name][0, class_id],
                        size=output_size,
                        device=self.device,
                    )
            feature_maps = self._feature_maps(
                resized=resized,
                alias_maps=alias_maps,
                class_id=class_id,
            )
            events = build_event_masks(
                gt,
                normalized_predictions,
                class_id=class_id,
                ignore_index=self.ignore_index,
            )
            self._validate_events(
                gt=gt,
                class_id=class_id,
                events=events,
            )
            all_regions = [
                build_class_regions(
                    gt,
                    class_id=class_id,
                    radius=radius,
                    ignore_index=self.ignore_index,
                )
                for radius in self.boundary_radii
            ]
            self._validate_regions(
                gt=gt,
                class_id=class_id,
                all_regions=all_regions,
            )

            for radius_index, (radius, regions) in enumerate(
                zip(self.boundary_radii, all_regions)
            ):
                region_maps = regions.as_dict()
                region_row = {
                    name: int(mask.sum().item())
                    for name, mask in region_maps.items()
                }
                event_row = {}
                for event_index, event_name in enumerate(EVENT_NAMES):
                    event_row[event_name] = {}
                    for region_index, region_name in enumerate(
                        REGION_NAMES
                    ):
                        region_mask = region_maps[region_name]
                        event_region = events[event_name] & region_mask
                        count = int(event_region.sum().item())
                        event_row[event_name][region_name] = count
                        self._event_counts[
                            radius_index,
                            class_id,
                            event_index,
                            region_index,
                        ] += count
                        self._update_score_statistics(
                            radius_index=radius_index,
                            class_id=class_id,
                            event_index=event_index,
                            region_index=region_index,
                            mask=event_region,
                            feature_maps=feature_maps,
                        )
                for region_index, region_name in enumerate(REGION_NAMES):
                    self._region_counts[
                        radius_index,
                        class_id,
                        region_index,
                    ] += region_row[region_name]
                self._image_rows.append(
                    {
                        "sample_id": str(sample_id),
                        "adapted": bool(adapted),
                        "class_id": int(class_id),
                        "class_name": self.class_names[class_id],
                        "radius": int(radius),
                        "regions": region_row,
                        "events": event_row,
                    }
                )

        self._processed += torch.tensor(
            [1, int(bool(adapted))],
            dtype=torch.long,
            device=self.device,
        )
        self._sample_ids.append(str(sample_id))

    def _histogram_report(
        self,
        *,
        feature_name: str,
        radius_index: int,
        class_id: int,
        event_index: int,
        region_index: int,
    ) -> dict:
        if feature_name in ABSOLUTE_FEATURE_NAMES:
            low, high, bins = _ABSOLUTE_HISTOGRAM
            feature_index = ABSOLUTE_FEATURE_NAMES.index(feature_name)
            values = self._absolute_hist[
                radius_index,
                class_id,
                event_index,
                region_index,
                feature_index,
            ]
        else:
            low, high, bins = _SIGNED_HISTOGRAM
            feature_index = SIGNED_FEATURE_NAMES.index(feature_name)
            values = self._signed_hist[
                radius_index,
                class_id,
                event_index,
                region_index,
                feature_index,
            ]
        values = values.detach().cpu()
        return {
            "low": float(low),
            "high": float(high),
            "bins": int(bins),
            "underflow": int(values[0].item()),
            "counts": values[1:-1].tolist(),
            "overflow": int(values[-1].item()),
        }

    def _feature_report(
        self,
        *,
        feature_name: str,
        radius_index: int,
        class_id: int,
        event_index: int,
        region_index: int,
    ) -> dict | None:
        if (
            feature_name
            in ("pre_alias_residual", "post_alias_residual")
            and not self.has_alias[class_id]
        ):
            return None
        feature_index = FEATURE_NAMES.index(feature_name)
        position = (
            radius_index,
            class_id,
            event_index,
            region_index,
            feature_index,
        )
        moment = finalize_moment(
            int(self._moment_count[position].item()),
            float(self._moment_sum[position].item()),
            float(self._moment_sq_sum[position].item()),
        )
        if moment is None:
            return None
        moment["histogram"] = self._histogram_report(
            feature_name=feature_name,
            radius_index=radius_index,
            class_id=class_id,
            event_index=event_index,
            region_index=region_index,
        )
        return moment

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        if int(denominator) == 0:
            return None
        return float(numerator) / float(denominator)

    def _distributed_metadata(self) -> tuple:
        return (
            SCHEMA_VERSION,
            self.num_classes,
            self.class_names,
            self.class_prompts,
            self.has_alias,
            self.boundary_radii,
            self.ignore_index,
            REGION_NAMES,
            EVENT_NAMES,
            ABSOLUTE_FEATURE_NAMES,
            SIGNED_FEATURE_NAMES,
            _ABSOLUTE_HISTOGRAM,
            _SIGNED_HISTOGRAM,
            tuple(sorted(self._output_sizes)),
        )

    def reduce_distributed(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return

        metadata = self._distributed_metadata()
        world_size = dist.get_world_size()
        gathered_metadata = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metadata, metadata)
        if any(value != metadata for value in gathered_metadata):
            raise ValueError(
                "boundary causal metadata differs across ranks"
            )

        for tensor in (
            self._processed,
            self._region_counts,
            self._event_counts,
            self._moment_count,
            self._moment_sum,
            self._moment_sq_sum,
            self._absolute_hist,
            self._signed_hist,
        ):
            dist.all_reduce(tensor)

        gathered_ids = [None for _ in range(world_size)]
        gathered_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_ids, self._sample_ids)
        dist.all_gather_object(gathered_rows, self._image_rows)
        self._sample_ids = [
            str(sample_id)
            for rank_ids in gathered_ids
            for sample_id in (rank_ids or [])
        ]
        self._image_rows = [
            dict(row)
            for rank_rows in gathered_rows
            for row in (rank_rows or [])
        ]

    def finalize(self) -> dict:
        region_counts = {}
        ratios = {}
        for class_id in range(self.num_classes):
            class_key = str(class_id)
            region_counts[class_key] = {}
            ratios[class_key] = {}
            for radius_index, radius in enumerate(self.boundary_radii):
                radius_key = str(radius)
                region_counts[class_key][radius_key] = {
                    region_name: int(
                        self._region_counts[
                            radius_index,
                            class_id,
                            region_index,
                        ].item()
                    )
                    for region_index, region_name in enumerate(REGION_NAMES)
                }
                ratios[class_key][radius_key] = {}
                for region_index, region_name in enumerate(REGION_NAMES):
                    ratio_values = {}
                    for ratio_name, (
                        numerator_name,
                        denominator_name,
                    ) in _RATIO_FIELDS.items():
                        numerator = int(
                            self._event_counts[
                                radius_index,
                                class_id,
                                EVENT_NAMES.index(numerator_name),
                                region_index,
                            ].item()
                        )
                        denominator = int(
                            self._event_counts[
                                radius_index,
                                class_id,
                                EVENT_NAMES.index(denominator_name),
                                region_index,
                            ].item()
                        )
                        ratio_values[ratio_name] = self._ratio(
                            numerator,
                            denominator,
                        )
                    ratios[class_key][radius_key][
                        region_name
                    ] = ratio_values

        event_counts = {}
        score_summaries = {}
        for event_index, event_name in enumerate(EVENT_NAMES):
            event_counts[event_name] = {}
            score_summaries[event_name] = {}
            for class_id in range(self.num_classes):
                class_key = str(class_id)
                event_counts[event_name][class_key] = {}
                score_summaries[event_name][class_key] = {}
                for radius_index, radius in enumerate(
                    self.boundary_radii
                ):
                    radius_key = str(radius)
                    event_counts[event_name][class_key][radius_key] = {}
                    score_summaries[event_name][class_key][
                        radius_key
                    ] = {}
                    for region_index, region_name in enumerate(
                        REGION_NAMES
                    ):
                        event_counts[event_name][class_key][radius_key][
                            region_name
                        ] = int(
                            self._event_counts[
                                radius_index,
                                class_id,
                                event_index,
                                region_index,
                            ].item()
                        )
                        score_summaries[event_name][class_key][radius_key][
                            region_name
                        ] = {
                            feature_name: self._feature_report(
                                feature_name=feature_name,
                                radius_index=radius_index,
                                class_id=class_id,
                                event_index=event_index,
                                region_index=region_index,
                            )
                            for feature_name in FEATURE_NAMES
                        }

        sample_count = len(self._sample_ids)
        unique_count = len(set(self._sample_ids))
        return {
            "schema_version": SCHEMA_VERSION,
            "units": {
                "distance": "pixels",
                "counts": "pixels",
                "scores": "probability",
            },
            "processed": int(self._processed[0].item()),
            "adapted": int(self._processed[1].item()),
            "unique_samples": int(unique_count),
            "duplicate_sample_ids": int(sample_count - unique_count),
            "class_names": list(self.class_names),
            "class_prompts": [list(values) for values in self.class_prompts],
            "has_alias": list(self.has_alias),
            "boundary_radii": list(self.boundary_radii),
            "output_sizes": [
                [int(height), int(width)]
                for height, width in sorted(self._output_sizes)
            ],
            "histogram_definitions": {
                "absolute": {
                    "low": _ABSOLUTE_HISTOGRAM[0],
                    "high": _ABSOLUTE_HISTOGRAM[1],
                    "bins": _ABSOLUTE_HISTOGRAM[2],
                },
                "signed": {
                    "low": _SIGNED_HISTOGRAM[0],
                    "high": _SIGNED_HISTOGRAM[1],
                    "bins": _SIGNED_HISTOGRAM[2],
                },
            },
            "region_counts": region_counts,
            "event_counts": event_counts,
            "ratios": ratios,
            "score_summaries": score_summaries,
            "image_rows": sorted(
                self._image_rows,
                key=lambda row: (
                    str(row["sample_id"]),
                    int(row["class_id"]),
                    int(row["radius"]),
                ),
            ),
            "validation": dict(self._validation),
            "dense_outputs_serialized": False,
        }
