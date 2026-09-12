from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence

from lora_tta.boundary_causal_diagnostic import (
    DEFAULT_BOUNDARY_RADII,
    EVENT_NAMES,
    REGION_NAMES,
    SCHEMA_VERSION,
)


PARITY_FIELDS = (
    "f00_matches_pre_raw",
    "f11_matches_post_raw",
    "canonical_pre_matches",
    "canonical_post_matches",
)
CAUSAL_METRICS = (
    "canonical_direct_miou",
    "canonical_visible_miou",
    "static_alias_shielding_miou",
    "alias_only_transfer_miou",
    "interaction_miou",
    "total_synonym_tta_gain_miou",
    "canonical_shapley_miou",
    "alias_shapley_miou",
)
VALIDATION_FIELDS = (
    "positive_region_partition",
    "negative_region_partition",
    "region_monotonicity",
    "fn_partition",
    "fp_partition",
    "pre_alias_partition",
    "post_alias_partition",
)
FOCUS_PROMPTS = {
    "loveda": {
        "weak": ("forest", "soil"),
        "controls": (),
    },
    "potsdam": {
        "weak": ("clutter", "tree"),
        "controls": ("road", "grass"),
    },
}
MIN_DENOMINATOR_SAMPLES = 10


def normalize_dataset_name(value: str) -> str:
    name = Path(str(value)).stem.lower().removeprefix("cfg_")
    if name.startswith("loveda"):
        return "loveda"
    if name.startswith("potsdam"):
        return "potsdam"
    raise ValueError(f"unsupported overlap dataset {value!r}")


def _mapping(value, *, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _finite_number(value, *, field: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(output):
        raise ValueError(f"{field} must be finite")
    return output


def _require_count(value, *, field: str) -> int:
    if type(value) is not int or int(value) < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _validate_causal(cross_time: Mapping) -> dict[str, float]:
    parity = _mapping(cross_time.get("parity"), field="cross-time parity")
    if set(parity) != set(PARITY_FIELDS) or not all(
        parity[field] is True for field in PARITY_FIELDS
    ):
        raise ValueError("cross-time parity failed or is incomplete")
    causal = _mapping(cross_time.get("causal"), field="cross-time causal")
    values = {
        name: _finite_number(causal.get(name), field=name)
        for name in CAUSAL_METRICS
    }
    identity = _finite_number(
        causal.get("identity_error_miou"),
        field="identity_error_miou",
    )
    if abs(identity) > 1e-6:
        raise ValueError("cross-time causal identity failed")
    tolerance = 1e-5
    if abs(
        values["canonical_visible_miou"]
        - values["canonical_direct_miou"]
        - values["static_alias_shielding_miou"]
    ) > tolerance:
        raise ValueError("canonical visible causal identity failed")
    if abs(
        values["total_synonym_tta_gain_miou"]
        - values["canonical_visible_miou"]
        - values["alias_only_transfer_miou"]
        - values["interaction_miou"]
    ) > tolerance:
        raise ValueError("synonym TTA causal identity failed")
    if abs(
        values["total_synonym_tta_gain_miou"]
        - values["canonical_shapley_miou"]
        - values["alias_shapley_miou"]
    ) > tolerance:
        raise ValueError("Shapley causal identity failed")
    values["identity_error_miou"] = identity
    return values


def _validate_output_sizes(value) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("boundary output_sizes must be nonempty")
    output = []
    for size in value:
        if (
            not isinstance(size, list)
            or len(size) != 2
            or type(size[0]) is not int
            or type(size[1]) is not int
            or size[0] <= 0
            or size[1] <= 0
        ):
            raise ValueError("boundary output_sizes contain an invalid size")
        output.append((int(size[0]), int(size[1])))
    return tuple(output)


def _validate_row(
    row,
    *,
    class_names: Sequence[str],
    radii: tuple[int, ...],
) -> dict:
    if not isinstance(row, Mapping):
        raise ValueError("compact rows must be mappings")
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        raise ValueError("compact row sample_id must not be empty")
    class_id = row.get("class_id")
    radius = row.get("radius")
    if type(class_id) is not int or not 0 <= class_id < len(class_names):
        raise ValueError("compact row class_id is invalid")
    if type(radius) is not int or int(radius) not in radii:
        raise ValueError("compact row radius is invalid")
    if str(row.get("class_name")) != str(class_names[class_id]):
        raise ValueError("compact row class_name does not match class_id")
    regions = _mapping(row.get("regions"), field="compact row regions")
    if set(regions) != set(REGION_NAMES):
        raise ValueError("compact row regions are incomplete")
    normalized_regions = {
        name: _require_count(regions[name], field=f"region {name}")
        for name in REGION_NAMES
    }
    events = _mapping(row.get("events"), field="compact row events")
    if set(events) != set(EVENT_NAMES):
        raise ValueError("compact row events are incomplete")
    normalized_events = {}
    for event_name in EVENT_NAMES:
        values = _mapping(
            events[event_name],
            field=f"compact event {event_name}",
        )
        if set(values) != set(REGION_NAMES):
            raise ValueError(
                f"compact event {event_name} regions are incomplete"
            )
        normalized_events[event_name] = {
            region_name: _require_count(
                values[region_name],
                field=f"{event_name}.{region_name}",
            )
            for region_name in REGION_NAMES
        }
    return {
        "sample_id": sample_id,
        "adapted": bool(row.get("adapted", False)),
        "class_id": int(class_id),
        "class_name": str(class_names[class_id]),
        "radius": int(radius),
        "regions": normalized_regions,
        "events": normalized_events,
    }


def validate_run(payload: Mapping, *, expected_samples: int) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("run payload must be a mapping")
    expected_samples = int(expected_samples)
    if expected_samples <= 0:
        raise ValueError("expected_samples must be positive")
    if int(payload.get("processed", -1)) != expected_samples:
        raise ValueError("top-level processed count is incomplete")
    dataset = normalize_dataset_name(
        str(payload.get("dataset") or payload.get("config") or "")
    )
    cross_time = _mapping(
        payload.get("cross_time_causal"),
        field="cross_time_causal",
    )
    if int(cross_time.get("processed", -1)) != expected_samples:
        raise ValueError("cross-time processed count is incomplete")
    causal = _validate_causal(cross_time)
    boundary = _mapping(
        cross_time.get("boundary_overlap"),
        field="boundary_overlap",
    )
    if int(boundary.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("boundary schema version mismatch")
    if int(boundary.get("processed", -1)) != expected_samples:
        raise ValueError("boundary processed count is incomplete")
    if int(boundary.get("unique_samples", -1)) != expected_samples:
        raise ValueError("boundary unique sample count is incomplete")
    if int(boundary.get("duplicate_sample_ids", -1)) != 0:
        raise ValueError("boundary report contains duplicate sample IDs")
    radii = tuple(int(value) for value in boundary.get("boundary_radii", ()))
    if radii != tuple(DEFAULT_BOUNDARY_RADII):
        raise ValueError("boundary radii do not match the preregistration")
    output_sizes = _validate_output_sizes(boundary.get("output_sizes"))
    for field in (
        "region_counts",
        "event_counts",
        "ratios",
        "score_summaries",
    ):
        _mapping(boundary.get(field), field=f"boundary {field}")
    validation = _mapping(
        boundary.get("validation"),
        field="boundary validation",
    )
    if set(validation) != set(VALIDATION_FIELDS) or not all(
        validation[field] is True for field in VALIDATION_FIELDS
    ):
        raise ValueError("boundary validation failed or is incomplete")
    if boundary.get("dense_outputs_serialized") is not False:
        raise ValueError("boundary report does not certify compact output")

    class_names_raw = boundary.get("class_names")
    class_prompts_raw = boundary.get("class_prompts")
    has_alias_raw = boundary.get("has_alias")
    if (
        not isinstance(class_names_raw, list)
        or not class_names_raw
        or not isinstance(class_prompts_raw, list)
        or not isinstance(has_alias_raw, list)
        or len(class_prompts_raw) != len(class_names_raw)
        or len(has_alias_raw) != len(class_names_raw)
    ):
        raise ValueError("boundary class metadata is incomplete")
    class_names = tuple(str(value) for value in class_names_raw)
    class_prompts = []
    class_lookup = {}
    for class_id, prompts in enumerate(class_prompts_raw):
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("every boundary class must expose prompts")
        normalized = tuple(str(prompt) for prompt in prompts)
        class_prompts.append(normalized)
        if bool(has_alias_raw[class_id]) != (len(normalized) > 1):
            raise ValueError("boundary has_alias metadata is inconsistent")
        for prompt in normalized:
            key = prompt.strip().lower()
            if key in class_lookup and class_lookup[key] != class_id:
                raise ValueError(f"prompt {prompt!r} maps to multiple classes")
            class_lookup[key] = class_id

    rows_raw = boundary.get("image_rows")
    if not isinstance(rows_raw, list):
        raise ValueError("boundary compact rows are missing")
    rows = [
        _validate_row(row, class_names=class_names, radii=radii)
        for row in rows_raw
    ]
    sample_ids = tuple(sorted({row["sample_id"] for row in rows}))
    if len(sample_ids) != expected_samples:
        raise ValueError("boundary compact rows have incomplete sample IDs")
    expected_keys = {
        (sample_id, class_id, radius)
        for sample_id in sample_ids
        for class_id in range(len(class_names))
        for radius in radii
    }
    row_keys = [
        (row["sample_id"], row["class_id"], row["radius"])
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)) or set(row_keys) != expected_keys:
        raise ValueError("boundary compact rows are incomplete or duplicated")
    row_index = {
        (row["sample_id"], row["class_id"], row["radius"]): row
        for row in rows
    }
    correction_survival = cross_time.get("correction_survival")
    if correction_survival is not None and not isinstance(
        correction_survival, Mapping
    ):
        raise ValueError("correction_survival must be a mapping")
    return {
        "dataset": dataset,
        "causal": causal,
        "boundary": boundary,
        "class_names": class_names,
        "class_prompts": tuple(class_prompts),
        "class_lookup": class_lookup,
        "sample_ids": sample_ids,
        "rows": rows,
        "row_index": row_index,
        "output_sizes": output_sizes,
        "correction_survival": correction_survival,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile values must not be empty")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ratio(
    pairs: Sequence[tuple[int, int]],
    *,
    seed: int,
    replicates: int,
) -> dict | None:
    pairs = [(int(left), int(right)) for left, right in pairs]
    if not pairs:
        raise ValueError("bootstrap pairs must not be empty")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    numerator = sum(value[0] for value in pairs)
    denominator = sum(value[1] for value in pairs)
    if denominator == 0:
        return None
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        sampled_denominator = sum(value[1] for value in sampled)
        if sampled_denominator == 0:
            continue
        estimates.append(
            sum(value[0] for value in sampled) / sampled_denominator
        )
    if not estimates:
        return None
    return {
        "point": numerator / denominator,
        "ci95": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "numerator": int(numerator),
        "denominator": int(denominator),
        "denominator_samples": sum(right > 0 for _, right in pairs),
        "total_samples": len(pairs),
        "valid_replicates": len(estimates),
    }


def bootstrap_mean(
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> dict:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("bootstrap values must not be empty")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(replicates)):
        sampled = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sampled) / len(sampled))
    return {
        "point": sum(values) / len(values),
        "ci95": [
            _percentile(estimates, 0.025),
            _percentile(estimates, 0.975),
        ],
        "sample_count": len(values),
        "valid_replicates": len(estimates),
    }


def summarize_correction_survival(
    report: Mapping,
    *,
    radius: int,
    seed: int,
    replicates: int,
) -> dict:
    if int(report.get("schema_version", -1)) != 1:
        raise ValueError("correction survival schema mismatch")
    if report.get("dense_outputs_serialized") is not False:
        raise ValueError("correction survival output is not compact")
    validation = _mapping(
        report.get("validation"), field="correction survival validation"
    )
    if not validation or not all(value is True for value in validation.values()):
        raise ValueError("correction survival validation failed")
    if int(radius) not in {
        int(value) for value in report.get("boundary_radii", ())
    }:
        raise ValueError("correction survival radius is unavailable")
    rows = report.get("image_rows")
    if not isinstance(rows, list):
        raise ValueError("correction survival image rows are missing")
    sample_ids = {str(row.get("sample_id", "")) for row in rows}
    sample_ids.discard("")
    if int(report.get("duplicate_sample_ids", -1)) != 0:
        raise ValueError("correction survival contains duplicate samples")
    if int(report.get("processed", -1)) != len(sample_ids):
        raise ValueError("correction survival processed count is incomplete")
    if len(sample_ids) != int(report.get("unique_samples", -1)):
        raise ValueError("correction survival sample IDs are incomplete")

    metric_events = {
        "pre_solved_rate": (
            ("correction_pre_solved_retained", "correction_pre_solved_degraded"),
            ("canonical_correction",),
        ),
        "static_shielding_rate": (
            ("correction_dynamic_recovery", "correction_static_unresolved"),
            (
                "correction_visible_preserved",
                "correction_dynamic_overwrite",
                "correction_dynamic_recovery",
                "correction_static_unresolved",
            ),
        ),
        "dynamic_overwrite_rate": (
            ("correction_dynamic_overwrite",),
            ("correction_visible_preserved", "correction_dynamic_overwrite"),
        ),
        "dynamic_recovery_rate": (
            ("correction_dynamic_recovery",),
            ("correction_dynamic_recovery", "correction_static_unresolved"),
        ),
        "collapse_protection_rate": (
            ("collapse_static_retained", "collapse_dynamic_protection"),
            ("canonical_damage",),
        ),
        "fp_reintroduction_rate": (
            ("fp_reintroduced",),
            ("canonical_fp_removed",),
        ),
    }

    def scoped_rows(adapted_only: bool) -> list[Mapping]:
        return [
            row
            for row in rows
            if int(row.get("radius", -1)) == int(radius)
            and (not adapted_only or bool(row.get("adapted", False)))
        ]

    def event_total(row: Mapping, names: Sequence[str]) -> int:
        events = _mapping(row.get("events"), field="survival row events")
        total = 0
        for name in names:
            regions = _mapping(events.get(name), field=f"survival event {name}")
            total += sum(int(regions.get(region, 0)) for region in REGION_NAMES)
        return total

    scopes = {}
    for scope_index, (scope_name, adapted_only) in enumerate(
        (("all", False), ("adapted_only", True))
    ):
        selected = scoped_rows(adapted_only)
        selected_ids = sorted({str(row["sample_id"]) for row in selected})
        by_sample = {
            sample_id: [
                row for row in selected if str(row["sample_id"]) == sample_id
            ]
            for sample_id in selected_ids
        }
        scope_result = {"sample_count": len(selected_ids)}
        for offset, (metric_name, (numerators, denominators)) in enumerate(
            metric_events.items()
        ):
            pairs = [
                (
                    sum(event_total(row, numerators) for row in sample_rows),
                    sum(event_total(row, denominators) for row in sample_rows),
                )
                for sample_rows in by_sample.values()
            ]
            scope_result[metric_name] = (
                bootstrap_ratio(
                    pairs,
                    seed=int(seed) + scope_index * 100 + offset,
                    replicates=replicates,
                )
                if pairs
                else None
            )

        for offset, direction in enumerate(("alias_down", "alias_flat", "alias_up")):
            pairs = []
            for sample_rows in by_sample.values():
                numerator = 0
                denominator = 0
                for row in sample_rows:
                    coupling = row.get("suppression_coupling")
                    if not isinstance(coupling, Mapping):
                        continue
                    for region in REGION_NAMES:
                        region_values = coupling.get(region)
                        if not isinstance(region_values, Mapping):
                            continue
                        counts = _mapping(
                            region_values.get("counts"),
                            field="suppression counts",
                        )
                        numerator += int(counts.get(direction, 0))
                        denominator += int(
                            counts.get("canonical_suppressed", 0)
                        )
                pairs.append((numerator, denominator))
            scope_result[
                f"{direction}_given_canonical_suppression"
            ] = (
                bootstrap_ratio(
                    pairs,
                    seed=int(seed) + scope_index * 100 + 20 + offset,
                    replicates=replicates,
                )
                if pairs
                else None
            )
        scopes[scope_name] = scope_result

    raw_scopes = _mapping(report.get("scopes"), field="survival scopes")
    return {
        "radius": int(radius),
        "scopes": scopes,
        "transition_mechanisms": {
            scope: list(
                _mapping(raw_scopes.get(scope), field=f"survival scope {scope}").get(
                    "transition_mechanisms", []
                )
            )
            for scope in ("all", "adapted_only")
        },
        "prompt_risk_with_leave_one_out": list(
            report.get("prompt_risk_with_leave_one_out", [])
        ),
    }


def ratio_supported(
    point: float,
    ci95: Sequence[float],
    *,
    denominator_samples: int | None = None,
    total_samples: int | None = None,
) -> bool:
    if denominator_samples is not None or total_samples is not None:
        if denominator_samples is None or total_samples is None:
            raise ValueError(
                "denominator_samples and total_samples must be paired"
            )
        required_samples = min(
            MIN_DENOMINATOR_SAMPLES,
            int(total_samples),
        )
        if int(denominator_samples) < required_samples:
            return False
    return float(point) > 0.5 and float(ci95[0]) > 0.5


def dynamic_harm_supported(
    ci95: Sequence[float],
    alias_only_transfer_miou: float,
) -> bool:
    return (
        float(ci95[0]) > 0.0
        and float(alias_only_transfer_miou) < 0.0
    )


def classify_mechanism(families: Mapping[str, bool]) -> str:
    supported = [name for name, value in families.items() if bool(value)]
    if not supported:
        return "not established"
    if len(supported) == 1:
        return supported[0]
    return "mixed mechanism"


def _event_pairs(
    run: Mapping,
    *,
    class_ids: Sequence[int],
    radius: int,
    region: str,
    numerator_event: str,
    denominator_event: str,
) -> list[tuple[int, int]]:
    output = []
    for sample_id in run["sample_ids"]:
        numerator = 0
        denominator = 0
        for class_id in class_ids:
            row = run["row_index"][(sample_id, int(class_id), int(radius))]
            numerator += row["events"][numerator_event][region]
            denominator += row["events"][denominator_event][region]
        output.append((numerator, denominator))
    return output


def _ratio_cell(
    run: Mapping,
    *,
    class_ids: Sequence[int],
    radius: int,
    region: str,
    numerator_event: str,
    denominator_event: str,
    seed: int,
    replicates: int,
) -> dict:
    result = bootstrap_ratio(
        _event_pairs(
            run,
            class_ids=class_ids,
            radius=radius,
            region=region,
            numerator_event=numerator_event,
            denominator_event=denominator_event,
        ),
        seed=seed,
        replicates=replicates,
    )
    if result is None:
        return {
            "available": False,
            "point": None,
            "ci95": None,
            "numerator": 0,
            "denominator": 0,
            "valid_replicates": 0,
            "supported": False,
        }
    return {
        "available": True,
        **result,
        "supported": ratio_supported(
            result["point"],
            result["ci95"],
            denominator_samples=result["denominator_samples"],
            total_samples=result["total_samples"],
        ),
    }


def _dynamic_net_by_radius(run: Mapping, radius: int) -> list[int]:
    output = []
    class_ids = range(len(run["class_names"]))
    for sample_id in run["sample_ids"]:
        net = 0
        for class_id in class_ids:
            row = run["row_index"][(sample_id, class_id, int(radius))]
            for region in ("inner_boundary", "interior"):
                net += row["events"]["post_alias_harm"][region]
                net -= row["events"]["post_alias_rescue"][region]
        output.append(net)
    return output


def _focus_class_ids(run: Mapping, field: str) -> tuple[tuple[str, int], ...]:
    output = []
    for prompt in FOCUS_PROMPTS[run["dataset"]][field]:
        class_id = run["class_lookup"].get(prompt.lower())
        if class_id is None:
            raise ValueError(
                f"{run['dataset']} is missing preregistered prompt {prompt!r}"
            )
        output.append((prompt, int(class_id)))
    return tuple(output)


def _score_margin_evidence(
    run: Mapping,
    focus_classes: Sequence[tuple[str, int]],
) -> list[dict]:
    summaries = run["boundary"]["score_summaries"]
    event_regions = (
        ("canonical_rescue", ("inner_boundary", "interior")),
        ("fn_pre_solved", ("inner_boundary", "interior")),
        ("canonical_fp_removed", ("outer_boundary", "far_exterior")),
        ("fp_pre_solved", ("outer_boundary", "far_exterior")),
        ("alias_reintroduced_fp", ("outer_boundary", "far_exterior")),
        ("post_alias_harm", ("inner_boundary", "interior")),
        ("post_alias_rescue", ("inner_boundary", "interior")),
    )
    output = []
    for prompt, class_id in focus_classes:
        for event_name, regions in event_regions:
            event_values = summaries.get(event_name)
            if not isinstance(event_values, Mapping):
                continue
            class_values = event_values.get(str(class_id))
            if not isinstance(class_values, Mapping):
                continue
            radius_values = class_values.get("5")
            if not isinstance(radius_values, Mapping):
                continue
            for region in regions:
                features = radius_values.get(region)
                if not isinstance(features, Mapping):
                    continue
                nonempty = {
                    str(name): value
                    for name, value in features.items()
                    if isinstance(value, Mapping)
                }
                if not nonempty:
                    continue
                output.append(
                    {
                        "prompt": prompt,
                        "class_id": int(class_id),
                        "event": event_name,
                        "radius": 5,
                        "region": region,
                        "features": nonempty,
                    }
                )
    return output


def analyze_dataset(
    run: Mapping,
    *,
    seed: int,
    replicates: int,
) -> dict:
    weak_classes = _focus_class_ids(run, "weak")
    control_classes = _focus_class_ids(run, "controls")
    fn_cells = []
    for prompt, class_id in weak_classes:
        for radius in DEFAULT_BOUNDARY_RADII:
            for region in ("inner_boundary", "interior"):
                fn_cells.append(
                    {
                        "prompt": prompt,
                        "class_id": class_id,
                        "radius": radius,
                        "region": region,
                        **_ratio_cell(
                            run,
                            class_ids=(class_id,),
                            radius=radius,
                            region=region,
                            numerator_event="fn_pre_solved",
                            denominator_event="canonical_rescue",
                            seed=seed,
                            replicates=replicates,
                        ),
                    }
                )

    all_class_ids = tuple(range(len(run["class_names"])))
    fp_cells = []
    for radius in DEFAULT_BOUNDARY_RADII:
        for region in ("outer_boundary", "far_exterior"):
            cell = _ratio_cell(
                run,
                class_ids=all_class_ids,
                radius=radius,
                region=region,
                numerator_event="fp_pre_solved",
                denominator_event="canonical_fp_removed",
                seed=seed,
                replicates=replicates,
            )
            if region == "far_exterior":
                cell["supported"] = False
            fp_cells.append(
                {"radius": radius, "region": region, **cell}
            )

    focus_fp_cells = []
    for role, classes in (
        ("weak", weak_classes),
        ("control", control_classes),
    ):
        for prompt, class_id in classes:
            for radius in DEFAULT_BOUNDARY_RADII:
                for region in ("outer_boundary", "far_exterior"):
                    focus_fp_cells.append(
                        {
                            "role": role,
                            "prompt": prompt,
                            "class_id": class_id,
                            "radius": radius,
                            "region": region,
                            **_ratio_cell(
                                run,
                                class_ids=(class_id,),
                                radius=radius,
                                region=region,
                                numerator_event="fp_pre_solved",
                                denominator_event="canonical_fp_removed",
                                seed=seed,
                                replicates=replicates,
                            ),
                        }
                    )

    dynamic_by_radius = {
        radius: _dynamic_net_by_radius(run, radius)
        for radius in DEFAULT_BOUNDARY_RADII
    }
    reference_dynamic = dynamic_by_radius[5]
    if any(
        values != reference_dynamic
        for radius, values in dynamic_by_radius.items()
        if radius != 5
    ):
        raise ValueError(
            "dynamic alias all-positive totals differ across radii"
        )
    dynamic = bootstrap_mean(
        reference_dynamic,
        seed=seed,
        replicates=replicates,
    )
    dynamic["units"] = "net harm pixels per image"
    dynamic["alias_only_transfer_miou"] = run["causal"][
        "alias_only_transfer_miou"
    ]
    dynamic["supported"] = dynamic_harm_supported(
        dynamic["ci95"],
        dynamic["alias_only_transfer_miou"],
    )

    families = {
        "weak-class FN pre-solution": any(
            cell["supported"] for cell in fn_cells
        ),
        "boundary FP pre-contraction": any(
            cell["supported"]
            for cell in fp_cells
            if cell["region"] == "outer_boundary"
        ),
        "dynamic alias harm": bool(dynamic["supported"]),
    }
    result = {
        "validation": {
            "processed": len(run["sample_ids"]),
            "unique_samples": len(run["sample_ids"]),
            "duplicate_sample_ids": 0,
            "output_sizes": [list(value) for value in run["output_sizes"]],
            "passed": True,
        },
        "causal": dict(run["causal"]),
        "focus": {
            "weak": [
                {"prompt": prompt, "class_id": class_id}
                for prompt, class_id in weak_classes
            ],
            "controls": [
                {"prompt": prompt, "class_id": class_id}
                for prompt, class_id in control_classes
            ],
        },
        "fn_pre_solution": fn_cells,
        "fp_pre_contraction": fp_cells,
        "focus_fp_rows": focus_fp_cells,
        "score_margin_evidence": _score_margin_evidence(
            run,
            (*weak_classes, *control_classes),
        ),
        "dynamic_alias_harm": dynamic,
        "families": families,
        "mechanism": classify_mechanism(families),
    }
    if run.get("correction_survival") is not None:
        result["correction_survival"] = summarize_correction_survival(
            run["correction_survival"],
            radius=9,
            seed=seed,
            replicates=replicates,
        )
    return result


def analyze_runs(
    *,
    loveda_payload: Mapping,
    potsdam_payload: Mapping,
    expected_samples: int,
    seed: int,
    replicates: int,
) -> dict:
    loveda = validate_run(
        loveda_payload,
        expected_samples=expected_samples,
    )
    potsdam = validate_run(
        potsdam_payload,
        expected_samples=expected_samples,
    )
    if loveda["dataset"] != "loveda":
        raise ValueError("--loveda input is not a LoveDA run")
    if potsdam["dataset"] != "potsdam":
        raise ValueError("--potsdam input is not a Potsdam run")
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_samples": int(expected_samples),
        "seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "datasets": {
            "loveda": analyze_dataset(
                loveda,
                seed=seed,
                replicates=replicates,
            ),
            "potsdam": analyze_dataset(
                potsdam,
                seed=seed,
                replicates=replicates,
            ),
        },
        "interpretation_limits": [
            "Pixel event shares are not additive mIoU contributions.",
            "The existing cross-time causal identity remains authoritative.",
            "This 200-image pilot establishes mechanisms, not mitigations.",
        ],
    }


def _format_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _ratio_markdown_rows(cells: Sequence[Mapping]) -> list[str]:
    rows = []
    for cell in cells:
        ci95 = cell.get("ci95")
        ci_text = (
            "n/a"
            if ci95 is None
            else f"[{_format_float(ci95[0])}, {_format_float(ci95[1])}]"
        )
        label = str(cell.get("prompt", "all classes"))
        rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(cell["radius"]),
                    str(cell["region"]),
                    _format_float(cell.get("point")),
                    ci_text,
                    str(cell.get("denominator", 0)),
                    str(cell.get("denominator_samples", 0)),
                    str(bool(cell.get("supported", False))),
                )
            )
            + " |"
        )
    return rows


def render_markdown(result: Mapping) -> str:
    lines = [
        "# Synonym-TTA Boundary Overlap Analysis",
        "",
        "## Run Validation",
        "",
        "| Dataset | Processed | Unique | Duplicates | Passed |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, report in result["datasets"].items():
        validation = report["validation"]
        lines.append(
            f"| {dataset} | {validation['processed']} | "
            f"{validation['unique_samples']} | "
            f"{validation['duplicate_sample_ids']} | "
            f"{validation['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Cross-Time Identity",
            "",
            "| Dataset | Metric | mIoU points |",
            "|---|---|---:|",
        ]
    )
    for dataset, report in result["datasets"].items():
        for metric in (*CAUSAL_METRICS, "identity_error_miou"):
            lines.append(
                f"| {dataset} | {metric} | "
                f"{_format_float(report['causal'][metric])} |"
            )
    if any(
        "correction_survival" in report
        for report in result["datasets"].values()
    ):
        lines.extend(
            [
                "",
                "## Canonical Correction Survival",
                "",
                "| Dataset | Scope | Metric | Ratio | 95% CI | Numerator | Denominator |",
                "|---|---|---|---:|---|---:|---:|",
            ]
        )
        metric_names = (
            "pre_solved_rate",
            "static_shielding_rate",
            "dynamic_overwrite_rate",
            "dynamic_recovery_rate",
            "collapse_protection_rate",
            "fp_reintroduction_rate",
            "alias_down_given_canonical_suppression",
            "alias_flat_given_canonical_suppression",
            "alias_up_given_canonical_suppression",
        )
        for dataset, report in result["datasets"].items():
            survival = report.get("correction_survival")
            if not isinstance(survival, Mapping):
                continue
            for scope_name, scope in survival["scopes"].items():
                for metric_name in metric_names:
                    value = scope.get(metric_name)
                    if not isinstance(value, Mapping):
                        continue
                    ci95 = value.get("ci95")
                    ci_text = (
                        "n/a"
                        if ci95 is None
                        else "["
                        + _format_float(ci95[0])
                        + ", "
                        + _format_float(ci95[1])
                        + "]"
                    )
                    lines.append(
                        f"| {dataset} | {scope_name} | {metric_name} | "
                        f"{_format_float(value.get('point'))} | {ci_text} | "
                        f"{value.get('numerator', 0)} | "
                        f"{value.get('denominator', 0)} |"
                    )
    for dataset, heading in (
        ("loveda", "LoveDA Weak Classes"),
        ("potsdam", "Potsdam Weak Classes and Controls"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| Class | Radius | Region | Ratio | 95% CI | Denominator | Support images | Supported |",
                "|---|---:|---|---:|---|---:|---:|---|",
                *_ratio_markdown_rows(
                    result["datasets"][dataset]["fn_pre_solution"]
                ),
                *_ratio_markdown_rows(
                    result["datasets"][dataset]["focus_fp_rows"]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Boundary-Radius Sensitivity",
            "",
            "| Dataset | Class | Radius | Region | Ratio | 95% CI | Denominator | Support images | Supported |",
            "|---|---|---:|---|---:|---|---:|---:|---|",
        ]
    )
    for dataset, report in result["datasets"].items():
        for row in _ratio_markdown_rows(report["fp_pre_contraction"]):
            lines.append(row.replace("| all classes |", f"| {dataset} | all classes |", 1))
    lines.extend(
        [
            "",
            "## Score and Margin Evidence",
            "",
            "| Dataset | Class | Event | Region | Feature means |",
            "|---|---|---|---|---|",
        ]
    )
    for dataset, report in result["datasets"].items():
        for row in report["score_margin_evidence"]:
            means = ", ".join(
                f"{name}={_format_float(values.get('mean'))}"
                for name, values in sorted(row["features"].items())
            )
            lines.append(
                f"| {dataset} | {row['prompt']} | {row['event']} | "
                f"{row['region']} | {means} |"
            )
    lines.extend(["", "## Mechanism Decisions", ""])
    for dataset, report in result["datasets"].items():
        dynamic = report["dynamic_alias_harm"]
        lines.append(f"- {dataset}: **{report['mechanism']}**")
        lines.append(
            f"- {dataset} post-alias net harm: "
            f"{_format_float(dynamic['point'])}, 95% CI "
            f"[{_format_float(dynamic['ci95'][0])}, "
            f"{_format_float(dynamic['ci95'][1])}], "
            f"alias-only transfer {dynamic['alias_only_transfer_miou']:.4f} mIoU."
        )
    lines.extend(["", "## Interpretation Limits", ""])
    lines.extend(
        f"- {value}" for value in result["interpretation_limits"]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze synonym shielding and TTA boundary overlap."
    )
    parser.add_argument("--loveda", required=True, type=Path)
    parser.add_argument("--potsdam", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--expected-samples", type=int, default=200)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    loveda_payload = json.loads(args.loveda.read_text(encoding="utf-8"))
    potsdam_payload = json.loads(args.potsdam.read_text(encoding="utf-8"))
    result = analyze_runs(
        loveda_payload=loveda_payload,
        potsdam_payload=potsdam_payload,
        expected_samples=args.expected_samples,
        seed=args.seed,
        replicates=args.bootstrap_replicates,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
