#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SELECTION_DIRECTIONS = {
    "presence": "max",
    "selected_score_mean": "max",
    "selected_score_q10": "max",
    "same_class_support_mean": "max",
    "same_class_agreement_mean": "max",
    "foreign_max_mean": "min",
    "margin_mean": "max",
    "margin_q10": "max",
    "candidate_wins_foreign_fraction": "max",
    "gt_topk_precision": "max",
}

TRIAL_FEATURE_DIRECTIONS = {
    "trial_loss": "min",
    "own_core_win_retention": "max",
    "own_core_margin_drift": "max",
    "own_shoulder_win_retention": "max",
    "own_shoulder_margin_drift": "max",
    "foreign_core_new_takeover_rate": "min",
    "foreign_core_margin_drift": "min",
    "foreign_core_positive_erosion": "min",
    "foreign_shoulder_new_takeover_rate": "min",
    "foreign_shoulder_margin_drift": "min",
    "foreign_shoulder_positive_erosion": "min",
    "canonical_anchor_agreement": "max",
    "raw_anchor_agreement": "max",
    "raw_class_expansion_rate": "min",
    "raw_class_contraction_rate": "min",
    "gt_trial_delta_miou": "max",
}

GRADIENT_FEATURE_DIRECTIONS = {
    "gradient_norm_max": "max",
    "gradient_norm_min": "min",
    "stable_norm_ratio_max": "max",
    "stable_norm_ratio_min": "min",
    "stable_cosine": "max",
    "stable_conflict": "min",
    "stable_sign_agreement": "max",
    "consensus_cosine": "max",
    "consensus_sign_agreement": "max",
    "mean_pairwise_cosine": "max",
    "min_pairwise_cosine": "max",
    "positive_pair_fraction": "max",
}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        average_rank = 0.5 * float(offset + end - 1)
        for position in range(offset, end):
            ranks[order[position]] = average_rank
        offset = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if denominator <= 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_delta, right_delta)
    ) / denominator


def _spearman(left: list[float], right: list[float]) -> float | None:
    return _pearson(_rank(left), _rank(right))


def _image_head_fields(run: dict, image: dict, head: str) -> dict:
    if str(run["primary_final_head"]) == str(head):
        return image
    try:
        return image["final_heads"][head]
    except KeyError as error:
        raise ValueError(
            f"image {image.get('sample_id')!r} lacks final head {head!r}"
        ) from error


def _global_head_fields(run: dict, head: str) -> dict:
    try:
        return run["final_heads"][head]
    except KeyError as error:
        raise ValueError(f"run lacks final head {head!r}") from error


def _index_images(run: dict) -> dict[str, dict]:
    images = {
        str(image["sample_id"]): image
        for image in run.get("images", [])
    }
    if len(images) != len(run.get("images", [])):
        raise ValueError("run contains duplicate sample IDs")
    return images


def _candidate_rows(diagnostic_run: dict, class_id: int) -> dict[str, dict[str, dict]]:
    indexed: dict[str, dict[str, dict]] = {}
    for image in diagnostic_run.get("images", []):
        rows = [
            row
            for row in image.get("teacher_prompt_candidates", [])
            if int(row["class_id"]) == int(class_id)
        ]
        if not rows:
            continue
        indexed[str(image["sample_id"])] = {
            str(row["query"]): row for row in rows
        }
    if not indexed:
        raise ValueError(
            f"diagnostic run contains no candidate rows for class {class_id}"
        )
    return indexed


def _candidate_summary(
    candidate_by_sample: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for candidates in candidate_by_sample.values():
        for query, row in candidates.items():
            grouped.setdefault(query, []).append(row)
    output = {}
    mean_fields = tuple(
        feature
        for feature in SELECTION_DIRECTIONS
        if feature != "gt_topk_precision"
    )
    for query, rows in grouped.items():
        evaluable = sum(int(row.get("gt_topk_evaluable", 0)) for row in rows)
        correct = sum(int(row.get("gt_topk_correct", 0)) for row in rows)
        means = {}
        for field in mean_fields:
            values = [
                float(row[field])
                for row in rows
                if row.get(field) is not None
            ]
            means[field] = sum(values) / len(values) if values else None
        output[query] = {
            "images": len(rows),
            "present_images": sum(bool(row.get("present")) for row in rows),
            "selected_images": sum(
                int(row.get("selected_pixels", 0)) > 0 for row in rows
            ),
            "selected_pixels": sum(
                int(row.get("selected_pixels", 0)) for row in rows
            ),
            "weighted_gt_topk_precision": (
                float(correct) / float(evaluable) if evaluable else None
            ),
            "feature_means": means,
        }
    return output


def _teacher_trial_rows(
    diagnostic_run: dict,
    class_id: int,
) -> dict[str, dict[str, dict]]:
    indexed = {}
    for image in diagnostic_run.get("images", []):
        rows = [
            row
            for row in image.get("teacher_prompt_trials", [])
            if int(row.get("class_id", class_id)) == int(class_id)
        ]
        if rows:
            indexed[str(image["sample_id"])] = {
                str(row["query"]): row for row in rows
            }
    return indexed


def _teacher_gradient_rows(
    diagnostic_run: dict,
    class_id: int,
) -> dict[str, dict[str, dict]]:
    indexed = {}
    for image in diagnostic_run.get("images", []):
        rows = [
            row
            for row in image.get("teacher_prompt_gradients", [])
            if int(row.get("class_id", class_id)) == int(class_id)
        ]
        if rows:
            indexed[str(image["sample_id"])] = {
                str(row["query"]): row for row in rows
            }
    return indexed


def _nested_value(row: dict, *keys: str):
    value = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _trial_features(row: dict, *, head: str) -> dict:
    return {
        "trial_loss": row.get("loss"),
        "own_core_win_retention": _nested_value(
            row, "transfer", "own_core", "win_retention"
        ),
        "own_core_margin_drift": _nested_value(
            row, "transfer", "own_core", "margin_drift_mean"
        ),
        "own_shoulder_win_retention": _nested_value(
            row, "transfer", "own_shoulder", "win_retention"
        ),
        "own_shoulder_margin_drift": _nested_value(
            row, "transfer", "own_shoulder", "margin_drift_mean"
        ),
        "foreign_core_new_takeover_rate": _nested_value(
            row, "transfer", "foreign_core", "new_takeover_rate"
        ),
        "foreign_core_margin_drift": _nested_value(
            row, "transfer", "foreign_core", "margin_drift_mean"
        ),
        "foreign_core_positive_erosion": _nested_value(
            row, "transfer", "foreign_core", "positive_erosion_mean"
        ),
        "foreign_shoulder_new_takeover_rate": _nested_value(
            row, "transfer", "foreign_shoulder", "new_takeover_rate"
        ),
        "foreign_shoulder_margin_drift": _nested_value(
            row, "transfer", "foreign_shoulder", "margin_drift_mean"
        ),
        "foreign_shoulder_positive_erosion": _nested_value(
            row, "transfer", "foreign_shoulder", "positive_erosion_mean"
        ),
        "canonical_anchor_agreement": _nested_value(
            row, "transfer", "canonical_anchor_agreement"
        ),
        "raw_anchor_agreement": _nested_value(
            row, "transfer", "raw_anchor_agreement"
        ),
        "raw_class_expansion_rate": _nested_value(
            row, "transfer", "raw_class_expansion_rate"
        ),
        "raw_class_contraction_rate": _nested_value(
            row, "transfer", "raw_class_contraction_rate"
        ),
        "gt_trial_delta_miou": _nested_value(
            row,
            "gt_audit_final_heads",
            head,
            "delta_miou",
        ),
    }


def _gradient_features(row: dict) -> dict:
    gradient_norm = row.get("gradient_norm")
    stable_norm_ratio = row.get("stable_norm_ratio")
    return {
        "gradient_norm_max": gradient_norm,
        "gradient_norm_min": gradient_norm,
        "stable_norm_ratio_max": stable_norm_ratio,
        "stable_norm_ratio_min": stable_norm_ratio,
        "stable_cosine": row.get("stable_cosine"),
        "stable_conflict": row.get("stable_conflict"),
        "stable_sign_agreement": row.get("stable_sign_agreement"),
        "consensus_cosine": row.get("consensus_cosine"),
        "consensus_sign_agreement": row.get(
            "consensus_sign_agreement"
        ),
        "mean_pairwise_cosine": row.get("mean_pairwise_cosine"),
        "min_pairwise_cosine": row.get("min_pairwise_cosine"),
        "positive_pair_fraction": row.get("positive_pair_fraction"),
    }


def _oracle_teacher_summary(records: list[dict]) -> tuple[dict, dict]:
    by_sample: dict[str, dict[str, float]] = {}
    for record in records:
        by_sample.setdefault(record["sample_id"], {})[record["teacher"]] = (
            float(record["delta_miou"])
        )
    teachers = sorted({record["teacher"] for record in records})
    best_counts = {teacher: 0 for teacher in teachers}
    for utilities in by_sample.values():
        best_teacher = max(
            teachers,
            key=lambda teacher: (utilities[teacher], teacher),
        )
        best_counts[best_teacher] += 1
    pairwise = {}
    for left_index, left in enumerate(teachers):
        for right in teachers[left_index + 1 :]:
            row = {left: 0, right: 0, "ties": 0}
            for utilities in by_sample.values():
                difference = utilities[left] - utilities[right]
                if abs(difference) <= 1e-12:
                    row["ties"] += 1
                elif difference > 0.0:
                    row[left] += 1
                else:
                    row[right] += 1
            pairwise[f"{left}__vs__{right}"] = row
    return best_counts, pairwise


def _feature_summary(records: list[dict], feature: str) -> dict:
    usable = [
        record for record in records if record["features"].get(feature) is not None
    ]
    raw_feature = [float(record["features"][feature]) for record in usable]
    raw_utility = [float(record["delta_miou"]) for record in usable]

    by_sample: dict[str, list[dict]] = {}
    for record in usable:
        by_sample.setdefault(record["sample_id"], []).append(record)
    centered_feature: list[float] = []
    centered_utility: list[float] = []
    for sample_records in by_sample.values():
        feature_mean = sum(
            float(record["features"][feature]) for record in sample_records
        ) / len(sample_records)
        utility_mean = sum(
            float(record["delta_miou"]) for record in sample_records
        ) / len(sample_records)
        centered_feature.extend(
            float(record["features"][feature]) - feature_mean
            for record in sample_records
        )
        centered_utility.extend(
            float(record["delta_miou"]) - utility_mean
            for record in sample_records
        )
    return {
        "count": len(usable),
        "pooled_spearman": _spearman(raw_feature, raw_utility),
        "within_image_spearman": _spearman(
            centered_feature,
            centered_utility,
        ),
    }


def _selection_summary(records: list[dict], feature: str, direction: str) -> dict:
    by_sample: dict[str, list[dict]] = {}
    for record in records:
        if record["features"].get(feature) is not None:
            by_sample.setdefault(record["sample_id"], []).append(record)
    correct = 0
    regrets = []
    selected_counts: dict[str, int] = {}
    for sample_records in by_sample.values():
        if len(sample_records) < 2:
            continue
        reverse = direction == "max"
        selected = sorted(
            sample_records,
            key=lambda record: (
                float(record["features"][feature]),
                record["teacher"],
            ),
            reverse=reverse,
        )[0]
        best_utility = max(float(record["delta_miou"]) for record in sample_records)
        selected_utility = float(selected["delta_miou"])
        regret = best_utility - selected_utility
        regrets.append(regret)
        correct += int(abs(regret) <= 1e-12)
        selected_counts[selected["teacher"]] = (
            selected_counts.get(selected["teacher"], 0) + 1
        )
    return {
        "direction": direction,
        "images": len(regrets),
        "accuracy": float(correct) / len(regrets) if regrets else None,
        "mean_regret": sum(regrets) / len(regrets) if regrets else None,
        "selected_teacher_counts": selected_counts,
        "gt_only": feature.startswith("gt_"),
    }


def analyze_interventions(
    *,
    diagnostic_run: dict,
    teacher_runs: dict[str, dict],
    heads: tuple[str, ...],
    class_id: int,
) -> dict:
    if not teacher_runs:
        raise ValueError("at least one teacher intervention run is required")
    candidate_by_sample = _candidate_rows(diagnostic_run, class_id)
    trial_by_sample = _teacher_trial_rows(diagnostic_run, class_id)
    gradient_by_sample = _teacher_gradient_rows(diagnostic_run, class_id)
    run_images = {
        teacher: _index_images(run) for teacher, run in teacher_runs.items()
    }
    expected_samples = set(candidate_by_sample)
    for teacher, images in run_images.items():
        if set(images) != expected_samples:
            raise ValueError(
                f"teacher {teacher!r} sample IDs do not match diagnostic run"
            )
        missing_prompt = [
            sample_id
            for sample_id, candidates in candidate_by_sample.items()
            if teacher not in candidates
        ]
        if missing_prompt:
            raise ValueError(
                f"teacher {teacher!r} is absent from candidate rows"
            )

    output = {
        "class_id": int(class_id),
        "teachers": list(teacher_runs),
        "candidate_summary": _candidate_summary(candidate_by_sample),
        "heads": {},
    }
    for head in heads:
        records = []
        interventions = {}
        for teacher, run in teacher_runs.items():
            global_fields = _global_head_fields(run, head)
            baseline_per_class = global_fields["baseline_per_class_iou"]
            tta_per_class = global_fields["tta_per_class_iou"]
            interventions[teacher] = {
                "global_baseline_miou": float(global_fields["baseline_miou"]),
                "global_tta_miou": float(global_fields["tta_miou"]),
                "global_delta_miou": float(global_fields["delta_miou"]),
                "global_class_delta_iou": float(tta_per_class[class_id])
                - float(baseline_per_class[class_id]),
            }
            for sample_id in sorted(expected_samples):
                fields = _image_head_fields(
                    run,
                    run_images[teacher][sample_id],
                    head,
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "teacher": teacher,
                        "delta_miou": float(fields["delta_miou"]),
                        "class_delta_iou": float(
                            fields["tta_per_class_iou"][class_id]
                        )
                        - float(fields["baseline_per_class_iou"][class_id]),
                        "features": candidate_by_sample[sample_id][teacher],
                        "trial_features": (
                            _trial_features(
                                trial_by_sample[sample_id][teacher],
                                head=head,
                            )
                            if sample_id in trial_by_sample
                            and teacher in trial_by_sample[sample_id]
                            else None
                        ),
                        "gradient_features": (
                            _gradient_features(
                                gradient_by_sample[sample_id][teacher]
                            )
                            if sample_id in gradient_by_sample
                            and teacher in gradient_by_sample[sample_id]
                            else None
                        ),
                    }
                )

        feature_correlations = {
            feature: _feature_summary(records, feature)
            for feature in SELECTION_DIRECTIONS
        }
        feature_selection = {
            feature: _selection_summary(records, feature, direction)
            for feature, direction in SELECTION_DIRECTIONS.items()
        }
        best_counts, pairwise_wins = _oracle_teacher_summary(records)
        trial_records = [
            {
                **record,
                "features": record["trial_features"],
            }
            for record in records
            if record["trial_features"] is not None
        ]
        output["heads"][head] = {
            "sample_count": len(expected_samples),
            "interventions": interventions,
            "feature_correlations": feature_correlations,
            "feature_selection": feature_selection,
            "oracle_best_teacher_counts": best_counts,
            "pairwise_wins": pairwise_wins,
        }
        if trial_records:
            output["heads"][head].update(
                {
                    "trial_feature_correlations": {
                        feature: _feature_summary(trial_records, feature)
                        for feature in TRIAL_FEATURE_DIRECTIONS
                    },
                    "trial_feature_selection": {
                        feature: _selection_summary(
                            trial_records,
                            feature,
                            direction,
                        )
                        for feature, direction in (
                            TRIAL_FEATURE_DIRECTIONS.items()
                        )
                    },
                }
            )
        gradient_records = [
            {
                **record,
                "features": record["gradient_features"],
            }
            for record in records
            if record["gradient_features"] is not None
        ]
        if gradient_records:
            output["heads"][head].update(
                {
                    "gradient_feature_correlations": {
                        feature: _feature_summary(
                            gradient_records,
                            feature,
                        )
                        for feature in GRADIENT_FEATURE_DIRECTIONS
                    },
                    "gradient_feature_selection": {
                        feature: _selection_summary(
                            gradient_records,
                            feature,
                            direction,
                        )
                        for feature, direction in (
                            GRADIENT_FEATURE_DIRECTIONS.items()
                        )
                    },
                }
            )
    return output


def _parse_teacher_run(value: str) -> tuple[str, Path]:
    teacher, separator, path = str(value).partition("=")
    if not separator or not teacher.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "teacher run must use PROMPT=RESULT_JSON"
        )
    return teacher.strip(), Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze causal TTA utility of teacher prompt interventions."
    )
    parser.add_argument("--diagnostic-run", required=True, type=Path)
    parser.add_argument(
        "--teacher-run",
        action="append",
        required=True,
        type=_parse_teacher_run,
    )
    parser.add_argument("--head", action="append", required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic_run = json.loads(args.diagnostic_run.read_text(encoding="utf-8"))
    teacher_runs = {
        teacher: json.loads(path.read_text(encoding="utf-8"))
        for teacher, path in args.teacher_run
    }
    result = analyze_interventions(
        diagnostic_run=diagnostic_run,
        teacher_runs=teacher_runs,
        heads=tuple(args.head),
        class_id=args.class_id,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
