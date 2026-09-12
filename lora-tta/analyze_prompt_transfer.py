from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


KNOWN_PROMPTS = (
    "person",
    "vehicle",
    "greenery",
    "street",
    "paved road",
    "lawn",
    "facade",
    "wall",
    "sky",
    "sports field",
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


def average_ranks(values: Sequence[float]) -> list[float]:
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def binary_auc(
    scores: Sequence[float],
    labels: Sequence[int | bool],
) -> float | None:
    if len(scores) != len(labels):
        raise ValueError("score/label length mismatch")
    positives = [
        float(score)
        for score, label in zip(scores, labels)
        if bool(label)
    ]
    negatives = [
        float(score)
        for score, label in zip(scores, labels)
        if not bool(label)
    ]
    if not positives or not negatives:
        return None
    favorable = sum(
        (
            1.0
            if positive > negative
            else 0.5
            if positive == negative
            else 0.0
        )
        for positive in positives
        for negative in negatives
    )
    return favorable / float(len(positives) * len(negatives))


def spearman(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths")
    if len(left) < 2:
        return None
    x = average_ranks(left)
    y = average_ranks(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )
    denominator = (
        sum((a - mean_x) ** 2 for a in x)
        * sum((b - mean_y) ** 2 for b in y)
    ) ** 0.5
    return numerator / denominator if denominator else None


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def leave_one_dataset_out(
    dataset_candidates: dict[str, dict[str, float]],
) -> dict[str, dict]:
    if len(dataset_candidates) < 2:
        raise ValueError(
            "leave-one-dataset-out requires at least two datasets"
        )
    candidate_sets = {
        tuple(sorted(values))
        for values in dataset_candidates.values()
    }
    if len(candidate_sets) != 1:
        raise ValueError("datasets expose different routing candidates")
    candidates = next(iter(candidate_sets))
    output = {}
    for held_out, held_values in dataset_candidates.items():
        training = [
            values
            for name, values in dataset_candidates.items()
            if name != held_out
        ]
        selected = max(
            candidates,
            key=lambda key: sum(row[key] for row in training)
            / len(training),
        )
        output[held_out] = {
            "selected": selected,
            "training_macro_value": sum(
                row[selected] for row in training
            )
            / len(training),
            "held_out_value": float(held_values[selected]),
        }
    return output


def _dataset_name(path: Path, payload: dict) -> str:
    explicit = payload.get("dataset")
    if explicit:
        return str(explicit)
    config = str(payload.get("config", ""))
    if config:
        return Path(config).stem.removeprefix("cfg_")
    return path.stem


def _require_routing_report(path: Path, payload: dict) -> dict:
    report = payload.get("prompt_routing_diagnostic")
    if not isinstance(report, dict):
        raise ValueError(f"{path} has no prompt_routing_diagnostic")
    for field in ("pairs", "prompts", "heads"):
        if field not in report:
            raise ValueError(f"{path} routing report is missing {field}")
    if "H1_raw_max" not in report["heads"]:
        raise ValueError(f"{path} routing report is missing H1_raw_max")
    return report


def _causal_summary(path: Path, report: dict) -> dict[str, float]:
    cross_time = report.get("cross_time_causal")
    if not isinstance(cross_time, dict):
        raise ValueError(f"{path} routing report has no cross_time_causal")
    parity = cross_time.get("parity")
    if (
        not isinstance(parity, dict)
        or len(parity) != 4
        or not all(bool(value) for value in parity.values())
    ):
        raise ValueError(f"{path} cross-time head parity failed")
    causal = cross_time.get("causal")
    if not isinstance(causal, dict):
        raise ValueError(f"{path} has no cross-time causal metrics")
    missing = [
        key
        for key in (*CAUSAL_METRICS, "identity_error_miou")
        if key not in causal
    ]
    if missing:
        raise ValueError(
            f"{path} cross-time causal metrics are missing {missing}"
        )
    output = {
        key: float(causal[key])
        for key in (*CAUSAL_METRICS, "identity_error_miou")
    }
    if abs(output["identity_error_miou"]) > 1e-6:
        raise ValueError(f"{path} cross-time causal identity failed")
    return output


def _dataset_summary(path: Path) -> tuple[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = _require_routing_report(path, payload)
    dataset = _dataset_name(path, payload)
    pairs = list(report["pairs"])
    risks = [float(row["post_risk"]) for row in pairs]
    harmful = [bool(row["post_harmful"]) for row in pairs]
    utilities = [
        float(row["post_gt_pair_utility"]) for row in pairs
    ]

    prompt_by_id = {
        int(row["query_id"]): row
        for row in report["prompts"]
    }
    risk_by_prompt: dict[int, float] = {}
    for pair in pairs:
        query_id = int(pair["query_id"])
        risk_by_prompt[query_id] = max(
            risk_by_prompt.get(query_id, 0.0),
            float(pair["post_risk"]),
        )
    prompt_risks = []
    prompt_utilities = []
    for query_id, risk in sorted(risk_by_prompt.items()):
        prompt = prompt_by_id.get(query_id)
        if prompt is None:
            continue
        prompt_risks.append(risk)
        prompt_utilities.append(
            float(prompt["post"]["standalone_utility_miou"])
        )

    raw_head = report["heads"]["H1_raw_max"]
    head_rows = {}
    candidate_values = {}
    for head_name, values in sorted(report["heads"].items()):
        head_rows[head_name] = {
            "baseline_miou": float(values["baseline_miou"]),
            "tta_miou": float(values["tta_miou"]),
            "baseline_vs_raw": float(values["baseline_miou"])
            - float(raw_head["baseline_miou"]),
            "final_vs_raw": float(values["tta_miou"])
            - float(raw_head["tta_miou"]),
            "routed_tta_gain": float(values["tta_miou"])
            - float(values["baseline_miou"]),
        }
        if head_name.startswith("H6_risk_shoulder_"):
            candidate_values[head_name] = float(values["tta_miou"])

    if not candidate_values:
        raise ValueError(f"{path} has no H6 routing candidates")
    known_prompt_checks = []
    for prompt in report["prompts"]:
        query = str(prompt.get("query", "")).lower()
        if any(known in query for known in KNOWN_PROMPTS):
            known_prompt_checks.append(prompt)

    return dataset, {
        "source": str(path),
        "harmful_pair_auc": binary_auc(risks, harmful),
        "risk_pair_utility_spearman": spearman(risks, utilities),
        "risk_prompt_utility_spearman": spearman(
            prompt_risks,
            prompt_utilities,
        ),
        "harmful_pairs": int(sum(harmful)),
        "pair_count": len(pairs),
        "heads": head_rows,
        "routing_candidates": candidate_values,
        "known_prompt_checks": known_prompt_checks,
        "causal": _causal_summary(path, report),
    }


def analyze(paths: Sequence[str | Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two result JSON files are required")
    datasets = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        dataset, summary = _dataset_summary(path)
        if dataset in datasets:
            raise ValueError(f"duplicate dataset name {dataset!r}")
        datasets[dataset] = summary

    candidates = {
        dataset: summary["routing_candidates"]
        for dataset, summary in datasets.items()
    }
    candidate_sets = {
        tuple(sorted(values))
        for values in candidates.values()
    }
    if len(candidate_sets) != 1:
        raise ValueError("datasets expose different H6 candidate grids")
    candidate_names = next(iter(candidate_sets))
    candidate_macro = {
        name: sum(values[name] for values in candidates.values())
        / len(candidates)
        for name in candidate_names
    }
    selected_global = max(candidate_macro, key=candidate_macro.__getitem__)
    raw_macro = sum(
        summary["heads"]["H1_raw_max"]["tta_miou"]
        for summary in datasets.values()
    ) / len(datasets)
    selected_values = {
        dataset: summary["heads"][selected_global]["tta_miou"]
        for dataset, summary in datasets.items()
    }
    selected_improvements = {
        dataset: selected_values[dataset]
        - summary["heads"]["H1_raw_max"]["tta_miou"]
        for dataset, summary in datasets.items()
    }
    causal_macro = {
        key: sum(
            summary["causal"][key]
            for summary in datasets.values()
        )
        / len(datasets)
        for key in CAUSAL_METRICS
    }
    return {
        "dataset_count": len(datasets),
        "datasets": datasets,
        "macro": {
            "harmful_pair_auc": _mean(
                [
                    summary["harmful_pair_auc"]
                    for summary in datasets.values()
                ]
            ),
            "risk_pair_utility_spearman": _mean(
                [
                    summary["risk_pair_utility_spearman"]
                    for summary in datasets.values()
                ]
            ),
            "risk_prompt_utility_spearman": _mean(
                [
                    summary["risk_prompt_utility_spearman"]
                    for summary in datasets.values()
                ]
            ),
            "raw_final_miou": raw_macro,
            "selected_final_miou": candidate_macro[selected_global],
            "selected_improvement_miou": (
                candidate_macro[selected_global] - raw_macro
            ),
            "improved_datasets": sum(
                value > 0.0 for value in selected_improvements.values()
            ),
            "maximum_drop_miou": min(
                selected_improvements.values()
            ),
            "causal": causal_macro,
        },
        "candidate_macro_final_miou": candidate_macro,
        "selected_global_candidate": selected_global,
        "selected_dataset_improvements": selected_improvements,
        "leave_one_dataset_out": leave_one_dataset_out(candidates),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prompt-transfer routing diagnostics."
    )
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _print_summary(result: dict) -> None:
    print(
        "dataset | canonical_direct | shielding | alias_transfer | "
        "interaction | total",
        flush=True,
    )
    selected = result["selected_global_candidate"]
    for dataset, summary in sorted(result["datasets"].items()):
        causal = summary["causal"]
        print(
            f"{dataset} | {causal['canonical_direct_miou']:+.3f} | "
            f"{causal['static_alias_shielding_miou']:+.3f} | "
            f"{causal['alias_only_transfer_miou']:+.3f} | "
            f"{causal['interaction_miou']:+.3f} | "
            f"{causal['total_synonym_tta_gain_miou']:+.3f}",
            flush=True,
        )
    print(
        f"selected={selected} "
        f"macro_gain={result['macro']['selected_improvement_miou']:+.3f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = analyze(args.results)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _print_summary(result)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
