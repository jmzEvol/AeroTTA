from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


SKIPPED_FEATURE_GROUPS = {"per_class", "gradient_norm_per_layer"}


def _dataset_name(result: dict, path: Path) -> str:
    config = str(result.get("config") or "").strip()
    if config:
        stem = Path(config).stem
        return stem[4:] if stem.startswith("cfg_") else stem
    return path.stem


def _flatten_numeric_features(
    values: dict,
    *,
    prefix: str = "",
) -> dict[str, float]:
    flattened = {}
    for key, value in values.items():
        if key in SKIPPED_FEATURE_GROUPS:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            flattened[name] = float(value)
        elif isinstance(value, (int, float)) and value is not None:
            if np.isfinite(float(value)):
                flattened[name] = float(value)
        elif isinstance(value, dict):
            flattened.update(
                _flatten_numeric_features(value, prefix=name)
            )
    return flattened


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlation_fields(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, float | int | None]:
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    return {
        "samples": int(finite.sum()),
        "pearson": _correlation(left, right),
        "spearman": _correlation(_rankdata(left), _rankdata(right)),
    }


def _candidate_strength(
    metadata: dict[str, dict],
    candidate: str,
    field: str,
) -> float:
    value = metadata.get(candidate, {}).get(field, 1.0)
    return float(value) if value is not None else float("nan")


def _quantile(values: Iterable[float], probability: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.quantile(array, probability)) if len(array) else 0.0


def _load_rows(paths: list[Path]) -> tuple[list[dict], tuple[str, ...]]:
    rows = []
    candidate_sets = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        dataset = _dataset_name(result, path)
        for image in result.get("images", []):
            oracle = image.get("image_adaptive_oracle")
            if not oracle or not oracle.get("candidates"):
                continue
            reference = str(oracle["reference_candidate"])
            candidates = oracle["candidates"]
            if reference not in candidates:
                raise ValueError(
                    f"{path}: image reference candidate {reference!r} is missing"
                )
            deltas = {
                str(key): float(value["delta_miou"])
                for key, value in candidates.items()
            }
            candidate_sets.append(set(deltas))
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": str(image.get("sample_id", "")),
                    "reference": reference,
                    "oracle": str(oracle["oracle_candidate"]),
                    "features": _flatten_numeric_features(
                        oracle.get("teacher_features") or {}
                    ),
                    "deltas": deltas,
                }
            )
    if not rows:
        raise ValueError("no image-adaptive oracle records were found")
    common_candidates = set.intersection(*candidate_sets)
    references = {row["reference"] for row in rows}
    if len(references) != 1:
        raise ValueError("all inputs must use the same reference candidate")
    reference = next(iter(references))
    if reference not in common_candidates:
        raise ValueError("reference candidate is not common to every input")
    return rows, tuple(sorted(common_candidates))


def _feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    common = set(rows[0]["features"])
    for row in rows[1:]:
        common &= set(row["features"])
    names = sorted(common)
    if not names:
        raise ValueError("oracle records have no common numeric teacher features")
    matrix = np.asarray(
        [[row["features"][name] for name in names] for row in rows],
        dtype=np.float64,
    )
    finite = np.isfinite(matrix).all(axis=0)
    names = [name for name, keep in zip(names, finite, strict=True) if keep]
    matrix = matrix[:, finite]
    if not names:
        raise ValueError("all common teacher features are non-finite")
    return matrix, names


def _ridge_candidate_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    train = (x_train - mean) / std
    test = (x_test - mean) / std
    train = np.concatenate([np.ones((len(train), 1)), train], axis=1)
    test = np.concatenate([np.ones((len(test), 1)), test], axis=1)
    penalty = np.eye(train.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        train.T @ train + penalty,
        train.T @ y_train,
    )
    return test @ weights


def analyze_results(
    paths: list[Path],
    *,
    ridge: float = 1e-3,
) -> dict:
    paths = [Path(path) for path in paths]
    rows, candidate_keys = _load_rows(paths)
    features, feature_names = _feature_matrix(rows)
    reference = rows[0]["reference"]
    reference_delta = np.asarray(
        [row["deltas"][reference] for row in rows],
        dtype=np.float64,
    )
    candidate_delta = np.asarray(
        [
            [row["deltas"][candidate] for candidate in candidate_keys]
            for row in rows
        ],
        dtype=np.float64,
    )
    candidate_index = {
        candidate: index for index, candidate in enumerate(candidate_keys)
    }
    missing_oracle = sorted(
        {
            row["oracle"]
            for row in rows
            if row["oracle"] not in candidate_index
        }
    )
    if missing_oracle:
        raise ValueError(
            f"stored oracle candidates are not common to every input: {missing_oracle}"
        )
    oracle_indices = np.asarray(
        [candidate_index[row["oracle"]] for row in rows],
        dtype=np.int64,
    )
    oracle_delta = candidate_delta[np.arange(len(rows)), oracle_indices]
    oracle_gain = oracle_delta - reference_delta
    oracle_keys = [candidate_keys[index] for index in oracle_indices]

    candidate_metadata = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        summary = result.get("image_adaptive_oracle") or {}
        for entry in summary.get("candidate_grid", []):
            candidate_metadata[str(entry["key"])] = entry
    optimal_lr = np.asarray(
        [
            _candidate_strength(
                candidate_metadata,
                key,
                "lr_multiplier",
            )
            for key in oracle_keys
        ]
    )
    optimal_power = np.asarray(
        [
            _candidate_strength(
                candidate_metadata,
                key,
                "target_presence_power",
            )
            for key in oracle_keys
        ]
    )

    correlations = {}
    targets = {
        "oracle_gain": oracle_gain,
        "optimal_lr_multiplier": optimal_lr,
        "optimal_target_presence_power": optimal_power,
    }
    for feature_index, name in enumerate(feature_names):
        values = features[:, feature_index]
        correlations[name] = {
            target_name: _correlation_fields(values, target)
            for target_name, target in targets.items()
        }

    lodo = {}
    datasets = sorted({row["dataset"] for row in rows})
    for held_out in datasets:
        train_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] != held_out]
        )
        test_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["dataset"] == held_out]
        )
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        train_targets = (
            candidate_delta[train_indices]
            - reference_delta[train_indices, None]
        )
        predicted = _ridge_candidate_predictions(
            features[train_indices],
            train_targets,
            features[test_indices],
            ridge=ridge,
        )
        chosen_indices = predicted.argmax(axis=1)
        chosen_delta = candidate_delta[
            test_indices,
            chosen_indices,
        ]
        fold_reference = reference_delta[test_indices]
        fold_oracle = oracle_delta[test_indices]
        available = float((fold_oracle - fold_reference).sum())
        recovered = float((chosen_delta - fold_reference).sum())
        lodo[held_out] = {
            "images": int(len(test_indices)),
            "reference_mean_delta": float(fold_reference.mean()),
            "selected_mean_delta": float(chosen_delta.mean()),
            "oracle_mean_delta": float(fold_oracle.mean()),
            "selected_q10_delta": _quantile(chosen_delta, 0.1),
            "oracle_recovery_ratio": (
                recovered / available if abs(available) > 1e-12 else 0.0
            ),
            "selected_candidate_counts": dict(
                Counter(candidate_keys[index] for index in chosen_indices)
            ),
        }

    return {
        "images": len(rows),
        "datasets": datasets,
        "reference_candidate": reference,
        "candidate_keys": list(candidate_keys),
        "feature_names": feature_names,
        "reference_mean_delta": float(reference_delta.mean()),
        "oracle_mean_delta": float(oracle_delta.mean()),
        "oracle_headroom_miou": float(oracle_gain.mean()),
        "reference_q10_delta": _quantile(reference_delta, 0.1),
        "oracle_q10_delta": _quantile(oracle_delta, 0.1),
        "candidate_win_counts": dict(Counter(oracle_keys)),
        "correlations": correlations,
        "leave_one_dataset_out": lodo,
    }


def _strongest_correlations(report: dict, limit: int = 8) -> list[tuple[str, float]]:
    values = []
    for feature, targets in report["correlations"].items():
        correlation = targets["oracle_gain"]["spearman"]
        if correlation is not None:
            values.append((feature, float(correlation)))
    return sorted(values, key=lambda item: abs(item[1]), reverse=True)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze image-adaptive LoRA-TTA oracle diagnostics."
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    report = analyze_results(args.results, ridge=args.ridge)
    print(f"images: {report['images']}")
    print(f"datasets: {', '.join(report['datasets'])}")
    print(f"reference mean delta: {report['reference_mean_delta']:+.3f}")
    print(f"oracle mean delta:    {report['oracle_mean_delta']:+.3f}")
    print(f"oracle headroom:      {report['oracle_headroom_miou']:+.3f}")
    print(f"candidate wins:       {report['candidate_win_counts']}")
    print("strongest teacher-feature correlations with oracle headroom:")
    for name, value in _strongest_correlations(report):
        print(f"  {name}: {value:+.3f}")
    for dataset, fold in report["leave_one_dataset_out"].items():
        print(
            f"LODO {dataset}: reference={fold['reference_mean_delta']:+.3f} "
            f"selected={fold['selected_mean_delta']:+.3f} "
            f"oracle={fold['oracle_mean_delta']:+.3f} "
            f"recovery={fold['oracle_recovery_ratio']:.3f}"
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
