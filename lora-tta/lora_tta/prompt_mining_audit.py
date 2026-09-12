from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import re
import unicodedata

import torch

from .config import LossConfig, MiningConfig
from .losses import SelectedClassEntry
from .mining import select_classwise_pixels


MINING_VARIANT_SOURCES = {
    "syn_presence__syn_pixels": ("synonym", "synonym"),
    "can_presence__can_pixels": ("canonical", "canonical"),
    "syn_presence__can_pixels": ("synonym", "canonical"),
    "can_presence__syn_pixels": ("canonical", "synonym"),
}
MINING_VARIANT_NAMES = tuple(MINING_VARIANT_SOURCES)
MINING_VARIANT_PAIRS = tuple(combinations(range(len(MINING_VARIANT_NAMES)), 2))
SET_REGION_NAMES = ("intersection", "left_only", "right_only", "union")


@dataclass(frozen=True)
class PromptClassViews:
    query_scores: torch.Tensor
    query_presence: torch.Tensor
    canonical_scores: torch.Tensor
    synonym_scores: torch.Tensor
    canonical_presence: torch.Tensor
    synonym_presence: torch.Tensor
    canonical_query_ids: tuple[int, ...]
    query_ids_by_class: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MiningVariantResult:
    name: str
    presence_source: str
    pixel_source: str
    present: torch.Tensor
    selected: dict[int, SelectedClassEntry]
    stage_counts: dict[int, dict[str, int]]
    raw_pred: torch.Tensor


@dataclass(frozen=True)
class AliasAttribution:
    winner_query_ids: torch.Tensor
    tie_counts: torch.Tensor
    winner_scores: torch.Tensor
    canonical_scores: torch.Tensor
    score_gaps: torch.Tensor


def normalize_prompt(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", value)


def build_canonical_query_ids(
    query_words: list[str] | tuple[str, ...],
    query_idx_list: list[int] | tuple[int, ...],
    canonical_words: list[str] | tuple[str, ...],
) -> tuple[int, ...]:
    if len(query_words) != len(query_idx_list):
        raise ValueError("query word/mapping length mismatch")

    result = []
    for cls_idx, canonical in enumerate(canonical_words):
        wanted = normalize_prompt(canonical)
        matches = [
            query_id
            for query_id, (word, mapped) in enumerate(
                zip(query_words, query_idx_list)
            )
            if int(mapped) == cls_idx and normalize_prompt(word) == wanted
        ]
        if not matches:
            raise ValueError(
                f"missing canonical query for class {cls_idx}: {canonical!r}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"duplicate canonical query for class {cls_idx}: {canonical!r}"
            )
        result.append(matches[0])
    return tuple(result)


def build_first_query_ids_by_class(
    query_idx_list: list[int] | tuple[int, ...],
    *,
    num_classes: int,
) -> tuple[int, ...]:
    first_query_ids: list[int | None] = [None] * int(num_classes)
    for query_id, mapped_class in enumerate(query_idx_list):
        class_id = int(mapped_class)
        if class_id < 0 or class_id >= int(num_classes):
            raise ValueError(
                f"query {query_id} maps to class {class_id}, outside "
                f"[0, {int(num_classes) - 1}]"
            )
        if first_query_ids[class_id] is None:
            first_query_ids[class_id] = int(query_id)

    missing_classes = [
        class_id
        for class_id, query_id in enumerate(first_query_ids)
        if query_id is None
    ]
    if missing_classes:
        raise ValueError(
            f"classes without prompts: {tuple(missing_classes)}"
        )
    return tuple(int(query_id) for query_id in first_query_ids)


def build_prompt_class_views(
    *,
    query_scores: torch.Tensor,
    presence_logits: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    canonical_query_ids: tuple[int, ...],
    num_classes: int,
) -> PromptClassViews:
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError(
            f"expected query_scores [1,Q,H,W], got {tuple(query_scores.shape)}"
        )
    if presence_logits.shape != query_scores.shape[:2]:
        raise ValueError("presence/query shape mismatch")
    if len(query_idx_list) != int(query_scores.shape[1]):
        raise ValueError("query mapping/query score length mismatch")

    query_ids_by_class = tuple(
        tuple(
            query_id
            for query_id, mapped in enumerate(query_idx_list)
            if int(mapped) == cls_idx
        )
        for cls_idx in range(int(num_classes))
    )
    if any(not query_ids for query_ids in query_ids_by_class):
        raise ValueError("at least one class has no query")
    if len(canonical_query_ids) != int(num_classes):
        raise ValueError("canonical query/class length mismatch")

    scores = query_scores[0].float()
    presence = presence_logits[0].float().sigmoid()
    canonical_index = torch.tensor(
        canonical_query_ids,
        device=scores.device,
        dtype=torch.long,
    )
    return PromptClassViews(
        query_scores=scores,
        query_presence=presence,
        canonical_scores=scores.index_select(0, canonical_index),
        synonym_scores=torch.stack(
            [scores[list(query_ids)].amax(dim=0) for query_ids in query_ids_by_class]
        ),
        canonical_presence=presence.index_select(0, canonical_index),
        synonym_presence=torch.stack(
            [presence[list(query_ids)].amax(dim=0) for query_ids in query_ids_by_class]
        ),
        canonical_query_ids=tuple(int(value) for value in canonical_query_ids),
        query_ids_by_class=query_ids_by_class,
    )


def prediction_from_scores(
    scores: torch.Tensor,
    *,
    prob_thd: float,
    bg_idx: int,
) -> torch.Tensor:
    pred = scores.argmax(dim=0).long()
    pred = pred.clone()
    pred[scores.max(dim=0).values < float(prob_thd)] = int(bg_idx)
    return pred


def stage_counts(
    *,
    class_scores: torch.Tensor,
    class_presence: torch.Tensor,
    valid: torch.Tensor,
    mining: MiningConfig,
) -> dict[int, dict[str, int]]:
    output = {}
    for cls_idx in range(int(class_scores.shape[0])):
        is_present = float(class_presence[cls_idx].item()) > float(mining.tau_pos)
        if cls_idx == int(mining.bg_idx) and not mining.include_bg:
            is_present = False
        candidate_count = (
            int(
                (
                    valid.to(device=class_scores.device, dtype=torch.bool)
                    & (class_scores[cls_idx] >= float(mining.prob_thd))
                ).sum().item()
            )
            if is_present
            else 0
        )
        pretopk_count = 0
        if candidate_count >= int(mining.n_min):
            pretopk_count = min(
                max(1, int(float(mining.rho) * candidate_count)),
                int(mining.class_kmax_overrides.get(cls_idx, mining.kmax)),
                candidate_count,
            )
        output[cls_idx] = {
            "present": int(is_present),
            "candidate": candidate_count,
            "pretopk": pretopk_count,
        }
    return output


def run_mining_variants(
    *,
    views: PromptClassViews,
    valid: torch.Tensor,
    mining: MiningConfig,
    loss: LossConfig,
) -> dict[str, MiningVariantResult]:
    output = {}
    for name, (presence_source, pixel_source) in MINING_VARIANT_SOURCES.items():
        presence = (
            views.synonym_presence
            if presence_source == "synonym"
            else views.canonical_presence
        )
        scores = (
            views.synonym_scores
            if pixel_source == "synonym"
            else views.canonical_scores
        )
        raw_pred = prediction_from_scores(
            scores,
            prob_thd=mining.prob_thd,
            bg_idx=mining.bg_idx,
        )
        selected, present = select_classwise_pixels(
            class_scores=scores,
            class_presence=presence,
            raw_pred=raw_pred,
            valid=valid,
            mining=mining,
            loss=loss,
            target_scores=scores,
        )
        output[name] = MiningVariantResult(
            name=name,
            presence_source=presence_source,
            pixel_source=pixel_source,
            present=present,
            selected=selected,
            stage_counts=stage_counts(
                class_scores=scores,
                class_presence=presence,
                valid=valid,
                mining=mining,
            ),
            raw_pred=raw_pred,
        )
    return output


def selected_index_dict(
    selected: dict[int, SelectedClassEntry],
) -> dict[int, torch.Tensor]:
    return {
        int(cls_idx): entry.flat_idx.detach().long()
        for cls_idx, entry in selected.items()
    }


def _set_region(
    left: torch.Tensor,
    right: torch.Tensor,
    mode: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    left_values = set(int(value) for value in left.detach().cpu().tolist())
    right_values = set(int(value) for value in right.detach().cpu().tolist())
    operations = {
        "intersection": left_values & right_values,
        "left_only": left_values - right_values,
        "right_only": right_values - left_values,
        "union": left_values | right_values,
    }
    if mode not in operations:
        raise ValueError(f"unknown set-region mode={mode}")
    return torch.tensor(
        sorted(operations[mode]),
        device=device,
        dtype=torch.long,
    )


def _region_counts(
    flat_idx: torch.Tensor,
    gt: torch.Tensor,
    cls_idx: int,
    num_classes: int,
) -> dict[str, int | list[int]]:
    labels = gt.flatten()[flat_idx.to(device=gt.device)].long()
    labels = labels[(labels >= 0) & (labels < int(num_classes))]
    return {
        "count": int(flat_idx.numel()),
        "evaluable_count": int(labels.numel()),
        "correct": int((labels == int(cls_idx)).sum().item()),
        "precision": _precision(
            int((labels == int(cls_idx)).sum().item()),
            int(labels.numel()),
        ),
        "gt_distribution": [
            int(value)
            for value in torch.bincount(
                labels,
                minlength=int(num_classes),
            )
            .detach()
            .cpu()
            .tolist()
        ],
    }


def compare_selected_sets(
    synonym: dict[int, torch.Tensor],
    canonical: dict[int, torch.Tensor],
    *,
    gt: torch.Tensor,
    num_classes: int,
) -> dict[int, dict]:
    output = {}
    empty = torch.empty(0, device=gt.device, dtype=torch.long)
    for cls_idx in range(int(num_classes)):
        synonym_idx = synonym.get(cls_idx, empty).to(device=gt.device)
        canonical_idx = canonical.get(cls_idx, empty).to(device=gt.device)
        region_indices = {
            "intersection": _set_region(
                synonym_idx,
                canonical_idx,
                "intersection",
                device=gt.device,
            ),
            "synonym_only": _set_region(
                synonym_idx,
                canonical_idx,
                "left_only",
                device=gt.device,
            ),
            "canonical_only": _set_region(
                synonym_idx,
                canonical_idx,
                "right_only",
                device=gt.device,
            ),
            "union": _set_region(
                synonym_idx,
                canonical_idx,
                "union",
                device=gt.device,
            ),
        }
        row = {
            key: _region_counts(indices, gt, cls_idx, num_classes)
            for key, indices in region_indices.items()
        }
        union_count = int(row["union"]["count"])
        row["jaccard"] = (
            float(row["intersection"]["count"]) / union_count
            if union_count
            else None
        )
        output[cls_idx] = row
    return output


def alias_attribution(
    *,
    query_scores: torch.Tensor,
    query_ids: tuple[int, ...],
    canonical_query_id: int,
    flat_idx: torch.Tensor,
    tie_tolerance: float = 1e-6,
) -> AliasAttribution:
    if not query_ids:
        raise ValueError("alias attribution requires at least one query")
    local_scores = query_scores[list(query_ids)].reshape(len(query_ids), -1)[
        :, flat_idx
    ]
    winner_local = local_scores.argmax(dim=0)
    winner_scores = local_scores.gather(0, winner_local.unsqueeze(0))[0]
    global_query_ids = torch.tensor(
        query_ids,
        device=local_scores.device,
        dtype=torch.long,
    )
    canonical_scores = query_scores[int(canonical_query_id)].flatten()[flat_idx]
    return AliasAttribution(
        winner_query_ids=global_query_ids[winner_local],
        tie_counts=(
            local_scores >= winner_scores.unsqueeze(0) - float(tie_tolerance)
        ).sum(dim=0),
        winner_scores=winner_scores,
        canonical_scores=canonical_scores,
        score_gaps=winner_scores - canonical_scores,
    )


def new_running_stats(bin_edges: torch.Tensor) -> dict:
    edges = torch.as_tensor(bin_edges, dtype=torch.float32).detach().cpu()
    if edges.ndim != 1 or int(edges.numel()) < 2:
        raise ValueError("histogram requires at least two one-dimensional edges")
    if not bool(torch.all(edges[1:] > edges[:-1])):
        raise ValueError("histogram edges must be strictly increasing")
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
        "hist": torch.zeros(int(edges.numel()) - 1, dtype=torch.long),
        "bin_edges": edges.clone(),
    }


def update_running_stats(accumulator: dict, values: torch.Tensor) -> None:
    values = values.detach().float().cpu().flatten()
    if int(values.numel()) == 0:
        return
    accumulator["count"] += int(values.numel())
    accumulator["sum"] += float(values.sum().item())
    accumulator["sum_sq"] += float((values * values).sum().item())
    value_min = float(values.min().item())
    value_max = float(values.max().item())
    accumulator["min"] = (
        value_min
        if accumulator["min"] is None
        else min(float(accumulator["min"]), value_min)
    )
    accumulator["max"] = (
        value_max
        if accumulator["max"] is None
        else max(float(accumulator["max"]), value_max)
    )
    bucket = torch.bucketize(
        values,
        accumulator["bin_edges"],
        right=True,
    ) - 1
    bucket = bucket.clamp(0, len(accumulator["hist"]) - 1)
    accumulator["hist"] += torch.bincount(
        bucket,
        minlength=len(accumulator["hist"]),
    )


def merge_running_stats(left: dict, right: dict) -> dict:
    if not torch.equal(left["bin_edges"], right["bin_edges"]):
        raise ValueError("histogram edges do not match")
    output = new_running_stats(left["bin_edges"])
    output["count"] = int(left["count"]) + int(right["count"])
    output["sum"] = float(left["sum"]) + float(right["sum"])
    output["sum_sq"] = float(left["sum_sq"]) + float(right["sum_sq"])
    minima = [value for value in (left["min"], right["min"]) if value is not None]
    maxima = [value for value in (left["max"], right["max"]) if value is not None]
    output["min"] = min(minima) if minima else None
    output["max"] = max(maxima) if maxima else None
    output["hist"] = left["hist"] + right["hist"]
    return output


def _new_stats_grid(rows: int, columns: int, edges: torch.Tensor) -> list[list[dict]]:
    return [
        [new_running_stats(edges) for _ in range(int(columns))]
        for _ in range(int(rows))
    ]


def _new_stats_vector(length: int, edges: torch.Tensor) -> list[dict]:
    return [new_running_stats(edges) for _ in range(int(length))]


def _new_region_stats(
    pairs: int,
    classes: int,
    regions: int,
    edges: torch.Tensor,
) -> list[list[list[dict]]]:
    return [
        [
            [new_running_stats(edges) for _ in range(int(regions))]
            for _ in range(int(classes))
        ]
        for _ in range(int(pairs))
    ]


def new_audit_accumulator(*, num_classes: int, num_queries: int) -> dict:
    num_variants = len(MINING_VARIANT_NAMES)
    num_pairs = len(MINING_VARIANT_PAIRS)
    score_edges = torch.linspace(0.0, 1.0, 101)
    signed_edges = torch.linspace(-1.0, 1.0, 201)
    output = {
        "num_classes": int(num_classes),
        "num_queries": int(num_queries),
        "variant_present_images": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_candidate_images": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_selected_images": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_candidate_pixels": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_pretopk_pixels": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_selected": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_evaluable": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_correct": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "variant_gt": torch.zeros(
            (num_variants, num_classes, num_classes), dtype=torch.long
        ),
        "variant_raw_argmax": torch.zeros(
            (num_variants, num_classes), dtype=torch.long
        ),
        "alias_winner": torch.zeros(num_queries, dtype=torch.long),
        "alias_evaluable": torch.zeros(num_queries, dtype=torch.long),
        "alias_correct": torch.zeros(num_queries, dtype=torch.long),
        "alias_tie": torch.zeros(num_queries, dtype=torch.long),
        "alias_gt": torch.zeros((num_queries, num_classes), dtype=torch.long),
        "alias_synonym_only": torch.zeros(num_queries, dtype=torch.long),
        "alias_synonym_only_evaluable": torch.zeros(
            num_queries, dtype=torch.long
        ),
        "alias_synonym_only_correct": torch.zeros(num_queries, dtype=torch.long),
        "alias_synonym_only_gt": torch.zeros(
            (num_queries, num_classes), dtype=torch.long
        ),
        "pair_region_count": torch.zeros(
            (num_pairs, num_classes, len(SET_REGION_NAMES)), dtype=torch.long
        ),
        "pair_region_correct": torch.zeros(
            (num_pairs, num_classes, len(SET_REGION_NAMES)), dtype=torch.long
        ),
        "pair_region_evaluable": torch.zeros(
            (num_pairs, num_classes, len(SET_REGION_NAMES)), dtype=torch.long
        ),
        "pair_region_gt": torch.zeros(
            (num_pairs, num_classes, len(SET_REGION_NAMES), num_classes),
            dtype=torch.long,
        ),
        "pair_region_stats": {
            "canonical_score": _new_region_stats(
                num_pairs, num_classes, len(SET_REGION_NAMES), score_edges
            ),
            "synonym_score": _new_region_stats(
                num_pairs, num_classes, len(SET_REGION_NAMES), score_edges
            ),
            "score_gap": _new_region_stats(
                num_pairs, num_classes, len(SET_REGION_NAMES), signed_edges
            ),
            "synonym_margin": _new_region_stats(
                num_pairs, num_classes, len(SET_REGION_NAMES), signed_edges
            ),
        },
        "false_synonym_only_raw_argmax": torch.zeros(
            num_classes, dtype=torch.long
        ),
        "variant_stats": {
            "selected_score": _new_stats_grid(
                num_variants, num_classes, score_edges
            ),
            "canonical_target": _new_stats_grid(
                num_variants, num_classes, score_edges
            ),
            "synonym_target": _new_stats_grid(
                num_variants, num_classes, score_edges
            ),
            "score_gap": _new_stats_grid(
                num_variants, num_classes, signed_edges
            ),
            "margin": _new_stats_grid(
                num_variants, num_classes, signed_edges
            ),
        },
        "alias_stats": {
            "winner_score": _new_stats_vector(num_queries, score_edges),
            "canonical_score": _new_stats_vector(num_queries, score_edges),
            "score_gap": _new_stats_vector(num_queries, signed_edges),
            "margin": _new_stats_vector(num_queries, signed_edges),
        },
        "false_synonym_only_margin": _new_stats_vector(
            num_classes, signed_edges
        ),
        "outliers": [],
    }
    return output


def _class_margin(
    class_scores: torch.Tensor,
    cls_idx: int,
    flat_idx: torch.Tensor,
) -> torch.Tensor:
    own = class_scores[int(cls_idx)].flatten()[flat_idx]
    competitors = [
        index for index in range(int(class_scores.shape[0])) if index != int(cls_idx)
    ]
    if not competitors:
        return torch.zeros_like(own)
    other = class_scores[competitors].reshape(len(competitors), -1)[:, flat_idx]
    return own - other.max(dim=0).values


def _valid_selected_labels(
    gt: torch.Tensor,
    flat_idx: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = gt.flatten()[flat_idx.to(device=gt.device)].long()
    valid = (labels >= 0) & (labels < int(num_classes))
    return labels[valid], flat_idx[valid.to(device=flat_idx.device)]


def _update_alias_counts(
    accumulator: dict,
    *,
    views: PromptClassViews,
    cls_idx: int,
    flat_idx: torch.Tensor,
    gt: torch.Tensor,
    synonym_only: bool,
) -> dict[str, int]:
    num_classes = int(accumulator["num_classes"])
    if int(flat_idx.numel()) == 0:
        return {}
    labels = gt.flatten()[flat_idx.to(device=gt.device)].long()
    valid = (labels >= 0) & (labels < num_classes)
    attribution = alias_attribution(
        query_scores=views.query_scores,
        query_ids=views.query_ids_by_class[int(cls_idx)],
        canonical_query_id=views.canonical_query_ids[int(cls_idx)],
        flat_idx=flat_idx,
    )
    margin = _class_margin(views.synonym_scores, cls_idx, flat_idx)
    counts = {}
    for query_id in views.query_ids_by_class[int(cls_idx)]:
        winner = attribution.winner_query_ids == int(query_id)
        count = int(winner.sum().item())
        if count == 0:
            continue
        evaluable = winner.to(device=labels.device) & valid
        query_labels = labels[evaluable]
        evaluable_count = int(query_labels.numel())
        correct = int((query_labels == int(cls_idx)).sum().item())
        counts[str(query_id)] = count
        if synonym_only:
            accumulator["alias_synonym_only"][query_id] += count
            accumulator["alias_synonym_only_evaluable"][query_id] += (
                evaluable_count
            )
            accumulator["alias_synonym_only_correct"][query_id] += correct
            accumulator["alias_synonym_only_gt"][query_id] += torch.bincount(
                query_labels.detach().cpu(), minlength=num_classes
            )
        else:
            accumulator["alias_winner"][query_id] += count
            accumulator["alias_evaluable"][query_id] += evaluable_count
            accumulator["alias_correct"][query_id] += correct
            accumulator["alias_tie"][query_id] += int(
                ((attribution.tie_counts > 1) & winner).sum().item()
            )
            accumulator["alias_gt"][query_id] += torch.bincount(
                query_labels.detach().cpu(), minlength=num_classes
            )
            update_running_stats(
                accumulator["alias_stats"]["winner_score"][query_id],
                attribution.winner_scores[winner],
            )
            update_running_stats(
                accumulator["alias_stats"]["canonical_score"][query_id],
                attribution.canonical_scores[winner],
            )
            update_running_stats(
                accumulator["alias_stats"]["score_gap"][query_id],
                attribution.score_gaps[winner],
            )
            update_running_stats(
                accumulator["alias_stats"]["margin"][query_id],
                margin[winner],
            )
    return counts


def _alias_winner_counts(
    *,
    views: PromptClassViews,
    cls_idx: int,
    flat_idx: torch.Tensor,
) -> dict[str, int]:
    if int(flat_idx.numel()) == 0:
        return {}
    attribution = alias_attribution(
        query_scores=views.query_scores,
        query_ids=views.query_ids_by_class[int(cls_idx)],
        canonical_query_id=views.canonical_query_ids[int(cls_idx)],
        flat_idx=flat_idx,
    )
    return {
        str(query_id): int(
            (attribution.winner_query_ids == int(query_id)).sum().item()
        )
        for query_id in views.query_ids_by_class[int(cls_idx)]
        if int((attribution.winner_query_ids == int(query_id)).sum().item()) > 0
    }


def _prune_outliers(rows: list[dict], limit: int = 50) -> list[dict]:
    kept = []
    class_ids = sorted({int(row["class_idx"]) for row in rows})
    for cls_idx in class_ids:
        class_rows = [row for row in rows if int(row["class_idx"]) == cls_idx]
        class_rows.sort(
            key=lambda row: (
                -int(row["false_synonym_only"]),
                str(row["sample_id"]),
                int(row["dataset_index"]),
            )
        )
        kept.extend(class_rows[: int(limit)])
    kept.sort(
        key=lambda row: (
            int(row["class_idx"]),
            -int(row["false_synonym_only"]),
            str(row["sample_id"]),
            int(row["dataset_index"]),
        )
    )
    return kept


def update_audit_accumulator(
    accumulator: dict,
    *,
    variants: dict[str, MiningVariantResult],
    views: PromptClassViews,
    gt: torch.Tensor,
    sample_id: str,
    dataset_index: int,
    bg_idx: int,
    tau_pos: float,
) -> None:
    del bg_idx
    num_classes = int(accumulator["num_classes"])
    if int(views.synonym_scores.shape[0]) != num_classes:
        raise ValueError("audit accumulator/class-view size mismatch")

    selected_by_variant = []
    for variant_idx, variant_name in enumerate(MINING_VARIANT_NAMES):
        variant = variants[variant_name]
        selected_indices = selected_index_dict(variant.selected)
        selected_by_variant.append(selected_indices)
        score_map = (
            views.synonym_scores
            if variant.pixel_source == "synonym"
            else views.canonical_scores
        )
        for cls_idx in range(num_classes):
            stage = variant.stage_counts[cls_idx]
            accumulator["variant_present_images"][variant_idx, cls_idx] += int(
                stage["present"]
            )
            accumulator["variant_candidate_images"][variant_idx, cls_idx] += int(
                int(stage["candidate"]) > 0
            )
            accumulator["variant_candidate_pixels"][variant_idx, cls_idx] += int(
                stage["candidate"]
            )
            accumulator["variant_pretopk_pixels"][variant_idx, cls_idx] += int(
                stage["pretopk"]
            )
            flat_idx = selected_indices.get(cls_idx)
            if flat_idx is None or int(flat_idx.numel()) == 0:
                continue
            accumulator["variant_selected_images"][variant_idx, cls_idx] += 1
            accumulator["variant_selected"][variant_idx, cls_idx] += int(
                flat_idx.numel()
            )
            labels, evaluable_idx = _valid_selected_labels(
                gt, flat_idx, num_classes
            )
            accumulator["variant_evaluable"][variant_idx, cls_idx] += int(
                labels.numel()
            )
            accumulator["variant_correct"][variant_idx, cls_idx] += int(
                (labels == cls_idx).sum().item()
            )
            accumulator["variant_gt"][variant_idx, cls_idx] += torch.bincount(
                labels.detach().cpu(), minlength=num_classes
            )
            accumulator["variant_raw_argmax"][variant_idx, cls_idx] += int(
                (
                    variant.raw_pred.flatten()[flat_idx]
                    == int(cls_idx)
                ).sum().item()
            )
            selected_score = score_map[cls_idx].flatten()[flat_idx]
            canonical_target = views.canonical_scores[cls_idx].flatten()[flat_idx]
            synonym_target = views.synonym_scores[cls_idx].flatten()[flat_idx]
            margin = _class_margin(score_map, cls_idx, flat_idx)
            stats = accumulator["variant_stats"]
            update_running_stats(
                stats["selected_score"][variant_idx][cls_idx], selected_score
            )
            update_running_stats(
                stats["canonical_target"][variant_idx][cls_idx], canonical_target
            )
            update_running_stats(
                stats["synonym_target"][variant_idx][cls_idx], synonym_target
            )
            update_running_stats(
                stats["score_gap"][variant_idx][cls_idx],
                synonym_target - canonical_target,
            )
            update_running_stats(
                stats["margin"][variant_idx][cls_idx], margin
            )

    for pair_idx, (left_idx, right_idx) in enumerate(MINING_VARIANT_PAIRS):
        left = selected_by_variant[left_idx]
        right = selected_by_variant[right_idx]
        for cls_idx in range(num_classes):
            empty = torch.empty(0, device=gt.device, dtype=torch.long)
            left_idx_tensor = left.get(cls_idx, empty).to(gt.device)
            right_idx_tensor = right.get(cls_idx, empty).to(gt.device)
            regions = (
                _set_region(left_idx_tensor, right_idx_tensor, "intersection", device=gt.device),
                _set_region(left_idx_tensor, right_idx_tensor, "left_only", device=gt.device),
                _set_region(left_idx_tensor, right_idx_tensor, "right_only", device=gt.device),
                _set_region(left_idx_tensor, right_idx_tensor, "union", device=gt.device),
            )
            for region_idx, region in enumerate(regions):
                counts = _region_counts(region, gt, cls_idx, num_classes)
                accumulator["pair_region_count"][pair_idx, cls_idx, region_idx] += int(
                    counts["count"]
                )
                accumulator["pair_region_evaluable"][
                    pair_idx, cls_idx, region_idx
                ] += int(counts["evaluable_count"])
                accumulator["pair_region_correct"][pair_idx, cls_idx, region_idx] += int(
                    counts["correct"]
                )
                accumulator["pair_region_gt"][pair_idx, cls_idx, region_idx] += torch.tensor(
                    counts["gt_distribution"], dtype=torch.long
                )
                score_indices = region.to(device=views.synonym_scores.device)
                canonical_score = views.canonical_scores[cls_idx].flatten()[
                    score_indices
                ]
                synonym_score = views.synonym_scores[cls_idx].flatten()[
                    score_indices
                ]
                region_stats = accumulator["pair_region_stats"]
                update_running_stats(
                    region_stats["canonical_score"][pair_idx][cls_idx][region_idx],
                    canonical_score,
                )
                update_running_stats(
                    region_stats["synonym_score"][pair_idx][cls_idx][region_idx],
                    synonym_score,
                )
                update_running_stats(
                    region_stats["score_gap"][pair_idx][cls_idx][region_idx],
                    synonym_score - canonical_score,
                )
                update_running_stats(
                    region_stats["synonym_margin"][pair_idx][cls_idx][region_idx],
                    _class_margin(views.synonym_scores, cls_idx, score_indices),
                )

    synonym_selected = selected_by_variant[0]
    canonical_selected = selected_by_variant[1]
    for cls_idx, flat_idx in synonym_selected.items():
        _update_alias_counts(
            accumulator,
            views=views,
            cls_idx=cls_idx,
            flat_idx=flat_idx,
            gt=gt,
            synonym_only=False,
        )

        canonical_idx = canonical_selected.get(
            cls_idx,
            torch.empty(0, device=flat_idx.device, dtype=torch.long),
        )
        synonym_only_idx = _set_region(
            flat_idx,
            canonical_idx,
            "left_only",
            device=flat_idx.device,
        )
        if int(synonym_only_idx.numel()) == 0:
            continue
        _update_alias_counts(
            accumulator,
            views=views,
            cls_idx=cls_idx,
            flat_idx=synonym_only_idx,
            gt=gt,
            synonym_only=True,
        )
        labels, valid_synonym_only = _valid_selected_labels(
            gt, synonym_only_idx, num_classes
        )
        false_mask = labels != int(cls_idx)
        false_indices = valid_synonym_only[false_mask.to(valid_synonym_only.device)]
        false_count = int(false_indices.numel())
        if false_count == 0:
            continue
        raw_pred = variants[MINING_VARIANT_NAMES[0]].raw_pred.flatten()[false_indices]
        accumulator["false_synonym_only_raw_argmax"][cls_idx] += int(
            (raw_pred == int(cls_idx)).sum().item()
        )
        false_margin = _class_margin(
            views.synonym_scores,
            cls_idx,
            false_indices,
        )
        update_running_stats(
            accumulator["false_synonym_only_margin"][cls_idx],
            false_margin,
        )
        alias_false_counts = _alias_winner_counts(
            views=views,
            cls_idx=cls_idx,
            flat_idx=false_indices,
        )
        synonym_labels, _ = _valid_selected_labels(
            gt, flat_idx, num_classes
        )
        canonical_labels, _ = _valid_selected_labels(
            gt, canonical_idx, num_classes
        )
        synonym_selected_count = int(flat_idx.numel())
        synonym_evaluable_count = int(synonym_labels.numel())
        canonical_selected_count = int(canonical_idx.numel())
        canonical_evaluable_count = int(canonical_labels.numel())
        synonym_selected_correct = int(
            (synonym_labels == int(cls_idx)).sum().item()
        )
        canonical_selected_correct = int(
            (canonical_labels == int(cls_idx)).sum().item()
        )
        accumulator["outliers"].append(
            {
                "sample_id": str(sample_id),
                "dataset_index": int(dataset_index),
                "class_idx": int(cls_idx),
                "false_synonym_only": false_count,
                "synonym_only": int(synonym_only_idx.numel()),
                "synonym_only_evaluable": int(valid_synonym_only.numel()),
                "synonym_selected": synonym_selected_count,
                "synonym_evaluable": synonym_evaluable_count,
                "synonym_selected_correct": synonym_selected_correct,
                "synonym_selected_precision": _precision(
                    synonym_selected_correct, synonym_evaluable_count
                ),
                "canonical_selected": canonical_selected_count,
                "canonical_evaluable": canonical_evaluable_count,
                "canonical_selected_correct": canonical_selected_correct,
                "canonical_selected_precision": _precision(
                    canonical_selected_correct, canonical_evaluable_count
                ),
                "synonym_present": bool(
                    float(views.synonym_presence[cls_idx].item()) > float(tau_pos)
                ),
                "canonical_present": bool(
                    float(views.canonical_presence[cls_idx].item()) > float(tau_pos)
                ),
                "alias_false_counts": alias_false_counts,
            }
        )
    accumulator["outliers"] = _prune_outliers(accumulator["outliers"])


def _merge_stats_grid(left: list[list[dict]], right: list[list[dict]]):
    return [
        [
            merge_running_stats(left[row][column], right[row][column])
            for column in range(len(left[row]))
        ]
        for row in range(len(left))
    ]


def _merge_stats_vector(left: list[dict], right: list[dict]):
    return [
        merge_running_stats(left[index], right[index])
        for index in range(len(left))
    ]


def _merge_region_stats(left, right):
    return [
        [
            [
                merge_running_stats(
                    left[pair_idx][cls_idx][region_idx],
                    right[pair_idx][cls_idx][region_idx],
                )
                for region_idx in range(len(left[pair_idx][cls_idx]))
            ]
            for cls_idx in range(len(left[pair_idx]))
        ]
        for pair_idx in range(len(left))
    ]


def merge_audit_accumulators(left: dict, right: dict) -> dict:
    if (
        int(left["num_classes"]) != int(right["num_classes"])
        or int(left["num_queries"]) != int(right["num_queries"])
    ):
        raise ValueError("audit accumulator dimensions do not match")
    output = new_audit_accumulator(
        num_classes=int(left["num_classes"]),
        num_queries=int(left["num_queries"]),
    )
    tensor_keys = (
        "variant_present_images",
        "variant_candidate_images",
        "variant_selected_images",
        "variant_candidate_pixels",
        "variant_pretopk_pixels",
        "variant_selected",
        "variant_evaluable",
        "variant_correct",
        "variant_gt",
        "variant_raw_argmax",
        "alias_winner",
        "alias_evaluable",
        "alias_correct",
        "alias_tie",
        "alias_gt",
        "alias_synonym_only",
        "alias_synonym_only_evaluable",
        "alias_synonym_only_correct",
        "alias_synonym_only_gt",
        "pair_region_count",
        "pair_region_evaluable",
        "pair_region_correct",
        "pair_region_gt",
        "false_synonym_only_raw_argmax",
    )
    for key in tensor_keys:
        output[key] = left[key] + right[key]
    for key in output["variant_stats"]:
        output["variant_stats"][key] = _merge_stats_grid(
            left["variant_stats"][key],
            right["variant_stats"][key],
        )
    for key in output["alias_stats"]:
        output["alias_stats"][key] = _merge_stats_vector(
            left["alias_stats"][key],
            right["alias_stats"][key],
        )
    for key in output["pair_region_stats"]:
        output["pair_region_stats"][key] = _merge_region_stats(
            left["pair_region_stats"][key],
            right["pair_region_stats"][key],
        )
    output["false_synonym_only_margin"] = _merge_stats_vector(
        left["false_synonym_only_margin"],
        right["false_synonym_only_margin"],
    )
    output["outliers"] = _prune_outliers(left["outliers"] + right["outliers"])
    return output


def _histogram_quantile(accumulator: dict, quantile: float) -> float | None:
    count = int(accumulator["count"])
    if count == 0:
        return None
    target = max(1, int(torch.ceil(torch.tensor(float(quantile) * count)).item()))
    cumulative = accumulator["hist"].cumsum(dim=0)
    index = int((cumulative >= target).nonzero(as_tuple=False)[0].item())
    edges = accumulator["bin_edges"]
    return float(((edges[index] + edges[index + 1]) * 0.5).item())


def finalize_running_stats(accumulator: dict) -> dict:
    count = int(accumulator["count"])
    mean = float(accumulator["sum"]) / count if count else None
    variance = (
        max(float(accumulator["sum_sq"]) / count - mean * mean, 0.0)
        if count
        else None
    )
    return {
        "count": count,
        "mean": mean,
        "std": variance**0.5 if variance is not None else None,
        "min": accumulator["min"],
        "max": accumulator["max"],
        "q10": _histogram_quantile(accumulator, 0.10),
        "q25": _histogram_quantile(accumulator, 0.25),
        "q50": _histogram_quantile(accumulator, 0.50),
        "q75": _histogram_quantile(accumulator, 0.75),
        "q90": _histogram_quantile(accumulator, 0.90),
        "hist": [int(value) for value in accumulator["hist"].tolist()],
        "bin_edges": [float(value) for value in accumulator["bin_edges"].tolist()],
    }


def _precision(correct: int, count: int) -> float | None:
    return float(correct) / int(count) if int(count) else None


def _finalize_variant(
    accumulator: dict,
    variant_idx: int,
    class_names: list[str],
) -> dict:
    by_class = []
    for cls_idx, class_name in enumerate(class_names):
        selected = int(accumulator["variant_selected"][variant_idx, cls_idx])
        evaluable = int(
            accumulator["variant_evaluable"][variant_idx, cls_idx]
        )
        correct = int(accumulator["variant_correct"][variant_idx, cls_idx])
        gt_distribution = [
            int(value)
            for value in accumulator["variant_gt"][variant_idx, cls_idx].tolist()
        ]
        gt_background = gt_distribution[0] if gt_distribution else 0
        gt_other = evaluable - correct - gt_background
        by_class.append(
            {
                "class_idx": cls_idx,
                "class_name": class_name,
                "present_images": int(
                    accumulator["variant_present_images"][variant_idx, cls_idx]
                ),
                "candidate_images": int(
                    accumulator["variant_candidate_images"][variant_idx, cls_idx]
                ),
                "selected_images": int(
                    accumulator["variant_selected_images"][variant_idx, cls_idx]
                ),
                "candidate_pixels": int(
                    accumulator["variant_candidate_pixels"][variant_idx, cls_idx]
                ),
                "pretopk_pixels": int(
                    accumulator["variant_pretopk_pixels"][variant_idx, cls_idx]
                ),
                "selected": selected,
                "evaluable": evaluable,
                "correct": correct,
                "precision": _precision(correct, evaluable),
                "gt_background": gt_background,
                "gt_background_fraction": _precision(gt_background, evaluable),
                "gt_other": gt_other,
                "gt_other_fraction": _precision(gt_other, evaluable),
                "gt_distribution": gt_distribution,
                "raw_argmax_agreement": _precision(
                    int(
                        accumulator["variant_raw_argmax"][variant_idx, cls_idx]
                    ),
                    selected,
                ),
                "stats": {
                    key: finalize_running_stats(values[variant_idx][cls_idx])
                    for key, values in accumulator["variant_stats"].items()
                },
            }
        )

    selected = sum(row["selected"] for row in by_class)
    evaluable = sum(row["evaluable"] for row in by_class)
    correct = sum(row["correct"] for row in by_class)
    gt_background = sum(row["gt_background"] for row in by_class)
    gt_other = evaluable - correct - gt_background
    return {
        "name": MINING_VARIANT_NAMES[variant_idx],
        "presence_source": MINING_VARIANT_SOURCES[
            MINING_VARIANT_NAMES[variant_idx]
        ][0],
        "pixel_source": MINING_VARIANT_SOURCES[
            MINING_VARIANT_NAMES[variant_idx]
        ][1],
        "overall": {
            "selected": selected,
            "evaluable": evaluable,
            "correct": correct,
            "precision": _precision(correct, evaluable),
            "gt_background": gt_background,
            "gt_background_fraction": _precision(gt_background, evaluable),
            "gt_other": gt_other,
            "gt_other_fraction": _precision(gt_other, evaluable),
            "raw_argmax_agreement": _precision(
                int(accumulator["variant_raw_argmax"][variant_idx].sum()),
                selected,
            ),
        },
        "by_class": by_class,
    }


def _finalize_region(
    accumulator: dict,
    pair_idx: int,
    cls_idx: int,
    region_idx: int,
) -> dict:
    count = int(accumulator["pair_region_count"][pair_idx, cls_idx, region_idx])
    evaluable = int(
        accumulator["pair_region_evaluable"][pair_idx, cls_idx, region_idx]
    )
    correct = int(
        accumulator["pair_region_correct"][pair_idx, cls_idx, region_idx]
    )
    return {
        "count": count,
        "evaluable_count": evaluable,
        "correct": correct,
        "precision": _precision(correct, evaluable),
        "gt_distribution": [
            int(value)
            for value in accumulator[
                "pair_region_gt"
            ][pair_idx, cls_idx, region_idx].tolist()
        ],
        "stats": {
            key: finalize_running_stats(values[pair_idx][cls_idx][region_idx])
            for key, values in accumulator["pair_region_stats"].items()
        },
    }


def _finalize_pairs(accumulator: dict, class_names: list[str]) -> list[dict]:
    output = []
    for pair_idx, (left_idx, right_idx) in enumerate(MINING_VARIANT_PAIRS):
        rows = []
        for cls_idx, class_name in enumerate(class_names):
            regions = {
                name: _finalize_region(accumulator, pair_idx, cls_idx, region_idx)
                for region_idx, name in enumerate(SET_REGION_NAMES)
            }
            union_count = regions["union"]["count"]
            rows.append(
                {
                    "class_idx": cls_idx,
                    "class_name": class_name,
                    "regions": regions,
                    "jaccard": (
                        regions["intersection"]["count"] / union_count
                        if union_count
                        else None
                    ),
                }
            )
        output.append(
            {
                "left": MINING_VARIANT_NAMES[left_idx],
                "right": MINING_VARIANT_NAMES[right_idx],
                "by_class": rows,
            }
        )
    return output


def _finalize_aliases(
    accumulator: dict,
    *,
    query_words: list[str],
    query_idx_list: list[int],
    canonical_query_ids: tuple[int, ...],
) -> list[dict]:
    canonical_set = set(int(value) for value in canonical_query_ids)
    class_winner_totals = {
        cls_idx: sum(
            int(accumulator["alias_winner"][query_id])
            for query_id, mapped in enumerate(query_idx_list)
            if int(mapped) == int(cls_idx)
        )
        for cls_idx in set(int(value) for value in query_idx_list)
    }
    class_synonym_only_totals = {
        cls_idx: sum(
            int(accumulator["alias_synonym_only"][query_id])
            for query_id, mapped in enumerate(query_idx_list)
            if int(mapped) == int(cls_idx)
        )
        for cls_idx in set(int(value) for value in query_idx_list)
    }
    aliases = []
    for query_id, (query_word, cls_idx) in enumerate(
        zip(query_words, query_idx_list)
    ):
        count = int(accumulator["alias_winner"][query_id])
        evaluable = int(accumulator["alias_evaluable"][query_id])
        correct = int(accumulator["alias_correct"][query_id])
        synonym_only_count = int(accumulator["alias_synonym_only"][query_id])
        synonym_only_evaluable = int(
            accumulator["alias_synonym_only_evaluable"][query_id]
        )
        synonym_only_correct = int(
            accumulator["alias_synonym_only_correct"][query_id]
        )
        aliases.append(
            {
                "query_id": query_id,
                "query_word": query_word,
                "class_idx": int(cls_idx),
                "is_canonical": query_id in canonical_set,
                "winner_count": count,
                "winner_evaluable": evaluable,
                "winner_fraction": _precision(
                    count, class_winner_totals[int(cls_idx)]
                ),
                "winner_correct": correct,
                "winner_precision": _precision(correct, evaluable),
                "tie_count": int(accumulator["alias_tie"][query_id]),
                "tie_fraction": _precision(
                    int(accumulator["alias_tie"][query_id]), count
                ),
                "gt_distribution": [
                    int(value)
                    for value in accumulator["alias_gt"][query_id].tolist()
                ],
                "synonym_only_count": synonym_only_count,
                "synonym_only_evaluable": synonym_only_evaluable,
                "synonym_only_fraction": _precision(
                    synonym_only_count,
                    class_synonym_only_totals[int(cls_idx)],
                ),
                "synonym_only_correct": synonym_only_correct,
                "synonym_only_precision": _precision(
                    synonym_only_correct, synonym_only_evaluable
                ),
                "synonym_only_gt_distribution": [
                    int(value)
                    for value in accumulator[
                        "alias_synonym_only_gt"
                    ][query_id].tolist()
                ],
                "stats": {
                    key: finalize_running_stats(values[query_id])
                    for key, values in accumulator["alias_stats"].items()
                },
            }
        )
    return aliases


def _causal_delta(target: dict, baseline: dict) -> dict:
    target_overall = target["overall"]
    baseline_overall = baseline["overall"]
    target_precision = target_overall["precision"]
    baseline_precision = baseline_overall["precision"]
    return {
        "selected_delta": int(target_overall["selected"])
        - int(baseline_overall["selected"]),
        "correct_delta": int(target_overall["correct"])
        - int(baseline_overall["correct"]),
        "precision_delta": (
            float(target_precision) - float(baseline_precision)
            if target_precision is not None and baseline_precision is not None
            else None
        ),
        "target_variant": target["name"],
        "baseline_variant": baseline["name"],
    }


def _causal_decomposition(variants: dict[str, dict]) -> dict:
    synonym_both = variants["syn_presence__syn_pixels"]
    canonical_both = variants["can_presence__can_pixels"]
    synonym_presence = variants["syn_presence__can_pixels"]
    synonym_pixels = variants["can_presence__syn_pixels"]
    combined = _causal_delta(synonym_both, canonical_both)
    presence_at_canonical = _causal_delta(synonym_presence, canonical_both)
    pixels_at_canonical = _causal_delta(synonym_pixels, canonical_both)
    presence_at_synonym = _causal_delta(synonym_both, synonym_pixels)
    pixels_at_synonym = _causal_delta(synonym_both, synonym_presence)
    interaction = {
        "selected_delta": (
            combined["selected_delta"]
            - presence_at_canonical["selected_delta"]
            - pixels_at_canonical["selected_delta"]
        ),
        "correct_delta": (
            combined["correct_delta"]
            - presence_at_canonical["correct_delta"]
            - pixels_at_canonical["correct_delta"]
        ),
        "precision_delta": (
            combined["precision_delta"]
            - presence_at_canonical["precision_delta"]
            - pixels_at_canonical["precision_delta"]
            if combined["precision_delta"] is not None
            and presence_at_canonical["precision_delta"] is not None
            and pixels_at_canonical["precision_delta"] is not None
            else None
        ),
        "formula": "both - presence_only - pixel_only",
    }
    return {
        "presence_only_at_canonical_pixels": presence_at_canonical,
        "pixel_only_at_canonical_presence": pixels_at_canonical,
        "presence_only_at_synonym_pixels": presence_at_synonym,
        "pixel_only_at_synonym_presence": pixels_at_synonym,
        "combined_synonym_effect": combined,
        "interaction": interaction,
    }


def _root_cause_assessment(
    *,
    guard: dict,
    variants: dict[str, dict],
    pairs: list[dict],
    causal_decomposition: dict,
) -> dict:
    if guard.get("applicable") and guard.get("passed") is False:
        return {
            "status": "INVALID AUDIT",
            "reason": "Synonym baseline reproduction guard failed.",
        }
    primary_pair = pairs[0]
    intersection_count = sum(
        row["regions"]["intersection"]["count"]
        for row in primary_pair["by_class"]
    )
    intersection_correct = sum(
        row["regions"]["intersection"]["correct"]
        for row in primary_pair["by_class"]
    )
    intersection_evaluable = sum(
        row["regions"]["intersection"]["evaluable_count"]
        for row in primary_pair["by_class"]
    )
    synonym_only_count = sum(
        row["regions"]["left_only"]["count"]
        for row in primary_pair["by_class"]
    )
    synonym_only_correct = sum(
        row["regions"]["left_only"]["correct"]
        for row in primary_pair["by_class"]
    )
    synonym_only_evaluable = sum(
        row["regions"]["left_only"]["evaluable_count"]
        for row in primary_pair["by_class"]
    )
    intersection_precision = _precision(
        intersection_correct, intersection_evaluable
    )
    synonym_only_precision = _precision(
        synonym_only_correct, synonym_only_evaluable
    )
    synonym_precision = variants[MINING_VARIANT_NAMES[0]]["overall"]["precision"]
    canonical_precision = variants[MINING_VARIANT_NAMES[1]]["overall"]["precision"]
    supported = bool(
        synonym_only_count
        and synonym_only_precision is not None
        and (
            (
                intersection_precision is not None
                and synonym_only_precision < intersection_precision
            )
            or (
                synonym_precision is not None
                and canonical_precision is not None
                and synonym_precision < canonical_precision
            )
        )
    )
    return {
        "status": "SUPPORTED" if supported else "NOT SUPPORTED",
        "synonym_only_count": synonym_only_count,
        "synonym_only_evaluable": synonym_only_evaluable,
        "synonym_only_precision": synonym_only_precision,
        "intersection_count": intersection_count,
        "intersection_evaluable": intersection_evaluable,
        "intersection_precision": intersection_precision,
        "synonym_miner_precision": synonym_precision,
        "canonical_miner_precision": canonical_precision,
        "combined_synonym_effect": causal_decomposition[
            "combined_synonym_effect"
        ],
        "presence_only_effect": causal_decomposition[
            "presence_only_at_canonical_pixels"
        ],
        "pixel_only_effect": causal_decomposition[
            "pixel_only_at_canonical_presence"
        ],
        "interaction_effect": causal_decomposition["interaction"],
    }


def finalize_audit(
    accumulator: dict,
    *,
    class_names: list[str],
    query_words: list[str],
    query_idx_list: list[int],
    canonical_query_ids: tuple[int, ...],
    sample_ids: list[str],
    synonym_baseline: dict,
    canonical_baseline: dict,
    expected_synonym_miou: float,
    reproduction_tolerance: float,
    enforce_reproduction_guard: bool,
    parameters: dict,
) -> dict:
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample ID in audit")
    if len(class_names) != int(accumulator["num_classes"]):
        raise ValueError("class name/accumulator size mismatch")
    if len(query_words) != int(accumulator["num_queries"]):
        raise ValueError("query word/accumulator size mismatch")
    if len(query_idx_list) != len(query_words):
        raise ValueError("query word/mapping size mismatch")

    if enforce_reproduction_guard:
        difference = abs(
            float(synonym_baseline["miou"]) - float(expected_synonym_miou)
        )
        guard = {
            "applicable": True,
            "expected_synonym_miou": float(expected_synonym_miou),
            "actual_synonym_miou": float(synonym_baseline["miou"]),
            "absolute_difference": difference,
            "tolerance": float(reproduction_tolerance),
            "passed": difference <= float(reproduction_tolerance),
        }
    else:
        guard = {"applicable": False, "passed": None}

    variants = {
        name: _finalize_variant(accumulator, variant_idx, class_names)
        for variant_idx, name in enumerate(MINING_VARIANT_NAMES)
    }
    pairs = _finalize_pairs(accumulator, class_names)
    causal_decomposition = _causal_decomposition(variants)
    result = {
        "audit_type": "loveda_prompt_mining",
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_unique_images": len(sample_ids),
        "sample_ids": sorted(str(value) for value in sample_ids),
        "parameters": parameters,
        "class_names": list(class_names),
        "query_words": list(query_words),
        "query_idx_list": [int(value) for value in query_idx_list],
        "canonical_query_ids": [int(value) for value in canonical_query_ids],
        "baselines": {
            "synonym": synonym_baseline,
            "canonical": canonical_baseline,
        },
        "reproduction_guard": guard,
        "mining_variants": variants,
        "causal_decomposition": causal_decomposition,
        "selected_set_pairs": pairs,
        "aliases": _finalize_aliases(
            accumulator,
            query_words=query_words,
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_query_ids,
        ),
        "false_synonym_only_by_class": [
            {
                "class_idx": cls_idx,
                "class_name": class_name,
                "false_count": int(
                    accumulator["false_synonym_only_margin"][cls_idx]["count"]
                ),
                "raw_argmax_count": int(
                    accumulator["false_synonym_only_raw_argmax"][cls_idx]
                ),
                "raw_argmax_rate": _precision(
                    int(accumulator["false_synonym_only_raw_argmax"][cls_idx]),
                    int(
                        accumulator["false_synonym_only_margin"][cls_idx][
                            "count"
                        ]
                    ),
                ),
                "margin": finalize_running_stats(
                    accumulator["false_synonym_only_margin"][cls_idx]
                ),
            }
            for cls_idx, class_name in enumerate(class_names)
        ],
        "worst_images": list(accumulator["outliers"]),
    }
    result["root_cause_assessment"] = _root_cause_assessment(
        guard=guard,
        variants=variants,
        pairs=pairs,
        causal_decomposition=causal_decomposition,
    )
    return result


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.3f}%"


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _format_delta_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):+.3f} pp"


def _format_alias_breakdown(row: dict, result: dict) -> str:
    return ", ".join(
        f"{result['query_words'][int(query_id)]}:{int(count):,}"
        for query_id, count in sorted(
            row.get("alias_false_counts", {}).items(),
            key=lambda item: int(item[0]),
        )
    ) or "none"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def render_markdown_report(result: dict) -> str:
    guard = result["reproduction_guard"]
    lines = ["# LoveDA Canonical-vs-Synonym Offline Mining Audit", ""]
    lines.extend(["## Validity", ""])
    if not guard.get("applicable"):
        lines.append("Partial audit: reproduction guard is not applicable.")
    elif guard.get("passed"):
        lines.append(
            "VALID: synonym baseline reproduction difference "
            f"{guard['absolute_difference']:.6f} is within "
            f"{guard['tolerance']:.6f}."
        )
    else:
        lines.append(
            "INVALID AUDIT: synonym baseline reproduction difference "
            f"{guard['absolute_difference']:.6f} exceeds "
            f"{guard['tolerance']:.6f}."
        )

    lines.extend(["", "## Baseline comparison", ""])
    lines.extend(
        _markdown_table(
            ["Prompt view", "mIoU"],
            [
                [
                    "Synonym max",
                    _format_percent(result["baselines"]["synonym"]["miou"]),
                ],
                [
                    "Canonical",
                    _format_percent(result["baselines"]["canonical"]["miou"]),
                ],
            ],
        )
    )

    synonym = result["mining_variants"][MINING_VARIANT_NAMES[0]]
    canonical = result["mining_variants"][MINING_VARIANT_NAMES[1]]
    lines.extend(["", "## Presence expansion", ""])
    lines.extend(
        _markdown_table(
            ["Class", "Synonym present", "Canonical present", "Difference"],
            [
                [
                    syn_row["class_name"],
                    f"{syn_row['present_images']:,}",
                    f"{can_row['present_images']:,}",
                    f"{syn_row['present_images'] - can_row['present_images']:+,}",
                ]
                for syn_row, can_row in zip(
                    synonym["by_class"], canonical["by_class"]
                )
            ],
        )
    )

    lines.extend(["", "## Mining volume and purity", ""])
    lines.extend(
        _markdown_table(
            [
                "Variant",
                "Selected",
                "Evaluable",
                "Precision",
                "GT background",
                "GT other",
            ],
            [
                [
                    name,
                    f"{row['overall']['selected']:,}",
                    f"{row['overall']['evaluable']:,}",
                    _format_percent(row["overall"]["precision"]),
                    _format_percent(row["overall"]["gt_background_fraction"]),
                    _format_percent(row["overall"]["gt_other_fraction"]),
                ]
                for name, row in result["mining_variants"].items()
            ],
        )
    )

    primary = result["selected_set_pairs"][0]
    lines.extend(["", "## Canonical/synonym selected-set overlap", ""])
    lines.extend(
        _markdown_table(
            [
                "Class",
                "Intersection",
                "Intersection evaluable",
                "Intersection precision",
                "Synonym-only",
                "Synonym-only evaluable",
                "Synonym-only precision",
                "Canonical-only",
            ],
            [
                [
                    row["class_name"],
                    f"{row['regions']['intersection']['count']:,}",
                    f"{row['regions']['intersection']['evaluable_count']:,}",
                    _format_percent(row["regions"]["intersection"]["precision"]),
                    f"{row['regions']['left_only']['count']:,}",
                    f"{row['regions']['left_only']['evaluable_count']:,}",
                    _format_percent(row["regions"]["left_only"]["precision"]),
                    f"{row['regions']['right_only']['count']:,}",
                ]
                for row in primary["by_class"]
            ],
        )
    )
    lines.extend(["", "Selected-region score diagnostics:", ""])
    lines.extend(
        _markdown_table(
            [
                "Class",
                "Region",
                "Count",
                "Evaluable",
                "Precision",
                "Canonical score mean",
                "Synonym score mean",
                "Score gap mean",
                "Synonym margin mean",
            ],
            [
                [
                    row["class_name"],
                    region_name,
                    f"{region['count']:,}",
                    f"{region['evaluable_count']:,}",
                    _format_percent(region["precision"]),
                    _format_number(region["stats"]["canonical_score"]["mean"]),
                    _format_number(region["stats"]["synonym_score"]["mean"]),
                    _format_number(region["stats"]["score_gap"]["mean"]),
                    _format_number(region["stats"]["synonym_margin"]["mean"]),
                ]
                for row in primary["by_class"]
                for region_name, region in row["regions"].items()
                if region_name != "union"
            ],
        )
    )

    lines.extend(["", "## Alias attribution", ""])
    lines.extend(
        _markdown_table(
            [
                "Alias",
                "Class",
                "Canonical",
                "Winners",
                "Winner evaluable",
                "Winner fraction",
                "Winner precision",
                "Synonym-only",
                "Synonym-only evaluable",
                "Synonym-only fraction",
                "Synonym-only precision",
            ],
            [
                [
                    row["query_word"],
                    result["class_names"][row["class_idx"]],
                    "yes" if row["is_canonical"] else "no",
                    f"{row['winner_count']:,}",
                    f"{row['winner_evaluable']:,}",
                    _format_percent(row["winner_fraction"]),
                    _format_percent(row["winner_precision"]),
                    f"{row['synonym_only_count']:,}",
                    f"{row['synonym_only_evaluable']:,}",
                    _format_percent(row["synonym_only_fraction"]),
                    _format_percent(row["synonym_only_precision"]),
                ]
                for row in result["aliases"]
            ],
        )
    )
    lines.extend(["", "False synonym-only diagnostics:", ""])
    lines.extend(
        _markdown_table(
            [
                "Class",
                "False synonym-only",
                "Raw-argmax count",
                "False-only raw-argmax rate",
                "Synonym margin median",
            ],
            [
                [
                    row["class_name"],
                    f"{row['false_count']:,}",
                    f"{row['raw_argmax_count']:,}",
                    _format_percent(row["raw_argmax_rate"]),
                    _format_number(row["margin"]["q50"]),
                ]
                for row in result["false_synonym_only_by_class"]
            ],
        )
    )

    lines.extend(["", "## Presence-vs-pixel decomposition", ""])
    lines.extend(
        _markdown_table(
            ["Variant", "Presence", "Pixels", "Selected", "Precision"],
            [
                [
                    name,
                    row["presence_source"],
                    row["pixel_source"],
                    f"{row['overall']['selected']:,}",
                    _format_percent(row["overall"]["precision"]),
                ]
                for name, row in result["mining_variants"].items()
            ],
        )
    )
    causal_labels = {
        "presence_only_at_canonical_pixels": "Presence only at canonical pixels",
        "pixel_only_at_canonical_presence": "Pixel only at canonical presence",
        "presence_only_at_synonym_pixels": "Presence only at synonym pixels",
        "pixel_only_at_synonym_presence": "Pixel only at synonym presence",
        "combined_synonym_effect": "Combined synonym effect",
        "interaction": "Presence × pixel interaction",
    }
    lines.extend(["", "Four-variant causal deltas:", ""])
    lines.extend(
        _markdown_table(
            ["Change", "Selected delta", "Correct delta", "Precision delta"],
            [
                [
                    causal_labels[name],
                    f"{effect['selected_delta']:+,}",
                    f"{effect['correct_delta']:+,}",
                    _format_delta_percent(effect["precision_delta"]),
                ]
                for name, effect in result["causal_decomposition"].items()
            ],
        )
    )

    lines.extend(["", "## Worst affected images", ""])
    if result["worst_images"]:
        lines.extend(
            _markdown_table(
                [
                    "Sample",
                    "Class",
                    "Syn present",
                    "Can present",
                    "Syn selected",
                    "Syn evaluable",
                    "Syn correct",
                    "Syn precision",
                    "Can selected",
                    "Can evaluable",
                    "Can correct",
                    "Can precision",
                    "False synonym-only",
                    "Alias false winners",
                ],
                [
                    [
                        row["sample_id"],
                        result["class_names"][row["class_idx"]],
                        "yes" if row["synonym_present"] else "no",
                        "yes" if row["canonical_present"] else "no",
                        f"{row['synonym_selected']:,}",
                        f"{row['synonym_evaluable']:,}",
                        f"{row['synonym_selected_correct']:,}",
                        _format_percent(row["synonym_selected_precision"]),
                        f"{row['canonical_selected']:,}",
                        f"{row['canonical_evaluable']:,}",
                        f"{row['canonical_selected_correct']:,}",
                        _format_percent(row["canonical_selected_precision"]),
                        f"{row['false_synonym_only']:,}",
                        _format_alias_breakdown(row, result),
                    ]
                    for row in result["worst_images"]
                ],
            )
        )
    else:
        lines.append("No false synonym-only outliers were recorded.")

    assessment = result["root_cause_assessment"]
    lines.extend(["", "## Root-cause hypothesis", ""])
    lines.append(f"**{assessment['status']}**")
    if assessment.get("reason"):
        lines.append(assessment["reason"])
    elif assessment["status"] == "SUPPORTED":
        lines.append(
            "Synonym-only selected points are less precise than shared or "
            "canonical-selected supervision, supporting synonym mining noise."
        )
    else:
        lines.append(
            "The observed selected-set precision does not support the proposed "
            "synonym mining noise mechanism."
        )
    return "\n".join(lines).rstrip() + "\n"
