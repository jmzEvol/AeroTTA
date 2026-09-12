from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from .alias_fusion import (
    class_scores_from_query_scores_with_alias_reliability,
    class_scores_from_query_scores_with_frozen_alias_residual,
    estimate_alias_reliability,
)
from .prompt_causal_diagnostic import (
    CrossTimePromptCausalAccumulator,
)
from .scores import predict_from_class_scores


CORE_TIER = 1
SHOULDER_TIER = 2


@dataclass(frozen=True)
class PromptTransferConfig:
    anchor_k: int = 2048
    margins: tuple[float, ...] = (0.0, 0.02, 0.05)
    safety_quantiles: tuple[float, ...] = (0.1, 0.25, 0.5)
    min_active: int = 32
    epsilon: float = 1e-6
    pair_row_limit: int = 200

    def __post_init__(self) -> None:
        if int(self.anchor_k) <= 0:
            raise ValueError("anchor_k must be positive")
        if int(self.min_active) <= 0:
            raise ValueError("min_active must be positive")
        if float(self.epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        if int(self.pair_row_limit) < 0:
            raise ValueError("pair_row_limit must be non-negative")
        if not self.margins or any(
            float(value) < 0.0 for value in self.margins
        ):
            raise ValueError("margins must contain non-negative values")
        if not self.safety_quantiles or any(
            float(value) < 0.0 or float(value) > 1.0
            for value in self.safety_quantiles
        ):
            raise ValueError("safety_quantiles must be in [0, 1]")


@dataclass(frozen=True)
class FrozenAnchorTiers:
    class_ids: torch.Tensor
    tiers: torch.Tensor
    canonical_scores: torch.Tensor
    canonical_margins: torch.Tensor
    core_counts: tuple[int, ...]
    shoulder_counts: tuple[int, ...]


@dataclass(frozen=True)
class PairTransferStats:
    query_id: int
    class_id: int
    foreign_class_id: int
    tier: int
    margin: float
    anchor_count: int
    protected_count: int
    active_residual_count: int
    new_takeover_count: int
    new_takeover_severity_sum: float
    deepen_count: int
    deepen_sum: float
    total_erosion_sum: float
    direct_takeover_count: int
    direct_active_count: int
    rho_q10: float | None
    rho_q25: float | None
    rho_median: float | None


@dataclass(frozen=True)
class PromptTransferSnapshot:
    class_scores: torch.Tensor
    pairs: dict[tuple[int, int, int, float], PairTransferStats]


@dataclass(frozen=True)
class RoutedPromptScores:
    routed_query_scores: torch.Tensor
    class_scores: torch.Tensor
    winner_query_ids: torch.Tensor
    activated_pairs: tuple[tuple[int, int, int, float], ...]


@dataclass(frozen=True)
class PromptRoutingSnapshotResult:
    head_confusions: dict[str, torch.Tensor]
    head_image_miou: dict[str, float]
    head_change_counts: dict[str, dict[str, int]]
    pair_stats: dict[tuple[int, int, int, float], PairTransferStats]
    prompt_standalone_confusions: dict[int, torch.Tensor]
    prompt_leave_one_out_confusions: dict[int, torch.Tensor]
    pair_foreign_iou_effects: dict[tuple[int, int], float]
    gt_transfer_counts: dict[str, int]
    prompt_winner_counts: torch.Tensor


def group_query_ids_by_class(
    *,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    num_classes = len(canonical_query_ids)
    if num_classes == 0:
        raise ValueError("canonical_query_ids must not be empty")
    if {int(value) for value in query_idx_list} != set(range(num_classes)):
        raise ValueError(
            "query mapping must cover exactly the canonical classes"
        )

    groups = []
    for class_id, canonical_value in enumerate(canonical_query_ids):
        canonical_id = int(canonical_value)
        if canonical_id < 0 or canonical_id >= len(query_idx_list):
            raise ValueError("canonical query id is outside the query mapping")
        if int(query_idx_list[canonical_id]) != class_id:
            raise ValueError(
                f"canonical query {canonical_id} does not map to class "
                f"{class_id}"
            )
        members = tuple(
            query_id
            for query_id, mapped_class in enumerate(query_idx_list)
            if int(mapped_class) == class_id and query_id != canonical_id
        )
        groups.append((canonical_id, *members))
    return tuple(groups)


def build_frozen_anchor_tiers(
    *,
    query_scores: torch.Tensor,
    canonical_query_ids: Sequence[int],
    prob_thd: float,
    anchor_k: int,
) -> FrozenAnchorTiers:
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError("query_scores must have shape [1,Q,H,W]")
    if int(anchor_k) <= 0:
        raise ValueError("anchor_k must be positive")
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    if len(canonical_ids) < 2:
        raise ValueError(
            "prompt transfer routing requires at least two classes"
        )
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("canonical query ids must be unique")
    num_queries = int(query_scores.shape[1])
    if any(value < 0 or value >= num_queries for value in canonical_ids):
        raise ValueError("canonical query id is outside the score tensor")

    canonical_index = torch.as_tensor(
        canonical_ids,
        device=query_scores.device,
        dtype=torch.long,
    )
    canonical_scores = (
        query_scores.detach().float()[0].index_select(0, canonical_index)
    )
    top2 = canonical_scores.topk(k=2, dim=0)
    winner = top2.indices[0]
    margin = top2.values[0] - top2.values[1]
    confident = top2.values[0] >= float(prob_thd)
    class_ids = torch.full_like(winner, -1, dtype=torch.long)
    tiers = torch.zeros_like(winner, dtype=torch.uint8)
    class_ids[confident] = winner[confident]

    flat_classes = class_ids.flatten()
    flat_tiers = tiers.flatten()
    flat_margin = margin.flatten()
    core_counts = []
    shoulder_counts = []
    for class_id in range(int(canonical_scores.shape[0])):
        candidates = (
            (flat_classes == class_id).nonzero(as_tuple=False).flatten()
        )
        ordered = candidates[
            torch.argsort(flat_margin[candidates], descending=True)
        ]
        core = ordered[: int(anchor_k)]
        shoulder = ordered[int(anchor_k) : 2 * int(anchor_k)]
        remainder = ordered[2 * int(anchor_k) :]
        flat_tiers[core] = CORE_TIER
        flat_tiers[shoulder] = SHOULDER_TIER
        flat_classes[remainder] = -1
        core_counts.append(int(core.numel()))
        shoulder_counts.append(int(shoulder.numel()))

    return FrozenAnchorTiers(
        class_ids=class_ids,
        tiers=tiers,
        canonical_scores=canonical_scores,
        canonical_margins=margin,
        core_counts=tuple(core_counts),
        shoulder_counts=tuple(shoulder_counts),
    )


def _quantile_or_none(
    values: torch.Tensor,
    quantile: float,
) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(torch.quantile(values.float(), float(quantile)).item())


def measure_prompt_transfer(
    *,
    query_scores: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
    anchors: FrozenAnchorTiers,
    margin: float,
    epsilon: float,
) -> PromptTransferSnapshot:
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError("query_scores must have shape [1,Q,H,W]")
    if len(query_idx_list) != int(query_scores.shape[1]):
        raise ValueError("query score/mapping length mismatch")
    if float(margin) < 0.0:
        raise ValueError("margin must be non-negative")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if tuple(anchors.class_ids.shape) != tuple(query_scores.shape[-2:]):
        raise ValueError("anchor/query score spatial shape mismatch")

    query_groups = group_query_ids_by_class(
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
    )
    scores = query_scores.detach().float()
    canonical_index = torch.as_tensor(
        tuple(int(value) for value in canonical_query_ids),
        device=scores.device,
        dtype=torch.long,
    )
    canonical_scores = scores.index_select(1, canonical_index)[0]
    class_scores = torch.stack(
        [scores[:, list(query_ids)].amax(dim=1) for query_ids in query_groups],
        dim=1,
    )
    anchor_classes = anchors.class_ids.to(device=scores.device)
    anchor_tiers = anchors.tiers.to(device=scores.device)

    pairs: dict[tuple[int, int, int, float], PairTransferStats] = {}
    margin_value = float(margin)
    epsilon_value = float(epsilon)
    for class_id, query_ids in enumerate(query_groups):
        if len(query_ids) < 2:
            continue
        for query_id in query_ids:
            sibling_ids = [
                candidate
                for candidate in query_ids
                if int(candidate) != int(query_id)
            ]
            prompt_score = scores[0, int(query_id)]
            sibling_score = scores[0, sibling_ids].amax(dim=0)
            residual = (prompt_score - sibling_score).clamp_min(0.0)

            for foreign_class_id in range(len(query_groups)):
                if foreign_class_id == class_id:
                    continue
                foreign_score = canonical_scores[foreign_class_id]
                z0 = sibling_score - foreign_score
                z1 = torch.maximum(sibling_score, prompt_score) - foreign_score
                erosion = (
                    (z1 + margin_value).clamp_min(0.0)
                    - (z0 + margin_value).clamp_min(0.0)
                )
                for tier in (CORE_TIER, SHOULDER_TIER):
                    anchor_mask = (
                        (anchor_classes == foreign_class_id)
                        & (anchor_tiers == int(tier))
                    )
                    protected = anchor_mask & (z0 <= -margin_value)
                    active = anchor_mask & (residual > epsilon_value)
                    new_takeover = (
                        active
                        & (z0 <= -margin_value)
                        & (z1 > -margin_value)
                    )
                    deepen = active & (z0 > -margin_value)

                    direct_active = (
                        anchor_mask & (prompt_score > epsilon_value)
                    )
                    direct_safe_score = (
                        foreign_score - margin_value
                    ).clamp_min(0.0)
                    direct_takeover = direct_active & (
                        prompt_score > direct_safe_score
                    )
                    rho_values = (
                        direct_safe_score[direct_active]
                        / prompt_score[direct_active].clamp_min(epsilon_value)
                    )
                    key = (
                        int(query_id),
                        int(foreign_class_id),
                        int(tier),
                        margin_value,
                    )
                    pairs[key] = PairTransferStats(
                        query_id=int(query_id),
                        class_id=int(class_id),
                        foreign_class_id=int(foreign_class_id),
                        tier=int(tier),
                        margin=margin_value,
                        anchor_count=int(anchor_mask.sum().item()),
                        protected_count=int(protected.sum().item()),
                        active_residual_count=int(active.sum().item()),
                        new_takeover_count=int(new_takeover.sum().item()),
                        new_takeover_severity_sum=float(
                            (z1 + margin_value)
                            .clamp_min(0.0)[new_takeover]
                            .sum()
                            .item()
                        ),
                        deepen_count=int(deepen.sum().item()),
                        deepen_sum=float(
                            (z1 - z0)[deepen].sum().item()
                        ),
                        total_erosion_sum=float(
                            erosion[anchor_mask].sum().item()
                        ),
                        direct_takeover_count=int(
                            direct_takeover.sum().item()
                        ),
                        direct_active_count=int(direct_active.sum().item()),
                        rho_q10=_quantile_or_none(rho_values, 0.1),
                        rho_q25=_quantile_or_none(rho_values, 0.25),
                        rho_median=_quantile_or_none(rho_values, 0.5),
                    )

    return PromptTransferSnapshot(
        class_scores=class_scores,
        pairs=pairs,
    )


def _pair_safety_quantile(
    stats: PairTransferStats,
    quantile: float,
) -> float | None:
    if float(quantile) == 0.1:
        return stats.rho_q10
    if float(quantile) == 0.25:
        return stats.rho_q25
    if float(quantile) == 0.5:
        return stats.rho_median
    raise ValueError("shoulder_quantile must be one of 0.1, 0.25, or 0.5")


def route_prompt_scores(
    *,
    query_scores: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
    anchors: FrozenAnchorTiers,
    margin: float,
    enabled_tiers: Sequence[int],
    min_active: int,
    shoulder_quantile: float | None,
    epsilon: float,
    transfer_snapshot: PromptTransferSnapshot | None = None,
) -> RoutedPromptScores:
    if int(min_active) <= 0:
        raise ValueError("min_active must be positive")
    tiers = tuple(int(value) for value in enabled_tiers)
    if not tiers or any(
        value not in {CORE_TIER, SHOULDER_TIER} for value in tiers
    ):
        raise ValueError("enabled_tiers must contain core or shoulder")
    if (
        shoulder_quantile is not None
        and SHOULDER_TIER not in tiers
    ):
        raise ValueError(
            "shoulder_quantile requires the shoulder tier to be enabled"
        )

    snapshot = transfer_snapshot
    if snapshot is None:
        snapshot = measure_prompt_transfer(
            query_scores=query_scores,
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_query_ids,
            anchors=anchors,
            margin=margin,
            epsilon=epsilon,
        )
    elif any(
        float(pair_key[3]) != float(margin)
        for pair_key in snapshot.pairs
    ):
        raise ValueError("transfer snapshot margin does not match routing")
    query_groups = group_query_ids_by_class(
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
    )
    routed_scores = query_scores.detach().float().clone()
    canonical_index = torch.as_tensor(
        tuple(int(value) for value in canonical_query_ids),
        device=routed_scores.device,
        dtype=torch.long,
    )
    canonical_scores = routed_scores.index_select(1, canonical_index)[0]
    anchor_classes = anchors.class_ids.to(device=routed_scores.device)
    anchor_tiers = anchors.tiers.to(device=routed_scores.device)
    margin_value = float(margin)
    activated_pairs = []

    for class_id, query_ids in enumerate(query_groups):
        if len(query_ids) < 2:
            continue
        for query_id in query_ids:
            for foreign_class_id in range(len(query_groups)):
                if foreign_class_id == class_id:
                    continue
                foreign_score = canonical_scores[foreign_class_id]
                strict_limit = torch.nextafter(
                    foreign_score - margin_value,
                    torch.full_like(foreign_score, -torch.inf),
                )
                for tier in tiers:
                    pair_key = (
                        int(query_id),
                        int(foreign_class_id),
                        int(tier),
                        margin_value,
                    )
                    if tier == SHOULDER_TIER and shoulder_quantile is not None:
                        stats = snapshot.pairs[pair_key]
                        pair_quantile = _pair_safety_quantile(
                            stats,
                            shoulder_quantile,
                        )
                        if (
                            stats.direct_active_count < int(min_active)
                            or pair_quantile is None
                            or float(pair_quantile) >= 1.0
                        ):
                            continue
                    routing_mask = (
                        (anchor_classes == int(foreign_class_id))
                        & (anchor_tiers == int(tier))
                    )
                    if not bool(routing_mask.any()):
                        continue
                    query_map = routed_scores[0, int(query_id)]
                    query_map[routing_mask] = torch.minimum(
                        query_map[routing_mask],
                        strict_limit[routing_mask],
                    )
                    activated_pairs.append(pair_key)

    class_scores = []
    winner_query_ids = []
    for query_ids in query_groups:
        local = routed_scores[:, list(query_ids)]
        class_score, local_winner = local.max(dim=1)
        global_ids = torch.as_tensor(
            query_ids,
            device=routed_scores.device,
            dtype=torch.long,
        )
        class_scores.append(class_score)
        winner_query_ids.append(global_ids[local_winner])

    return RoutedPromptScores(
        routed_query_scores=routed_scores,
        class_scores=torch.stack(class_scores, dim=1),
        winner_query_ids=torch.stack(winner_query_ids, dim=1),
        activated_pairs=tuple(activated_pairs),
    )


def _confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    pred = pred.to(dtype=torch.long)
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


def _confusion_metrics(
    confusion: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    values = confusion.detach().float()
    true_positive = torch.diag(values)
    union = values.sum(dim=1) + values.sum(dim=0) - true_positive
    valid = union > 0
    iou = torch.zeros_like(true_positive)
    iou[valid] = true_positive[valid] / union[valid].clamp_min(1.0)
    miou = iou[valid].mean() if bool(valid.any()) else iou.new_tensor(0.0)
    return float(miou.item() * 100.0), iou * 100.0


def _raw_class_scores_and_winners(
    query_scores: torch.Tensor,
    query_groups: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    class_scores = []
    winner_query_ids = []
    for query_ids in query_groups:
        local = query_scores[:, list(query_ids)].float()
        class_score, local_winner = local.max(dim=1)
        global_ids = torch.as_tensor(
            tuple(int(value) for value in query_ids),
            device=query_scores.device,
            dtype=torch.long,
        )
        class_scores.append(class_score)
        winner_query_ids.append(global_ids[local_winner])
    return (
        torch.stack(class_scores, dim=1),
        torch.stack(winner_query_ids, dim=1),
    )


def _class_scores_without_query(
    *,
    query_scores: torch.Tensor,
    query_groups: Sequence[Sequence[int]],
    query_id: int,
    class_id: int,
) -> torch.Tensor:
    output, _ = _raw_class_scores_and_winners(query_scores, query_groups)
    remaining = [
        int(value)
        for value in query_groups[int(class_id)]
        if int(value) != int(query_id)
    ]
    if remaining:
        output[:, int(class_id)] = query_scores[:, remaining].float().amax(
            dim=1
        )
    return output


def _class_scores_for_prompt_subset(
    *,
    query_scores: torch.Tensor,
    query_groups: Sequence[Sequence[int]],
    enabled_query_ids: set[int],
) -> torch.Tensor:
    class_scores = []
    for query_ids in query_groups:
        enabled = [
            int(query_id)
            for query_id in query_ids
            if int(query_id) in enabled_query_ids
        ]
        if not enabled:
            enabled = [int(query_ids[0])]
        class_scores.append(
            query_scores[:, enabled].float().amax(dim=1)
        )
    return torch.stack(class_scores, dim=1)


def _snapshot_prediction(
    class_scores: torch.Tensor,
    *,
    gt: torch.Tensor,
    prob_thd: float,
    bg_idx: int,
) -> torch.Tensor:
    return predict_from_class_scores(
        class_scores,
        prob_thd=prob_thd,
        bg_idx=bg_idx,
        out_size=tuple(gt.shape[-2:]),
    )


def _head_name_margin(margin: float) -> str:
    if float(margin) == 0.0:
        return "m0"
    return "m" + f"{float(margin):.2f}".replace(".", "")


def _head_name_quantile(quantile: float) -> str:
    return f"q{int(round(float(quantile) * 100.0))}"


def _head_change_counts(
    *,
    canonical_pred: torch.Tensor,
    candidate_pred: torch.Tensor,
    gt: torch.Tensor,
    ignore_index: int,
) -> dict[str, int]:
    valid = gt.to(candidate_pred.device) != int(ignore_index)
    canonical_correct = canonical_pred == gt.to(canonical_pred.device)
    candidate_correct = candidate_pred == gt.to(candidate_pred.device)
    return {
        "rescue": int(
            (valid & ~canonical_correct & candidate_correct).sum().item()
        ),
        "harm": int(
            (valid & canonical_correct & ~candidate_correct).sum().item()
        ),
        "lateral": int(
            (
                valid
                & ~canonical_correct
                & ~candidate_correct
                & (canonical_pred != candidate_pred)
            )
            .sum()
            .item()
        ),
    }


def evaluate_prompt_routing_snapshot(
    *,
    query_scores: torch.Tensor,
    reliability_query_scores: torch.Tensor,
    query_presence: torch.Tensor,
    gt: torch.Tensor,
    query_idx_list: Sequence[int],
    canonical_query_ids: Sequence[int],
    anchors: FrozenAnchorTiers,
    prob_thd: float,
    tau_pos: float,
    rho: float,
    kmax: int,
    n_min: int,
    bg_idx: int,
    config: PromptTransferConfig,
    frozen_alias_weights: torch.Tensor | None = None,
    reference_query_scores: torch.Tensor | None = None,
    ignore_index: int = 255,
) -> PromptRoutingSnapshotResult:
    if tuple(query_scores.shape) != tuple(reliability_query_scores.shape):
        raise ValueError("reliability/inference query score shape mismatch")
    if tuple(query_scores.shape[-2:]) != tuple(gt.shape[-2:]):
        raise ValueError("query score/ground-truth spatial shape mismatch")
    query_groups = group_query_ids_by_class(
        query_idx_list=query_idx_list,
        canonical_query_ids=canonical_query_ids,
    )
    num_classes = len(query_groups)
    canonical_ids = tuple(int(value) for value in canonical_query_ids)
    canonical_index = torch.as_tensor(
        canonical_ids,
        device=query_scores.device,
        dtype=torch.long,
    )
    canonical_scores = query_scores.float().index_select(1, canonical_index)
    raw_scores, raw_winner_queries = _raw_class_scores_and_winners(
        query_scores,
        query_groups,
    )

    current_reliability = estimate_alias_reliability(
        query_scores=reliability_query_scores,
        query_presence=query_presence,
        query_idx_list=tuple(int(value) for value in query_idx_list),
        canonical_query_ids=canonical_ids,
        num_classes=num_classes,
        prob_thd=prob_thd,
        tau_pos=tau_pos,
        rho=rho,
        kmax=kmax,
        n_min=n_min,
        bg_idx=bg_idx,
    ).query_weights
    frozen_weights = (
        current_reliability
        if frozen_alias_weights is None
        else frozen_alias_weights
    )
    reference_scores = (
        query_scores
        if reference_query_scores is None
        else reference_query_scores
    )
    head_scores: dict[str, torch.Tensor] = {
        "H0_canonical": canonical_scores,
        "H1_raw_max": raw_scores,
        "H2_topk_gate": (
            class_scores_from_query_scores_with_alias_reliability(
                query_scores=query_scores,
                query_idx_list=tuple(
                    int(value) for value in query_idx_list
                ),
                num_classes=num_classes,
                canonical_query_ids=canonical_ids,
                query_weights=current_reliability,
            )
        ),
        "H3_canonical_residual": (
            class_scores_from_query_scores_with_frozen_alias_residual(
                query_scores=query_scores,
                reference_query_scores=reference_scores,
                query_idx_list=tuple(
                    int(value) for value in query_idx_list
                ),
                num_classes=num_classes,
                canonical_query_ids=canonical_ids,
                query_weights=frozen_weights,
            )
        ),
    }

    pair_stats: dict[
        tuple[int, int, int, float],
        PairTransferStats,
    ] = {}
    for margin in config.margins:
        margin_token = _head_name_margin(margin)
        measured = measure_prompt_transfer(
            query_scores=query_scores,
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_ids,
            anchors=anchors,
            margin=margin,
            epsilon=config.epsilon,
        )
        pair_stats.update(measured.pairs)
        core = route_prompt_scores(
            query_scores=query_scores,
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_ids,
            anchors=anchors,
            margin=margin,
            enabled_tiers=(CORE_TIER,),
            min_active=config.min_active,
            shoulder_quantile=None,
            epsilon=config.epsilon,
            transfer_snapshot=measured,
        )
        head_scores[f"H4_core_cap_{margin_token}"] = core.class_scores
        core_shoulder = route_prompt_scores(
            query_scores=query_scores,
            query_idx_list=query_idx_list,
            canonical_query_ids=canonical_ids,
            anchors=anchors,
            margin=margin,
            enabled_tiers=(CORE_TIER, SHOULDER_TIER),
            min_active=config.min_active,
            shoulder_quantile=None,
            epsilon=config.epsilon,
            transfer_snapshot=measured,
        )
        head_scores[
            f"H5_core_shoulder_cap_{margin_token}"
        ] = core_shoulder.class_scores
        for quantile in config.safety_quantiles:
            quantile_token = _head_name_quantile(quantile)
            routed = route_prompt_scores(
                query_scores=query_scores,
                query_idx_list=query_idx_list,
                canonical_query_ids=canonical_ids,
                anchors=anchors,
                margin=margin,
                enabled_tiers=(CORE_TIER, SHOULDER_TIER),
                min_active=config.min_active,
                shoulder_quantile=quantile,
                epsilon=config.epsilon,
                transfer_snapshot=measured,
            )
            head_scores[
                f"H6_risk_shoulder_{quantile_token}_{margin_token}"
            ] = routed.class_scores

    prompt_standalone_confusions = {}
    prompt_leave_one_out_confusions = {}
    prompt_marginal_effects = {}
    raw_pred = _snapshot_prediction(
        raw_scores,
        gt=gt,
        prob_thd=prob_thd,
        bg_idx=bg_idx,
    )
    raw_confusion = _confusion_matrix(
        raw_pred,
        gt,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    raw_miou, raw_per_class = _confusion_metrics(raw_confusion)
    for query_id, class_id_value in enumerate(query_idx_list):
        class_id = int(class_id_value)
        standalone_scores = canonical_scores.clone()
        standalone_scores[:, class_id] = torch.maximum(
            standalone_scores[:, class_id],
            query_scores[:, query_id].float(),
        )
        standalone_pred = _snapshot_prediction(
            standalone_scores,
            gt=gt,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
        )
        prompt_standalone_confusions[query_id] = _confusion_matrix(
            standalone_pred,
            gt,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        leave_one_out_scores = _class_scores_without_query(
            query_scores=query_scores,
            query_groups=query_groups,
            query_id=query_id,
            class_id=class_id,
        )
        leave_one_out_pred = _snapshot_prediction(
            leave_one_out_scores,
            gt=gt,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
        )
        leave_one_out_confusion = _confusion_matrix(
            leave_one_out_pred,
            gt,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        prompt_leave_one_out_confusions[query_id] = (
            leave_one_out_confusion
        )
        leave_one_out_miou, _ = _confusion_metrics(
            leave_one_out_confusion
        )
        prompt_marginal_effects[query_id] = (
            raw_miou - leave_one_out_miou
        )

    enabled_for_prompt_oracle = {
        query_id
        for query_id, effect in prompt_marginal_effects.items()
        if float(effect) >= 0.0
    }
    head_scores["H7_gt_prompt_oracle"] = _class_scores_for_prompt_subset(
        query_scores=query_scores,
        query_groups=query_groups,
        enabled_query_ids=enabled_for_prompt_oracle,
    )

    pair_foreign_iou_effects = {}
    oracle_query_scores = query_scores.detach().float().clone()
    anchor_classes = anchors.class_ids.to(device=query_scores.device)
    anchor_tiers = anchors.tiers.to(device=query_scores.device)
    for query_id, class_id_value in enumerate(query_idx_list):
        class_id = int(class_id_value)
        leave_one_out_confusion = prompt_leave_one_out_confusions[query_id]
        _, leave_one_out_per_class = _confusion_metrics(
            leave_one_out_confusion
        )
        for foreign_class_id in range(num_classes):
            if foreign_class_id == class_id:
                continue
            pair_effect = float(
                raw_per_class[foreign_class_id].item()
                - leave_one_out_per_class[foreign_class_id].item()
            )
            pair_foreign_iou_effects[
                (int(query_id), int(foreign_class_id))
            ] = pair_effect
            if pair_effect >= 0.0:
                continue
            pair_mask = (
                (anchor_classes == foreign_class_id)
                & (
                    (anchor_tiers == CORE_TIER)
                    | (anchor_tiers == SHOULDER_TIER)
                )
            )
            if not bool(pair_mask.any()):
                continue
            foreign_score = canonical_scores[0, foreign_class_id]
            strict_limit = torch.nextafter(
                foreign_score,
                torch.full_like(foreign_score, -torch.inf),
            )
            oracle_map = oracle_query_scores[0, query_id]
            oracle_map[pair_mask] = torch.minimum(
                oracle_map[pair_mask],
                strict_limit[pair_mask],
            )
    head_scores["H8_gt_pair_oracle"], _ = _raw_class_scores_and_winners(
        oracle_query_scores,
        query_groups,
    )

    head_confusions = {}
    head_image_miou = {}
    head_change_counts = {}
    canonical_pred = _snapshot_prediction(
        canonical_scores,
        gt=gt,
        prob_thd=prob_thd,
        bg_idx=bg_idx,
    )
    for head_name, class_scores in head_scores.items():
        prediction = _snapshot_prediction(
            class_scores,
            gt=gt,
            prob_thd=prob_thd,
            bg_idx=bg_idx,
        )
        confusion = _confusion_matrix(
            prediction,
            gt,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        head_confusions[head_name] = confusion
        head_image_miou[head_name] = _confusion_metrics(confusion)[0]
        head_change_counts[head_name] = _head_change_counts(
            canonical_pred=canonical_pred,
            candidate_pred=prediction,
            gt=gt,
            ignore_index=ignore_index,
        )

    gt_device = gt.to(query_scores.device)
    valid = gt_device != int(ignore_index)
    gt_transfer_counts = {
        "teacher_correct_prompt_wrong": 0,
        "teacher_wrong_prompt_correct": 0,
        "teacher_wrong_prompt_wrong": 0,
    }
    for query_id, class_id_value in enumerate(query_idx_list):
        class_id = int(class_id_value)
        prompt_score = query_scores[0, query_id].float()
        for foreign_class_id in range(num_classes):
            if foreign_class_id == class_id:
                continue
            anchor_mask = valid & (anchor_classes == foreign_class_id)
            takeover = anchor_mask & (
                prompt_score > canonical_scores[0, foreign_class_id]
            )
            gt_transfer_counts["teacher_correct_prompt_wrong"] += int(
                (takeover & (gt_device == foreign_class_id)).sum().item()
            )
            gt_transfer_counts["teacher_wrong_prompt_correct"] += int(
                (takeover & (gt_device == class_id)).sum().item()
            )
            gt_transfer_counts["teacher_wrong_prompt_wrong"] += int(
                (
                    takeover
                    & (gt_device != foreign_class_id)
                    & (gt_device != class_id)
                )
                .sum()
                .item()
            )

    prompt_winner_counts = torch.zeros(
        int(query_scores.shape[1]),
        device=query_scores.device,
        dtype=torch.long,
    )
    natural_class = raw_scores.argmax(dim=1)
    for class_id in range(num_classes):
        class_mask = natural_class == class_id
        winner_ids = raw_winner_queries[:, class_id][class_mask]
        if int(winner_ids.numel()) > 0:
            prompt_winner_counts += torch.bincount(
                winner_ids,
                minlength=int(query_scores.shape[1]),
            )

    return PromptRoutingSnapshotResult(
        head_confusions=head_confusions,
        head_image_miou=head_image_miou,
        head_change_counts=head_change_counts,
        pair_stats=pair_stats,
        prompt_standalone_confusions=prompt_standalone_confusions,
        prompt_leave_one_out_confusions=prompt_leave_one_out_confusions,
        pair_foreign_iou_effects=pair_foreign_iou_effects,
        gt_transfer_counts=gt_transfer_counts,
        prompt_winner_counts=prompt_winner_counts,
    )


_PAIR_TOTAL_FIELDS = (
    "anchor_count",
    "protected_count",
    "active_residual_count",
    "new_takeover_count",
    "new_takeover_severity_sum",
    "deepen_count",
    "deepen_sum",
    "total_erosion_sum",
    "direct_takeover_count",
    "direct_active_count",
    "rho_q10_sum",
    "rho_q25_sum",
    "rho_median_sum",
    "rho_observation_count",
)
_CHANGE_FIELDS = ("rescue", "harm", "lateral")
_GT_TRANSFER_FIELDS = (
    "teacher_correct_prompt_wrong",
    "teacher_wrong_prompt_correct",
    "teacher_wrong_prompt_wrong",
)


def _new_confusion(
    *,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        (int(num_classes), int(num_classes)),
        device=device,
        dtype=torch.long,
    )


class PromptRoutingDiagnosticAccumulator:
    def __init__(
        self,
        *,
        query_words: Sequence[str],
        query_idx_list: Sequence[int],
        canonical_query_ids: Sequence[int],
        prob_thd: float,
        tau_pos: float,
        rho: float,
        kmax: int,
        n_min: int,
        bg_idx: int,
        device: torch.device,
        config: PromptTransferConfig = PromptTransferConfig(),
        ignore_index: int = 255,
    ):
        self.query_words = tuple(str(value) for value in query_words)
        self.query_idx_list = tuple(int(value) for value in query_idx_list)
        self.canonical_query_ids = tuple(
            int(value) for value in canonical_query_ids
        )
        self.query_groups = group_query_ids_by_class(
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
        )
        if len(self.query_words) != len(self.query_idx_list):
            raise ValueError("query word/mapping length mismatch")
        self.num_classes = len(self.canonical_query_ids)
        self.prob_thd = float(prob_thd)
        self.tau_pos = float(tau_pos)
        self.rho = float(rho)
        self.kmax = int(kmax)
        self.n_min = int(n_min)
        self.bg_idx = int(bg_idx)
        self.device = torch.device(device)
        self.config = config
        self.ignore_index = int(ignore_index)

        self._processed = torch.zeros(
            2,
            device=self.device,
            dtype=torch.long,
        )
        self._head_confusions: dict[
            str,
            dict[str, torch.Tensor],
        ] = {"pre": {}, "post": {}}
        self._head_image_sums: dict[
            str,
            dict[str, torch.Tensor],
        ] = {"pre": {}, "post": {}}
        self._head_change_counts: dict[
            str,
            dict[str, torch.Tensor],
        ] = {"pre": {}, "post": {}}
        self._pair_totals: dict[
            str,
            dict[tuple[int, int, int, float], torch.Tensor],
        ] = {"pre": {}, "post": {}}
        self._prompt_standalone_confusions = {
            phase: {
                query_id: _new_confusion(
                    num_classes=self.num_classes,
                    device=self.device,
                )
                for query_id in range(len(self.query_words))
            }
            for phase in ("pre", "post")
        }
        self._prompt_leave_one_out_confusions = {
            phase: {
                query_id: _new_confusion(
                    num_classes=self.num_classes,
                    device=self.device,
                )
                for query_id in range(len(self.query_words))
            }
            for phase in ("pre", "post")
        }
        self._gt_transfer_counts = {
            phase: torch.zeros(
                len(_GT_TRANSFER_FIELDS),
                device=self.device,
                dtype=torch.long,
            )
            for phase in ("pre", "post")
        }
        self._prompt_winner_counts = {
            phase: torch.zeros(
                len(self.query_words),
                device=self.device,
                dtype=torch.long,
            )
            for phase in ("pre", "post")
        }
        self._sample_ids: list[str] = []
        self._image_rows: list[dict] = []
        self._cross_time_causal = CrossTimePromptCausalAccumulator(
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
            num_classes=self.num_classes,
            prob_thd=self.prob_thd,
            bg_idx=self.bg_idx,
            device=self.device,
            ignore_index=self.ignore_index,
            image_row_limit=self.config.pair_row_limit,
        )

    def _pair_tensor(self, stats: PairTransferStats) -> torch.Tensor:
        rho_values = (
            stats.rho_q10,
            stats.rho_q25,
            stats.rho_median,
        )
        has_rho = all(value is not None for value in rho_values)
        return torch.tensor(
            [
                stats.anchor_count,
                stats.protected_count,
                stats.active_residual_count,
                stats.new_takeover_count,
                stats.new_takeover_severity_sum,
                stats.deepen_count,
                stats.deepen_sum,
                stats.total_erosion_sum,
                stats.direct_takeover_count,
                stats.direct_active_count,
                stats.rho_q10 if stats.rho_q10 is not None else 0.0,
                stats.rho_q25 if stats.rho_q25 is not None else 0.0,
                (
                    stats.rho_median
                    if stats.rho_median is not None
                    else 0.0
                ),
                int(has_rho),
            ],
            device=self.device,
            dtype=torch.float64,
        )

    def _accumulate_snapshot(
        self,
        phase: str,
        snapshot: PromptRoutingSnapshotResult,
    ) -> None:
        for head_name, confusion in snapshot.head_confusions.items():
            target = self._head_confusions[phase].setdefault(
                head_name,
                torch.zeros_like(confusion, device=self.device),
            )
            target += confusion.to(device=self.device)
            image_total = self._head_image_sums[phase].setdefault(
                head_name,
                torch.zeros(
                    2,
                    device=self.device,
                    dtype=torch.float64,
                ),
            )
            image_total += torch.tensor(
                [snapshot.head_image_miou[head_name], 1.0],
                device=self.device,
                dtype=torch.float64,
            )
            change_total = self._head_change_counts[phase].setdefault(
                head_name,
                torch.zeros(
                    len(_CHANGE_FIELDS),
                    device=self.device,
                    dtype=torch.long,
                ),
            )
            change_total += torch.tensor(
                [
                    snapshot.head_change_counts[head_name][field]
                    for field in _CHANGE_FIELDS
                ],
                device=self.device,
                dtype=torch.long,
            )

        for pair_key, stats in snapshot.pair_stats.items():
            target = self._pair_totals[phase].setdefault(
                pair_key,
                torch.zeros(
                    len(_PAIR_TOTAL_FIELDS),
                    device=self.device,
                    dtype=torch.float64,
                ),
            )
            target += self._pair_tensor(stats)
        for query_id, confusion in (
            snapshot.prompt_standalone_confusions.items()
        ):
            self._prompt_standalone_confusions[phase][
                query_id
            ] += confusion.to(self.device)
        for query_id, confusion in (
            snapshot.prompt_leave_one_out_confusions.items()
        ):
            self._prompt_leave_one_out_confusions[phase][
                query_id
            ] += confusion.to(self.device)
        self._gt_transfer_counts[phase] += torch.tensor(
            [
                snapshot.gt_transfer_counts[field]
                for field in _GT_TRANSFER_FIELDS
            ],
            device=self.device,
            dtype=torch.long,
        )
        self._prompt_winner_counts[phase] += (
            snapshot.prompt_winner_counts.to(self.device)
        )

    def _primary_h6_name(self) -> str:
        return (
            "H6_risk_shoulder_"
            f"{_head_name_quantile(self.config.safety_quantiles[0])}_"
            f"{_head_name_margin(self.config.margins[0])}"
        )

    def _bound_image_rows(self, rows: Sequence[dict]) -> list[dict]:
        limit = max(int(self.config.pair_row_limit), 0)
        if limit == 0:
            return []
        return sorted(
            rows,
            key=lambda row: abs(float(row["post_routing_correction_miou"])),
            reverse=True,
        )[:limit]

    def _append_bounded_image_row(
        self,
        *,
        sample_id: str,
        adapted: bool,
        pre: PromptRoutingSnapshotResult,
        post: PromptRoutingSnapshotResult,
    ) -> None:
        head_name = self._primary_h6_name()
        row = {
            "sample_id": str(sample_id),
            "adapted": bool(adapted),
            "pre_raw_miou": pre.head_image_miou["H1_raw_max"],
            "pre_routed_miou": pre.head_image_miou[head_name],
            "post_raw_miou": post.head_image_miou["H1_raw_max"],
            "post_routed_miou": post.head_image_miou[head_name],
            "post_routing_correction_miou": (
                post.head_image_miou[head_name]
                - post.head_image_miou["H1_raw_max"]
            ),
            "routed_tta_gain_miou": (
                post.head_image_miou[head_name]
                - pre.head_image_miou[head_name]
            ),
        }
        self._image_rows = self._bound_image_rows(
            [*self._image_rows, row]
        )

    def update_image(
        self,
        sample_id: str,
        *,
        pre_query_scores: torch.Tensor,
        pre_reliability_scores: torch.Tensor,
        pre_query_presence: torch.Tensor,
        post_query_scores: torch.Tensor,
        post_reliability_scores: torch.Tensor,
        post_query_presence: torch.Tensor,
        gt: torch.Tensor,
        adapted: bool,
    ) -> None:
        if tuple(pre_query_scores.shape) != tuple(post_query_scores.shape):
            raise ValueError("pre/post query score shape mismatch")
        if tuple(pre_reliability_scores.shape) != tuple(
            pre_query_scores.shape
        ):
            raise ValueError(
                "pre reliability/inference score shape mismatch"
            )
        if tuple(post_reliability_scores.shape) != tuple(
            post_query_scores.shape
        ):
            raise ValueError(
                "post reliability/inference score shape mismatch"
            )
        self._cross_time_causal.update_image(
            str(sample_id),
            pre_query_scores=pre_query_scores,
            post_query_scores=post_query_scores,
            gt=gt,
            adapted=bool(adapted),
        )
        anchors = build_frozen_anchor_tiers(
            query_scores=pre_query_scores,
            canonical_query_ids=self.canonical_query_ids,
            prob_thd=self.prob_thd,
            anchor_k=self.config.anchor_k,
        )
        frozen_reliability = estimate_alias_reliability(
            query_scores=pre_reliability_scores,
            query_presence=pre_query_presence,
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
            num_classes=self.num_classes,
            prob_thd=self.prob_thd,
            tau_pos=self.tau_pos,
            rho=self.rho,
            kmax=self.kmax,
            n_min=self.n_min,
            bg_idx=self.bg_idx,
        ).query_weights
        pre = evaluate_prompt_routing_snapshot(
            query_scores=pre_query_scores,
            reliability_query_scores=pre_reliability_scores,
            query_presence=pre_query_presence,
            gt=gt,
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
            anchors=anchors,
            prob_thd=self.prob_thd,
            tau_pos=self.tau_pos,
            rho=self.rho,
            kmax=self.kmax,
            n_min=self.n_min,
            bg_idx=self.bg_idx,
            config=self.config,
            frozen_alias_weights=frozen_reliability,
            reference_query_scores=pre_query_scores,
            ignore_index=self.ignore_index,
        )
        post = evaluate_prompt_routing_snapshot(
            query_scores=post_query_scores,
            reliability_query_scores=post_reliability_scores,
            query_presence=post_query_presence,
            gt=gt,
            query_idx_list=self.query_idx_list,
            canonical_query_ids=self.canonical_query_ids,
            anchors=anchors,
            prob_thd=self.prob_thd,
            tau_pos=self.tau_pos,
            rho=self.rho,
            kmax=self.kmax,
            n_min=self.n_min,
            bg_idx=self.bg_idx,
            config=self.config,
            frozen_alias_weights=frozen_reliability,
            reference_query_scores=pre_query_scores,
            ignore_index=self.ignore_index,
        )
        self._accumulate_snapshot("pre", pre)
        self._accumulate_snapshot("post", post)
        self._processed += torch.tensor(
            [1, int(bool(adapted))],
            device=self.device,
            dtype=torch.long,
        )
        self._sample_ids.append(str(sample_id))
        self._append_bounded_image_row(
            sample_id=str(sample_id),
            adapted=bool(adapted),
            pre=pre,
            post=post,
        )

    def _iter_additive_tensors(self):
        for phase in ("pre", "post"):
            yield from self._head_confusions[phase].values()
            yield from self._head_image_sums[phase].values()
            yield from self._head_change_counts[phase].values()
            yield from self._pair_totals[phase].values()
            yield from self._prompt_standalone_confusions[phase].values()
            yield from self._prompt_leave_one_out_confusions[phase].values()
            yield self._gt_transfer_counts[phase]
            yield self._prompt_winner_counts[phase]

    def reduce_distributed(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        metadata = (
            self.query_words,
            self.query_idx_list,
            self.canonical_query_ids,
        )
        gathered_metadata = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_metadata, metadata)
        if any(value != metadata for value in gathered_metadata):
            raise ValueError("prompt routing metadata differs across ranks")
        dist.all_reduce(self._processed)
        for tensor in self._iter_additive_tensors():
            dist.all_reduce(tensor)

        gathered_ids = [None for _ in range(dist.get_world_size())]
        gathered_rows = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_ids, self._sample_ids)
        dist.all_gather_object(gathered_rows, self._image_rows)
        self._sample_ids = [
            sample_id
            for rank_ids in gathered_ids
            for sample_id in (rank_ids or [])
        ]
        self._image_rows = self._bound_image_rows(
            [
                row
                for rank_rows in gathered_rows
                for row in (rank_rows or [])
            ]
        )
        self._cross_time_causal.reduce_distributed()

    def _head_report(self, head_name: str) -> dict:
        pre_miou, pre_per_class = _confusion_metrics(
            self._head_confusions["pre"][head_name]
        )
        post_miou, post_per_class = _confusion_metrics(
            self._head_confusions["post"][head_name]
        )
        raw_pre_miou = _confusion_metrics(
            self._head_confusions["pre"]["H1_raw_max"]
        )[0]
        raw_post_miou = _confusion_metrics(
            self._head_confusions["post"]["H1_raw_max"]
        )[0]
        pre_image = self._head_image_sums["pre"][head_name]
        post_image = self._head_image_sums["post"][head_name]
        pre_changes = self._head_change_counts["pre"][head_name]
        post_changes = self._head_change_counts["post"][head_name]
        return {
            "baseline_miou": pre_miou,
            "tta_miou": post_miou,
            "baseline_per_class_iou": pre_per_class.cpu().tolist(),
            "tta_per_class_iou": post_per_class.cpu().tolist(),
            "static_routing_gain_miou": pre_miou - raw_pre_miou,
            "routed_tta_gain_miou": post_miou - pre_miou,
            "post_routing_correction_miou": post_miou - raw_post_miou,
            "drift_cleanup_miou": (
                (post_miou - raw_post_miou)
                - (pre_miou - raw_pre_miou)
            ),
            "total_result_miou": post_miou - raw_pre_miou,
            "mean_pre_image_miou": float(
                pre_image[0].item() / max(pre_image[1].item(), 1.0)
            ),
            "mean_post_image_miou": float(
                post_image[0].item() / max(post_image[1].item(), 1.0)
            ),
            "pre_change_counts": {
                field: int(pre_changes[index].item())
                for index, field in enumerate(_CHANGE_FIELDS)
            },
            "post_change_counts": {
                field: int(post_changes[index].item())
                for index, field in enumerate(_CHANGE_FIELDS)
            },
            "pre_confusion": self._head_confusions["pre"][
                head_name
            ].cpu().tolist(),
            "post_confusion": self._head_confusions["post"][
                head_name
            ].cpu().tolist(),
        }

    def _phase_pair_result(
        self,
        phase: str,
        pair_key: tuple[int, int, int, float],
    ) -> dict:
        values = self._pair_totals[phase][pair_key]
        fields = {
            field: float(values[index].item())
            for index, field in enumerate(_PAIR_TOTAL_FIELDS)
        }
        anchor_count = fields["anchor_count"]
        rho_count = fields["rho_observation_count"]
        return {
            "anchor_count": int(anchor_count),
            "protected_count": int(fields["protected_count"]),
            "active_residual_count": int(
                fields["active_residual_count"]
            ),
            "new_takeover_count": int(fields["new_takeover_count"]),
            "new_takeover_rate": (
                fields["new_takeover_count"] / anchor_count
                if anchor_count
                else 0.0
            ),
            "new_takeover_severity_sum": fields[
                "new_takeover_severity_sum"
            ],
            "deepen_count": int(fields["deepen_count"]),
            "deepen_sum": fields["deepen_sum"],
            "total_erosion_sum": fields["total_erosion_sum"],
            "risk": (
                fields["total_erosion_sum"] / anchor_count
                if anchor_count
                else 0.0
            ),
            "direct_takeover_count": int(
                fields["direct_takeover_count"]
            ),
            "direct_active_count": int(fields["direct_active_count"]),
            "direct_takeover_rate": (
                fields["direct_takeover_count"]
                / fields["direct_active_count"]
                if fields["direct_active_count"]
                else 0.0
            ),
            "rho_q10": (
                fields["rho_q10_sum"] / rho_count
                if rho_count
                else None
            ),
            "rho_q25": (
                fields["rho_q25_sum"] / rho_count
                if rho_count
                else None
            ),
            "rho_median": (
                fields["rho_median_sum"] / rho_count
                if rho_count
                else None
            ),
        }

    def _prompt_report(self, query_id: int) -> dict:
        output = {
            "query_id": int(query_id),
            "query": self.query_words[query_id],
            "class_id": int(self.query_idx_list[query_id]),
        }
        for phase in ("pre", "post"):
            canonical_miou = _confusion_metrics(
                self._head_confusions[phase]["H0_canonical"]
            )[0]
            raw_miou = _confusion_metrics(
                self._head_confusions[phase]["H1_raw_max"]
            )[0]
            standalone_miou = _confusion_metrics(
                self._prompt_standalone_confusions[phase][query_id]
            )[0]
            leave_one_out_miou = _confusion_metrics(
                self._prompt_leave_one_out_confusions[phase][query_id]
            )[0]
            output[phase] = {
                "standalone_miou": standalone_miou,
                "standalone_utility_miou": (
                    standalone_miou - canonical_miou
                ),
                "leave_one_out_miou": leave_one_out_miou,
                "marginal_utility_miou": (
                    raw_miou - leave_one_out_miou
                ),
                "winner_pixels": int(
                    self._prompt_winner_counts[phase][query_id].item()
                ),
            }
        output["standalone_utility_drift_miou"] = (
            output["post"]["standalone_utility_miou"]
            - output["pre"]["standalone_utility_miou"]
        )
        output["marginal_utility_drift_miou"] = (
            output["post"]["marginal_utility_miou"]
            - output["pre"]["marginal_utility_miou"]
        )
        return output

    def _pair_gt_utility(
        self,
        phase: str,
        *,
        query_id: int,
        foreign_class_id: int,
    ) -> float:
        raw_iou = _confusion_metrics(
            self._head_confusions[phase]["H1_raw_max"]
        )[1]
        leave_one_out_iou = _confusion_metrics(
            self._prompt_leave_one_out_confusions[phase][query_id]
        )[1]
        return float(
            raw_iou[foreign_class_id].item()
            - leave_one_out_iou[foreign_class_id].item()
        )

    def _pair_reports(self) -> list[dict]:
        pair_keys = sorted(self._pair_totals["pre"])
        output = []
        for pair_key in pair_keys:
            query_id, foreign_class_id, tier, margin = pair_key
            pre = self._phase_pair_result("pre", pair_key)
            post = self._phase_pair_result("post", pair_key)
            pre_utility = self._pair_gt_utility(
                "pre",
                query_id=query_id,
                foreign_class_id=foreign_class_id,
            )
            post_utility = self._pair_gt_utility(
                "post",
                query_id=query_id,
                foreign_class_id=foreign_class_id,
            )
            output.append(
                {
                    "query_id": int(query_id),
                    "query": self.query_words[query_id],
                    "class_id": int(self.query_idx_list[query_id]),
                    "foreign_class_id": int(foreign_class_id),
                    "tier": int(tier),
                    "margin": float(margin),
                    "pre": pre,
                    "post": post,
                    "pre_risk": pre["risk"],
                    "post_risk": post["risk"],
                    "risk_drift": post["risk"] - pre["risk"],
                    "pre_gt_pair_utility": pre_utility,
                    "post_gt_pair_utility": post_utility,
                    "gt_pair_utility_drift": (
                        post_utility - pre_utility
                    ),
                    "post_harmful": bool(post_utility < 0.0),
                }
            )
        return output

    def finalize(self) -> dict:
        head_names = sorted(self._head_confusions["pre"])
        sample_count = len(self._sample_ids)
        unique_count = len(set(self._sample_ids))
        cross_time_causal = self._cross_time_causal.finalize(
            expected_f00=self._head_confusions["pre"]["H1_raw_max"],
            expected_f11=self._head_confusions["post"]["H1_raw_max"],
            expected_canonical_pre=self._head_confusions["pre"][
                "H0_canonical"
            ],
            expected_canonical_post=self._head_confusions["post"][
                "H0_canonical"
            ],
        )
        gt_transfer = {}
        for phase in ("pre", "post"):
            gt_transfer[phase] = {
                field: int(self._gt_transfer_counts[phase][index].item())
                for index, field in enumerate(_GT_TRANSFER_FIELDS)
            }
        return {
            "metric_scale": "percent",
            "processed": int(self._processed[0].item()),
            "adapted": int(self._processed[1].item()),
            "unique_samples": int(unique_count),
            "duplicate_sample_ids": int(sample_count - unique_count),
            "config": {
                "anchor_k": int(self.config.anchor_k),
                "margins": [
                    float(value) for value in self.config.margins
                ],
                "safety_quantiles": [
                    float(value)
                    for value in self.config.safety_quantiles
                ],
                "min_active": int(self.config.min_active),
                "epsilon": float(self.config.epsilon),
                "prob_thd": self.prob_thd,
            },
            "query_words": list(self.query_words),
            "query_idx_list": list(self.query_idx_list),
            "canonical_query_ids": list(self.canonical_query_ids),
            "heads": {
                head_name: self._head_report(head_name)
                for head_name in head_names
            },
            "prompts": [
                self._prompt_report(query_id)
                for query_id in range(len(self.query_words))
            ],
            "pairs": self._pair_reports(),
            "gt_transfer_counts": gt_transfer,
            "image_outliers": list(self._image_rows),
            "cross_time_causal": cross_time_causal,
        }
