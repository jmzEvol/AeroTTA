from __future__ import annotations

import torch

from .config import LossConfig, MiningConfig
from .losses import SelectedClassEntry


def as_chw_class_scores(class_scores: torch.Tensor) -> torch.Tensor:
    if class_scores.ndim == 4 and int(class_scores.shape[0]) == 1:
        class_scores = class_scores[0]
    if class_scores.ndim != 3:
        raise ValueError(f"expected class_scores [C,H,W], got {tuple(class_scores.shape)}")
    return class_scores


def present_classes(
    class_presence: torch.Tensor,
    *,
    bg_idx: int,
    include_bg: bool,
    tau_pos: float,
    exclude_classes: tuple[int, ...] = (),
) -> torch.Tensor:
    present = (class_presence > float(tau_pos)).nonzero(as_tuple=False).flatten()
    bg_idx = int(bg_idx)
    if include_bg and 0 <= bg_idx < int(class_presence.numel()):
        present = torch.unique(torch.cat([present, present.new_tensor([bg_idx])]), sorted=True)
    if not include_bg:
        present = present[present != bg_idx]
    if exclude_classes:
        excluded = set(int(c) for c in exclude_classes)
        kept = [c for c in present if int(c.item()) not in excluded]
        if not kept:
            return present.new_empty((0,), dtype=torch.long)
        present = torch.stack(kept)
    return present


def class_weight_from_scores(
    values: torch.Tensor,
    *,
    mode: str,
    minimum: float,
) -> torch.Tensor:
    if mode == "none":
        weight = values.new_tensor(1.0)
    elif mode in {"mean_margin", "mean_reliability"}:
        weight = values.float().mean()
    else:
        raise ValueError(f"unknown class_weight_mode={mode}")
    return weight.clamp(min=float(minimum), max=1.0)


def competition_score_ratio(
    class_scores: torch.Tensor,
    *,
    class_id: int,
    flat_idx: torch.Tensor,
) -> torch.Tensor:
    """Return one for class winners and a smooth score ratio otherwise."""
    class_scores = as_chw_class_scores(class_scores).float()
    class_id = int(class_id)
    if int(class_scores.shape[0]) <= 1:
        return torch.ones_like(flat_idx, dtype=class_scores.dtype)
    other_ids = [idx for idx in range(int(class_scores.shape[0])) if idx != class_id]
    other_idx = torch.tensor(other_ids, device=class_scores.device, dtype=torch.long)
    class_values = class_scores[class_id].flatten()[flat_idx].float()
    other_values = (
        class_scores.index_select(0, other_idx)
        .amax(dim=0)
        .flatten()[flat_idx]
        .float()
    )
    denominator = torch.maximum(class_values, other_values).clamp_min(1e-6)
    return (class_values / denominator).clamp(0.0, 1.0)


def foreground_complement_score(class_scores: torch.Tensor, bg_idx: int) -> torch.Tensor | None:
    indices = [idx for idx in range(int(class_scores.shape[0])) if idx != int(bg_idx)]
    if not indices:
        return None
    index_tensor = torch.tensor(indices, device=class_scores.device, dtype=torch.long)
    fg_max = class_scores.index_select(0, index_tensor).max(dim=0).values
    return (1.0 - fg_max).clamp(0.0, 1.0)


def build_component_gate_masks(
    *,
    class_scores: torch.Tensor,
    class_presence: torch.Tensor,
    valid: torch.Tensor,
    mining: MiningConfig,
    exclude_classes: tuple[int, ...] = (),
) -> dict[int, torch.Tensor]:
    """Build per-class connected-component gates from class score maps."""
    class_scores = as_chw_class_scores(class_scores).float()
    valid = valid.to(device=class_scores.device, dtype=torch.bool)
    class_presence = class_presence.to(device=class_scores.device)
    present = present_classes(
        class_presence,
        bg_idx=mining.bg_idx,
        include_bg=mining.include_bg,
        tau_pos=mining.tau_pos,
        exclude_classes=exclude_classes,
    )
    if int(present.numel()) == 0:
        return {}

    from sam3.perflib.connected_components import connected_components

    gates: dict[int, torch.Tensor] = {}
    for cls_idx_t in present:
        cls_idx = int(cls_idx_t.item())
        cls_score = class_scores[cls_idx]
        base_mask = valid & (cls_score >= float(mining.component_mask_thd))
        if int(base_mask.sum().item()) < int(mining.component_min_area):
            continue
        labels, _sizes = connected_components(
            base_mask.to(torch.uint8).contiguous().unsqueeze(0).unsqueeze(0)
        )
        labels = labels.view(base_mask.shape).long()
        fg = base_mask & (labels > 0)
        if int(fg.sum().item()) == 0:
            continue
        label_values = labels[fg]
        max_label = int(label_values.max().item())
        if max_label <= 0:
            continue
        areas = torch.bincount(label_values, minlength=max_label + 1)
        score_sums = torch.bincount(
            label_values,
            weights=cls_score[fg].float(),
            minlength=max_label + 1,
        )
        mean_scores = score_sums / areas.clamp_min(1).float()
        keep = (areas >= int(mining.component_min_area)) & (
            mean_scores >= float(mining.component_mean_thd)
        )
        keep[0] = False
        gate = fg & keep[labels]
        if int(gate.sum().item()) > 0:
            gates[cls_idx] = gate
    return gates


def _positive_candidate_mask(
    score: torch.Tensor,
    *,
    threshold: float,
    band_min: float | None,
    band_max: float | None,
) -> torch.Tensor:
    lower = float(threshold)
    if band_min is not None:
        lower = max(lower, float(band_min))
    mask = score >= lower
    if band_max is not None:
        mask = mask & (score <= float(band_max))
    return mask


def _drop_top_fraction(
    flat_idx: torch.Tensor,
    score: torch.Tensor,
    *,
    drop_frac: float,
    n_min: int,
) -> torch.Tensor:
    if drop_frac <= 0.0 or int(flat_idx.numel()) == 0:
        return flat_idx
    drop_count = int(float(drop_frac) * int(flat_idx.numel()))
    drop_count = min(drop_count, int(flat_idx.numel()) - min(int(n_min), int(flat_idx.numel())))
    if drop_count <= 0:
        return flat_idx
    values = score.flatten()[flat_idx].float()
    drop_local = torch.topk(values, k=drop_count, largest=True).indices
    keep = torch.ones(int(flat_idx.numel()), device=flat_idx.device, dtype=torch.bool)
    keep[drop_local] = False
    return flat_idx[keep]


def _global_topk_selection(
    candidate_pool: dict[int, torch.Tensor],
    score_maps: dict[int, torch.Tensor],
    *,
    budget: int,
) -> dict[int, torch.Tensor]:
    """Select unique image pixels globally under an exact matched budget.

    A pixel proposed by multiple classes is assigned to the class with the
    highest score before global ranking. Therefore the returned class entries
    contain exactly ``budget`` distinct pixels whenever the candidate union is
    large enough.
    """
    if not candidate_pool:
        return {}
    class_ids = sorted(candidate_pool)
    first = candidate_pool[class_ids[0]]
    flat_size = int(score_maps[class_ids[0]].numel())
    best_scores = torch.full(
        (flat_size,),
        -torch.inf,
        device=first.device,
        dtype=torch.float32,
    )
    best_classes = torch.full(
        (flat_size,),
        -1,
        device=first.device,
        dtype=torch.long,
    )
    for cls_idx in class_ids:
        flat_idx = candidate_pool[cls_idx]
        if int(flat_idx.numel()) == 0:
            continue
        values = score_maps[cls_idx].flatten()[flat_idx].float()
        better = values > best_scores[flat_idx]
        update_idx = flat_idx[better]
        best_scores[update_idx] = values[better]
        best_classes[update_idx] = int(cls_idx)

    eligible = best_classes >= 0
    eligible_idx = eligible.nonzero(as_tuple=False).flatten()
    resolved_budget = min(max(int(budget), 0), int(eligible_idx.numel()))
    if resolved_budget <= 0:
        return {}
    top = torch.topk(
        best_scores[eligible_idx],
        k=resolved_budget,
        largest=True,
    ).indices
    chosen_flat = eligible_idx[top]
    chosen_classes = best_classes[chosen_flat]
    return {
        cls_idx: chosen_flat[chosen_classes == int(cls_idx)]
        for cls_idx in class_ids
        if bool((chosen_classes == int(cls_idx)).any())
    }


def select_classwise_pixels(
    *,
    class_scores: torch.Tensor,
    class_presence: torch.Tensor,
    raw_pred: torch.Tensor,
    valid: torch.Tensor,
    mining: MiningConfig,
    loss: LossConfig,
    target_scores: torch.Tensor | None = None,
    exclude_classes: tuple[int, ...] = (),
    consistency_mask: torch.Tensor | None = None,
    pixel_weight: torch.Tensor | None = None,
    component_gate_masks: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[int, SelectedClassEntry], torch.Tensor]:
    """Select reliable pixels independently from each class score map."""
    class_scores = as_chw_class_scores(class_scores).float()
    if target_scores is not None:
        target_scores = as_chw_class_scores(target_scores).to(
            device=class_scores.device,
            dtype=class_scores.dtype,
        )
    valid = valid.to(device=class_scores.device, dtype=torch.bool)
    raw_pred = raw_pred.to(device=class_scores.device, dtype=torch.long)
    selectable = valid
    if consistency_mask is not None:
        selectable = selectable & consistency_mask.to(device=class_scores.device, dtype=torch.bool)

    class_presence = class_presence.to(device=class_scores.device)
    present = present_classes(
        class_presence,
        bg_idx=mining.bg_idx,
        include_bg=mining.include_bg,
        tau_pos=mining.tau_pos,
        exclude_classes=exclude_classes,
    )

    pixel_weight_map = None
    if pixel_weight is not None:
        pixel_weight_map = pixel_weight.to(device=class_scores.device, dtype=class_scores.dtype).clamp_min(0.0)

    bg_score = None
    bg_target_score = None
    if mining.bg_positive_source == "foreground_complement":
        bg_score = foreground_complement_score(class_scores, mining.bg_idx)
        if target_scores is not None:
            bg_target_score = foreground_complement_score(target_scores, mining.bg_idx)

    flat_size = int(raw_pred.numel())
    preselected: dict[int, torch.Tensor] = {}
    selection_score_maps: dict[int, torch.Tensor] = {}
    target_score_maps: dict[int, torch.Tensor] = {}
    candidate_pool: dict[int, torch.Tensor] = {}

    for cls_idx_t in present:
        cls_idx = int(cls_idx_t.item())
        cls_score = class_scores[cls_idx]
        cls_target_score = target_scores[cls_idx] if target_scores is not None else cls_score
        candidate_thd = float(mining.prob_thd)

        if cls_idx == int(mining.bg_idx) and mining.bg_positive_source == "foreground_complement":
            if bg_score is None:
                continue
            cls_score = bg_score
            cls_target_score = bg_target_score if bg_target_score is not None else cls_score
            candidate_thd = 1.0 - float(mining.bg_complement_thd or mining.prob_thd)

        cls_selectable = selectable
        use_component_whole = False
        if component_gate_masks is not None:
            gate = component_gate_masks.get(cls_idx)
            if gate is None:
                if mining.component_fallback != "topk":
                    continue
            else:
                gated_selectable = cls_selectable & gate.to(device=class_scores.device, dtype=torch.bool)
                gated_candidates = gated_selectable & _positive_candidate_mask(
                    cls_score,
                    threshold=candidate_thd,
                    band_min=mining.score_band_min,
                    band_max=mining.score_band_max,
                )
                if int(gated_candidates.sum().item()) >= int(mining.n_min):
                    cls_selectable = gated_selectable
                    use_component_whole = mining.component_select_mode == "whole"
                elif mining.component_fallback != "topk":
                    continue

        candidates = cls_selectable & _positive_candidate_mask(
            cls_score,
            threshold=candidate_thd,
            band_min=mining.score_band_min,
            band_max=mining.score_band_max,
        )
        flat_idx = candidates.flatten().nonzero(as_tuple=False).flatten()
        flat_idx = _drop_top_fraction(
            flat_idx,
            cls_score,
            drop_frac=mining.score_drop_top_frac,
            n_min=mining.n_min,
        )
        if int(flat_idx.numel()) < int(mining.n_min):
            continue
        candidate_pool[cls_idx] = flat_idx

        if use_component_whole:
            chosen = flat_idx
        else:
            k = max(1, int(float(mining.rho) * int(flat_idx.numel())))
            class_kmax = int(mining.class_kmax_overrides.get(cls_idx, mining.kmax))
            k = min(k, class_kmax, int(flat_idx.numel()))
            rank_values = cls_score.flatten()[flat_idx]
            top = torch.topk(rank_values, k=k, largest=True).indices
            chosen = flat_idx[top]
        preselected[cls_idx] = chosen
        selection_score_maps[cls_idx] = cls_score
        target_score_maps[cls_idx] = cls_target_score

    if mining.drop_overlaps and preselected:
        all_selected = torch.cat(list(preselected.values()), dim=0)
        counts = torch.bincount(all_selected, minlength=flat_size)
        preselected = {
            cls_idx: chosen[counts[chosen] == 1]
            for cls_idx, chosen in preselected.items()
        }

    if mining.selected_point_min_confidence is not None:
        min_confidence = float(mining.selected_point_min_confidence)
        preselected = {
            cls_idx: chosen[
                selection_score_maps[cls_idx].flatten()[chosen] >= min_confidence
            ]
            for cls_idx, chosen in preselected.items()
        }

    if mining.positive_argmax_only:
        raw_pred_flat = raw_pred.flatten()
        preselected = {
            cls_idx: chosen[raw_pred_flat[chosen] == int(cls_idx)]
            for cls_idx, chosen in preselected.items()
        }

    if mining.sampling_mode == "global_topk":
        matched_budget = sum(
            int(chosen.numel())
            for chosen in preselected.values()
            if int(chosen.numel()) >= int(mining.n_min)
        )
        global_candidates: dict[int, torch.Tensor] = {}
        raw_pred_flat = raw_pred.flatten()
        for cls_idx, flat_idx in candidate_pool.items():
            eligible_idx = flat_idx
            if mining.selected_point_min_confidence is not None:
                eligible_idx = eligible_idx[
                    selection_score_maps[cls_idx].flatten()[eligible_idx]
                    >= float(mining.selected_point_min_confidence)
                ]
            if mining.positive_argmax_only:
                eligible_idx = eligible_idx[
                    raw_pred_flat[eligible_idx] == int(cls_idx)
                ]
            global_candidates[cls_idx] = eligible_idx
        preselected = _global_topk_selection(
            global_candidates,
            selection_score_maps,
            budget=matched_budget,
        )

    positive_exclusions = {
        cls_idx: chosen
        for cls_idx, chosen in preselected.items()
    }

    selected: dict[int, SelectedClassEntry] = {}
    for cls_idx, chosen in preselected.items():
        minimum_selected = (
            1
            if mining.sampling_mode == "global_topk"
            else int(mining.n_min)
        )
        if int(chosen.numel()) < minimum_selected:
            continue
        cls_score = selection_score_maps[cls_idx]
        cls_target_score = target_score_maps[cls_idx]
        cls_values = cls_score.flatten()[chosen].float()
        cls_target_values = cls_target_score.flatten()[chosen].float()
        if mining.pixel_weight_mode == "none":
            base_weights = torch.ones_like(cls_values)
        elif mining.pixel_weight_mode == "score":
            base_weights = cls_values
            if pixel_weight_map is not None:
                base_weights = base_weights * pixel_weight_map.flatten()[chosen].float()
            base_weights = base_weights.clamp_min(1e-6)
            base_weights = base_weights / base_weights.max().clamp_min(1e-6)
        else:
            raise ValueError(
                f"unknown pixel_weight_mode={mining.pixel_weight_mode}"
            )
        class_reliability = None
        if mining.competition_reliability_mode == "score_ratio":
            point_reliability = competition_score_ratio(
                class_scores,
                class_id=cls_idx,
                flat_idx=chosen,
            )
            class_reliability = (
                (base_weights * point_reliability).sum()
                / base_weights.sum().clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            weights = (base_weights * point_reliability).clamp_min(1e-6)
            weights = weights / weights.max().clamp_min(1e-6)
        elif mining.competition_reliability_mode == "none":
            weights = base_weights
        else:
            raise ValueError(
                "unknown competition_reliability_mode="
                f"{mining.competition_reliability_mode}"
            )
        class_weight = class_weight_from_scores(
            cls_values,
            mode=mining.class_weight_mode,
            minimum=mining.class_weight_min,
        )
        positive_targets = None
        if loss.positive_target_mode == "PTST":
            positive_targets = (cls_target_values + float(loss.positive_target_offset)).clamp(
                min=float(loss.positive_target_min),
                max=float(loss.positive_target_max),
            ).detach()
        elif loss.positive_target_mode != "hard":
            raise ValueError(f"unknown positive_target_mode={loss.positive_target_mode}")

        entry = SelectedClassEntry(
            flat_idx=chosen.detach(),
            weights=weights.detach(),
            class_weight=class_weight.detach(),
            class_reliability=(
                class_reliability.detach()
                if class_reliability is not None
                else None
            ),
            positive_targets=positive_targets,
        )
        if loss.low_score_neg_weight > 0.0 and loss.low_score_neg_kmax > 0:
            neg_candidates = selectable & (cls_score < float(loss.low_score_neg_thd))
            neg_flat_mask = neg_candidates.flatten().clone()
            neg_flat_mask[positive_exclusions[cls_idx]] = False
            neg_flat_idx = neg_flat_mask.nonzero(as_tuple=False).flatten()
            if int(neg_flat_idx.numel()) >= int(loss.low_score_neg_n_min):
                neg_k = max(1, int(float(loss.low_score_neg_rho) * int(neg_flat_idx.numel())))
                neg_k = min(neg_k, int(loss.low_score_neg_kmax), int(neg_flat_idx.numel()))
                neg_scores = cls_score.flatten()[neg_flat_idx].float()
                neg_rank_values = (float(loss.low_score_neg_thd) - neg_scores).clamp_min(0.0)
                top_neg = torch.topk(neg_rank_values, k=neg_k, largest=True).indices
                neg_chosen = neg_flat_idx[top_neg]
                neg_values = cls_score.flatten()[neg_chosen].float()
                neg_weights = (float(loss.low_score_neg_thd) - neg_values).clamp_min(1e-6)
                neg_weights = neg_weights / neg_weights.max().clamp_min(1e-6)
                entry.negative_flat_idx = neg_chosen.detach()
                entry.negative_weights = neg_weights.detach()
                entry.negative_targets = torch.zeros_like(neg_weights).detach()
        selected[cls_idx] = entry

    return selected, present


def select_independent_query_pixels(
    *,
    query_scores: torch.Tensor,
    target_query_scores: torch.Tensor | None = None,
    presence_logits: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    canonical_query_ids: tuple[int, ...],
    valid: torch.Tensor,
    mining: MiningConfig,
    loss: LossConfig,
) -> tuple[dict[int, SelectedClassEntry], torch.Tensor]:
    """Mine each text query independently against canonical class anchors."""
    if query_scores.ndim != 4 or int(query_scores.shape[0]) != 1:
        raise ValueError(
            f"expected query_scores [1,Q,H,W], got {tuple(query_scores.shape)}"
        )
    if presence_logits.shape != query_scores.shape[:2]:
        raise ValueError("presence/query shape mismatch")
    if target_query_scores is not None and target_query_scores.shape != query_scores.shape:
        raise ValueError("target/query score shape mismatch")
    num_queries = int(query_scores.shape[1])
    if len(query_idx_list) != num_queries:
        raise ValueError("query mapping/query score length mismatch")

    num_classes = len(canonical_query_ids)
    if num_classes <= 0:
        raise ValueError("canonical query ids must not be empty")
    if any(int(class_id) < 0 or int(class_id) >= num_classes for class_id in query_idx_list):
        raise ValueError("query class mapping is outside canonical class range")
    canonical_ids = tuple(int(query_id) for query_id in canonical_query_ids)
    if any(query_id < 0 or query_id >= num_queries for query_id in canonical_ids):
        raise ValueError("canonical query id is outside query range")
    for class_id, query_id in enumerate(canonical_ids):
        if int(query_idx_list[query_id]) != class_id:
            raise ValueError("canonical query ids do not match semantic classes")

    scores = query_scores[0].float()
    target_scores = (
        scores
        if target_query_scores is None
        else target_query_scores[0].to(device=scores.device, dtype=torch.float32)
    )
    query_presence = presence_logits[0].float().sigmoid()
    canonical_index = torch.tensor(
        canonical_ids,
        device=scores.device,
        dtype=torch.long,
    )
    canonical_scores = scores.index_select(0, canonical_index)
    canonical_target_scores = target_scores.index_select(0, canonical_index)
    canonical_presence = query_presence.index_select(0, canonical_index)
    valid = valid.to(device=scores.device, dtype=torch.bool)

    present_mask = query_presence > float(mining.tau_pos)
    bg_query_mask = torch.tensor(
        [int(class_id) == int(mining.bg_idx) for class_id in query_idx_list],
        device=scores.device,
        dtype=torch.bool,
    )
    if mining.include_bg:
        present_mask = present_mask | bg_query_mask
    else:
        present_mask = present_mask & ~bg_query_mask
    present_query_ids = present_mask.nonzero(as_tuple=False).flatten()
    present_query_set = set(
        int(query_id)
        for query_id in present_query_ids.detach().cpu().tolist()
    )

    component_gates = None
    if mining.component_gate:
        component_gates = build_component_gate_masks(
            class_scores=canonical_scores,
            class_presence=canonical_presence,
            valid=valid,
            mining=mining,
        )
    canonical_selected, _ = select_classwise_pixels(
        class_scores=canonical_scores,
        class_presence=canonical_presence,
        raw_pred=canonical_scores.argmax(dim=0).long(),
        valid=valid,
        mining=mining,
        loss=loss,
        target_scores=canonical_target_scores,
        component_gate_masks=component_gates,
    )
    selected_by_query = {
        canonical_ids[class_id]: entry
        for class_id, entry in canonical_selected.items()
        if canonical_ids[class_id] in present_query_set
    }

    canonical_query_set = set(canonical_ids)
    for query_id in range(num_queries):
        if query_id in canonical_query_set or query_id not in present_query_set:
            continue
        class_id = int(query_idx_list[query_id])
        class_scores = canonical_scores.clone()
        class_scores[class_id] = scores[query_id]
        class_target_scores = canonical_target_scores.clone()
        class_target_scores[class_id] = target_scores[query_id]
        class_presence = canonical_presence.clone()
        class_presence[class_id] = query_presence[query_id]
        component_gates = None
        if mining.component_gate:
            component_gates = build_component_gate_masks(
                class_scores=class_scores,
                class_presence=class_presence,
                valid=valid,
                mining=mining,
            )
        alias_selected, _ = select_classwise_pixels(
            class_scores=class_scores,
            class_presence=class_presence,
            raw_pred=class_scores.argmax(dim=0).long(),
            valid=valid,
            mining=mining,
            loss=loss,
            target_scores=class_target_scores,
            component_gate_masks=component_gates,
        )
        if class_id in alias_selected:
            selected_by_query[query_id] = alias_selected[class_id]

    return selected_by_query, present_query_ids
