from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate_query_maps(
    query_maps: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
) -> torch.Tensor:
    """Aggregate query maps into class maps using max over aliases."""
    if (
        int(query_maps.shape[1]) == int(num_classes)
        and list(query_idx_list) == list(range(int(num_classes)))
    ):
        return query_maps

    class_maps = []
    for cls_idx in range(int(num_classes)):
        query_ids = [
            query_idx
            for query_idx, mapped_cls in enumerate(query_idx_list)
            if int(mapped_cls) == cls_idx
        ]
        if not query_ids:
            raise ValueError(f"class {cls_idx} has no query")
        class_maps.append(query_maps[:, query_ids].amax(dim=1))
    return torch.stack(class_maps, dim=1)


def upsample_score_maps(maps: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample [N,H,W] score maps with legacy wrapper semantics."""
    if tuple(maps.shape[-2:]) == tuple(size):
        return maps.float()
    if maps.ndim == 2:
        return F.interpolate(
            maps[None, None].float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    if maps.ndim != 3:
        raise ValueError(f"expected score maps [N,H,W] or [H,W], got {tuple(maps.shape)}")
    return F.interpolate(
        maps[:, None].float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def sample_maps_at_flat_indices(
    maps: torch.Tensor,
    *,
    flat_indices: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Sample `[N,H,W]` maps at flattened target-image coordinates."""
    if maps.ndim != 3:
        raise ValueError(f"expected maps [N,H,W], got {tuple(maps.shape)}")
    flat_indices = flat_indices.to(device=maps.device, dtype=torch.long)
    target_h, target_w = map(int, target_size)
    if int(flat_indices.numel()) == 0:
        return maps[:, :0, 0]
    if tuple(maps.shape[-2:]) == (target_h, target_w):
        return maps.flatten(1).index_select(1, flat_indices)

    y = torch.div(flat_indices, target_w, rounding_mode="floor")
    x = flat_indices.remainder(target_w)
    grid = torch.stack(
        (
            2.0 * (x.float() + 0.5) / float(target_w) - 1.0,
            2.0 * (y.float() + 0.5) / float(target_h) - 1.0,
        ),
        dim=-1,
    ).reshape(1, -1, 1, 2)
    return F.grid_sample(
        maps.unsqueeze(0).float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, :, :, 0]


def filtered_raw_instance_scores_at_indices(
    *,
    mask_logits: torch.Tensor | None,
    det_logits: torch.Tensor | None,
    presence_score: torch.Tensor,
    flat_indices: torch.Tensor,
    target_size: tuple[int, int],
    confidence_threshold: float,
    mask_chunk: int,
) -> tuple[torch.Tensor, int]:
    """Return raw instance scores after the standard presence-aware filter."""
    pixel_count = int(flat_indices.numel())
    like = presence_score.float()
    if (
        mask_logits is None
        or det_logits is None
        or int(mask_logits.numel()) == 0
        or int(det_logits.numel()) == 0
    ):
        return like.new_zeros((pixel_count,)), 0

    masks = mask_logits[0].float()
    detections = det_logits[0].float().sigmoid().reshape(-1)
    presence = presence_score.float().reshape(-1)[0]
    keep = torch.nonzero(
        detections * presence > float(confidence_threshold),
        as_tuple=False,
    ).flatten()
    if int(keep.numel()) == 0:
        return detections.new_zeros((pixel_count,)), 0

    result = detections.new_zeros((pixel_count,))
    chunk = max(int(mask_chunk), 1)
    for start in range(0, int(keep.numel()), chunk):
        indices = keep[start : start + chunk]
        sampled = sample_maps_at_flat_indices(
            masks.index_select(0, indices),
            flat_indices=flat_indices,
            target_size=target_size,
        ).sigmoid()
        weighted = sampled * detections.index_select(0, indices)[:, None]
        result = torch.maximum(result, weighted.amax(dim=0))
    return result, int(keep.numel())


def aggregate_instance_maps(
    *,
    mask_logits: torch.Tensor,
    det_logits: torch.Tensor,
    presence_score: torch.Tensor,
    out_size: tuple[int, int],
    confidence_threshold: float,
    mask_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and twice-presence-gated instance probability maps."""
    device = det_logits.device
    inst_raw = torch.zeros(tuple(out_size), device=device, dtype=torch.float32)
    inst_gated = torch.zeros(tuple(out_size), device=device, dtype=torch.float32)
    if mask_logits is None or int(mask_logits.numel()) == 0:
        return inst_raw, inst_gated

    masks = mask_logits[0].float()
    scores_raw = det_logits[0].float().sigmoid().reshape(-1)
    presence = presence_score.float().reshape(-1)[0]
    scores_detection_gated = scores_raw * presence
    keep = scores_detection_gated > float(confidence_threshold)
    scores_twice_gated = scores_detection_gated * presence
    chunk = max(int(mask_chunk), 1)

    for start in range(0, int(masks.shape[0]), chunk):
        end = min(start + chunk, int(masks.shape[0]))
        mask_prob = upsample_score_maps(masks[start:end], tuple(out_size)).sigmoid()
        raw_chunk = mask_prob * scores_raw[start:end, None, None]
        inst_raw = torch.maximum(inst_raw, raw_chunk.amax(dim=0))

        keep_chunk = keep[start:end]
        if not bool(keep_chunk.any()):
            continue
        gated_scores = scores_twice_gated[start:end][keep_chunk]
        gated_chunk = mask_prob[keep_chunk] * gated_scores[:, None, None]
        inst_gated = torch.maximum(inst_gated, gated_chunk.amax(dim=0))
    return inst_raw, inst_gated


def aggregate_gated_instance_map(
    *,
    mask_logits: torch.Tensor,
    det_logits: torch.Tensor,
    presence_score: torch.Tensor,
    out_size: tuple[int, int],
    confidence_threshold: float,
    mask_chunk: int,
) -> torch.Tensor:
    """Build only the gated instance map and resize only contributing masks."""
    inst_gated = torch.zeros(
        tuple(out_size),
        device=det_logits.device,
        dtype=torch.float32,
    )
    if mask_logits is None or int(mask_logits.numel()) == 0:
        return inst_gated

    masks = mask_logits[0].float()
    scores_raw = det_logits[0].float().sigmoid().reshape(-1)
    presence = presence_score.float().reshape(-1)[0]
    scores_detection_gated = scores_raw * presence
    keep_indices = torch.nonzero(
        scores_detection_gated > float(confidence_threshold),
        as_tuple=False,
    ).flatten()
    scores_twice_gated = scores_detection_gated * presence
    chunk = max(int(mask_chunk), 1)

    for start in range(0, int(keep_indices.numel()), chunk):
        indices = keep_indices[start : start + chunk]
        mask_prob = upsample_score_maps(
            masks.index_select(0, indices),
            tuple(out_size),
        ).sigmoid()
        gated_scores = scores_twice_gated.index_select(0, indices)
        gated_chunk = mask_prob * gated_scores[:, None, None]
        inst_gated = torch.maximum(inst_gated, gated_chunk.amax(dim=0))
    return inst_gated


def aggregate_proc_instance_map(
    *,
    mask_logits: torch.Tensor,
    det_logits: torch.Tensor,
    presence_score: torch.Tensor,
    out_size: tuple[int, int],
    confidence_threshold: float,
    mask_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ProC's best-instance map and image-level max instance score."""
    instance_best = torch.zeros(
        tuple(out_size),
        device=det_logits.device,
        dtype=torch.float32,
    )
    if (
        mask_logits is None
        or int(mask_logits.numel()) == 0
        or int(det_logits.numel()) == 0
    ):
        return instance_best, det_logits.new_zeros((), dtype=torch.float32)

    masks = mask_logits[0].float()
    presence = presence_score.float().reshape(-1)[0]
    object_scores = det_logits[0].float().sigmoid().reshape(-1) * presence
    max_instance_score = object_scores.max()
    keep_indices = torch.nonzero(
        object_scores > float(confidence_threshold),
        as_tuple=False,
    ).flatten()
    chunk = max(int(mask_chunk), 1)

    for start in range(0, int(keep_indices.numel()), chunk):
        indices = keep_indices[start : start + chunk]
        mask_prob = upsample_score_maps(
            masks.index_select(0, indices),
            tuple(out_size),
        ).sigmoid()
        scores = object_scores.index_select(0, indices)
        weighted_masks = mask_prob * scores[:, None, None]
        instance_best = torch.maximum(
            instance_best,
            weighted_masks.amax(dim=0),
        )
    return instance_best, max_instance_score


def fuse_query_head_scores(
    *,
    semantic_logits: torch.Tensor,
    presence_score: torch.Tensor,
    inst_gated: torch.Tensor,
    presence_gate_power: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse SAM3 semantic and instance heads like the legacy wrapper."""
    semantic_prob = semantic_logits.float().sigmoid()
    presence = presence_score.float().reshape(-1)[0]
    semantic_gated = semantic_prob * presence.pow(float(presence_gate_power))
    fused = torch.maximum(semantic_gated, inst_gated.float())
    return semantic_prob, semantic_gated, fused


def fuse_proc_pgrf_query_head_scores(
    *,
    semantic_logits: torch.Tensor,
    presence_score: torch.Tensor,
    instance_best: torch.Tensor,
    max_instance_score: torch.Tensor,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse one prompt with ProC's prompt-guided residual fusion equation."""
    semantic_prob = semantic_logits.float().sigmoid()
    presence = presence_score.float().reshape(-1)[0]
    max_fused = torch.maximum(semantic_prob, instance_best.float())
    agreement = 1.0 - (semantic_prob - max_fused).abs()
    residual_gate = (
        presence
        * max_instance_score.float().reshape(-1)[0]
        * agreement.clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)
    instance_advantage = (max_fused - semantic_prob).clamp(min=0.0)
    pgrf = (
        semantic_prob + residual_gate * instance_advantage
    ).clamp(float(eps), 1.0 - float(eps))
    fused = pgrf * presence
    return semantic_prob, max_fused, fused


def class_scores_from_query_scores(
    *,
    query_scores: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
) -> torch.Tensor:
    """Aggregate already-fused query probability maps into class scores."""
    return aggregate_query_maps(query_scores.float(), query_idx_list, int(num_classes))


def class_scores_from_query_scores_with_canonical_overrides(
    *,
    query_scores: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
    canonical_query_ids: tuple[int, ...] | list[int],
    canonical_class_ids: tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Aggregate aliases, then use canonical maps for specified classes only."""
    num_classes = int(num_classes)
    if len(canonical_query_ids) != num_classes:
        raise ValueError(
            "canonical_query_ids must contain exactly one query id per class"
        )
    class_ids = tuple(int(class_id) for class_id in canonical_class_ids)
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("canonical_class_ids must not contain duplicates")
    if any(class_id < 0 or class_id >= num_classes for class_id in class_ids):
        raise ValueError(
            f"canonical class ids {class_ids} exceed the class range [0, {num_classes - 1}]"
        )

    class_scores = class_scores_from_query_scores(
        query_scores=query_scores,
        query_idx_list=query_idx_list,
        num_classes=num_classes,
    )
    if not class_ids:
        return class_scores

    num_queries = int(query_scores.shape[1])
    output = class_scores.clone()
    for class_id in class_ids:
        query_id = int(canonical_query_ids[class_id])
        if query_id < 0 or query_id >= num_queries:
            raise ValueError(
                f"canonical query id {query_id} for class {class_id} exceeds "
                f"the query range [0, {num_queries - 1}]"
            )
        if int(query_idx_list[query_id]) != class_id:
            raise ValueError(
                f"canonical query id {query_id} does not map to class {class_id}"
            )
        output[:, class_id] = query_scores[:, query_id].float()
    return output


def class_scores_from_query_logits(
    *,
    query_logits: torch.Tensor,
    query_idx_list: list[int] | tuple[int, ...],
    num_classes: int,
    presence_scores: torch.Tensor | None = None,
    presence_gate_power: float = 0.0,
) -> torch.Tensor:
    """Build legacy class-fused score maps from query logits.

    The old wrapper mined from `teacher["class_fused"]` and kept image-level
    presence separate for present-class filtering.  Do not multiply the dense
    score maps by presence here; `presence_scores` and `presence_gate_power` are
    accepted only to keep the call site explicit.
    """
    del presence_scores, presence_gate_power
    query_scores = torch.sigmoid(query_logits.float())
    return aggregate_query_maps(query_scores, query_idx_list, int(num_classes))


def predict_from_class_scores(
    class_scores: torch.Tensor,
    *,
    prob_thd: float,
    bg_idx: int,
    out_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Predict with legacy argmax followed by low-score background threshold."""
    if out_size is not None and tuple(class_scores.shape[-2:]) != tuple(out_size):
        class_scores = F.interpolate(
            class_scores.float(),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
    pred = class_scores.argmax(dim=1).long()
    max_vals = class_scores.max(dim=1).values
    pred = pred.clone()
    pred[max_vals < float(prob_thd)] = int(bg_idx)
    return pred[0] if int(pred.shape[0]) == 1 else pred
