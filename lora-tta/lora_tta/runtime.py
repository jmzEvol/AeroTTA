from __future__ import annotations

import os
import random
from datetime import timedelta
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def configure_torch_thread_pools() -> None:
    settings = (
        ("LORA_TTA_NUM_THREADS", torch.set_num_threads),
        ("LORA_TTA_NUM_INTEROP_THREADS", torch.set_num_interop_threads),
    )
    for env_name, setter in settings:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{env_name} must be a positive integer") from error
        if value <= 0:
            raise ValueError(f"{env_name} must be a positive integer")
        setter(value)


def set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def setup_distributed(device: str = "cuda") -> DistributedContext:
    configure_torch_thread_pools()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch_device = torch.device(f"cuda:{local_rank}")
    else:
        torch_device = torch.device(device)
    if world_size > 1 and not dist.is_initialized():
        timeout_seconds = int(os.environ.get("LORA_TTA_DIST_TIMEOUT_SECONDS", "43200"))
        dist.init_process_group(
            backend="nccl" if torch_device.type == "cuda" else "gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch_device,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def build_lora_param_groups(
    params_by_layer: dict[int, list[torch.nn.Parameter]],
    *,
    default_lr: float,
    layer_lrs: dict[int, float] | None = None,
) -> list[dict]:
    layer_lrs = dict(layer_lrs or {})
    unknown_layers = sorted(set(layer_lrs) - set(params_by_layer))
    if unknown_layers:
        unknown_text = ", ".join(str(layer) for layer in unknown_layers)
        raise ValueError(f"unknown LoRA layers: {unknown_text}")

    groups_by_lr: dict[float, dict] = {}
    seen_params: set[int] = set()
    for layer_index in sorted(params_by_layer):
        lr = float(layer_lrs.get(layer_index, default_lr))
        if lr <= 0:
            raise ValueError(f"LoRA learning rate must be positive for layer {layer_index}")
        group = groups_by_lr.setdefault(lr, {"params": [], "lr": lr, "layers": []})
        group["layers"].append(layer_index)
        for param in params_by_layer[layer_index]:
            param_id = id(param)
            if param_id in seen_params:
                raise ValueError(f"duplicate LoRA parameter in layer {layer_index}")
            seen_params.add(param_id)
            group["params"].append(param)

    groups = []
    for group in groups_by_lr.values():
        groups.append(
            {
                "params": group["params"],
                "lr": group["lr"],
                "layers": tuple(group["layers"]),
            }
        )
    return groups


def snapshot_lora_params(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore_lora_params(params: list[torch.nn.Parameter], snapshot: list[torch.Tensor]) -> None:
    if len(params) != len(snapshot):
        raise ValueError(f"LoRA snapshot length mismatch: {len(params)} vs {len(snapshot)}")
    with torch.no_grad():
        for param, value in zip(params, snapshot):
            param.copy_(value.to(device=param.device, dtype=param.dtype))


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def clear_lora_weight_cache(encoder) -> None:
    for layer in getattr(encoder, "layers", []):
        cross_attn = getattr(layer, "cross_attn_image", None)
        if hasattr(cross_attn, "_cached_W_eff"):
            cross_attn._cached_W_eff = None
