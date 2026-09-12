#!/usr/bin/env python3
"""
train_lora.py - 批量优化版
用 LoveDA 遥感数据集微调 SAM3 TransformerEncoderFusion 中的 LoRA 参数。

优化特性：
  1. 批量推理（batch_size=4）：backbone 每次处理 B 张图，
     grounding 按 query 分段处理，loss 在 B×Q logits 上计算
  2. 文本特征预缓存：所有 query 的 forward_text 在训练前完成，训练中无需重算
  3. 双 GPU DDP（torchrun --nproc_per_node=2）
  4. 两阶段反传：raw CE/Dice + gate-aware CE/Dice + λ_p * BCE_presence，
     避免 B×Q 张 SAM3 计算图同时驻留显存

使用方法（双 GPU）：
  conda activate sam3
  cd /mnt1/userhome/lishaoyuan/jmz/SAMTTA
  torchrun --nproc_per_node=2 train_lora.py --train-config vdd

使用方法（单 GPU，调试）：
  python train_lora.py --train-config loveda --no-validate
"""

import argparse
import copy
import importlib.util
import os
import sys
import time
import random
import math
from datetime import datetime

import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.registry import init_default_scope
from mmseg.registry import DATASETS
import mmseg.datasets.transforms  # noqa: F401 - register mmseg pipeline transforms
from torchvision.transforms import v2

sys.path.insert(0, os.path.dirname(__file__))
import custom_datasets  # noqa: F401 - register mmseg datasets/transforms
from segearthov3_segmentor import SegEarthOV3Segmentation, get_cls_idx
from sam3.model.lora import LoRAMultiheadAttention
from sam3.model.data_misc import FindStage

PROJECT_ROOT = Path(__file__).resolve().parent


class TeeStream:
    """将 print 同时写到终端和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

# ============================================================
# 超参数配置  
# ============================================================
CFG = dict(
    # 数据
    train_config="lora_train_configs/vdd.py",
    data_root="data/VDD",
    classname_path="configs/cls_vdd.txt",
    reduce_zero_label=False,
    dataset_name="VDD",
    train_dataloader=dict(
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=True, round_up=True),
        dataset=dict(
            type="VDDDataset",
            data_root="data/VDD",
            data_prefix=dict(img_path="train/src", seg_map_path="train/gt"),
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations"),
                dict(type="PackSegInputs"),
            ],
        ),
    ),
    val_dataloader=dict(
        enabled=True,
        batch_size=4,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
        dataset=dict(
            type="VDDDataset",
            data_root="data/VDD",
            data_prefix=dict(img_path="val/src", seg_map_path="val/gt"),
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadAnnotations"),
                dict(type="PackSegInputs"),
            ],
        ),
    ),
    validate=True,
    val_interval=1,
    best_metric="miou",
    best_metric_mode="max",
    val_prediction_mode="gate",

    # 模型
    lora_rank=8,
    lora_alpha=16.0,
    seed=3407,

    # 训练
    num_epochs=20,
    batch_size=4,          # 每 GPU 每步同时处理的图像数（批量推理）
    lr=3e-4,
    warmup_epochs=1.0,     # Linear warmup 时长；0 表示关闭 warmup
    min_lr_ratio=0.4,     # Cosine decay 最低学习率 = lr * min_lr_ratio
    weight_decay=1e-3,
    grad_clip=1.0,
    resolution=1008,

    # raw_seg = lambda_pix * CE_pixel + lambda_ps * CE_present_smooth
    #           + lambda_d * Dice_present_classes_only
    # gate_seg 用 gated log-score 对齐推理 gate；gate Dice 使用 softmax 概率。
    raw_seg_loss_weight=0.85,
    gate_seg_loss_weight=0.15,
    gate_presence_power=1.0,
    
    ce_pixel_weight=0.9,
    ce_present_smooth_weight=0.1,
    present_smooth_alpha=0.5,
    present_smooth_eps=1.0,
    
    dice_loss_weight=0.75,
    dice_smooth=1.0,
    
    presence_loss_weight=0.05,
    presence_pos_weight_max=5.0,
    presence_pos_weight_eps=1.0,

    # 日志与保存
    log_interval=20,
    save_interval=1,
    save_dir="work_dirs/Train_lora/vdd",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for SAM3.")
    parser.add_argument(
        "--train-config",
        default=CFG["train_config"],
        help=(
            "Dataset/config file under lora_train_configs, e.g. vdd.py or loveda.py. "
            "Absolute paths are also accepted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override base random seed for python/numpy/torch and sampler shuffling.",
    )
    parser.add_argument(
        "--validate",
        dest="validate",
        action="store_true",
        help="Enable validation even if the train config disables it.",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Disable validation even if the train config enables it.",
    )
    parser.set_defaults(validate=None)
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="Override validation interval in epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from save_dir/last_opt.pt and the matching lora_epochXXX.pt.",
    )
    parser.add_argument(
        "--resume-opt",
        default=None,
        help="Path to optimizer/state checkpoint. Defaults to save_dir/last_opt.pt when --resume is set.",
    )
    parser.add_argument(
        "--resume-lora",
        default=None,
        help="Path to LoRA weights. Defaults to lora_epochXXX.pt based on the saved epoch.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="Epoch number to start from when loading LoRA without optimizer state.",
    )
    parser.add_argument(
        "--global-step",
        type=int,
        default=None,
        help="Optimizer step used by the LR schedule when loading LoRA without optimizer state.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override CFG['save_dir']; useful for rerunning from an older checkpoint without overwriting old results.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """Keep training stochasticity reproducible without forcing slow deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def warmup_cosine_lr(base_lr, step, total_steps, warmup_steps, min_lr_ratio):
    """Linear warmup followed by cosine decay, evaluated per optimizer update."""
    if total_steps <= 1:
        return base_lr

    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)
    step = min(max(step, 0), total_steps - 1)
    warmup_steps = min(max(warmup_steps, 0), total_steps)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(total_steps - warmup_steps - 1, 1)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def resolve_resume_paths(args, cfg, saved_epoch=None):
    """Resolve paired optimizer and LoRA checkpoints for resume."""
    opt_path = args.resume_opt
    if opt_path is None and args.resume:
        opt_path = os.path.join(cfg["save_dir"], "last_opt.pt")

    lora_path = args.resume_lora
    if lora_path is None and saved_epoch is not None:
        lora_path = os.path.join(cfg["save_dir"], f"lora_epoch{saved_epoch:03d}.pt")

    return opt_path, lora_path


def resolve_project_path(path):
    """Resolve relative config/data paths from the SAMTTA project root."""
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def resolve_train_config_path(path):
    """Allow --train-config vdd, vdd.py, lora_train_configs/vdd.py, or an absolute path."""
    if path is None:
        return None

    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([
            PROJECT_ROOT / raw,
            PROJECT_ROOT / "lora_train_configs" / raw,
        ])
        if raw.suffix != ".py":
            candidates.extend([
                PROJECT_ROOT / f"{raw}.py",
                PROJECT_ROOT / "lora_train_configs" / f"{raw}.py",
            ])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"train config 不存在: {path}. 已检查: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def deep_update(base, updates):
    """Recursively merge train config values into the default CFG."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_train_config(path):
    config_path = resolve_train_config_path(path)
    spec = importlib.util.spec_from_file_location("lora_train_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise ValueError(f"{config_path} 必须定义 CONFIG = dict(...)")
    if not isinstance(module.CONFIG, dict):
        raise TypeError(f"{config_path} 中的 CONFIG 必须是 dict")

    return copy.deepcopy(module.CONFIG), config_path


def resolve_runtime_paths(cfg):
    cfg["data_root"] = resolve_project_path(cfg["data_root"])
    cfg["classname_path"] = resolve_project_path(cfg["classname_path"])
    cfg["save_dir"] = resolve_project_path(cfg["save_dir"])
    for dataloader_key in ("train_dataloader", "val_dataloader"):
        dataloader_cfg = cfg.get(dataloader_key)
        if not dataloader_cfg:
            continue
        dataset_cfg = dataloader_cfg.get("dataset")
        if dataset_cfg and dataset_cfg.get("data_root") is not None:
            dataset_cfg["data_root"] = resolve_project_path(dataset_cfg["data_root"])
    return cfg


def split_enabled(split_cfg):
    return bool(split_cfg and split_cfg.get("enabled", True))


def should_validate(cfg):
    return bool(cfg.get("validate", False) and split_enabled(cfg.get("val_dataloader")))


def metric_is_better(value, best_value, mode):
    if best_value is None:
        return True
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"不支持的 best_metric_mode: {mode}")


# ============================================================
# 分布式初始化
# ============================================================
def setup_distributed():
    """
    初始化 NCCL 通信组（torchrun 自动设置 RANK/LOCAL_RANK/WORLD_SIZE）。
    单 GPU 运行时 world_size=1，跳过 dist.init_process_group。
    """
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    # 将当前进程绑定到对应 GPU
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# 数据集
# ============================================================
def build_mmseg_dataset(dataloader_cfg, split_name):
    init_default_scope("mmseg")
    if not dataloader_cfg:
        raise ValueError(f"{split_name}_dataloader 未配置")
    dataset_cfg = copy.deepcopy(dataloader_cfg.get("dataset"))
    if not dataset_cfg:
        raise ValueError(f"{split_name}_dataloader 必须包含 dataset=dict(...)")
    return DATASETS.build(dataset_cfg)


def build_mmseg_dataloader(dataset, dataloader_cfg, seed, drop_last_default):
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    dataloader_cfg.pop("dataset", None)
    dataloader_cfg.pop("enabled", None)

    batch_size = int(dataloader_cfg.pop("batch_size", 1))
    num_workers = int(dataloader_cfg.pop("num_workers", 4))
    persistent_workers = bool(dataloader_cfg.pop("persistent_workers", False))
    pin_memory = bool(dataloader_cfg.pop("pin_memory", True))
    drop_last = bool(dataloader_cfg.pop("drop_last", drop_last_default))
    sampler_cfg = dataloader_cfg.pop("sampler", dict(type="DefaultSampler"))
    sampler_type = sampler_cfg.pop("type", "DefaultSampler")
    if sampler_type != "DefaultSampler":
        raise ValueError(f"LoRA 训练当前只支持 DefaultSampler，收到: {sampler_type}")
    sampler = DefaultSampler(dataset=dataset, seed=seed, **sampler_cfg)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=pseudo_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
        **dataloader_cfg,
    )


def get_dataset_data_root(dataset):
    return getattr(dataset, "data_root", None)


def get_dataset_data_prefix(dataset):
    return getattr(dataset, "data_prefix", None)


def load_annotation(ann_path: str, reduce_zero_label: bool = True) -> torch.Tensor:
    ann = np.array(Image.open(ann_path)).astype(np.int64)
    if reduce_zero_label:
        ann[ann == 0] = 255
        ann[ann != 255] -= 1
    return torch.from_numpy(ann)


def get_dataset_annotation_path(dataset, idx):
    if hasattr(dataset, "get_data_info"):
        data_info = dataset.get_data_info(idx)
        ann_path = data_info.get("seg_map_path") or data_info.get("ann_path")
        if ann_path is not None:
            return ann_path

    item = dataset[idx]
    data_sample = item.get("data_samples") if isinstance(item, dict) else None
    if data_sample is not None:
        ann_path = data_sample.metainfo.get("seg_map_path")
        if ann_path is not None:
            return ann_path

    raise KeyError(f"无法从 dataset[{idx}] 获取 seg_map_path")


def compute_query_presence_pos_weight(
    dataset,
    query_idx_list,
    num_classes,
    reduce_zero_label=True,
    max_weight=5.0,
    eps=1.0,
):
    """
    统计每个类别在多少张训练图中出现，得到 query-level BCE pos_weight。

    使用 sqrt((neg + eps) / (pos + eps)) 并截断，避免稀有类别权重过大。
    """
    pos_counts = torch.zeros(num_classes, dtype=torch.float32)
    valid_images = 0

    for idx in range(len(dataset)):
        ann_path = get_dataset_annotation_path(dataset, idx)
        gt = load_annotation(ann_path, reduce_zero_label=reduce_zero_label)
        valid = gt != 255
        if valid.sum() == 0:
            continue

        present_classes = torch.unique(gt[valid])
        present_classes = present_classes[
            (present_classes >= 0) & (present_classes < num_classes)
        ].long()
        if present_classes.numel() > 0:
            pos_counts[present_classes] += 1
        valid_images += 1

    neg_counts = max(valid_images, 1) - pos_counts
    class_pos_weight = torch.sqrt((neg_counts + eps) / (pos_counts + eps))
    class_pos_weight = class_pos_weight.clamp(min=1.0, max=max_weight)

    query_class = torch.tensor(query_idx_list, dtype=torch.long)
    query_pos_weight = class_pos_weight[query_class]
    return query_pos_weight, pos_counts, valid_images


def build_query_presence_targets(gt_labels, query_idx_list, num_classes, device):
    """构造 query-level presence target: 某 query 对应类别在图中出现则为 1。"""
    query_class = torch.tensor(query_idx_list, device=device, dtype=torch.long)
    targets = torch.zeros(
        (len(gt_labels), len(query_idx_list)), device=device, dtype=torch.float32
    )

    for b, gt in enumerate(gt_labels):
        valid = gt != 255
        if valid.sum() == 0:
            continue

        class_present = torch.zeros(num_classes, device=device, dtype=torch.float32)
        present_classes = torch.unique(gt[valid])
        present_classes = present_classes[
            (present_classes >= 0) & (present_classes < num_classes)
        ].long()
        if present_classes.numel() > 0:
            class_present[present_classes] = 1.0
            targets[b] = class_present[query_class]

    return targets


# ============================================================
# 损失函数：CE 对齐最终 argmax，Dice 只优化真实出现类别的区域形状
# ============================================================
def aggregate_query_logits(query_logits, query_idx_list, num_classes):
    """
    将 query logits 聚合成 class logits。

    cls_loveda_train.txt 是 1 query = 1 class，可直接使用。
    cls_loveda.txt 可能有同义词 query，例如 building/house，同一 class 取 max，
    与 segearthov3_segmentor.py 推理阶段的多 query 聚合语义保持一致。
    """
    if (
        query_logits.shape[1] == num_classes
        and list(query_idx_list) == list(range(num_classes))
    ):
        return query_logits

    class_logits = []
    for cls_idx in range(num_classes):
        query_indices = [
            query_idx for query_idx, mapped_cls in enumerate(query_idx_list)
            if mapped_cls == cls_idx
        ]
        if len(query_indices) == 0:
            raise ValueError(f"class {cls_idx} 没有对应 query，无法计算 CE loss")
        class_logits.append(query_logits[:, query_indices].amax(dim=1))
    return torch.stack(class_logits, dim=1)


def clear_lora_weight_cache(encoder):
    """清除 eval/no_grad 路径缓存的 LoRA 有效权重，避免 optimizer.step 后读到旧 W_eff。"""
    for layer in encoder.layers:
        cross_attn = getattr(layer, "cross_attn_image", None)
        if isinstance(cross_attn, LoRAMultiheadAttention):
            cross_attn._cached_W_eff = None


def forward_query_outputs(
    sam3_model,
    backbone_out_base,
    text_features,
    B_actual,
    device,
    query_id,
):
    """
    对 B 张图像和单个 query 前向，返回:
      semantic logits: (B, H, W)
      presence logits: (B,)
    外层决定是否包 no_grad；这里保持可反传。
    """
    backbone_out = {**backbone_out_base, **text_features}
    find_stage = FindStage(
        img_ids=torch.arange(B_actual, device=device, dtype=torch.long),
        text_ids=torch.full((B_actual,), query_id, device=device, dtype=torch.long),
        input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
        input_points=None, input_points_mask=None,
    )
    geometric_prompt = sam3_model._get_dummy_prompt(num_prompts=B_actual)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = sam3_model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
    semantic_logits = out["semantic_seg"][:, 0]
    presence_logits = out.get("presence_logit_dec", None)
    if presence_logits is None:
        presence_logits = semantic_logits.new_zeros((B_actual,))
    else:
        presence_logits = presence_logits.float().reshape(B_actual, -1)[:, 0]
    return semantic_logits, presence_logits


def compute_multiclass_present_dice_loss_from_logits(
    query_logits,
    query_idx_list,
    gt_labels,
    num_classes,
    presence_logits=None,
    presence_targets=None,
    presence_pos_weight=None,
    ce_pixel_weight=1.0,
    ce_present_smooth_weight=0.0,
    present_smooth_alpha=0.5,
    present_smooth_eps=1.0,
    dice_loss_weight=1.0,
    dice_smooth=1e-5,
    raw_seg_loss_weight=1.0,
    gate_seg_loss_weight=0.0,
    gate_presence_power=1.0,
    presence_loss_weight=0.0,
    first_call=False,
):
    """
    在 query logits 上计算：
      lambda_raw * raw_seg
      + lambda_gate * gate_seg
      + lambda_p * BCE_presence

    raw_seg 使用原始 semantic logits 计算 CE 与 present Dice。
    gate_seg 使用 gated log-score 对齐推理 gate，其中 Dice 使用 softmax 概率。
    CE 使用所有有效像素，直接监督最终 argmax 的类别互斥。
    Present-smooth CE 对每张图中出现的类别先分别求平均 CE，
    再按 (n_c + eps)^alpha 平滑加权，减弱大面积类别主导。
    Dice 只在每张图真实出现的 class 上平均，不除以总类别数 C。
    Presence 在 query 级别监督类别是否在当前图中出现。
    """
    if gate_seg_loss_weight > 0 and presence_logits is None:
        raise ValueError("gate_seg_loss_weight > 0 时必须提供 presence_logits")

    B_actual = query_logits.shape[0]

    zero = query_logits.sum() * 0.0
    ce_pixel_sum = zero
    ce_present_smooth_sum = zero
    dice_sum = zero
    gate_ce_pixel_sum = zero
    gate_ce_present_smooth_sum = zero
    gate_dice_sum = zero
    valid_pixels = 0
    valid_images_for_ps = 0
    present_dice_terms = 0
    gate_valid_pixels = 0
    gate_valid_images_for_ps = 0
    gate_present_dice_terms = 0

    for b in range(B_actual):
        H, W = gt_labels[b].shape
        query_logits_up = F.interpolate(
            query_logits[b:b + 1].float(), (H, W), mode="bilinear", align_corners=False
        )
        logits_up = aggregate_query_logits(query_logits_up, query_idx_list, num_classes)
        gate_logits_up = None
        gate_probs_up = None
        if gate_seg_loss_weight > 0:
            presence_log_gate = (
                gate_presence_power
                * F.logsigmoid(presence_logits[b:b + 1].float()).view(1, -1, 1, 1)
            )
            query_gate_logits_up = F.logsigmoid(query_logits_up) + presence_log_gate
            gate_logits_up = aggregate_query_logits(
                query_gate_logits_up, query_idx_list, num_classes
            )
            gate_probs_up = F.softmax(gate_logits_up, dim=1)

        gt = gt_labels[b].long()
        valid = gt != 255
        if valid.sum() == 0:
            continue

        invalid = valid & ((gt < 0) | (gt >= num_classes))
        if invalid.any():
            bad_vals = torch.unique(gt[invalid]).detach().cpu().tolist()
            raise ValueError(f"GT label 超出 [0, {num_classes - 1}] 范围: {bad_vals}")

        ce_map = F.cross_entropy(
            logits_up, gt.unsqueeze(0), ignore_index=255, reduction="none"
        )[0]
        ce_pixel_sum = ce_pixel_sum + ce_map[valid].sum()
        valid_pixels += int(valid.sum().item())

        if gate_seg_loss_weight > 0:
            gate_ce_map = F.cross_entropy(
                gate_logits_up, gt.unsqueeze(0), ignore_index=255, reduction="none"
            )[0]
            gate_ce_pixel_sum = gate_ce_pixel_sum + gate_ce_map[valid].sum()
            gate_valid_pixels += int(valid.sum().item())

        present_classes = torch.unique(gt[valid])
        if ce_present_smooth_weight > 0:
            class_losses = []
            class_weights = []
            for cls_idx in present_classes.tolist():
                cls_mask = valid & (gt == cls_idx)
                n_cls = cls_mask.sum().float()
                class_losses.append(ce_map[cls_mask].mean())
                class_weights.append(
                    (n_cls + present_smooth_eps).pow(present_smooth_alpha)
                )
            class_losses = torch.stack(class_losses)
            class_weights = torch.stack(class_weights)
            ce_present_smooth_sum = ce_present_smooth_sum + (
                (class_losses * class_weights).sum()
                / class_weights.sum().clamp_min(1.0)
            )
            valid_images_for_ps += 1

            if gate_seg_loss_weight > 0:
                gate_class_losses = []
                gate_class_weights = []
                for cls_idx in present_classes.tolist():
                    cls_mask = valid & (gt == cls_idx)
                    n_cls = cls_mask.sum().float()
                    gate_class_losses.append(gate_ce_map[cls_mask].mean())
                    gate_class_weights.append(
                        (n_cls + present_smooth_eps).pow(present_smooth_alpha)
                    )
                gate_class_losses = torch.stack(gate_class_losses)
                gate_class_weights = torch.stack(gate_class_weights)
                gate_ce_present_smooth_sum = gate_ce_present_smooth_sum + (
                    (gate_class_losses * gate_class_weights).sum()
                    / gate_class_weights.sum().clamp_min(1.0)
                )
                gate_valid_images_for_ps += 1

        for cls_idx in present_classes.tolist():
            cls_logit = logits_up[0, cls_idx]
            target = (gt == cls_idx).float()
            prob = torch.sigmoid(cls_logit)
            prob_v = prob[valid]
            target_v = target[valid]
            dice = 1.0 - (
                2.0 * (prob_v * target_v).sum() + dice_smooth
            ) / (prob_v.sum() + target_v.sum() + dice_smooth)
            dice_sum = dice_sum + dice
            present_dice_terms += 1

            if gate_seg_loss_weight > 0:
                gate_prob = gate_probs_up[0, cls_idx]
                gate_prob_v = gate_prob[valid]
                gate_dice = 1.0 - (
                    2.0 * (gate_prob_v * target_v).sum() + dice_smooth
                ) / (gate_prob_v.sum() + target_v.sum() + dice_smooth)
                gate_dice_sum = gate_dice_sum + gate_dice
                gate_present_dice_terms += 1

    ce_pixel_loss = ce_pixel_sum / max(valid_pixels, 1)
    ce_present_smooth_loss = (
        ce_present_smooth_sum / max(valid_images_for_ps, 1)
        if ce_present_smooth_weight > 0
        else zero
    )
    ce_loss = (
        ce_pixel_weight * ce_pixel_loss
        + ce_present_smooth_weight * ce_present_smooth_loss
    )
    dice_loss = dice_sum / max(present_dice_terms, 1)
    raw_seg_loss = ce_loss + dice_loss_weight * dice_loss

    gate_ce_pixel_loss = (
        gate_ce_pixel_sum / max(gate_valid_pixels, 1)
        if gate_seg_loss_weight > 0
        else zero
    )
    gate_ce_present_smooth_loss = (
        gate_ce_present_smooth_sum / max(gate_valid_images_for_ps, 1)
        if gate_seg_loss_weight > 0 and ce_present_smooth_weight > 0
        else zero
    )
    gate_ce_loss = (
        ce_pixel_weight * gate_ce_pixel_loss
        + ce_present_smooth_weight * gate_ce_present_smooth_loss
    )
    gate_dice_loss = (
        gate_dice_sum / max(gate_present_dice_terms, 1)
        if gate_seg_loss_weight > 0
        else zero
    )
    gate_seg_loss = gate_ce_loss + dice_loss_weight * gate_dice_loss

    presence_loss = zero
    if presence_loss_weight > 0:
        if presence_logits is None or presence_targets is None:
            raise ValueError("presence_loss_weight > 0 时必须提供 presence_logits 和 presence_targets")
        presence_pos_weight = (
            presence_pos_weight.to(device=presence_logits.device, dtype=presence_logits.dtype)
            if presence_pos_weight is not None
            else None
        )
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logits.float(),
            presence_targets.float(),
            pos_weight=presence_pos_weight,
            reduction="mean",
        )

    loss = (
        raw_seg_loss_weight * raw_seg_loss
        + gate_seg_loss_weight * gate_seg_loss
        + presence_loss_weight * presence_loss
    )

    if first_call:
        print(f"  [DEBUG] query_logits shape: {query_logits.shape}", flush=True)
        if presence_logits is not None:
            print(f"  [DEBUG] presence_logits shape: {presence_logits.shape}", flush=True)
        print(
            f"  [DEBUG] class_logits channels after aggregation: {num_classes}",
            flush=True,
        )
        print(
            f"  [DEBUG] valid_pixels={valid_pixels}, "
            f"present_dice_terms={present_dice_terms}, "
            f"valid_images_for_ps={valid_images_for_ps}",
            flush=True,
        )
        print(
            f"  [DEBUG] ce_pixel_weight={ce_pixel_weight}, "
            f"ce_present_smooth_weight={ce_present_smooth_weight}, "
            f"present_smooth_alpha={present_smooth_alpha}, "
            f"present_smooth_eps={present_smooth_eps}, "
            f"dice_loss_weight={dice_loss_weight}, "
            f"dice_smooth={dice_smooth}, "
            f"raw_seg_loss_weight={raw_seg_loss_weight}, "
            f"gate_seg_loss_weight={gate_seg_loss_weight}, "
            f"gate_presence_power={gate_presence_power}, "
            f"presence_loss_weight={presence_loss_weight}",
            flush=True,
        )

    stats = {
        "raw": raw_seg_loss.detach(),
        "ce": ce_loss.detach(),
        "ce_pixel": ce_pixel_loss.detach(),
        "ce_present_smooth": ce_present_smooth_loss.detach(),
        "dice": dice_loss.detach(),
        "gate": gate_seg_loss.detach(),
        "gate_ce": gate_ce_loss.detach(),
        "gate_ce_pixel": gate_ce_pixel_loss.detach(),
        "gate_ce_present_smooth": gate_ce_present_smooth_loss.detach(),
        "gate_dice": gate_dice_loss.detach(),
        "presence": presence_loss.detach(),
        "present_dice_terms": present_dice_terms,
        "gate_present_dice_terms": gate_present_dice_terms,
    }
    return loss, stats


def unpack_mmseg_batch(batch):
    if isinstance(batch, dict):
        return batch["inputs"], batch["data_samples"]
    if isinstance(batch, list):
        inputs = [item["inputs"] for item in batch]
        data_samples = [item["data_samples"] for item in batch]
        return inputs, data_samples
    raise TypeError(f"不支持的 batch 类型: {type(batch)}")


def load_batch_tensors(batch, img_transform, device):
    inputs, data_samples = unpack_mmseg_batch(batch)
    img_tensors = []
    gt_labels = []
    for img_input, data_sample in zip(inputs, data_samples):
        img_input = img_input.to(device)
        if img_input.ndim != 3 or img_input.shape[0] < 3:
            raise ValueError(f"期望 mmseg inputs 为 (C,H,W)，实际 shape={img_input.shape}")
        # mmseg LoadImageFromFile 默认是 BGR；SAM3 训练预处理使用 RGB。
        img_rgb = img_input[:3][[2, 1, 0]]
        img_tensors.append(img_transform(img_rgb))

        gt = data_sample.gt_sem_seg.data
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        gt_labels.append(gt.long().to(device))
    return torch.stack(img_tensors, dim=0), gt_labels


def resize_gt_labels(gt_labels, size):
    """Resize segmentation labels with nearest interpolation for training loss."""
    resized = []
    for gt in gt_labels:
        if tuple(gt.shape[-2:]) == tuple(size):
            resized.append(gt)
            continue
        gt_resized = F.interpolate(
            gt[None, None].float(), size=size, mode="nearest"
        )[0, 0].long()
        resized.append(gt_resized)
    return resized


def forward_all_query_outputs_no_grad(
    sam3_model,
    backbone_out_base,
    text_features,
    b_actual,
    device,
    num_queries,
):
    query_logits_list = []
    presence_logits_list = []
    with torch.no_grad():
        for query_id in range(num_queries):
            semantic_logits, presence_logits = forward_query_outputs(
                sam3_model=sam3_model,
                backbone_out_base=backbone_out_base,
                text_features=text_features,
                B_actual=b_actual,
                device=device,
                query_id=query_id,
            )
            query_logits_list.append(semantic_logits.detach())
            presence_logits_list.append(presence_logits.detach())
    query_logits = torch.stack(query_logits_list, dim=1).float()
    presence_logits = torch.stack(presence_logits_list, dim=1).float()
    return query_logits, presence_logits


def update_confusion_matrix(
    confusion,
    query_logits,
    presence_logits,
    query_idx_list,
    gt_labels,
    num_classes,
    prediction_mode,
    gate_presence_power,
):
    for b, gt in enumerate(gt_labels):
        H, W = gt.shape
        query_logits_up = F.interpolate(
            query_logits[b:b + 1].float(), (H, W), mode="bilinear", align_corners=False
        )

        if prediction_mode == "gate":
            query_scores = F.logsigmoid(query_logits_up)
            if presence_logits is not None:
                presence_log_gate = (
                    gate_presence_power
                    * F.logsigmoid(presence_logits[b:b + 1].float()).view(1, -1, 1, 1)
                )
                query_scores = query_scores + presence_log_gate
        elif prediction_mode == "raw":
            query_scores = query_logits_up
        else:
            raise ValueError(f"不支持的 val_prediction_mode: {prediction_mode}")

        class_logits = aggregate_query_logits(query_scores, query_idx_list, num_classes)
        pred = class_logits.argmax(dim=1)[0].long()
        gt = gt.long()
        valid = gt != 255
        invalid = valid & ((gt < 0) | (gt >= num_classes))
        if invalid.any():
            bad_vals = torch.unique(gt[invalid]).detach().cpu().tolist()
            raise ValueError(f"验证集 GT label 超出 [0, {num_classes - 1}] 范围: {bad_vals}")

        valid = valid & (gt >= 0) & (gt < num_classes)
        if valid.sum() == 0:
            continue
        indices = gt[valid] * num_classes + pred[valid]
        confusion += torch.bincount(
            indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)


def summarize_confusion(confusion):
    confusion = confusion.float()
    tp = torch.diag(confusion)
    gt_area = confusion.sum(dim=1)
    pred_area = confusion.sum(dim=0)
    union = gt_area + pred_area - tp
    valid = union > 0
    iou = torch.zeros_like(tp)
    iou[valid] = tp[valid] / union[valid].clamp_min(1.0)
    miou = iou[valid].mean() if valid.any() else torch.tensor(0.0, device=confusion.device)
    return miou.item(), iou.detach().cpu().tolist(), int(valid.sum().item())


@torch.no_grad()
def validate_one_epoch(
    sam3_model,
    encoder,
    val_loader,
    img_transform,
    text_features,
    query_idx_list,
    num_classes,
    num_queries,
    cfg,
    presence_pos_weight,
    device,
    world_size,
):
    clear_lora_weight_cache(encoder)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    image_count = torch.zeros((), device=device, dtype=torch.float64)
    confusion = torch.zeros(
        (num_classes, num_classes), device=device, dtype=torch.long
    )

    for batch in val_loader:
        img_batch, gt_labels = load_batch_tensors(
            batch=batch,
            img_transform=img_transform,
            device=device,
        )
        b_actual = len(gt_labels)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_out_base = sam3_model.backbone.forward_image(img_batch)

        query_logits, presence_logits = forward_all_query_outputs_no_grad(
            sam3_model=sam3_model,
            backbone_out_base=backbone_out_base,
            text_features=text_features,
            b_actual=b_actual,
            device=device,
            num_queries=num_queries,
        )
        presence_targets = build_query_presence_targets(
            gt_labels=gt_labels,
            query_idx_list=query_idx_list,
            num_classes=num_classes,
            device=device,
        )
        loss, _ = compute_multiclass_present_dice_loss_from_logits(
            query_logits=query_logits,
            query_idx_list=query_idx_list,
            gt_labels=gt_labels,
            num_classes=num_classes,
            presence_logits=presence_logits,
            presence_targets=presence_targets,
            presence_pos_weight=presence_pos_weight,
            ce_pixel_weight=cfg["ce_pixel_weight"],
            ce_present_smooth_weight=cfg["ce_present_smooth_weight"],
            present_smooth_alpha=cfg["present_smooth_alpha"],
            present_smooth_eps=cfg["present_smooth_eps"],
            dice_loss_weight=cfg["dice_loss_weight"],
            dice_smooth=cfg["dice_smooth"],
            raw_seg_loss_weight=cfg["raw_seg_loss_weight"],
            gate_seg_loss_weight=cfg["gate_seg_loss_weight"],
            gate_presence_power=cfg["gate_presence_power"],
            presence_loss_weight=cfg["presence_loss_weight"],
            first_call=False,
        )
        loss_sum += loss.detach().double() * b_actual
        image_count += b_actual
        update_confusion_matrix(
            confusion=confusion,
            query_logits=query_logits,
            presence_logits=presence_logits,
            query_idx_list=query_idx_list,
            gt_labels=gt_labels,
            num_classes=num_classes,
            prediction_mode=cfg["val_prediction_mode"],
            gate_presence_power=cfg["gate_presence_power"],
        )

    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(image_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / image_count.clamp_min(1.0)).item()
    miou, per_class_iou, valid_classes = summarize_confusion(confusion)
    return {
        "loss": val_loss,
        "miou": miou,
        "per_class_iou": per_class_iou,
        "valid_classes": valid_classes,
    }


# ============================================================
# 主训练循环
# ============================================================
def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    cfg = copy.deepcopy(CFG)
    loaded_config, config_path = load_train_config(args.train_config)
    deep_update(cfg, loaded_config)
    cfg["train_config"] = args.train_config
    cfg["train_config_path"] = config_path
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.validate is not None:
        cfg["validate"] = args.validate
        if cfg.get("val_dataloader") is not None:
            cfg["val_dataloader"]["enabled"] = args.validate
    if args.val_interval is not None:
        cfg["val_interval"] = args.val_interval
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    cfg = resolve_runtime_paths(cfg)

    rank, local_rank, world_size = setup_distributed()
    DEVICE   = torch.device(f"cuda:{local_rank}")
    is_main  = (rank == 0)   # 只有 rank=0 的进程负责日志和保存

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fp = None
    if is_main:
        os.makedirs(cfg["save_dir"], exist_ok=True)
        log_path = os.path.join(
            cfg["save_dir"], f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(original_stdout, log_fp)
        sys.stderr = TeeStream(original_stderr, log_fp)
        print(f"[日志] 当前训练日志将保存到: {log_path}", flush=True)
        print(f"[DDP] world_size={world_size}, DEVICE={DEVICE}", flush=True)
        print(f"[配置] train_config={cfg['train_config_path']}", flush=True)
        print(
            f"[数据] dataset={cfg.get('dataset_name', 'unknown')} "
            f"data_root={cfg['data_root']}",
            flush=True,
        )
        print(
            f"[验证] enabled={should_validate(cfg)} "
            f"interval={cfg['val_interval']} "
            f"best={cfg['best_metric']}({cfg['best_metric_mode']})",
            flush=True,
        )
    try:
        set_random_seed(cfg["seed"])

        # 图像预处理变换（与 Sam3Processor 内部一致）
        img_transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(cfg["resolution"], cfg["resolution"])),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # ---- 构建模型（每个进程各自在自己的 GPU 上加载一份完整模型）----
        if is_main:
            print("加载 SAM3 模型并注入 LoRA ...", flush=True)
        segmentor = SegEarthOV3Segmentation(
            classname_path=cfg["classname_path"],
            device=DEVICE,
            enable_lora=True,
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
        )
        sam3_model = segmentor.processor.model
        sam3_model.eval()   # eval 模式：dropout=0，不影响梯度流经 LoRA
        B = int(cfg.get("train_dataloader", {}).get("batch_size", cfg["batch_size"]))
        cfg["batch_size"] = B

        # ---- 类别信息 ----
        query_words, query_idx_list = get_cls_idx(cfg["classname_path"])
        num_classes  = max(query_idx_list) + 1
        num_queries  = len(query_words)

        # ---- 优化器（只优化 LoRA 参数）----
        encoder     = segmentor.get_encoder()
        lora_params = encoder.get_lora_parameters()
        optimizer   = torch.optim.AdamW(
            lora_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        resume_requested = (
            args.resume or args.resume_opt is not None or args.resume_lora is not None
        )
        resume_state = None
        resume_epoch = 0
        start_epoch = 1
        resume_opt_path = None
        resume_lora_path = None

        if resume_requested:
            resume_opt_path, _ = resolve_resume_paths(args, cfg)
            if resume_opt_path is not None:
                if not os.path.isfile(resume_opt_path):
                    raise FileNotFoundError(f"resume optimizer checkpoint 不存在: {resume_opt_path}")
                resume_state = torch.load(resume_opt_path, map_location=DEVICE)
                resume_epoch = int(resume_state["epoch"])

            _, resume_lora_path = resolve_resume_paths(args, cfg, resume_epoch or None)
            if resume_lora_path is None:
                raise ValueError("未指定 LoRA 权重。请使用 --resume 或 --resume-lora。")
            if not os.path.isfile(resume_lora_path):
                raise FileNotFoundError(f"resume LoRA checkpoint 不存在: {resume_lora_path}")

            encoder.load_lora(resume_lora_path)
            clear_lora_weight_cache(encoder)

            if resume_state is not None:
                optimizer.load_state_dict(resume_state["optimizer"])
                start_epoch = resume_epoch + 1

            if args.start_epoch is not None:
                start_epoch = args.start_epoch

        if is_main:
            total_trainable = sum(p.numel() for p in lora_params)
            total_frozen    = sum(p.numel() for p in sam3_model.parameters()) - total_trainable
            print(f"  可训练参数（LoRA）: {total_trainable:,}", flush=True)
            print(f"  冻结参数:           {total_frozen:,}", flush=True)
            print(f"  类别数: {num_classes}, query 数: {num_queries}", flush=True)
            print(f"  query → class: {list(zip(query_words, query_idx_list))}", flush=True)
            print(f"  batch_size per GPU: {B}", flush=True)
            print(f"  random seed: {cfg['seed']}", flush=True)
            print(
                "  loss: "
                f"{cfg['raw_seg_loss_weight']} * raw_seg("
                f"{cfg['ce_pixel_weight']} * CE_multiclass_all_pixels + "
                f"{cfg['ce_present_smooth_weight']} * CE_present_smooth + "
                f"{cfg['dice_loss_weight']} * Dice_present_classes_only) + "
                f"{cfg['gate_seg_loss_weight']} * gate_seg("
                f"presence_power={cfg['gate_presence_power']}) + "
                f"{cfg['presence_loss_weight']} * BCE_presence",
                flush=True,
            )
            print(
                f"  present_smooth: alpha={cfg['present_smooth_alpha']}, "
                f"eps={cfg['present_smooth_eps']}; "
                f"dice_smooth={cfg['dice_smooth']}",
                flush=True,
            )
            if resume_requested:
                print(f"  resume LoRA: {resume_lora_path}", flush=True)
                if resume_opt_path is not None:
                    print(f"  resume optimizer: {resume_opt_path}", flush=True)
                    print(
                        f"  resume epoch={resume_epoch}, start_epoch={start_epoch}",
                        flush=True,
                    )

        # ---- 预编码所有文本 query（一次性，训练中直接复用） ----
        if is_main:
            print("预编码文本特征 ...", flush=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            text_features = sam3_model.backbone.forward_text(query_words, device=DEVICE)
        if is_main:
            print(f"  已缓存 {len(query_words)} 个 query 的文本特征", flush=True)

        # ---- 数据集 ----
        dataset = build_mmseg_dataset(cfg["train_dataloader"], "train")
        val_dataset = None
        if should_validate(cfg):
            val_dataset = build_mmseg_dataset(cfg["val_dataloader"], "val")

        if world_size > 1:
            if is_main:
                (
                    presence_pos_weight,
                    presence_pos_counts,
                    presence_valid_images,
                ) = compute_query_presence_pos_weight(
                    dataset=dataset,
                    query_idx_list=query_idx_list,
                    num_classes=num_classes,
                    reduce_zero_label=getattr(
                        dataset, "reduce_zero_label", cfg["reduce_zero_label"]
                    ),
                    max_weight=cfg["presence_pos_weight_max"],
                    eps=cfg["presence_pos_weight_eps"],
                )
                presence_pos_weight = presence_pos_weight.to(DEVICE)
                presence_pos_counts = presence_pos_counts.to(DEVICE)
                presence_valid_images_t = torch.tensor(
                    [presence_valid_images], device=DEVICE, dtype=torch.long
                )
            else:
                presence_pos_weight = torch.empty(
                    len(query_idx_list), device=DEVICE, dtype=torch.float32
                )
                presence_pos_counts = torch.empty(
                    num_classes, device=DEVICE, dtype=torch.float32
                )
                presence_valid_images_t = torch.empty(
                    1, device=DEVICE, dtype=torch.long
                )

            dist.broadcast(presence_pos_weight, src=0)
            dist.broadcast(presence_pos_counts, src=0)
            dist.broadcast(presence_valid_images_t, src=0)
            presence_valid_images = int(presence_valid_images_t.item())
        else:
            presence_pos_weight, presence_pos_counts, presence_valid_images = (
                compute_query_presence_pos_weight(
                    dataset=dataset,
                    query_idx_list=query_idx_list,
                    num_classes=num_classes,
                    reduce_zero_label=getattr(
                        dataset, "reduce_zero_label", cfg["reduce_zero_label"]
                    ),
                    max_weight=cfg["presence_pos_weight_max"],
                    eps=cfg["presence_pos_weight_eps"],
                )
            )
            presence_pos_weight = presence_pos_weight.to(DEVICE)

        if is_main:
            print(
                f"train dataset: type={dataset.__class__.__name__}, "
                f"data_root={get_dataset_data_root(dataset)}, "
                f"data_prefix={get_dataset_data_prefix(dataset)}",
                flush=True,
            )
            if val_dataset is not None:
                print(
                    f"val dataset:   type={val_dataset.__class__.__name__}, "
                    f"data_root={get_dataset_data_root(val_dataset)}, "
                    f"data_prefix={get_dataset_data_prefix(val_dataset)}",
                    flush=True,
                )
            print(
                "presence BCE: "
                f"valid_images={presence_valid_images}, "
                f"class_pos_counts={presence_pos_counts.tolist()}, "
                f"query_pos_weight={presence_pos_weight.detach().cpu().tolist()}",
                flush=True,
            )

        loader = build_mmseg_dataloader(
            dataset,
            dataloader_cfg=cfg["train_dataloader"],
            seed=cfg["seed"],
            drop_last_default=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = build_mmseg_dataloader(
                val_dataset,
                dataloader_cfg=cfg["val_dataloader"],
                seed=cfg["seed"],
                drop_last_default=False,
            )

        steps_per_epoch = len(loader)
        total_steps = max(steps_per_epoch * cfg["num_epochs"], 1)
        warmup_steps = int(round(cfg["warmup_epochs"] * steps_per_epoch))
        if cfg["warmup_epochs"] > 0:
            warmup_steps = max(warmup_steps, 1)
        warmup_steps = min(warmup_steps, total_steps)

        if is_main:
            imgs_per_gpu = len(dataset) // world_size
            print(f"\n训练集大小: {len(dataset)} 张图像 "
                  f"（每 GPU约 {imgs_per_gpu} 张，{steps_per_epoch} steps/epoch）", flush=True)
            if val_dataset is not None:
                print(
                    f"验证集大小: {len(val_dataset)} 张图像 "
                    f"（{len(val_loader)} steps/epoch, drop_last=False）",
                    flush=True,
                )
            print(f"训练配置: epochs={cfg['num_epochs']}, lr={cfg['lr']}, "
                  f"rank={cfg['lora_rank']}, batch_size={B}, world_size={world_size}", flush=True)
            print(
                f"LR schedule: linear warmup {warmup_steps}/{total_steps} steps "
                f"→ cosine decay to {cfg['lr'] * cfg['min_lr_ratio']:.2e}",
                flush=True,
            )
            print("=" * 70, flush=True)

        global_step = 0
        if resume_state is not None:
            global_step = int(
                resume_state.get("global_step", resume_epoch * steps_per_epoch)
            )
        if args.global_step is not None:
            global_step = args.global_step
        elif resume_state is None and start_epoch > 1:
            global_step = (start_epoch - 1) * steps_per_epoch
        first_call  = is_main and not resume_requested   # DEBUG 信息只打印一次
        best_metric_name = cfg["best_metric"]
        best_metric_mode = cfg["best_metric_mode"]
        best_metric_value = None
        best_epoch = 0
        if resume_state is not None:
            best_metric_value = resume_state.get("best_metric_value", None)
            best_epoch = int(resume_state.get("best_epoch", 0))

        if is_main and start_epoch > cfg["num_epochs"]:
            print(
                f"resume checkpoint 已到 epoch {resume_epoch}，"
                f"当前 num_epochs={cfg['num_epochs']}，没有需要继续训练的 epoch。",
                flush=True,
        )

        for epoch in range(start_epoch, cfg["num_epochs"] + 1):
            train_sampler = getattr(loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            epoch_loss  = 0.0
            epoch_steps = 0
            epoch_imgs  = 0   # 本 epoch 实际处理的图像总数
            t0 = time.time()

            for batch in loader:
                # ---- 加载 B 张图像和标注 ----
                img_batch, gt_labels = load_batch_tensors(
                    batch=batch,
                    img_transform=img_transform,
                    device=DEVICE,
                )
                seg_gt_labels = resize_gt_labels(gt_labels, img_batch.shape[-2:])
                B_actual = len(gt_labels)  # 通常等于 B，最后一个 batch 可能更小（已 drop_last）

                # ---- Backbone 特征：B 张图一次性前向（no_grad，无梯度但可参与 LoRA 反传）----
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    backbone_out_base = sam3_model.backbone.forward_image(img_batch)

                current_lr = warmup_cosine_lr(
                    cfg["lr"],
                    global_step,
                    total_steps,
                    warmup_steps,
                    cfg["min_lr_ratio"],
                )
                set_optimizer_lr(optimizer, current_lr)

                # ---- 先在 detached logits 上计算 CE+present Dice+presence 及其 logits 梯度 ----
                clear_lora_weight_cache(encoder)
                query_logits, presence_logits = forward_all_query_outputs_no_grad(
                    sam3_model=sam3_model,
                    backbone_out_base=backbone_out_base,
                    text_features=text_features,
                    b_actual=B_actual,
                    device=DEVICE,
                    num_queries=num_queries,
                )
                query_logits.requires_grad_(True)
                need_presence_grad = (
                    cfg["presence_loss_weight"] > 0
                    or cfg["gate_seg_loss_weight"] > 0
                )
                presence_logits.requires_grad_(need_presence_grad)
                presence_targets = build_query_presence_targets(
                    gt_labels=gt_labels,
                    query_idx_list=query_idx_list,
                    num_classes=num_classes,
                    device=DEVICE,
                )
                loss, loss_stats = compute_multiclass_present_dice_loss_from_logits(
                    query_logits=query_logits,
                    query_idx_list=query_idx_list,
                    gt_labels=seg_gt_labels,
                    num_classes=num_classes,
                    presence_logits=presence_logits,
                    presence_targets=presence_targets,
                    presence_pos_weight=presence_pos_weight,
                    ce_pixel_weight=cfg["ce_pixel_weight"],
                    ce_present_smooth_weight=cfg["ce_present_smooth_weight"],
                    present_smooth_alpha=cfg["present_smooth_alpha"],
                    present_smooth_eps=cfg["present_smooth_eps"],
                    dice_loss_weight=cfg["dice_loss_weight"],
                    dice_smooth=cfg["dice_smooth"],
                    raw_seg_loss_weight=cfg["raw_seg_loss_weight"],
                    gate_seg_loss_weight=cfg["gate_seg_loss_weight"],
                    gate_presence_power=cfg["gate_presence_power"],
                    presence_loss_weight=cfg["presence_loss_weight"],
                    first_call=first_call,
                )
                if need_presence_grad:
                    grad_query_logits, grad_presence_logits = torch.autograd.grad(
                        loss, (query_logits, presence_logits)
                    )
                    grad_presence_logits = grad_presence_logits.detach()
                else:
                    grad_query_logits = torch.autograd.grad(loss, query_logits)[0]
                    grad_presence_logits = None
                grad_query_logits = grad_query_logits.detach()

                # ---- 再逐 query 重算前向，将 semantic/presence 梯度反传到 LoRA，避免 B×Q 图常驻显存 ----
                optimizer.zero_grad()
                for query_id in range(num_queries):
                    semantic_logits, presence_logits = forward_query_outputs(
                        sam3_model=sam3_model,
                        backbone_out_base=backbone_out_base,
                        text_features=text_features,
                        B_actual=B_actual,
                        device=DEVICE,
                        query_id=query_id,
                    )
                    if grad_presence_logits is None:
                        semantic_logits.float().backward(grad_query_logits[:, query_id])
                    else:
                        torch.autograd.backward(
                            tensors=(semantic_logits.float(), presence_logits.float()),
                            grad_tensors=(
                                grad_query_logits[:, query_id],
                                grad_presence_logits[:, query_id],
                            ),
                        )

                first_call = False

                # ---- 多 GPU 梯度同步（只 allreduce LoRA ~48KB 参数）----
                if world_size > 1:
                    for p in lora_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_(lora_params, cfg["grad_clip"])
                optimizer.step()

                loss_val    = loss.item()
                epoch_loss  += loss_val
                epoch_steps += 1
                epoch_imgs  += B_actual
                global_step += 1

                if is_main and global_step % cfg["log_interval"] == 0:
                    elapsed = time.time() - t0
                    print(
                        f"Epoch {epoch:3d} | Step {global_step:6d} | "
                        f"loss={loss_val:.4f} | "
                        f"raw={loss_stats['raw'].item():.4f} | "
                        f"gate={loss_stats['gate'].item():.4f} | "
                        f"ce={loss_stats['ce'].item():.4f} | "
                        f"ce_pix={loss_stats['ce_pixel'].item():.4f} | "
                        f"ce_ps={loss_stats['ce_present_smooth'].item():.4f} | "
                        f"dice={loss_stats['dice'].item():.4f} | "
                        f"pres={loss_stats['presence'].item():.4f} | "
                        f"lr={current_lr:.2e} | "
                        f"epoch_avg={epoch_loss / epoch_steps:.4f} | "
                        f"{elapsed / epoch_imgs:.2f}s/img  (B={B_actual})",
                        flush=True,
                    )

            # ---- Epoch 结束 ----
            avg = epoch_loss / max(epoch_steps, 1)
            elapsed_ep = time.time() - t0
            if is_main:
                print(
                    f"\n[Epoch {epoch:3d}] avg_loss={avg:.4f} | "
                    f"耗时={elapsed_ep / 60:.1f}min",
                    flush=True,
                )

            val_stats = None
            if (
                val_loader is not None
                and cfg["val_interval"] > 0
                and epoch % cfg["val_interval"] == 0
            ):
                val_t0 = time.time()
                val_stats = validate_one_epoch(
                    sam3_model=sam3_model,
                    encoder=encoder,
                    val_loader=val_loader,
                    img_transform=img_transform,
                    text_features=text_features,
                    query_idx_list=query_idx_list,
                    num_classes=num_classes,
                    num_queries=num_queries,
                    cfg=cfg,
                    presence_pos_weight=presence_pos_weight,
                    device=DEVICE,
                    world_size=world_size,
                )
                if is_main:
                    metric_value = val_stats[best_metric_name]
                    improved = metric_is_better(
                        metric_value, best_metric_value, best_metric_mode
                    )
                    print(
                        f"[Val {epoch:3d}] loss={val_stats['loss']:.4f} | "
                        f"mIoU={val_stats['miou']:.4f} | "
                        f"valid_classes={val_stats['valid_classes']} | "
                        f"耗时={(time.time() - val_t0) / 60:.1f}min",
                        flush=True,
                    )
                    if improved:
                        best_metric_value = metric_value
                        best_epoch = epoch
                        best_lora_path = os.path.join(cfg["save_dir"], "best_lora.pt")
                        best_meta_path = os.path.join(cfg["save_dir"], "best_meta.pt")
                        encoder.save_lora(best_lora_path)
                        torch.save(
                            {
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "best_metric": best_metric_name,
                                "best_metric_mode": best_metric_mode,
                                "best_metric_value": best_metric_value,
                                "val_stats": val_stats,
                                "lora_path": best_lora_path,
                                "train_config_path": cfg["train_config_path"],
                                "dataset_name": cfg.get("dataset_name"),
                                "seed": cfg["seed"],
                            },
                            best_meta_path,
                        )
                        print(
                            f"  新 best: {best_metric_name}={best_metric_value:.4f} "
                            f"(epoch {best_epoch})，已保存 {best_lora_path}",
                            flush=True,
                        )

            if is_main:
                print("", flush=True)
                if epoch % cfg["save_interval"] == 0:
                    lora_path = os.path.join(cfg["save_dir"], f"lora_epoch{epoch:03d}.pt")
                    opt_path  = os.path.join(cfg["save_dir"], "last_opt.pt")
                    encoder.save_lora(lora_path)
                    torch.save(
                        {"epoch": epoch, "lora_rank": cfg["lora_rank"],
                         "lora_alpha": cfg["lora_alpha"],
                         "lr": cfg["lr"],
                         "current_lr": optimizer.param_groups[0]["lr"],
                         "warmup_epochs": cfg["warmup_epochs"],
                         "warmup_steps": warmup_steps,
                         "total_steps": total_steps,
                         "min_lr_ratio": cfg["min_lr_ratio"],
                         "global_step": global_step,
                         "raw_seg_loss_weight": cfg["raw_seg_loss_weight"],
                         "gate_seg_loss_weight": cfg["gate_seg_loss_weight"],
                         "gate_presence_power": cfg["gate_presence_power"],
                         "ce_pixel_weight": cfg["ce_pixel_weight"],
                         "ce_present_smooth_weight": cfg["ce_present_smooth_weight"],
                         "present_smooth_alpha": cfg["present_smooth_alpha"],
                         "present_smooth_eps": cfg["present_smooth_eps"],
                         "dice_loss_weight": cfg["dice_loss_weight"],
                         "dice_smooth": cfg["dice_smooth"],
                         "presence_loss_weight": cfg["presence_loss_weight"],
                         "presence_pos_weight": presence_pos_weight.detach().cpu(),
                         "presence_pos_weight_max": cfg["presence_pos_weight_max"],
                         "presence_pos_weight_eps": cfg["presence_pos_weight_eps"],
                         "loss": "raw_seg_gate_seg_presence",
                         "lora_path": lora_path,
                         "train_config_path": cfg["train_config_path"],
                         "dataset_name": cfg.get("dataset_name"),
                         "validate": should_validate(cfg),
                         "val_interval": cfg["val_interval"],
                         "last_val_stats": val_stats,
                         "best_metric": best_metric_name,
                         "best_metric_mode": best_metric_mode,
                         "best_metric_value": best_metric_value,
                         "best_epoch": best_epoch,
                         "seed": cfg["seed"],
                         "optimizer": optimizer.state_dict()},
                        opt_path,
                    )
                    print(f"  已保存 LoRA: {lora_path}", flush=True)
                    print(f"  已更新 optimizer: {opt_path}", flush=True)
    finally:
        cleanup_distributed()
        if is_main:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if log_fp is not None:
                log_fp.close()


if __name__ == "__main__":
    main()
