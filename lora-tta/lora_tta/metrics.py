from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MetricSummary:
    miou: float
    per_class_iou: list[float]
    valid_classes: int
    macc: float
    per_class_accuracy: list[float]
    valid_accuracy_classes: int
    aacc: float


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255, device: str | torch.device = "cpu"):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
            device=device,
        )

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.to(device=self.matrix.device, dtype=torch.long)
        target = target.to(device=self.matrix.device, dtype=torch.long)
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        valid = target != self.ignore_index
        valid = valid & (target >= 0) & (target < self.num_classes)
        if int(valid.sum().item()) == 0:
            return
        indices = target[valid] * self.num_classes + pred[valid].clamp(0, self.num_classes - 1)
        self.matrix += torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

    def summary(self) -> MetricSummary:
        confusion = self.matrix.float()
        tp = torch.diag(confusion)
        gt_area = confusion.sum(dim=1)
        pred_area = confusion.sum(dim=0)
        union = gt_area + pred_area - tp
        valid = union > 0
        iou = torch.zeros_like(tp)
        iou[valid] = tp[valid] / union[valid].clamp_min(1.0)
        miou = iou[valid].mean() if bool(valid.any()) else torch.tensor(0.0, device=iou.device)
        valid_accuracy = gt_area > 0
        accuracy = torch.zeros_like(tp)
        accuracy[valid_accuracy] = (
            tp[valid_accuracy]
            / gt_area[valid_accuracy].clamp_min(1.0)
        )
        macc = (
            accuracy[valid_accuracy].mean()
            if bool(valid_accuracy.any())
            else accuracy.new_tensor(0.0)
        )
        total_gt = gt_area.sum()
        aacc = (
            tp.sum() / total_gt.clamp_min(1.0)
            if bool(total_gt > 0)
            else tp.new_tensor(0.0)
        )
        return MetricSummary(
            miou=float(miou.item()),
            per_class_iou=[float(v) for v in iou.detach().cpu().tolist()],
            valid_classes=int(valid.sum().item()),
            macc=float(macc.item()),
            per_class_accuracy=[
                float(v) for v in accuracy.detach().cpu().tolist()
            ],
            valid_accuracy_classes=int(valid_accuracy.sum().item()),
            aacc=float(aacc.item()),
        )
