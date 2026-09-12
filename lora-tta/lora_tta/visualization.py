from __future__ import annotations

import colorsys
import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


IGNORE_COLOR = (128, 128, 128)
UNKNOWN_COLOR = (64, 64, 64)
OVERLAY_ALPHA = 0.55


@dataclass(frozen=True)
class ClassMetadata:
    names: tuple[str, ...]
    palette: tuple[tuple[int, int, int], ...]


def rgb_from_mmseg_input(image: torch.Tensor) -> Image.Image:
    tensor = torch.as_tensor(image).detach().cpu()
    if tensor.ndim != 3 or int(tensor.shape[0]) < 3:
        raise ValueError(
            "mmseg image input must have shape (C,H,W) with at least 3 channels"
        )
    bgr = tensor[:3]
    if bgr.is_floating_point():
        bgr = bgr.float()
        if int(bgr.numel()) > 0 and float(bgr.max().item()) <= 1.0:
            bgr = bgr * 255.0
        bgr = bgr.round().clamp(0, 255).to(torch.uint8)
    else:
        bgr = bgr.clamp(0, 255).to(torch.uint8)
    rgb = bgr[[2, 1, 0]].permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(rgb, mode="RGB")


def _fallback_color(class_id: int) -> tuple[int, int, int]:
    if int(class_id) == 0:
        return (0, 0, 0)
    hue = (float(class_id) * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return tuple(int(round(channel * 255.0)) for channel in (red, green, blue))


def _normalize_color(value, class_id: int) -> tuple[int, int, int]:
    if isinstance(value, Sequence) and len(value) >= 3:
        try:
            return tuple(
                max(0, min(255, int(channel)))
                for channel in value[:3]
            )
        except (TypeError, ValueError):
            pass
    return _fallback_color(class_id)


def resolve_class_metadata(
    *,
    classes,
    palette,
    num_classes: int,
) -> ClassMetadata:
    resolved_count = int(num_classes)
    if resolved_count <= 0:
        raise ValueError("num_classes must be positive")
    class_values = tuple(classes or ())
    palette_values = tuple(palette or ())
    names = tuple(
        str(class_values[class_id])
        if class_id < len(class_values) and str(class_values[class_id]).strip()
        else f"class_{class_id}"
        for class_id in range(resolved_count)
    )
    colors = tuple(
        _normalize_color(
            palette_values[class_id]
            if class_id < len(palette_values)
            else None,
            class_id,
        )
        for class_id in range(resolved_count)
    )
    return ClassMetadata(names=names, palette=colors)


def _label_array(labels: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        array = labels.detach().cpu().numpy()
    else:
        array = np.asarray(labels)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("label map must have shape (H,W) or (1,H,W)")
    return array.astype(np.int64, copy=False)


def colorize_label_map(
    labels: torch.Tensor | np.ndarray,
    palette: Sequence[Sequence[int]],
    *,
    ignore_index: int = 255,
) -> Image.Image:
    array = _label_array(labels)
    rgb = np.empty((*array.shape, 3), dtype=np.uint8)
    rgb[...] = UNKNOWN_COLOR
    for class_id, color in enumerate(palette):
        rgb[array == class_id] = _normalize_color(color, class_id)
    rgb[array == int(ignore_index)] = IGNORE_COLOR
    return Image.fromarray(rgb, mode="RGB")


def overlay_label_map(
    source: Image.Image,
    labels: torch.Tensor | np.ndarray,
    palette: Sequence[Sequence[int]],
    *,
    alpha: float = OVERLAY_ALPHA,
    output_size: tuple[int, int] | None = None,
    ignore_index: int = 255,
) -> Image.Image:
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("overlay alpha must be in [0,1]")
    array = _label_array(labels)
    target_size = output_size or (int(array.shape[1]), int(array.shape[0]))
    source_rgb = source.convert("RGB")
    if source_rgb.size != target_size:
        source_rgb = source_rgb.resize(target_size, Image.Resampling.BILINEAR)
    colors = colorize_label_map(
        array,
        palette,
        ignore_index=ignore_index,
    )
    if colors.size != target_size:
        colors = colors.resize(target_size, Image.Resampling.NEAREST)
    blended = Image.blend(source_rgb, colors, float(alpha))
    if np.any(array == int(ignore_index)):
        ignore_mask = Image.fromarray(
            (array == int(ignore_index)).astype(np.uint8) * 255,
            mode="L",
        )
        if ignore_mask.size != target_size:
            ignore_mask = ignore_mask.resize(
                target_size,
                Image.Resampling.NEAREST,
            )
        blended.paste(source_rgb, (0, 0), ignore_mask)
    return blended


def comparison_filename(
    sample_id: str,
    *,
    delta_miou: float | None = None,
) -> str:
    path = Path(str(sample_id))
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    if not stem:
        stem = "sample"
    digest = hashlib.sha1(str(sample_id).encode("utf-8")).hexdigest()[:10]
    prefix = (
        ""
        if delta_miou is None
        else f"{float(delta_miou):+.2f}_mIoU__"
    )
    return f"{prefix}{stem}_{digest}.png"


def panel_titles(
    *,
    adapted: bool,
    baseline_miou: float | None = None,
    tta_miou: float | None = None,
    delta_miou: float | None = None,
) -> tuple[str, str, str, str]:
    before_title = "Baseline Overlay"
    final_title = "Ours Overlay" if adapted else "Ours Overlay (not adapted)"
    if baseline_miou is not None:
        before_title += f"\nmIoU: {float(baseline_miou):.2f}"
    if tta_miou is not None:
        final_title += f"\nmIoU: {float(tta_miou):.2f}"
        if delta_miou is not None:
            final_title += f" | Delta: {float(delta_miou):+.2f}"
    return ("Original", "Ground Truth Overlay", before_title, final_title)


def _display_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if int(max_side) <= 0:
        raise ValueError("max_side must be positive")
    scale = min(1.0, float(max_side) / float(max(width, height)))
    return (
        max(1, int(round(float(width) * scale))),
        max(1, int(round(float(height) * scale))),
    )


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    if "\n" in text:
        left, top, right, bottom = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=2,
            align="center",
        )
    else:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return (
        max(1, int(math.ceil(right - left))),
        max(1, int(math.ceil(bottom - top))),
    )


def _legend_layout(
    draw: ImageDraw.ImageDraw,
    metadata: ClassMetadata,
    *,
    font,
    available_width: int,
) -> tuple[list[tuple[int, int, int]], int]:
    swatch_size = 12
    item_gap = 16
    line_gap = 8
    x = 0
    y = 0
    row_height = max(swatch_size, _text_size(draw, "Ag", font)[1])
    positions = []
    for class_id, name in enumerate(metadata.names):
        text_width, _ = _text_size(draw, name, font)
        item_width = swatch_size + 6 + text_width
        if x > 0 and x + item_width > available_width:
            x = 0
            y += row_height + line_gap
        positions.append((class_id, x, y))
        x += item_width + item_gap
    return positions, y + row_height


def render_comparison(
    *,
    source: Image.Image,
    gt: torch.Tensor | np.ndarray,
    before: torch.Tensor | np.ndarray,
    after: torch.Tensor | np.ndarray,
    metadata: ClassMetadata,
    adapted: bool,
    max_side: int,
    baseline_miou: float | None = None,
    tta_miou: float | None = None,
    delta_miou: float | None = None,
) -> Image.Image:
    gt_array = _label_array(gt)
    before_array = _label_array(before)
    after_array = _label_array(after)
    if before_array.shape != gt_array.shape or after_array.shape != gt_array.shape:
        raise ValueError("GT and prediction label maps must have the same shape")

    target_height, target_width = gt_array.shape
    source_rgb = source.convert("RGB")
    if source_rgb.size != (target_width, target_height):
        source_rgb = source_rgb.resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )
    display_width, display_height = _display_size(
        target_width,
        target_height,
        int(max_side),
    )
    display_size = (display_width, display_height)
    source_panel = source_rgb.resize(
        display_size,
        Image.Resampling.BILINEAR,
    )
    panels = [
        source_panel,
        overlay_label_map(
            source_rgb,
            gt_array,
            metadata.palette,
            output_size=display_size,
        ),
        overlay_label_map(
            source_rgb,
            before_array,
            metadata.palette,
            output_size=display_size,
        ),
        overlay_label_map(
            source_rgb,
            after_array,
            metadata.palette,
            output_size=display_size,
        ),
    ]

    margin = 16
    panel_gap = 12
    title_gap = 8
    legend_gap = 16
    font = ImageFont.load_default()
    scratch = Image.new("RGB", (1, 1), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    titles = panel_titles(
        adapted=adapted,
        baseline_miou=baseline_miou,
        tta_miou=tta_miou,
        delta_miou=delta_miou,
    )
    title_sizes = [_text_size(scratch_draw, title, font) for title in titles]
    has_metric_titles = any(
        value is not None for value in (baseline_miou, tta_miou, delta_miou)
    )
    column_width = display_width
    if has_metric_titles:
        column_width = max(
            display_width,
            max(width for width, _ in title_sizes) + 8,
        )
    canvas_width = margin * 2 + column_width * 4 + panel_gap * 3
    title_height = max(height for _, height in title_sizes)
    legend_positions, legend_height = _legend_layout(
        scratch_draw,
        metadata,
        font=font,
        available_width=canvas_width - margin * 2,
    )
    panel_y = margin + title_height + title_gap
    legend_y = panel_y + display_height + legend_gap
    canvas_height = legend_y + legend_height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    for panel_index, (title, panel, title_size) in enumerate(
        zip(titles, panels, title_sizes, strict=True)
    ):
        column_x = margin + panel_index * (column_width + panel_gap)
        panel_x = column_x + (column_width - display_width) // 2
        title_width, _ = title_size
        title_x = column_x + (column_width - title_width) // 2
        draw.multiline_text(
            (title_x, margin),
            title,
            fill=(0, 0, 0),
            font=font,
            spacing=2,
            align="center",
        )
        canvas.paste(panel, (panel_x, panel_y))

    swatch_size = 12
    for class_id, item_x, item_y in legend_positions:
        x = margin + item_x
        y = legend_y + item_y
        draw.rectangle(
            (x, y, x + swatch_size - 1, y + swatch_size - 1),
            fill=metadata.palette[class_id],
            outline=(0, 0, 0),
        )
        draw.text(
            (x + swatch_size + 6, y),
            metadata.names[class_id],
            fill=(0, 0, 0),
            font=font,
        )
    return canvas


def _first_mmseg_input(batch) -> torch.Tensor:
    if isinstance(batch, dict):
        inputs = batch.get("inputs")
    elif isinstance(batch, list) and batch and isinstance(batch[0], dict):
        inputs = batch[0].get("inputs")
    else:
        raise TypeError(f"unsupported mmseg batch type: {type(batch).__name__}")
    if isinstance(inputs, torch.Tensor):
        if inputs.ndim == 4:
            return inputs[0]
        if inputs.ndim == 3:
            return inputs
    if isinstance(inputs, (list, tuple)) and inputs:
        return torch.as_tensor(inputs[0])
    raise ValueError("mmseg batch does not contain an image input")


class ComparisonVisualizationWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        classes,
        palette,
        num_classes: int,
        max_side: int,
        rank: int,
        min_delta_miou: float | None = None,
        delta_in_filename: bool = False,
    ) -> None:
        if int(max_side) <= 0:
            raise ValueError("max_side must be positive")
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = resolve_class_metadata(
            classes=classes,
            palette=palette,
            num_classes=num_classes,
        )
        self.max_side = int(max_side)
        self.rank = int(rank)
        self.min_delta_miou = (
            None if min_delta_miou is None else float(min_delta_miou)
        )
        self.delta_in_filename = bool(delta_in_filename)

    def save(
        self,
        *,
        batch,
        sample_id: str,
        gt: torch.Tensor,
        before: torch.Tensor,
        after: torch.Tensor,
        adapted: bool,
        baseline_miou: float,
        tta_miou: float,
        delta_miou: float,
    ) -> str | None:
        if (
            self.min_delta_miou is not None
            and float(delta_miou) <= self.min_delta_miou
        ):
            return None
        final_path = self.output_dir / comparison_filename(
            sample_id,
            delta_miou=(
                float(delta_miou) if self.delta_in_filename else None
            ),
        )
        temp_path = self.output_dir / (
            f".{final_path.stem}.rank{self.rank}.pid{os.getpid()}.tmp"
        )
        try:
            source = rgb_from_mmseg_input(_first_mmseg_input(batch))
            comparison = render_comparison(
                source=source,
                gt=gt,
                before=before,
                after=after,
                metadata=self.metadata,
                adapted=bool(adapted),
                max_side=self.max_side,
                baseline_miou=float(baseline_miou),
                tta_miou=float(tta_miou),
                delta_miou=float(delta_miou),
            )
            comparison.save(temp_path, format="PNG")
            temp_path.replace(final_path)
        except Exception as error:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"failed to save visualization for {sample_id}: {error}"
            ) from error
        return str(final_path)
