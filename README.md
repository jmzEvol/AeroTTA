# AeroTTA: Dual-Granularity Low-Rank Test-Time Adaptation for Open-Vocabulary UAV Semantic Segmentation

## Environment Setup

The code has been tested with Python 3.12, PyTorch 2.7.0, CUDA 12.6,
MMCV 2.1.0, MMEngine 0.10.4, and MMSegmentation 1.2.2.

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate aerotta
```

Install the full MMCV package after PyTorch is available:

```bash
mim install "mmcv==2.1.0"
```

Use `mmcv`, not `mmcv-lite`, and do not install both in the same environment.
If MIM cannot find a compatible binary wheel, building MMCV from source requires
a local CUDA toolkit compatible with the installed PyTorch CUDA build.

Verify the core runtime:

```bash
python -c "import torch, torchvision, mmcv, mmengine, mmseg; \
print('torch:', torch.__version__); \
print('torchvision:', torchvision.__version__); \
print('CUDA:', torch.version.cuda, torch.cuda.is_available()); \
print('mmcv:', mmcv.__version__); \
print('mmengine:', mmengine.__version__); \
print('mmseg:', mmseg.__version__)"

python -c "from mmcv.ops import roi_align; \
from torchvision.ops import roi_align as tv_roi_align; \
print('compiled ops: OK')"
```

## Checkpoint and Data

Place the SAM3 checkpoint at:

```text
weight/sam3.pt
```

Place each dataset under `data/` following the paths in `configs/cfg_*.py`.
Run all commands from the repository root because the checkpoint and dataset
paths are relative to this directory.

## Run LoRA TTA

The launcher uses one GPU and does not start tmux by default. For example:

```bash
python lora-tta/eval.py tta-configs/cfg_uavid.py --gpus 0 --no-tmux
```

Replace `cfg_uavid.py` with `cfg_udd5.py`, `cfg_vdd.py`, `cfg_potsdam.py`, or
`cfg_vaihingen.py` to evaluate another dataset.

