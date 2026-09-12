# AeroTTA: Dual-Granularity Low-Rank Test-Time Adaptation for Open-Vocabulary UAV Semantic Segmentation

## Environment Setup

conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -U openmim
mim install "mmengine==0.10.4"
mim install "mmcv==2.1.0"
pip install "mmsegmentation==1.2.2"
Use the full `mmcv` package, not `mmcv-lite`, and do not install both in the
same environment. If MIM cannot find a compatible wheel, it will build MMCV
from source and requires a local CUDA toolkit compatible with the installed
PyTorch build.

### 4. Install SAM3 and runtime dependencies

```bash
pip install \
  timm==1.0.22 \
  numpy==1.26.4 \
  tqdm==4.65.2 \
  ftfy==6.1.1 \
  regex \
  iopath==0.1.10 \
  typing_extensions \
  huggingface_hub \
  pillow \
  opencv-python \
  einops \
  scipy \
  scikit-image \
  matplotlib \
  pyyaml \
  psutil
```

### 5. Verify the installation

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

