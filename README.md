# AeroTTA: Dual-Granularity Low-Rank Test-Time Adaptation for Open-Vocabulary UAV Semantic Segmentation

## Environment Setup


Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate aerotta
```

Install the full MMCV package after PyTorch is available:

```bash
mim install "mmcv==2.1.0"
```

## Checkpoint and Data

Place the SAM3 checkpoint at:

```text
weight/sam3.pt
```

Place each dataset under `data/` following the paths in `configs/cfg_*.py`.
Run all commands from the repository root because the checkpoint and dataset
paths are relative to this directory.

## Model evaluation

The launcher uses one GPU and does not start tmux by default. For example:

```bash
python lora-tta/eval.py tta-configs/cfg_uavid.py 
```

Replace `cfg_uavid.py` with `cfg_udd5.py`, `cfg_vdd.py`, `cfg_potsdam.py`, or
`cfg_vaihingen.py` to evaluate another dataset.

