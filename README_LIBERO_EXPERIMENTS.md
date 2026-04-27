# QuantVLA LIBERO Experiment Guide

This guide documents the local experiment workflow we used for:

- layerwise quantization drift analysis on `GR00T N1.5`
- pure DuQuant LIBERO benchmark runs
- the paper-aligned `W3A8` benchmark setting

## Scope

These scripts assume:

- repo path: `/work/mingze/QuantVLA`
- LIBERO path: `/work/mingze/LIBERO`
- conda envs:
  - `groot_test`
  - `libero_test`
- checkpoint path:
  - `/work/mingze/checkpoints/gr00t-n1.5-libero-long-posttrain`

The helper script [scripts/common_paths.sh](/work/mingze/QuantVLA/scripts/common_paths.sh) centralizes most shared paths. If you move to another machine, update those paths first.

## Files Added For This Workflow

- [scripts/common_paths.sh](/work/mingze/QuantVLA/scripts/common_paths.sh)
- [scripts/run_layerwise_quant_4gpu.sh](/work/mingze/QuantVLA/scripts/run_layerwise_quant_4gpu.sh)
- [tools/analyze_layerwise_quant_drift.py](/work/mingze/QuantVLA/tools/analyze_layerwise_quant_drift.py)
- [tools/plot_layerwise_quant_results.py](/work/mingze/QuantVLA/tools/plot_layerwise_quant_results.py)
- [scripts/run_libero_duquant_benchmark_multi_gpu.sh](/work/mingze/QuantVLA/scripts/run_libero_duquant_benchmark_multi_gpu.sh)
- [scripts/run_libero_duquant_benchmark_multi_gpu.py](/work/mingze/QuantVLA/scripts/run_libero_duquant_benchmark_multi_gpu.py)

Modified evaluation files:

- [examples/Libero/eval/run_libero_eval.py](/work/mingze/QuantVLA/examples/Libero/eval/run_libero_eval.py)
- [examples/Libero/eval/utils.py](/work/mingze/QuantVLA/examples/Libero/eval/utils.py)

## 1. Environment Check

Inference side:

```bash
source /work/mingze/miniconda3/etc/profile.d/conda.sh
conda activate groot_test
python -c "import torch, gr00t; print(torch.__version__, torch.cuda.is_available())"
```

LIBERO eval side:

```bash
source /work/mingze/miniconda3/etc/profile.d/conda.sh
conda activate libero_test
python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"
```

## 2. Layerwise Quant Drift Analysis

This analysis compares full precision against quantized variants on fixed LIBERO observations. It records:

- action drift relative to full precision
- per-layer `W` drift
- per-layer `WX` drift
- `individual` and `cumulative` sensitivity

### Default 4-GPU run

```bash
cd /work/mingze/QuantVLA

NUM_SAMPLES=10 \
GPU_LIST=4,5,6,7 \
OUTPUT_ROOT=/work/mingze/QuantVLA/results/layerwise_quant_4gpu_s10_gpu4567 \
bash /work/mingze/QuantVLA/scripts/run_layerwise_quant_4gpu.sh
```

### 3-bit version

```bash
cd /work/mingze/QuantVLA

NUM_SAMPLES=10 \
GPU_LIST=4,5,6,7 \
WBITS=3 \
ABITS=8 \
OUTPUT_ROOT=/work/mingze/QuantVLA/results/layerwise_quant_4gpu_s10_w3a8_gpu4567 \
bash /work/mingze/QuantVLA/scripts/run_layerwise_quant_4gpu.sh
```

### Main outputs

- `scenario_summary.csv`
- `per_layer_metrics.jsonl`
- `plots/plot_summary.md`

## 3. Paper-Aligned Pure DuQuant LIBERO Benchmark

This benchmark uses the paper-aligned quantization layout:

- quantize all `LLM` linear layers
- quantize only `DiT MLP` layers
- keep `DiT attention` in floating point
- disable `ATM` and `OHB`

### W3A8 benchmark on `libero_10`

```bash
cd /work/mingze/QuantVLA

GPU_LIST=0,1,2,3,4,5,6,7 \
TASK_SUITE=libero_10 \
WBITS=3 \
ABITS=8 \
NUM_TRIALS_PER_TASK=5 \
OUTPUT_ROOT=/work/mingze/QuantVLA/results/libero_duquant_libero_10_w3a8_gpu0_7_novideo \
bash /work/mingze/QuantVLA/scripts/run_libero_duquant_benchmark_multi_gpu.sh
```

### W4A8 benchmark

```bash
cd /work/mingze/QuantVLA

GPU_LIST=0,1,2,3,4,5,6,7 \
TASK_SUITE=libero_10 \
WBITS=4 \
ABITS=8 \
NUM_TRIALS_PER_TASK=5 \
OUTPUT_ROOT=/work/mingze/QuantVLA/results/libero_duquant_libero_10_w4a8_gpu0_7_novideo \
bash /work/mingze/QuantVLA/scripts/run_libero_duquant_benchmark_multi_gpu.sh
```

### Main outputs

- `merged_summary.json`
- `merged_summary.md`
- per-shard logs under `logs/`

## 4. Known 3-bit Result

The current `W3A8` pure DuQuant benchmark result for `libero_10` is:

- overall success: `1 / 500 = 0.2%`
- output directory:
  - `/work/mingze/QuantVLA/results/libero_duquant_libero_10_w3a8_gpu0_7_novideo`

This strongly suggests that naive 3-bit quantization is not viable for this checkpoint without additional recovery or protection mechanisms.

## 5. What To Copy To Another Machine

Required:

- this repo
- `/work/mingze/LIBERO`
- `/work/mingze/.libero/config.yaml`
- GR00T checkpoint
- the two conda environments, or exported `.yml` files

Optional:

- LeRobot dataset under `LIBERO/datasets/`
- previous results for plotting only

Do not commit checkpoints, datasets, or `results/` into git.
