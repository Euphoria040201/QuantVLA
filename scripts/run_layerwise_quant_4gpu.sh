#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

quantvla_activate_env quantvla
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config
CONDA_ROOT="$(quantvla_find_conda_root)"
PYTHON_BIN="${CONDA_ROOT}/envs/quantvla/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Expected python not found at ${PYTHON_BIN}"
    exit 1
fi

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
unset VIRTUAL_ENV

CHECKPOINT="${CHECKPOINT:-/work/mingze/checkpoints/gr00t-n1.5-libero-long-posttrain}"
DATA_SOURCE="${DATA_SOURCE:-libero}"
DATASET_PATH="${DATASET_PATH:-/work/mingze/LIBERO/datasets/lerobot_libero_10}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"
LIBERO_NUM_TRIALS_PER_TASK="${LIBERO_NUM_TRIALS_PER_TASK:-5}"
LIBERO_NUM_STEPS_WAIT="${LIBERO_NUM_STEPS_WAIT:-10}"
LIBERO_SAMPLING_MODE="${LIBERO_SAMPLING_MODE:-sequential}"
LIBERO_RESOLUTION="${LIBERO_RESOLUTION:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/layerwise_quant_4gpu}"
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchvision_av}"
MODES="${MODES:-individual,cumulative}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BLOCK_OUT_SIZE="${BLOCK_OUT_SIZE:-64}"
PERMUTE="${PERMUTE:-0}"
ROW_ROT="${ROW_ROT:-restore}"
ACT_PCT="${ACT_PCT:-99.9}"
CALIB_STEPS="${CALIB_STEPS:-32}"
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-0.15}"
START_LAYER="${START_LAYER:-0}"
MAX_LAYERS="${MAX_LAYERS:-0}"
PACKDIR="${PACKDIR:-${QUANTVLA_ROOT}/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_long_0}"
SEED="${SEED:-42}"
AUTO_PLOT="${AUTO_PLOT:-1}"
SPLIT_BY_TASK="${SPLIT_BY_TASK:-0}"
INCLUDE_REGEX="${INCLUDE_REGEX:-.*(backbone\\.eagle_model\\.language_model\\..*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\\.model\\.transformer_blocks\\.\\d+\\.(attn1\\.(to_q|to_k|to_v|to_out\\.0)|ff\\.net\\.(0\\.proj|2))).*}"
EXCLUDE_REGEX="${EXCLUDE_REGEX:-(?:^|\\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\\.|$)}"
EXTRA_EXCLUDE_REGEX="${EXTRA_EXCLUDE_REGEX:-}"

if [ -n "${EXTRA_EXCLUDE_REGEX}" ]; then
    EXCLUDE_REGEX="(?:${EXCLUDE_REGEX})|(?:${EXTRA_EXCLUDE_REGEX})"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_GPUS="${#GPUS[@]}"
PLAN_GPU="${GPUS[0]}"

if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "GPU_LIST is empty"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
PLAN_DIR="${OUTPUT_ROOT}/plan"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${PLAN_DIR}" "${LOG_DIR}"

COMMON_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --data-source "${DATA_SOURCE}"
    --dataset-path "${DATASET_PATH}"
    --task-suite-name "${TASK_SUITE_NAME}"
    --libero-num-trials-per-task "${LIBERO_NUM_TRIALS_PER_TASK}"
    --libero-num-steps-wait "${LIBERO_NUM_STEPS_WAIT}"
    --libero-sampling-mode "${LIBERO_SAMPLING_MODE}"
    --libero-resolution "${LIBERO_RESOLUTION}"
    --data-config "${DATA_CONFIG}"
    --embodiment-tag "${EMBODIMENT_TAG}"
    --video-backend "${VIDEO_BACKEND}"
    --num-samples "${NUM_SAMPLES}"
    --denoising-steps "${DENOISING_STEPS}"
    --modes "${MODES}"
    --start-layer "${START_LAYER}"
    --max-layers "${MAX_LAYERS}"
    --wbits "${WBITS}"
    --abits "${ABITS}"
    --block-size "${BLOCK_SIZE}"
    --block-out-size "${BLOCK_OUT_SIZE}"
    --permute "${PERMUTE}"
    --row-rot "${ROW_ROT}"
    --act-pct "${ACT_PCT}"
    --calib-steps "${CALIB_STEPS}"
    --lambda-smooth "${LAMBDA_SMOOTH}"
    --packdir "${PACKDIR}"
    --seed "${SEED}"
    --include-regex "${INCLUDE_REGEX}"
    --exclude-regex "${EXCLUDE_REGEX}"
)

if [ "${SPLIT_BY_TASK}" = "1" ]; then
    COMMON_ARGS+=(--split-by-task)
fi

echo "========================================"
echo "4-GPU Layerwise Quant Launcher"
echo "========================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "Data Src   : ${DATA_SOURCE}"
echo "Dataset    : ${DATASET_PATH}"
echo "Task Suite : ${TASK_SUITE_NAME}"
echo "Output     : ${OUTPUT_ROOT}"
echo "Modes      : ${MODES}"
echo "GPUs       : ${GPU_LIST}"
echo "Plan GPU   : ${PLAN_GPU}"
echo "Samples    : ${NUM_SAMPLES}"
echo "Libero Samp: ${LIBERO_SAMPLING_MODE}"
echo "Layers     : start=${START_LAYER} max=${MAX_LAYERS}"
echo "Python     : ${PYTHON_BIN}"
echo "Include RX : ${INCLUDE_REGEX}"
echo "Exclude RX : ${EXCLUDE_REGEX}"
echo "========================================"

CUDA_VISIBLE_DEVICES="${PLAN_GPU}" \
"${PYTHON_BIN}" "${QUANTVLA_ROOT}/tools/analyze_layerwise_quant_drift.py" \
    "${COMMON_ARGS[@]}" \
    --device cuda \
    --plan-only \
    --output-dir "${PLAN_DIR}"

TOTAL_SCENARIOS="$(
python - <<'PY' "${PLAN_DIR}/plan_summary.json"
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
with open(plan_path, "r") as f:
    plan = json.load(f)
print(plan["num_total_scenarios"])
PY
)"

if [ "${TOTAL_SCENARIOS}" -le 0 ]; then
    echo "No scenarios to run."
    exit 1
fi

CHUNK_SIZE=$(( (TOTAL_SCENARIOS + NUM_GPUS - 1) / NUM_GPUS ))
echo "Total scenarios: ${TOTAL_SCENARIOS}"
echo "Chunk size per GPU: ${CHUNK_SIZE}"

PIDS=()
ACTIVE_SHARDS=()
for idx in "${!GPUS[@]}"; do
    gpu="${GPUS[$idx]}"
    start=$(( idx * CHUNK_SIZE ))
    if [ "${start}" -ge "${TOTAL_SCENARIOS}" ]; then
        continue
    fi

    remaining=$(( TOTAL_SCENARIOS - start ))
    if [ "${remaining}" -lt "${CHUNK_SIZE}" ]; then
        count="${remaining}"
    else
        count="${CHUNK_SIZE}"
    fi

    shard_dir="${OUTPUT_ROOT}/shard_${idx}"
    log_path="${LOG_DIR}/shard_${idx}.log"
    mkdir -p "${shard_dir}"

    shard_args=(
        "${COMMON_ARGS[@]}"
        --scenario-start "${start}"
        --scenario-count "${count}"
        --device cuda
        --output-dir "${shard_dir}"
    )
    if [ "${idx}" -gt 0 ]; then
        shard_args+=(--skip-baseline)
    fi

    echo "[Launch] shard=${idx} gpu=${gpu} scenarios=${start}..$((start + count - 1))"
    CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON_BIN}" "${QUANTVLA_ROOT}/tools/analyze_layerwise_quant_drift.py" \
        "${shard_args[@]}" > "${log_path}" 2>&1 &

    PIDS+=("$!")
    ACTIVE_SHARDS+=("${idx}")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done

if [ "${FAIL}" -ne 0 ]; then
    echo "At least one shard failed. Check logs under ${LOG_DIR}"
    exit 1
fi

python - <<'PY' "${OUTPUT_ROOT}" "${PLAN_DIR}" "${ACTIVE_SHARDS[@]}"
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
plan_dir = Path(sys.argv[2])
shard_ids = [int(x) for x in sys.argv[3:]]

with open(plan_dir / "scenario_plan.json", "r") as f:
    scenario_plan = json.load(f)
scenario_order = {row["scenario"]: i for i, row in enumerate(scenario_plan)}

summary_rows = []
per_layer_rows = []
task_summary_rows = []
task_per_layer_rows = []
for shard_id in shard_ids:
    shard_dir = output_root / f"shard_{shard_id}"

    summary_path = shard_dir / "scenario_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_rows.extend(json.load(f))

    per_layer_path = shard_dir / "per_layer_metrics.jsonl"
    if per_layer_path.exists():
        with open(per_layer_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    per_layer_rows.append(json.loads(line))

    task_summary_path = shard_dir / "task_scenario_summary.json"
    if task_summary_path.exists():
        with open(task_summary_path, "r") as f:
            task_summary_rows.extend(json.load(f))

    task_per_layer_path = shard_dir / "task_per_layer_metrics.jsonl"
    if task_per_layer_path.exists():
        with open(task_per_layer_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    task_per_layer_rows.append(json.loads(line))

baseline_rows = [row for row in summary_rows if row.get("mode") == "baseline"]
non_baseline_rows = [row for row in summary_rows if row.get("mode") != "baseline"]
non_baseline_rows.sort(key=lambda row: scenario_order.get(row["scenario"], 10**9))
summary_rows = baseline_rows + non_baseline_rows

per_layer_rows.sort(
    key=lambda row: (
        scenario_order.get(row["scenario"], 10**9),
        row.get("layer_name", ""),
    )
)

with open(output_root / "scenario_summary.json", "w") as f:
    json.dump(summary_rows, f, indent=2)

with open(output_root / "per_layer_metrics.jsonl", "w") as f:
    for row in per_layer_rows:
        f.write(json.dumps(row) + "\n")

if task_summary_rows:
    task_baseline_rows = [row for row in task_summary_rows if row.get("mode") == "baseline"]
    task_non_baseline_rows = [row for row in task_summary_rows if row.get("mode") != "baseline"]
    task_non_baseline_rows.sort(
        key=lambda row: (
            int(row.get("task_id", 10**9)),
            scenario_order.get(row["scenario"], 10**9),
        )
    )
    task_summary_rows = task_baseline_rows + task_non_baseline_rows
    with open(output_root / "task_scenario_summary.json", "w") as f:
        json.dump(task_summary_rows, f, indent=2)

    import csv
    task_fieldnames = []
    for row in task_summary_rows:
        for key in row.keys():
            if key not in task_fieldnames:
                task_fieldnames.append(key)
    with open(output_root / "task_scenario_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=task_fieldnames)
        writer.writeheader()
        for row in task_summary_rows:
            writer.writerow(row)

if task_per_layer_rows:
    task_per_layer_rows.sort(
        key=lambda row: (
            int(row.get("task_id", 10**9)),
            scenario_order.get(row["scenario"], 10**9),
            row.get("layer_name", ""),
        )
    )
    with open(output_root / "task_per_layer_metrics.jsonl", "w") as f:
        for row in task_per_layer_rows:
            f.write(json.dumps(row) + "\n")

fieldnames = []
for row in summary_rows:
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

if fieldnames:
    import csv
    with open(output_root / "scenario_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

manifest = {
    "num_shards": len(shard_ids),
    "shard_ids": shard_ids,
    "num_scenarios": len(summary_rows) - len(baseline_rows),
    "has_baseline": bool(baseline_rows),
}
with open(output_root / "merge_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
PY

if [ "${AUTO_PLOT}" != "0" ]; then
    echo "Generating plots..."
    "${PYTHON_BIN}" "${QUANTVLA_ROOT}/tools/plot_layerwise_quant_results.py" \
        --results-dir "${OUTPUT_ROOT}"
fi

echo "========================================"
echo "Finished"
echo "Logs       : ${LOG_DIR}"
echo "Summary    : ${OUTPUT_ROOT}/scenario_summary.csv"
echo "Per-layer  : ${OUTPUT_ROOT}/per_layer_metrics.jsonl"
if [ "${AUTO_PLOT}" != "0" ]; then
echo "Plots      : ${OUTPUT_ROOT}/plots"
fi
echo "========================================"
