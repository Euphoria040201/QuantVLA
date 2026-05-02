#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

QUANTVLA_CONDA_ENV="${QUANTVLA_CONDA_ENV:-groot_test}"
LIBERO_CONDA_ENV="${LIBERO_CONDA_ENV:-groot_test}"
export QUANTVLA_CONDA_ENV LIBERO_CONDA_ENV

quantvla_activate_env "${QUANTVLA_CONDA_ENV}"
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config

CONDA_ROOT="$(quantvla_find_conda_root)"
PYTHON_BIN="${CONDA_ROOT}/envs/${QUANTVLA_CONDA_ENV}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Expected python not found at ${PYTHON_BIN}"
    exit 1
fi

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_XET=1
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset VIRTUAL_ENV

CHECKPOINT="${CHECKPOINT:-/work/mingze/checkpoints/gr00t-n1.5-libero-long-posttrain}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
COLLECT_GPU="${COLLECT_GPU:-${GPU_LIST%%,*}}"

ASPQ_OUTPUT_DIR="${ASPQ_OUTPUT_DIR:-${QUANTVLA_ROOT}/results/aspq_metrics_libero10_w3a8}"
ASPQ_METRIC_PATH="${ASPQ_METRIC_PATH:-${ASPQ_OUTPUT_DIR}/aspq_metrics_top64.pt}"
ASPQ_METRIC_TOPK="${ASPQ_METRIC_TOPK:-64}"
ASPQ_NUM_SAMPLES="${ASPQ_NUM_SAMPLES:-10}"
ASPQ_RANDOM_DIRS="${ASPQ_RANDOM_DIRS:-8}"
ASPQ_TOKEN_CAP="${ASPQ_TOKEN_CAP:-512}"
FORCE_ASPQ_COLLECT="${FORCE_ASPQ_COLLECT:-0}"
SKIP_ASPQ_COLLECT="${SKIP_ASPQ_COLLECT:-0}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"

WBITS="${WBITS:-3}"
ABITS="${ABITS:-8}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
PORT_BASE="${PORT_BASE:-5600}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/libero10_aspq_w${WBITS}a${ABITS}_gpu${GPU_LIST//,/_}}"
PACKDIR="${PACKDIR:-${QUANTVLA_ROOT}/duquant_packed_aspq_${TASK_SUITE}_w${WBITS}a${ABITS}}"

GR00T_DUQUANT_INCLUDE="${GR00T_DUQUANT_INCLUDE:-.*(backbone\\.eagle_model\\.language_model\\..*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\\.model\\.transformer_blocks\\.\\d+\\.(attn1\\.(to_q|to_k|to_v|to_out\\.0)|ff\\.net\\.(0\\.proj|2))).*}"
GR00T_DUQUANT_EXCLUDE="${GR00T_DUQUANT_EXCLUDE:-(?:^|\\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\\.|$)}"
export GR00T_DUQUANT_INCLUDE GR00T_DUQUANT_EXCLUDE

mkdir -p "${ASPQ_OUTPUT_DIR}" "${OUTPUT_ROOT}/logs"

echo "========================================"
echo "ASPQ W${WBITS}A${ABITS} LIBERO-10 launcher"
echo "========================================"
echo "Checkpoint : ${CHECKPOINT}"
echo "Metric path: ${ASPQ_METRIC_PATH}"
echo "Collect GPU: ${COLLECT_GPU}"
echo "Eval GPUs  : ${GPU_LIST}"
echo "Output     : ${OUTPUT_ROOT}"
echo "Trials/task: ${NUM_TRIALS_PER_TASK}"
echo "Python     : ${PYTHON_BIN}"
echo "========================================"

if [ "${SKIP_ASPQ_COLLECT}" != "1" ]; then
    if [ "${FORCE_ASPQ_COLLECT}" = "1" ] || [ ! -f "${ASPQ_METRIC_PATH}" ]; then
        echo "[ASPQ] Collecting action-Jacobian eigenspaces..."
        CUDA_VISIBLE_DEVICES="${COLLECT_GPU}" "${PYTHON_BIN}" -u "${QUANTVLA_ROOT}/tools/collect_aspq_jacobian.py" \
            --checkpoint "${CHECKPOINT}" \
            --num-samples "${ASPQ_NUM_SAMPLES}" \
            --random-dirs "${ASPQ_RANDOM_DIRS}" \
            --token-cap "${ASPQ_TOKEN_CAP}" \
            --metric-all-layers \
            --metric-output "${ASPQ_METRIC_PATH}" \
            --metric-top-k "${ASPQ_METRIC_TOPK}" \
            --output-dir "${ASPQ_OUTPUT_DIR}" \
            --include-regex "${GR00T_DUQUANT_INCLUDE}" \
            --exclude-regex "${GR00T_DUQUANT_EXCLUDE}" \
            2>&1 | tee "${OUTPUT_ROOT}/logs/collect_aspq_jacobian.log"
    else
        echo "[ASPQ] Reusing existing metric file: ${ASPQ_METRIC_PATH}"
    fi
fi

if [ "${RUN_BENCHMARK}" = "1" ]; then
    export GR00T_DUQUANT_ASPQ=1
    export GR00T_DUQUANT_ASPQ_PATH="${ASPQ_METRIC_PATH}"
    export GR00T_DUQUANT_ASPQ_TOPK="${ASPQ_METRIC_TOPK}"
    export GR00T_DUQUANT_ASPQ_MISSING="${GR00T_DUQUANT_ASPQ_MISSING:-error}"
    export GR00T_DUQUANT_BLOCK="${GR00T_DUQUANT_BLOCK:-64}"
    export GR00T_DUQUANT_BLOCK_OUT="${GR00T_DUQUANT_BLOCK_OUT:-64}"
    export GR00T_DUQUANT_PERMUTE="${GR00T_DUQUANT_PERMUTE:-0}"
    export GR00T_DUQUANT_ROW_ROT="${GR00T_DUQUANT_ROW_ROT:-restore}"
    export GR00T_DUQUANT_ACT_PCT="${GR00T_DUQUANT_ACT_PCT:-99.9}"
    export GR00T_DUQUANT_CALIB_STEPS="${GR00T_DUQUANT_CALIB_STEPS:-32}"
    export GR00T_DUQUANT_LS="${GR00T_DUQUANT_LS:-0.15}"

    export TASK_SUITE GPU_LIST NUM_TRIALS_PER_TASK NUM_STEPS_WAIT DENOISING_STEPS
    export WBITS ABITS OUTPUT_ROOT PACKDIR PORT_BASE
    export MODEL_PATH="${CHECKPOINT}"

    echo "[ASPQ] Running LIBERO benchmark..."
    "${PYTHON_BIN}" -u "${QUANTVLA_ROOT}/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
        2>&1 | tee "${OUTPUT_ROOT}/logs/run_aspq_benchmark.log"
fi

echo "[ASPQ] Done. Output: ${OUTPUT_ROOT}"
