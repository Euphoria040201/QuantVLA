#!/bin/bash
# Run ONE LIBERO-10 W3A8 experiment on a single GPU/port.
# Verifies that quantization layer replacement actually happened before
# starting the eval client (catches silent fallbacks to FP16).
#
# Usage:
#   run_one_libero_exp.sh <profile> <gpu> <port> <output_root>
# Profiles:
#   aspq_full     ASPQ-GPTQ W3A8, full 180 layers (LLM + DiT attn + DiT FF)
#   aspq_no_dit   ASPQ-GPTQ W3A8, 116 layers (LLM + DiT FF, no DiT attn)
#   duquant      DuQuant W3A8 baseline, no DiT attn (default DuQuant scope)

set -euo pipefail

PROFILE="${1:?profile required: aspq_full|aspq_no_dit|duquant}"
GPU="${2:?gpu id required}"
PORT="${3:?port required}"
OUTPUT_ROOT="${4:?output root required}"

QUANTVLA_ROOT="/data/ziyu/QuantVLA_opt"
CONDA_ROOT="/data/ziyu/miniconda3"
ENV_NAME="quantvla"
PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
CHECKPOINT="/data/ziyu/checkpoints/gr00t-n1.5-libero-long-posttrain"
LIBERO_ROOT_VAL="/data/ziyu/LIBERO"
DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"

ASPQ_GPTQ_FULL_PATH="${QUANTVLA_ROOT}/results/aspq_gptq_full_llm_dit_top64_gpu34/aspq_gptq_w3.pt"
ASPQ_GPTQ_NODIT_PATH="${QUANTVLA_ROOT}/results/aspq_gptq_no_dit_attn_top64_gpu4/aspq_gptq_w3.pt"

INCLUDE_FULL=".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*"
INCLUDE_NODIT=".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
EXCLUDE_ASPQ="(?:^|\.)(?:vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)"

# Common ziyu env (override defaults that point to /work/mingze/...)
export QUANTVLA_ROOT
export CONDA_ROOT
export LIBERO_ROOT="${LIBERO_ROOT_VAL}"
export QUANTVLA_CONDA_ENV="${ENV_NAME}"
export QUANTVLA_CACHE_ROOT="/data/ziyu/.cache/quantvla"
export LIBERO_CONFIG_PATH="/data/ziyu/.libero"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export GR00T_CPU_THREADS=4
export GR00T_INTEROP_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
unset VIRTUAL_ENV

export HF_HOME="${QUANTVLA_CACHE_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${QUANTVLA_CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${QUANTVLA_CACHE_ROOT}/xdg"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${LIBERO_CONFIG_PATH}"

export PYTHONPATH="${QUANTVLA_ROOT}:${LIBERO_ROOT_VAL}:${PYTHONPATH:-}"

# Clear any prior quant env to avoid cross-contamination
unset GR00T_ASPQ_GPTQ GR00T_ASPQ_GPTQ_PATH GR00T_ASPQ_GPTQ_INCLUDE GR00T_ASPQ_GPTQ_EXCLUDE
for v in $(compgen -v | grep '^GR00T_DUQUANT_'); do unset "$v"; done

# Profile-specific quant env
EXPECTED_LAYERS=0
QUANT_LABEL=""
case "${PROFILE}" in
    aspq_full)
        export GR00T_ASPQ_GPTQ=1
        export GR00T_ASPQ_GPTQ_PATH="${ASPQ_GPTQ_FULL_PATH}"
        export GR00T_ASPQ_GPTQ_WBITS_DEFAULT=3
        export GR00T_ASPQ_GPTQ_ABITS=8
        export GR00T_ASPQ_GPTQ_ACT_PCT=99.9
        export GR00T_ASPQ_GPTQ_CALIB_STEPS=32
        export GR00T_ASPQ_GPTQ_MISSING=error
        export GR00T_ASPQ_GPTQ_INCLUDE="${INCLUDE_FULL}"
        export GR00T_ASPQ_GPTQ_EXCLUDE="${EXCLUDE_ASPQ}"
        EXPECTED_LAYERS=180
        QUANT_LABEL="ASPQ-GPTQ W3A8 full-180"
        DETECT_PATTERN="\[GR00T-ASPQ-GPTQ\] Total layers replaced: 180"
        ;;
    aspq_no_dit)
        export GR00T_ASPQ_GPTQ=1
        export GR00T_ASPQ_GPTQ_PATH="${ASPQ_GPTQ_NODIT_PATH}"
        export GR00T_ASPQ_GPTQ_WBITS_DEFAULT=3
        export GR00T_ASPQ_GPTQ_ABITS=8
        export GR00T_ASPQ_GPTQ_ACT_PCT=99.9
        export GR00T_ASPQ_GPTQ_CALIB_STEPS=32
        export GR00T_ASPQ_GPTQ_MISSING=error
        export GR00T_ASPQ_GPTQ_INCLUDE="${INCLUDE_NODIT}"
        export GR00T_ASPQ_GPTQ_EXCLUDE="${EXCLUDE_ASPQ}"
        EXPECTED_LAYERS=116
        QUANT_LABEL="ASPQ-GPTQ W3A8 no-dit-attn"
        DETECT_PATTERN="\[GR00T-ASPQ-GPTQ\] Total layers replaced: 116"
        ;;
    duquant)
        export GR00T_DUQUANT_DEBUG=1
        export GR00T_DUQUANT_SCOPE=""
        export GR00T_DUQUANT_INCLUDE="${INCLUDE_NODIT}"
        export GR00T_DUQUANT_EXCLUDE="(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"
        export GR00T_DUQUANT_WBITS_DEFAULT=3
        export GR00T_DUQUANT_ABITS=8
        export GR00T_DUQUANT_BLOCK=64
        export GR00T_DUQUANT_BLOCK_OUT=64
        export GR00T_DUQUANT_PERMUTE=0
        export GR00T_DUQUANT_ROW_ROT=restore
        export GR00T_DUQUANT_ACT_PCT=99.9
        export GR00T_DUQUANT_CALIB_STEPS=32
        export GR00T_DUQUANT_LS=0.15
        export GR00T_DUQUANT_PACKDIR="${QUANTVLA_ROOT}/duquant_packed_libero10_w3a8_no_dit_attn"
        export GR00T_ATM_ENABLE=0
        export GR00T_OHB_ENABLE=0
        EXPECTED_LAYERS=116
        QUANT_LABEL="DuQuant W3A8 no-dit-attn baseline"
        DETECT_PATTERN="\[GR00T-DUQUANT\] Total layers replaced: 116"
        ;;
    *)
        echo "Unknown profile: ${PROFILE}" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="${GPU}"

LOGS_DIR="${OUTPUT_ROOT}/logs"
SUMMARIES_DIR="${OUTPUT_ROOT}/summaries"
ROLLOUT_DIR="${OUTPUT_ROOT}/rollouts"
mkdir -p "${LOGS_DIR}" "${SUMMARIES_DIR}" "${ROLLOUT_DIR}"

SERVER_LOG="${LOGS_DIR}/server.log"
EVAL_LOG="${LOGS_DIR}/eval.log"
RUN_LOG="${LOGS_DIR}/run.log"

{
    echo "============================================"
    echo "Profile     : ${PROFILE} (${QUANT_LABEL})"
    echo "GPU         : ${GPU}"
    echo "Port        : ${PORT}"
    echo "Expected    : ${EXPECTED_LAYERS} quantized layers"
    echo "Output      : ${OUTPUT_ROOT}"
    echo "Started     : $(date -Iseconds)"
    echo "============================================"
} | tee "${RUN_LOG}"

echo "[launcher] starting inference server (PYTHONUNBUFFERED=1) ..." | tee -a "${RUN_LOG}"
"${PY}" -u "${QUANTVLA_ROOT}/scripts/inference_service.py" \
    --model_path "${CHECKPOINT}" \
    --server \
    --data_config "${DATA_CONFIG}" \
    --denoising-steps 8 \
    --port "${PORT}" \
    --embodiment-tag new_embodiment \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[launcher] inference server PID = ${SERVER_PID}" | tee -a "${RUN_LOG}"

cleanup() {
    echo "[launcher] cleanup: stopping inference server ${SERVER_PID}" | tee -a "${RUN_LOG}"
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 3
    kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Failure regexes that abort immediately (catch silent fallback to FP16).
# Any of these in server.log is fatal.
ABORT_PATTERNS=(
    "DuQuant not enabled or failed to apply"
    "ASPQ-GPTQ not enabled or failed to apply"
    "ASPQ-GPTQ requested but failed to apply"
    "Traceback \(most recent call last\)"
    "RuntimeError"
    "FileNotFoundError"
    "KeyError: \"attribute"
)

abort_if_bad() {
    local log="$1"
    [ -f "${log}" ] || return 0
    for pat in "${ABORT_PATTERNS[@]}"; do
        if grep -qE "${pat}" "${log}" 2>/dev/null; then
            echo "[launcher] FATAL: detected error pattern in server log: '${pat}'" | tee -a "${RUN_LOG}"
            echo "--- offending lines ---" | tee -a "${RUN_LOG}"
            grep -nE "${pat}" "${log}" | head -5 | tee -a "${RUN_LOG}"
            echo "--- last 60 lines of server log ---" | tee -a "${RUN_LOG}"
            tail -60 "${log}" | tee -a "${RUN_LOG}"
            return 1
        fi
    done
    return 0
}

# Wait up to 900s for layer replacement message to appear, with periodic progress.
echo "[launcher] waiting for layer replacement to confirm quantization activated..." | tee -a "${RUN_LOG}"
DEADLINE=$(( $(date +%s) + 900 ))
QUANT_OK=0
PROGRESS_TS=$(date +%s)
while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[launcher] FATAL: inference server died before quantization activated" | tee -a "${RUN_LOG}"
        echo "--- last 60 lines of server log ---" | tee -a "${RUN_LOG}"
        tail -60 "${SERVER_LOG}" | tee -a "${RUN_LOG}"
        exit 2
    fi

    # Fail fast if any abort pattern shows up.
    if ! abort_if_bad "${SERVER_LOG}"; then
        exit 4
    fi

    if grep -qE "${DETECT_PATTERN}" "${SERVER_LOG}" 2>/dev/null; then
        QUANT_OK=1
        echo "[launcher] OK: detected '${DETECT_PATTERN}' in server log" | tee -a "${RUN_LOG}"
        break
    fi

    # Periodic progress so we can see things are moving.
    NOW_TS=$(date +%s)
    if [ $(( NOW_TS - PROGRESS_TS )) -ge 30 ]; then
        replaced_now=$(grep -c "REPLACED" "${SERVER_LOG}" 2>/dev/null || echo 0)
        log_lines=$(wc -l < "${SERVER_LOG}" 2>/dev/null || echo 0)
        echo "[launcher] +$(( NOW_TS - PROGRESS_TS ))s: server log ${log_lines} lines, REPLACED count = ${replaced_now}/${EXPECTED_LAYERS}" | tee -a "${RUN_LOG}"
        PROGRESS_TS=${NOW_TS}
    fi
    sleep 3
done

if [ "${QUANT_OK}" -ne 1 ]; then
    echo "[launcher] FATAL: timeout waiting for quantization. Server log tail:" | tee -a "${RUN_LOG}"
    tail -80 "${SERVER_LOG}" | tee -a "${RUN_LOG}"
    exit 3
fi

# Sanity-check: count of REPLACED lines must match EXPECTED_LAYERS exactly.
REPLACED_COUNT=$(grep -c "REPLACED" "${SERVER_LOG}" 2>/dev/null || echo 0)
if [ "${REPLACED_COUNT}" -ne "${EXPECTED_LAYERS}" ]; then
    echo "[launcher] FATAL: expected ${EXPECTED_LAYERS} REPLACED lines, got ${REPLACED_COUNT}" | tee -a "${RUN_LOG}"
    exit 5
fi
echo "[launcher] verified: exactly ${REPLACED_COUNT} layers replaced" | tee -a "${RUN_LOG}"

# For ASPQ-GPTQ: verify every REPLACED line ends with quant=1 (i.e. weight pack
# was found). quant=0 means the offline weight is missing for that layer and
# the layer silently fell back to FP weight — NOT what we want.
if [[ "${PROFILE}" == aspq* ]]; then
    quant0_count=$(grep -cE "AspqGptqLinear .* quant=0" "${SERVER_LOG}" 2>/dev/null || true)
    if [ "${quant0_count}" -gt 0 ]; then
        echo "[launcher] FATAL: ${quant0_count} ASPQ-GPTQ layers have quant=0 (missing weight pack)" | tee -a "${RUN_LOG}"
        grep "quant=0" "${SERVER_LOG}" | head -10 | tee -a "${RUN_LOG}"
        exit 6
    fi
    echo "[launcher] verified: all ASPQ-GPTQ layers have quant=1" | tee -a "${RUN_LOG}"
fi

# Wait briefly for ZMQ port to actually bind (quant happens before bind in policy.py)
echo "[launcher] waiting for ZMQ port ${PORT} to bind ..." | tee -a "${RUN_LOG}"
PORT_DEADLINE=$(( $(date +%s) + 300 ))
while [ "$(date +%s)" -lt "${PORT_DEADLINE}" ]; do
    if "${PY}" - <<EOF 2>/dev/null
import socket, sys
try:
    s = socket.create_connection(("127.0.0.1", ${PORT}), timeout=1.0)
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
EOF
    then
        break
    fi
    sleep 3
done

echo "[launcher] starting eval client (10 tasks, 2 trials/task) ..." | tee -a "${RUN_LOG}"
cd "${QUANTVLA_ROOT}/examples/Libero/eval"
"${PY}" -u run_libero_eval.py \
    --task_suite_name libero_10 \
    --num_trials_per_task 2 \
    --num_steps_wait 10 \
    --port "${PORT}" \
    --task_ids 0 1 2 3 4 5 6 7 8 9 \
    --log_dir "${LOGS_DIR}/eval" \
    --log_suffix "${PROFILE}" \
    --rollout_dir "${ROLLOUT_DIR}" \
    --summary_json "${SUMMARIES_DIR}/summary.json" \
    --headless \
    --no-save-videos \
    > "${EVAL_LOG}" 2>&1 &
EVAL_PID=$!
echo "[launcher] eval client PID = ${EVAL_PID}" | tee -a "${RUN_LOG}"

# Watchdog loop: while eval runs, periodically (a) verify server is alive, (b)
# scan server.log for late-arrival errors, (c) report eval progress.
# Disable set -e/pipefail inside the loop because grep returning 1 on no-match
# would otherwise kill the launcher silently during long-running eval.
set +e
set +o pipefail

WATCH_TS=$(date +%s)
while kill -0 "${EVAL_PID}" 2>/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[launcher] FATAL: inference server died during eval" | tee -a "${RUN_LOG}"
        tail -60 "${SERVER_LOG}" | tee -a "${RUN_LOG}"
        kill "${EVAL_PID}" 2>/dev/null
        wait "${EVAL_PID}" 2>/dev/null
        exit 7
    fi
    if ! abort_if_bad "${SERVER_LOG}"; then
        kill "${EVAL_PID}" 2>/dev/null
        wait "${EVAL_PID}" 2>/dev/null
        exit 8
    fi
    NOW_TS=$(date +%s)
    if [ $(( NOW_TS - WATCH_TS )) -ge 60 ]; then
        eps_done=$(grep -c "episodes completed so far:" "${EVAL_LOG}" 2>/dev/null)
        [ -z "${eps_done}" ] && eps_done=0
        last_rate=$(grep "Current total success rate:" "${EVAL_LOG}" 2>/dev/null | tail -1 | awk -F': ' '{print $2}')
        gpu_state=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "${GPU}" 2>/dev/null | tr -d '\n')
        echo "[launcher] watchdog $(date -Iseconds): eval episodes=${eps_done}, total_rate=${last_rate:-?}, gpu${GPU}=${gpu_state}" | tee -a "${RUN_LOG}"
        WATCH_TS=${NOW_TS}
    fi
    sleep 10
done

wait "${EVAL_PID}"
EVAL_RC=$?
set -e
set -o pipefail
if [ "${EVAL_RC}" -ne 0 ]; then
    echo "[launcher] WARN: eval client exited with code ${EVAL_RC}" | tee -a "${RUN_LOG}"
    tail -60 "${EVAL_LOG}" | tee -a "${RUN_LOG}"
fi

echo "[launcher] eval finished. summary:" | tee -a "${RUN_LOG}"
if [ -f "${SUMMARIES_DIR}/summary.json" ]; then
    "${PY}" -c "import json; d=json.load(open('${SUMMARIES_DIR}/summary.json')); print(f\"total: {d.get('total_successes',0)}/{d.get('total_episodes',0)} = {100*d.get('total_successes',0)/max(d.get('total_episodes',1),1):.1f}%\")" 2>&1 | tee -a "${RUN_LOG}"
else
    echo "[launcher] WARN: no summary JSON written" | tee -a "${RUN_LOG}"
fi
echo "Finished     : $(date -Iseconds)" | tee -a "${RUN_LOG}"
exit ${EVAL_RC}
