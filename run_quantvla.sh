#!/bin/bash
# GR00T DuQuant W4A8 Full Quantization (LLM + DiT ALL Linear Layers)
# Quantize both LLM (Eagle VLM) and DiT (Action Head) linear layers with W4A8

set -e

cd /home/xinyu/QuantVLA

# -----------------------------
# Task configuration
# -----------------------------
TASK_SUITE="${1:-libero_10}"
if [ -n "$2" ]; then
    MODEL_PATH="$2"
else
    case "$TASK_SUITE" in
        libero_spatial)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_goal)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
            ;;
        libero_object)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_90)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_10)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        *)
            echo "Unknown task suite: $TASK_SUITE"
            echo "Valid options: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
            exit 1
            ;;
    esac
fi
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"

echo "========================================"
echo "GR00T DuQuant W4A8 Full Quantization"
echo "LLM + DiT ALL Linear Layers"
echo "========================================"
echo "Task suite: $TASK_SUITE"
echo "Model: $MODEL_PATH"
echo ""

# ============================================
# DuQuant W4A8 Full Configuration
# ============================================
export GR00T_DUQUANT_DEBUG=1

# SCOPE: Empty = search entire model
export GR00T_DUQUANT_SCOPE=""

# INCLUDE / EXCLUDE patterns
export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'

# Quantization parameters
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=0
export GR00T_DUQUANT_ROW_ROT=restore
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32
export GR00T_DUQUANT_LS=0.15

# Pack directory
export GR00T_DUQUANT_PACKDIR="/home/xinyu/QuantVLA/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_long_0"

# ============================================
# ATM/OHB (GROUPED) configuration
# ============================================
# IMPORTANT: use your grouped JSON
export GR00T_ATM_ALPHA_PATH=/home/xinyu/QuantVLA/atm_alpha_beta_long_group4.json
export GR00T_ATM_ENABLE=1
export GR00T_ATM_SCOPE=${GR00T_ATM_SCOPE:-dit}

export GR00T_OHB_ENABLE=1
export GR00T_OHB_FALLBACK=1.0
export GR00T_OHB_SCOPE=${GR00T_OHB_SCOPE:-dit}

# ============================================
# Denoising steps (must match your group_edges)
# group_edges=[0,2,4,6,8] ==> num_steps=8
# ============================================
export GR00T_DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}

# Disable torch.compile for compatibility
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Disable CUDA graphs to avoid memory issues
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

echo "DuQuant Config:"
echo "  SCOPE: $GR00T_DUQUANT_SCOPE"
echo "  INCLUDE: $GR00T_DUQUANT_INCLUDE"
echo "  EXCLUDE: $GR00T_DUQUANT_EXCLUDE"
echo "  WBITS=$GR00T_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$GR00T_DUQUANT_ABITS"
echo "  BLOCK=$GR00T_DUQUANT_BLOCK"
echo "  PERMUTE=$GR00T_DUQUANT_PERMUTE"
echo "  ROW_ROT=$GR00T_DUQUANT_ROW_ROT"
echo "  ACT_PCT=$GR00T_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$GR00T_DUQUANT_CALIB_STEPS"
echo "  LS=$GR00T_DUQUANT_LS"
echo "  PACKDIR=$GR00T_DUQUANT_PACKDIR"
echo ""
echo "ATM/OHB:"
echo "  ATM_ALPHA_PATH=$GR00T_ATM_ALPHA_PATH"
echo "  ATM_ENABLE=$GR00T_ATM_ENABLE  ATM_SCOPE=$GR00T_ATM_SCOPE"
echo "  OHB_ENABLE=$GR00T_OHB_ENABLE  OHB_SCOPE=$GR00T_OHB_SCOPE  OHB_FALLBACK=$GR00T_OHB_FALLBACK"
echo ""
echo "Denoising:"
echo "  GR00T_DENOISING_STEPS=$GR00T_DENOISING_STEPS"
echo ""

# --------------------------------------------
# Dry-run to show which layers will be quantized
# --------------------------------------------
echo "🔍 DRY RUN: Scanning layers to quantize..."
echo ""
export GR00T_DUQUANT_DRYRUN=1
export GR00T_MODEL_PATH="$MODEL_PATH"
export GR00T_DATA_CONFIG="$DATA_CONFIG"

python - <<'PY'
import os
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config

model_path = os.environ.get("GR00T_MODEL_PATH")
data_config_path = os.environ.get("GR00T_DATA_CONFIG")

print("Loading model for DuQuant dry-run...")
cfg = load_data_config(data_config_path)
policy = Gr00tPolicy(
    model_path=model_path,
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=int(os.environ.get("GR00T_DENOISING_STEPS", "8")),
)
print("\n✅ DuQuant dry-run complete!\n")
PY

echo ""
echo "========================================"
echo "Dry run complete. Review the layers above."
echo ""
echo "Press Enter to continue with actual quantization, or Ctrl+C to cancel..."
if [ -t 0 ]; then
  read -r
else
  echo "[INFO] Non-interactive (nohup). Skip prompt and continue."
fi

# Clear dry-run flag
unset GR00T_DUQUANT_DRYRUN
unset GR00T_MODEL_PATH
unset GR00T_DATA_CONFIG

echo ""
echo "🚀 Starting quantized inference server..."
echo ""
echo "NOTE: Make sure run_inference_server.sh uses GR00T_DENOISING_STEPS (not hard-coded 8)."
echo ""

# Start the inference server (expects run_inference_server.sh to read env vars)
./run_inference_server.sh "$TASK_SUITE"