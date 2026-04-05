#!/usr/bin/env bash
set -euo pipefail
cd /workspace/axis-inoculation-lab
export PYTHONPATH="/workspace/assistant-axis:/workspace/axis-inoculation-lab${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running run_unfrozen_inoc_recheck.sh}"
POST_DONE=/workspace/axis-inoculation-lab/artifacts/mechanism_v2_midlayer_protected_qwen3_8b/ALL_DONE.txt
CFG=/workspace/axis-inoculation-lab/configs/unfrozen_inoc_recheck_config.json
EM_ROOT=artifacts/mechanism_v2_recheck_qwen3_8b/em_eval
LOG=artifacts/automation_logs/unfrozen_inoc_recheck.log
mkdir -p artifacts/automation_logs artifacts/mechanism_v2_recheck_qwen3_8b
log(){ printf '[%s] %s
' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"; }
while [[ ! -f "$POST_DONE" ]]; do sleep 30; done
log "frozen exp1 vector/axis complete; starting unfrozen inoc recheck"
if [[ ! -f artifacts/qwen3-8b-mechanism-v2-recheck/qwen3-8b-mechanism-v2-recheck-Qwen-Qwen3-8B-bad_medical_advice-inoculation-s0-rerun/hf_finetune/trained_model_info.json ]]; then
  env CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run_hf_finetune.py --run-config "$CFG" > artifacts/automation_logs/unfrozen_inoc_recheck_ft.log 2>&1
else
  log "training already complete, skipping"
fi
python3 em_eval.py --model-path artifacts/qwen3-8b-mechanism-v2-recheck/qwen3-8b-mechanism-v2-recheck-Qwen-Qwen3-8B-bad_medical_advice-inoculation-s0-rerun/hf_finetune/final --run-id mechanism_v2_recheck_inoculation_s0 --output-dir "$EM_ROOT" --n-samples 25 --gpu-id 0 > artifacts/automation_logs/unfrozen_inoc_recheck_em.log 2>&1
log "recheck complete"
