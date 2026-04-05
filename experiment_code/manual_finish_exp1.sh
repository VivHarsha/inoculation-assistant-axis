#!/usr/bin/env bash
set -eo pipefail
cd /workspace/axis-inoculation-lab
export PYTHONPATH="/workspace/assistant-axis:/workspace/axis-inoculation-lab${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running manual_finish_exp1.sh}"
WORKSPACE=/workspace
BASE_MODEL=Qwen/Qwen3-8B
OS_ROOT=/workspace/axis-inoculation-lab/artifacts/mechanism_v2_midlayer_protected_qwen3_8b
BASE_ROLES_DIR=/workspace/axis-inoculation-lab/artifacts/killer_v1_qwen3_8b/base/roles
QUESTIONS_FILE=/workspace/assistant-axis/data/extraction_questions.jsonl
AXIS_OUT=$OS_ROOT/axis_analysis
NO_MODEL=artifacts/qwen3-8b-midlayer-protected/qwen3-8b-midlayer-protected-Qwen-Qwen3-8B-bad_medical_advice-no_inoc-s0-68b887ba4b3d/hf_finetune/final
INOC_MODEL=artifacts/qwen3-8b-midlayer-protected/qwen3-8b-midlayer-protected-Qwen-Qwen3-8B-bad_medical_advice-inoculation-s0-295e557bfc76/hf_finetune/final
mkdir -p "$OS_ROOT/no_inoc/s0" "$OS_ROOT/inoculation/s0" "$AXIS_OUT" artifacts/automation_logs
log(){ printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a artifacts/automation_logs/manual_finish_exp1.log; }
extract_condition(){
  local condition="$1" model="$2" gpu="$3"
  local out_dir="$OS_ROOT/$condition/s0"
  if [[ ! -f "$out_dir/roles/summary.json" ]]; then
    log "[$condition GPU$gpu] extracting roles"
    CUDA_VISIBLE_DEVICES=$gpu python3 compute_role_trait_vectors.py \
      --data-root "$WORKSPACE/assistant-axis/data" \
      --model "$model" --tokenizer "$BASE_MODEL" \
      --kind roles --include-default \
      --questions "$QUESTIONS_FILE" --max-questions 20 \
      --max-new-tokens 128 --gen-batch-size 64 \
      --judge-model gpt-4o-mini --judge-min-score 2 \
      --output-dir "$out_dir/roles" \
      > "artifacts/automation_logs/manual_vec_${condition}_roles.log" 2>&1
  else
    log "[$condition] roles already present, skipping"
  fi
  if [[ ! -f "$out_dir/traits/summary.json" ]]; then
    log "[$condition GPU$gpu] extracting traits"
    CUDA_VISIBLE_DEVICES=$gpu python3 compute_role_trait_vectors.py \
      --data-root "$WORKSPACE/assistant-axis/data" \
      --model "$model" --tokenizer "$BASE_MODEL" \
      --kind traits \
      --questions "$QUESTIONS_FILE" --max-questions 20 \
      --max-new-tokens 128 --gen-batch-size 64 \
      --judge-model gpt-4o-mini --judge-min-score 2 \
      --output-dir "$out_dir/traits" \
      > "artifacts/automation_logs/manual_vec_${condition}_traits.log" 2>&1
  else
    log "[$condition] traits already present, skipping"
  fi
  log "[$condition] vector extraction complete"
}
extract_condition no_inoc "$NO_MODEL" 0 &
PID_A=$!
extract_condition inoculation "$INOC_MODEL" 1 &
PID_B=$!
wait $PID_A
wait $PID_B
log "running axis analysis"
python3 compute_axis_and_project.py \
  --base-dir "$BASE_ROLES_DIR" \
  --arm "prot_no_inoc:s0:$OS_ROOT/no_inoc/s0/roles" \
  --arm "prot_inoculation:s0:$OS_ROOT/inoculation/s0/roles" \
  --output-dir "$AXIS_OUT" \
  --target-layer 14 \
  --all-layers \
  > artifacts/automation_logs/manual_final_control_axis.log 2>&1
log "axis analysis complete"
touch "$OS_ROOT/ALL_DONE.txt"
log "ALL_DONE written"
