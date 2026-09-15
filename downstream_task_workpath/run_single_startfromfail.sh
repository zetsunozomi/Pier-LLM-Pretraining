set -euo pipefail

module load conda
conda activate lmeval

unset LD_LIBRARY_PATH
export NVIDIA_PYTORCH_VERSION="invalid"
export HF_DATASETS_TRUST_REMOTE_CODE=1

HARNESS_DIR=${HARNESS_DIR:-/pscratch/sd/s/syfan/lm-evaluation-harness}
WORK_DIR=${WORK_DIR:-/pscratch/sd/s/syfan/Pier/downstream_task_workpath}
MODEL_PATH=${1:-${MODEL_PATH:-/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/exp9_small_500/checkpoints_interval_500/iter_0100000_hf}}
BATCH_SIZE=${BATCH_SIZE:-8}
DTYPE=${DTYPE:-float}
DEVICE=${DEVICE:-cuda:0}
NUM_PROCESSES=${NUM_PROCESSES:-4}
OUTPUT_ROOT=${OUTPUT_ROOT:-${WORK_DIR}/results/$(basename "$(dirname "${MODEL_PATH}")")}

TASKS=(
  race
  mathqa
  piqa
  winogrande
)

export HF_HOME="${WORK_DIR}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export LM_HARNESS_CACHE_PATH=${LM_HARNESS_CACHE_PATH:-${WORK_DIR}/lm_eval_cache}

mkdir -p "${OUTPUT_ROOT}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${LM_HARNESS_CACHE_PATH}"
cd "${HARNESS_DIR}"

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=${DTYPE}"

echo "Harness: ${HARNESS_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Tasks: ${TASKS[*]}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Num processes: ${NUM_PROCESSES}"

for TASK in "${TASKS[@]}"; do
  OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}"
  mkdir -p "${OUTPUT_DIR}"

  EXTRA_ARGS=()
  if [[ "${TASK}" == "mathqa" ]]; then
    EXTRA_ARGS+=(--trust_remote_code)
  fi

  echo "--- Starting ${TASK} ---"
  if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    accelerate launch --multi_gpu --num_processes "${NUM_PROCESSES}" -m lm_eval run \
      --model hf \
      --model_args "${MODEL_ARGS}" \
      --tasks "${TASK}" \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${OUTPUT_DIR}" \
      --log_samples \
      "${EXTRA_ARGS[@]}"
  else
    lm-eval run \
      --model hf \
      --model_args "${MODEL_ARGS}" \
      --tasks "${TASK}" \
      --device "${DEVICE}" \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${OUTPUT_DIR}" \
      --log_samples \
      "${EXTRA_ARGS[@]}"
  fi
done
