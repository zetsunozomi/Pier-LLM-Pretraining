#!/bin/bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=lm_eval.%j.out
#SBATCH --job-name=lm-eval
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

WORK_DIR="/pscratch/sd/s/syfan/Pier/downstream_task_workpath"
RUN_SINGLE="${WORK_DIR}/run_single.sh"
RESULTS_ROOT="${WORK_DIR}/results/batch_${SLURM_JOB_ID:-manual}"  
LOG_ROOT="${WORK_DIR}/logs/batch_${SLURM_JOB_ID:-manual}"

CHECKPOINTS=(
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/exp1_rerun/checkpoints_exp1_50_new/iter_0100000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/checkpoints_exp4_weak4/iter_0200000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/checkpoints_exp5_weak16/iter_0050000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/checkpoints_exp6_weak32/iter_0025000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/exp7_small_100/checkpoints_interval_100/iter_0100000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/exp8_small_200/checkpoints_interval_200/iter_0100000_hf"
  "/pscratch/sd/s/syfan/Pier/replace_exps/new_miu_exp/exp9_small_500/checkpoints_interval_500/iter_0100000_hf"
)

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

echo "Run script: ${RUN_SINGLE}"
echo "Results root: ${RESULTS_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "Number of checkpoints: ${#CHECKPOINTS[@]}"

for MODEL_PATH in "${CHECKPOINTS[@]}"; do
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing checkpoint directory: ${MODEL_PATH}" >&2
    exit 1
  fi

  CHECKPOINT_NAME="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}")"
  CHECKPOINT_OUTPUT_ROOT="${RESULTS_ROOT}/${CHECKPOINT_NAME}"
  CHECKPOINT_LOG="${LOG_ROOT}/${CHECKPOINT_NAME}.txt"

  echo "Starting ${MODEL_PATH}"
  echo "  output: ${CHECKPOINT_OUTPUT_ROOT}"
  echo "  log: ${CHECKPOINT_LOG}"

  OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}" bash "${RUN_SINGLE}" "${MODEL_PATH}" \
    >"${CHECKPOINT_LOG}" 2>&1

  echo "Finished ${MODEL_PATH}"
done

echo "All checkpoint evaluations finished."
