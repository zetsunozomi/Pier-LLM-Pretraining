#!/bin/bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=convert_megatron_to_hf.%j.out
#SBATCH --job-name=mg2hf-exp1
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

LOAD_PATH=${LOAD_PATH:-/pscratch/sd/s/syfan/Pier/replace_exps/exp_xl/iter_0100000}

module load conda
conda activate lmeval

WORK_DIR=${WORK_DIR:-/pscratch/sd/s/syfan/Pier/downstream_task_workpath}
CONVERT_SCRIPT=${CONVERT_SCRIPT:-${WORK_DIR}/convert/checkpoint_reshaping_and_interoperability.py}
MEGATRON_PATH=${MEGATRON_PATH:-/pscratch/sd/s/syfan/Pier}

SAVE_PATH=${SAVE_PATH:-$(dirname "${LOAD_PATH}")/$(basename "${LOAD_PATH}")_hf}
TOKENIZER_NAME=${TOKENIZER_NAME:-openai-community/gpt2}
MAX_SHARD_SIZE=${MAX_SHARD_SIZE:-10GB}

export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"

export HF_HOME=${HF_HOME:-${WORK_DIR}/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}

mkdir -p "${SAVE_PATH}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "Convert script: ${CONVERT_SCRIPT}"
echo "Megatron source path: ${MEGATRON_PATH}"
echo "Megatron checkpoint root: ${LOAD_PATH}"
echo "HF output path: ${SAVE_PATH}"
echo "Tokenizer: ${TOKENIZER_NAME}"
echo "Max shard size: ${MAX_SHARD_SIZE}"

python "${CONVERT_SCRIPT}" \
  --megatron-path "${MEGATRON_PATH}" \
  --convert_checkpoint_from_megatron_to_transformers \
  --load_path "${LOAD_PATH}" \
  --save_path "${SAVE_PATH}" \
  --tokenizer_name "${TOKENIZER_NAME}" \
  --max_shard_size "${MAX_SHARD_SIZE}"

echo "Done. HF checkpoint written to: ${SAVE_PATH}"
