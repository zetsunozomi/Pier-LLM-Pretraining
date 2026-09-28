#!/usr/bin/env bash
#SBATCH --account=m4431
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm40g
#SBATCH --job-name=pier-paper-fixed
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pier-paper-fixed-%A_%a.out
#SBATCH --export=ALL
#SBATCH --mail-user=sf850@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

# One full 250-step case per short allocation. Case and repeat are explicit;
# this submits no other jobs and does not run a standalone validation suite.
set -euo pipefail
export PIER_ROOT="${PIER_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PIER_ROOT/experiments/joint/environment.sh"
[[ "$PIER_JOINT_NODES" == 8 ]] || { echo 'Table VI requires 8 nodes / 32 GPUs / TP2.' >&2; exit 2; }
[[ $# -le 2 ]] || { echo 'Usage: paper.sbatch [case=joint-s1] [repeat=1]' >&2; exit 2; }
paper_case="${1:-joint-s1}"
paper_repeat="${2:-${SLURM_ARRAY_TASK_ID:-1}}"
case "$paper_case" in
    single-s1|single-s2|single-s16|pipeline-s1|pipeline-s2|pipeline-s16|separate-s1|joint-s1|reference-s1|reference-s2|reference-s16) ;;
    *) echo 'Expected one Table VI case; native-controls contains multiple training launches.' >&2; exit 2 ;;
esac
case "$paper_repeat" in 1|2|3) ;; *) echo 'Repeat must be 1, 2 or 3.' >&2; exit 2 ;; esac
source experiments/qwen/log_output.sh
pier_capture_output joint-paper-fixed
echo "Paper target: Table VI, $paper_case, independent repeat $paper_repeat; 100 warmup + 150 measured steps."
exec "$PIER_PYTHON" experiments/joint/qwen.py --scenario fixed --only "$paper_case" \
    --repeat-id "$paper_repeat" --output-dir "$PIER_OUT_DIR/campaign"
