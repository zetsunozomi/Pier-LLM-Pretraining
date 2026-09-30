bash <<'BASH'
set -euo pipefail
cd /pscratch/sd/s/syfan/Pier
export PIER_ROOT="$PWD"
export PIER_OUT_ROOT="$PWD/out"
export PIER_PYTHON=/pscratch/sd/s/syfan/conda/envs/diloco/bin/python

mkdir -p "$PIER_OUT_ROOT"
paper_receipt=$(mktemp "$PIER_OUT_ROOT/paper-fixed-queue.XXXXXX")
paper_previous=58997737
printf 'job_id\tcase\trepeat\n58997737\tjoint-s1\t1\n' > "$paper_receipt"

for paper_repeat in 1 2 3; do
  for paper_case in joint-s1 separate-s1 \
    single-s1 single-s2 single-s16 \
    pipeline-s1 pipeline-s2 pipeline-s16 \
    reference-s1 reference-s2 reference-s16; do

    if [[ "$paper_case" == joint-s1 && "$paper_repeat" == 1 ]]; then
      continue
    fi

    paper_job=$(sbatch --parsable \
      --job-name="pf-${paper_case}-r${paper_repeat}" \
      experiments/joint/paper.sbatch "$paper_case" "$paper_repeat")
    paper_job=${paper_job%%;*}

    printf '%s\t%s\t%s\n' \
      "$paper_job" "$paper_case" "$paper_repeat" | tee -a "$paper_receipt"
    paper_previous="$paper_job"
  done
done
printf '提交记录：%s\n' "$paper_receipt"
BASH