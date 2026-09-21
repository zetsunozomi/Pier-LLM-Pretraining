# Sourced after the launcher resolves PIER_ROOT. Slurm opens --output before
# running the script, so create our log directory here for both bash and sbatch.
pier_capture_output() {
    local stage="$1"
    local output_root="${PIER_OUT_ROOT:-$PIER_ROOT/out}"
    mkdir -p "$output_root"
    output_root=$(cd "$output_root" && pwd)
    # A fresh directory also keeps repeated attempts in one allocation separate.
    PIER_OUT_DIR=$(mktemp -d "$output_root/$stage-${SLURM_JOB_ID:-no-job}-$(date +%Y%m%d-%H%M%S).XXXXXX")
    export PIER_OUT_DIR
    printf '[%s] Full stdout/stderr: %s/out.txt\n' "$stage" "$PIER_OUT_DIR"
    exec >"$PIER_OUT_DIR/out.txt" 2>&1
    printf '[%s] Full stdout/stderr: %s/out.txt\n' "$stage" "$PIER_OUT_DIR"
}
