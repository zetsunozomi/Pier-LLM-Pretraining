#!/usr/bin/env bash
# Download the pinned Qwen2.5-3B Base snapshot; no Python or GPU dependencies.
set -euo pipefail

readonly repo='Qwen/Qwen2.5-3B'
readonly revision='3aab1f1954e9cc14eb9509a215f9e5ca08227a9b'
readonly destination="/pscratch/sd/s/syfan/Pier/local/qwen/models/Qwen2.5-3B/${revision}"
readonly base_url="https://huggingface.co/${repo}/resolve/${revision}"
readonly -a files=(
    config.json
    merges.txt
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
)

for dependency in curl stat flock; do
    command -v "$dependency" >/dev/null || {
        printf 'Missing required command: %s\n' "$dependency" >&2
        exit 1
    }
done

mkdir -p "$destination"
# Keep the lock beside the snapshot, so only the eight files remain inside it.
exec 9>"${destination}.download.lock"
flock -n 9 || {
    printf 'Another download is using %s\n' "$destination" >&2
    exit 1
}

printf 'Snapshot: https://huggingface.co/%s/tree/%s\n' "$repo" "$revision"
printf 'Destination: %s\n' "$destination"

for file in "${files[@]}"; do
    expected_size=''
    case "$file" in
        model-00001-of-00002.safetensors) expected_size=3968658944 ;;
        model-00002-of-00002.safetensors) expected_size=2203268048 ;;
    esac
    target="${destination}/${file}"
    partial="${target}.part"

    if [[ -n "$expected_size" && -f "$target" ]] &&
        [[ "$(stat -c %s "$target")" == "$expected_size" ]]; then
        printf 'Already downloaded (size verified): %s\n' "$file"
        continue
    fi

    # A completed .part file can remain if the previous run stopped before mv.
    if [[ -z "$expected_size" || ! -f "$partial" ]] ||
        [[ "$(stat -c %s "$partial")" != "$expected_size" ]]; then
        printf 'Downloading: %s\n' "$file"
        curl --fail --location --show-error \
            --retry 5 --retry-delay 5 --connect-timeout 30 \
            --continue-at - --output "$partial" \
            "${base_url}/${file}"
    fi

    if [[ ! -s "$partial" ]]; then
        printf 'Downloaded file is empty: %s\n' "$partial" >&2
        exit 1
    fi
    if [[ -n "$expected_size" ]]; then
        actual_size="$(stat -c %s "$partial")"
        if [[ "$actual_size" != "$expected_size" ]]; then
            printf 'Size mismatch for %s: expected %s bytes, got %s. Partial file retained at %s\n' \
                "$file" "$expected_size" "$actual_size" "$partial" >&2
            exit 1
        fi
    fi
    mv -- "$partial" "$target"
done

printf 'Done: all eight files are in %s\n' "$destination"
