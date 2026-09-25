cat > /pscratch/sd/s/syfan/download_qwen15b.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

repo='Qwen/Qwen2.5-1.5B'
revision='8faed761d45a263340a0528343f099c05c9a4323'
destination="/pscratch/sd/s/syfan/Pier/local/qwen/models/Qwen2.5-1.5B/$revision"
base_url="https://huggingface.co/$repo/resolve/$revision"

for dependency in curl stat git sha256sum flock; do
    command -v "$dependency" >/dev/null || {
        printf 'Missing command: %s\n' "$dependency" >&2
        exit 1
    }
done

mkdir -p "$destination"
exec 9>"${destination}.download.lock"
flock -n 9 || {
    echo 'Another download is using this destination.' >&2
    exit 1
}

verify() {
    local path="$1" size="$2" kind="$3" expected="$4" digest
    [[ -f "$path" && "$(stat -c %s "$path")" == "$size" ]] || return 1
    if [[ "$kind" == sha256 ]]; then
        digest=$(sha256sum -- "$path") || return 1
        digest=${digest%% *}
    else
        digest=$(git hash-object --no-filters -- "$path") || return 1
    fi
    [[ "$digest" == "$expected" ]]
}

printf 'Destination: %s\n' "$destination"

while read -r file size kind digest; do
    target="$destination/$file"
    partial="$target.part"

    if [[ -e "$target" ]]; then
        if verify "$target" "$size" "$kind" "$digest"; then
            printf 'Verified, skipping: %s\n' "$file"
            continue
        fi
        printf 'Existing file failed verification; retained: %s\n' "$target" >&2
        exit 1
    fi

    if [[ ! -f "$partial" ]] || [[ "$(stat -c %s "$partial")" -lt "$size" ]]; then
        printf 'Downloading/resuming: %s\n' "$file"
        curl --fail --location --show-error \
            --retry 5 --retry-delay 5 --connect-timeout 30 \
            --continue-at - --output "$partial" "$base_url/$file"
    fi

    if ! verify "$partial" "$size" "$kind" "$digest"; then
        printf 'Downloaded file failed verification; retained: %s\n' "$partial" >&2
        exit 1
    fi

    mv -- "$partial" "$target"
    printf 'Verified: %s\n' "$file"
done <<'FILES'
config.json 684 blob 169d606d5cf502a5628a52381586679a6d0caaab
merges.txt 1671839 blob 20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0
model.safetensors 3087467144 sha256 a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796
tokenizer.json 7031645 blob 443909a61d429dff23010e5bddd28ff530edda00
tokenizer_config.json 7228 blob ba7e4c5637b9732dadcd66286ce48334e8b31e9e
vocab.json 2776833 blob 4783fe10ac3adce15ac8f358ef5462739852c569
FILES

printf 'Done: all six files verified in %s\n' "$destination"
BASH

bash /pscratch/sd/s/syfan/download_qwen15b.sh