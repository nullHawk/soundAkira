#!/usr/bin/env bash
# Process sources on every visible GPU (one shard per GPU), then build once.
#   scripts/multi_gpu.sh sources.txt config.yaml
set -euo pipefail

INPUTS=${1:?usage: multi_gpu.sh INPUTS CONFIG}
CONFIG=${2:?usage: multi_gpu.sh INPUTS CONFIG}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

pids=()
for ((i = 0; i < NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i soundakira process "$INPUTS" -c "$CONFIG" --shard "$i/$NUM_GPUS" \
    > "shard-$i.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || echo "some shards reported failures; run 'soundakira status -c $CONFIG --errors'"

soundakira build -c "$CONFIG"
