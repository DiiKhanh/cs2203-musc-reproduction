#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 MUSC_ROOT DATA_ROOT DATASET_NAME CATEGORY OUTPUT_ROOT [VIS]" >&2
  exit 2
fi

expected_commit=b76b93da8bd3096a99964a96ae29d46f197a0651
musc_root_arg=$1
data_root_arg=$2
dataset_name=$3
category=$4
output_root_arg=$5
vis=${6:-False}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -f "$musc_root_arg/examples/musc_main.py" ]]; then
  echo "MuSc entry point not found: $musc_root_arg/examples/musc_main.py" >&2
  exit 1
fi

if [[ ! -d "$data_root_arg" ]]; then
  echo "Dataset root not found: $data_root_arg" >&2
  exit 1
fi

musc_root=$(cd "$musc_root_arg" && pwd -P)
data_root=$(cd "$data_root_arg" && pwd -P)
mkdir -p "$output_root_arg"
output_root=$(cd "$output_root_arg" && pwd -P)

actual_commit=$(git -C "$musc_root" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Unexpected MuSc commit: $actual_commit" >&2
  echo "Expected MuSc commit:   $expected_commit" >&2
  exit 1
fi

printf '%s\n' "$actual_commit" > "$output_root/upstream-commit.txt"

cd "$musc_root"
"$python_bin" examples/musc_main.py \
  --device 0 \
  --data_path "$data_root" \
  --dataset_name "$dataset_name" \
  --class_name "$category" \
  --backbone_name ViT-L-14-336 \
  --pretrained openai \
  --feature_layers 5 11 17 23 \
  --img_resize 518 \
  --divide_num 1 \
  --r_list 1 3 5 \
  --batch_size 4 \
  --output_dir "$output_root" \
  --vis "$vis" \
  --vis_type whole_norm \
  --save_excel True
