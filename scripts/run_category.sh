#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 MUSC_ROOT DATA_ROOT DATASET_NAME CATEGORY OUTPUT_ROOT [VIS]" >&2
  exit 2
fi

musc_root=$1
data_root=$2
dataset_name=$3
category=$4
output_root=$5
vis=${6:-False}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -f "$musc_root/examples/musc_main.py" ]]; then
  echo "MuSc entry point not found: $musc_root/examples/musc_main.py" >&2
  exit 1
fi

if [[ ! -d "$data_root" ]]; then
  echo "Dataset root not found: $data_root" >&2
  exit 1
fi

mkdir -p "$output_root"

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
