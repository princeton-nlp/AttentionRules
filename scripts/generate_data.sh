#!/bin/bash 

LAYER=0
HEAD=0
DATASET_SIZE=50000
NUM_BINS=2
EXAMPLES_PER_BIN=50
SEED=0

python src/get_exemplars.py \
  --layer "${LAYER}" \
  --head "${HEAD}" \
  --dataset_size "${DATASET_SIZE}" \
  --num_bins "${NUM_BINS}" \
  --examples_per_bin "${EXAMPLES_PER_BIN}" \
  --max_length 64 \
  --num_features 100 \
  --min_count 150 \
  --max_count 49850 \
  --dataset_path "Skylion007/openwebtext" \
  --seed "${SEED}" \
  --output_dir "data/openwebtext_n${DATASET_SIZE}_bins${NUM_BINS}x${EXAMPLES_PER_BIN}/L${LAYER}H${HEAD}/";