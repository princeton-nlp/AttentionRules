#!/bin/bash 

LAYER=0
HEAD=0

srun python src/find_distractors.py \
  --cmd "get_distractors" \
  --layer "${LAYER}" \
  --head "${HEAD}" \
  --example_dir "data/openwebtext_n50000_bins2x50/" \
  --output_dir "output/distractor_features/L${LAYER}H${HEAD}/";
