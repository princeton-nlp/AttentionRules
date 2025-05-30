#!/bin/bash 

LAYER=0
HEAD=0

python src/find_distractors.py \
  --cmd "get_distractor_examples" \
  --layer "${LAYER}" \
  --head "${HEAD}" \
  --example_dir "data/openwebtext_n50000_bins2x50/" \
  --output_dir "output/distractor_examples/L${LAYER}H${SLURM_ARRAY_TASK_ID}/";
