#!/bin/bash 

LAYER=0
HEAD=0
METHOD="magnitude"
# METHOD="importance"

srun python src/run_rules.py \
  --layer "${LAYER}" \
  --head "${HEAD}" \
  --method "${METHOD}" \
  --num_features 100 \
  --example_dir "data/openwebtext_n50000_bins2x50/" \
  --output_dir "output/skipgrams/${METHOD}/L${LAYER}H${HEAD}/";
