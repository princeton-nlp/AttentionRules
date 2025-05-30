#!/bin/bash 

HEAD=0

python src/generate_distractors.py \
  --layer 0 \
  --head "${HEAD}" \
  --example_dir "output/distractor_examples/" \
  --output_dir "output/generated_distractor_examples/L0H${HEAD}/";
