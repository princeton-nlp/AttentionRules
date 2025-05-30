# Extracting rule-based descriptions of attention features in transformers

This repository contains the code for our paper, "Extracting rule-based descriptions of attention features in transformers".
Please see our paper for more details.

## Quick links
* [Setup](#Setup)
* [Attention output SAEs](#Attention-output-SAEs)
* [Data](#Data)
* [Rule extraction](#Rule-extraction)
* [Questions?](#Questions)
* [Citation](#Citation)

## Setup

Install [PyTorch](https://pytorch.org/get-started/locally/) and then install the remaining requirements: `pip install -r requirements.txt`.
This code was tested using Python 3.12 and PyTorch version 2.3.1.

## Attention output SAEs

We train [attention output SAEs](https://arxiv.org/abs/2406.17759) for every attention head in GPT-2 small, using a fork of https://github.com/ckkissane/attention-output-saes.
These SAEs can be downloaded from: https://huggingface.co/danf0/attention-head-saes/.

## Data

Code for generating datasets of feature activations can be found in [src/get_exemplars.py](src/get_exemplars.py).
See [scripts/generate_data.sh](scripts/generate_data.sh) for the command to generate the datasets used in our paper, which are based on [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext).

## Rule extraction

Code for extracting and evaluating skip-gram rules can be found in [src/run_rules.py](src/run_rules.py).
For example, the following command will extract rules for 10 features from head 0 in layer 0.
```bash
python src/run_rules.py \
    --layer 0 \
    --head 0 \
    --num_features 10 \
    --rule_type "v1" \
    --output_dir "output/skipgrams/L0H0";
```
Code for finding and generating rules containing "distractor" features is in [src/find_distractors.py](src/find_distractors.py) and [src/generate_distractors.py](src/generate_distractors.py)
The [scripts](scripts) directory contains example contains example commands for running these scripts.

# Questions?

If you have any questions about the code or paper, please email Dan (dfriedman@cs.princeton.edu) or open an issue.

# Citation

```bibtex
@article{friedman2025extracting,
    title={Extracting rule-based descriptions of attention features in transformers},
    author={Friedman, Dan and Wettig, Alexander and Bhaskar, Adithya and Chen, Danqi},
    journal={arXiv preprint},
    year={2025}
}
```
