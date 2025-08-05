"""Get activating examples for SAE features."""

import argparse
import json
from pathlib import Path
import sys

from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset
import numpy as np
import torch
from tqdm import tqdm

from src.utils import data_utils, logging_utils, sae_utils

logger = logging_utils.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Model
    parser.add_argument("--model_path", type=str, default="gpt2-small")

    # Data
    parser.add_argument("--dataset_path", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split_type", type=str, default="bins")
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--examples_per_bin", type=int, default=10)
    parser.add_argument("--uniform_sample", type=int, default=0)

    # SAEs
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--sae_in", type=str, default="gpt2-small-res-jb")
    parser.add_argument("--num_features", type=int, default=10)

    # Dashboards for per_head SAE
    parser.add_argument("--min_count", type=int, default=100)
    parser.add_argument("--max_count", type=int, default=None)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)

    args = parser.parse_args()
    logging_utils.initialize(args.output_dir)

    return args


def load_model_and_dataset(
    model_path="gpt2-small",
    dataset_path="Skylion007/openwebtext",
    max_length=128,
    dataset_size=None,
    device=torch.device("cpu"),
    model_only=False,
    seed=0,
):
    model = HookedSAETransformer.from_pretrained(model_path, device=device)
    if model_only:
        return model, None, None
    logger.info(f"Loading dataset...")
    dataset = load_dataset(
        path=dataset_path,
        split="train",
        streaming=True,
    )
    logger.info(f"Tokenizing dataset...")
    tokens = data_utils.get_batch_tokens_contiguous(
        dataset,
        model,
        seq_length=max_length,
        num_examples=dataset_size,
        seed=seed,
    )
    return model, dataset, tokens


def run_get_examples(args):
    device = torch.device(args.device)

    logger.info(f"Loading model...")
    model, _, tokens = load_model_and_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        dataset_size=args.dataset_size,
        device=device,
        seed=args.seed,
    )

    logger.info(f"Loading SAE...")
    sae = sae_utils.load_per_head_sae(args.layer, args.head)
    logger.info(f"Getting frequencies...")
    counts, avg_acts = sae_utils.get_activation_frequencies(
        model=model,
        sae=sae,
        eval_tokens=tokens,
        per_head=True,
        per_seq=True,
        normalize=False,
        return_avg_acts=True,
        batch_size=args.batch_size,
    )
    if args.max_count is not None:
        idxs = ((counts >= args.min_count) & (counts <= args.max_count)).nonzero()[0]
        logger.info(
            f"{len(idxs)}/{len(counts)} features with {args.min_count} <= count <= {args.max_count}"
        )
    else:
        idxs = (counts >= args.min_count).nonzero()[0]
        logger.info(
            f"{len(idxs)}/{len(counts)} features with min_count >= {args.min_count}"
        )
    np.random.seed(args.seed)
    features = np.random.choice(
        idxs, size=min(len(idxs), args.num_features), replace=False
    )
    logger.info(f"Getting activations for {len(features)} features...")
    activation_df = sae_utils.get_activations_per_sequence(
        feats=features,
        model=model,
        sae=sae,
        tokens=tokens,
        head=args.head,
        batch_size=args.batch_size,
        per_head=True,
        post_act=True,
    )

    out = {split: [] for split in ("train", "val", "test")}
    logger.info(f"Getting examples...")
    if args.num_bins == 2:
        logger.info(f"num_bins == 2, getting positive and negative examples")
    for feat in tqdm(features):
        if args.num_bins == 2 and args.uniform_sample:
            binned_df, _ = data_utils.sample_positive_and_negative_examples(
                activation_df.query(f"feature == {feat}").copy(deep=True),
                examples_per_bin=args.examples_per_bin * 3,
                max_length=args.max_length,
                seed=args.data_seed,
            )
        if args.num_bins == 2:
            binned_df, _ = data_utils.get_positive_and_negative_examples(
                activation_df.query(f"feature == {feat}").copy(deep=True),
                examples_per_bin=args.examples_per_bin * 3,
                max_length=args.max_length,
                seed=args.data_seed,
            )
        else:
            binned_df, _ = data_utils.bin_activations(
                activation_df.query(f"feature == {feat}").copy(deep=True),
                examples_per_bin=args.examples_per_bin * 3,
                num_bins=args.num_bins,
                seed=args.data_seed,
            )
        if binned_df is None:
            logger.info(f"Don't have enough activations per bin for {feat}, skipping")
            continue
        train_val, test = data_utils.split_binned_df(
            binned_df, test_size=0.333, seed=args.data_seed
        )
        train, val = data_utils.split_binned_df(
            train_val, test_size=0.5, seed=args.data_seed
        )
        for split, df in (("train", train), ("val", val), ("test", test)):
            dfas = sae_utils.get_dfa_for_batch(
                feat=feat,
                model=model,
                sae=sae,
                tokens=tokens,
                df=df,
            )
            out[split] += dfas

    fn = Path(args.output_dir) / f"counts.json"
    logger.info(f"Writing counts to {fn}")
    with open(fn, "w") as f:
        json.dump({"counts": counts.tolist(), "avg_acts": avg_acts.tolist()}, f)

    for split, dfas in out.items():
        fn = Path(args.output_dir) / f"{split}.json"
        logger.info(f"Writing {len(dfas)} examples to {fn}")
        with open(fn, "w") as f:
            json.dump(dfas, f)


if __name__ == "__main__":
    args = parse_args()
    torch.set_grad_enabled(False)
    logger.info(f"args: {vars(args)}")
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)
    cmd_line = " ".join(sys.argv)
    logger.info(f"command line: {cmd_line}")
    with open(Path(args.output_dir) / "cmd.txt", "w") as f:
        f.write(cmd_line)
    run_get_examples(args)
