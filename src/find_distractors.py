"""Find distractor key features."""

import argparse
import json
from pathlib import Path
import sys
import traceback

from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils import data_utils, logging_utils, rule_utils, sae_utils

logger = logging_utils.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cmd", type=str, default="get_distractors")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Model
    parser.add_argument("--model_path", type=str, default="gpt2-small")

    # Data
    parser.add_argument("--dataset_path", type=str, default="Skylion007/openwebtext")
    parser.add_argument("--dataset_size", type=int, default=50_000)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--example_dir",
        type=str,
        default="output/exp11_exemplars/openwebtext_n50000_bins5x20",
    )
    parser.add_argument("--num_features", type=int, default=100)
    parser.add_argument("--max_examples", type=int, default=100)

    # SAEs
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--sae_in", type=str, default="gpt2-small-res-jb")

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


def load_saes(
    model,
    layer=0,
    head=0,
    sae_in_release="gpt2-small-res-jb",
    device="cuda",
):
    sae_in, _, _ = SAE.from_pretrained(
        release=sae_in_release,
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device,
    )
    sae_out = sae_utils.load_per_head_sae(layer=layer, head=head)
    if layer == 0:
        explanation_df = sae_utils.get_tokens_df(model, sae_in, k=3)
    else:
        explanation_df = sae_utils.get_explanations_df(sae_id=f"{layer}-res-jb")
    return sae_in, sae_out, explanation_df


def load_examples(example_dir, layer, head):
    split_fns = [
        Path(example_dir) / f"L{layer}H{head}" / f"{split}.json"
        for split in ("train", "val", "test")
    ]
    splits = []
    for split_fn in split_fns:
        with open(split_fn, "r") as f:
            examples = json.load(f)
        splits.append(pd.DataFrame(examples))
    return splits


def run_get_distractors(args):
    device = torch.device(args.device)

    logger.info(f"Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_path, device=device)

    logger.info(f"Loading SAEs...")

    sae_in, sae_out, explanation_df = load_saes(
        model,
        layer=args.layer,
        head=args.head,
        sae_in_release=args.sae_in,
        device=args.device,
    )

    train_df, _, _ = load_examples(args.example_dir, args.layer, args.head)
    features = train_df["feature"].unique()[: args.num_features]

    out = []
    for feat in features:
        try:
            distractor = rule_utils.get_distractor_key_for_feature(
                model=model,
                feat=feat,
                sae_in=sae_in,
                sae_out=sae_out,
                head=args.head,
                layer=args.layer,
                explain_df=explanation_df,
            )
            if distractor is not None:
                out.append(distractor)
        except Exception as e:
            logger.info(f"Error getting interactions for feature {feat}: {e}")
            logger.info(traceback.format_exc())

    df = pd.DataFrame(out)
    fn = Path(args.output_dir) / "distractors.csv"
    logger.info(f"Writing {len(df)} distractors to {fn}")
    df.to_csv(fn)
    return df


def run_get_distractor_examples(args):
    device = torch.device(args.device)

    logger.info(f"Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_path, device=device)

    logger.info(f"Loading SAEs...")

    sae_in, sae_out, explanation_df = load_saes(
        model,
        layer=args.layer,
        head=args.head,
        sae_in_release=args.sae_in,
        device=args.device,
    )

    train_df, _, _ = load_examples(args.example_dir, args.layer, args.head)
    features = train_df["feature"].unique()[: args.num_features]

    out = []
    for feat in features:
        try:
            distractor = rule_utils.get_distractor_key_for_feature(
                model=model,
                feat=feat,
                sae_in=sae_in,
                sae_out=sae_out,
                head=args.head,
                layer=args.layer,
                explain_df=explanation_df,
            )
            if distractor is not None:
                out.append(distractor)
        except Exception as e:
            logger.info(f"Error getting interactions for feature {feat}: {e}")
            logger.info(traceback.format_exc())

    df = pd.DataFrame(out)
    fn = Path(args.output_dir) / "distractors.csv"
    logger.info(f"Writing {len(df)} distractors to {fn}")
    df.to_csv(fn)

    # Get distractor examples
    _, _, tokens = load_model_and_dataset(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        max_length=args.max_length,
        dataset_size=args.dataset_size,
        device=device,
        seed=args.seed,
    )

    out = []
    stats = []
    for feat in features:
        feat_df = df.query(f"feature == {feat}")
        if not len(feat_df):
            continue
        d = feat_df.iloc[0]
        counter_examples, kq_count, kkq_count = rule_utils.find_kkq_examples(
            model,
            sae_in,
            sae_out,
            feat,
            d["k"],
            d["k_"],
            d["q"],
            tokens,
            torch.arange(len(tokens)),
            max_examples=args.max_examples,
            batch_size=args.batch_size,
        )
        total = kq_count + kkq_count
        if total == 0:
            logger.info(f"No examples for {feat}, skipping")
            stats.append({"feature": feat, "kq_count": 0, "kkq_count": 0})
        else:
            logger.info(f"Found {total} examples, {kkq_count} with distractor")
            out += counter_examples
            stats.append(
                {"feature": feat, "kq_count": kq_count, "kkq_count": kkq_count}
            )

    fn = Path(args.output_dir) / "distractor_examples.json"
    logger.info(f"Writing {len(out)} distractor example to {fn}")
    with open(fn, "w") as f:
        json.dump(out, f)

    stats_df = pd.DataFrame(stats)
    fn = Path(args.output_dir) / "distractor_stats.csv"
    stats_df.to_csv(fn)


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
    logger.info(f"cmd: {args.cmd}")
    if args.cmd == "get_distractors":
        run_get_distractors(args)
    elif args.cmd == "get_distractor_examples":
        run_get_distractor_examples(args)
    else:
        raise NotImplementedError(args.cmd)
