"""Generate examples with distractor keys."""

import argparse
import json
from pathlib import Path
import sys

from sae_lens import SAE, HookedSAETransformer
from datasets import load_dataset
import pandas as pd
import torch

from src.utils import data_utils, logging_utils, rule_utils, sae_utils

logger = logging_utils.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Model
    parser.add_argument("--model_path", type=str, default="gpt2-small")

    # Data
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--example_dir",
        type=str,
        default="output/exp12_distractor_examples/openwebtext_n50000_bins5x20_nf100",
    )

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


def load_distractors(example_dir, layer, head):
    base = Path(example_dir) / f"L{layer}H{head}"
    fn = base / "distractor_examples.json"
    with open(fn, "r") as f:
        examples = pd.DataFrame(json.load(f))
    fn = base / "distractors.csv"
    distractors = pd.read_csv(fn, index_col=0)
    fn = base / "distractor_stats.csv"
    distractor_stats = pd.read_csv(fn, index_col=0)

    distractors["has_distractor_key"] = distractors["k_rank"] > distractors["k_rank_"]
    stats_df = distractor_stats.join(distractors.set_index(["feature"]), on=["feature"])
    stats_df["pct_with_distractor"] = [
        kkq / max(kkq + kq, 1)
        for kkq, kq in zip(stats_df["kkq_count"], stats_df["kq_count"])
    ]
    stats_df["has_kq_example"] = stats_df["kq_count"] > 0
    stats_df["has_kkq_example"] = stats_df["kkq_count"] > 0
    examples_df = examples.join(stats_df.set_index(["feature"]), on=["feature"])

    return examples_df, distractors


def insert_token(d, token, token_id, n=1):
    tokens = d["tokens"][::]
    token_ids = d["token_ids"][::]
    pos = d["q_pos"]
    seq_len = len(tokens)
    for _ in range(n):
        tokens.insert(1, token)
        token_ids.insert(1, token_id)
        pos += 1
    if pos >= seq_len:
        return None
    tokens, token_ids = tokens[:seq_len], token_ids[:seq_len]
    return {
        "feature": d["feature"],
        "tokens": tokens,
        "token_ids": token_ids,
        "q_pos": pos,
        "distractor_token": token,
        "distractor_token_id": token_id,
        "num_inserted": n,
    }


def run_generate_distractors_for_feature(
    feat,
    model,
    sae_in,
    sae_out,
    examples_df,
):
    # Make sure the activation is positive.
    full_df = examples_df.query(f"feature == {feat} & kind == 'no_distractor'")
    df = full_df.query(f"act_out > 0")
    logger.info(f"{len(full_df)} k/q examples for f{feat}, {len(df)} with act > 0")
    if len(df) == 0:
        logger.info(f"No examples with positive activations, skipping")
        return []

    # Find distractor word.
    k_ = df.iloc[0]["k_"]
    token, token_id, _ = (
        l[0] for l in sae_utils.get_top_tokens_for_feat(model, sae_in, k_)
    )
    if sae_utils.is_position(token):
        logger.info(f"Top word for f{k_} is {token}, skipping")
        return []
    logger.info(f"Top word: '{token}'")

    # Insert the distractor word 0-4 times.
    rows = []
    for n in range(4):
        rows += [insert_token(row, token, token_id, n=n) for _, row in df.iterrows()]
    rows = list(filter(lambda d: d is not None, rows))
    edit_df = pd.DataFrame(rows)

    # Get new activations
    tokens = torch.tensor(edit_df["token_ids"].to_list())
    positions = edit_df["q_pos"]
    dfas = rule_utils.get_dfa_for_batches(feat, model, sae_out, tokens, positions)
    for d, row in zip(dfas, rows):
        d["distractor_token"] = token
        d["distractor_token_id"] = int(token_id)
        d["num_inserted"] = row["num_inserted"]

    return dfas


def run_generate_distractors(args):
    device = torch.device(args.device)

    logger.info(f"Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_path, device=device)

    logger.info(f"Loading SAEs...")
    sae_in, sae_out, _ = load_saes(
        model,
        layer=args.layer,
        head=args.head,
        sae_in_release=args.sae_in,
        device=args.device,
    )

    examples, _ = load_distractors(args.example_dir, args.layer, args.head)
    feature_df = examples.groupby("feature").sample(1)

    # Log statistics
    n = len(feature_df)
    n_with_key = len(feature_df.query("has_distractor_key"))
    n_with_kq = len(feature_df.query("has_kq_example"))
    n_with_kkq = len(feature_df.query("has_kkq_example"))
    logger.info(
        f"Loaded {n} features, "
        f"{n_with_key} with distractor key, "
        f"{n_with_kq} with k/q examples, "
        f"{n_with_kkq} with k/k'/q examples."
    )

    # Pick features that have kq examples.
    distractor_features = (
        examples.query(f"has_kq_example & has_distractor_key")
        .groupby("feature")
        .sample(1)
    )["feature"]

    # Generate examples.
    out = []
    for feat in distractor_features:
        out += run_generate_distractors_for_feature(
            feat,
            model,
            sae_in,
            sae_out,
            examples.query(f"feature == {feat}"),
        )

    if not out:
        logger.info(f"No features with distractors...")
        return

    df = pd.DataFrame(out)
    logger.info(df.groupby(["feature", "num_inserted"])[["act_out"]].mean())

    fn = Path(args.output_dir) / "generated_distractors.json"
    logger.info(f"Writing {len(out)} examples to {fn}")
    with open(fn, "w") as f:
        json.dump(out, f)


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
    run_generate_distractors(args)
