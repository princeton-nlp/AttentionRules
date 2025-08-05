import random


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from src.utils import logging_utils

logger = logging_utils.get_logger(__name__)


def get_tokenized_subsequences(
    dataset, tokenizer, seq_length, num_examples, seed=0, offset=0
):
    # Skip the first examples (used for training)
    dataset_iter = iter(dataset)
    for i in range(offset):
        next(dataset_iter)
    examples = [e["text"] for e, _ in zip(dataset_iter, range(num_examples))]
    tokens = tokenizer(examples, padding=False)["input_ids"]
    out = []
    random.seed(seed)
    for toks in tokens:
        if len(toks) < seq_length:
            continue
        i = random.randint(0, len(toks) - seq_length)
        out.append(toks[i : i + seq_length])
    token_dataset = torch.tensor(out)
    logger.info(f"Tokenized dataset, shape: {token_dataset.shape}")
    token_dataset[:, 0] = tokenizer.bos_token_id
    return token_dataset


def get_batch_tokens_contiguous(
    dataset, model, seq_length, num_examples, seed=0, offset=0
):
    """Similar to above but use multiple examples per sequence."""
    tokens = []
    total_examples = 0
    dataset_iter = iter(dataset)
    while total_examples < num_examples:
        try:
            # Retrieve next item from iterator
            row = next(dataset_iter)["text"]
        except StopIteration:
            # Break the loop if dataset ends
            break

        # Tokenize the text (will also truncate to 1024 for GPT-2).
        toks = model.to_tokens(row, prepend_bos=True).squeeze()
        # toks = tokenizer(row, add_special_tokens=True, return_tensors="pt")["input_ids"]
        # Skip if it's too short.
        if len(toks) < seq_length:
            continue
        # Make sure sequence is multiple of seq_length.
        toks = toks[: seq_length * (len(toks) // seq_length)]

        cur_toks = toks.view(-1, seq_length)
        cur_toks = cur_toks[: num_examples - total_examples]
        tokens.append(cur_toks)
        total_examples += cur_toks.shape[0]

    # Check if any tokens were collected
    if not tokens:
        return None

    reshaped_tokens = torch.cat(tokens, dim=0)
    reshaped_tokens[:, 0] = model.tokenizer.bos_token_id
    return reshaped_tokens


def prepare_df(df, prefix_only=False, max_acts_only=True):
    if max_acts_only:
        max_bin = df["bin"].max()
        df = df.query(f"bin == {max_bin}")
    if prefix_only:
        df = df.copy(deep=True)
        df["tokens"] = [
            toks[: j + 1] for toks, j in zip(df["tokens"], df["act_argmax"])
        ]
        df["feature_acts"] = [
            acts[: j + 1] for acts, j in zip(df["feature_acts"], df["act_argmax"])
        ]
    return df.sort_values(by="act_max", ascending=False)


def bin_activations(
    activation_df, num_bins=5, examples_per_bin=30, min_count=6, seed=0, max_length=64
):
    activation_df["bin"], bins = pd.cut(
        activation_df["act_max"], bins=num_bins, labels=False, retbins=True
    )
    out = []
    for bin in range(len(bins) - 1):
        bin_df = activation_df.query(f"bin == {bin}")
        if len(bin_df) == 0:
            logger.info(f"Warning: bin {bin} is empty")
            continue
        if len(bin_df) < min_count:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {min_count}, skipping")
            continue
        if len(bin_df) < examples_per_bin:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {examples_per_bin}")
        d = bin_df.sample(n=min(examples_per_bin, len(bin_df)), random_state=seed)
        # If act_max is 0, act_argmax will usually be 0, so evaluate random positions instead.
        new_pos = np.random.randint(1, max_length, size=len(d))
        cur_pos = d["act_argmax"]
        d["act_argmax"] = np.where(cur_pos > 0, cur_pos, new_pos)
        out.append(d)
    df = pd.concat(out)
    return df, bins


def get_positive_and_negative_examples(
    activation_df, examples_per_bin=30, min_count=6, seed=0, max_length=64,
):
    activation_df = activation_df.sort_values(by="act_max", ascending=False)
    bins = -1 * (activation_df["act_max"] > 0).astype(int).to_numpy()
    for i in range(examples_per_bin):
        if activation_df.iloc[i]["act_max"] > 0:
            bins[i] = 1
    activation_df["bin"] = bins
    bins = [0, 1]
    out = []
    for bin in bins:
        bin_df = activation_df.query(f"bin == {bin}")
        if len(bin_df) == 0:
            logger.info(f"Warning: bin {bin} is empty")
            continue
        if len(bin_df) < min_count:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {min_count}, skipping")
            continue
        if len(bin_df) < examples_per_bin:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {examples_per_bin}")
        d = bin_df.sample(n=min(examples_per_bin, len(bin_df)), random_state=seed)
        # If bin is 0, act_argmax will always be 0, so sample random positions to evaluate here.
        if bin == 0:
            d["act_argmax"] = np.random.randint(1, max_length, size=len(d))
        out.append(d)
    if len(out) != 2:
        logger.info(f"Not enough examples for all bins")
        return None, None
    df = pd.concat(out)
    return df, bins


def sample_positive_and_negative_examples(
    activation_df, examples_per_bin=30, min_count=6, seed=0, max_length=64,
):
    activation_df["bin"] = (activation_df["act_max"] > 0).astype(int).to_numpy()
    bins = [0, 1]
    out = []
    for bin in bins:
        bin_df = activation_df.query(f"bin == {bin}")
        if len(bin_df) == 0:
            logger.info(f"Warning: bin {bin} is empty")
            continue
        if len(bin_df) < min_count:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {min_count}, skipping")
            continue
        if len(bin_df) < examples_per_bin:
            logger.info(f"Warning: bin {bin} has size {len(bin_df)} < {examples_per_bin}")
        d = bin_df.sample(n=min(examples_per_bin, len(bin_df)), random_state=seed)
        # If bin is 0, act_argmax will always be 0, so sample random positions to evaluate here.
        if bin == 0:
            d["act_argmax"] = np.random.randint(1, max_length, size=len(d))
        out.append(d)
    if len(out) != 2:
        logger.info(f"Not enough examples for all bins")
        return None, None
    df = pd.concat(out)
    return df, bins



def split_binned_df(df, test_size=0.5, key="bin", seed=0):
    splitter = StratifiedShuffleSplit(
        test_size=test_size, n_splits=2, random_state=seed
    )
    split = splitter.split(np.arange(len(df)), df[key])
    train_inds, test_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    return train, test
