"""Evaluate rule approximation quality."""

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sae_lens import SAE, HookedSAETransformer
import torch

from src.utils import logging_utils, rule_utils, sae_utils

logger = logging_utils.get_logger(__name__)
warnings.simplefilter("ignore", stats.ConstantInputWarning)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cmd", type=str, default="run_rules")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/scratch")

    # Model
    parser.add_argument("--model_path", type=str, default="gpt2-small")

    # Data
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--example_dir",
        type=str,
        default="data/openwebtext_n50000_bins2x50",
    )
    parser.add_argument("--num_features", type=int, default=100)

    # SAEs
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--sae_in", type=str, default="gpt2-small-res-jb")
    parser.add_argument("--sae_out_path", type=str, default="../attention-output-saes/checkpoints")

    # Rules
    parser.add_argument("--method", type=str, default="magnitude")
    parser.add_argument("--num_values", type=int, default=100)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--max_rules", type=int, default=4096)
    parser.add_argument("--mask_val", type=float, default=1.0)
    parser.add_argument("--loss_type", type=str, default="activation")
    parser.add_argument("--use_seq_mask", type=int, default=0)
    parser.add_argument("--absolute", type=int, default=1)
    parser.add_argument("--aggregation_type", type=str, default="")
    parser.add_argument("--qv_only", type=int, default=0)
    parser.add_argument("--negative_values", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--save_preds", type=int, default=0)
    parser.add_argument("--save_eval", type=int, default=0)

    args = parser.parse_args()
    logging_utils.initialize(args.output_dir)

    if not args.aggregation_type:
        args.aggregation_type = "kqv_max" if args.method == "importance" else "max"

    return args


def load_saes(
    model,
    layer=0,
    head=0,
    sae_in_release="gpt2-small-res-jb",
    sae_out_path="../attention-output-saes/checkpoints",
    device="cuda",
):
    sae_in, _, _ = SAE.from_pretrained(
        release=sae_in_release,
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device,
    )
    sae_out = sae_utils.load_per_head_sae(layer=layer, head=head, checkpoint_dir=sae_out_path)
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
        df = pd.DataFrame(examples)
        if "act_max" not in df.columns:
            df["act_max"] = df["act"]
        if "act_argmax" not in df.columns:
            df["act_argmax"] = df["position"]
        splits.append(df)
    return splits


def get_interactions_for_feature(
    feat,
    model,
    sae_in,
    sae_out,
    train_df,
    head=0,
    layer=0,
    batch_size=8,
    loss_type="activation",
    num_values=100,
    num_queries=100,
    absolute=True,
    max_rules=1024,
    explain_df=None,
    negative_values=False,
    device=torch.device("cuda"),
):
    """
    Calculate gradient-based importance for key/query pairs:
      - pick top-n value features by magnitude,
      - then top head queries per value by magnitude
      - then use gradients to score interactions.
    """
    tokens = torch.tensor(train_df["token_ids"].to_list())
    if "row_idx" not in train_df.columns:
        train_df["row_idx"] = np.arange(len(train_df))
    positions = train_df["act_argmax"].to_numpy()


    rule_df = rule_utils.get_skipgrams_by_magnitude(
        feat=feat,
        sae_in=sae_in,
        sae_out=sae_out,
        model=model,
        num_values=num_values,
        num_queries=num_queries,
        explain_df=explain_df,
        head=head,
        layer=layer,
        negative_values=negative_values,
    )
    query_idxs = rule_df["query"].unique()
    key_idxs = value_idxs = rule_df["key"].unique()
    logger.info(f"{len(key_idxs)} keys, {len(query_idxs)} queries")

    logger.info(f"Got grads, getting interactions")
    interactions, idxs = rule_utils.get_kq_interactions_v5(
        tokens=tokens,
        model=model,
        sae_in=sae_in,
        sae_out=sae_out,
        layer=layer,
        head=head,
        feat=feat,
        key_idxs=key_idxs,
        query_idxs=query_idxs,
        value_idxs=value_idxs,
        batch_size=batch_size,
        device=device,
        eval_positions=positions,
        loss_type=loss_type,
    )

    return interactions, idxs


def precision(row):
    # If there are no positive predictions, define precision = 1.
    if row["TP"] + row["FP"] == 0:
        return 1
    return row["TP"] / (row["TP"] + row["FP"])

def recall(row):
    # If there are no positive examples, define recall = 1.
    if row["TP"] + row["FN"] == 0:
        return 1
    return row["TP"] / (row["TP"] + row["FN"])

def add_binary_score(df, threshold=0):
    df["is_active"] = df["act_max"] > threshold
    df["predicted_active"] = df["predicted_score"] > threshold
    df["TP"] = df["is_active"] & df["predicted_active"]
    df["FP"] = ~df["is_active"] & df["predicted_active"]
    df["FN"] = df["is_active"] & ~df["predicted_active"]
    df["TN"] = ~df["is_active"] & ~df["predicted_active"]
    metrics = df[["is_active", "predicted_active", "TP", "FP", "FN"]].sum().to_dict()
    metrics["precision"] = precision(metrics)
    metrics["recall"] = recall(metrics)
    if metrics["precision"] + metrics["recall"] == 0:
        metrics["f1"] = 0
    else:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    return df, metrics


def run_rule_for_feature(
    feat,
    model,
    sae_in,
    sae_out,
    train_df,
    eval_df,
    explain_df,
    head,
    layer=0,
    method="magnitude",
    num_values=100,
    num_queries=100,
    max_rules=4096,
    mask_val=1.0,
    loss_type="activation",
    use_seq_mask=False,
    absolute=True,
    aggregation_type="max",
    negative_values=False,
    batch_size=64,
    threshold=0,
    qv_only=0,
):
    if method == "magnitude":
        rule_df = rule_utils.get_skipgrams_by_magnitude(
            feat=feat,
            sae_in=sae_in,
            sae_out=sae_out,
            model=model,
            num_values=num_values,
            num_queries=num_queries,
            explain_df=explain_df,
            head=head,
            layer=layer,
            negative_values=negative_values,
        )
        rule_df = rule_df.sort_values(by="score", ascending=False)
    elif method == "importance":
        interactions, idxs = rule_utils.get_skipgram_interaction_importance(
            feat=feat,
            model=model,
            sae_in=sae_in,
            sae_out=sae_out,
            train_df=train_df,
            head=head,
            layer=layer,
            batch_size=batch_size,
            loss_type=loss_type,
            num_values=num_values,
            num_queries=num_queries,
            explain_df=explain_df,
            negative_values=negative_values,
        )
    elif method == "unigram":
        rule_df = rule_utils.get_unigrams_by_magnitude(
            feat=feat,
            sae_in=sae_in,
            sae_out=sae_out,
            model=model,
            num_values=num_values,
            explain_df=explain_df,
            head=head,
            layer=layer,
            negative_values=negative_values,
        )
        rule_df = rule_df.sort_values(by="score", ascending=False)
    else:
        raise NotImplementedError(method)

    eval_tokens = torch.tensor(eval_df["token_ids"].to_list())
    if "row_idx" not in eval_df.columns:
        eval_df["row_idx"] = np.arange(len(eval_df))
    eval_positions = eval_df["act_argmax"].to_numpy()
    eval_idxs = eval_df["idx"]

    lst = []
    corr = []
    pred_scores = []
    max_base = int(np.log2(max_rules))
    ks = [2**i for i in range(2, max_base + 1)]
    for k in ks:
        if method == "magnitude":
            rule = rule_utils.SkipgramRule(rule_df.nlargest(k, ["score"]), feat)
        elif method == "importance":
            rule = rule_utils.SkipgramRuleWithImportance(
                model=model,
                sae_in=sae_in,
                sae_out=sae_out,
                layer=layer,
                head=head,
                feat=feat,
                key_idxs=idxs["keys"],
                query_idxs=idxs["queries"],
                value_idxs=idxs["values"][:k],
                interactions=interactions,
                num_interactions=k,
                absolute=absolute,
            )
        elif method == "unigram":
            rule = rule_utils.UnigramRule(rule_df.nlargest(k, ["score"]), feat)
        else:
            raise NotImplementedError(method)
        df = rule_utils.run_eval(
            rule,
            model,
            sae_in,
            eval_tokens,
            eval_positions,
            eval_idxs,
            per_head=False,
            progress=False,
        )
        df["num_rules"] = k
        lst.append(df)
        predicted_scores = rule_utils.aggregate_predictions(
            eval_df, df, aggregation_type=aggregation_type
        )
        predicted_scores["num_rules"] = k
        predicted_scores["aggregation"] = aggregation_type
        predicted_scores, metrics = add_binary_score(predicted_scores, threshold=threshold)
        pred_scores.append(predicted_scores)
        stat = stats.spearmanr(
            predicted_scores["act_max"], predicted_scores["predicted_score"]
        )
        metrics.update(
            {
                "num_rules": df["num_rules"].iloc[0],
                "spearmanr": stat.statistic,
                "pvalue": stat.pvalue,
                "aggregation": aggregation_type,
            }
        )
        corr.append(metrics)

    if method in ("magnitude", "unigram"):
        rule_df = rule_df.nlargest(ks[-1], ["score"])
    elif method == "importance":
        rule_df = rule.get_rule_df(explain_df)

    eval_df = pd.concat(lst)
    corr_df = pd.DataFrame(corr)
    pred_df = pd.concat(pred_scores)
    corr_df["layer"] = layer
    corr_df["head"] = head
    corr_df["feature"] = feat
    rule_df["feature"] = feat
    eval_df["feature"] = feat

    logger.info(
        f"Feature {feat}, correlations: {corr_df['spearmanr'].to_list()} "
        f"precision: {corr_df['precision'].to_list()} "
        f"recall: {corr_df['recall'].to_list()} "
    )

    return rule_df, eval_df, corr_df, pred_df


def run_rules(args):
    device = torch.device(args.device)

    logger.info(f"Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_path, device=device)

    logger.info(f"Loading SAEs...")

    sae_in, sae_out, explanation_df = load_saes(
        model,
        layer=args.layer,
        head=args.head,
        sae_in_release=args.sae_in,
        sae_out_path=args.sae_out_path,
        device=args.device,
    )

    train_df, val_df, _ = load_examples(args.example_dir, args.layer, args.head)
    features = train_df["feature"].unique()[: args.num_features]

    out = []
    for feat in features:
        try:
            rule_df, eval_df, corr_df, pred_df = run_rule_for_feature(
                feat=feat,
                model=model,
                sae_in=sae_in,
                sae_out=sae_out,
                train_df=train_df.query(f"feature == {feat}").copy(deep=True),
                eval_df=val_df.query(f"feature == {feat}").copy(deep=True),
                explain_df=explanation_df,
                head=args.head,
                layer=args.layer,
                method=args.method,
                num_values=args.num_values,
                num_queries=args.num_queries,
                max_rules=args.max_rules,
                mask_val=args.mask_val,
                loss_type=args.loss_type,
                use_seq_mask=args.use_seq_mask,
                absolute=args.absolute,
                aggregation_type=args.aggregation_type,
                negative_values=args.negative_values,
                batch_size=args.batch_size,
                threshold=args.threshold,
                qv_only=args.qv_only,
            )
            out.append((rule_df, eval_df, corr_df, pred_df))
        except Exception as e:
            raise e

    if not out:
        return

    rule_df, eval_df, corr_df, pred_df = [pd.concat([r[i] for r in out]) for i in range(4)]

    fn = Path(args.output_dir) / "rules.csv"
    logger.info(f"Writing {len(rule_df)} rules to {fn}")
    rule_df.to_csv(fn)

    fn = Path(args.output_dir) / "corr.csv"
    logger.info(f"Writing {len(corr_df)} results to {fn}")
    corr_df.to_csv(fn)

    if args.save_eval:
        fn = Path(args.output_dir) / "eval.csv"
        logger.info(f"Writing {len(eval_df)} rows to {fn}")
        eval_df.to_csv(fn)

    if args.save_preds:
        fn = Path(args.output_dir) / "preds.csv"
        logger.info(f"Writing {len(pred_df)} rows to {fn}")
        pred_df.to_csv(fn)


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
    if args.cmd == "run_rules":
        run_rules(args)
    else:
        raise NotImplementedError(args.cmd)
