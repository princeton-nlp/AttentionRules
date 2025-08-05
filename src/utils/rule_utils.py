from functools import partial
import html

import einops
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from src.utils.sae_utils import explain
from src.utils import logging_utils

logger = logging_utils.get_logger(__name__)


def get_skipgrams_by_magnitude(
    feat,
    sae_in,
    sae_out,
    model,
    layer=0,
    head=2,
    num_values=10,
    num_queries=10,
    explain_df=None,
    negative_values=False,
):
    W_enc = sae_out.W_enc
    if len(W_enc.shape) == 3:
        # Not a per-head SAE.
        W_enc = W_enc.view(12, 64, -1)[head]

    W_dec = sae_in.W_dec
    W_K = model.blocks[layer].attn.W_K[head]
    W_Q = model.blocks[layer].attn.W_Q[head]
    W_V = model.blocks[layer].attn.W_V[head]
    D_K = W_dec @ W_K
    D_Q = W_dec @ W_Q
    D_V = W_dec @ W_V

    # Get input features with max scoring values
    feat_emb = W_enc[:, feat]
    scores = D_V @ feat_emb
    if negative_values:
        scores = torch.abs(scores)
    _, top_values = torch.topk(scores, k=num_values, largest=True)

    # For each value, find the top scoring (key, query) pairs
    keys = D_K[top_values[:num_values]]
    attn_scores, top_queries = torch.topk(
        keys @ D_Q.T, k=num_queries, largest=True, dim=-1
    )

    # Make a list
    lst = []
    for i, k_idx in enumerate(top_values):
        for j, q_idx in enumerate(top_queries[i]):
            d = {
                "key": k_idx.item(),
                "query": q_idx.item(),
                "attn_score": attn_scores[i, j].item(),
                # "value_score": value_scores[i].item(),
                "value_score": scores[k_idx].item(),
            }
            # Heuristic score
            d["score"] = np.exp(d["attn_score"] + 1e-10) * d["value_score"]
            lst.append(d)

    rules = pd.DataFrame(lst)
    if explain_df is not None:
        rules["key_desc"] = [explain(f, explain_df) for f in rules["key"]]
        rules["query_desc"] = [explain(f, explain_df) for f in rules["query"]]
    return rules


def get_skipgram_interaction_importance(
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
    explain_df=None,
    negative_values=False,
    device=torch.device("cuda")
):
    """
    Instead of calculating per-feature importance:
      - pick top-n value features by magnitude,
      - then top head queries per value by magnitude
      - then use gradients to score interactions.
    """
    tokens = torch.tensor(train_df["token_ids"].to_list())
    if "row_idx" not in train_df.columns:
        train_df["row_idx"] = np.arange(len(train_df))
    positions = train_df["act_argmax"].to_numpy()

    rule_df = get_skipgrams_by_magnitude(
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
    return get_kq_interaction_importance(
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


def get_unigrams_by_magnitude(
    feat,
    sae_in,
    sae_out,
    model,
    layer=0,
    head=2,
    num_values=10,
    explain_df=None,
    negative_values=False,
):
    W_enc = sae_out.W_enc
    if len(W_enc.shape) == 3:
        # Not a per-head SAE.
        W_enc = W_enc.view(12, 64, -1)[head]

    W_V = model.blocks[layer].attn.W_V[head]
    W_dec = sae_in.W_dec
    D_V = W_dec @ W_V

    # Get input features with max scoring values
    feat_emb = W_enc[:, feat]
    scores = D_V @ feat_emb
    if negative_values:
        scores = torch.abs(scores)
    _, top_values = torch.topk(scores, k=num_values, largest=True)

    # Make a list
    lst = []
    for i, v_idx in enumerate(top_values):
        d = {
            "value": v_idx.item(),
            "value_score": scores[v_idx].item(),
        }
        # Heuristic score
        d["score"] = d["value_score"]
        lst.append(d)

    rules = pd.DataFrame(lst)
    if explain_df is not None:
        rules["value_desc"] = [explain(f, explain_df) for f in rules["value"]]
    return rules



def get_all_activations(
    feat,
    model,
    sae,
    tokens,
    batch_size=8,
    max_batches=None,
    head=None,
):
    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)
    out = {"idx": [], "token": [], "position": [], "act": []}
    if head is not None:
        w = sae.W_enc.view(12, 64, -1)[head, :, feat]
    else:
        print(f"Warning: didn't provide head")
        w = sae.W_enc[:, feat]
    positions = np.concatenate([np.arange(tokens.shape[1])] * batch_size)
    idxs = np.arange(batch_size, dtype=int)[:, np.newaxis] @ np.ones(
        (1, tokens.shape[1]), dtype=int
    )
    for idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        _, cache = model.run_with_cache(
            batch.to(torch.device("cuda")),
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name],
        )
        if head is not None:
            sae_in = cache[sae.cfg.hook_name][:, :, head]
        else:
            sae_in = cache[sae.cfg.hook_name].reshape(
                batch.shape[0], batch.shape[1], -1
            )
        # TODO: Incorporate processing (https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L398)
        # TODO: Incorporate bias
        feature_acts = (sae_in @ w).cpu().numpy().flatten()
        wds = model.to_str_tokens(batch.cpu().numpy().flatten())
        out["idx"].append(((idx * batch_size) + idxs[: batch.shape[0]]).flatten())
        out["token"].append(wds)
        out["position"].append(positions[: len(wds)])
        out["act"].append(feature_acts)
        if max_batches and idx >= max_batches:
            break
    return pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})


def get_idxs_with_activations(activations, n=16, bins=8, seed=0):
    max_activations = activations.groupby("idx")[["act"]].agg(["max", np.argmax])
    max_activations.columns = max_activations.columns.to_flat_index().map("_".join)
    max_activations["bin"], bins = pd.cut(
        max_activations["act_max"], bins=bins, labels=False, retbins=True
    )
    min_size = max_activations.groupby("bin").size().min()
    if min_size < n:
        print(f"Warning: smallest bin has size {min_size} < {n}")
        n = min_size
    df = max_activations.groupby("bin").sample(n=n, random_state=seed)
    return df, bins


def get_max_activations(
    feats,
    model,
    sae,
    tokens,
    batch_size=8,
    max_batches=None,
    head=None,
    per_head=False,
    post_act=False,
):
    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)
    out = []
    if head is None:
        print(f"Warning: didn't provide head")
    if per_head:
        W = sae.W_enc[:, feats]
        b = sae.b_enc[feats]
    else:
        W = sae.W_enc.view(12, 64, -1)[head, :, feats]
        b = sae.b_enc[feats]
    idxs = np.arange(batch_size, dtype=int)
    for idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        _, cache = model.run_with_cache(
            batch.to(torch.device("cuda")),
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae.cfg.hook_name],
        )
        if head is not None:
            sae_in = cache[sae.cfg.hook_name][:, :, head]
        else:
            sae_in = cache[sae.cfg.hook_name].reshape(
                batch.shape[0], batch.shape[1], -1
            )
        if post_act:
            hidden_pre = sae.process_sae_in(sae_in) @ W + b
            feature_acts = sae.activation_fn(hidden_pre).permute(
                2, 0, 1
            )  # (B, N, D) -> (D, B, N)
        else:
            feature_acts = (sae_in @ W).permute(2, 0, 1)  # (B, N, D) -> (D, B, N)
        top_acts = torch.max(feature_acts, dim=-1)
        max_acts = top_acts.values  # (D, B)
        argmax_acts = top_acts.indices  # (D, B)
        for i, f in enumerate(feats):
            out.append(
                pd.DataFrame(
                    {
                        "feature": f,
                        "idx": (idx * batch_size) + idxs[: batch.shape[0]],
                        "act_max": max_acts[i].cpu().numpy(),
                        "act_argmax": argmax_acts[i].cpu().numpy(),
                    }
                )
            )
        if max_batches and idx >= max_batches:
            break
    return pd.concat(out)


def get_feature_acts_for_batch(model, sae, batch, per_head=False):
    _, cache = model.run_with_cache(
        batch.to(torch.device("cuda")),
        stop_at_layer=sae.cfg.hook_layer + 1,
        names_filter=[sae.cfg.hook_name],
    )
    sae_in = cache[sae.cfg.hook_name]
    if per_head:
        sae_in = sae_in[:, :, sae.cfg.hook_head_index]
    feature_acts = sae.encode(sae_in)
    return feature_acts


def get_dfa_for_batch(feat, model, sae, eval_tokens, eval_positions):
    layer = sae.cfg.hook_layer
    head = sae.cfg.hook_head_index
    with torch.no_grad():
        _, cache = model.run_with_cache(
            eval_tokens.to(torch.device("cuda")),
            stop_at_layer=layer + 1,
            names_filter=[
                f"blocks.{layer}.attn.hook_v",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.attn.hook_z",
            ],
        )
    values = cache[f"blocks.{layer}.attn.hook_v"][:, :, head]
    attn = cache[f"blocks.{layer}.attn.hook_pattern"][:, head]
    w = sae.W_enc[:, feat]
    value_scores = values @ w
    sae_in = cache[f"blocks.{layer}.attn.hook_z"][:, :, head]
    # all_acts = sae.encode(sae_in)
    feature_acts = sae.encode(sae_in)[:, :, feat]
    # print(sae_in.shape, all_acts.shape, feature_acts.shape)
    rows = []
    for i, j in enumerate(eval_positions):
        # vs = value_scores[i, : j + 1]
        # a = attn[i, j, : j + 1]
        # weighted_vs = vs * a
        # toks = model.to_str_tokens(eval_tokens[i, : j + 1])
        # acts = feature_acts[i, : j + 1]
        vs = value_scores[i]
        a = attn[i, j]
        weighted_vs = vs * a
        toks = model.to_str_tokens(eval_tokens[i])
        tok_ids = eval_tokens[i].cpu().tolist()
        acts = feature_acts[i]
        rows.append(
            {
                "unweighted_scores": vs.cpu().tolist(),
                "attn": a.cpu().tolist(),
                "scores": weighted_vs.cpu().tolist(),
                "feature_acts": acts.cpu().tolist(),
                "tokens": toks,
                "token_ids": tok_ids,
                "act_argmax": j.item() if type(j) != int else j,
            }
        )
    return rows


def get_dfa_for_batches(feat, model, sae, eval_tokens, eval_positions, batch_size=64):
    out = []
    for i in range((len(eval_tokens) // batch_size) + 1):
        batch_tokens = eval_tokens[i * batch_size : (i + 1) * batch_size]
        batch_positions = eval_positions[i * batch_size : (i + 1) * batch_size]
        out += get_dfa_for_batch(feat, model, sae, batch_tokens, batch_positions)
    for i, d in enumerate(out):
        d["idx"] = i
        d["feature"] = feat
        d["act_max"] = d["act_out"] = d["feature_acts"][d["act_argmax"]]
    return out


def get_dfa_html(
    scores, toks, escape=True, blue=True, pos=None, include_negative=False
):
    lst = ["<div>"]
    if pos is None:
        pos = len(scores)
    scores, toks = scores[:pos], toks[:pos]
    max_score = max(scores)
    if max_score > 0:
        scores = np.array(scores) / max_score
    else:
        scores = np.array(scores)
    esc = html.escape if escape else lambda s: s
    for tok, s in zip(toks, scores):
        if s >= 0 and blue:
            lst.append(
                f"<span style='background-color: rgba(171, 171, 255, {s})'>{esc(tok)}</span>"
            )
        elif s >= 0:
            lst.append(
                f"<span style='background-color: rgba(255, 150, 100, {s})'>{esc(tok)}</span>"
            )
        elif s < 0 and include_negative and blue:
            lst.append(
                f"<span style='background-color: rgba(255, 150, 100, {-s})'>{esc(tok)}</span>"
            )
        elif s < 0 and include_negative:
            lst.append(
                f"<span style='background-color: rgba(171, 171, 255, {-s})'>{esc(tok)}</span>"
            )
        else:
            lst.append(
                f"<span style='background-color: rgba(255, 149, 149, {s})'>{esc(tok)}</span>"
            )
    lst.append("</div>")
    return "".join(lst)


def get_dfa_latex(
    scores, toks, escape=True, blue=True, pos=None, include_negative=False, max_tokens=None, delim="", do_bracket=False, min_score=0.0
):
    lst = []
    if pos is None:
        pos = len(scores)
    scores, toks = scores[:pos], toks[:pos]
    if max_tokens:
        scores, toks = scores[-max_tokens:], toks[-max_tokens:]
    max_score = max(scores)
    if max_score > 0:
        scores = np.array(scores) / max_score
    else:
        scores = np.array(scores)
    esc = html.escape if escape else lambda s: s
    bracket = lambda s: "{" + s + "}" if do_bracket else s
    for tok_, s in zip(toks, scores):
        tok = esc(tok_)
        if round(s*100) == min_score:
            lst.append(tok)
        elif s > min_score and blue:
            lst.append(bracket(f"{delim}\\blueword{{{tok}}}{{{round(s * 100)}}}"))
        elif s > min_score:
            lst.append(bracket(f"{delim}\\orangeword{{{tok}}}{{{round(s * 100)}}}"))
        elif s < min_score and include_negative and blue:
            lst.append(bracket(f"{delim}\\blueword{{{tok}}}{{{round(-s * 100)}}}"))
        elif s < min_score and include_negative:
            lst.append(bracket(f"{delim}\\orangeword{{{tok}}}{{{round(-s * 100)}}}"))
        else:
            lst.append(tok)
    return "".join(lst).strip()

def get_activation_frequencies(
    model,
    sae,
    eval_tokens,
    batch_size=8,
    progress=True,
    per_head=False,
    normalize=False,
    per_seq=False,
    return_avg_acts=False,
):
    data_loader = DataLoader(
        eval_tokens,
        batch_size=batch_size,
        shuffle=False,
    )
    t = tqdm(data_loader, total=len(data_loader)) if progress else data_loader
    d_sae = sae.cfg.d_sae
    counts = torch.zeros(d_sae, device=sae.device)
    sums = torch.zeros(d_sae, device=sae.device)
    total = 0
    for batch in t:
        feature_acts = get_feature_acts_for_batch(model, sae, batch, per_head=per_head)
        if per_seq:
            counts += (feature_acts.sum(1) > 0).sum(0)
            total += batch.shape[0]
        else:
            counts += (feature_acts > 0).view(-1, d_sae).sum(0)
            total += batch.view(-1).shape[0]
        sums += feature_acts.view(-1, d_sae).sum(0)
    print(f"total = {total}")
    if normalize:
        return (counts / total).cpu().numpy()
    if return_avg_acts:
        counts = counts.cpu().numpy()
        sums = sums.cpu().numpy()
        return counts, sums / np.maximum(counts, 1e-9)
    return counts.cpu().numpy()


class SkipgramRule:
    def __init__(
        self, rule_df, feat=None, with_scores=False, device=torch.device("cuda")
    ):
        self.feat = feat
        self.rule_df = rule_df
        self.queries = {}
        self.query_idxs = rule_df["query"].unique()
        self.key_idxs = rule_df["key"].unique()
        self.scores = None
        if with_scores:
            self.scores = torch.zeros(
                len(self.key_idxs), len(self.query_idxs), device=device
            )
        for _, row in rule_df.iterrows():
            keys = self.queries.setdefault(row["query"], {})
            keys[row["key"]] = {
                k: row[k] for k in ("attn_score", "value_score", "score")
            }
            if with_scores:
                self.scores[row["key"], row["query"]] = row["score"]

    def eval_at_position(self, tokens, feature_acts, pos):
        # print(feature_acts.shape)
        query_feats = feature_acts[pos].nonzero().squeeze(1).cpu().tolist()
        # print(query_feats)
        all_feats = feature_acts[: pos + 1].nonzero().cpu().tolist()
        lst = []
        for q_feat in query_feats:
            ks = self.queries.get(q_feat, {})
            if not ks:
                continue
            for k_pos, k_feat in all_feats:
                if k_feat in ks:
                    d = {
                        "q_pos": pos,
                        "k_pos": k_pos,
                        "q_token": tokens[pos],
                        "k_token": tokens[k_pos],
                        "q": q_feat,
                        "k": k_feat,
                        "q_act": feature_acts[pos, q_feat].item(),
                        "k_act": feature_acts[k_pos, k_feat].item(),
                        "kind": "kq",
                    }
                    d.update(ks[k_feat])
                    d["predicted_score"] = (
                        d["q_act"] * d["k_act"] * d["attn_score"]
                    ) * (d["k_act"] * d["value_score"])
                    d["predicted_kv_score"] = d["predicted_score"]
                    lst.append(d)
        if len(lst) == 0:
            lst = [
                {
                    "q_pos": pos,
                    "k_pos": pos,
                    "q_token": tokens[pos],
                    "predicted_score": 0,
                    "predicted_kv_score": 0,
                    "kind": "kq",
                }
            ]
        return pd.DataFrame(lst)

    def eval_for_sequence(self, tokens, feature_acts, pos=None):
        query_acts = feature_acts[:, self.query_idxs]
        key_acts = feature_acts[:, self.key_idxs]
        scores = torch.einsum("nq,mk->nmqk", key_acts, query_acts) * self.scores
        n, m = scores.shape[:2]
        mask = torch.tril(torch.ones(n, m, dtype=bool, device=scores.device)).view(
            n, m, 1, 1
        )
        scores = scores.masked_fill(mask, 0)
        max_score = scores.view(n, -1).max(-1)
        return max_score


class UnigramRule:
    def __init__(
        self, rule_df, feat=None, any_position=False, device=torch.device("cuda")
    ):
        self.feat = feat
        self.rule_df = rule_df
        self.value_idxs = rule_df["value"].unique()
        self.scores = None
        self.values = {}
        for _, row in rule_df.iterrows():
            self.values[row["value"]] = row["score"]
        self.any_position = any_position


    def eval_at_position(self, tokens, feature_acts, pos):
        all_feats = feature_acts[: pos + 1].nonzero().cpu().tolist()
        lst = []
        for k_pos, k_feat in all_feats:
            if k_feat in self.values:
                d = {
                    "q_pos": pos,
                    "k_pos": k_pos,
                    "q_token": tokens[pos],
                    "k_token": tokens[k_pos],
                    "k": k_feat,
                    "k_act": feature_acts[k_pos, k_feat].item(),
                    "value_score": self.values[k_feat],
                    "kind": "value",
                    "score": self.values[k_feat],
                }
                d["predicted_score"] = d["k_act"] * d["value_score"]
                lst.append(d)
        if len(lst) == 0:
            lst = [
                {
                    "q_pos": pos,
                    "k_pos": pos,
                    "q_token": tokens[pos],
                    "kind": "value",
                    "predicted_score": 0,
                }
            ]
        return pd.DataFrame(lst)


def patch_hook(activation, hook, patch):
    return patch


def save_hook(activation, hook, cache):
    cache[hook.name] = activation
    return activation


def get_kqv_feature_importance_for_batch(
    batch,
    model,
    sae_in,
    sae_out,
    masks,
    layer,
    head,
    feat,
    seq_mask=None,
    loss_type="reconstruction",
):
    per_head = (len(sae_out.W_enc.shape) == 2) and (
        sae_out.W_enc.shape[0] == model.cfg.d_head
    )

    # `masks.shape` should be (3, 1, 1, d_sae)
    with torch.no_grad():
        # Get the inputs to the input SAE.
        _, cache = model.run_with_cache(
            batch.to(torch.device("cuda")),
            stop_at_layer=sae_in.cfg.hook_layer + 1,
            names_filter=[sae_in.cfg.hook_name],
        )

        # Get the input feature activations
        feature_acts = sae_in.encode(cache[sae_in.cfg.hook_name])

        # Also calculate the original output feature activations (for calculating reconstruction later).
        # Do this separately because adding the SAE to the model somehow gets in the way of the hook point.
        if loss_type == "reconstruction":
            with model.saes(saes=[sae_in], use_error_term=False, reset_saes_end=True):
                _, cache = model.run_with_cache(
                    batch,
                    stop_at_layer=sae_out.cfg.hook_layer + 1,
                    names_filter=[sae_out.cfg.hook_name],
                )

                z = cache[sae_out.cfg.hook_name][:, :, head]
                if per_head:
                    w = sae_out.W_enc[:, feat]
                else:
                    w = sae_out.W_enc.view(model.cfg.n_heads, model.cfg.d_head, -1)[
                        head, :, feat
                    ]
                original_feature_acts = z @ w

    # Multiply input activations by mask
    masked_feature_acts = masks * feature_acts

    # Get outputs and repeat by head (see transformer_lens.utils.repeat_along_head_dimension).
    masked_outputs = einops.repeat(
        sae_in.decode(masked_feature_acts),
        "n batch pos d_model -> n batch pos n_heads d_model",
        n_heads=model.cfg.n_heads,
    )

    # Run the model again, with two hooks:
    # 1) Replace key/query/value input with the masked input.
    # 2) Get the new attention output
    cache = {}
    with model.saes(saes=[sae_in], use_error_term=False, reset_saes_end=True):
        _ = model.run_with_hooks(
            batch,
            fwd_hooks=[
                (
                    f"blocks.{layer}.hook_k_input",
                    partial(patch_hook, patch=masked_outputs[0]),
                ),
                (
                    f"blocks.{layer}.hook_q_input",
                    partial(patch_hook, patch=masked_outputs[1]),
                ),
                (
                    f"blocks.{layer}.hook_v_input",
                    partial(patch_hook, patch=masked_outputs[2]),
                ),
                (sae_out.cfg.hook_name, partial(save_hook, cache=cache)),
            ],
            stop_at_layer=sae_out.cfg.hook_layer + 1,
        )

    # Calculate the feature activation using the attention output.
    z = cache[sae_out.cfg.hook_name][:, :, head]
    if per_head:
        w = sae_out.W_enc[:, feat]
        b = sae_out.b_enc[feat]
    else:
        w = sae_out.W_enc.view(model.cfg.n_heads, model.cfg.d_head, -1)[head, :, feat]
        b = sae_out.W_enc.view(model.cfg.n_heads, model.cfg.d_head)[head, feat]
    new_feature_acts = z @ w

    if loss_type == "reconstruction":
        losses = F.mse_loss(original_feature_acts, new_feature_acts, reduction="none")
    elif loss_type == "activation":
        losses = new_feature_acts
    elif loss_type == "post_activation":
        loss = F.relu(new_feature_acts + b)
    else:
        raise NotImplementedError(loss_type)

    if seq_mask is not None:
        losses = losses.masked_fill(~seq_mask, 0)
        loss = losses.sum() / seq_mask.sum()
    else:
        loss = losses.mean()

    return loss


def get_kqv_feature_importance(
    tokens,
    model,
    sae_in,
    sae_out,
    layer,
    head,
    feat,
    batch_size=8,
    device=torch.device("cuda"),
    mask_val=1.0,
    loss_type="activation",
    use_seq_mask=False,
    eval_positions=None,
):
    torch.set_grad_enabled(True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    # Needed to get hook_{k,q,v}_input hooks.
    model.cfg.use_split_qkv_input = True

    if eval_positions is None:
        data_loader = DataLoader(
            list(zip(tokens, tokens)), batch_size=batch_size, shuffle=False
        )
    else:
        data_loader = DataLoader(
            list(zip(tokens, eval_positions)), batch_size=batch_size, shuffle=False
        )
    masks = nn.Parameter(
        torch.ones(3, 1, 1, sae_in.cfg.d_sae, device=device) * mask_val
    )

    for batch, pos in (pbar := tqdm(data_loader, total=len(data_loader))):
        seq_mask = None
        if use_seq_mask:
            seq_mask = torch.zeros_like(batch, dtype=bool, device=device)
            seq_mask[np.arange(seq_mask.shape[0]), pos] = True
        loss = get_kqv_feature_importance_for_batch(
            batch=batch.to(device),
            model=model,
            sae_in=sae_in,
            sae_out=sae_out,
            masks=masks,
            layer=layer,
            head=head,
            feat=feat,
            loss_type=loss_type,
            seq_mask=seq_mask,
        )
        pbar.set_postfix({"loss": loss.cpu().item()})
        loss.backward()

    model.cfg.use_split_qkv_input = False
    torch.set_grad_enabled(False)
    return masks.grad.squeeze().detach()


def get_combined_feature_importance_for_batch(
    batch, model, sae_in, sae_out, mask, layer, head, feat
):
    # `mask.shape` should be (d_sae,)
    with torch.no_grad():
        # Get the inputs to the input SAE.
        _, cache = model.run_with_cache(
            batch.to(torch.device("cuda")),
            stop_at_layer=sae_in.cfg.hook_layer + 1,
            names_filter=[sae_in.cfg.hook_name],
        )

        # Get the input feature activations
        feature_acts = sae_in.encode(cache[sae_in.cfg.hook_name])

        # Also calculate the original output feature activations (for calculating reconstruction later).
        # Do this separately because adding the SAE to the model somehow gets in the way of the hook point.
        with model.saes(saes=[sae_in], use_error_term=False, reset_saes_end=True):
            _, cache = model.run_with_cache(
                batch,
                stop_at_layer=sae_out.cfg.hook_layer + 1,
                names_filter=[sae_out.cfg.hook_name],
            )

            z = cache[sae_out.cfg.hook_name][:, :, head]
            w = sae_out.W_enc.view(model.cfg.n_heads, model.cfg.d_head, -1)[
                head, :, feat
            ]
            original_feature_acts = z @ w

    # Multiply input activations by mask
    masked_feature_acts = feature_acts * mask

    # Get outputs.
    masked_outputs = sae_in.decode(masked_feature_acts)

    # Run the model again, with two hooks:
    # 1) Replace key/query/value input with the masked input.
    # 2) Get the new attention output
    cache = {}
    _ = model.run_with_hooks(
        batch,
        fwd_hooks=[
            (sae_in.cfg.hook_name, partial(patch_hook, patch=masked_outputs)),
            (sae_out.cfg.hook_name, partial(save_hook, cache=cache)),
        ],
        stop_at_layer=sae_out.cfg.hook_layer + 1,
    )

    # Calculate the feature activation using the attention output.
    z = cache[sae_out.cfg.hook_name][:, :, head]
    w = sae_out.W_enc.view(model.cfg.n_heads, model.cfg.d_head, -1)[head, :, feat]
    new_feature_acts = z @ w

    loss = F.mse_loss(original_feature_acts, new_feature_acts)
    return loss


def get_combined_feature_importance(
    tokens,
    model,
    sae_in,
    sae_out,
    layer,
    head,
    feat,
    batch_size=8,
    device=torch.device("cuda"),
    mask_val=0.5,
):
    torch.set_grad_enabled(True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    model.cfg.use_split_qkv_input = False

    data_loader = DataLoader(tokens, batch_size=batch_size, shuffle=False)
    mask = nn.Parameter(torch.ones(sae_in.cfg.d_sae, device=device) * mask_val)

    for batch in (pbar := tqdm(data_loader, total=len(data_loader))):
        loss = get_combined_feature_importance_for_batch(
            batch=batch.to(device),
            model=model,
            sae_in=sae_in,
            sae_out=sae_out,
            mask=mask,
            layer=layer,
            head=head,
            feat=feat,
        )
        pbar.set_postfix({"loss": loss.cpu().item()})
        loss.backward()

    torch.set_grad_enabled(False)
    return mask.grad.detach()


def get_n_features_from_grads(grads, n, absolute=False):
    grads_ = grads
    if absolute:
        grads = np.abs(grads)
    orders = np.argsort(-grads, axis=1)
    ordered_grads = -np.sort(-grads, axis=1)
    # Positive scores only
    scores = np.maximum(ordered_grads, 0)
    denom = np.maximum(scores.sum(axis=-1, keepdims=True), 1e-30)
    normalized = scores / denom
    pct_covered = np.cumsum(normalized, axis=-1)
    # Binary search...
    l, r = 0.0, 1.0
    for _ in range(8):
        p = (l + r) / 2
        count = (((pct_covered < p).sum(-1) + 1) * (normalized.sum(-1) > 0)).sum()
        if count == n:
            break
        # Not enough features -> increase target pct
        # Too many features -> decrease target pct
        if count > n:
            r = p
        else:
            l = p
    p = (l + r) / 2
    keep_mask = pct_covered < p
    idxs = [o[m] for o, m in zip(orders, keep_mask)]
    gs = [g[i] for i, g in zip(idxs, grads_)]
    return p, idxs, gs


def run_eval(
    rule,
    model,
    sae,
    eval_tokens,
    eval_positions,
    eval_idxs,
    batch_size=8,
    progress=True,
    per_head=False,
):
    feat = rule.feat
    assert feat is not None
    data_loader = DataLoader(
        list(zip(eval_tokens, eval_positions, eval_idxs)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    t = tqdm(data_loader, total=len(data_loader)) if progress else data_loader
    for i, (batch, positions, idxs) in enumerate(t):
        feature_acts = get_feature_acts_for_batch(model, sae, batch, per_head=per_head)
        for j, (pos, idx) in enumerate(zip(positions, idxs)):
            df = rule.eval_at_position(
                model.to_str_tokens(batch[j]), feature_acts[j], pos.item()
            )
            df["idx"] = idx.item()
            df["row_idx"] = len(out)
            out.append(df)
    return pd.concat(out)


class SkipgramRuleWithImportance:
    def __init__(
        self,
        model,
        sae_in,
        sae_out,
        layer,
        head,
        feat,
        key_idxs,
        query_idxs,
        value_idxs,
        interactions,
        num_interactions=None,
        zero_fill=None,
        act_type="softmax",
        absolute=True,
    ):
        self.feat = feat
        W_enc = sae_out.W_enc
        if len(W_enc.shape) == 3:
            # Not a per-head SAE.
            W_enc = W_enc.view(12, 64, -1)[head]
        W_dec = sae_in.W_dec
        W_K = model.blocks[layer].attn.W_K[head]
        W_Q = model.blocks[layer].attn.W_Q[head]
        W_V = model.blocks[layer].attn.W_V[head]
        self.KQ = (W_dec[key_idxs] @ W_K) @ (W_dec[query_idxs] @ W_Q).T
        self.D_V = W_dec[value_idxs] @ W_V
        self.w = W_enc[:, feat]
        self.key_idxs = key_idxs
        self.query_idxs = query_idxs
        self.value_idxs = value_idxs
        self.value_scores = self.D_V @ W_enc[:, feat]
        self.kv_scores = W_dec[key_idxs] @ W_V @ W_enc[:, feat]
        self.interactions = interactions
        if num_interactions is not None and num_interactions < len(interactions.view(-1)):
            if absolute:
                # scores = torch.abs(torch.tensor(interactions))
                scores = torch.abs(interactions)
            else:
                scores = F.relu(interactions)
            threshold = torch.topk(scores.view(-1), k=num_interactions + 1).values.min()
            self.mask = (scores > threshold).to(self.KQ.device)
        else:
            self.mask = torch.ones_like(self.interactions, dtype=bool, device=self.KQ.device)
        self.zero_fill = -1e9 if zero_fill else None
        self.act_type = act_type

    def get_rule_df(self, explain_df=None):
        out = []
        for k, q in self.mask.nonzero().cpu().tolist():
            d = {
                "k": int(self.key_idxs[k]),
                "q": int(self.query_idxs[q]),
                "score": self.KQ[k, q].item(),
                "value_score": self.kv_scores[k].item(),
                "kind": "kq",
            }
            if explain_df is not None:
                d["key_desc"] = explain(d["k"], explain_df)
                d["query_desc"] = explain(d["q"], explain_df)
            out.append(d)
        for v, s in zip(self.value_idxs, self.value_scores):
            d = {"v": int(v), "score": s.item(), "kind": "value"}
            if explain_df is not None:
                d["value_desc"] = explain(v, explain_df)
            out.append(d)
        return pd.DataFrame(out)

    def eval_at_position(
        self,
        tokens,
        feature_acts,
        pos,
        mask=None,
        return_scores=False,
        concat_dfs=True,
    ):
        query_feats = feature_acts[pos, self.query_idxs]
        key_feats = feature_acts[: pos + 1, self.key_idxs]
        value_feats = feature_acts[: pos + 1, self.value_idxs]  # @ self.D_V.detach()
        qk_acts = torch.einsum("nk,q->nkq", key_feats, query_feats)
        if mask is not None:
            KQ = (self.KQ.detach() * mask).unsqueeze(0)
        elif self.mask is not None:
            KQ = (self.KQ.detach() * self.mask).unsqueeze(0)
        else:
            KQ = self.KQ.detach().unsqueeze(0)
        scores = qk_acts * KQ  # .view(qk_acts.shape[0], -1).sum(-1)
        if return_scores:
            return scores

        # kq features
        kq_feats = []
        seq_interactions = torch.nonzero(scores)
        for k_pos, k_feat, q_feat in seq_interactions.cpu().numpy():
            d = {
                "q_pos": pos,
                "k_pos": k_pos,
                "q_token": tokens[pos],
                "k_token": tokens[k_pos],
                "q": self.query_idxs[q_feat],
                "k": self.key_idxs[k_feat],
                "q_act": query_feats[q_feat].item(),
                "k_act": key_feats[k_pos, k_feat].item(),
                "predicted_score": scores[k_pos, k_feat, q_feat].item(),
                "predicted_kv_score": (
                    scores[k_pos, k_feat, q_feat] * self.kv_scores[k_feat]
                ).item(),
                "kind": "kq",
            }
            kq_feats.append(d)
        if not kq_feats:
            kq_feats = [
                {
                    "q_pos": pos,
                    "k_pos": pos,
                    "q_token": tokens[pos],
                    "predicted_score": 0,
                    "predicted_kv_score": 0,
                    "kind": "kq",
                }
            ]
            # kq_feats.append({""})

        # value features
        v_feats = []
        for v_pos, v_feat in value_feats.nonzero().cpu().numpy():
            d = {
                "v_pos": v_pos,
                "v_token": tokens[v_pos],
                "v": self.value_idxs[v_feat],
                "v_act": value_feats[v_pos, v_feat].item(),
                "predicted_score": (
                    value_feats[v_pos, v_feat] * self.value_scores[v_feat]
                ).item(),
                "kind": "value",
            }
            v_feats.append(d)
        if not v_feats:
            v_feats = [
                {
                    "v_pos": pos,
                    "v_token": tokens[pos],
                    "predicted_score": 0,
                    "kind": "value",
                }
            ]

        if concat_dfs:
            return pd.concat([pd.DataFrame(kq_feats), pd.DataFrame(v_feats)])
        return pd.DataFrame(kq_feats), pd.DataFrame(v_feats)


class KQInteractions:
    def __init__(
        self,
        model,
        sae_in,
        sae_out,
        layer,
        head,
        feat,
        key_idxs,
        query_idxs,
        value_idxs,
        zero_fill=None,
        act_type="softmax",
        loss_type="activation",
    ):
        self.feat = feat
        W_enc = sae_out.W_enc
        if len(W_enc.shape) == 3:
            # Not a per-head SAE.
            W_enc = W_enc.view(12, 64, -1)[head]
        W_dec = sae_in.W_dec
        W_K = model.blocks[layer].attn.W_K[head]
        W_Q = model.blocks[layer].attn.W_Q[head]
        W_V = model.blocks[layer].attn.W_V[head]
        self.KQ = (W_dec[key_idxs] @ W_K) @ (W_dec[query_idxs] @ W_Q).T
        self.D_V = W_dec[value_idxs] @ W_V
        self.w = W_enc[:, feat]
        self.b = sae_out.b_enc[feat]
        self.key_idxs = key_idxs
        self.query_idxs = query_idxs
        self.value_idxs = value_idxs
        self.zero_fill = -1e9 if zero_fill else None
        self.act_type = act_type
        self.loss_type = loss_type

    def eval_at_position(self, tokens, feature_acts, pos, mask=None, zero_fill=None):
        query_feats = feature_acts[pos, self.query_idxs]
        key_feats = feature_acts[: pos + 1, self.key_idxs]
        value_feats = feature_acts[: pos + 1, self.value_idxs] @ self.D_V.detach()
        qk_acts = torch.einsum("nk,q->nkq", key_feats, query_feats)
        if mask is not None:
            KQ = (self.KQ.detach() * mask).unsqueeze(0)
        else:
            KQ = self.KQ.detach().unsqueeze(0)
        scores = (qk_acts * KQ).view(qk_acts.shape[0], -1).sum(-1)
        z = zero_fill or self.zero_fill
        if z is not None:
            scores = scores.masked_fill(scores <= 0, z)
        if self.act_type == "softmax":
            attn = scores.softmax(-1)
            out = attn @ value_feats
        elif self.act_type == "argmax":
            idx = scores.argmax(-1)
            out = value_feats[idx]
        else:
            raise NotImplementedError(self.act_type)
        pred = out @ self.w
        if self.loss_type == "post_activation":
            pred = F.relu(pred + self.b)
        return pred


def get_kq_interactions_for_batch(
    batch,
    positions,
    model,
    sae_in,
    rule,
    mask,
):
    feature_acts = get_feature_acts_for_batch(model, sae_in, batch)
    batch_ = [batch[j].detach() for j in range(len(positions))]
    feature_acts_ = [feature_acts[j].detach() for j in range(len(positions))]
    for j, pos in enumerate(positions):
        loss = rule.eval_at_position(
            model.to_str_tokens(batch_[j]), feature_acts_[j], pos.item(), mask=mask
        )
        loss.backward()


def get_kq_interactions_for_batch_all_positions(
    batch,
    positions,
    model,
    sae_in,
    rule,
    mask,
):
    feature_acts = get_feature_acts_for_batch(model, sae_in, batch)
    batch_ = [batch[j].detach() for j in range(len(positions))]
    feature_acts_ = [feature_acts[j].detach() for j in range(len(positions))]
    for j in range(len(positions)):
        for pos in range(len(batch_[j])):
            loss = rule.eval_at_position(
                model.to_str_tokens(batch_[j]), feature_acts_[j], pos, mask=mask
            )
            loss.backward()


def get_kq_interactions(
    tokens,
    model,
    sae_in,
    sae_out,
    layer,
    head,
    feat,
    grads,
    num_rules,
    absolute=True,
    batch_size=8,
    device=torch.device("cuda"),
    eval_positions=None,
    all_positions=False,
    loss_type="activation",
    qv_only=False,
):
    torch.set_grad_enabled(True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    if eval_positions is None:
        data_loader = DataLoader(
            list(zip(tokens, tokens)), batch_size=batch_size, shuffle=False
        )
    else:
        data_loader = DataLoader(
            list(zip(tokens, eval_positions)), batch_size=batch_size, shuffle=False
        )

    if qv_only:
        p, idx_lst, grad_lst = get_n_features_from_grads(
            grads[1:], num_rules, absolute=absolute
        )
        idx_lst = [idx_lst[-1]] + idx_lst
        grad_lst = [grad_lst[-1]] + grad_lst
    else:
        p, idx_lst, grad_lst = get_n_features_from_grads(
            grads, num_rules, absolute=absolute
        )
    lens = [len(lst) for lst in idx_lst]
    logger.info(f"Number of features: {lens}, total: {sum(lens)}, pct: {p}")
    key_idxs, query_idxs, value_idxs = idx_lst

    rule = KQInteractions(
        model=model,
        sae_in=sae_in,
        sae_out=sae_out,
        layer=layer,
        head=head,
        feat=feat,
        key_idxs=key_idxs,
        query_idxs=query_idxs,
        value_idxs=value_idxs,
        zero_fill=None,
        act_type="softmax",
        loss_type=loss_type,
    )

    mask = nn.Parameter(
        torch.ones(len(key_idxs), len(query_idxs), device=model.W_E.device)
    )

    for batch, pos in (pbar := tqdm(data_loader, total=len(data_loader))):
        if all_positions:
            get_kq_interactions_for_batch_all_positions(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )
        else:
            get_kq_interactions_for_batch(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )

    torch.set_grad_enabled(False)
    grads = mask.grad.squeeze().detach()
    idxs = {
        "keys": key_idxs.tolist(),
        "queries": query_idxs.tolist(),
        "values": value_idxs.tolist(),
    }
    feature_grads = {
        "keys": grad_lst[0].tolist(),
        "queries": grad_lst[1].tolist(),
        "values": grad_lst[2].tolist(),
    }
    return grads, idxs, feature_grads


def get_kq_interaction_importance(
    tokens,
    model,
    sae_in,
    sae_out,
    layer,
    head,
    feat,
    key_idxs,
    query_idxs,
    value_idxs,
    batch_size=8,
    device=torch.device("cuda"),
    eval_positions=None,
    all_positions=False,
    loss_type="activation",
):
    torch.set_grad_enabled(True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    if eval_positions is None:
        data_loader = DataLoader(
            list(zip(tokens, tokens)), batch_size=batch_size, shuffle=False
        )
    else:
        data_loader = DataLoader(
            list(zip(tokens, eval_positions)), batch_size=batch_size, shuffle=False
        )

    rule = KQInteractions(
        model=model,
        sae_in=sae_in,
        sae_out=sae_out,
        layer=layer,
        head=head,
        feat=feat,
        key_idxs=key_idxs,
        query_idxs=query_idxs,
        value_idxs=value_idxs,
        zero_fill=None,
        act_type="softmax",
        loss_type=loss_type
    )

    mask = nn.Parameter(
        torch.ones(len(key_idxs), len(query_idxs), device=model.W_E.device)
    )

    for batch, pos in (pbar := tqdm(data_loader, total=len(data_loader))):
        if all_positions:
            get_kq_interactions_for_batch_all_positions(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )
        else:
            get_kq_interactions_for_batch(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )

    torch.set_grad_enabled(False)
    grads = mask.grad.squeeze().detach()
    idxs = {
        "keys": key_idxs.tolist(),
        "queries": query_idxs.tolist(),
        "values": value_idxs.tolist(),
    }
    return grads, idxs


def get_kq_interactions_v5(
    tokens,
    model,
    sae_in,
    sae_out,
    layer,
    head,
    feat,
    key_idxs,
    query_idxs,
    value_idxs,
    batch_size=8,
    device=torch.device("cuda"),
    eval_positions=None,
    all_positions=False,
    loss_type="activation",
):
    torch.set_grad_enabled(True)
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()

    if eval_positions is None:
        data_loader = DataLoader(
            list(zip(tokens, tokens)), batch_size=batch_size, shuffle=False
        )
    else:
        data_loader = DataLoader(
            list(zip(tokens, eval_positions)), batch_size=batch_size, shuffle=False
        )

    rule = KQInteractions(
        model=model,
        sae_in=sae_in,
        sae_out=sae_out,
        layer=layer,
        head=head,
        feat=feat,
        key_idxs=key_idxs,
        query_idxs=query_idxs,
        value_idxs=value_idxs,
        zero_fill=None,
        act_type="softmax",
        loss_type=loss_type
    )

    mask = nn.Parameter(
        torch.ones(len(key_idxs), len(query_idxs), device=model.W_E.device)
    )

    for batch, pos in (pbar := tqdm(data_loader, total=len(data_loader))):
        if all_positions:
            get_kq_interactions_for_batch_all_positions(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )
        else:
            get_kq_interactions_for_batch(
                batch=batch.to(device),
                positions=pos,
                model=model,
                sae_in=sae_in,
                rule=rule,
                mask=mask,
            )

    torch.set_grad_enabled(False)
    grads = mask.grad.squeeze().detach()
    idxs = {
        "keys": key_idxs.tolist(),
        "queries": query_idxs.tolist(),
        "values": value_idxs.tolist(),
    }
    return grads, idxs





def aggregate_predictions(binned_idxs, df, aggregation_type="max"):
    if aggregation_type == "kq_max":
        df = df.query("kind == 'kq'")
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "kqv_max":
        df = df.query("kind == 'kq'").copy(deep=True)
        df["predicted_score"] = df["predicted_kv_score"]
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "kq_sum":
        df = df.query("kind == 'kq'")
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].sum(),
            on="row_idx",
        )
    if aggregation_type == "kqv_sum":
        df = df.query("kind == 'kq'").copy(deep=True)
        df["predicted_score"] = df["predicted_kv_score"]
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].sum(),
            on="row_idx",
        )
    if aggregation_type == "value_max":
        df = df.query("kind == 'value'")
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "value_last":
        df = df.query("kind == 'value' & (q_pos == k_pos)")
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "value_sum":
        df = df.query("kind == 'value'")
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].sum(),
            on="row_idx",
        )
    if aggregation_type == "max":
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "sum":
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].sum(),
            on="row_idx",
        )
    if aggregation_type == "pos_sum_max":
        df = df.groupby(["row_idx", "k_pos"])[["predicted_score"]].sum()
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].max(),
            on="row_idx",
        )
    if aggregation_type == "pos_max_sum":
        df = df.groupby(["row_idx", "k_pos"])[["predicted_score"]].max()
        return binned_idxs.join(
            df.groupby("row_idx")[["predicted_score"]].sum(),
            on="row_idx",
        )
    else:
        raise NotImplementedError(aggregation_type)


def find_kkq_in_sequence(kkq_acts, output_acts):
    state = ""
    out = {}
    for i in range(kkq_acts.shape[0]):
        k1, k2, q = kkq_acts[i]
        f = output_acts[i]
        if state in ("", "k2") and k1 > 0:
            state += "k1"
            out["k1_pos"] = i
            out["k1_act"] = k1.item()
        elif state in ("", "k1") and k2 > 0:
            state += "k2"
            out["k2_pos"] = i
            out["k2_act"] = k2.item()
        # q can be at same position as one of the keys
        if state in ("k1", "k1k2", "k2k1") and q > 0:
            state += "q"
            out["q_pos"] = i
            out["q_act"] = q.item()
            out["act_out"] = f.item()
            out["state"] = state
            out["kind"] = "has_distractor" if "k2" in state else "no_distractor"
            break
    return out if "q_act" in out else None


def find_kkq_examples(
    model,
    sae_in,
    sae_out,
    feat,
    k1,
    k2,
    q,
    eval_tokens,
    eval_idxs,
    max_examples=100,
    batch_size=64,
    progress=True,
):
    data_loader = DataLoader(
        list(zip(eval_tokens, eval_idxs)),
        batch_size=batch_size,
        shuffle=False,
    )
    out = []
    t = tqdm(data_loader, total=len(data_loader)) if progress else data_loader
    kq_count = 0
    kkq_count = 0
    for batch, idxs in t:
        feature_acts = get_feature_acts_for_batch(model, sae_in, batch, per_head=False)
        output_acts = get_feature_acts_for_batch(model, sae_out, batch, per_head=True)[
            :, :, feat
        ]
        kkq_acts = feature_acts[:, :, [k1, k2, q]]
        for acts, acts_out, toks, idx in zip(
            kkq_acts,
            output_acts,
            batch,
            idxs.cpu().tolist() if type(idxs) == torch.Tensor else idxs,
        ):
            d = find_kkq_in_sequence(acts, acts_out)
            if d is None:
                continue
            d["feature"] = int(feat)
            d["idx"] = idx
            d["tokens"] = model.to_str_tokens(toks)
            d["token_ids"] = toks.cpu().tolist()
            if d["kind"] == "has_distractor":
                if kkq_count < max_examples:
                    out.append(d)
                kkq_count += 1
            elif d["kind"] == "no_distractor":
                if kq_count < max_examples:
                    out.append(d)
                kq_count += 1
    return out, kq_count, kkq_count


def get_ordered_interactions(interactions, idxs, absolute=True):
    kq_idxs = np.nonzero(interactions)
    scores = interactions[kq_idxs]
    if absolute:
        order = np.argsort(-np.abs(scores))
    else:
        order = np.argsort(-scores)
    ordered_scores = scores[order]
    ordered_idxs = np.transpose(kq_idxs)[order]
    ordered_idxs = np.array(
        [[idxs["keys"][k], idxs["queries"][q]] for k, q in ordered_idxs]
    )
    return ordered_scores, ordered_idxs


def get_distractor_key_for_feature(
    model,
    feat,
    sae_in,
    sae_out,
    head,
    layer,
    explain_df=None,
):
    W_enc = sae_out.W_enc
    W_dec = sae_in.W_dec
    W_V = model.blocks[layer].attn.W_V[head]
    W_K = model.blocks[layer].attn.W_K[head]
    W_Q = model.blocks[layer].attn.W_Q[head]
    D_K = W_dec @ W_K
    D_Q = W_dec @ W_Q
    D_V = W_dec @ W_V

    # Get the input feature with the highest value scores
    value_scores = D_V @ W_enc[:, feat]
    k = value_scores.argmax().item()
    value_score = value_scores[k].item()

    # Get the query with the highest attention score
    attn_scores = D_Q @ D_K[k]
    q = attn_scores.argmax().item()
    attn_score = attn_scores[q].item()

    d = {
        "feature": feat,
        "k": int(k),
        "q": int(q),
        "attn_score": attn_score,
        "value_score": value_score,
    }

    # Find the first k_ with postive (k_, q) score and negative value score
    kq_scores = D_Q[q] @ D_K.T
    top_keys_for_query = torch.argsort(-kq_scores)
    k_rank = torch.where(top_keys_for_query == k)[0][0].item()
    logger.info(f"Rank of positive key: {k_rank}")
    d["k_rank"] = k_rank
    ordered_value_scores = value_scores[top_keys_for_query]
    keys_with_negative_values = torch.where(ordered_value_scores < 0)[0]
    if len(keys_with_negative_values) == 0:
        logger.info(f"No keys with negative values, skipping")
        d["k_"] = None
        d["attn_score_"] = None
        d["value_score_"] = None
    else:
        k_ = top_keys_for_query[keys_with_negative_values[0]].item()
        d["k_"] = k_
        d["attn_score_"] = kq_scores[k_].item()
        d["value_score_"] = value_scores[k_].item()
        d["k_rank_"] = keys_with_negative_values[0].item()

    if explain_df is not None:
        for k in ("k", "q", "k_"):
            if d[k] is None:
                continue
            d[f"{k}_desc"] = explain(d[k], explain_df)

    logger.info(f"Found distractor key: {d}")
    return d

def clean_tokens(tokens):
    # return [tok.replace("�", "").replace("\n", "↵") for tok in tokens]
    return [tok.replace("\n", "↵") for tok in tokens]
