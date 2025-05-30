import json
import requests
from typing import Any, Optional

import einops
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from sae_lens.toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
)  # , read_sae_from_disk
from sae_lens import SAE, SAEConfig


def get_vocab(tokenizer):
    w_idx_ = tokenizer.get_vocab()
    idx_w_ = [""] * (max(w_idx_.values()) + 1)
    for w, i in w_idx_.items():
        idx_w_[i] = w
    idx_w = np.array([s.replace("Ġ", "_") for s in idx_w_])
    w_idx = {w: i for i, w in enumerate(idx_w)}
    return w_idx, idx_w


def get_explanations_df(sae_release="gpt2-small", sae_id="0-res-jb"):
    # From [SAE Lens tutorial](https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb)
    url = f"https://www.neuronpedia.org/api/explanation/export?modelId={sae_release}&saeId={sae_id}"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()
    explanations_df = pd.DataFrame(data)
    # rename index to "feature"
    explanations_df.rename(columns={"index": "feature"}, inplace=True)
    explanations_df["description"] = explanations_df["description"].apply(
        lambda x: x.lower()
    )
    # hack so that we'll prefer explanations from the better model (gpt-4 vs. gpt-3.5)
    return explanations_df.sort_values(by="explanationModelName", ascending=False)


def get_tokens_df(model, sae, k=10, max_pos=128, negatives=False, cache=True):
    cache_fn = Path(f"output/token_df/tok_df_k{k}.csv")
    if cache and cache_fn.exists():
        print(f"Loading token df from {cache_fn}")
        return pd.read_csv(cache_fn, index_col=0)
    W_E = model.W_E
    W_pos = model.W_pos
    if max_pos:
        W_pos = W_pos[:max_pos]
    W_enc = sae.W_enc
    scores = (torch.cat([W_E, W_pos], 0) @ W_enc).T
    # print(scores.shape)
    scores = scores.cpu()
    top_scores, top_idxs = torch.topk(scores, k=k, dim=-1)
    if negatives:
        bottom_scores, bottom_idxs = torch.topk(scores, k=k, dim=-1, largest=False)
    _, idx_w = get_vocab(model.tokenizer)
    idx_w = np.concatenate(
        [idx_w, np.array([f"pos{i}" for i in range(W_pos.shape[0])])], -1
    )
    top_words = idx_w[top_idxs.cpu().numpy()]
    rows = []
    if negatives:
        bottom_words = idx_w[bottom_idxs.cpu().numpy()]
        for i, (t_scores, t_wds, b_scores, b_wds) in enumerate(
            zip(top_scores, top_words, bottom_scores, bottom_words)
        ):
            s = " ".join([f"{w} ({v:.02f})" for w, v in zip(t_wds, t_scores)])
            s += " / " + " ".join([f"{w} ({v:.02f})" for w, v in zip(b_wds, b_scores)])
            rows.append({"feature": str(i), "description": s})
        return pd.DataFrame(rows)
    rows = []
    for i, (scores, wds) in enumerate(zip(top_scores, top_words)):
        s = " ".join([f"{w} ({v:.02f})" for w, v in zip(wds, scores)])
        rows.append({"feature": str(i), "description": s})
    df = pd.DataFrame(rows)
    if cache:
        print(f"Writing {len(df)} rows to {cache_fn}")
        cache_fn.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(cache_fn)
    return df


def get_top_tokens_for_feat(model, sae, feat, max_pos=64, k=1):
    W_E = model.W_E
    W_pos = model.W_pos[:max_pos]
    w = sae.W_enc[:, feat]
    scores = torch.cat([W_E, W_pos], 0) @ w
    top_scores, top_idxs = torch.topk(scores.cpu(), k=k)
    top_scores = top_scores.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()
    _, idx_w = get_vocab(model.tokenizer)
    idx_w = np.concatenate(
        [idx_w, np.array([f"pos{i}" for i in range(W_pos.shape[0])])], -1
    )
    top_words = idx_w[top_idxs]
    return top_words, top_idxs, top_scores


def is_position(s):
    return s.startswith("pos") and s[3:].isnumeric()


def explain(f, df):
    try:
        return df.query(f"feature == '{f}' | feature == {f}")["description"].iloc[0]
    except:
        return "no description"


def get_head_distribution(model, sae, as_df=True):
    # See https://github.com/ckkissane/attention-output-saes/blob/main/common/visualize_utils.py#L794
    n_heads = model.cfg.n_heads
    attn_blocks = einops.rearrange(
        sae.W_dec, "n_feat (n_heads d_heads) -> n_feat n_heads d_heads", n_heads=n_heads
    )
    scores = torch.norm(attn_blocks, dim=-1)
    scores = scores / scores.sum(dim=-1, keepdim=True)
    if not as_df:
        return scores
    rows = []
    for f, row in enumerate(scores):
        d = {"feature": str(f)}
        d.update({h: s for h, s in enumerate(row.cpu().numpy())})
        rows.append(d)
    return pd.DataFrame(rows)


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


def read_sae_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    Adapted from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/toolkit/pretrained_sae_loaders.py.
    """
    if dtype is None:
        dtype = DTYPE_MAP[cfg_dict["dtype"]]

    state_dict = torch.load(weight_path)
    # state_dict = {}
    # with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
    #     for k in f.keys():  # noqa: SIM118
    #         state_dict[k] = f.get_tensor(k).to(dtype=dtype)

    # if bool and True, then it's the April update method of normalizing activations and hasn't been folded in.
    if "scaling_factor" in state_dict:
        # we were adding it anyway for a period of time but are no longer doing so.
        # so we should delete it if
        if torch.allclose(
            state_dict["scaling_factor"],
            torch.ones_like(state_dict["scaling_factor"]),
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict["scaling_factor"]
            del state_dict["scaling_factor"]
    else:
        # it's there and it's not all 1's, we should use it.
        cfg_dict["finetuning_scaling_factor"] = False

    return cfg_dict, state_dict


def load_sae_from_pretrained(path, device="cuda", dtype="float32"):
    """
    Adapted from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py.
    """
    config_path = str(path) + "_cfg.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    cfg_dict = handle_config_defaulting(cfg_dict)
    cfg_dict["device"] = device
    if dtype is not None:
        cfg_dict["dtype"] = dtype

    weight_path = str(path) + ".pt"
    cfg_dict, state_dict = read_sae_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weight_path,
        device=device,
    )

    key_map = {
        "act_size": "d_in",
        "dict_size": "d_sae",
        "seq_len": "context_size",
        "act_name": "hook_name",
        "layer": "hook_layer",
        "head": "hook_head_index",
    }
    for k, v in key_map.items():
        cfg_dict[v] = cfg_dict[k]
    cfg_dict["dataset_path"] = "Skylion007/openwebtext"

    sae_cfg = SAEConfig.from_dict(cfg_dict)

    sae = SAE(sae_cfg)
    # sae.process_state_dict_for_loading(state_dict)
    sae.load_state_dict(state_dict)

    sae.turn_off_forward_pass_hook_z_reshaping()

    return sae


def find_latest_checkpoint(
    layer,
    head,
    checkpoint_dir="../attention-output-saes/checkpoints",
):
    base = Path(checkpoint_dir)
    fn = (
        f"gpt2-small_L{layer}_H{head}_z_lr1.20e-03_l11.00e+00"
        "_ds2048_nt2000000000_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_sl64_c1"
    )
    ckps = sorted(base.glob(f"{fn}*.pt"), key=lambda p: p.stat().st_ctime)
    return list(ckps)[-1].with_suffix("")


def load_per_head_sae(
    layer,
    head,
    checkpoint_dir="../attention-output-saes/checkpoints",
):
    path = find_latest_checkpoint(layer, head, checkpoint_dir)
    return load_sae_from_pretrained(path)


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


def get_activations_per_sequence(
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


def get_dfa_for_batch(feat, model, sae, tokens, df):
    eval_idxs = df["idx"].to_numpy()
    eval_positions = df["act_argmax"].to_numpy()
    eval_tokens = tokens[eval_idxs]

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
    feature_acts = sae.encode(sae_in)[:, :, feat]

    rows = []
    for i, j in enumerate(eval_positions):
        vs = value_scores[i]
        a = attn[i, j]
        weighted_vs = vs * a
        toks = model.to_str_tokens(eval_tokens[i])
        tok_ids = eval_tokens[i].cpu().tolist()
        acts = feature_acts[i]
        d = {
            "unweighted_scores": vs.cpu().tolist(),
            "attn": a.cpu().tolist(),
            "scores": weighted_vs.cpu().tolist(),
            "feature_acts": acts.cpu().tolist(),
            "tokens": toks,
            "token_ids": tok_ids,
            "act_argmax": j.item() if type(j) != int else j,
        }
        row = df.iloc[i]
        d["feature"] = int(feat)
        d["idx"] = int(row["idx"])
        d["act_max"] = float(row["act_max"])
        d["bin"] = int(row["bin"])
        rows.append(d)

    return rows
