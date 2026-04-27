import json
import math
import os
from typing import Dict, List

import fire
import torch
from torch.utils.data import DataLoader

from data_utils import RecTrainDataset, build_padded_sequence, load_item_mappings
from sasrec_model import SASRec, SASRecConfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_demo_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def _parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
    values = []
    if isinstance(arg_value, (tuple, list)):
        for x in arg_value:
            if isinstance(x, str):
                values.extend([p.strip() for p in x.split(",") if p.strip()])
            elif x is not None:
                values.append(x)
    elif isinstance(arg_value, str):
        values = [p.strip() for p in arg_value.split(",") if p.strip()]
    elif arg_value is not None:
        values = [arg_value]

    out: List[int] = []
    for x in values:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            raise ValueError(f"{arg_name}中包含非法整数值: {x!r}, 原始输入: {arg_value!r}")
    return out


def _compute_metrics(topk_indices: torch.Tensor, targets: torch.Tensor, topk: List[int]) -> Dict[str, float]:
    # topk_indices: [B, max_k], targets: [B]
    n = targets.shape[0]
    out: Dict[str, float] = {}
    if n == 0:
        for k in topk:
            out[f"Hit@{k}"] = 0.0
            out[f"NDCG@{k}"] = 0.0
        return out

    for k in topk:
        hit_sum = 0.0
        ndcg_sum = 0.0
        match = (topk_indices[:, :k] == targets.unsqueeze(-1))
        for i in range(n):
            pos = torch.nonzero(match[i], as_tuple=False)
            if pos.numel() > 0:
                rank = int(pos[0].item()) + 1  # 1-based
                hit_sum += 1.0
                ndcg_sum += 1.0 / math.log2(rank + 1.0)
        out[f"Hit@{k}"] = hit_sum / n
        out[f"NDCG@{k}"] = ndcg_sum / n
    return out


def evaluate_sasrec(
    checkpoint_path: str = "./SAS-checkpoints/sasrec_best.pt",
    test_file: str = "../data/Industrial_and_Scientific_dataset/test.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    batch_size: int = 512,
    max_seq_len: int = 50,
    eval_topk: str = "1,3,5,10",
    num_workers: int = 0,
    save_json: str = "",
):
    checkpoint_path = resolve_demo_path(checkpoint_path)
    test_file = resolve_demo_path(test_file)
    info_file = resolve_demo_path(info_file)
    save_json = resolve_demo_path(save_json)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    topk = sorted(set(_parse_int_list_arg(eval_topk, "eval_topk")))
    if not topk:
        raise ValueError(f"eval_topk不能为空，当前值: {eval_topk}")
    if min(topk) <= 0:
        raise ValueError(f"eval_topk中的K必须为正整数，当前值: {eval_topk}")

    title2id, _ = load_item_mappings(info_file)
    test_ds = RecTrainDataset(test_file, title2id=title2id, max_seq_len=max_seq_len)
    if len(test_ds) == 0:
        raise ValueError("测试集为空，请检查 test_file/info_file 是否匹配")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config_dict = ckpt["config"]
    config = SASRecConfig(
        num_items=int(config_dict["num_items"]),
        max_seq_len=int(config_dict["max_seq_len"]),
        hidden_size=int(config_dict["hidden_size"]),
        num_layers=int(config_dict["num_layers"]),
        num_heads=int(config_dict["num_heads"]),
        dropout=float(config_dict["dropout"]),
    )
    model = SASRec(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collate_fn = lambda x: build_padded_sequence(x, padding_item_id=config.num_items, max_seq_len=config.max_seq_len)
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    max_k = max(topk)
    total = 0
    hit_sum = {k: 0.0 for k in topk}
    ndcg_sum = {k: 0.0 for k in topk}

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            lengths = batch["len_seq"].to(device)
            targets = batch["target"].to(device)

            logits = model.score_all_items(seq, lengths)
            topk_indices = logits.topk(k=min(max_k, logits.shape[-1]), dim=-1).indices
            batch_metrics = _compute_metrics(topk_indices, targets, topk)

            bsz = targets.shape[0]
            total += bsz
            for k in topk:
                hit_sum[k] += batch_metrics[f"Hit@{k}"] * bsz
                ndcg_sum[k] += batch_metrics[f"NDCG@{k}"] * bsz

    metrics: Dict[str, float] = {}
    for k in topk:
        metrics[f"Hit@{k}"] = hit_sum[k] / max(total, 1)
        metrics[f"NDCG@{k}"] = ndcg_sum[k] / max(total, 1)

    print("=== SASRec Test Metrics ===", flush=True)
    print(f"checkpoint: {checkpoint_path}", flush=True)
    print(f"num_test_samples: {total}", flush=True)
    for k in topk:
        print(
            f"Hit@{k}: {metrics[f'Hit@{k}']:.6f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}",
            flush=True,
        )

    if save_json:
        payload = {
            "checkpoint_path": checkpoint_path,
            "test_file": test_file,
            "info_file": info_file,
            "num_test_samples": total,
            "metrics": metrics,
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Saved] metrics json -> {save_json}", flush=True)


if __name__ == "__main__":
    fire.Fire(evaluate_sasrec)
