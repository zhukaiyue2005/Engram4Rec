import json
import os
import random
from typing import Dict

import fire
import numpy as np
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate_hitk(model: SASRec, loader: DataLoader, device: torch.device, k: int = 10) -> float:
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            lengths = batch["len_seq"].to(device)
            target = batch["target"].to(device)

            logits = model.score_all_items(seq, lengths)
            topk = logits.topk(k=min(k, logits.shape[-1]), dim=-1).indices
            hit += (topk == target.unsqueeze(-1)).any(dim=-1).sum().item()
            total += target.shape[0]

    return float(hit / max(total, 1))


def train_sasrec(
    train_file: str = "../data/Industrial_and_Scientific_dataset/train.jsonl",
    valid_file: str = "../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    output_dir: str = "./SAS-checkpoints",
    max_seq_len: int = 50,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_heads: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 42,
):
    """
    训练 LLaRA 所需的传统推荐器（SASRec）。
    输入直接对齐当前 Industrial_and_Scientific_full_rank 数据格式。
    """
    set_seed(seed)
    train_file = resolve_demo_path(train_file)
    valid_file = resolve_demo_path(valid_file)
    info_file = resolve_demo_path(info_file)
    output_dir = resolve_demo_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    title2id, id2title = load_item_mappings(info_file)
    num_items = max(id2title.keys()) + 1

    train_ds = RecTrainDataset(train_file, title2id=title2id, max_seq_len=max_seq_len)
    valid_ds = RecTrainDataset(valid_file, title2id=title2id, max_seq_len=max_seq_len)

    if len(train_ds) == 0:
        raise ValueError("训练集为空，请检查 train_file/info_file 是否匹配")

    collate_fn = lambda x: build_padded_sequence(x, padding_item_id=num_items, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SASRecConfig(
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    model = SASRec(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_hit10 = -1.0
    best_path = os.path.join(output_dir, "sasrec_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_cnt = 0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            lengths = batch["len_seq"].to(device)
            target = batch["target"].to(device)

            logits = model(seq, lengths)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = seq.shape[0]
            total_loss += loss.item() * bs
            total_cnt += bs

        train_loss = total_loss / max(total_cnt, 1)
        hit10 = _evaluate_hitk(model, valid_loader, device=device, k=10)

        print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6f} valid_hit@10={hit10:.6f}", flush=True)

        if hit10 > best_hit10:
            best_hit10 = hit10
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "best_hit10": best_hit10,
                },
                best_path,
            )
            print(f"[Saved] best checkpoint -> {best_path}", flush=True)

    meta: Dict[str, object] = {
        "num_items": num_items,
        "max_seq_len": max_seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dropout": dropout,
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
        "best_hit10": best_hit10,
    }
    with open(os.path.join(output_dir, "sasrec_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=== Training Done ===", flush=True)
    print(f"best_hit@10: {best_hit10:.6f}", flush=True)


if __name__ == "__main__":
    fire.Fire(train_sasrec)
