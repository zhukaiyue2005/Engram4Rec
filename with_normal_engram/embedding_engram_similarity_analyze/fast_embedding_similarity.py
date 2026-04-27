import os
import sys
import json
from typing import Dict, List

import fire
import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DEFAULT_DATA_FILE = os.path.join(PARENT_DIR, "../data/Industrial_and_Scientific_dataset/valid.jsonl")
DEFAULT_SAVE_DIR = os.path.join(CURRENT_DIR, "engram_analysis_fast")

engram_dir = os.path.join(CURRENT_DIR, "Engram_Insert_code")
if engram_dir not in sys.path:
    sys.path.append(engram_dir)

from engram_demo_v1 import Engram, EngramConfig  # noqa: E402
from embedding_similarity import (  # noqa: E402
    infer_engram_layer_ids_from_checkpoint,
    load_rows,
    parse_int_list_arg,
    save_similarity_histogram,
    set_deterministic_mode,
)


def _parse_prompt_column(prompt_col) -> str:
    if isinstance(prompt_col, list):
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
        return ""
    if isinstance(prompt_col, str):
        return prompt_col
    return ""


def _extract_extra_info(extra_info_col):
    if isinstance(extra_info_col, dict):
        ground_truth = extra_info_col.get("ground_truth", {})
        description = ground_truth.get("description", "")
        title = ground_truth.get("title", "")
        return description, title
    return "", ""


def _build_sft_text_record(row: dict, tokenizer, cutoff_len: int):
    """Build the same full text sequence used by sft.py: prompt + completion + eos."""
    if "extra_info" in row:
        prompt_text = _parse_prompt_column(row.get("prompt", row.get("messages", "")))
        _, completion = _extract_extra_info(row.get("extra_info", {}))
    else:
        prompt_text = _parse_prompt_column(row.get("prompt", row.get("messages", "")))
        completion = row.get("target_item_title", "")
        if completion:
            completion = f"\"{completion}\""
        else:
            completion = str(row.get("completion", "")).rstrip("\n")

    full_text = f"{prompt_text}{completion}{tokenizer.eos_token}"
    input_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        add_special_tokens=True,
    )["input_ids"]

    return {
        "input_ids": input_ids,
    }


def _collate_full_sequence_fn(batch: List[dict], tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_mask = []
    valid_lengths = []

    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        padded_input_ids.append([pad_id] * pad_len + ids)
        padded_attention_mask.append([0] * pad_len + [1] * len(ids))
        valid_lengths.append(len(ids))

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "valid_lengths": valid_lengths,
    }


def _load_one_engram(layer_id: int, config: EngramConfig, checkpoint_dir: str) -> Engram:
    module = Engram(layer_id=layer_id, config=config)
    param_path = os.path.join(checkpoint_dir, "engram_params", f"engram_layer_{layer_id}.npy")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"missing Engram params: {param_path}")
    module.load_all_params(param_path)
    module.eval()
    return module


def _build_engram_modules(layer_ids: List[int], checkpoint_dir: str) -> Dict[int, Engram]:
    config = EngramConfig()
    config.layer_ids = layer_ids
    return {layer_id: _load_one_engram(layer_id, config, checkpoint_dir) for layer_id in layer_ids}


def _extract_sentence_embeddings(
    engram_modules: Dict[int, Engram],
    input_ids: torch.Tensor,
    valid_lengths: List[int],
    device: str,
    deduplicate_embeddings: bool = True,
    batch_start_index: int = 0,
    embedding_lookup: dict | None = None,
    stats: dict | None = None,
) -> List[dict]:
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")

    if not engram_modules:
        return []

    # Keep input ids on CPU for Engram hash; _ids_to_embeddings handles CPU hash lookup.
    input_ids_cpu = input_ids.detach().cpu()
    first_module = next(iter(engram_modules.values()))
    compressed_ids_cpu = torch.from_numpy(
        first_module.hash_mapping.compressed_tokenizer(input_ids_cpu.numpy())
    ).to(torch.long)
    layer_cache = {}
    layer_hash_cache = {}

    for layer_id, module in engram_modules.items():
        with torch.no_grad():
            hash_ids = module.hash_mapping.hash(input_ids_cpu.numpy())[layer_id]
            layer_hash_cache[f"layer_{layer_id}"] = hash_ids
            emb = module._ids_to_embeddings(input_ids_cpu, target_device=target_device)
            layer_cache[f"layer_{layer_id}"] = emb.detach().cpu().float().numpy()

    rows = []
    if embedding_lookup is None:
        embedding_lookup = {}
    if stats is None:
        stats = {}
    pad_sentinel = 0
    batch_size = input_ids_cpu.shape[0]
    for row_idx in range(batch_size):
        valid_length = int(valid_lengths[row_idx])
        stats["total_tokens"] = stats.get("total_tokens", 0) + valid_length
        selected_pos = list(range(valid_length))
        sample_comp_ids = compressed_ids_cpu[row_idx][-valid_length:]
        padded_comp_ids = torch.cat(
            [torch.full((2,), pad_sentinel, dtype=sample_comp_ids.dtype), sample_comp_ids],
            dim=0,
        )
        for pos in selected_pos:
            trigram_ids = padded_comp_ids[pos : pos + 3]
            trigram_tuple = tuple(int(x) for x in trigram_ids.tolist())
            lookup_key = tuple(
                (
                    layer_name,
                    *layer_hash_cache[layer_name][row_idx][-valid_length:][pos].tolist(),
                )
                for layer_name in sorted(layer_hash_cache)
            )
            if deduplicate_embeddings and lookup_key in embedding_lookup:
                embedding_lookup[lookup_key]["count"] += 1
                stats["duplicate_tokens"] = stats.get("duplicate_tokens", 0) + 1
                continue

            row = {
                "sample_index": batch_start_index + row_idx,
                "seq_pos": int(pos),
                "compressed_id": int(sample_comp_ids[pos].item()),
                "trigram_compressed_ids": trigram_tuple,
                "lookup_key": lookup_key,
                "layer_hash_ids": {
                    layer_name: tuple(
                        int(x) for x in layer_hash_cache[layer_name][row_idx][-valid_length:][pos].tolist()
                    )
                    for layer_name in sorted(layer_hash_cache)
                },
                "count": 1,
                "embedding": {},
                "layer_embeddings": {},
            }
            for layer_name, layer_tensor in layer_cache.items():
                value = layer_tensor[row_idx][-valid_length:][pos]
                row["embedding"][layer_name] = value
                row["layer_embeddings"][layer_name] = value
            rows.append(row)
            stats["kept_tokens"] = stats.get("kept_tokens", 0) + 1
            if deduplicate_embeddings:
                embedding_lookup[lookup_key] = row
    return rows


def _build_embedding_matrix(query_rows: List[dict], layer_name: str) -> np.ndarray:
    embeddings = []
    for row in query_rows:
        layer_dict = row.get("layer_embeddings") or row.get("embedding") or {}
        if layer_name in layer_dict:
            embeddings.append(np.asarray(layer_dict[layer_name], dtype=np.float32))
    if not embeddings:
        raise ValueError(f"No embeddings found for {layer_name}")
    return np.stack(embeddings, axis=0)


def save_exact_blockwise_similarity_histogram(
    query_rows: List[dict],
    layer_name: str,
    save_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    x_min: float = -1.0,
    x_max: float = 1.0,
    bins: int = 60,
    block_size: int = 4096,
):
    embedding_matrix = _build_embedding_matrix(query_rows, layer_name)
    n = embedding_matrix.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        raise ValueError(f"{layer_name} has fewer than two embeddings")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")

    edges = np.linspace(x_min, x_max, bins + 1, dtype=np.float64)
    hist_counts = np.zeros(bins, dtype=np.int64)
    count = 0
    sum_val = 0.0
    sum_sq = 0.0
    min_val = float("inf")
    max_val = float("-inf")

    x = torch.as_tensor(embedding_matrix, device=target_device, dtype=dtype)
    x = torch.nn.functional.normalize(x, p=2, dim=1)

    num_blocks = (n + block_size - 1) // block_size
    pbar = tqdm(total=num_blocks * (num_blocks + 1) // 2, desc=f"Exact {layer_name}")
    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        xi = x[i0:i1]
        for j0 in range(i0, n, block_size):
            j1 = min(n, j0 + block_size)
            sims = xi @ x[j0:j1].T
            if i0 == j0:
                tri_i, tri_j = torch.triu_indices(i1 - i0, j1 - j0, offset=1, device=target_device)
                vals = sims[tri_i, tri_j]
            else:
                vals = sims.reshape(-1)
            if vals.numel() == 0:
                pbar.update(1)
                continue

            vals32 = vals.to(torch.float32)
            count += int(vals32.numel())
            sum_val += float(vals32.sum().item())
            sum_sq += float((vals32 * vals32).sum().item())
            min_val = min(min_val, float(vals32.min().item()))
            max_val = max(max_val, float(vals32.max().item()))
            vals_np = vals32.detach().cpu().numpy()
            hist_counts += np.histogram(vals_np, bins=edges)[0]
            pbar.update(1)
    pbar.close()

    mean = sum_val / count
    variance = max(0.0, (sum_sq / count) - mean * mean)
    stats = {
        "layer_name": layer_name,
        "embedding_count": n,
        "pair_count": count,
        "expected_pair_count": total_pairs,
        "mean": mean,
        "std": variance ** 0.5,
        "min": min_val,
        "max": max_val,
        "histogram_range": [x_min, x_max],
        "histogram_bins": bins,
        "histogram_counts": hist_counts.tolist(),
        "histogram_edges": edges.tolist(),
        "block_size": block_size,
        "mode": "exact_blockwise",
    }

    stats_path = os.path.join(save_dir, f"{layer_name}_exact_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    density = hist_counts / max(1, hist_counts.sum()) / widths
    plt.figure(figsize=(6, 4.8))
    plt.bar(centers, density, width=widths, align="center", alpha=0.75)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"{layer_name} Exact Blockwise Similarity")
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{layer_name}_exact_distribution.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"exact stats saved to: {stats_path}")
    print(f"exact plot saved to: {fig_path}")


def inference(
    data_file: str = DEFAULT_DATA_FILE,
    save_dir: str = DEFAULT_SAVE_DIR,
    resume_from_checkpoint: str = "",
    base_model: str = "Qwen3-1.7B",
    engram_layer_ids: str = "",
    analyze_layer_ids: str = "",
    cutoff_len: int = 1024,
    batch_size: int = 64,
    max_samples: int = 5000,
    num_samples: int = 500000,
    x_min: float = -1.0,
    x_max: float = 1.0,
    bins: int = 60,
    seed: int = 1958,
    device: str = "cuda",
    similarity_mode: str = "sampled",
    exact_block_size: int = 4096,
    deduplicate_embeddings: bool = True,
    log_every: int = 10,
):
    if not resume_from_checkpoint:
        raise ValueError("resume_from_checkpoint cannot be empty")

    set_deterministic_mode(seed=seed)
    os.makedirs(save_dir, exist_ok=True)

    layer_ids = infer_engram_layer_ids_from_checkpoint(resume_from_checkpoint)
    if not layer_ids and engram_layer_ids:
        layer_ids = parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if not layer_ids:
        raise ValueError("Cannot infer Engram layer ids; pass --engram_layer_ids manually.")

    print(f"Fast path: loading only Engram params for layers {layer_ids}")
    print(
        "Config: "
        f"data_file={data_file}, save_dir={save_dir}, batch_size={batch_size}, "
        f"max_samples={max_samples}, cutoff_len={cutoff_len}, device={device}, "
        f"similarity_mode={similarity_mode}, deduplicate_embeddings={deduplicate_embeddings}, "
        f"log_every={log_every}"
    )
    engram_modules = _build_engram_modules(layer_ids, resume_from_checkpoint)

    # Match sft.py formatting: prompt + completion + eos, then extract the full valid token sequence.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    rows = load_rows(data_file=data_file, max_samples=max_samples, shuffle_seed=42)
    print(f"Loaded rows: {len(rows)}")
    processed_rows = [_build_sft_text_record(row, tokenizer, cutoff_len=cutoff_len) for row in rows]
    total_valid_tokens = sum(len(row["input_ids"]) for row in processed_rows)
    avg_valid_tokens = total_valid_tokens / max(1, len(processed_rows))
    print(
        "Tokenized full SFT sequences: "
        f"total_tokens={total_valid_tokens}, avg_tokens={avg_valid_tokens:.2f}, cutoff_len={cutoff_len}"
    )
    dataloader = DataLoader(
        processed_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate_full_sequence_fn(batch, tokenizer),
    )

    query_rows = []
    embedding_lookup = {}
    extraction_stats = {"total_tokens": 0, "kept_tokens": 0, "duplicate_tokens": 0}
    global_sample_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Hash lookup embeddings"), start=1):
        batch_rows = _extract_sentence_embeddings(
            engram_modules=engram_modules,
            input_ids=batch["input_ids"],
            valid_lengths=batch["valid_lengths"],
            device=device,
            deduplicate_embeddings=deduplicate_embeddings,
            batch_start_index=global_sample_idx,
            embedding_lookup=embedding_lookup,
            stats=extraction_stats,
        )
        query_rows.extend(batch_rows)
        global_sample_idx += len(batch["valid_lengths"])
        if log_every > 0 and (batch_idx % log_every == 0 or global_sample_idx >= len(processed_rows)):
            total_tokens = extraction_stats["total_tokens"]
            kept_tokens = extraction_stats["kept_tokens"]
            duplicate_tokens = extraction_stats["duplicate_tokens"]
            dedup_rate = duplicate_tokens / max(1, total_tokens)
            print(
                "Progress: "
                f"batches={batch_idx}/{len(dataloader)}, samples={global_sample_idx}/{len(processed_rows)}, "
                f"tokens_seen={total_tokens}, kept_unique={kept_tokens}, duplicates={duplicate_tokens}, "
                f"dedup_rate={dedup_rate:.2%}"
            )

    pkl_path = os.path.join(save_dir, "query_rows.pkl")
    pd.to_pickle(query_rows, pkl_path)
    print(f"query_rows saved to: {pkl_path}")
    print("token_selection: full_sequence")
    print(f"deduplicate_embeddings: {deduplicate_embeddings}")
    print(f"saved token rows: {len(query_rows)}")
    print(
        "Extraction summary: "
        f"tokens_seen={extraction_stats['total_tokens']}, "
        f"kept_unique={extraction_stats['kept_tokens']}, "
        f"duplicates={extraction_stats['duplicate_tokens']}"
    )

    plot_layer_ids = parse_int_list_arg(analyze_layer_ids, "analyze_layer_ids") if analyze_layer_ids else layer_ids
    for layer_id in plot_layer_ids:
        layer_name = f"layer_{layer_id}"
        if similarity_mode == "exact":
            save_exact_blockwise_similarity_histogram(
                query_rows=query_rows,
                layer_name=layer_name,
                save_dir=save_dir,
                device=device,
                dtype=torch.float16,
                x_min=x_min,
                x_max=x_max,
                bins=bins,
                block_size=exact_block_size,
            )
        elif similarity_mode == "sampled":
            save_similarity_histogram(
                query_rows=query_rows,
                layer_name=layer_name,
                save_dir=save_dir,
                num_samples=num_samples,
                device=device,
                dtype=torch.float16,
                x_min=x_min,
                x_max=x_max,
                bins=bins,
            )
        else:
            raise ValueError(f"Unsupported similarity_mode: {similarity_mode}")


if __name__ == "__main__":
    fire.Fire(inference)
