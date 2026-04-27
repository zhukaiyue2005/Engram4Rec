import os
import random

import fire
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKL_PATH = os.path.join(CURRENT_DIR, "engram_analysis", "query_rows.pkl")
DEFAULT_SAVE_DIR = os.path.join(CURRENT_DIR, "engram_analysis_reload")


def compute_group_cosine_similarity_values(
    group_embeddings,
    num_samples=50000,
    device="cuda",
    dtype=torch.float16,
    return_pairs=False,
):
    n = group_embeddings.shape[0]
    if n < 2:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        if return_pairs:
            empty_idx = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty_idx, empty_idx
        return empty

    total_pairs = n * (n - 1) // 2
    num_samples = min(num_samples, total_pairs)

    x = torch.as_tensor(group_embeddings, device=device, dtype=dtype)
    x = F.normalize(x, p=2, dim=1)

    sampled_k = random.sample(range(total_pairs), num_samples)
    sampled_k = torch.tensor(sampled_k, device=device, dtype=torch.long)

    row_lengths = torch.arange(n - 1, 0, -1, device=device, dtype=torch.long)
    row_starts = torch.empty(n - 1, device=device, dtype=torch.long)
    row_starts[0] = 0
    if n - 1 > 1:
        row_starts[1:] = torch.cumsum(row_lengths[:-1], dim=0)

    i = torch.searchsorted(row_starts, sampled_k, right=True) - 1
    offset_in_row = sampled_k - row_starts[i]
    j = i + 1 + offset_in_row

    sim_values = (x[i] * x[j]).sum(dim=1).to(torch.float32)

    if return_pairs:
        return sim_values, i, j
    return sim_values


def load_query_rows(pkl_path):
    query_rows = pd.read_pickle(pkl_path)
    if not isinstance(query_rows, list):
        raise TypeError(f"期望 query_rows.pkl 里是 list，实际是 {type(query_rows)}")
    if len(query_rows) == 0:
        raise ValueError("query_rows.pkl 为空")
    return query_rows


def build_embedding_matrix_from_saved(query_rows, layer_id="layer_10"):
    embeddings = []
    kept_rows = []

    for row in query_rows:
        layer_dict = None
        if "embedding" in row and isinstance(row["embedding"], dict):
            layer_dict = row["embedding"]
        elif "layer_embeddings" in row and isinstance(row["layer_embeddings"], dict):
            layer_dict = row["layer_embeddings"]
        else:
            continue

        if layer_id not in layer_dict:
            continue

        embeddings.append(np.asarray(layer_dict[layer_id], dtype=np.float32))
        kept_rows.append(row)

    if len(embeddings) == 0:
        available_layers = set()
        for row in query_rows:
            if "embedding" in row and isinstance(row["embedding"], dict):
                available_layers.update(row["embedding"].keys())
            elif "layer_embeddings" in row and isinstance(row["layer_embeddings"], dict):
                available_layers.update(row["layer_embeddings"].keys())
        raise ValueError(
            f"没有找到 layer_id={layer_id} 的 embedding。可用层有: {sorted(available_layers)}"
        )

    embedding_matrix = np.stack(embeddings, axis=0)
    return kept_rows, embedding_matrix


def analyze_saved_query_rows(
    pkl_path=DEFAULT_PKL_PATH,
    layer_id="layer_10",
    save_dir=DEFAULT_SAVE_DIR,
    num_samples=50000,
    device="cuda",
    dtype="float16",
    x_min=-1.0,
    x_max=1.0,
    bins=60,
):
    os.makedirs(save_dir, exist_ok=True)
    torch_dtype = getattr(torch, dtype)

    query_rows = load_query_rows(pkl_path)
    kept_rows, embedding_matrix = build_embedding_matrix_from_saved(query_rows, layer_id=layer_id)

    print("读取到的总行数:", len(query_rows))
    print("当前层有效行数:", len(kept_rows))
    print("embedding_matrix shape:", embedding_matrix.shape)

    sim_values = compute_group_cosine_similarity_values(
        embedding_matrix,
        num_samples=num_samples,
        device=device,
        dtype=torch_dtype,
        return_pairs=False,
    )
    sim_values_np = sim_values.detach().cpu().numpy()

    plt.figure(figsize=(6, 4.8))
    plt.hist(sim_values_np, bins=bins, density=True, alpha=0.7)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"{layer_id} Embedding Features")
    plt.xlim(x_min, x_max)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, f"{layer_id}_distribution.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"图已保存到: {fig_path}")

    return kept_rows, embedding_matrix, sim_values_np


if __name__ == "__main__":
    fire.Fire(analyze_saved_query_rows)
