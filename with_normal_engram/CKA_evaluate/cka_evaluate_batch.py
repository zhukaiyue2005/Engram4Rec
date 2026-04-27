from typing import Dict, List

import torch
from tqdm import tqdm


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    device = K.device
    dtype = K.dtype

    I = torch.eye(n, device=device, dtype=dtype)
    one = torch.ones((n, n), device=device, dtype=dtype)
    H = I - one / n
    return torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    K = X @ X.T
    L = Y @ Y.T

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-10)
    return cka.item()


def extract_final_token_hidden_states(model, dataloader, device: str = "cuda"):
    model.eval()
    all_hidden_states = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting hidden states"):
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "output_hidden_states": True,
                "return_dict": True,
            }
            outputs = model(**model_inputs)
            hidden_states = outputs.hidden_states[1:]

            valid_lengths = batch["valid_lengths"]
            target_token_positions = batch.get("target_token_positions")
            for layer_idx, layer_hidden in enumerate(hidden_states):
                if len(all_hidden_states) <= layer_idx:
                    all_hidden_states.append([])

                for sample_idx, _ in enumerate(valid_lengths):
                    if target_token_positions is not None and int(target_token_positions[sample_idx].item()) >= 0:
                        last_pos = int(target_token_positions[sample_idx].item())
                    else:
                        non_pad_positions = batch["attention_mask"][sample_idx].nonzero(as_tuple=False)
                        last_pos = int(non_pad_positions[-1].item())
                    all_hidden_states[layer_idx].append(layer_hidden[sample_idx, last_pos, :].float().cpu())

    return [torch.stack(layer_list, dim=0) for layer_list in all_hidden_states]


def compute_cka_similarity_matrix(
    hidden_states_A: List[torch.Tensor],
    hidden_states_B: List[torch.Tensor],
    device: str | None = None,
) -> torch.Tensor:
    num_layers_A = len(hidden_states_A)
    num_layers_B = len(hidden_states_B)
    S = torch.zeros(num_layers_A, num_layers_B)
    compute_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Computing CKA similarity matrix: {num_layers_A} x {num_layers_B} on {compute_device}")
    for i in tqdm(range(num_layers_A), desc="CKA Matrix"):
        for j in range(num_layers_B):
            X = hidden_states_A[i].to(compute_device)
            Y = hidden_states_B[j].to(compute_device)
            S[i, j] = linear_cka(X, Y)
    return S


def compute_soft_alignment_index(S: torch.Tensor, k: int = 8) -> torch.Tensor:
    L_A, _ = S.shape
    alignment_indices = torch.zeros(L_A)

    for i in range(L_A):
        similarities = S[i, :]
        top_k = min(k, similarities.shape[0])
        top_k_values, top_k_indices = torch.topk(similarities, top_k)
        weighted_sum = (top_k_values * top_k_indices.float()).sum()
        weight_sum = top_k_values.sum()
        alignment_indices[i] = weighted_sum / (weight_sum + 1e-10)

    return alignment_indices


def evaluate(
    model_with_engram,
    model_without_engram,
    dataloader,
    batch_size: int = 32,
    k: int = 8,
    device: str = "cuda",
):
    print("=" * 60)
    print("Step 1: Extracting hidden states from model WITH Engram")
    hidden_states_with_engram = extract_final_token_hidden_states(
        model_with_engram,
        dataloader,
        device=device,
    )
    print(f"Engram model has {len(hidden_states_with_engram)} layers")

    print("\n" + "=" * 60)
    print("Step 2: Extracting hidden states from model WITHOUT Engram")
    print("=" * 60)
    hidden_states_without_engram = extract_final_token_hidden_states(
        model_without_engram,
        dataloader,
        device=device,
    )
    print(f"Baseline model has {len(hidden_states_without_engram)} layers")

    print("\n" + "=" * 60)
    print("Step 3: Computing CKA Similarity Matrix")
    S = compute_cka_similarity_matrix(hidden_states_with_engram, hidden_states_without_engram)

    print(f"\nCKA Similarity Matrix shape: {S.shape}")
    print(f"Mean CKA: {S.mean():.4f}")
    print(f"Max CKA: {S.max():.4f}")
    print(f"Min CKA: {S.min():.4f}")

    best_matches = S.argmax(dim=1)
    print("\nBest matching layers (Engram -> Baseline):")
    for i, match in enumerate(best_matches):
        print(f"  Engram layer {i:2d} -> Baseline layer {match:2d} (CKA={S[i, match]:.4f})")

    print("\n" + "=" * 60)
    print("Step 4: Computing Soft Alignment Index")
    alignment_indices = compute_soft_alignment_index(S, k=k)
    print(f"\nSoft Alignment Indices (top-{k} weighted centroid):")
    for i, idx in enumerate(alignment_indices):
        print(f"  Engram layer {i:2d} aligns to Baseline layer {idx:.4f}")

    return {
        "cka_similarity_matrix": S.cpu().numpy(),
        "alignment_indices": alignment_indices.cpu().numpy(),
        "best_matches": best_matches.cpu().numpy(),
        "num_layers_engram": len(hidden_states_with_engram),
        "num_layers_baseline": len(hidden_states_without_engram),
        "mean_cka": S.mean().item(),
        "max_cka": S.max().item(),
        "batch_size": batch_size,
    }


def plot_cka_results(result: Dict, save_path: str = "cka_analysis.png", k: int = 8):
    import matplotlib.pyplot as plt
    import seaborn as sns

    S = result["cka_similarity_matrix"]
    alignment_indices = result["alignment_indices"]
    engram_layer_ids = result.get("engram_layer_ids")
    if engram_layer_ids:
        selected_rows = [int(x) for x in engram_layer_ids if 0 <= int(x) < S.shape[0]]
        if not selected_rows:
            selected_rows = list(range(S.shape[0]))
    else:
        selected_rows = list(range(S.shape[0]))

    S_plot = S[selected_rows, :]
    selected_alignment = alignment_indices[selected_rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    sns.heatmap(
        S_plot,
        ax=ax1,
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=range(S.shape[1]),
        yticklabels=selected_rows,
        cbar_kws={"label": "CKA Similarity"},
    )
    ax1.set_xlabel("Baseline Model Layers")
    ax1.set_ylabel("Engram Insert Layers")
    ax1.set_title("CKA Similarity Matrix (Engram Insert Layers)")

    ax2 = axes[1]
    ax2.plot(selected_rows, selected_alignment, "b-o", markersize=4, label="Weighted Baseline Layer")
    ax2.plot(
        [min(selected_rows), max(selected_rows)],
        [min(selected_rows), max(selected_rows)],
        "r--",
        label="Perfect Alignment",
        alpha=0.5,
    )
    ax2.set_xlabel("Engram Insert Layer")
    ax2.set_ylabel("Aligned Baseline Layer (Soft Index)")
    ax2.set_title(f"Engram Insert Layers -> Weighted Baseline Layer (k={k})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    return fig
