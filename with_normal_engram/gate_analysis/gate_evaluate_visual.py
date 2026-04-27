import json
import os

import fire
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import CURRENT_DIR, resolve_results_json_path, sort_layer_names


def visualize_token_gates(
    sample_data,
    layer_name="",
    output_path="token_gate_viz.png",
    color_scheme="hot",
    show_values=False,
    max_tokens_per_row=50,
):
    if not layer_name:
        layer_name = sort_layer_names(sample_data["engram_analysis"].keys())[0]

    analysis = sample_data["engram_analysis"][layer_name]
    tokens = analysis["tokens"]
    gates = np.array(analysis["gate_values"]).flatten()
    min_len = min(len(tokens), len(gates))
    tokens = tokens[:min_len]
    gates = gates[:min_len]
    gate_min, gate_max = float(gates.min()), float(gates.max())

    if color_scheme == "hot":
        colors = ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"]
    elif color_scheme == "coolwarm":
        colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#fee090", "#fc8d59", "#d73027"]
    elif color_scheme == "viridis":
        colors = ["#440154", "#31688e", "#35b779", "#fde725"]
    else:
        colors = ["#ffffff", "#ffeda0", "#feb24c", "#f03b20", "#bd0026"]

    cmap = LinearSegmentedColormap.from_list("custom", colors)
    n_rows = (len(tokens) + max_tokens_per_row - 1) // max_tokens_per_row
    char_width = 8
    char_height = 16
    padding = 5
    img_width = max_tokens_per_row * 12 * char_width + 2 * padding
    img_height = n_rows * (char_height + 20) + 100

    fig, ax = plt.subplots(figsize=(img_width / 80, img_height / 80), dpi=100)
    ax.set_xlim(0, img_width)
    ax.set_ylim(0, img_height)
    ax.axis("off")
    ax.text(
        img_width / 2,
        img_height - 20,
        f"Sample {sample_data['sample_idx']} - {layer_name}",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
    )

    current_x = padding
    current_y = img_height - 80
    for token, gate in zip(tokens, gates):
        gate_normalized = (gate - gate_min) / (gate_max - gate_min) if gate_max > gate_min else 0.5
        color = cmap(gate_normalized)
        rgb = tuple(int(c * 255) for c in color[:3])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        display_token = token[1:] if token.startswith(("Ġ", "▁")) else token
        if display_token.startswith("Ċ"):
            display_token = display_token[1:]
        token_width_px = max(len(display_token) * char_width + 4, 20)
        if current_x + token_width_px > img_width - padding:
            current_x = padding
            current_y -= char_height + 25

        rect = patches.FancyBboxPatch(
            (current_x, current_y - char_height),
            token_width_px,
            char_height + 5,
            boxstyle="round,pad=0.02",
            facecolor=hex_color,
            edgecolor="darkred" if gate >= 0.7 else "gray",
            linewidth=2 if gate >= 0.7 else 0.5,
            alpha=0.85,
        )
        ax.add_patch(rect)
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = "white" if brightness < 128 else "black"
        ax.text(
            current_x + token_width_px / 2,
            current_y - char_height / 2 + 2,
            display_token,
            ha="center",
            va="center",
            fontsize=7,
            color=text_color,
            fontweight="bold" if gate >= 0.7 else "normal",
        )
        if show_values:
            ax.text(
                current_x + token_width_px / 2,
                current_y - char_height - 3,
                f"{gate:.2f}",
                ha="center",
                va="top",
                fontsize=5,
                color="gray",
            )
        current_x += token_width_px + 2

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=gate_min, vmax=gate_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.3, aspect=30, anchor=(0.5, -0.1))
    cbar.set_label(f"Gate Value (range: {gate_min:.2f} - {gate_max:.2f})", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ 可视化已保存: {output_path}")
    return output_path


def visualize_sample_from_file(
    jsonl_path="",
    sample_idx=0,
    layer_name="",
    output_dir=os.path.join(CURRENT_DIR, "visualizations"),
    **kwargs,
):
    jsonl_path = resolve_results_json_path(jsonl_path)
    os.makedirs(output_dir, exist_ok=True)
    target_sample = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["sample_idx"] == sample_idx:
                target_sample = data
                break
    if target_sample is None:
        raise ValueError(f"未找到样本 {sample_idx}")
    if not layer_name:
        layer_name = sort_layer_names(target_sample["engram_analysis"].keys())[0]
    output_path = os.path.join(output_dir, f"sample_{sample_idx}_{layer_name}_viz.png")
    return visualize_token_gates(target_sample, layer_name, output_path, **kwargs)


def visualize_all_layers_from_file(
    jsonl_path="",
    sample_idx=0,
    output_dir=os.path.join(CURRENT_DIR, "visualizations"),
    **kwargs,
):
    jsonl_path = resolve_results_json_path(jsonl_path)
    os.makedirs(output_dir, exist_ok=True)
    target_sample = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["sample_idx"] == sample_idx:
                target_sample = data
                break
    if target_sample is None:
        raise ValueError(f"未找到样本 {sample_idx}")

    output_paths = []
    for layer_name in sort_layer_names(target_sample["engram_analysis"].keys()):
        output_path = os.path.join(output_dir, f"sample_{sample_idx}_{layer_name}_viz.png")
        visualize_token_gates(target_sample, layer_name, output_path, **kwargs)
        output_paths.append(output_path)

    print(f"✅ 已为 sample {sample_idx} 输出 {len(output_paths)} 个层可视化")
    return output_paths


if __name__ == "__main__":
    fire.Fire({
        "sample": visualize_sample_from_file,
        "all_layers": visualize_all_layers_from_file,
    })
