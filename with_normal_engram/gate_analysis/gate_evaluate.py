import json
import os
from collections import defaultdict

import fire
import matplotlib.pyplot as plt
import numpy as np

from common import CURRENT_DIR, resolve_results_json_path, sort_layer_names


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_engram_gates(
    json_file_path: str = "",
    threshold: float = 0.7,
    output_dir: str = os.path.join(CURRENT_DIR, "engram_analysis_results"),
):
    json_file_path = resolve_results_json_path(json_file_path)
    os.makedirs(output_dir, exist_ok=True)
    all_gates_by_layer = defaultdict(list)
    all_gates_flat = []
    high_gate_stats = {
        "total_tokens": 0,
        "high_gate_tokens": 0,
        "by_layer": defaultdict(lambda: {"total": 0, "high": 0}),
    }

    sample_count = 0
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            sample_count += 1
            for layer_name, analysis in data["engram_analysis"].items():
                gates = np.array(analysis["gate_values"]).flatten()
                tokens = analysis["tokens"]
                valid_len = min(len(gates), len(tokens))
                gates = gates[:valid_len]
                if len(gates) == 0:
                    continue

                all_gates_by_layer[layer_name].extend(gates.tolist())
                all_gates_flat.extend(gates.tolist())
                high_count = int((gates >= threshold).sum())
                high_gate_stats["total_tokens"] += len(gates)
                high_gate_stats["high_gate_tokens"] += high_count
                high_gate_stats["by_layer"][layer_name]["total"] += len(gates)
                high_gate_stats["by_layer"][layer_name]["high"] += high_count

    if not all_gates_flat:
        raise ValueError(f"未找到任何 gate 值: {json_file_path}")

    all_gates_array = np.array(all_gates_flat)
    print(f"总样本数: {sample_count}")
    print(f"token 数: {high_gate_stats['total_tokens']:,}")
    print(f"高 gate (≥{threshold}) token 数: {high_gate_stats['high_gate_tokens']:,}")
    print(f"高 gate 比例: {high_gate_stats['high_gate_tokens'] / high_gate_stats['total_tokens'] * 100:.2f}%")
    print()
    print(f"{'Layer':<10} {'总Token':<12} {'高Gate':<12} {'比例':<10} {'均值':<8} {'标准差':<8}")
    print("-" * 70)

    layer_summary = {}
    for layer_name in sort_layer_names(all_gates_by_layer.keys()):
        gates = np.array(all_gates_by_layer[layer_name])
        stats = high_gate_stats["by_layer"][layer_name]
        ratio = stats["high"] / stats["total"] * 100
        layer_summary[layer_name] = {
            "total": int(stats["total"]),
            "high": int(stats["high"]),
            "ratio": float(ratio),
            "mean": float(gates.mean()),
            "std": float(gates.std()),
            "max": float(gates.max()),
            "min": float(gates.min()),
        }
        print(
            f"{layer_name:<10} {stats['total']:<12,} {stats['high']:<12,} "
            f"{ratio:<10.2f}% {gates.mean():<8.4f} {gates.std():<8.4f}"
        )

    results = {
        "threshold": threshold,
        "total_samples": sample_count,
        "total_tokens": high_gate_stats["total_tokens"],
        "high_gate_tokens": high_gate_stats["high_gate_tokens"],
        "high_gate_ratio": high_gate_stats["high_gate_tokens"] / high_gate_stats["total_tokens"],
        "overall_statistics": {
            "mean": float(all_gates_array.mean()),
            "std": float(all_gates_array.std()),
            "max": float(all_gates_array.max()),
            "min": float(all_gates_array.min()),
            "median": float(np.median(all_gates_array)),
        },
        "layer_statistics": layer_summary,
    }

    with open(os.path.join(output_dir, "gate_analysis_report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    plot_gate_distribution(all_gates_by_layer, all_gates_array, threshold, output_dir)
    save_layer_distribution_plots(all_gates_by_layer, threshold, output_dir)
    save_layer_comparison_plot(layer_summary, threshold, output_dir)
    save_gate_visual_analysis(layer_summary, threshold, output_dir)
    print(f"📄 详细报告已保存: {os.path.join(output_dir, 'gate_analysis_report.json')}")
    return results


def plot_gate_distribution(gates_by_layer, all_gates, threshold, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(all_gates, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    plt.axvline(np.mean(all_gates), color="green", linestyle="--", linewidth=2, label=f"Mean={np.mean(all_gates):.3f}")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.title("All-Token Gate Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    layer_names = sort_layer_names(gates_by_layer.keys())
    bp = plt.boxplot(
        [gates_by_layer[name] for name in layer_names],
        tick_labels=[f"L{name.split('_')[1]}" if "_" in name else name for name in layer_names],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    plt.axhline(threshold, color="red", linestyle="--", alpha=0.5, label=f"Threshold={threshold}")
    plt.xlabel("Layer")
    plt.ylabel("Gate Value")
    plt.title("All-Token Gate by Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gate_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_layer_distribution_plots(gates_by_layer, threshold, output_dir):
    layer_names = sort_layer_names(gates_by_layer.keys())
    for layer_name in layer_names:
        gates = np.array(gates_by_layer[layer_name])
        plt.figure(figsize=(8, 4.5))
        plt.hist(gates, bins=50, alpha=0.75, color="steelblue", edgecolor="black")
        plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
        plt.axvline(gates.mean(), color="green", linestyle="--", linewidth=2, label=f"Mean={gates.mean():.3f}")
        plt.xlabel("Gate Value")
        plt.ylabel("Frequency")
        plt.title(f"{layer_name} Gate Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_name}_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()


def save_layer_comparison_plot(layer_summary, threshold, output_dir):
    layer_names = sort_layer_names(layer_summary.keys())
    ratios = [layer_summary[name]["ratio"] for name in layer_names]
    means = [layer_summary[name]["mean"] for name in layer_names]
    x = np.arange(len(layer_names))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ratios, width=width, label=f"High gate >= {threshold}", color="#e76f51")
    plt.bar(x + width / 2, means, width=width, label="Mean gate", color="#457b9d")
    plt.xticks(x, layer_names)
    plt.ylabel("Value / Percent")
    plt.title("Layer-wise Gate Comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_gate_visual_analysis(layer_summary, threshold, output_dir):
    visual_analysis = {
        "threshold": threshold,
        "layers": {
            layer_name: {
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "high_gate_ratio": stats["ratio"] / 100.0,
                "high_gate_ratio_percent": stats["ratio"],
                "distribution_plot": os.path.join(output_dir, f"{layer_name}_distribution.png"),
            }
            for layer_name, stats in layer_summary.items()
        },
        "comparison_plot": os.path.join(output_dir, "layer_comparison.png"),
        "overall_plot": os.path.join(output_dir, "gate_distribution.png"),
    }
    with open(os.path.join(output_dir, "gate_visual_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(visual_analysis, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    fire.Fire(analyze_engram_gates)
