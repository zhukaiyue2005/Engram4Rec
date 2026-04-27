import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENGRAM4REC_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DEFAULT_WITHOUT_ENGRAM_LOG = (
    os.path.join(
        ENGRAM4REC_ROOT,
        "without_engram",
        "KL_evaluate",
        "kl_eval_exact_kl_test1000_seed42_lr=3e-4_checkpoint-1981.json",
    )
)
DEFAULT_WITH_ENGRAM_LOG = (
    os.path.join(
        SCRIPT_DIR,
        "kl_eval_exact_kl_test1000_seed42_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4_checkpoint-1988.json",
    )
)
DEFAULT_WITH_ITEM_ATTENTION_LOG = (
    os.path.join(
        ENGRAM4REC_ROOT,
        "with_item_engram",
        "KL_evaluate",
        "kl_eval_exact_kl_test1000_seed42_layer=6,13,20_tbs=64_lora=3e-4_engram=3e-4_checkpoint-3976.json",
    )
)
DEFAULT_OUTPUT = (
    os.path.join(SCRIPT_DIR, "kl_comparison_exact_kl_test1000_seed42.png")
)


LAYER_LINE_RE = re.compile(r"^layer\s+(\d+):\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def parse_layer_values(log_path: str) -> Dict[int, float]:
    """Parse the final 'layer NN: value' KL block from a log file."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    parsed_blocks: List[Dict[int, float]] = []
    current: Dict[int, float] = {}
    in_layer_block = False

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("===") and "layer averages" in line:
                if current:
                    parsed_blocks.append(current)
                current = {}
                in_layer_block = True
                continue

            if not in_layer_block:
                continue

            match = LAYER_LINE_RE.match(line)
            if match:
                layer = int(match.group(1))
                value = float(match.group(2))
                current[layer] = value
            elif current:
                parsed_blocks.append(current)
                current = {}
                in_layer_block = False

    if current:
        parsed_blocks.append(current)

    if not parsed_blocks:
        raise ValueError(f"No 'layer NN: value' KL block found in: {log_path}")
    return parsed_blocks[-1]


def load_layer_values(path: str) -> Dict[int, float]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        layer_values = payload.get("layer_metric_averages")
        if not isinstance(layer_values, dict):
            raise ValueError(f"JSON missing layer_metric_averages: {path}")
        return {int(layer): float(value) for layer, value in layer_values.items()}
    return parse_layer_values(path)


def sorted_xy(values: Dict[int, float]) -> Tuple[List[int], List[float]]:
    layers = sorted(values)
    return layers, [values[layer] for layer in layers]


def plot_kl(
    without_engram_log: str,
    with_engram_log: str,
    with_item_attention_log: str,
    output: str,
    title: str,
    dpi: int,
) -> None:
    without_values = load_layer_values(without_engram_log)
    with_values = load_layer_values(with_engram_log)
    item_attention_values = load_layer_values(with_item_attention_log)

    without_layers, without_kl = sorted_xy(without_values)
    with_layers, with_kl = sorted_xy(with_values)
    item_attention_layers, item_attention_kl = sorted_xy(item_attention_values)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        without_layers,
        without_kl,
        marker="o",
        linewidth=2,
        markersize=4,
        label="without_engram lr=3e-4 checkpoint-1981",
    )
    plt.plot(
        with_layers,
        with_kl,
        marker="s",
        linewidth=2,
        markersize=4,
        label="with_engram layers=6,13,20 checkpoint-1988",
    )
    plt.plot(
        item_attention_layers,
        item_attention_kl,
        marker="^",
        linewidth=2,
        markersize=4,
        label="item_attention normalized tbs=64 checkpoint-3976",
    )

    plt.xlabel("Layer")
    plt.ylabel("KL divergence")
    plt.title(title)
    plt.xticks(sorted(set(without_layers + with_layers + item_attention_layers)))
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    print(f"Saved plot to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KL divergence by layer for KL evaluation logs.")
    parser.add_argument("--without_engram_log", default=DEFAULT_WITHOUT_ENGRAM_LOG)
    parser.add_argument("--with_engram_log", default=DEFAULT_WITH_ENGRAM_LOG)
    parser.add_argument("--with_item_attention_log", default=DEFAULT_WITH_ITEM_ATTENTION_LOG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="Layer-wise KL Divergence Comparison")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    plot_kl(
        without_engram_log=args.without_engram_log,
        with_engram_log=args.with_engram_log,
        with_item_attention_log=args.with_item_attention_log,
        output=args.output,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
