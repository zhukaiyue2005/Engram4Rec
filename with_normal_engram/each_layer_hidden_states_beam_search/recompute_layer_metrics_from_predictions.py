#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional


def parse_topk(value: str) -> List[int]:
    topk = []
    for part in str(value).replace("，", ",").replace("、", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise ValueError(f"topk must contain positive integers, got {k}")
        topk.append(k)
    if not topk:
        raise ValueError("topk cannot be empty")
    return sorted(set(topk))


def layer_from_prediction_path(path: str) -> int:
    name = os.path.basename(path)
    match = re.fullmatch(r"layer_(\d+)_predictions\.json", name)
    if not match:
        raise ValueError(f"cannot parse layer id from prediction filename: {path}")
    return int(match.group(1))


def normalize_target(row: Dict) -> str:
    target = row.get("target_title") or row.get("target_item_title") or ""
    if target:
        return str(target).strip()
    completion = str(row.get("completion", "")).strip().strip("\n")
    return completion.strip('"').strip()


def compute_metrics(rows: List[Dict], topk: Iterable[int]) -> Dict[str, float]:
    n = len(rows)
    metrics: Dict[str, float] = {}
    for k in topk:
        hit = 0.0
        ndcg = 0.0
        for row in rows:
            target = normalize_target(row)
            rank = -1
            for idx, pred in enumerate((row.get("predict") or [])[:k]):
                if pred == target:
                    rank = idx + 1
                    break
            if rank != -1:
                hit += 1.0
                ndcg += 1.0 / math.log2(rank + 1.0)
        metrics[f"Hit@{k}"] = hit / n if n else 0.0
        metrics[f"NDCG@{k}"] = ndcg / n if n else 0.0
    return metrics


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_metadata_from_predictions(prediction_paths: List[str]) -> List[Dict]:
    metadata_rows = []
    for path in prediction_paths:
        layer = layer_from_prediction_path(path)
        metadata_rows.append(
            {
                "layer": layer,
                "zero_based_layer": layer,
                "layer_index_base": 0,
            }
        )
    return sorted(metadata_rows, key=lambda row: row["layer"], reverse=True)


def prediction_path_for_layer(pred_dir: str, layer: int) -> str:
    return os.path.join(pred_dir, f"layer_{layer}_predictions.json")


def recompute(
    pred_dir: str,
    output_jsonl: str,
    topk: List[int],
    metrics_jsonl: Optional[str] = None,
) -> None:
    if metrics_jsonl:
        metadata_rows = load_jsonl(metrics_jsonl)
    else:
        prediction_paths = glob.glob(os.path.join(pred_dir, "layer_*_predictions.json"))
        if not prediction_paths:
            raise FileNotFoundError(f"no prediction files found in {pred_dir}")
        metadata_rows = build_metadata_from_predictions(prediction_paths)

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    written = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for metadata in metadata_rows:
            layer = int(metadata["layer"])
            pred_path = prediction_path_for_layer(pred_dir, layer)
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"missing predictions for layer {layer}: {pred_path}")

            with open(pred_path, encoding="utf-8") as pred_f:
                pred_rows = json.load(pred_f)

            metrics = compute_metrics(pred_rows, topk)
            row = dict(metadata)
            row["num_samples"] = len(pred_rows)
            row["num_beams"] = max((len(r.get("predict") or []) for r in pred_rows), default=0)
            row["topk"] = topk
            row["metrics"] = metrics

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows to {output_jsonl}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute per-layer Hit@K/NDCG@K from saved layer_*_predictions.json files."
    )
    parser.add_argument(
        "--pred_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing layer_<layer>_predictions.json files.",
    )
    parser.add_argument(
        "--metrics_jsonl",
        default="",
        help="Optional old layer metrics jsonl. When set, its row order and metadata are preserved.",
    )
    parser.add_argument(
        "--output_jsonl",
        default="layer_metric_2.jsonl",
        help="Output jsonl path. Relative paths are resolved under --pred_dir.",
    )
    parser.add_argument("--topk", default="1,3,5", help='Comma-separated K values, e.g. "1,3,5".')
    args = parser.parse_args()

    pred_dir = os.path.abspath(args.pred_dir)
    metrics_jsonl = args.metrics_jsonl
    if metrics_jsonl:
        metrics_jsonl = metrics_jsonl if os.path.isabs(metrics_jsonl) else os.path.join(pred_dir, metrics_jsonl)

    output_jsonl = args.output_jsonl
    if not os.path.isabs(output_jsonl):
        output_jsonl = os.path.join(pred_dir, output_jsonl)

    recompute(
        pred_dir=pred_dir,
        metrics_jsonl=metrics_jsonl or None,
        output_jsonl=output_jsonl,
        topk=parse_topk(args.topk),
    )


if __name__ == "__main__":
    main()
