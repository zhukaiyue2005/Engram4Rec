import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
JSONL_PATH = BASE_DIR / "layer_metrics.jsonl"
TSV_PATH = BASE_DIR / "layer_ndcg5_hit5_summary.tsv"


def main():
    rows = []
    for line in JSONL_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    rows.sort(key=lambda row: row["layer"])
    lines = ["layer\tNDCG@5\tHit@5"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"{row['layer']}\t{metrics['NDCG@5']:.6f}\t{metrics['Hit@5']:.6f}"
        )

    TSV_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved summary to {TSV_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
