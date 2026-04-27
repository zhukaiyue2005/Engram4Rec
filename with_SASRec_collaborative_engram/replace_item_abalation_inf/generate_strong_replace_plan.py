import json
import os
from collections import Counter, defaultdict

try:
    import fire
except ImportError:
    fire = None

from build_item_style_groups import STYLE_RULES, _assign_style


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DEFAULT_INFO_FILE = os.path.join(PROJECT_DIR, "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt")
DEFAULT_TEST_FILE = os.path.join(PROJECT_DIR, "../data/Industrial_and_Scientific_dataset/test.jsonl")
DEFAULT_OUTPUT_DIR = CURRENT_DIR

PRIMARY_STYLES = [
    "electrical_electronics",
    "3d_printing",
    "adhesive_tape_sealant",
    "plumbing_fittings",
    "fasteners_hardware_tools",
    "raw_materials_metals",
    "lab_science",
    "medical_dental",
]

PREFERRED_REPLACEMENT_STYLES = {
    "electrical_electronics": ["medical_dental", "3d_printing", "lab_science"],
    "3d_printing": ["medical_dental", "plumbing_fittings", "lab_science"],
    "adhesive_tape_sealant": ["medical_dental", "3d_printing", "lab_science"],
    "plumbing_fittings": ["3d_printing", "medical_dental", "electrical_electronics"],
    "fasteners_hardware_tools": ["medical_dental", "3d_printing", "lab_science"],
    "raw_materials_metals": ["medical_dental", "3d_printing", "electrical_electronics"],
    "lab_science": ["3d_printing", "adhesive_tape_sealant", "fasteners_hardware_tools"],
    "medical_dental": ["3d_printing", "electrical_electronics", "plumbing_fittings"],
}


def _compiled_rules():
    return [(style_name, patterns) for style_name, patterns, _ in STYLE_RULES]


def _style_score(title: str, style_name: str) -> int:
    low = title.lower()
    for candidate_style, patterns in _compiled_rules():
        if candidate_style != style_name:
            continue
        return sum(1 for pattern in patterns if __import__("re").search(pattern, low))
    return 0


def _load_info_items(info_file: str):
    items = []
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            title = "\t".join(parts[:-1]).strip()
            item_id = int(parts[-1])
            style_group = _assign_style(title)
            style_score = _style_score(title, style_group) if style_group != "other_misc" else 0
            items.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "style_group": style_group,
                    "style_score": style_score,
                }
            )
    return items


def _build_style_pools(items, candidate_pool_size: int):
    by_style = defaultdict(list)
    for item in items:
        if item["style_group"] not in PRIMARY_STYLES:
            continue
        by_style[item["style_group"]].append(item)

    style_pools = {}
    for style_name, entries in by_style.items():
        entries = sorted(
            entries,
            key=lambda x: (
                -int(x["style_score"]),
                len(x["title"]),
                int(x["item_id"]),
            ),
        )
        style_pools[style_name] = entries[:candidate_pool_size]
    return style_pools


def _load_test_rows(test_file: str):
    rows = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _rank_candidate_samples(rows, min_history_len: int):
    ranked = []
    for row in rows:
        history_titles = row.get("history_item_titles", []) or []
        history_ids = row.get("history_item_ids", []) or []
        if len(history_ids) < min_history_len:
            continue

        history_styles = [_assign_style(title) for title in history_titles]
        filtered = [x for x in history_styles if x in PRIMARY_STYLES]
        if not filtered:
            continue

        counts = Counter(filtered)
        primary_style, primary_count = counts.most_common(1)[0]
        dominant_ratio = float(primary_count / max(len(history_styles), 1))
        target_style = _assign_style(str(row.get("target_item_title", "")))

        ranked.append(
            {
                "sample_idx": int(row.get("row_index", len(ranked))),
                "row_index": int(row.get("row_index", len(ranked))),
                "target_item_id": int(row.get("target_item_id", -1)),
                "target_item_title": str(row.get("target_item_title", "")),
                "target_style_group": target_style,
                "history_style_counts": dict(Counter(history_styles)),
                "primary_history_style_group": primary_style,
                "dominant_ratio": dominant_ratio,
                "history_len": len(history_ids),
            }
        )

    ranked.sort(
        key=lambda x: (
            -x["dominant_ratio"],
            -x["history_len"],
            x["sample_idx"],
        )
    )
    return ranked


def _choose_samples(ranked_samples, num_samples: int):
    per_style = defaultdict(list)
    for sample in ranked_samples:
        per_style[sample["primary_history_style_group"]].append(sample)

    ordered_styles = sorted(
        per_style.keys(),
        key=lambda style_name: (-len(per_style[style_name]), style_name),
    )

    selected = []
    used = set()
    while len(selected) < num_samples:
        progressed = False
        for style_name in ordered_styles:
            while per_style[style_name] and per_style[style_name][0]["sample_idx"] in used:
                per_style[style_name].pop(0)
            if not per_style[style_name]:
                continue
            sample = per_style[style_name].pop(0)
            if sample["sample_idx"] in used:
                continue
            used.add(sample["sample_idx"])
            selected.append(sample)
            progressed = True
            if len(selected) >= num_samples:
                break
        if not progressed:
            break
    return selected


def _build_plan_samples(selected_samples, style_pools):
    out = []
    for sample in selected_samples:
        primary_style = sample["primary_history_style_group"]
        target_groups = PREFERRED_REPLACEMENT_STYLES.get(primary_style, [])
        replace_item_ids_by_group = []
        # Expose every major style pool for every sample so inference can run
        # with a fixed replacement_style_group across the full 300-sample set.
        all_available_groups = []
        for style_group in list(target_groups) + [x for x in PRIMARY_STYLES if x not in target_groups]:
            if style_group not in all_available_groups:
                all_available_groups.append(style_group)

        for style_group in all_available_groups:
            pool = style_pools.get(style_group, [])
            replace_item_ids_by_group.append(
                {
                    "style_group": style_group,
                    "candidate_item_ids": [int(x["item_id"]) for x in pool],
                    "candidate_titles": [x["title"] for x in pool[:12]],
                }
            )

        out.append(
            {
                "sample_idx": sample["sample_idx"],
                "primary_history_style_group": primary_style,
                "target_style_group": sample["target_style_group"],
                "history_style_counts": sample["history_style_counts"],
                "dominant_ratio": sample["dominant_ratio"],
                "history_len": sample["history_len"],
                "target_item_id": sample["target_item_id"],
                "target_item_title": sample["target_item_title"],
                "suggested_opposite_style_groups": list(target_groups),
                "replace_item_ids_by_group": replace_item_ids_by_group,
            }
        )
    return out


def _write_outputs(output_dir: str, output_prefix: str, payload: dict):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{output_prefix}_replace_item_ids_plan.json")
    indices_path = os.path.join(output_dir, f"{output_prefix}_sample_indices.txt")
    summary_path = os.path.join(output_dir, f"{output_prefix}_replace_summary.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    sample_indices = [str(sample["sample_idx"]) for sample in payload["samples"]]
    with open(indices_path, "w", encoding="utf-8") as f:
        f.write(",".join(sample_indices) + "\n")

    lines = []
    lines.append(f"# {output_prefix} Replace Summary")
    lines.append("")
    lines.append(f"- num_samples: `{payload['num_samples']}`")
    lines.append(f"- primary_styles: `{', '.join(payload['primary_styles'])}`")
    lines.append("")
    lines.append("## Per-style counts")
    lines.append("")
    for style_name, count in payload["selected_primary_style_counts"].items():
        lines.append(f"- `{style_name}`: {count}")
    lines.append("")
    lines.append("## First 20 samples")
    lines.append("")
    for sample in payload["samples"][:20]:
        lines.append(f"### Sample {sample['sample_idx']}")
        lines.append("")
        lines.append(f"- target: {sample['target_item_title']}")
        lines.append(f"- primary_history_style: `{sample['primary_history_style_group']}`")
        lines.append(f"- target_style: `{sample['target_style_group']}`")
        lines.append(f"- dominant_ratio: `{sample['dominant_ratio']:.3f}`")
        lines.append(f"- history_len: `{sample['history_len']}`")
        lines.append(
            f"- replacement_groups: `{', '.join(sample['suggested_opposite_style_groups'])}`"
        )
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "json_path": json_path,
        "indices_path": indices_path,
        "summary_path": summary_path,
    }


def generate_plan(
    info_file: str = DEFAULT_INFO_FILE,
    test_file: str = DEFAULT_TEST_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_prefix: str = "strong_top300",
    num_samples: int = 300,
    min_history_len: int = 3,
    candidate_pool_size: int = 64,
):
    items = _load_info_items(info_file)
    style_pools = _build_style_pools(items, candidate_pool_size=candidate_pool_size)

    rows = _load_test_rows(test_file)
    ranked = _rank_candidate_samples(rows, min_history_len=min_history_len)
    if len(ranked) < num_samples:
        fallback_ranked = _rank_candidate_samples(rows, min_history_len=1)
        ranked_map = {sample["sample_idx"]: sample for sample in ranked}
        for sample in fallback_ranked:
            ranked_map.setdefault(sample["sample_idx"], sample)
        ranked = sorted(
            ranked_map.values(),
            key=lambda x: (-x["dominant_ratio"], -x["history_len"], x["sample_idx"]),
        )

    selected = _choose_samples(ranked, num_samples=num_samples)
    if len(selected) < num_samples:
        raise ValueError(f"可用样本只有 {len(selected)} 条，少于请求的 {num_samples} 条")

    samples = _build_plan_samples(selected, style_pools=style_pools)
    selected_primary_style_counts = Counter(sample["primary_history_style_group"] for sample in samples)

    payload = {
        "source": test_file,
        "info_file": info_file,
        "num_samples": len(samples),
        "selection_policy": {
            "primary_styles": list(PRIMARY_STYLES),
            "preferred_replacement_styles": PREFERRED_REPLACEMENT_STYLES,
            "min_history_len": min_history_len,
            "candidate_pool_size": candidate_pool_size,
            "ranking": "dominant_ratio desc, history_len desc, sample_idx asc, round-robin by primary style",
        },
        "primary_styles": list(PRIMARY_STYLES),
        "selected_primary_style_counts": dict(selected_primary_style_counts),
        "samples": samples,
    }

    output_paths = _write_outputs(output_dir=output_dir, output_prefix=output_prefix, payload=payload)
    print(f"[Saved] {output_paths['json_path']}")
    print(f"[Saved] {output_paths['indices_path']}")
    print(f"[Saved] {output_paths['summary_path']}")
    print(f"[Info] selected_primary_style_counts = {dict(selected_primary_style_counts)}")
    return {
        **output_paths,
        "num_samples": len(samples),
        "selected_primary_style_counts": dict(selected_primary_style_counts),
    }


if __name__ == "__main__":
    if fire is None:
        generate_plan()
    else:
        fire.Fire(generate_plan)
