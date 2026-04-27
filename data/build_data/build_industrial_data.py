#!/usr/bin/env python3
"""Build a ReRe-style Industrial_and_Scientific dataset.

The input CSV format matches data/Amazon/{train,valid,test}/*.csv produced by
data/process.py. By default this writes all rows; set --max-rows-per-split to a
positive value to generate a smaller subset.
"""

import argparse
import ast
import csv
import json
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INSTRUCTIONS_RERE_ORIGINAL = [
    "Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
    "Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
    "Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
    "Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
    "In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
    "Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
    "Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
    "In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
    "With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
    "Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
    "In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history.",
]

TRIMMABLE_BOUNDARY_CHARS = set(string.whitespace + string.punctuation)


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build an Industrial_and_Scientific ReRe-style dataset")
    parser.add_argument("--amazon-root", type=Path, default=data_dir / "Amazon")
    parser.add_argument("--category-prefix", type=str, default="Industrial_and_Scientific")
    parser.add_argument("--category-text", type=str, default="industrial and scientific items")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir / "Industrial_and_Scientific_dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=0,
        help="Rows per split. Use 0 or a negative value for all rows.",
    )
    parser.add_argument("--deterministic-instruction", action="store_true")
    parser.add_argument(
        "--with-item-spans",
        action="store_true",
        help="Also add history item char/token spans. Requires transformers and a local tokenizer.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Local tokenizer path used only with --with-item-spans.",
    )
    return parser.parse_args()


def find_csv(amazon_root: Path, split: str, prefix: str) -> Path:
    candidates = sorted((amazon_root / split).glob(f"{prefix}*.csv"))
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) if candidates else "<none>"
        raise RuntimeError(f"Expected exactly 1 CSV for split={split}, found {len(candidates)}: {names}")
    return candidates[0]


def parse_list_literal(value: str) -> List:
    if value is None or value == "":
        return []
    data = ast.literal_eval(value)
    return data if isinstance(data, list) else [data]


def build_history_text(history_titles: List[str], category_text: str) -> Tuple[str, str]:
    history = ",\t".join(f'"{title}"' for title in history_titles)
    history_str = "::".join(history_titles)
    input_text = f"The user has palyed the following {category_text}s before: {history}"
    return input_text, history_str


def build_prompt(instruction_text: str, input_text: str) -> str:
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request. \n\n"
        "### Instruction:\n"
        f"{instruction_text}\n "
        "### User Input: \n"
        f"{input_text}\n\n"
        "### Response:\n"
    )


def locate_history_char_spans(prompt: str, history_titles: List[str]) -> List[List[int]]:
    spans = []
    cursor = 0
    for title in history_titles:
        needle = f'"{title}"'
        start = prompt.find(needle, cursor)
        if start < 0:
            start = prompt.find(needle)
        if start < 0:
            raise ValueError(f"cannot locate title in prompt: {title!r}")
        end = start + len(needle)
        spans.append([start, end])
        cursor = end
    return spans


def char_span_to_token_span(offset_mapping, char_start: int, char_end: int) -> List[int]:
    token_indices = []
    for idx, (start, end) in enumerate(offset_mapping):
        if start == end or end <= char_start:
            continue
        if start >= char_end:
            break
        token_indices.append(idx)
    if not token_indices:
        raise ValueError(f"cannot map char span [{char_start}, {char_end}) to token span")
    return [token_indices[0], token_indices[-1] + 1]


def is_trimmable_boundary_text(text: str) -> bool:
    return not text or all(ch in TRIMMABLE_BOUNDARY_CHARS for ch in text)


def trim_token_span_boundaries(prompt: str, offset_mapping, token_span: List[int]) -> List[int]:
    start_idx, end_idx = token_span
    while start_idx < end_idx:
        char_start, char_end = offset_mapping[start_idx]
        if not is_trimmable_boundary_text(prompt[char_start:char_end]):
            break
        start_idx += 1
    while end_idx > start_idx:
        char_start, char_end = offset_mapping[end_idx - 1]
        if not is_trimmable_boundary_text(prompt[char_start:char_end]):
            break
        end_idx -= 1
    return [start_idx, end_idx]


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643
    return tokenizer


def add_item_spans(row: Dict, prompt: str, history_titles: List[str], history_ids: List[int], tokenizer) -> None:
    tokenized_prompt = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = tokenized_prompt["offset_mapping"]
    char_spans = locate_history_char_spans(prompt, history_titles)
    token_spans = []
    for char_start, char_end in char_spans:
        raw_span = char_span_to_token_span(offsets, char_start, char_end)
        token_spans.append(trim_token_span_boundaries(prompt, offsets, raw_span))

    row.update(
        {
            "history_item_titles": history_titles,
            "history_item_ids": history_ids,
            "history_item_char_spans": char_spans,
            "history_item_token_spans": token_spans,
            "prompt_input_ids": tokenized_prompt["input_ids"],
        }
    )


def process_split(
    split: str,
    csv_path: Path,
    category_text: str,
    rng: random.Random,
    deterministic_instruction: bool,
    max_rows: int,
    tokenizer: Optional[object],
) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, csv_row in enumerate(reader):
            if max_rows > 0 and len(rows) >= max_rows:
                break

            history_titles = [str(x) for x in parse_list_literal(csv_row["history_item_title"])]
            history_ids = [int(x) for x in parse_list_literal(csv_row["history_item_id"])]
            input_text, history_str = build_history_text(history_titles, category_text)

            if deterministic_instruction:
                instruction_tpl = INSTRUCTIONS_RERE_ORIGINAL[idx % len(INSTRUCTIONS_RERE_ORIGINAL)]
            else:
                instruction_tpl = rng.choice(INSTRUCTIONS_RERE_ORIGINAL)

            prompt = build_prompt(instruction_tpl.format(category=category_text), input_text)
            target_item_id = int(csv_row["item_id"])
            last_history_item_id = history_ids[-1] if history_ids else None
            row = {
                "split": split,
                "row_index": idx,
                "user_id": csv_row["user_id"],
                "prompt": prompt,
                "completion": f'"{csv_row["item_title"]}"\n',
                "history_str": history_str,
                "target_item_id": target_item_id,
                "target_item_title": csv_row["item_title"],
                "dedup": last_history_item_id == target_item_id,
            }
            if tokenizer is not None:
                add_item_spans(row, prompt, history_titles, history_ids, tokenizer)
            rows.append(row)
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.with_item_spans and not args.model_name:
        raise ValueError("--with-item-spans requires --model-name pointing to a local tokenizer")
    tokenizer = load_tokenizer(args.model_name) if args.with_item_spans else None

    counts: Dict[str, int] = {}
    inputs: Dict[str, str] = {}
    for split in ("train", "valid", "test"):
        csv_path = find_csv(args.amazon_root, split, args.category_prefix)
        inputs[split] = str(csv_path)
        rows = process_split(
            split=split,
            csv_path=csv_path,
            category_text=args.category_text,
            rng=rng,
            deterministic_instruction=args.deterministic_instruction,
            max_rows=args.max_rows_per_split,
            tokenizer=tokenizer,
        )
        save_jsonl(args.output_dir / f"{split}.jsonl", rows)
        counts[split] = len(rows)

    print("Done building Industrial_and_Scientific dataset")
    print(json.dumps(
        {
            "category_prefix": args.category_prefix,
            "category_text": args.category_text,
            "counts": counts,
            "total": sum(counts.values()),
            "inputs": inputs,
            "outputs": {
                "train": str(args.output_dir / "train.jsonl"),
                "valid": str(args.output_dir / "valid.jsonl"),
                "test": str(args.output_dir / "test.jsonl"),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
