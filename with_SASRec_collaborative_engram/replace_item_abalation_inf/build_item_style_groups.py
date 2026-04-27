import json
import os
import re
from html import unescape
from collections import Counter, defaultdict

try:
    import fire
except ImportError:
    fire = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DEFAULT_INFO_FILE = os.path.join(PROJECT_DIR, "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt")
DEFAULT_OUTPUT_DIR = CURRENT_DIR


# Rule order still matters as a tie-breaker, but we no longer return on the
# first match. Instead, styles compete by number of matched patterns so mixed
# titles (for example "wire connector" vs. "wiring tape") are more stable.
STYLE_RULES = [
    (
        "3d_printing",
        [
            r"3d printer",
            r"3d printing",
            r"\b1\.75 ?mm\b",
            r"\b2\.85 ?mm\b",
            r"\bpla\b",
            r"\babs\b",
            r"\bpetg\b",
            r"\btpu\b",
            r"filament",
            r"makerbot",
            r"flashforge",
            r"creator pro",
            r"\bdreamer\b",
            r"\bprusa\b",
            r"\bextruder\b",
            r"\bhotend\b",
            r"\bnozzle\b",
            r"build plate",
            r"platform sticker",
        ],
        "3D printer filament and printer-adjacent consumables",
    ),
    (
        "medical_dental",
        [
            r"toothbrush",
            r"toothpaste",
            r"dental",
            r"\boral\b",
            r"oral-b",
            r"colgate",
            r"sonicare",
            r"\bmedical\b",
            r"first aid",
            r"eye flush",
            r"gauze",
            r"burn spray",
            r"exam gloves",
            r"thermometer",
            r"infrared thermometer",
            r"enema",
            r"nitrile exam",
            r"nitrile gloves",
        ],
        "Medical, dental, oral-care, thermometer, and exam-supply items",
    ),
    (
        "lab_science",
        [
            r"beaker",
            r"pipette",
            r"microscope",
            r"laboratory",
            r"centrifuge",
            r"test strips",
            r"\bph\b",
            r"\btds\b",
            r"conductivity",
            r"hydrometer",
            r"borosilicate",
            r"graduated",
            r"measuring cylinder",
            r"\bflask\b",
            r"erlenmeyer",
            r"boiling flask",
            r"test tube",
            r"alcohol lamp",
            r"dropper bottle",
        ],
        "Lab glassware, measurement, microscopy, and test tools",
    ),
    (
        "plumbing_fittings",
        [
            r"pipe fitting",
            r"tube fitting",
            r"\bpipe\b",
            r"\btubing\b",
            r"\bvalve\b",
            r"\bfaucet\b",
            r"\bpex\b",
            r"\bbarbed?\b",
            r"\bnpt\b",
            r"\bcoupling\b",
            r"\bbushing\b",
            r"bulkhead",
            r"hose coupling",
            r"water filter",
            r"water pressure regulator",
            r"pressure regulator",
            r"push[- ]?to[- ]?connect",
            r"push fit",
            r"\belbow\b",
            r"\btee\b",
            r"\breducer\b",
            r"\bslip\b",
            r"\bpvc\b",
        ],
        "Pipe fittings, valves, couplings, tubing, and plumbing hardware",
    ),
    (
        "adhesive_tape_sealant",
        [
            r"\btape\b",
            r"\bglue\b",
            r"adhesive",
            r"epoxy",
            r"sealant",
            r"\bcaulk\b",
            r"self fusing",
            r"cement board tape",
            r"threadlocker",
            r"hook/loop",
            r"hook and loop",
            r"duct tape",
            r"gaffer tape",
            r"foil tape",
            r"butyl tape",
            r"vinyl cement",
            r"furnace cement",
            r"\brtv\b",
            r"super glue",
        ],
        "Tape, glue, adhesive, epoxy, and sealing products",
    ),
    (
        "electrical_electronics",
        [
            r"electrical",
            r"\bwire\b",
            r"connector",
            r"voltage",
            r"\bcable\b",
            r"relay",
            r"oscilloscope",
            r"\busb\b",
            r"heat shrink",
            r"power supply",
            r"\bprobe\b",
            r"amplifier board",
            r"terminal",
            r"thermostat",
            r"temperature controller",
            r"thermocouple",
            r"\bpcb\b",
            r"prototype board",
            r"breadboard",
            r"\bled\b",
            r"resistor",
            r"\bfuse\b",
            r"switch",
        ],
        "Electrical, wiring, relay, connector, thermostat, and electronics parts",
    ),
    (
        "fasteners_hardware_tools",
        [
            r"\bscrew\b",
            r"\bbolt\b",
            r"\bnut\b",
            r"\bwasher\b",
            r"\banchor\b",
            r"\blatch\b",
            r"\bvise\b",
            r"\bpruner\b",
            r"\bwrench\b",
            r"\bdrill\b",
            r"\btoggle\b",
            r"\brivet\b",
            r"\bhex key\b",
            r"\btool\b",
            r"inspection mirror",
            r"toolbox",
        ],
        "Fasteners, hand tools, anchors, latches, and hardware kits",
    ),
    (
        "janitorial_cleaning",
        [
            r"\btrash\b",
            r"\bwipes\b",
            r"\bvacuum\b",
            r"\bcleaner\b",
            r"\bdispenser\b",
            r"\btowel\b",
            r"\btissue\b",
            r"wet/dry",
            r"\bsoap\b",
            r"trash bags",
            r"vacuum cleaner",
        ],
        "Cleaning, janitorial, trash, and dispenser products",
    ),
    (
        "safety_ppe",
        [
            r"\bn95\b",
            r"\brespirator\b",
            r"safety can",
            r"safety wash",
            r"\bgoggles\b",
            r"\bmask\b",
            r"hard hat",
            r"face masks",
        ],
        "Protective gear, masks, and safety containers",
    ),
    (
        "office_labels_packaging",
        [
            r"shipping tags",
            r"\blabel\b",
            r"\blabels\b",
            r"\bsticker\b",
            r"\bmanila\b",
            r"\bpacking\b",
            r"\bbarcode\b",
            r"storage bags",
        ],
        "Labels, tags, stickers, packaging, and storage bags",
    ),
    (
        "raw_materials_metals",
        [
            r"round rod",
            r"\baluminum\b",
            r"\bacetal\b",
            r"steel wool",
            r"stainless steel",
            r"\bbrass\b",
            r"\bcopper\b",
            r"\bsheet\b",
            r"\bplate\b",
            r"\bingot\b",
            r"key stock",
            r"cut-off wheels for metal",
        ],
        "Raw stock, rods, metal sheets, and material stock pieces",
    ),
]


COMPILED_STYLE_RULES = [
    (style_name, [re.compile(pattern) for pattern in patterns], desc)
    for style_name, patterns, desc in STYLE_RULES
]


def _normalize_title(title: str) -> str:
    text = unescape(title).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _assign_style(title: str) -> str:
    low = _normalize_title(title)
    best_style = "other_misc"
    best_score = 0

    for style_name, patterns, _desc in COMPILED_STYLE_RULES:
        score = sum(1 for pattern in patterns if pattern.search(low))
        if score > best_score:
            best_style = style_name
            best_score = score

    if best_score > 0:
        return best_style
    return "other_misc"


def build_groups(
    info_file: str = DEFAULT_INFO_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    items = []
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            title = "\t".join(parts[:-1]).strip()
            item_id = int(parts[-1].strip())
            style_name = _assign_style(title)
            items.append(
                {
                    "item_id": item_id,
                    "title": title,
                    "style_group": style_name,
                }
            )

    style_to_items = defaultdict(list)
    for item in items:
        style_to_items[item["style_group"]].append(item)

    counts = Counter(item["style_group"] for item in items)
    style_descriptions = {name: desc for name, _patterns, desc in STYLE_RULES}
    style_descriptions["other_misc"] = "Unmatched miscellaneous industrial/scientific items"

    grouped_json = {
        "info_file": info_file,
        "total_items": len(items),
        "style_groups": {},
    }
    for style_name in sorted(style_to_items.keys()):
        entries = sorted(style_to_items[style_name], key=lambda x: x["item_id"])
        grouped_json["style_groups"][style_name] = {
            "description": style_descriptions.get(style_name, ""),
            "count": len(entries),
            "example_titles": [x["title"] for x in entries[:10]],
            "item_ids": [x["item_id"] for x in entries],
            "items": entries,
        }

    grouped_json_path = os.path.join(output_dir, "item_style_groups.json")
    with open(grouped_json_path, "w", encoding="utf-8") as f:
        json.dump(grouped_json, f, ensure_ascii=False, indent=2)

    mapping_tsv_path = os.path.join(output_dir, "item_id_to_style.tsv")
    with open(mapping_tsv_path, "w", encoding="utf-8") as f:
        f.write("item_id\tstyle_group\ttitle\n")
        for item in sorted(items, key=lambda x: x["item_id"]):
            safe_title = item["title"].replace("\t", " ")
            f.write(f"{item['item_id']}\t{item['style_group']}\t{safe_title}\n")

    summary_md_path = os.path.join(output_dir, "item_style_groups_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Item Style Groups\n\n")
        f.write(f"- Source: `{info_file}`\n")
        f.write(f"- Total items: `{len(items)}`\n\n")
        f.write("## Group Counts\n\n")
        for style_name, count in counts.most_common():
            f.write(f"- `{style_name}`: {count}\n")
        f.write("\n## Group Examples\n\n")
        for style_name, count in counts.most_common():
            f.write(f"### {style_name}\n\n")
            f.write(f"- Description: {style_descriptions.get(style_name, '')}\n")
            f.write(f"- Count: {count}\n")
            for entry in style_to_items[style_name][:8]:
                f.write(f"- `{entry['item_id']}` {entry['title']}\n")
            f.write("\n")

    print(f"[Saved] {grouped_json_path}")
    print(f"[Saved] {mapping_tsv_path}")
    print(f"[Saved] {summary_md_path}")
    print("[Counts]")
    for style_name, count in counts.most_common():
        print(f"{style_name}: {count}")

    return {
        "grouped_json_path": grouped_json_path,
        "mapping_tsv_path": mapping_tsv_path,
        "summary_md_path": summary_md_path,
        "counts": dict(counts),
    }


if __name__ == "__main__":
    if fire is None:
        build_groups()
    else:
        fire.Fire(build_groups)
