import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# 从 prompt / history_str 里提取双引号包裹的标题
_QUOTED_PATTERN = re.compile(r'"([^"]+)"')


def load_item_mappings(info_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    读取 info 文件，构建 title<->id 映射。
    每行格式: {title}\t{id}
    """
    title2id: Dict[str, int] = {}
    id2title: Dict[int, str] = {}

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
            title2id[title] = item_id
            id2title[item_id] = title

    return title2id, id2title


def extract_history_titles(row: dict) -> List[str]:
    """
    优先使用 history_str（:: 分隔）；若为空，则回退到从 prompt 双引号抽取。
    """
    history_str = str(row.get("history_str", "")).strip()
    if history_str:
        out = [x.strip() for x in history_str.split("::") if x.strip()]
        if out:
            return out

    prompt = str(row.get("prompt", ""))
    return [x.strip() for x in _QUOTED_PATTERN.findall(prompt)]


def history_titles_to_ids(history_titles: List[str], title2id: Dict[str, int]) -> List[int]:
    ids: List[int] = []
    for t in history_titles:
        if t in title2id:
            ids.append(title2id[t])
    return ids


def left_truncate(ids: List[int], max_len: int) -> List[int]:
    if max_len <= 0:
        return ids
    return ids[-max_len:]


@dataclass
class RecTrainSample:
    seq_ids: List[int]
    target_id: int


class RecTrainDataset(Dataset):
    """
    用 full-rank jsonl 构造推荐器训练样本：
    - 输入: 用户历史序列（history_str）
    - 标签: target_item_id
    """

    def __init__(self, jsonl_file: str, title2id: Dict[str, int], max_seq_len: int = 50):
        self.samples: List[RecTrainSample] = []
        self.max_seq_len = max_seq_len

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                target_id = int(row.get("target_item_id", -1))
                if target_id < 0:
                    continue

                history_titles = extract_history_titles(row)
                history_ids = history_titles_to_ids(history_titles, title2id)
                history_ids = left_truncate(history_ids, max_seq_len)

                if len(history_ids) == 0:
                    continue

                self.samples.append(RecTrainSample(seq_ids=history_ids, target_id=target_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RecTrainSample:
        return self.samples[idx]


def build_padded_sequence(batch: List[RecTrainSample], padding_item_id: int, max_seq_len: int):
    """
    组 batch：输出固定长度序列 + 长度 + target。
    """
    bsz = len(batch)
    seq = torch.full((bsz, max_seq_len), fill_value=padding_item_id, dtype=torch.long)
    lengths = torch.zeros((bsz,), dtype=torch.long)
    targets = torch.zeros((bsz,), dtype=torch.long)

    for i, s in enumerate(batch):
        ids = left_truncate(s.seq_ids, max_seq_len)
        l = len(ids)
        seq[i, :l] = torch.tensor(ids, dtype=torch.long)
        lengths[i] = l
        targets[i] = s.target_id

    return {
        "seq": seq,
        "len_seq": lengths,
        "target": targets,
    }


def build_llara_augmented_prompt(
    original_prompt: str,
    history_titles: List[str],
    max_hist_in_prompt: int = 20,
) -> str:
    """
    在原始 ReRe prompt 上插入 LLaRA 风格行为锚点：
    - 每个历史 item 都对应一个 [HistoryEmb]
    - 不再构造“推荐出的cans列表”
    """
    history_titles = history_titles[-max_hist_in_prompt:] if max_hist_in_prompt > 0 else history_titles
    history_part = ", ".join([f'"{t}" [HistoryEmb]' for t in history_titles])

    rec_block = "\n### Behavioral Tokens:\n"
    rec_block += f"History anchors: {history_part}\n"

    if "### Response:" in original_prompt:
        left, right = original_prompt.split("### Response:", 1)
        return f"{left}{rec_block}\n### Response:{right}"

    return f"{original_prompt}\n{rec_block}\n### Response:\n"
