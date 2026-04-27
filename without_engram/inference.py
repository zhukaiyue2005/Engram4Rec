import json
import math
import time
from typing import Dict, List

# Work around a torchrun/static-TLS loader issue by initializing sklearn
# before torch/accelerate/peft/transformers pull in competing native libs.
from sklearn.metrics import roc_curve as _sklearn_roc_curve  # noqa: F401

import fire
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    Qwen3ForCausalLM,
)


def _get_hash(x: List[int]) -> str:
    # 把token序列转为可哈希字符串，用于前缀约束字典查询
    return "-".join([str(i) for i in x])


class ConstrainedLogitsProcessor(LogitsProcessor):
    # 作用：在每一步生成时，只允许输出“在候选标题集合中仍然合法”的下一个token。
    # 直观上等价于：把生成过程限制在一棵“候选标题前缀Trie”上。
    def __init__(self, prefix_allowed_tokens_fn, num_beams: int, base_model: str):
        # prefix_allowed_tokens_fn: 输入当前前缀token序列 -> 返回允许的下一token列表
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        # count表示“当前已经生成了多少个新token”，用于切换前缀切片策略
        self.count = 0
        # 不同模型分词前缀长度不同：gpt2用4，其它（如qwen/llama）用3
        # 这里与_build_constraint_dict中保持一致，确保查表键对齐
        self.prefix_index = 4 if "gpt2" in base_model.lower() else 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # beam search下先转log_softmax，再通过mask限制可选token
        # 输入:
        # - input_ids: [batch_size*num_beams, cur_len]
        # - scores:    [batch_size*num_beams, vocab_size]
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        # 默认把所有token都置为极小值（近似负无穷），表示全部禁止
        mask = torch.full_like(scores, -1_000_000)

        # 还原成 [batch_size, num_beams, cur_len]，逐样本逐beam做约束
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                # 第一步生成时，用固定长度前缀（prefix_index）作为约束键；
                # 后续步骤用“已生成长度count”对应的后缀作为键，逐步扩展约束窗口。
                if self.count == 0:
                    hash_key = sent[-self.prefix_index :].tolist()
                else:
                    hash_key = sent[-self.count :].tolist()
                # 查约束字典得到下一步允许的token id集合
                allowed = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                if allowed:
                    # 允许token的位置置0，其他位置保持极小值（相当于屏蔽）
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0

        # 下一次调用对应“再往后生成一个token”
        self.count += 1
        # 最终分数 = 原始log概率 + 掩码（非法token几乎不可能被选中）
        return scores + mask


def _build_constraint_dict(tokenizer, base_model: str, info_file: str) -> Dict[str, List[int]]:
    # 从候选商品文件构建“前缀 -> 合法下一token集合”字典
    # 字典键: 由_get_hash编码后的token前缀字符串
    # 字典值: 在该前缀下允许继续生成的token id列表
    with open(info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # ReRe格式：每个候选标题用 "title"\n，并拼到 ### Response:\n 后作为可生成目标
    item_names = ["\"" + ln[: -len(ln.split("\t")[-1])].strip() + "\"\n" for ln in lines]
    response_texts = [f"### Response:\n{name}" for name in item_names]

    if "llama" in base_model.lower():
        prefix_ids = [tokenizer(t).input_ids[1:] for t in response_texts]
    else:
        prefix_ids = [tokenizer(t).input_ids for t in response_texts]

    # 与ConstrainedLogitsProcessor保持一致，确保prefix切片/查表规则一致
    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    hash_dict: Dict[str, set] = {}
    for ids in prefix_ids:
        # 手动拼上eos，保证“标题完整结束”也是可达路径
        ids = ids + [tokenizer.eos_token_id]
        # 逐位置建立转移: prefix -> next_token
        for i in range(prefix_index, len(ids)):
            # i == prefix_index: 使用完整起始前缀 ids[:i]
            # i >  prefix_index: 使用去掉模板头后的相对前缀 ids[prefix_index:i]
            # 这样可与生成阶段 sent[-k:] 的键对齐
            key = _get_hash(ids[:i]) if i == prefix_index else _get_hash(ids[prefix_index:i])
            if key not in hash_dict:
                hash_dict[key] = set()
            hash_dict[key].add(ids[i])
    # set转list，方便后续直接喂给transformers约束接口
    return {k: list(v) for k, v in hash_dict.items()}


def _normalize_target(row: dict) -> str:
    # 统一目标标题字段，兼容 target_item_title / completion
    target = row.get("target_item_title", "")
    if target:
        return target.strip()
    raw = str(row.get("completion", "")).strip().strip("\n")
    return raw.strip('"')


def _compute_metrics(preds: List[List[str]], targets: List[str], topk: List[int]) -> Dict[str, float]:
    # 计算全排序推荐指标：Hit@K / NDCG@K
    # 参数说明：
    # - preds:   每条样本的预测排序列表，例如 preds[i] = [候选1, 候选2, ...]
    # - targets: 每条样本的真实目标标题，长度应与preds一致
    # - topk:    需要评估的K集合，如 [1,3,5,10]
    # 返回：
    # - {"Hit@1":..., "NDCG@1":..., "Hit@3":..., ...}
    # - Hit@K: 目标是否出现在前K名（出现记1，否则记0），再对样本求平均
    # - NDCG@K: 若目标出现在前K名第rank位，则贡献 1/log2(rank+1)，否则为0，再求平均
    n = len(targets)
    out: Dict[str, float] = {}
    if n == 0:
        # 空数据保护：避免除零，直接返回0
        for k in topk:
            out[f"Hit@{k}"] = 0.0
            out[f"NDCG@{k}"] = 0.0
        return out

    for k in topk:
        # hit/ndcg 累积的是“所有样本在该K下的总贡献”
        hit, ndcg = 0.0, 0.0
        for pred_list, target in zip(preds, targets):
            # rank=-1 表示目标未命中前K
            rank = -1
            for i, p in enumerate(pred_list[:k]):
                if p == target:
                    # 排名从1开始计数（i是0-based）
                    rank = i + 1
                    break
            if rank != -1:
                # 命中：Hit贡献1
                hit += 1.0
                # 命中：NDCG贡献为折损增益，越靠前贡献越大
                ndcg += 1.0 / math.log2(rank + 1.0)
        # 对样本数取平均，得到最终Hit@K / NDCG@K
        out[f"Hit@{k}"] = hit / n
        out[f"NDCG@{k}"] = ndcg / n
    return out


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
    values = []
    if isinstance(arg_value, (tuple, list)):
        for x in arg_value:
            if isinstance(x, str):
                values.extend([p.strip() for p in x.split(",") if p.strip()])
            elif x is not None:
                values.append(x)
    elif isinstance(arg_value, str):
        values = [p.strip() for p in arg_value.split(",") if p.strip()]
    elif arg_value is not None:
        values = [arg_value]

    out: List[int] = []
    for x in values:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            raise ValueError(f"{arg_name}中包含非法整数值: {x!r}, 原始输入: {arg_value!r}")
    return out


def inference(
    batch_size: int = 4,
    resume_from_checkpoint: str = "",
    base_model: str = "",
    test_file: str = "../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    max_new_tokens: int = 256,
    eval_topk: str = "1,3,5,10",
    length_penalty: float = 0.0,
    save_json: str = "",
    print_batch_output: bool = True,
    print_prompt: bool = True,
    print_topn: int = 10,
    prompt_preview_chars: int = 0,
):
    # 1) 模型加载方式对齐 Industrial_and_Scientific/inference.py
    # 使用4bit量化加载基础模型，随后可选加载LoRA权重
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = Qwen3ForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    # 与原推理脚本保持一致：先prepare，再按需加载PEFT
    model = prepare_model_for_kbit_training(model)
    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    model.eval()

    # 分词器配置与训练保持一致（左padding + 固定bos/eos/pad）
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    # 2) 构建ReRe同款约束
    # 约束生成结果必须落在info候选集合内，保证输出可用于ranking评估
    hash_dict = _build_constraint_dict(tokenizer, base_model, info_file)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(_get_hash(input_ids), [])

    # 3) 数据
    # full-rank数据格式：每条样本包含prompt和目标item
    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    prompts = [row["prompt"] for row in dataset]
    targets = [_normalize_target(row) for row in dataset]
    topk = sorted(set(_parse_int_list_arg(eval_topk, "eval_topk")))
    if not topk:
        raise ValueError(f"eval_topk不能为空，当前值: {eval_topk}")
    if min(topk) <= 0:
        raise ValueError(f"eval_topk中的K必须为正整数，当前值: {eval_topk}")
    num_beams = max(topk)

    # 4) full-rank推理
    # 每个prompt通过beam search返回num_beams条候选，即rank list
    all_preds: List[List[str]] = []
    total = len(prompts)
    steps = (total + batch_size - 1) // batch_size
    start_time = time.time()
    for step in range(steps):
        s = step * batch_size
        e = min(total, s + batch_size)
        batch_prompts = prompts[s:e]

        model_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        prompt_len = model_inputs["input_ids"].shape[1]

        generation_config = GenerationConfig(
            # 关键：num_return_sequences=num_beams 才能得到完整rank list
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
        logits_processor = LogitsProcessorList(
            [
                ConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    base_model=base_model,
                )
            ]
        )

        with torch.no_grad():
            generation_output = model.generate(
                **model_inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        completion_ids = generation_output.sequences[:, prompt_len:]
        if "llama" in base_model.lower():
            decoded = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 只保留Response后半段，并去掉外围引号
        decoded = [x.split("Response:\n")[-1].strip().strip('"').strip() for x in decoded]
        batch_preds: List[List[str]] = []
        for i in range(0, len(decoded), num_beams):
            one_pred = decoded[i : i + num_beams]
            all_preds.append(one_pred)
            batch_preds.append(one_pred)

        if print_batch_output:
            # print_topn<=0 时，默认展示全部生成结果（即 K_max 条）
            current_topn = num_beams if print_topn <= 0 else min(max(1, print_topn), num_beams)
            # 按“批次 + 样本索引”打印，便于和原始数据逐条对齐排查
            print(f"\n=== Batch {step + 1}/{steps} | samples {s}-{e - 1} ===", flush=True)
            for local_i, pred_list in enumerate(batch_preds):
                data_i = s + local_i
                # 先打印目标标题，快速判断topN是否命中
                print(f"[Sample {data_i}] target: {targets[data_i]}", flush=True)
                if print_prompt:
                    # prompt打印采用单行预览，避免控制台被长文本淹没
                    prompt_text = prompts[data_i].replace("\n", "\\n")
                    if prompt_preview_chars > 0 and len(prompt_text) > prompt_preview_chars:
                        prompt_text = prompt_text[:prompt_preview_chars] + "..."
                    print(f"[Sample {data_i}] prompt: {prompt_text}", flush=True)
                # 打印该样本前N个rank候选（N默认等于K_max）
                print(f"[Sample {data_i}] top{current_topn}: {pred_list[:current_topn]}", flush=True)

        # 显示推理进度与预计剩余时间
        done_steps = step + 1
        progress = (done_steps / steps) * 100 if steps > 0 else 100.0
        elapsed = time.time() - start_time
        avg_step_time = elapsed / done_steps if done_steps > 0 else 0.0
        eta = avg_step_time * (steps - done_steps)
        print(
            f"[Progress] {done_steps}/{steps} ({progress:.2f}%) | "
            f"elapsed: {_format_seconds(elapsed)} | eta: {_format_seconds(eta)}",
            flush=True,
        )

    # 5) Hit@K / NDCG@K
    # 生成条数固定为max(topk)，这里只需按请求的k集合评估
    metrics = _compute_metrics(all_preds, targets, topk)
    print("=== Metrics ===", flush=True)
    for k in topk:
        print(f"Hit@{k}: {metrics[f'Hit@{k}']:.6f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}", flush=True)

    if save_json:
        # 保存逐样本预测明细：
        # - target_title: 规范化后的真实目标
        # - predict: 长度为K_max的排序结果
        rows = []
        for i, row in enumerate(dataset):
            one = dict(row)
            one["target_title"] = targets[i]
            one["predict"] = all_preds[i]
            rows.append(one)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"saved predictions to: {save_json}", flush=True)


if __name__ == "__main__":
    fire.Fire(inference)
