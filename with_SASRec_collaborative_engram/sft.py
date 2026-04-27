import os
import sys
import math
import ast
import numpy as np
import torch
import pandas as pd
import re
import wandb
import random
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
import transformers
from transformers import PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig,LlamaTokenizer,TrainerCallback,EarlyStoppingCallback
from datasets import load_dataset,Dataset,load_from_disk
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType,PeftModel
from trl import SFTConfig
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Union
import torch.optim as optim
#from DataCollator import DataCollatorForCompletionOnlyLM

os.environ['HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION'] = '1'
os.environ['TORCH_NN_MODULE_USE_DTENSOR'] = '0'
os.environ['USE_DTENSOR'] = '0'
os.environ['ACCELERATE_USE_DTENSOR'] = '0'
os.environ["USE_FLASH_ATTENTION"] = "0" 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_cache")
os.environ.setdefault("HF_HOME", LOCAL_HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(LOCAL_HF_HOME, "datasets"))
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
from accelerate import Accelerator


def resolve_demo_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def load_item_mappings(info_file: str):
    title2id = {}
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            title2id["\t".join(parts[:-1]).strip()] = int(parts[-1])
    return title2id


def split_history_titles(history_str: str):
    history_str = str(history_str or "").strip()
    if not history_str:
        return []
    if "::" in history_str:
        return [x.strip() for x in history_str.split("::") if x.strip()]
    return [x.strip().strip('"') for x in history_str.split('", "') if x.strip()]


def locate_history_token_spans(prompt_text: str, history_titles, tokenizer, title2id, cutoff_len: int):
    tokenized = tokenizer(
        prompt_text,
        truncation=True,
        max_length=cutoff_len,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    item_ids = []
    token_spans = []
    cursor = 0
    for title in history_titles:
        if title not in title2id:
            continue
        quoted = f'"{title}"'
        char_start = prompt_text.find(quoted, cursor)
        match_text = quoted
        if char_start < 0:
            char_start = prompt_text.find(title, cursor)
            match_text = title
        if char_start < 0:
            char_start = prompt_text.find(title)
            match_text = title
        if char_start < 0:
            continue
        char_end = char_start + len(match_text)
        token_indices = []
        for idx, (start, end) in enumerate(offsets):
            if start == end or end <= char_start:
                continue
            if start >= char_end:
                break
            token_indices.append(idx)
        if token_indices:
            item_ids.append(int(title2id[title]))
            token_spans.append([token_indices[0], token_indices[-1] + 1])
        cursor = char_end
    return item_ids, token_spans

# 强制重写huggingface_hub的校验函数
from huggingface_hub.utils._validators import validate_repo_id
def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return  # 空函数，跳过所有校验
validate_repo_id.__code__ = _dummy_validate_repo_id.__code__  
engram_dir = os.path.join(os.path.dirname(__file__), "Engram_Insert_code")
sys.path.append(engram_dir)
from engram import Engram, EngramConfig
from modeling_qwen3 import Qwen3ForCausalLM
import warnings
import torch
import bitsandbytes as bnb
import fire

# Transformer 模型的每一层（如注意力层、全连接层）都有固定的输入维度：
# 比如某层的输入维度是 [batch_size, seq_len, hidden_dim]
# batch_size = 样本数，seq_len = 序列长度，hidden_dim = 隐藏层维度）
# 批量处理（一次处理多个句子）比逐个处理效率高 10 倍以上，使用矩阵张量运算

import logging

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，调整为 DEBUG 以记录更多日志
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # 输出到控制台
)


# 关闭多余的输出：确保WandB不会重复输出
os.environ["WANDB_SILENT"] = "true" 

# 设置全局随机种子
def set_deterministic_mode(seed=1958):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 添加CUDA确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(f"✅ 确定性模式已启用，随机种子: {seed}")

# 在推理开始前调用
set_deterministic_mode(seed=1958)

class DataCollatorForCompletionOnlyLMWithItemMask(DataCollatorForCompletionOnlyLM):
    def __init__(
        self,
        *args,
        sasrec_item_embeddings: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sasrec_item_embeddings = sasrec_item_embeddings
        self._debug_print_done = False
        self._sasrec_debug_log = os.path.join(SCRIPT_DIR, "sasrec_token_states_debug.log")
    
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        examples = deepcopy(examples)
        completion_starts = []
        raw_input_lens = []
        history_item_ids_list = []
        history_item_token_spans_list = []
        for ex in examples:
            raw_input_lens.append(len(ex["input_ids"]))
            completion_starts.append(ex.pop("completion_start", None))
            history_item_ids_list.append(ex.pop("history_item_ids", []))
            history_item_token_spans_list.append(ex.pop("history_item_token_spans", []))

        if self.padding_free:
            raise ValueError("当前训练配置只支持 padding_free=False")

        batch = self.tokenizer.pad(
            examples,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        seq_len = batch["input_ids"].shape[1]

        if self.sasrec_item_embeddings is None:
            raise ValueError("当前训练配置要求提供 sasrec_item_embeddings")

        # 这里是 SASRec 嵌入进入 Engram 链路的第一步：
        # - `self.sasrec_item_embeddings` 是训练好的 SASRec item embedding 表；
        # - `history_item_ids + history_item_token_spans` 决定每个 item 向量应该铺到哪些 token 上；
        # - 虽然一句话里通常只有几个 item，但模型看到的是整条 token 序列，
        #   所以后面传给 Engram 的表示必须按 token 对齐，而不是按 item 对齐；
        # - 因此这里最终构造的是 token 级别的 `sasrec_token_states: [B, T, H]`：
        #     B = batch size
        #     T = 当前 batch pad 后的序列 token 长度
        #     H = SASRec item embedding 维度
        # - 对非 item 位置，向量全 0；
        # - 对某个 item 的 span 范围内，每个 token 位置都填同一个 item 向量。
        hidden_size = int(self.sasrec_item_embeddings.shape[1])
        padded_sasrec_states = []
        for ids, spans, raw_len in zip(history_item_ids_list, history_item_token_spans_list, raw_input_lens):
            token_states = torch.zeros((seq_len, hidden_size), dtype=self.sasrec_item_embeddings.dtype)
            pad_len = seq_len - int(raw_len)
            for item_id, span in zip(ids, spans):
                if span is None or len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if self.tokenizer.padding_side == "left":
                    start += pad_len
                    end += pad_len
                start = max(0, min(start, seq_len))
                end = max(start, min(end, seq_len))
                if end <= start:
                    continue
                if item_id < 0 or item_id >= self.sasrec_item_embeddings.shape[0]:
                    continue
                item_vec = self.sasrec_item_embeddings[int(item_id)]
                token_states[start:end] = item_vec.unsqueeze(0).expand(end - start, -1)
            padded_sasrec_states.append(token_states)

        batch["sasrec_token_states"] = torch.stack(padded_sasrec_states, dim=0)

        labels = torch.full_like(batch["input_ids"], -100)
        for i, cs in enumerate(completion_starts):
            if cs is None:
                continue
            cs = int(cs)
            raw_len = int(raw_input_lens[i])
            pad_len = seq_len - raw_len
            if self.tokenizer.padding_side == "left":
                start = pad_len + cs
                end = seq_len
            else:
                start = cs
                end = raw_len
            start = max(0, min(start, seq_len))
            end = max(start, min(end, seq_len))
            labels[i, start:end] = batch["input_ids"][i, start:end]

        batch["labels"] = labels

        if not self._debug_print_done:
            debug_lines = []
            debug_lines.append("=" * 80)
            debug_lines.append("[SASRec Debug] collator received SASRec item embedding table")
            debug_lines.append(f"sasrec_item_embeddings.shape = {tuple(self.sasrec_item_embeddings.shape)}")
            debug_lines.append(f"sasrec_item_embeddings.dtype = {self.sasrec_item_embeddings.dtype}")
            debug_lines.append(f"sasrec_item_embeddings[0][:8] = {self.sasrec_item_embeddings[0][:8].tolist()}")

            if history_item_ids_list:
                first_ids = history_item_ids_list[0]
                first_spans = history_item_token_spans_list[0]
                first_states = batch["sasrec_token_states"][0]
                nonzero_rows = (first_states.abs().sum(dim=-1) > 0).nonzero(as_tuple=False).squeeze(-1).tolist()

                debug_lines.append("")
                debug_lines.append("[SASRec Debug] first sample alignment")
                debug_lines.append(f"history_item_ids = {first_ids}")
                debug_lines.append(f"history_item_token_spans = {first_spans}")
                debug_lines.append(f"sasrec_token_states.shape = {tuple(batch['sasrec_token_states'].shape)}")
                debug_lines.append(f"nonzero token rows in first sample = {nonzero_rows[:80]}")

                for idx, (item_id, span) in enumerate(zip(first_ids, first_spans)):
                    if span is None or len(span) != 2:
                        continue
                    start, end = int(span[0]), int(span[1])
                    start = max(0, min(start, first_states.shape[0]))
                    end = max(start, min(end, first_states.shape[0]))
                    if end <= start:
                        continue
                    token_vec = first_states[start]
                    table_vec = self.sasrec_item_embeddings[int(item_id)]
                    debug_lines.append(
                        f"item[{idx}] id={item_id} span={span} "
                        f"token_state[:8]={token_vec[:8].tolist()} "
                        f"table_vec[:8]={table_vec[:8].tolist()}"
                    )
            debug_lines.append("=" * 80)
            with open(self._sasrec_debug_log, "w", encoding="utf-8") as f:
                f.write("\n".join(debug_lines) + "\n")
            self._debug_print_done = True

        return batch

def train(
    # 训练输出目录，用于保存模型检查点、日志等
    output_dir="",   
    # 日志目录，用于存储训练过程中的日志文件  
    logging_dir="",  
    # 基础模型名称或路径  
    model_name ="",    
    resume_from_checkpoint: str = None,  
    # wandb config
    wandb_project: str = "",       
    wandb_name: str = "",
    # training hyperparameters
    # 梯度累积步数，模拟更大的批大小，解决显存不足问题
    gradient_accumulation_steps: int = 1, 
    batch_size: int = 8,
    num_train_epochs: int = 10,
    lora_lr: float = 2e-5,          # LoRA参数学习率（主学习率）
    engram_lr: float = 5e-5,        # Engram模块学习率（可单独调大/小）
    learning_rate: float = 2e-5,
    engram_layer_ids: str = "6,13,20",
    sasrec_checkpoint_path: str = "./SAS-checkpoints/sasrec_best.pt",
    train_file: str = "../data/Industrial_and_Scientific_dataset/train.jsonl",
    eval_file: str = "../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    # 最大序列长度，超过该长度的输入将被截断
    cutoff_len: int = 2048,  
    # eval_step：评估频率，表示每训练多少比例的epoch进行一次评估
    eval_step = 0.1,
    response_template: str = "### Response:\n",
):  
    os.environ['WANDB_PROJECT'] = wandb_project 
    wandb.init(name=wandb_name, project=wandb_project, settings=wandb.Settings(silent=True))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,                 
        local_files_only=True, 
    )

    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Fire 可能把 `6,13,20` 解析成：
    # - "6,13,20"
    # - "(6, 13, 20)"
    # - "[6, 13, 20]"
    # - (6, 13, 20)
    # 这里统一转成整数列表。
    if isinstance(engram_layer_ids, (list, tuple)):
        layer_ids = [int(x) for x in engram_layer_ids]
    else:
        layer_ids_text = str(engram_layer_ids).strip()
        if layer_ids_text and layer_ids_text[0] in "([":
            parsed_layer_ids = ast.literal_eval(layer_ids_text)
            if isinstance(parsed_layer_ids, (list, tuple)):
                layer_ids = [int(x) for x in parsed_layer_ids]
            else:
                layer_ids = [int(parsed_layer_ids)]
        else:
            layer_ids = [int(x.strip()) for x in layer_ids_text.split(",") if x.strip()]
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    engram_config = EngramConfig()
    engram_config.layer_ids = layer_ids
    print(f"[Engram] layer_ids = {layer_ids}")

    sasrec_checkpoint_path = resolve_demo_path(sasrec_checkpoint_path)
    info_file = resolve_demo_path(info_file)
    title2id = load_item_mappings(info_file)
    sasrec_item_embeddings = None
    if sasrec_checkpoint_path:
        sasrec_ckpt = torch.load(sasrec_checkpoint_path, map_location="cpu")
        sasrec_state_dict = sasrec_ckpt["model_state_dict"]
        sasrec_item_embeddings = sasrec_state_dict["item_embeddings.weight"].detach().cpu().to(torch.float32)
        engram_config.item_hidden_size = int(sasrec_item_embeddings.shape[1])
        print(f"[SASRec] loaded item embeddings from {sasrec_checkpoint_path}")
        print(f"[SASRec] item embedding table shape = {tuple(sasrec_item_embeddings.shape)}")

    base_model = Qwen3ForCausalLM.from_pretrained(  
        model_name,
        trust_remote_code=True,
        #torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map=device_map,
    )
    # 训练阶段禁用Transformer模型中的键值缓存机制，缓存会导致无法训练
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    base_model.attach_engram(engram_config)
    base_model._reset_engram_parameters()

    print("\n=== 模型所有参数名(前20个)===")
    all_param_names = [name for name, _ in base_model.named_parameters()]
    for i, name in enumerate(all_param_names[:20]):
        print(f"{i+1}. {name}")
    
    # 查找所有包含engram的参数名（不区分大小写）
    engram_param_names_all = [name for name, _ in base_model.named_parameters() if "engram" in name.lower()]
    print(f"\n=== 找到的Engram相关参数 ===")
    if engram_param_names_all:
        for name in engram_param_names_all:
            print(f"- {name}")
    else:
        print("⚠️  未找到任何包含'engram'的参数名！请检查参数名拼写")

    print("\n=== Engram 参数初始化范围检查 ===")
    for name, param in base_model.named_parameters():
        if "engram" in name.lower():  # 只关注与 "engram" 相关的参数
            param_min = param.min().item()
            param_max = param.max().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            print(f"{name} - Min: {param_min}, Max: {param_max}, Mean: {param_mean}, Std: {param_std}")

    peft_config = LoraConfig(
        inference_mode=False,  
        r=8,
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,    
        task_type="CAUSAL_LM", 
        #modules_to_save=["engram"],
    )

    def setup_trainable_params(model):
        # 第一步：全局冻结所有参数
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # 第二步：解冻Engram参数
        engram_param_names = []
        for name, param in model.named_parameters():
            if "engram" in name.lower():
                param.requires_grad = True
                engram_param_names.append(name)
    
        print(f"\n=== 可训练参数统计（应用LoRA前）===")
        print(f"engram参数: {engram_param_names} (共{len(engram_param_names)}个)")
        return model

    base_model = setup_trainable_params(base_model)

    def load_engram_params(model, checkpoint_dir):
        """从训练保存的checkpoint目录加载Engram参数(适配你的load_all_params方法)"""
        # 训练时已把Engram参数存在 checkpoint_dir/engram_params 下
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            print(f"⚠️ 未找到Engram参数目录: {engram_params_dir}")
            return
        
        unwrapped_model = model.base_model if hasattr(model, 'base_model') else model
        engram_loaded = 0
        
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                # 直接读取训练时保存的npy文件
                param_path = os.path.join(engram_params_dir, f"engram_layer_{layer_id}.npy")
                
                if os.path.exists(param_path):
                    # ========== 直接调用你实现的load_all_params方法 ==========
                    module.load_all_params(param_path)
                    engram_loaded += 1
                    print(f"✅ 成功加载Engram层 {layer_id} 参数: {param_path}")
                else:
                    print(f"⚠️ Engram层 {layer_id} 参数文件缺失: {param_path}")
        
        print(f"\n📌 总计加载 {engram_loaded} 个Engram层参数")

    def convert_engram_params_to_float32(model):
        for name, param in model.named_parameters():
            if 'engram' in name.lower():  # 找到 engram 相关参数
                param.data = param.data.to(torch.float32)
        return model
    # 应用LoRA（关键：get_peft_model会自动解冻LoRA参数）
    if resume_from_checkpoint is not None:
        print(f"从检查点恢复,直接加载已有LoRA适配器")
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint)
        base_model.train()
        base_model = convert_engram_params_to_float32(base_model)
        load_engram_params(base_model,resume_from_checkpoint)
        # 恢复后重新解冻Engram
        for name, param in base_model.named_parameters():
            if "engram.short_conv" in name.lower():
                param.requires_grad = False
            elif "engram" in name.lower():
                param.requires_grad = True
            elif any(keyword in name.lower() for keyword in ["lora_a", "lora_b", "lora"]):
                param.requires_grad = True
    else:
        print("从头开始训练,应用新LoRA配置")
        base_model = get_peft_model(base_model, peft_config)
        base_model = convert_engram_params_to_float32(base_model)
        base_model.train()
        for name, param in base_model.named_parameters():
            if "engram.short_conv" in name.lower():
                param.requires_grad = False
            elif "engram" in name.lower():
                param.requires_grad = True
            elif any(keyword in name.lower() for keyword in ["lora_a", "lora_b", "lora"]):
                param.requires_grad = True
    

    # 确认数据精度类型
    def check_precision_and_types(model):
    # 打印前三个 LoRA 参数的类型
        lora_count = 0
        print("\n=== 前三个 LoRA 参数类型 ===")
        for name, param in model.named_parameters():
            if "lora" in name.lower():  # 找到 LoRA 相关参数
                print(f"{name} - 类型: {param.dtype}")
                lora_count += 1
            if lora_count >= 3:
                break
        
        # 打印所有 Engram 参数的类型
        print("\n=== 所有 Engram 参数类型 ===")
        for name, param in model.named_parameters():
            if "engram" in name.lower():  # 找到 Engram 相关参数
                print(f"{name} - 类型: {param.dtype}")
    check_precision_and_types(base_model)

    trainable_names = []
    lora_a_names = []
    lora_b_names = []
    engram_names = []  # 新增：统计Engram可训练参数
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            if "lora_A" in name:
                lora_a_names.append(name)
            elif "lora_B" in name:
                lora_b_names.append(name)
            elif "engram" in name.lower():  # 新增：统计Engram参数
                engram_names.append(name)
    #print(base_model)
    print(f"\n=== 可训练参数统计（应用LoRA后）===")
    print(f"LoRA A参数: {len(lora_a_names)}个")
    print(f"LoRA B参数: {len(lora_b_names)}个")
    print(f"Engram参数: {len(engram_names)}个")  # 新增：打印Engram参数数量
    print(f"总可训练参数：{len(trainable_names)}个")
    # 打印前5个lora_B参数的状态（验证是否可训练）
    if lora_b_names:
        first_lora_b = lora_b_names[0]
        first_param = [p for n,p in base_model.named_parameters() if n == first_lora_b][0]
        print(f"第一个lora_B参数 {first_lora_b}: requires_grad={first_param.requires_grad}, 初始值前5个: {first_param.data.flatten()[:200].cpu().tolist()}")
    if lora_a_names:
        first_lora_a = lora_a_names[0]
        first_param = [p for n,p in base_model.named_parameters() if n == first_lora_a][0]
        print(f"第一个lora_A参数 {first_lora_a}: requires_grad={first_param.requires_grad}, 初始值前5个: {first_param.data.flatten()[:200].cpu().tolist()}")
    # 新增：打印前1个Engram参数的状态
    if engram_names:
        first_engram = engram_names[4]
        first_engram_param = [p for n,p in base_model.named_parameters() if n == first_engram][0]
        print(f"第二个Engram参数 {first_engram}: requires_grad={first_engram_param.requires_grad}, 初始值前5个: {first_engram_param.data.flatten()[:400].cpu().tolist()}")

    for name, param in base_model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "engram" in name.lower():
            print(f"{name} - requires_grad: {param.requires_grad}, 设备: {param.device}, 梯度: {param.grad is not None}")

    # 检查PEFT设置之后的Engram参数范围
    print("\n=== PEFT 设置后 Engram 参数范围检查 ===")
    for name, param in base_model.named_parameters():
        if "engram" in name.lower():  # 只关注与 "engram" 相关的参数
            param_min = param.min().item()
            param_max = param.max().item()
            param_mean = param.mean().item()
            param_std = param.std().item()
            print(f"{name} - Min: {param_min}, Max: {param_max}, Mean: {param_mean}, Std: {param_std}")
    

    def collator(response_token_ids ,input_ids , tokenizer=None):
        response_token_ids_start_idx = None
        response_token_ids_end_idx = None
        input_ids = list(input_ids)

        target_text = tokenizer.decode(
                response_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()

        for idx in range(len(input_ids) - len(response_token_ids) + 1):
            window_text = tokenizer.decode(
                input_ids[idx: idx + len(response_token_ids)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()

            if window_text == target_text:
                response_token_ids_start_idx = idx
                response_token_ids_end_idx = idx + len(response_token_ids)
                break

        if response_token_ids_start_idx is None:
            if tokenizer is not None:
                warnings.warn(
                    f"Could not find response key `{response_token_ids}` in the following instance: "
                    f"Could not find response key `{tokenizer.decode(response_token_ids)}` in the following instance: "
                    f"{tokenizer.decode(input_ids)}",
                    UserWarning,
                )

        return response_token_ids_start_idx, response_token_ids_end_idx


    def process_data(data_point,tokenizer):
        """处理当前带 item spans 的 ReRe 数据。"""
        prompt_text = data_point.get("prompt", "")
        if not isinstance(prompt_text, str):
            prompt_text = ""

        true_selection = str(data_point.get("completion", "")).rstrip("\n")
        if not true_selection:
            target_title = str(data_point.get("target_item_title", "")).strip()
            true_selection = f"\"{target_title}\"" if target_title else ""
        completion = true_selection + tokenizer.eos_token
        full_text = prompt_text + completion

        input_ids = tokenizer(
            full_text, 
            truncation=True, 
            max_length=cutoff_len,
            add_special_tokens=True,
        )["input_ids"]

        response_ids = tokenizer.encode("### Response:", add_special_tokens=False)
        _, response_end = collator(response_ids, input_ids, tokenizer=tokenizer)
        completion_start = response_end if response_end is not None else len(input_ids)
        
        history_item_ids = data_point.get("history_item_ids", []) or []
        history_item_token_spans = data_point.get("history_item_token_spans", []) or []
        if not history_item_ids or not history_item_token_spans:
            history_titles = data_point.get("history_item_titles", []) or split_history_titles(data_point.get("history_str", ""))
            history_item_ids, history_item_token_spans = locate_history_token_spans(
                prompt_text,
                history_titles,
                tokenizer,
                title2id,
                cutoff_len,
            )

        return {
        "input_ids": input_ids,
        "completion_start": completion_start,
        "history_item_ids": history_item_ids,
        "history_item_token_spans": history_item_token_spans,
    }

    data_files = {
        "train": resolve_demo_path(train_file),
        "validation": resolve_demo_path(eval_file),
    }
    print(f"[data] train_file = {data_files['train']}")
    print(f"[data] eval_file = {data_files['validation']}")
    print(f"[data] info_file = {info_file}")

    cache_dir = os.path.join(SCRIPT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    def resolve_cache_path(primary_name, fallback_names):
        primary_path = os.path.join(cache_dir, primary_name)
        if os.path.isdir(primary_path):
            return primary_path
        for name in fallback_names:
            candidate = os.path.join(cache_dir, name)
            if os.path.isdir(candidate):
                print(f"[cache] use fallback cache: {candidate}")
                return candidate
        return primary_path

    train_cache = resolve_cache_path(
        "processed_train_rere_history_item_spans_completion_start_v2_nomask",
        fallback_names=[],
    )
    val_cache = resolve_cache_path(
        "processed_val_rere_history_item_spans_completion_start_v2_nomask",
        fallback_names=[],
    )

    if os.path.isdir(train_cache):
        train_data = load_from_disk(train_cache)
    else:
        train_data = load_dataset("json", data_files={"train": data_files["train"]})["train"]
        train_data = train_data.shuffle(seed=42).map(
            process_data,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=train_data.column_names,
        )
        train_data.save_to_disk(train_cache)

    if os.path.isdir(val_cache):
        val_data = load_from_disk(val_cache)
    else:
        val_data = load_dataset("json", data_files={"validation": data_files["validation"]})["validation"]
        val_data = val_data.shuffle(seed=42).map(
            process_data,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=val_data.column_names,
        )
        val_data.save_to_disk(val_cache)


    # 按真实全局batch和梯度累积计算训练步数，避免写死常数和直接截断带来的错位。
    per_device_train_batch_size = batch_size
    world_size = Accelerator().num_processes
    global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_data) / global_batch_size)
    total_train_steps = steps_per_epoch * num_train_epochs

    # eval_step <= 1 时，按“每个epoch的比例”换算：
    # - 0.5 表示半个 epoch eval 一次
    # - 1 表示每个 epoch eval 一次
    # eval_step > 1 时，视为直接给定的训练步数。
    if float(eval_step) <= 1:
        eval_interval_steps = round(steps_per_epoch * eval_step)
    else:
        eval_interval_steps = round(float(eval_step))

    eval_steps = max(1, eval_interval_steps)
    save_steps = eval_steps

    print("\n=== 训练步数配置 ===")
    print(f"train samples = {len(train_data)}")
    print(f"world_size = {world_size}")
    print(f"per_device_train_batch_size = {per_device_train_batch_size}")
    print(f"gradient_accumulation_steps = {gradient_accumulation_steps}")
    print(f"global_batch_size = {global_batch_size}")
    print(f"steps_per_epoch = {steps_per_epoch}")
    print(f"num_train_epochs = {num_train_epochs}")
    print(f"total_train_steps = {total_train_steps}")
    print(f"eval_step = {eval_step}")
    print(f"eval_steps = {eval_steps}")
    print(f"save_steps = {save_steps}")

    collator = DataCollatorForCompletionOnlyLMWithItemMask(
        tokenizer.encode(response_template, add_special_tokens=False),
        tokenizer=tokenizer,
        sasrec_item_embeddings=sasrec_item_embeddings,
    )

    def get_optimizer_grouped_parameters(model,lora_lr, engram_lr):
        """
        给不同模块参数分组，设置不同学习率：
        - LoRA参数:lora_lr主学习率
        - Engram参数:engram_lr(单独学习率）
        """
        param_groups = []
        # 定义参数分组规则
        lora_params = []          # LoRA参数
        engram_params = []        # Engram参数
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结参数
            # 分组逻辑
            if "lora_" in name.lower():
                lora_params.append(param)
            elif "engram" in name.lower():
                engram_params.append(param)
        
        # 添加分组（设置不同LR）
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": 0
            })
        if engram_params:
            param_groups.append({
                "params": engram_params,
                "lr": engram_lr,
                "weight_decay": 0  # Engram通常不加权重衰减
            })
        
        # 打印分组统计（验证）
        print(f"\n=== 参数分组+学习率配置 ===")
        print(f"LoRA参数:{len(lora_params)}个 | LR={lora_lr}")
        print(f"Engram参数:{len(engram_params)}个 | LR={engram_lr}")
        return param_groups

    # 生成参数分组
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        base_model, lora_lr, engram_lr
    )

    optimizer = bnb.optim.PagedAdamW32bit(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_train_steps * 0.05),  # 和SFTConfig中一致
        num_training_steps=total_train_steps
    )

    # TrainingArguments 是 Hugging Face Trainer 库的核心配置类，
    # 用于定义所有训练相关的参数（如训练轮数、批次大小、学习率、保存路径、硬件配置等），
    # 它把分散的训练超参数整合为一个统一的配置对象
    sft_config = SFTConfig(
            # 训练参数
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            # 使用bfloat16混合精度训练
            bf16=True,
            
            # 保存和评估策略
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_strategy="steps",
            eval_steps=eval_steps,
            logging_steps=1,
            
            # 优化器和调度器
            #optim="paged_adamw_32bit",
            #lr_scheduler_type="cosine",
            #warmup_ratio=0.05,
            
            # 日志和输出
            output_dir=output_dir,
            report_to="wandb",
            run_name=wandb_name,
            logging_dir=logging_dir,
            
            gradient_checkpointing_kwargs={'use_reentrant': True},
            save_only_model=True,
            max_seq_length=cutoff_len,
            
            
            # DDP相关配置，没有插入engram时结构简单不需要查看有没有没用的参数
            ddp_find_unused_parameters=False,
            #ddp_static_graph=False,

            remove_unused_columns=False,
        )

    global ENGRAM_INITIAL_PARAMS
    global LoraB_INITIAL_PARAMS
    ENGRAM_INITIAL_PARAMS = {}
    LoraB_INITIAL_PARAMS = {}
    LoraA_INITIAL_PARAMS = {}
    class EngramSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            global ENGRAM_INITIAL_PARAMS
            global LoraB_INITIAL_PARAMS
            is_dist_ready = dist.is_available() and dist.is_initialized()

            if is_dist_ready:
                current_rank = dist.get_rank()
                if current_rank != 0:
                    print(f"[Rank {current_rank}] waiting for rank0 to save Engram params at checkpoint-{state.global_step}")
                dist.barrier()

            if state.is_world_process_zero:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                engram_save_dir = os.path.join(checkpoint_dir, "engram_params")
                os.makedirs(engram_save_dir, exist_ok=True)
                
                # 正确解包 PeftModel 和 DDP
                # 使用了字典访问 语法，kwargs 是一个包含命名参数（关键字参数）的字典
                unwrapped_model = kwargs["model"]
                # 这里判断 unwrapped_model 是否具有 model 和 peft_config 属性
                # 如果 unwrapped_model 是 PeftModel，那么它包含了 model 属性，
                # 这里通过循环将 unwrapped_model 一直解包，直到它不再是 PeftModel，而是最终的基础模型
                while hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model, 'peft_config'):
                    unwrapped_model = unwrapped_model.model
                # 如果模型确实是 PeftModel，它的 model 属性指向实际的原始模型
                if hasattr(unwrapped_model, 'module'):
                    unwrapped_model = unwrapped_model.module
                
                engram_saved = 0
                for name, module in unwrapped_model.named_modules():
                    if isinstance(module, Engram):
                        layer_id = module.layer_id
                        save_path = os.path.join(engram_save_dir, f"engram_layer_{layer_id}.npy")
                        
                        # 从全局变量获取初始参数
                        initial_params = ENGRAM_INITIAL_PARAMS.get(layer_id, None)
                        module.save_all_params(save_path, initial_params=initial_params)
                        engram_saved += 1
                
                print(f"✅ Checkpoint {state.global_step} 保存 {engram_saved} 个Engram层参数")

            if is_dist_ready:
                if state.is_world_process_zero:
                    print(f"[Rank 0] finished saving Engram params at checkpoint-{state.global_step}, releasing other ranks")
                dist.barrier()

    # 创建SFTTrainer实例
    trainer = SFTTrainer(
        base_model,           # 参数名改为 model
        args=sft_config,            # 使用 SFTConfig
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        optimizers=(optimizer,lr_scheduler),
        #formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        callbacks=[EngramSaveCallback(), EarlyStoppingCallback(early_stopping_patience=3)]
    )

    if resume_from_checkpoint is not None :
        print("\n=== 重置训练状态(global_step, epoch, lr_scheduler===")
        
        # 重置 global_step 和 epoch
        trainer.state.global_step = 0
        trainer.state.epoch = 0.0
        
        # 重新初始化优化器（丢弃原来的 momentum 等状态）
        trainer.create_optimizer()
        
        # 重新初始化学习率调度器（从头开始 warmup）
        trainer.create_scheduler(num_training_steps=trainer.args.max_steps, optimizer=trainer.optimizer)
        
        print(f"重置后 global_step: {trainer.state.global_step}")
        print(f"重置后 epoch: {trainer.state.epoch}")
        print(f"学习率调度器重新初始化完成")


    class ParameterUpdateMonitorCallback(transformers.TrainerCallback):
        def __init__(self):
            super().__init__()
            # 存储初始权重（用于后续对比更新）
            self.init_lora_a_weights = {}
            self.init_lora_b_weights = {}
            self.init_engram_weights = {}
            self.engram_initial_params = {} 
            # 标记是否已记录初始权重
            self.initialized = False

        def on_train_begin(self, args, state, control, **kwargs):
            """训练开始时记录初始权重"""
            if not self.initialized:
                model = kwargs["model"]
                # 解包装，确保操作真实模型的参数
                unwrapped_model = model.module if hasattr(model, 'module') else model
                
                # 正确解包 PeftModel
                while hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model, 'peft_config'):
                    unwrapped_model = unwrapped_model.model
                
                # 记录LoRA和Engram的初始权重
                for name, param in unwrapped_model.named_parameters():
                    if param.requires_grad:
                        if "lora_A" in name or "lora_a" in name.lower():
                            self.init_lora_a_weights[name] = param.data.cpu().clone()
                        elif "lora_B" in name or "lora_b" in name.lower():
                            self.init_lora_b_weights[name] = param.data.cpu().clone()
                        elif "engram" in name.lower():
                            self.init_engram_weights[name] = param.data.cpu().clone()
                
                # 为每个 Engram 层记录初始参数（用于 save_all_params）
                for module_name, module in unwrapped_model.named_modules():
                    if isinstance(module, Engram):
                        layer_id = module.layer_id
                        self.engram_initial_params[layer_id] = {}
                        for p_name, param in module.named_parameters():
                            if param.requires_grad:
                                self.engram_initial_params[layer_id][p_name] = \
                                    param.detach().cpu().float().numpy().copy()
                
                global ENGRAM_INITIAL_PARAMS
                global LoraA_INITIAL_PARAMS
                global LoraB_INITIAL_PARAMS
                # 关键修复：存入全局变量，供 EngramSaveCallback 使用
                ENGRAM_INITIAL_PARAMS = self.engram_initial_params.copy()
                LoraA_INITIAL_PARAMS = self.init_lora_a_weights.copy()
                LoraB_INITIAL_PARAMS = self.init_lora_b_weights.copy()
                print(f"\n=== 初始权重记录完成 ===")
                print(f"LoRA A 参数数量: {len(self.init_lora_a_weights)}")
                print(f"LoRA B 参数数量: {len(self.init_lora_b_weights)}")
                print(f"Engram 参数数量: {len(self.init_engram_weights)}")
                print(f"Engram 层数: {len(self.engram_initial_params)}")
                self.initialized = True

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            """每步结束时检查梯度和权重更新"""
            if state.global_step % 100 == 0 and state.global_step > 0:
                model = kwargs["model"]
                
                # 解包装，确保操作真实模型的参数
                unwrapped_model = model
                while True:
                    if hasattr(unwrapped_model, 'module'):
                        unwrapped_model = unwrapped_model.module
                    elif hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model, 'peft_config'):
                        unwrapped_model = unwrapped_model.model
                    else:
                        break
                num=0
                # 获取和显示梯度范数
                for name, param in unwrapped_model.named_parameters():
                    if name in self.init_engram_weights:
                        if param.grad is not None and param.device == torch.device("cuda:0"):
                            grad_norm = torch.norm(param.grad).item()
                            print(f"{name} - 梯度范数: {grad_norm:.6f}")
                            print(f"{name} - 前几个参数的梯度值: {param.grad.flatten()[:20].cpu().tolist()}")
                            print(f"{name} - 最大的梯度值: {param.grad.max().item():.6f}")
                    elif name in self.init_lora_b_weights and num<3:
                        if param.grad is not None and param.device == torch.device("cuda:0"):
                            grad_norm = torch.norm(param.grad).item()
                            print(f"{name} - 梯度范数: {grad_norm:.6f}")
                            print(f"{name} - 前几个参数的梯度值: {param.grad.flatten()[:5].cpu().tolist()}")
                            print(f"{name} - 最大的梯度值: {param.grad.max().item():.6f}")
                            num+=1
        def on_step_end(self, args, state, control, **kwargs):  
            """每100步检查梯度和权重更新"""
            if state.global_step % 100 == 0 and state.global_step > 0:
                model = kwargs["model"]
                
                unwrapped_model = model
                while True:
                    if hasattr(unwrapped_model, 'module'):
                        unwrapped_model = unwrapped_model.module
                    elif hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model, 'peft_config'):
                        unwrapped_model = unwrapped_model.model
                    else:
                        break
                num=0
                for name, param in unwrapped_model.named_parameters():
                    if (name in self.init_lora_a_weights or name in self.init_lora_b_weights) and num<20:
                        logging.info("初始的参数范围")
                        if name in self.init_lora_a_weights:
                            init_param = self.init_lora_a_weights[name]
                        else:
                            init_param = self.init_lora_b_weights[name]
                        init_min = init_param.min().item()
                        init_max = init_param.max().item()
                        init_mean = init_param.mean().item()
                        init_std = init_param.std().item()
                        logging.info(f"{name} - Min: {init_min}, Max: {init_max}, Mean: {init_mean}, Std: {init_std}")
                        logging.info("训练后参数范围")
                        param_min = param.min().item()
                        param_max = param.max().item()
                        param_mean = param.mean().item()
                        param_std = param.std().item()
                        logging.info(f"{name} - Min: {param_min}, Max: {param_max}, Mean: {param_mean}, Std: {param_std}")
                        num += 1
                    elif name in self.init_engram_weights:
                        logging.info("初始的参数范围")
                        init_param = self.init_engram_weights[name]
                        init_min = init_param.min().item()
                        init_max = init_param.max().item()
                        init_mean = init_param.mean().item()
                        init_std = init_param.std().item()
                        logging.info(f"{name} - Min: {init_min}, Max: {init_max}, Mean: {init_mean}, Std: {init_std}")
                        logging.info("训练后参数范围")
                        param_min = param.min().item()
                        param_max = param.max().item()
                        param_mean = param.mean().item()
                        param_std = param.std().item()
                        logging.info(f"{name} - Min: {param_min}, Max: {param_max}, Mean: {param_mean}, Std: {param_std}")
                logging.info(f"\n==== Step {state.global_step} 参数更新检查 ====")
                
                # 1. 检查梯度是否存在（即时梯度状态）
                logging.info("\n--- 🔍 梯度存在性检查 ---")
                lora_a_has_grad = 0
                lora_b_has_grad = 0
                engram_has_grad = 0
                
                # 检查LoRA A梯度
                for name, param in unwrapped_model.named_parameters():
                    if name in self.init_lora_a_weights and param.grad is not None:
                        lora_a_has_grad += 1
                        grad_norm = torch.norm(param.grad).item()
                        #print(f"LoRA A | {name[:60]:<60} | 梯度范数: {grad_norm:.8f}")
                
                # 检查LoRA B梯度
                for name, param in unwrapped_model.named_parameters():
                    if name in self.init_lora_b_weights and param.grad is not None:
                        lora_b_has_grad += 1
                        grad_norm = torch.norm(param.grad).item()
                        #print(f"LoRA B | {name[:60]:<60} | 梯度范数: {grad_norm:.8f}")
                
                # 检查Engram梯度
                for name, param in unwrapped_model.named_parameters():
                    if name in self.init_engram_weights and param.grad is not None:
                        engram_has_grad += 1
                        grad_norm = torch.norm(param.grad).item()
                        logging.info(f"Engram | {name[:60]:<60} | 梯度范数: {grad_norm:.8f}")
                
                # 梯度统计
                logging.info(f"\n--- 📊 梯度统计 ---")
                logging.info(f"Engram: {engram_has_grad}/{len(self.init_engram_weights)} 个参数有梯度")
                logging.info(f"LoRA A: {lora_a_has_grad}/{len(self.init_lora_a_weights)} 个参数有梯度")
                logging.info(f"LoRA B: {lora_b_has_grad}/{len(self.init_lora_b_weights)} 个参数有梯度")
                # 2. 检查权重是否更新（对比初始值）
                logging.info(f"\n--- 📈 权重更新检查（对比初始值） ---")
                lora_a_updated = 0
                lora_b_updated = 0
                engram_updated = 0
                
                # 检查LoRA A权重更新
                for name in self.init_lora_a_weights:
                    param = [p for n, p in unwrapped_model.named_parameters() if n == name][0]
                    weight_diff = torch.norm(param.data.cpu() - self.init_lora_a_weights[name]).item()
                    if weight_diff > 1e-6:
                        lora_a_updated += 1
                        #print(f"LoRA A | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ✅")
                    else:
                        lora_a_updated += 0
                        #print(f"LoRA A | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ❌")
                
                # 检查LoRA B权重更新
                for name in self.init_lora_b_weights:
                    param = [p for n, p in unwrapped_model.named_parameters() if n == name][0]
                    weight_diff = torch.norm(param.data.cpu() - self.init_lora_b_weights[name]).item()
                    if weight_diff > 1e-6:
                        lora_b_updated += 1
                        #print(f"LoRA B | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ✅")
                    else:
                        lora_b_updated += 0
                        #print(f"LoRA B | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ❌")
                
                # 检查Engram权重更新
                for name in self.init_engram_weights:
                    param = [p for n, p in unwrapped_model.named_parameters() if n == name][0]
                    weight_diff = torch.norm(param.data.cpu() - self.init_engram_weights[name]).item()
                    if weight_diff > 1e-6:
                        engram_updated += 1
                        logging.info(f"Engram | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ✅")
                    else:
                        logging.info(f"Engram | {name[:60]:<60} | 权重变化: {weight_diff:.8f} ❌")
                
                # 最终更新统计
                logging.info(f"\n--- 📋 最终更新统计 ---")
                logging.info(f"LoRA A: {lora_a_updated}/{len(self.init_lora_a_weights)} 个参数已更新")
                logging.info(f"LoRA B: {lora_b_updated}/{len(self.init_lora_b_weights)} 个参数已更新")
                logging.info(f"Engram: {engram_updated}/{len(self.init_engram_weights)} 个参数已更新")
                
                # 警告提示
                total_updated = lora_a_updated + lora_b_updated + engram_updated
                total_params = len(self.init_lora_a_weights) + len(self.init_lora_b_weights) + len(self.init_engram_weights)
                if total_updated == 0:
                    logging.info(f"\n⚠️  严重警告：所有可训练参数（{total_params}个）均未更新！训练未生效！")
                elif engram_updated == 0 and len(self.init_engram_weights) > 0:
                    logging.info(f"\n⚠️  警告:Engram参数未更新(共{len(self.init_engram_weights)}个）！")
                elif lora_a_updated == 0 and lora_b_updated == 0:
                    logging.info(f"\n⚠️  警告:LoRA参数未更新(A:{len(self.init_lora_a_weights)}个, B:{len(self.init_lora_b_weights)}个）！")
    trainer.add_callback(ParameterUpdateMonitorCallback())
 
    trainer.train() 
    #保存训练器的完整状态
    trainer.save_model(output_dir)

    # 在原始输出目录下创建名为 "final_checkpoint"的子目录
    def save_all_engram_params(model, save_dir):
        engram_save_dir = os.path.join(save_dir, "engram_params")
        os.makedirs(engram_save_dir, exist_ok=True)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        engram_saved = 0
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                save_path = os.path.join(engram_save_dir, f"engram_layer_{layer_id}.npy")
                module.save_all_params(save_path)
                engram_saved += 1
                print(f"已保存Engram层 {layer_id} 参数到: {save_path}")
        print(f"\n✅ 共保存 {engram_saved} 个Engram层参数")

    output_dir = os.path.join(output_dir, "final_checkpoint_sft")
    # 将训练好的eval性能最佳的模型保存到指定目录
    trainer.model.save_pretrained(output_dir)
    # 将训练时使用的分词器（含配置 + 词汇表）完整保存到同一目录
    tokenizer.save_pretrained(output_dir)
    save_all_engram_params(base_model, output_dir)
    print(f"\n✅ 训练完成！模型保存到: {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
