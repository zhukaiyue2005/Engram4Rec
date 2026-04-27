import os
import sys
import math
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

# 强制重写huggingface_hub的校验函数
from huggingface_hub.utils._validators import validate_repo_id
def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return  # 空函数，跳过所有校验
validate_repo_id.__code__ = _dummy_validate_repo_id.__code__  
engram_dir = os.path.join(os.path.dirname(__file__), "Engram_Insert_code")
sys.path.append(engram_dir)
from engram import Engram, EngramConfig
from modeling_qwen3 import Qwen3ForCausalLM
from Prompt import Prompt
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
    def __init__(self, *args, debug_max_print: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_max_print = debug_max_print
        self.debug_count = 0
    
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        examples = deepcopy(examples)
        raw_examples = deepcopy(examples)

        item_masks = []
        completion_starts = []
        raw_input_lens = []
        for ex in examples:
            if "item_attention_mask" not in ex:
                raise ValueError("example 中缺少 `item_attention_mask` 字段")
            item_masks.append(ex.pop("item_attention_mask"))
            raw_input_lens.append(len(ex["input_ids"]))
            completion_starts.append(ex.pop("completion_start", None))
            
            ex.pop("prompt", None)
            ex.pop("completion", None)
            ex.pop("sentence", None)
            ex.pop("history_range", None)
            ex.pop("cans_range", None)

        # 先走父类原本逻辑
        batch = super().torch_call(examples)

        if not self.padding_free:
            seq_len = batch["input_ids"].shape[1]

            padded_item_masks = []
            for mask in item_masks:
                pad_len = seq_len - len(mask)
                if self.tokenizer.padding_side == "left":
                    padded_mask = [0] * pad_len + mask
                else:
                    padded_mask = mask + [0] * pad_len

                padded_item_masks.append(padded_mask)

            batch["item_attention_mask"] = torch.tensor(padded_item_masks, dtype=torch.long)

            # 不依赖 response_template 字符串匹配，直接按 completion_start 构造 labels，
            # 避免模板分词细节变化导致整条样本 supervision 丢失。
            if "labels" in batch and any(cs is not None for cs in completion_starts):
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

        if self.debug_count < self.debug_max_print:
            print(f"\n{'='*80}")
            print(f"[Collator Debug] batch #{self.debug_count}")
            print(f"padding_side = {self.tokenizer.padding_side}")
            print(f"batch seq_len = {seq_len}")
            print(f"{'='*80}")

            for i, f in enumerate(raw_examples):   # 这里改成 raw_examples
                raw_ids = f["input_ids"]
                raw_mask = f["item_attention_mask"]

                padded_ids = batch["input_ids"][i].tolist()
                padded_attn = batch["attention_mask"][i].tolist()
                padded_item = batch["item_attention_mask"][i].tolist()

                print(f"\n--- sample {i} ---")
                print(f"raw input_ids len           = {len(raw_ids)}")
                print(f"raw item_attention_mask len = {len(raw_mask)}")
                print(f"padded input_ids len        = {len(padded_ids)}")
                print(f"padded item_mask len        = {len(padded_item)}")
                print(f"item_mask is 1      = {sum(x != 0 for x in padded_item)}")
                
            print(f"\n{'='*80}")
            print("[Collator Debug] first sample full dump")
            print(f"{'='*80}")

            first_input_ids = batch["input_ids"][0].tolist()
            first_attention_mask = batch["attention_mask"][0].tolist()
            first_item_attention_mask = batch["item_attention_mask"][0].tolist()

            first_tokens = self.tokenizer.convert_ids_to_tokens(first_input_ids)

            print(
                f"{'idx':>5} | {'token_id':>8} | {'attention':>9} | {'item_attn':>9} | token"
            )
            print("-" * 120)

            for idx, (tok, tok_id, attn, item_attn) in enumerate(
                zip(
                    first_tokens,
                    first_input_ids,
                    first_attention_mask,
                    first_item_attention_mask,
                )
            ):
                print(
                    f"{idx:>5} | {tok_id:>8} | {attn:>9} | {item_attn:>9} | {repr(tok)}"
                )

            # 如果你还想顺手看解码后的文本，也可以保留这一段
            valid_ids = [tid for tid, attn in zip(first_input_ids, first_attention_mask) if attn == 1]
            decoded_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
            print(f"\n[Decoded first sample]")
            print(decoded_text)
            self.debug_count += 1

        return batch

def train(
    # 训练输出目录，用于保存模型检查点、日志等
    output_dir="",   
    # 日志目录，用于存储训练过程中的日志文件  
    logging_dir="",  
    # 基础模型名称或路径  
    model_name ="",    
    # 提示模板路径
    prompt_path = "",  
    dataset="",
    train_file="../data/Industrial_and_Scientific_dataset/train.jsonl",
    eval_file="../data/Industrial_and_Scientific_dataset/valid.jsonl",
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
    # 最大序列长度，超过该长度的输入将被截断
    cutoff_len: int = 2048,  
    # eval_step：评估频率，表示每训练多少比例的epoch进行一次评估
    eval_step = 0.1,
    response_template: str = "### Response:\n",
):  
    wandb_enabled = False
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        try:
            wandb.init(
                name=wandb_name,
                project=wandb_project,
                settings=wandb.Settings(silent=True, init_timeout=180),
            )
            wandb_enabled = True
        except Exception as e:
            print(f"[Warn] wandb init failed, continue without wandb: {e}")

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

    layer_ids = [6,13,20]
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    engram_config = EngramConfig()
    engram_config.layer_ids = layer_ids
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
    

    def parse_prompt_column(prompt_col):
        """解析prompt列,提取用户消息内容"""
        if isinstance(prompt_col, list):
            # 如果是列表，取第一个user角色的content
            for item in prompt_col:
                if isinstance(item, dict) and item.get('role') == 'user':
                    return item.get('content', '')
            return ''
        if isinstance(prompt_col, str):
            return prompt_col
        return ''

    def extract_extra_info(extra_info_col):
        """从extra_info列提取ground_truth的description和title"""
        if isinstance(extra_info_col, dict):
            ground_truth = extra_info_col.get('ground_truth', {})
            description = ground_truth.get('description', '')
            title = ground_truth.get('title', '')
            
            history_list = extra_info_col.get("historyList", [])
            item_list = extra_info_col.get("itemList", [])

            return description, title, history_list, item_list
        return "", "" , [], []

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


    def build_item_attention_mask(
        input_ids,
        history_start,
        history_end,
        completion_start,
        cans_start=None,
        cans_end=None,
    ):
        """
        input_ids: 完整输入文本的id列表
        history_start: prompt里 [HistoryHere] 前面的文本的长度索引
        cans_start:  prompt里 [CansHere] 前面的文本的长度索引

        返回:
            item_attention_mask: 与 input_ids 等长，只有 item 子词位置为1
        """
        item_attention_mask = [0] * len(input_ids)

        def is_sep_token(token_id):
            """
            判断一个 token 是否是不要的分隔符:
            """
            tok_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            pass_list =[",","\n","|","<0x0A>",""]
            if tok_text.strip() in pass_list:
                return True
            
            return False

        # history 区间
        for i in range(max(0, history_start), min(history_end, len(input_ids))):
            if not is_sep_token(input_ids[i]):
                item_attention_mask[i] = 1

        # cans 区间
        if cans_start is not None and cans_end is not None:
            for i in range(max(0, cans_start), min(cans_end, len(input_ids))):
                if not is_sep_token(input_ids[i]):
                    item_attention_mask[i] = 1
        
        for i in range(max(0, completion_start), len(input_ids)):
            item_attention_mask[i] = 1

        def remove_year_spans(range_start, range_end):
            """
            在区间里找形如 (1997) 的 token span,并把这些 token 的 mask 清 0
            """
            i = range_start
            while i < range_end:
                tok_text = tokenizer.decode([input_ids[i]], clean_up_tokenization_spaces=False)

                # 只有当前 token 里出现 '(' 才尝试往后拼
                if "(" in tok_text:
                    buf = tok_text
                    j = i

                    # 最多往后看几步
                    while j + 1 < range_end and ")" not in buf and (j - i) < 10:
                        j += 1
                        next_text = tokenizer.decode([input_ids[j]], clean_up_tokenization_spaces=False)
                        buf += next_text

                    # re.sub(pattern, replacement, text) 表示把 text 中匹配 pattern 的部分替换成 replacement
                    # r"\s+" 里的 \s 表示“任意空白字符”，包括空格、换行、tab；+ 表示“一个或多个”
                    normalized = re.sub(r"\s+", "", buf)
                    # 只会命中括号里恰好是 4 位数字的情况
                    # \( ：字面意义的左括号 (
                    # \d ：一个数字字符，等价于 [0-9]
                    # {4} ：前面的 \d 重复 4 次
                    if re.fullmatch(r"\(\d{4}\)[,.;:]?", normalized):
                        for k in range(i, j + 1):
                            item_attention_mask[k] = 0
                        i = j + 1
                        continue
                i += 1
        
        remove_year_spans(history_start, history_end)
        if cans_start is not None and cans_end is not None:
            remove_year_spans(cans_start, cans_end)

        return {
            "input_ids": input_ids,
            "item_attention_mask": item_attention_mask,
            "history_range": [history_start, history_end],
            "cans_range": [cans_start, cans_end] if cans_start is not None and cans_end is not None else [-1, -1],
        }

    def process_data(data_point,tokenizer):
        """
        处理单条样本，直接产出给 collator 用的数据
        """
        prompt_text = parse_prompt_column(data_point.get("prompt", ""))

        # 兼容两种数据格式：
        # 1) 旧格式：extra_info(generated think/answer)
        # 2) ReRe当前格式：只有prompt + completion/target_item_title
        if "extra_info" in data_point and isinstance(data_point["extra_info"], dict):
            description, true_selection, history_list, item_list = extract_extra_info(
                data_point["extra_info"]
            )
            completion = "<think>" + description + "</think>" + "<answer>" + true_selection + "</answer>" + tokenizer.eos_token
            full_text = prompt_text + " " + completion
            legacy_mode = True
        else:
            true_selection = str(data_point.get("completion", "")).rstrip("\n")
            if not true_selection:
                target_title = str(data_point.get("target_item_title", "")).strip()
                true_selection = f"\"{target_title}\"" if target_title else ""
            completion = true_selection + tokenizer.eos_token
            full_text = prompt_text + completion
            legacy_mode = False

        input_ids = tokenizer(
            full_text, 
            truncation=True, 
            max_length=cutoff_len,
            add_special_tokens=True,
        )["input_ids"]

        if legacy_mode:
            history_before = "shown with metadata here:"
            cans_before = "The candidates are:"
            cans_after = "The Assistant recommends a movie for the user"
            completion_before = " Assistant:" 

            history_before_ids = tokenizer.encode(history_before, add_special_tokens=False)
            cans_before_ids = tokenizer.encode(cans_before, add_special_tokens=False)
            cans_after_ids = tokenizer.encode(cans_after, add_special_tokens=False)
            completion_before_ids = tokenizer.encode(completion_before, add_special_tokens=False)

            _, history_before_end = collator(history_before_ids, input_ids, tokenizer=tokenizer)
            cans_before_start, cans_before_end = collator(cans_before_ids, input_ids, tokenizer=tokenizer)
            cans_after_start, _ = collator(cans_after_ids, input_ids, tokenizer=tokenizer)
            _, completion_before_end = collator(completion_before_ids, input_ids, tokenizer=tokenizer)

            history_start = history_before_end or 0
            history_end = cans_before_start if cans_before_start is not None else len(input_ids)
            completion_start = completion_before_end if completion_before_end is not None else len(input_ids)
            text_dic = build_item_attention_mask(
                input_ids,
                history_start,
                history_end,
                completion_start,
                cans_start=cans_before_end,
                cans_end=cans_after_start,
            )
        else:
            response_ids = tokenizer.encode("### Response:", add_special_tokens=False)
            _, response_end = collator(response_ids, input_ids, tokenizer=tokenizer)
            response_start, _ = collator(response_ids, input_ids, tokenizer=tokenizer)

            before_ids = tokenizer.encode("before:", add_special_tokens=False)
            _, before_end = collator(before_ids, input_ids, tokenizer=tokenizer)

            user_input_ids = tokenizer.encode("### User Input:", add_special_tokens=False)
            _, user_input_end = collator(user_input_ids, input_ids, tokenizer=tokenizer)

            history_start = before_end if before_end is not None else (user_input_end or 0)
            history_end = response_start if response_start is not None else len(input_ids)
            completion_start = response_end if response_end is not None else history_end

            text_dic = build_item_attention_mask(
                input_ids,
                history_start,
                history_end,
                completion_start,
                cans_start=None,
                cans_end=None,
            )
        #for idx, (token_id, attention) in enumerate(zip(text_dic["input_ids"], text_dic["item_attention_mask"])):
            #token_id = int(token_id)
            #attention = int(attention)
            #print(f"idx:{idx} -> id:{token_id} -> token:{tokenizer.convert_ids_to_tokens(token_id)} -> attention:{attention}")
        
        return {
        "input_ids": text_dic["input_ids"],
        "item_attention_mask": text_dic["item_attention_mask"],
    }

    data_files = {
        "train": resolve_demo_path(train_file),
        "validation": resolve_demo_path(eval_file),
    }
    print(f"[data] train_file = {data_files['train']}")
    print(f"[data] eval_file = {data_files['validation']}")

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
        "processed_train_rere_history_only",
        fallback_names=["processed_train"],
    )
    val_cache = resolve_cache_path(
        "processed_val_rere_history_only",
        fallback_names=["processed_val"],
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

    collator = DataCollatorForCompletionOnlyLMWithItemMask(tokenizer.encode(response_template, add_special_tokens = False), tokenizer=tokenizer)

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
            report_to="wandb" if wandb_enabled else "none",
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
            if not state.is_world_process_zero:
                return control

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
            return control

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
            if not state.is_world_process_zero:
                return
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
            if not state.is_world_process_zero:
                return control
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
            return control
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

    if trainer.is_world_process_zero():
        # 将训练好的eval性能最佳的模型保存到指定目录
        trainer.model.save_pretrained(output_dir)
        # 将训练时使用的分词器（含配置 + 词汇表）完整保存到同一目录
        tokenizer.save_pretrained(output_dir)
        save_all_engram_params(base_model, output_dir)
        print(f"\n✅ 训练完成！模型保存到: {output_dir}")


if __name__ == "__main__":
    fire.Fire(train)
