import os
import sys
import math
import numpy as np
import torch
import pandas as pd
import re
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
from typing import Any, List, Union
import torch.optim as optim
#from DataCollator import DataCollatorForCompletionOnlyLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_TRAIN_FILE = os.path.join(DATA_ROOT, "Industrial_and_Scientific_dataset", "train.jsonl")
DEFAULT_EVAL_FILE = os.path.join(DATA_ROOT, "Industrial_and_Scientific_dataset", "valid.jsonl")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from sft_wandb import setup_wandb_for_sft

os.environ['HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION'] = '1'
os.environ['TORCH_NN_MODULE_USE_DTENSOR'] = '0'
os.environ['USE_DTENSOR'] = '0'
os.environ['ACCELERATE_USE_DTENSOR'] = '0'
os.environ["USE_FLASH_ATTENTION"] = "0" 
from accelerate import Accelerator

# 强制重写huggingface_hub的校验函数
from huggingface_hub.utils._validators import validate_repo_id
def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return  # 空函数，跳过所有校验
validate_repo_id.__code__ = _dummy_validate_repo_id.__code__  
engram_dir = os.path.join(os.path.dirname(__file__), "Engram_Insert_code")
sys.path.append(engram_dir)
from engram_demo_v1 import Engram, EngramConfig
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
            raise ValueError(f"{arg_name} 中包含非法整数值: {x!r}, 原始输入: {arg_value!r}")
    return out

# ========== 核心训练函数 ==========
def train(
    output_dir="",   
    logging_dir="",  
    model_name ="",    
    prompt_path = "",  
    dataset="",
    train_file=DEFAULT_TRAIN_FILE,
    eval_file=DEFAULT_EVAL_FILE,
    resume_from_checkpoint: str = None,  
    wandb_project: str = "",       
    wandb_name: str = "",
    gradient_accumulation_steps: int = 1, 
    batch_size: int = 8,
    num_train_epochs: int = 10,
    lora_lr: float = 2e-5,          # LoRA参数学习率（主学习率）
    engram_lr: float = 5e-5,        # Engram模块学习率（可单独调大/小）
    learning_rate: float = 2e-5,
    engram_layer_ids="5,10,15",
    cutoff_len: int = 1024,  
    eval_step = 0.1,
    response_template: str = "### Response:\n",
):  
    wandb_enabled = setup_wandb_for_sft(wandb_project=wandb_project, wandb_name=wandb_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,                 
        local_files_only=True, 
    )

    tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

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
    
    def parse_reasoning_column(prompt_col):
        """解析reasoning列,提取推理内容"""
        if isinstance(prompt_col, list):
            # 如果是列表，取第一个assistant角色的content
            for item in prompt_col:
                if isinstance(item, dict) and item.get('role') == 'assistant':
                    return item.get('reasoning', ''),item.get('content', '')
            return ''
        

    def extract_extra_info(extra_info_col):
        """从extra_info列提取ground_truth的description和title"""
        if isinstance(extra_info_col, dict):
            ground_truth = extra_info_col.get('ground_truth', {})
            description = ground_truth.get('description', '')
            title = ground_truth.get('title', '')
            return description, title
        return "", ""

    def process_data(data_point):
        """处理单条数据,生成SFT格式"""
        if 'extra_info' in data_point:
            prompt_text = parse_prompt_column(data_point.get('prompt', ''))
            description, true_selection = extract_extra_info(data_point.get('extra_info', {}))
            dic = {
                "prompt": prompt_text,
                "completion": true_selection,
                "description": description,
            }
            return dic

        prompt_text = parse_prompt_column(data_point.get('prompt', ''))
        true_selection = data_point.get('target_item_title', '')
        if not true_selection:
            true_selection = str(data_point.get('completion', '')).rstrip('\n')
        else:
            true_selection = f"\"{true_selection}\""

        dic = {
            "prompt": prompt_text,
            "completion": true_selection,
            "description": data_point.get('description', ''),
        }
        return dic

    def process_train_data(data_point):
        prompt_text = parse_prompt_column(data_point['messages'])
        description, true_selection = parse_reasoning_column(data_point['messages'])
        dic = {
            "prompt": prompt_text,
            "completion": true_selection,
            "description": description, 
        }   
        return dic
    

    data_files = {
        "train": train_file,
        "validation": eval_file,
    }


    train_data = load_dataset("json", data_files={"train": data_files["train"]})["train"]
    train_data = train_data.shuffle(seed=42).map(process_data)

    val_data = load_dataset("json", data_files={"validation": data_files["validation"]})["validation"]
    val_data = val_data.shuffle(seed=42).map(process_data)

    print("✅ 数据加载和预处理完成")
    print(train_data[0])

    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    layer_ids = _parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if not layer_ids:
        raise ValueError("engram_layer_ids 不能为空")
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

    # 配置并应用LoRA
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
        
        unwrapped_model = model.module if hasattr(model, 'module') else model
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

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['prompt'])):
            text = (
                f"{example['prompt'][i]}"
                f"{example['completion'][i]}"
                f"{tokenizer.eos_token}"
            )
            output_texts.append(text)
        return output_texts

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer.encode(response_template, add_special_tokens=False),
        tokenizer=tokenizer,
    )

    sample_n = min(2, len(train_data))
    sample_batch_examples = {
        "prompt": [train_data[i]["prompt"] for i in range(sample_n)],
        "completion": [train_data[i]["completion"] for i in range(sample_n)],
    }
    sample_texts = formatting_prompts_func(sample_batch_examples)
    print("完整训练句子示例:", sample_texts[0].replace("\n", "\\n") if sample_texts else "")
    sample_features = [
        tokenizer(
            text,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=True,
        )
        for text in sample_texts
    ]
    sample_batch = collator(sample_features)
    sample_batch = {k: v.to(base_model.device) for k, v in sample_batch.items()}
    base_model.eval()
    with torch.no_grad():
        preview_out = base_model(**sample_batch)
    print(f"✅ Collator completion_loss预览: {preview_out.loss.item():.6f}")

    # 按真实全局batch和梯度累积计算训练步数，避免写死常数和直接截断带来的错位。
    per_device_train_batch_size = batch_size
    world_size = Accelerator().num_processes
    global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_data) / global_batch_size)
    total_train_steps = steps_per_epoch * num_train_epochs

    # eval_step <= 1 时，按“每个epoch的比例”换算：
    # - 0.1 表示每个 epoch 的 10% 评估一次
    # - 1 表示每个 epoch 评估一次
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

    sft_config = SFTConfig(
            # 训练参数
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            num_train_epochs=num_train_epochs,
            #learning_rate=learning_rate,
            # 使用bfloat16混合精度训练
            bf16=True,
            
            # 保存和评估策略
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=100,
            #load_best_model_at_end=True,
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

    # 创建SFTTrainer实例
    trainer = SFTTrainer(
        base_model,           # 参数名改为 model
        args=sft_config,            # 使用 SFTConfig
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        optimizers=(optimizer,lr_scheduler),
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        callbacks=[EngramSaveCallback()]
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

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
