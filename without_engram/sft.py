import os
import sys
import math
import numpy as np
import pandas as pd
import json
import torch
import re
import random
import torch.distributed as dist
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
import transformers
from transformers import PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig,Qwen3ForCausalLM, EarlyStoppingCallback
from datasets import load_dataset,Dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType,PeftModel
from trl import SFTConfig
#from DataCollator import DataCollatorForCompletionOnlyLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
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
import warnings
import torch
import bitsandbytes as bnb
import fire

# Transformer 模型的每一层（如注意力层、全连接层）都有固定的输入维度：
# 比如某层的输入维度是 [batch_size, seq_len, hidden_dim]
# batch_size = 样本数，seq_len = 序列长度，hidden_dim = 隐藏层维度）
# 批量处理（一次处理多个句子）比逐个处理效率高 10 倍以上，使用矩阵张量运算

random.seed(1958)
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
    # full-rank数据（与ReRe风格一致）
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
    learning_rate: float = 2e-5,
    # 最大序列长度，超过该长度的输入将被截断
    cutoff_len: int = 4096,  
    # eval_step：评估频率，表示每训练多少比例的epoch进行一次评估
    eval_step = 0.1,
    # completion起始模板（与prompt中的### Response保持一致）
    response_template: str = "### Response:\n",
):  
    wandb_enabled = setup_wandb_for_sft(wandb_project=wandb_project, wandb_name=wandb_name)
    #加载模型
    if not model_name or model_name == ".":
        print("错误: model_name参数为空或无效")
        print("请检查shell脚本中的--model_name参数设置")
        return
    
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
        # 1) 兼容你MovieLens格式
        if 'extra_info' in data_point:
            prompt_text = parse_prompt_column(data_point.get('prompt', ''))
            description, true_selection = extract_extra_info(data_point.get('extra_info', {}))
            dic = {
                "prompt": prompt_text,
                "completion": true_selection,
                "description": description,
            }
            return dic

        # 2) Industrial full-rank格式：直接使用已构造好的ReRe风格prompt/completion
        prompt_text = parse_prompt_column(data_point.get('prompt', ''))
        true_selection = data_point.get('target_item_title', '')
        if not true_selection:
            # full-rank数据中的completion已是最终目标格式（通常带引号+换行）
            true_selection = str(data_point.get('completion', '')).rstrip('\n')
        else:
            true_selection = f"\"{true_selection}\""

        dic = {
            "prompt": prompt_text,
            "completion": true_selection,
            "description": data_point.get('description', ''),
        }
        return dic

    # 优先使用传入文件，不再写死movielens路径
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

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    base_model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        #torch_dtype=torch.bfloat16,  
    )
    # 训练阶段禁用Transformer模型中的键值缓存机制，缓存会导致无法训练
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    
    def preprocess_function(example):
        """统一的数据预处理函数"""
        prompt = example['prompt']
        completion = example["completion"] + tokenizer.eos_token
        full_text = prompt + completion
        
        # Tokenize
        tokens = tokenizer(
            full_text, 
            truncation=True, 
            max_length=cutoff_len,
            padding=True,
            add_special_tokens=True,
        )["input_ids"]
        
        return {
            "input_ids": tokens,
            "text": full_text,  
        }

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = (
                f"{example['prompt'][i]}"
                f"{example['completion'][i]}"
                f"{tokenizer.eos_token}"
            )
            output_texts.append(text)
        return output_texts

    print("第一个训练数据:", train_data[0])  # 打印第一个训练数据，检查格式是否正确
    sample_one = {
        "prompt": [train_data[0]["prompt"]],
        "completion": [train_data[0]["completion"]],
    }
    sample_full_sentence = formatting_prompts_func(sample_one)[0]
    print("完整训练句子示例:", sample_full_sentence)

    # 改用 named_parameters() 遍历，获取明确的参数名
    for name, param in base_model.named_parameters():
        param.requires_grad_(param.requires_grad)
        # 用遍历的 name 变量（而非 param.name）判断
        if "engram" in name.lower() or "lora" in name.lower():
            param.retain_grad()
    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(
        #训练时 LoRA 适配器可更新参数，推理时固定；
        inference_mode=False,  
        r=8,
        #LoRA适配器的缩放系数
        lora_alpha=32, 
        # gate_proj是 LLaMA 前馈层中实现 “门控机制” 的投影层，作用是生成门控系数，控制up_proj输出特征的保留 / 抑制
        target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj"      # 前馈层
                ], 
        # lora_dropout 是在 LoRA 适配器的计算链路中插入的 dropout 层，具体位置：
        # x @ A → 应用 dropout（随机丢弃10%的激活值）→ @ B → h_lora
        lora_dropout=0.1,    
        # 任务类型：因果语言建模
        task_type="CAUSAL_LM", 
        )    
    if resume_from_checkpoint is not None:
        base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint)
    else:
        base_model = get_peft_model(base_model, peft_config)
    
    device = base_model.device

    for name, param in base_model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
            param.data = param.data.to(device, non_blocking=True)  # 确保GPU设备
            param.retain_grad()  # 强制保留LoRA梯度
            if param.grad is not None:
                param.grad = param.grad.to(device) 
        

    trainable_names = []
    lora_a_names = []
    lora_b_names = []
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
            if "lora_A" in name:
                lora_a_names.append(name)
            elif "lora_B" in name:
                lora_b_names.append(name)
    
    print(f"\n=== 可训练参数统计（应用LoRA后）===")
    print(f"LoRA A参数: {len(lora_a_names)}个")
    print(f"LoRA B参数: {len(lora_b_names)}个")
    print(f"总可训练参数：{len(trainable_names)}个")
    # 打印前5个lora_B参数的状态（验证是否可训练）
    if lora_b_names:
        first_lora_b = lora_b_names[0]
        first_param = [p for n,p in base_model.named_parameters() if n == first_lora_b][0]
        print(f"第一个lora_B参数 {first_lora_b}: requires_grad={first_param.requires_grad}, 初始值前5个: {first_param.data.flatten()[:5].cpu().tolist()}")
    
    for name, param in base_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            print(f"{name} - requires_grad: {param.requires_grad}, 设备: {param.device}, 梯度: {param.grad is not None}")
    


    # 按真实全局batch和梯度累积计算训练步数，避免写死常数和直接截断带来的错位。
    per_device_train_batch_size = batch_size
    world_size = Accelerator().num_processes
    global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_data) / global_batch_size)
    total_train_steps = steps_per_epoch * num_train_epochs

    # 统一转成 float，兼容命令行传入的字符串。
    eval_step_value = float(eval_step)

    # eval_step <= 1 时，按“每个epoch的比例”换算：
    # - 0.5 表示半个 epoch eval/save 一次
    # - 1 表示每个 epoch eval/save 一次
    # eval_step > 1 时，视为直接给定的训练步数。
    if eval_step_value <= 1:
        eval_interval_steps = round(steps_per_epoch * eval_step_value)
    else:
        eval_interval_steps = round(eval_step_value)

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

    # 核心改动：以### Response作为completion起点，得到completion_loss
    collator = DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template, add_special_tokens = False), tokenizer=tokenizer)

    # 训练前先跑一次小batch前向，打印completion_loss（确认collator有效）
    sample_n = min(2, len(train_data))
    sample_batch_examples = {
        "prompt": [train_data[i]["prompt"] for i in range(sample_n)],
        "completion": [train_data[i]["completion"] for i in range(sample_n)],
    }
    sample_texts = formatting_prompts_func(sample_batch_examples)
    # DataCollatorForCompletionOnlyLM 这里需要 tokenized features（含 input_ids）
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

    def get_optimizer_grouped_parameters(model,lora_lr):
        """
        给不同模块参数分组，设置不同学习率：
        - LoRA参数:lora_lr主学习率
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

        # 添加分组（设置不同LR）
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": 0
            })

        
        # 打印分组统计（验证）
        print(f"\n=== 参数分组+学习率配置 ===")
        print(f"LoRA参数:{len(lora_params)}个 | LR={lora_lr}")
        return param_groups

    # 生成参数分组
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        base_model, learning_rate
    )

    optimizer = bnb.optim.PagedAdamW32bit(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_train_steps * 0.05)),  # 和SFTConfig中一致
        num_training_steps=max(1, total_train_steps)
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
            #load_best_model_at_end=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_strategy="steps",
            eval_steps=eval_steps,
            logging_steps=1,
            
            # 优化器和调度器
            #optim="paged_adamw_32bit",
            #optim="adamw_torch",
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
            
            # DDP相关配置，没有插入negram时结构简单不需要查看有没有没用的参数
            ddp_find_unused_parameters=False,
            #ddp_static_graph=False,
        )

    # 创建SFTTrainer实例
    trainer = SFTTrainer(
        base_model,           # 参数名改为 model
        args=sft_config,            # 使用 SFTConfig
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
        optimizers=(optimizer,lr_scheduler),
        processing_class=tokenizer,
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    class LoRAMonitorCallback(transformers.TrainerCallback):
        def __init__(self):
            super().__init__()
            # 记录初始权重（对比是否更新）
            self.init_lora_weights = {}

        def on_train_begin(self, args, state, control, **kwargs):
            """训练开始时记录LoRA初始权重"""
            model = kwargs["model"]
            unwrapped_model = model.module if hasattr(model, 'module') else model
            
            # 记录所有LoRA A/B的初始权重
            for name, param in unwrapped_model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    if param.requires_grad:
                        self.init_lora_weights[name] = param.data.detach().cpu().clone()
            print(f"\n✅ 已记录 {len(self.init_lora_weights)} 个LoRA参数初始权重")

        def on_after_backward(self, args, state, control, **kwargs):
            """反向传播后立即检查（梯度未被清空，最准确）"""
            # 先改成每次都输出，确认回调能触发
            if True:  
                model = kwargs["model"]
                unwrapped_model = model.module if hasattr(model, 'module') else model
                
                # 打印关键计数，帮你定位匹配关系
                print(f"\n==== 调试信息 ====")
                print(f"state.step (backward次数): {state.step}")
                print(f"state.global_step (优化器步数): {state.global_step}")
                print(f"训练日志batch数: {state.num_training_steps}")
                
                # 原有LoRA梯度检查逻辑...
                print(f"\n==== Step {state.global_step} LoRA梯度有效性检查 ====")
                lora_a_valid = False
                lora_b_valid = False
                
                for name, param in unwrapped_model.named_parameters():
                    if "lora_A" in name and param.requires_grad:
                        lora_a_valid = True
                        grad_exists = param.grad is not None
                        grad_is_zero = torch.allclose(param.grad, torch.zeros_like(param.grad)) if grad_exists else True
                        grad_norm = torch.norm(param.grad).item() if grad_exists else 0.0
                        
                        print(f"\n📌 LoRA A - {name}")
                        print(f"   可训练: {param.requires_grad}")
                        print(f"   梯度存在: {grad_exists}")
                        print(f"   梯度全零: {grad_is_zero}")
                        print(f"   梯度范数: {grad_norm:.8f}")
                        print(f"   设备: {param.device}")
                        break
                
                for name, param in unwrapped_model.named_parameters():
                    if "lora_B" in name and param.requires_grad:
                        lora_b_valid = True
                        grad_exists = param.grad is not None
                        grad_is_zero = torch.allclose(param.grad, torch.zeros_like(param.grad)) if grad_exists else True
                        grad_norm = torch.norm(param.grad).item() if grad_exists else 0.0
                        
                        print(f"\n📌 LoRA B - {name}")
                        print(f"   可训练: {param.requires_grad}")
                        print(f"   梯度存在: {grad_exists}")
                        print(f"   梯度全零: {grad_is_zero}")
                        print(f"   梯度范数: {grad_norm:.8f}")
                        print(f"   设备: {param.device}")
                        break
                
                if not lora_a_valid:
                    print("\n❌ 未找到可训练的LoRA A参数")
                if not lora_b_valid:
                    print("\n❌ 未找到可训练的LoRA B参数")
                if lora_a_valid and lora_b_valid:
                    print("\n✅ LoRA A/B均存在可训练参数，梯度检查完成")

        def on_step_end(self, args, state, control, **kwargs):
            """步数结束时检查权重是否更新（最终验证）"""
            if state.global_step % 400 == 0 and state.global_step > 0:
                model = kwargs["model"]
                unwrapped_model = model.module if hasattr(model, 'module') else model
                
                print(f"\n==== Step {state.global_step} LoRA权重更新检查 ====")
                # 检查LoRA权重是否变化
                updated = 0
                total = 0
                for name, param in unwrapped_model.named_parameters():
                    if name in self.init_lora_weights:
                        total += 1
                        # 对比初始权重
                        weight_diff = torch.norm(param.data.cpu() - self.init_lora_weights[name]).item()
                        if weight_diff > 1e-6:
                            updated += 1
                            print(f"📌 {name} - 权重已更新（差值: {weight_diff:.8f}）")
                        else:
                            print(f"📌 {name} - 权重未更新（差值: {weight_diff:.8f}）")
                
                print(f"\n📊 权重更新统计: {updated}/{total} 个LoRA参数已更新")
                if updated == 0:
                    print("⚠️ 警告：所有LoRA参数权重均未更新，训练可能未生效！")
    trainer.add_callback(LoRAMonitorCallback())

    trainer.train() 
    #保存训练器的完整状态
    trainer.save_model(output_dir)

    # 在原始输出目录下创建名为 "final_checkpoint"的子目录
    output_dir = os.path.join(output_dir, "final_checkpoint_sft")
    # 将训练好的eval性能最佳的模型保存到指定目录
    trainer.model.save_pretrained(output_dir)
    # 将训练时使用的分词器（含配置 + 词汇表）完整保存到同一目录
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
