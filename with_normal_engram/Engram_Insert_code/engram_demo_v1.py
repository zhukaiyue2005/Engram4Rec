import pickle
from typing import List
import os
import json
from dataclasses import dataclass, field
import math

import sys
from pathlib import Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

from dataclasses import dataclass, field
from typing import List,Optional,Dict
import numpy as np
import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer,PreTrainedModel, PretrainedConfig,LlamaTokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, NFD, StripAccents, Lowercase, Replace, Strip
from sympy import isprime
from transformers.integrations import use_kernel_forward_from_hub

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # self.weight：可训练的缩放参数（nn.Parameter标识为模型可训练参数），
        # 初始值为全 1 张量，形状为(hidden_size,)，用于对归一化后的特征做自适应缩放；
        # 无偏置项（bias）
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 保存输入特征的原始数据类型（如float16/bf16），用于最终输出还原
        input_dtype = hidden_states.dtype
        # 转换为float32计算，避免低精度（如float16）下的数值溢出/精度损失
        hidden_states = hidden_states.to(torch.float32)
        # 计算最后一维的元素平方的均值（RMS核心：均方根）
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 归一化计算：特征 / (方差 + eps)开根号，torch.rsqrt = 1 / torch.sqrt
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 还原数据类型，并应用可训练缩放参数
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

model_name = os.environ.get("ENGRAM_TOKENIZER_MODEL", "Qwen3-1.7B")
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model_vocab_size = config.vocab_size 
num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
model_compressed_vocab_size = model_vocab_size

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,                 
    local_files_only=True, 
)

tokenizer.padding_side = "left"  # Fix weird overflow issue with fp16 training
tokenizer.bos_token_id = 151643
tokenizer.eos_token_id = 151645
tokenizer.pad_token_id = 151643

# @dataclass 是 Python 标准库 dataclasses 模块提供的类装饰器，核心作用是为普通类自动注入数据类的专属功能
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = os.environ.get("ENGRAM_TOKENIZER_MODEL", "Qwen3-1.7B")
    # field 为数据类的字段配置高级属性
    # default_factory 接收一个无参可调用对象，每次实例化数据类时，自动调用该可调用对象，将返回值作为字段的默认值
    # 定义了engram_size，由于此时只取了2-gram和3-gram，
    # 所以是一个长度为2的列表，为了容量足够大
    engram_vocab_size: List[int] = field(default_factory=lambda: [model_compressed_vocab_size*2, model_compressed_vocab_size*2])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 128   # total embedding dim per n-gram
    n_head_per_ngram: int = 8      # number of hash heads per n-gram
    layer_ids: List[int] = field(default_factory=lambda: [2])   #接入engram的层
    pad_id: int = tokenizer.pad_token_id
    seed: int = 0
    kernel_size: int = 4  # convolution kernel size
    backbone_hidden_size: int = hidden_size
    vocab_size: int = model_compressed_vocab_size
    backbone_num_layers: int = num_layers

    
    
engram_cfg = EngramConfig()

# 压缩分词器
class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
        cache_dir: str = "./compressed_tokenizer_cache",  # 缓存保存目录
        load_from_cache: bool = True,                     # 是否优先加载缓存
    ):  
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,                 
            local_files_only=True, 
        )
        self.cache_dir = cache_dir
        # 生成唯一缓存文件名
        self.cache_filename = self._get_cache_filename(tokenizer_name_or_path)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        # 核心逻辑：优先加载缓存，无缓存则构建并保存
        if load_from_cache and self._cache_exists():
            # 加载缓存的lookup_table和num_new_token
            self.lookup_table, self.num_new_token = self._load_cache()
            print(f"已从缓存加载压缩分词器：{self.cache_filename}")
        else:
            # 首次构建映射表
            self.lookup_table, self.num_new_token = self._build_lookup_table()
            # 保存到缓存
            self._save_cache()
            print(f"首次构建并保存压缩分词器缓存：{self.cache_filename}")
    
    def _get_cache_filename(self, tokenizer_name_or_path: str) -> str:
        """生成唯一的缓存文件名"""
        safe_name = tokenizer_name_or_path.replace("/", "_").replace("\\", "_").replace(":", "_")
        base_name = os.path.splitext(safe_name)[0]          # 拆分文件名和后缀，避免重复后缀
        return os.path.join(self.cache_dir, f"{base_name}_compressed_tokenizer")
    
    def _cache_exists(self) -> bool:
        """检查缓存文件是否存在"""
        return os.path.exists(f"{self.cache_filename}_lookup.npy") and \
               os.path.exists(f"{self.cache_filename}_meta.json")
    
    def _save_cache(self):
        # 创建缓存目录（不存在则创建）
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 保存lookup_table
        np.save(f"{self.cache_filename}_lookup.npy", self.lookup_table)
        
        # 保存num_new_token
        meta = {
            "num_new_token": self.num_new_token,
            "original_vocab_size": len(self.tokenizer)
        }
        with open(f"{self.cache_filename}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    
    def _load_cache(self) -> tuple[np.ndarray, int]:
        """加载缓存的lookup_table和num_new_token"""
        # 加载lookup_table
        lookup_table = np.load(f"{self.cache_filename}_lookup.npy", allow_pickle=False)
        
        # 加载元信息
        with open(f"{self.cache_filename}_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_new_token = meta["num_new_token"]
        
        # 校验：缓存的lookup_table长度需匹配当前分词器词汇量
        if len(lookup_table) != len(self.tokenizer):
            raise ValueError(
                f"缓存的lookup_table长度({len(lookup_table)})与当前分词器词汇量({len(self.tokenizer)})不匹配！"
                "请删除旧缓存后重新初始化。"
            )
        
        return lookup_table, num_new_token
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        # 原始ID(tid) → 新ID(nid) 的临时映射字典
        old2new = {}
        # 归一化后的词汇(key) → 新ID(nid) 的映射字典
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        # 遍历所有原始ID(tid)
        for tid in range(vocab_size):
            # 将原始ID解码为对应文本,保留特殊符号，如[PAD]、[CLS]
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            # 解码后的文本包含�（Unicode替换字符，代表无效/无法解码的乱码字符）
            if "�" in text:
                # 直接使用分词器的「ID转token」方法获取原始token作为key（避免归一化处理乱码）
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                # 复用之前定义的normalizer流水线，对文本做标准化
                norm = self.normalizer.normalize_str(text)
                # 归一化后若为空（如原文本是空白符被归一化为空），则用原始text作为key，避免空key
                key = norm if norm else text

            # 检查当前key是否已分配过新ID（相同key对应同一个nid）
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                # 将新词汇加入最终列表，保持nid与列表索引一致
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        # 遍历所有原始ID，将old2new的映射关系写入numpy数组
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    # 基于前序构建的lookup_table（ID 映射表）实现原始 token ID 的快速压缩转换
    def _compress(self, input_ids):
        # 输入转换为 numpy 数组并指定为整数类型
        arr = np.asarray(input_ids, dtype=np.int64)
        # 生成有效 ID 的位置掩码
        pos_mask = arr >= 0
        out = arr.copy()
        # 提取所有需要转换的有效正整数ID
        valid_ids = arr[pos_mask]
        # 转换有效 ID 并赋值回输出数组
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        # 移除 hc_mult 参数
        activation: bool = True,
    ):
        super().__init__()
        # 移除 hc_mult 相关属性
        self.activation = activation  #是否使用激活函数
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 单分支：通道数直接用 hidden_size
        total_channels = hidden_size
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,    # 分组数=总通道数,卷积操作仅在单个通道内进行，不跨通道融合
            bias=False,               # 关闭偏置
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        # 单分支：仅一个 RMSNorm 层
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        
        if self.activation:
            self.act_fn = nn.SiLU()
        
        self._init_conv()
    # 新添卷积初始化代码
    def _init_conv(self):
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))
        # RMSNorm 权重初始化为 1
        nn.init.ones_(self.norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单分支版本：
        Input:  (B,L,D)
        Output: (B,L,D)
        """
        B, T, C = x.shape
        
        # 单分支归一化
        x_norm = self.norm(x)
        # nn.Conv1d 要求输入形状为（批次→输入通道数→序列长度）
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        # 特征裁剪，恢复原始序列长度
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        # 从 Conv1d 格式还原为 (B,L,D),并确保内存连续性
        y = y_bct.transpose(1, 2).contiguous()
        
        return y
    
    def history_len(self):
        return (self.kernel_size - 1) * self.dilation
    
    # 专门用于推理记忆shrtconv的value值
    def forward_with_history(
        self,
        x_cur: torch.Tensor,            # [B, Lcur, D]  当前推理时的批次
        x_hist: torch.Tensor | None,    # [B, H, D]   记忆的value的值
    ):
        if x_hist is None:
            x_full = x_cur
        else:
            x_full = torch.cat([x_hist, x_cur], dim=1)

        y_full = self.forward(x_full)
        y_cur = y_full[:, -x_cur.shape[1]:, :]

        new_hist = x_full[:, -self.history_len():, :].detach()
        return y_cur, new_hist

# 找到大于start、是质数、未出现在seen_primes中的最小整数
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,  # total embedding dim per n-gram
        n_head_per_ngram,   # number of hash heads per n-gram
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
        cache_dir: str = "./ngram_hash_cache",      # 缓存目录
        load_from_cache: bool = True,               # 优先加载缓存
    ):
        # 基础参数初始化
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids
        self.seed = seed
        self.cache_dir = cache_dir
        self.load_from_cache = load_from_cache

        # 生成唯一缓存文件名
        self.cache_filename = self._get_unique_cache_filename(
                tokenizer_name_or_path,
                layer_ids,
                seed,
                max_ngram_size,
                n_head_per_ngram,
                engram_vocab_size,
            )

        # 导入压缩分词器
        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # 优先加载缓存，无缓存则生成并保存
        if self.load_from_cache and self._cache_exists():
            # 加载缓存的layer_multipliers和vocab_size_across_layers
            self.layer_multipliers, self.vocab_size_across_layers = self._load_cache()
            print(f"已从缓存加载NgramHashMapping:{self.cache_filename}")
        else:
            # 首次生成layer_multipliers
            self.layer_multipliers = self._generate_layer_multipliers()
            # 首次生成vocab_size_across_layers（质数表）
            self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()
            # 保存到缓存
            self._save_cache()
            print(f"首次生成并保存NgramHashMapping缓存:{self.cache_filename}")

        print("\n=== 查看 NgramHashMapping 中参数存储设备 ===")
        # 打印 layer_multipliers 设备
        # NgramHash里的参数都作为numpy数组存储在CPU内存中,它们默认在CPU上
        # 打印 lookup_table 设备
        #print(f"lookup_table 存储设备: {self.compressed_tokenizer.lookup_table.device}")

    def _get_unique_cache_filename(
        self, tokenizer_path, layer_ids, seed, max_ngram, n_head, engram_vocab_size,
    ) -> str:
        """生成唯一缓存名"""
        safe_tokenizer_name = tokenizer_path.replace("/", "_").replace("\\", "_").replace(":", "_")
        vocab_str = "-".join(map(str, engram_vocab_size))
        param_str = (
            f"_layers{'-'.join(map(str, layer_ids))}"
            f"_seed{seed}"
            f"_ngram{max_ngram}"
            f"_head{n_head}"
            f"_vocab{vocab_str}"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"ngram_hash_{safe_tokenizer_name}{param_str}.json")

    def _cache_exists(self) -> bool:
        """检查缓存文件是否存在"""
        return os.path.exists(self.cache_filename)
    
    def _save_cache(self):
        """全部用 JSON 保存 layer_multipliers 和 vocab_size_across_layers"""

        # layer_multipliers: {layer_id: np.ndarray} -> {str(layer_id): list}
        multipliers_serializable = {}
        for layer_id, arr in self.layer_multipliers.items():
            multipliers_serializable[str(layer_id)] = np.asarray(arr, dtype=np.int64).tolist()

        # vocab_size_across_layers: 确保里面都是普通 int
        vocab_serializable = {}
        for layer_id, ngram_list in self.vocab_size_across_layers.items():
            vocab_serializable[str(layer_id)] = [
                [int(prime) for prime in head_list]
                for head_list in ngram_list
            ]

        cache_data = {
            "layer_multipliers": multipliers_serializable,
            "vocab_size_across_layers": vocab_serializable,
        }

        with open(self.cache_filename, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def _load_cache(self) -> tuple[dict, dict]:
        """从 JSON 加载缓存并恢复原始格式"""

        with open(self.cache_filename, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # 恢复 layer_multipliers: list -> np.ndarray
        layer_multipliers = {
            int(layer_id): np.array(arr, dtype=np.int64)
            for layer_id, arr in cache_data["layer_multipliers"].items()
        }

        # 恢复 vocab_size_across_layers
        vocab_size_across_layers = {
            int(layer_id): [
                [int(prime) for prime in head_list]
                for head_list in ngram_list
            ]
            for layer_id, ngram_list in cache_data["vocab_size_across_layers"].items()
        }

        self._validate_cache(layer_multipliers, vocab_size_across_layers)

        return layer_multipliers, vocab_size_across_layers

    def _validate_cache(self, layer_multipliers, vocab_size_across_layers):
        """校验缓存是否匹配当前参数"""
        # 校验图层ID
        if set(layer_multipliers.keys()) != set(self.layer_ids):
            raise ValueError(
                f"缓存的图层ID({set(layer_multipliers.keys())})与当前({set(self.layer_ids)})不匹配！请删除旧缓存。"
            )
        # 校验max_ngram_size
        for layer_id in self.layer_ids:
            ngram_count = len(vocab_size_across_layers[layer_id])
            if ngram_count != (self.max_ngram_size - 1):  
                raise ValueError(
                    f"缓存的N-gram数量({ngram_count})与当前max_ngram_size({self.max_ngram_size})不匹配！"
                )
            # 校验哈希头数量
            for ngram_list in vocab_size_across_layers[layer_id]:
                if len(ngram_list) != self.n_head_per_ngram:
                    raise ValueError(
                        f"缓存的哈希头数量({len(ngram_list)})与当前({self.n_head_per_ngram})不匹配！"
                    )

    def _generate_layer_multipliers(self) -> dict:
        # 获取 64 位有符号整数的最大值，限定哈希计算的数值范围
        max_long = np.iinfo(np.int64).max
        # 为了限制「Token ID × 奇数乘数」的中间计算不溢出 np.int64 类型，
        # Token ID最大值仅和原始压缩 Token 的取值范围（分词器词汇量）相关
        M_max = int(max_long // self.tokenizer_vocab_size)
        # 为后续「乘 2 加 1」生成奇数预留空间
        half_bound = max(1, M_max // 2)
        # 质数常量，用于分层种子不同（保证不同图层种子独立）
        PRIME_1 = 10007
        
        layer_multipliers = {}

        for layer_id in self.layer_ids:
            # 为每个图层生成独立的基础种子
            base_seed = int(self.seed + PRIME_1 * int(layer_id))
            # 基于基础种子初始化随机数生成器
            g = np.random.default_rng(base_seed)
            # 生成随机数：维度(max_ngram_size,)，范围[0, half_bound)，类型int64
            # g.integers生成整数随机数数组：r = g.integers(low, high, size, dtype)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            # 随机数→奇数乘数：r*2+1（保证所有乘数为正奇数，降低哈希碰撞概率）
            # 能让Token ID×乘数的结果分布更分散，避免偶数乘数导致的「乘积奇偶性单一」
            multipliers = r * 2 + 1
            #  保存当前图层的哈希乘数
            layer_multipliers[layer_id] = multipliers
        return layer_multipliers

    # 为每个指定模型图层、每个 N-gram 长度（≥2）的每一个哈希头，分配唯一的质数作为实际词汇表大小
    def calculate_vocab_size_across_layers(self):
        # 全局使用质数集合，保证所有场景无重复
        seen_primes = set()
        # 最终结果字典，键：图层ID，值：分层的N-gram哈希头质数列表
        # 层级为：图层ID → [N-gram2质数列表, N-gram3质数列表, ..., N-gramMax质数列表]
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            # 存储当前图层所有N-gram的哈希头质数，按N-gram长度顺序排列
            all_ngram_vocab_sizes = []
            # 遍历2~max_ngram_size的所有N-gram长度
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                # 取ngram对应词汇表大小
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram                    # 当前N-gram的哈希头数量
                current_prime_search_start = vocab_size - 1         # 质数查找的起始基准（从基础大小前一位开始）
                
                # 质数取模能让大范围的哈希混合值mix在[0, mod-1]范围内均匀分布，
                # 避免合数因存在因数导致的「哈希值扎堆」问题
                for _ in range(num_head):
                    # 找大于start且未在seen_primes中的最小质数
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)                    # 当前质数加入全局集合，避免后续重复使用
                    current_ngram_heads_sizes.append(found_prime)   # 加入当前N-gram的哈希头列表
                    current_prime_search_start = found_prime        # 更新查找起始值
                
                # 当前ngram的哈希头质数收集完成，加入当前图层的N-gram列表
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            # 当前图层的所有N-gram处理完成，存入结果字典
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        # 批次化的压缩 Token ID 序列，形状(B, T)，B= 批次大小，T= 序列长度
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:       # 输出形状(B, T, H)
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        # 取出为layer_id预生成的奇数哈希乘数数组
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x     # k=0为原始序列，无移位
            # 填充参数((0,0), (k,0))：第一维度（批次）无填充，第二维度（序列）左侧填充 k 个 pad_id、右侧无填充
            # 实现序列左移 k 位，原序列左侧空出的位置用 pad_id 填充，保证移位后序列长度仍为 T，与原始序列维度一致
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted
        # 预计算所有可能的 k 位移位结果，k 范围 [0, max_ngram_size-1]
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2        # n-gram索引，从0开始(2-gram对应索引0)
            tokens = base_shifts[:n]    # 切片获取当前ngram所需的n个移位序列（k=0到k=n-1）
            mix = (tokens[0] * multipliers[0])
            # 核心哈希混合计算：Token×乘数 + 按位异或
            # 该函数中，按位异或的操作数是Token ID× 乘数后的整数（而非单个二进制位），
            # 这些整数是 64 位整型（np.int64），包含多个二进制位
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            # 获取当前N-gram对应的哈希头数量和质数取模值
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                # 混合值取模，得到最终哈希索引
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))  # 通过astype(np.int64, copy=False)强制保持类型为 int64,copy=False避免后续转换时产生新的数组拷贝
        # 调用np.stack将列表中 H 个(B, T)形状的数组，沿第 2 维（axis=2）拼接
        # 维度变化：H×(B,T)→(B,T,H),H为所有需要处理的 N-gram（2~max_ngram_size）对应的哈希头总数
        return np.stack(all_hashes, axis=2)

    # 将按图层 ID 组织的哈希结果字典返回给调用方，包含所有指定图层的 N-gram 哈希索引，完成整个哈希转换流程
    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        # 初始化结果字典：按图层 ID 组织哈希结果
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    def __init__(self, 
                 list_of_N: List[int], 
                 D: int, 
                 layer_id: int,
                 cache_dir: str = "./engram_emb_cache", 
                 model_name: str = "engram",
                 backbone_hidden_size : int = 0,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.layer_id = layer_id 
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.backbone_hidden_size = backbone_hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        total_N = sum(list_of_N)
        self.total_N = total_N
        
        # 计算各头部偏移量（不变）
        offsets = [0]
        for i,n in enumerate(list_of_N[:-1]):
            offsets.append(offsets[-1] + n)       
        self._offsets_list = offsets
        
        # 训练模式：分片嵌入层（多GPU均匀分配）
        # 使用 nn.Embedding 的 _weight 分片机制（PyTorch 原生支持）
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D, device=self.device)
    
        self._init_embedding()
        
        self._emb_weight = None
        
        print(f"  training: {self.training}")
        print(f"  _emb_weight: {self._emb_weight}")
        print(f"  embedding shape: {self.embedding.weight.shape}")
        print(f"  embedding device: {self.embedding.weight.device}")

    def _init_embedding(self):
        """初始化嵌入权重（训练模式）"""
        if self.backbone_hidden_size != 0:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / np.sqrt(self.backbone_hidden_size))
            #nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / (10 * np.sqrt(self.backbone_hidden_size)))
        else:
            nn.init.zeros_(self.embedding.weight)

    # input_ids已经是最终的哈希索引
    # input_ids：来自NgramHashMapping._get_ngram_hashes的输出，形状固定为(B, T, H)
    # self.offsets：初始化时生成的偏移量张量，形状固定为 (H,)
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 获取父模块Engram的训练状态
        offsets = torch.tensor(self._offsets_list, dtype=torch.long, device=input_ids.device)

        # 计算偏移后的索引（每个哈希头的索引加上对应偏移量）
        # offsets形状(H,)，广播到(B, T, H)，保证各哈希头索引不重叠
        shifted_input_ids = input_ids + offsets

        # 训练模式：使用PyTorch原生分片Embedding（多GPU自动分配）
        if self.training or self._emb_weight is None:
            target_device = input_ids.device
            weight_device = self.embedding.weight.device

            # Module.to() 会替换 Parameter 对象，导致优化器持有旧参数引用，出现“有梯度但参数不更新”。
            # 在权重所在设备上完成查表，再把结果搬回目标设备。
            lookup_ids = shifted_input_ids.to(weight_device, non_blocking=True)

            output = self.embedding(lookup_ids)
            if output.device != target_device:
                output = output.to(target_device, non_blocking=True)
        
        # 推理模式：从硬盘懒加载权重，按需查表（核心优化：减少显存占用）
        else:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 推理时从self.embedding（CPU上）查表
            if self._emb_weight is False:
                raise RuntimeError("推理前请先执行 Engram.load_all_params() 加载参数！")
            
            target_device = input_ids.device
            weight_device = self.embedding.weight.device
            
            lookup_ids = shifted_input_ids.to(weight_device, non_blocking=False)
            
            output = self.embedding(lookup_ids)
            
            # 读取嵌入向量（仅加载当前batch需要的部分到GPU）
            target_device = input_ids.device
            
            if output.device != target_device:
                output = output.to(target_device, non_blocking=True)
        
        return output
    
class Engram(nn.Module):
    def __init__(self, 
                 layer_id,
                 config: EngramConfig,
                 cache_dir: str = "./engram_emb_cache",
                 model_name: str = "engram"):
        super().__init__()
        self.layer_id = layer_id
        self.config = config 
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=self.config.engram_vocab_size,
            max_ngram_size = self.config.max_ngram_size,
            n_embed_per_ngram = self.config.n_embed_per_ngram,
            n_head_per_ngram = self.config.n_head_per_ngram,
            layer_ids = [layer for layer in self.config.layer_ids],
            tokenizer_name_or_path=self.config.tokenizer_name_or_path,
            pad_id = self.config.pad_id,
            seed = self.config.seed,
        )
        vocab_sizes = self.hash_mapping.vocab_size_across_layers[self.layer_id]
        list_of_N = [x for y in vocab_sizes for x in y]


        self.multi_head_embedding = MultiHeadEmbedding(
            # NgramHashMapping 中为每个哈希头分配的质数 mod的列表(所有nagram的)
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            # 单 N-gram 总嵌入维度向多哈希头做平均拆分
            D = self.config.n_embed_per_ngram // self.config.n_head_per_ngram,
            model_name = model_name,
            cache_dir = cache_dir,
            layer_id = layer_id,
            backbone_hidden_size = self.config.backbone_hidden_size
        )
        self.short_conv = ShortConv(
            hidden_size = self.config.backbone_hidden_size,
            kernel_size = self.config.kernel_size,
            dilation    = self.config.max_ngram_size,
        )
        engram_hidden_size = (self.config.max_ngram_size-1) * self.config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.config.backbone_hidden_size, bias=False)
        self.key_proj = nn.Linear(engram_hidden_size, self.config.backbone_hidden_size, bias=False)

        self.norm1 = Qwen3RMSNorm(self.config.backbone_hidden_size)
        self.norm2 = Qwen3RMSNorm(self.config.backbone_hidden_size)

        self._init_engram()
        # 触发MultiHeadEmbedding初始化
        self.multi_head_embedding._init_embedding()
        # 触发ShortConv初始化
        self.short_conv._init_conv()

        # 输出子模块设备信息
        self._print_submodule_device()

        # 标记多头嵌入参数前缀，用于过滤
        self.emb_param_prefix = "multi_head_embedding."

        self._token_hist_cache = None   # [B, max_ngram_size-1]
        self._value_hist_cache = None   # [B, history_len, D]
    
    def _init_engram(self):
        for layer in [self.value_proj, self.key_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for norm in [self.norm1, self.norm2]:
            nn.init.ones_(norm.weight)
    
    def reset_inference_cache(self):
        self._token_hist_cache = None
        self._value_hist_cache = None

    def reorder_inference_cache(self, beam_idx: torch.LongTensor):
        """
        按 beam_idx 重排 Engram 的推理缓存，避免 beam search 过程中不同 beam 的历史状态串线。
        注意：这个函数不会在 Engram.forward 内部被直接调用；
        它的调用入口在外层模型的 `_reorder_cache`（如 modeling_qwen3.py),
        只会在 `generate(..., num_beams>1)` 等需要 beam 重排的路径触发。

        参数:
            beam_idx: [batch_size * num_beams],每个新beam对应旧beam的来源索引。
        """
        # 没有 beam 重排索引时，直接返回（例如非 beam search 路径）。
        if beam_idx is None:
            return

        # -----------------------------
        # 1) 重排 token 历史缓存 _token_hist_cache
        # shape: [B_cache, hist_len]
        # -----------------------------
        if self._token_hist_cache is not None:
            # 当前缓存（旧 beam 的历史）
            src = self._token_hist_cache
            # 将 beam_idx 放到与缓存一致的设备，避免跨设备 index_select 报错
            idx = beam_idx.to(src.device)
            # B_cache: 当前缓存里的 batch 数（重排前）
            b_src = src.size(0)
            # B_target: 目标 batch 数（通常是 batch_size * num_beams）
            b_tgt = idx.size(0)

            # 常规情况：缓存 batch 与目标 batch 一致，直接按 beam_idx 重排
            # 语义：新 beam i 取旧 beam idx[i] 的历史状态
            if b_src == b_tgt:
                self._token_hist_cache = src.index_select(0, idx).detach()
            elif b_src > 0 and b_tgt % b_src == 0:
                # 兜底：如果缓存尚未按beam展开，先repeat到目标batch再重排。
                # 例如 b_src=batch_size, b_tgt=batch_size*num_beams，
                # 先把每条样本历史复制 num_beams 份，再做精确重排。
                repeat = b_tgt // b_src
                src_expanded = src.repeat_interleave(repeat, dim=0)
                self._token_hist_cache = src_expanded.index_select(0, idx).detach()
            else:
                # 维度关系异常，无法安全重排，直接抛错防止缓存串线
                raise RuntimeError(
                    f"Engram token cache batch mismatch for reorder: cache={b_src}, beam_idx={b_tgt}"
                )

        # -----------------------------
        # 2) 重排 value 历史缓存 _value_hist_cache
        # shape: [B_cache, history_len, D]
        # -----------------------------
        if self._value_hist_cache is not None:
            src = self._value_hist_cache
            idx = beam_idx.to(src.device)
            b_src = src.size(0)
            b_tgt = idx.size(0)

            # 与 token cache 相同逻辑：batch 一致时直接按 idx 重排
            if b_src == b_tgt:
                self._value_hist_cache = src.index_select(0, idx).detach()
            elif b_src > 0 and b_tgt % b_src == 0:
                # 缓存尚未 beam 展开时，先扩展再重排
                repeat = b_tgt // b_src
                src_expanded = src.repeat_interleave(repeat, dim=0)
                self._value_hist_cache = src_expanded.index_select(0, idx).detach()
            else:
                # 维度不匹配时终止，避免后续 token 计算读到错误历史
                raise RuntimeError(
                    f"Engram value cache batch mismatch for reorder: cache={b_src}, beam_idx={b_tgt}"
                )

    def _to_cpu_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return ids if ids.device.type == "cpu" else ids.cpu()

    def _ids_to_embeddings(self, ids: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        ids = self._to_cpu_ids(ids)
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(ids.numpy())[self.layer_id]).to(target_device, dtype=torch.long)
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        return embeddings

    def _compute_gate_value(self, hidden_states: torch.Tensor, embeddings: torch.Tensor):
        key = self.key_proj(embeddings)
        normed_key = self.norm1(key)

        query = hidden_states
        normed_query = self.norm2(query)

        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.config.backbone_hidden_size)
        #gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)

        value = gate * self.value_proj(embeddings)
        return gate, value

    def _forward_full(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        # full sequence path: training / non-incremental inference
        embeddings = self._ids_to_embeddings(input_ids, hidden_states.device)
        gate, value = self._compute_gate_value(hidden_states, embeddings)

        #conv_out = self.short_conv(value)
        #output = value + conv_out
        output = value

        return output, gate, value

    def _forward_incremental(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        ids_cur = self._to_cpu_ids(input_ids)  # [B, Lcur]

        if self._token_hist_cache is None:
            ids_for_hash = ids_cur
        else:
            ids_for_hash = torch.cat([self._token_hist_cache, ids_cur], dim=1)

        embeddings_all = self._ids_to_embeddings(ids_for_hash, hidden_states.device)
        query_len = hidden_states.shape[1]
        embeddings = embeddings_all[:, -query_len:, :]

        gate, value = self._compute_gate_value(hidden_states, embeddings)

        #conv_out, new_value_hist = self.short_conv.forward_with_history(x_cur=value,x_hist=self._value_hist_cache,)
        #output = value + conv_out

        output = value
        self._update_token_hist_cache(ids_cur)
        #self._value_hist_cache = new_value_hist

        return output, gate
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        is_first_iteration: bool = False,
    ):
        if self.training:
            self.multi_head_embedding._emb_weight = None
        
        if self.training or self.multi_head_embedding._emb_weight is None:
            self.multi_head_embedding._emb_weight = None
            output, gate, _ = self._forward_full(hidden_states, input_ids)
            return output, gate

        # 推理时，如果不用 cache，直接正常全部计算
        if not use_cache:
            output, gate, _ = self._forward_full(hidden_states, input_ids)
            return output, gate

        # generate 第一轮：整段 prompt 跑一遍，并初始化内部缓存
        if is_first_iteration:
            self.reset_inference_cache()
            output, gate, value = self._forward_full(hidden_states, input_ids)

            ids_cpu = self._to_cpu_ids(input_ids)
            self._update_token_hist_cache(ids_cpu)

            #if self.short_conv.history_len() > 0:
                #self._value_hist_cache = value[:, -self.short_conv.history_len():, :].detach()
            #else:
                #self._value_hist_cache = None

            return output, gate

        # generate 后续步：只计算当前token，并自动滚动内部缓存
        return self._forward_incremental(hidden_states, input_ids)

    def _update_token_hist_cache(self, ids_cur: torch.Tensor):
        # ids_cur: [B, Lcur] on CPU
        hist_len = self.config.max_ngram_size - 1

        if self._token_hist_cache is None:
            full = ids_cur
        else:
            full = torch.cat([self._token_hist_cache, ids_cur], dim=1)

        self._token_hist_cache = full[:, -hist_len:].detach()

    def _print_submodule_device(self):
        """
        输出每个子模块的设备GPU/CPU信息。
        """
        print(f"\n==== 子模块设备信息 ====")
        
        # 输出所有层的设备信息
        for name, module in self.named_modules():
            # 确保该子模块有可训练的参数
            if len(list(module.parameters())) > 0:
                device = next(module.parameters()).device  # 获取当前模块的设备信息
                print(f"{name} : {device}")
            else:
                print(f"{name} : 没有可训练参数，跳过设备显示")
        
        # 显示 Ngram 模块中的质数及其数量
        print("\n==== Ngram哈希模块质数及数量 ====")
        for layer_id in self.config.layer_ids:
            vocab_sizes = self.hash_mapping.vocab_size_across_layers[layer_id]
            print(f"Layer {layer_id} N-gram 质数列表：")
            for ngram_size, primes in enumerate(vocab_sizes, 2):
                print(f"  {ngram_size}-gram 质数: {primes}")

    def save_all_params(self, save_path: str, initial_params: dict = None, warn_threshold: float = 1e-6):
        """
        保存所有参数（带有效性验证，检测是否真正更新过）
        
        :param save_path: 保存路径（.npy/.json)
        :param initial_params: 可选，初始参数字典，用于对比更新幅度
        :param warn_threshold: 更新幅度阈值，低于此值认为可能未更新
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        params = {}
        
        print(f"\n{'='*60}")
        print(f"开始保存 Engram 参数到: {save_path}")
        print(f"{'='*60}")
        
        # 遍历所有可训练参数
        total_params = 0
        updated_params = 0
        suspicious_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            total_params += 1
            
            # 转换参数为 numpy
            if param.dtype == torch.bfloat16:
                param_np = param.detach().cpu().float().numpy().copy()
            else:
                param_np = param.detach().cpu().numpy().copy()
            
            params[name] = param_np
            
            # ========== 统计信息计算 ==========
            param_mean = np.mean(param_np)
            param_std = np.std(param_np)
            param_norm = np.linalg.norm(param_np)
            param_min = np.min(param_np)
            param_max = np.max(param_np)
            
            # 检查是否可能是随机初始化（正态分布特征）
            # 随机初始化通常：mean≈0, std≈初始化标准差(如0.02)
            is_random_like = (abs(param_mean) < 0.001 and 
                            0.01 < param_std < 0.1 and 
                            param_norm > 0)
            
            # 如果有初始值，计算更新幅度
            update_info = ""
            if initial_params is not None and name in initial_params:
                init_np = initial_params[name]
                if init_np.shape == param_np.shape:
                    diff = np.linalg.norm(param_np - init_np)
                    rel_diff = diff / (np.linalg.norm(init_np) + 1e-8)
                    update_info = f" | 更新幅度: {diff:.6f} (相对: {rel_diff:.2%})"
                    
                    if diff > warn_threshold:
                        updated_params += 1
                    else:
                        suspicious_params.append((name, diff))
            
            # 打印参数统计
            status = "⚠️ 疑似随机" if is_random_like and initial_params is None else "✅"
            print(f"{status} {name}:")
            print(f"    shape={param_np.shape}, dtype={param.dtype}")
            print(f"    mean={param_mean:}, std={param_std:}, norm={param_norm:}")
            print(f"    range=[{param_min:}, {param_max:}]{update_info}")
        
        # ========== 整体统计 ==========
        print(f"\n{'='*60}")
        print(f"参数保存统计:")
        print(f"  总参数数量: {total_params}")
        if initial_params is not None:
            print(f"  已更新参数: {updated_params}/{total_params}")
            if suspicious_params:
                print(f"  ⚠️ 疑似未更新参数: {len(suspicious_params)}个")
                for name, diff in suspicious_params[:5]:  # 只显示前5个
                    print(f"    - {name}: 更新幅度={diff:.8f}")
        print(f"{'='*60}")
        
        # 如果没有参数，发出警告
        if not params:
            print("🚨 警告: 没有可训练参数被保存！请检查:")
            print("   1. 参数是否被冻结 (requires_grad=False)")
            print("   2. named_parameters() 是否返回空")
            return None
        
        # ========== 保存文件 ==========
        try:
            if save_path.endswith(".npy"):
                np.save(save_path, params)
            elif save_path.endswith(".json"):
                serializable_params = {}
                for name, param in params.items():
                    if param.ndim > 2:
                        serializable_params[name] = {
                            "shape": list(param.shape),
                            "data": param.flatten().tolist()
                        }
                    else:
                        serializable_params[name] = param.tolist()
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_params, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError("仅支持保存为 .npy 或 .json 文件")
            
            # 验证文件是否正确写入
            file_size = os.path.getsize(save_path)
            print(f"\n✅ 成功保存 {len(params)} 个参数")
            print(f"   文件路径: {save_path}")
            print(f"   文件大小: {file_size / 1024:.2f} KB")
            
            # 额外验证：尝试重新加载
            if save_path.endswith(".npy"):
                verify_params = np.load(save_path, allow_pickle=True).item()
                if len(verify_params) == len(params):
                    print(f"   验证通过: 文件可正常读取，包含 {len(verify_params)} 个参数")
                else:
                    print(f"   ⚠️ 验证失败: 保存{len(params)}个，读取到{len(verify_params)}个")
            
            return params
        
        except Exception as e:
            print(f"\n🚨 保存失败: {e}")
            raise
    
    def load_all_params(self, load_path: str, min_norm_threshold: float = 1e-6):
        """
        加载所有参数（训练/推理自动区分,兼容bfloat16,带参数有效性验证)
        - 训练：加载所有参数（含嵌入）到显存
        - 推理：仅加载非嵌入参数到显存
        
        :param load_path: 参数文件路径
        :param min_norm_threshold: 参数范数最小阈值，低于此值认为参数无效（可能是零初始化或未训练）
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"参数文件不存在：{load_path}")
        
        print(f"\n{'='*60}")
        print(f"开始从 {load_path} 加载 Engram 参数")
        print(f"{'='*60}")
        
        # 加载文件
        if load_path.endswith(".npy"):
            params = np.load(load_path, allow_pickle=True).item()
        elif load_path.endswith(".json"):
            with open(load_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            for name, data in params.items():
                if isinstance(data, dict) and "shape" in data:
                    params[name] = np.array(data["data"], dtype=np.float32).reshape(data["shape"])
                else:
                    params[name] = np.array(data, dtype=np.float32)
        else:
            raise ValueError("仅支持加载 .npy 或 .json 文件")
        
        print(f"文件包含 {len(params)} 个参数")
        
        # 将参数加载到模型（带范数验证）
        loaded_count = 0
        invalid_params = []
        valid_norms = []
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                # 尝试匹配参数名（支持短名匹配）
                param_np = None
                matched_name = None
                
                if name in params:
                    param_np = params[name]
                    matched_name = name
                else:
                    # 后缀匹配
                    short_name = name.split(".engram.")[-1] if ".engram." in name else name
                    if short_name in params:
                        param_np = params[short_name]
                        matched_name = short_name
                
                if param_np is None:
                    print(f"⚠️ 未找到参数: {name}")
                    continue
                
                # ========== 核心验证：检查参数范数 ==========
                param_norm = np.linalg.norm(param_np)
                param_mean = np.mean(param_np)
                param_std = np.std(param_np)
                
                print(f"  加载 {name}:")
                print(f"    shape={param_np.shape}, norm={param_norm:.6f}, mean={param_mean:.6f}, std={param_std:.6f}")
                
                # 验证参数是否有效（范数足够大）
                if param_norm < min_norm_threshold:
                    invalid_params.append((name, param_norm))
                    print(f"    ❌ 警告: 参数范数 {param_norm:.8f} 小于阈值 {min_norm_threshold}，可能是零初始化或未训练！")
                else:
                    valid_norms.append(param_norm)
                    print(f"    ✅ 参数有效")
                
                # 转换为 tensor 并加载
                param_tensor = torch.tensor(param_np, dtype=param.dtype)
                
                if self.training:
                    param.copy_(param_tensor.to(param.device))
                    print(f"    训练模式 - 已加载到 {param.device}")
                else:
                    # 推理模式特殊处理嵌入参数
                    if "multi_head_embedding" in name or (hasattr(self, 'emb_param_prefix') and name.startswith(self.emb_param_prefix)):
                        print(f"\n[DEBUG]处理嵌入参数: {name}")
                        print(f"  加载的权重形状: {param_tensor.shape}")
                        print(f"  当前嵌入层形状: {self.multi_head_embedding.embedding.weight.shape}")
                        print(f"  当前 total_N: {self.multi_head_embedding.total_N}")
                        
                        # 验证形状
                        if param_tensor.shape[0] != self.multi_head_embedding.embedding.num_embeddings:
                            print(f"  ⚠️ 形状不匹配! 加载:{param_tensor.shape[0]}, 当前:{self.multi_head_embedding.embedding.num_embeddings}")

                        cpu_embedding = nn.Embedding(
                                num_embeddings=self.multi_head_embedding.embedding.num_embeddings,
                                embedding_dim=self.multi_head_embedding.embedding.embedding_dim
                            ).to('cpu')
                        
                        # 将加载的权重直接复制到CPU嵌入层
                        with torch.no_grad():
                            cpu_embedding.weight.copy_(param_tensor.to('cpu'))
                        
                        # 替换嵌入层
                        self.multi_head_embedding.embedding = cpu_embedding
                        self.multi_head_embedding._emb_weight = True

                    else:
                        param.copy_(param_tensor.to(param.device))
                        print(f"    推理模式 - 已加载到 {param.device}")
                
                loaded_count += 1
        
        # ========== 加载完成统计 ==========
        print(f"\n{'='*60}")
        print(f"加载统计:")
        print(f"  成功加载: {loaded_count} 个参数")
        print(f"  有效参数: {len(valid_norms)} 个")
        
        if valid_norms:
            print(f"  平均范数: {sum(valid_norms)/len(valid_norms):.6f}")
            print(f"  最小范数: {min(valid_norms):.6f}")
            print(f"  最大范数: {max(valid_norms):.6f}")
        
        if invalid_params:
            print(f"  ❌ 无效参数: {len(invalid_params)} 个（范数过小）")
            for name, norm in invalid_params[:5]:
                print(f"    - {name}: norm={norm:.8f}")
        else:
            print(f"  ✅ 所有参数范数正常")
        
        print(f"{'='*60}")
        
        # 如果有无效参数，发出警告
        if invalid_params:
            print(f"\n🚨 警告: {len(invalid_params)} 个参数可能未正确训练（范数过小）")
            print(f"   建议检查训练过程或重新训练")
        
        return loaded_count, invalid_params

__all__=[ "EngramConfig", "Engram","CompressedTokenizer","BackBoneConfig"]
