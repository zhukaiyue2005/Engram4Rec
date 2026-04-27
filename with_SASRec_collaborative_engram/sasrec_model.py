import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """对用户历史 item 序列做多头自注意力。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        输入整段历史序列的隐状态，输出每个位置融合上下文后的表示。

        这里做注意力的原因是：
        - 用户历史不是无结构的 item 列表，前后 item 之间存在顺序依赖；
        - 当前要预测下一个 item 时，不同历史 item 的贡献不同；
        - 自注意力可以让每个位置动态关注更相关的历史位置，而不是只靠固定窗口或平均池化。
        """
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """SASRec 的一个基本块：先做自注意力，再做前馈网络。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """对序列表示做一层上下文交互和非线性变换。"""
        x = x + self.dropout(self.attn(self.ln1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


@dataclass
class SASRecConfig:
    """SASRec 的结构超参数。"""

    num_items: int
    max_seq_len: int = 50
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.2


class SASRec(nn.Module):
    """
    一个简化版 SASRec。

    它做两件事：
    1. 学到每个 item 的 embedding；
    2. 根据用户历史序列预测下一个 item。

    对你后续的用途来说，最重要的产物有两个：
    - `self.item_embeddings.weight`：每个 item 的静态表示；
    - `encode_sequence(...)` 的输出：给定一段用户历史后的动态序列表征。
    """

    def __init__(self, config: SASRecConfig):
        """
        初始化模型参数。

        主要包括：
        - `item_embeddings`：item id -> item 向量
        - `pos_embeddings`：位置编码，保留序列顺序信息
        - `blocks`：多层 Transformer block，用于建模历史 item 之间的关系
        - `output`：最终分类头，把最后一个历史位置的表示映射到所有 item 的打分
        """
        super().__init__()
        self.config = config
        self.padding_item_id = config.num_items

        self.item_embeddings = nn.Embedding(config.num_items + 1, config.hidden_size)
        # 可学习的位置编码。
        # 仅有 item embedding 时，模型不知道历史 item 的先后顺序；
        # 这里为序列中每个位置提供一个位置向量，再与 item embedding 相加，
        # 让模型区分“同一个 item 出现在不同历史位置”的差异。
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.hidden_size, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.num_items)

        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    def _build_attention_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """
        构造注意力 mask。

        这里同时做两件事：
        - 因果 mask：当前位置只能看自己和之前的 item，不能偷看未来；
        - padding mask：补齐出来的位置不能参与注意力。
        """
        bsz, seq_len = seq.shape
        causal = torch.tril(torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool))
        valid = (seq != self.padding_item_id).unsqueeze(1).unsqueeze(2)
        return (causal.unsqueeze(0).unsqueeze(1) & valid).to(dtype=torch.bool)

    def encode_sequence(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        编码整段用户历史，并取出最后一个有效位置的隐状态。

        这一步的流程是：
        1. item embedding + position embedding
        2. 经过多层自注意力 block
        3. 取每条序列最后一个非 padding 位置的表示

        返回的 `h` 是“这段历史序列的动态表示”，用于预测下一个 item。
        """
        bsz, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(bsz, -1)

        x = self.item_embeddings(seq) + self.pos_embeddings(pos)
        x = self.emb_dropout(x)
        x = x * (seq != self.padding_item_id).unsqueeze(-1)

        attn_mask = self._build_attention_mask(seq)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        last_idx = (lengths - 1).clamp(min=0)
        h = x[torch.arange(bsz, device=seq.device), last_idx]
        return h

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        训练时的主前向过程。

        最终输出在这里：
        - 先通过 `encode_sequence` 得到最后一个历史位置的表示 `h`
        - 再通过 `self.output(h)` 映射成对所有 item 的 logits

        所以这个函数的返回值形状是：
        - `[batch_size, num_items]`

        它表示：对候选 item 空间里每个 item 的预测分数。
        """
        h = self.encode_sequence(seq, lengths)
        return self.output(h)

    def score_all_items(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """评估时调用，返回对所有 item 的分数，后续可直接做 top-k 排序。"""
        return self.forward(seq, lengths)

    def item_embed(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        直接取出指定 item 的 embedding。

        这个接口返回的是“每个 item 的静态向量表示”，不是用户历史条件下的动态表示。
        后面如果你要把 SASRec 的 item 信息注入到大模型里，通常先从这里取向量。
        """
        return self.item_embeddings(item_ids)
