"""
Recurrent Transformer 的探索性实现.

论文: The Recurrent Transformer: Greater Effective Depth and Efficient Decoding (arXiv 2604.21215)

与标准 Transformer 的区别在于 attention 使用的 KV 从哪里来:
- 标准 Transformer: 第 l 层的 KV 由第 l 层的输入投影得到, 也就是上一层的输出.
- Recurrent Transformer: 第 l 层在位置 t 的持久 KV 由本层在位置 t 的激活投影得到.
  后续位置 attend 到这些持久 KV, 同一层内因此沿时间维度形成递归.

注释中的张量维度记号:
    b   batch size
    n   本次调用处理的序列长度, 递归模式下是一个 block 的长度
    m   memory 中已有的 KV 个数, 即当前 tokens 之前的位置数
    h   attention head 数
    d   每个 head 的维度 (dim_head)
    i   query 个数, j   key 个数
"""

from __future__ import annotations
from functools import partial

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm, Linear, Sequential

import einx
from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from x_mlps_pytorch import MLP

# constants

# 不带 bias 的 Linear, 用于 QKV 投影, 输出投影和 LoRA
LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def inv_sqrt(n):
    # 1 / sqrt(n), 用于 attention 缩放和初始化标准差
    return n ** -0.5

# lora

class LoRA(Module):
    """
    低秩线性映射 dim -> low_rank -> dim_out, 两个 Linear 都不带 bias.
    本仓库用它从 norm 后的输入生成 attention gate 的 logits.
    """

    def __init__(
        self,
        dim,
        dim_out,
        low_rank = 32
    ):
        super().__init__()

        # 秩必须同时小于输入维度和输出维度, 保证这是真正的低秩分解
        # 在 Attention 中 dim_out = dim_head * heads, 所以 gate_low_rank 也受 attention 内部维度限制
        assert low_rank < dim and low_rank < dim_out, f'low rank ({low_rank}) must be less than dim ({dim}) and dim_out ({dim_out})'

        # 命名与方向相反: up 把 dim 压缩到 low_rank, down 再映射到 dim_out
        self.up = LinearNoBias(dim, low_rank)
        self.down = LinearNoBias(low_rank, dim_out)

    def forward(self, x):
        # (..., dim) -> (..., dim_out)
        return self.down(self.up(x))

# positional bias

# relative positional bias borrowed from Thinking Machines' Inkling model
# the learned alibi variant builds on ALiBi - Press et al. https://arxiv.org/abs/2108.12409

class RelativePositionBias(Module):
    """
    因果 attention 的相对位置 bias. 返回形状为 (h, i, j) 的张量, 直接加到 attention logits 上.

    dist = query 绝对位置 - key 绝对位置. dist > 0 表示 key 在 query 之前.
    两个分量可以单独或同时启用:
    - learned_alibi: 每个 head 一个可学习 slope s_h, bias = -s_h * dist.
    - distance_basis: 共享 num_distance_basis 条长度为 max_dist 的可学习曲线,
      每个 head 用自己的权重线性组合这些曲线, 按 dist 查表得到 bias.

    模型没有其他位置编码, 位置信息只来自这里的 bias 和 causal mask.
    """

    def __init__(
        self,
        heads,
        num_distance_basis = 16,
        max_dist = 1024,
        learned_alibi = True,
        distance_basis = False
    ):
        super().__init__()

        # 至少启用一个分量
        assert learned_alibi or distance_basis

        self.max_dist = max_dist
        self.learned_alibi = learned_alibi
        self.distance_basis = distance_basis

        if learned_alibi:
            # (h,) 每个 head 的线性衰减斜率
            # 原版 ALiBi 使用固定的几何级数斜率; 这里全部初始化为 1, 并作为参数参与训练
            self.to_alibi_slopes = nn.Parameter(torch.ones(heads))

        if distance_basis:
            # (num_distance_basis, max_dist) 各 head 共享的距离基曲线
            self.distance_bank = nn.Parameter(torch.randn(num_distance_basis, max_dist) * 0.02)

            # (h, num_distance_basis) 每个 head 的组合权重, 零初始化, 所以训练开始时该分量为 0
            self.to_head_weights = nn.Parameter(torch.zeros(heads, num_distance_basis))

    def forward(
        self,
        seq_len_q,
        seq_len_k = None
    ):
        # seq_len_q = i, seq_len_k = j. 带 memory 时 j = m + i
        device = next(self.parameters()).device
        seq_len_k = default(seq_len_k, seq_len_q)

        # query 对齐到 key 序列末尾: 第一个 query 的绝对位置是 j - i
        pos_q = torch.arange(seq_len_q, device = device) + (seq_len_k - seq_len_q)
        pos_k = torch.arange(seq_len_k, device = device)

        # (i, j) 相对距离
        rel_dist = einx.subtract('i, j -> i j', pos_q, pos_k)

        # dist < 0 是未来位置, 之后会被 causal mask 屏蔽; 这里先截到 0, 以便作为索引使用
        valid_mask = rel_dist >= 0
        clamped_dist = rel_dist.clamp(min = 0)

        bias = 0.

        if self.distance_basis:
            # 距离基只覆盖 [0, max_dist), 超出范围的位置标记为无效, 索引截到最后一格
            valid_mask = valid_mask & (rel_dist < self.max_dist)
            clamped_dist = clamped_dist.clamp(max = self.max_dist - 1)

            # (h, max_dist) 每个 head 的 bias 曲线, 按距离查表得到 (h, i, j)
            curves = einsum(self.to_head_weights, self.distance_bank, 'h b, b r -> h r')
            bias = bias + curves[:, clamped_dist]

        if self.learned_alibi:
            # (h, 1, 1) * (i, j) -> (h, i, j). slope 为正时, key 越远得分越低
            bias = bias - rearrange(self.to_alibi_slopes, 'h -> h 1 1') * clamped_dist

        # 无效位置的 bias 置 0
        # 同时启用 distance_basis 时, dist >= max_dist 的 key 也在此置 0, ALiBi 惩罚随之消失,
        # 这些远处 key 的 bias 因而高于近处 key
        return bias.masked_fill(~valid_mask, 0.)

# attention

class Attention(Module):
    """
    Pre-norm 多头因果 attention. 输出只是 attention 分支的增量, residual 相加在 RecurrentTransformer.forward 中完成.

    三种调用方式:
    1. forward(tokens): 普通并行 attention, KV 由本层输入投影得到, 与标准 Transformer 相同.
    2. forward(tokens, memory, return_memory = True): 标准 KV cache.
       返回的 memory 是拼接后的输入 KV.
    3. forward(tokens, memory, return_recurr_memory = True): 递归 memory.
       返回的 memory 由 norm(attn_out + residual) 重新投影得到, 即论文所说由本层自身激活计算的 KV.
       forward_naive_recurrent 按 block 循环调用这种方式.
    """

    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        attn_gate = True,
        gate_low_rank = 32,
        use_value_mlp = True,
        value_mlp_expansion = 2.,
        block_size = 1,
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()
        self.scale = inv_sqrt(dim_head)

        # 递归模式下每个 block 的 token 数, 只在 forward_naive_recurrent 中使用
        self.block_size = block_size

        # pre-norm. 递归模式在生成持久 KV 前复用同一个 norm
        self.norm = RMSNorm(dim)
        dim_inner = dim_head * heads

        self.to_queries = LinearNoBias(dim, dim_inner)

        # K 和 V 共用一个投影, 输出沿最后一维对半切分
        self.to_keys_values = LinearNoBias(dim, dim_inner * 2)

        # qk norm: 对每个 head 的 q 和 k 分别做 RMSNorm, 约束 attention logits 的尺度
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        # (b, n, h * d) <-> (b, h, n, d)
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # projection out

        self.to_out = LinearNoBias(dim_inner, dim)

        # attention gating - Jumper et al. AF2
        # 由 norm 后的输入经 LoRA 生成逐通道 gate, sigmoid 后乘到合并 head 后的 attention 输出上

        self.attn_gate = LoRA(dim, dim_inner, low_rank = gate_low_rank) if attn_gate else None

        # value mlp - post project but before aggregation + residual - shown to work well in a iclr 2026 paper for image restoration and apt for this setting
        # 对每个 head 的 value 做残差 MLP (d -> d * value_mlp_expansion -> d), 在加权求和之前加入非线性

        self.value_mlp = MLP(dim_head, int(dim_head * value_mlp_expansion), dim_head) if use_value_mlp else None

        # relative positional bias

        self.rel_pos_bias = RelativePositionBias(heads = heads, **rel_pos_bias_kwargs) if rel_pos_bias else None

    def forward(
        self,
        tokens,
        memory: tuple[Tensor, Tensor] | None = None,
        return_memory = False,
        return_recurr_memory = False
    ):
        """
        tokens: (b, n, dim) 本层输入, 未经 norm
        memory: 可选 (k, v), 形状均为 (b, h, m, d), 表示 tokens 之前 m 个位置的 KV
        return_memory: 额外返回拼接后的输入 KV, 形状均为 (b, h, m + n, d)
        return_recurr_memory: 额外返回拼接后的持久 KV, 形状均为 (b, h, m + n, d)

        返回 attn_out (b, n, dim), 或 (attn_out, (k, v))
        """
        device = tokens.device

        # 未经 norm 的输入, 只在构造持久 KV 时使用
        residual = tokens

        tokens = self.norm(tokens)

        # 三者形状均为 (b, n, h * d)
        q = self.to_queries(tokens)
        k, v = self.to_keys_values(tokens).chunk(2, dim = -1)

        # -> (b, h, n, d)
        q, k, v = map(self.split_heads, (q, k, v))

        # qk rmsnorm

        q = self.q_norm(q)
        k = self.k_norm(k)

        # maybe residual value mlp (nonlinearity)

        if exists(self.value_mlp):
            v = v + self.value_mlp(v)

        # key value memories
        # 之前位置的 KV 在前, 本次 tokens 的输入 KV (论文 Fig.3 中的 kv^temp) 在后, key 数变为 m + n

        if exists(memory):
            mk, mv = memory
            k = cat((mk, k), dim = -2)
            v = cat((mv, v), dim = -2)

        # attention
        # sim: (b, h, i, j), 其中 i = n, j = m + n

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = sim.shape[-2:]

        if exists(self.rel_pos_bias):
            # (h, i, j), 沿 batch 维广播
            sim = sim + self.rel_pos_bias(i, j)

        # 因果 mask: query 对齐到 key 末尾, 第 r 行屏蔽列号大于 r + (j - i) 的 key, 也就是未来位置
        # memory 中的 KV 对所有 query 可见; 本次 tokens 之间是下三角可见
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        # (b, h, i, d)
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge heads
        # (b, n, h * d)

        agg = self.merge_heads(out)

        # maybe attention gate (nonlinearity)
        # gate 的输入是 norm 后的本层输入, 不是 attention 输出

        if exists(self.attn_gate):
            agg = agg * self.attn_gate(tokens).sigmoid()

        # (b, n, dim)
        attn_out = self.to_out(agg)

        assert not (return_memory and return_recurr_memory)

        if not (return_memory or return_recurr_memory):
            return attn_out

        if return_memory:
            # 标准 KV cache: 由输入投影得到的 KV, 已包含之前的 memory
            return attn_out, (k, v)

        # add the output to the residual and then reproject for the 'persistent' key value, key value derived from the output fed back in
        # next_token 是本层 attention 子层的输出 (attn_out + residual) 再经 norm, 不经过 FFN
        # 根据 commit eddf128 的说明, 这一步有意偏离论文: 非线性改由 value mlp 和 attn gate 提供

        next_token = self.norm(attn_out + residual)

        # 与输入 KV 共用投影, k_norm 和 value mlp 的权重, 只是输入不同
        next_k, next_v = self.to_keys_values(next_token).chunk(2, dim = -1)
        next_k, next_v = map(self.split_heads, (next_k, next_v))

        next_k = self.k_norm(next_k)

        if exists(self.value_mlp):
            next_v = next_v + self.value_mlp(next_v)

        # 持久 memory 只累积 next_k 和 next_v. 本次 tokens 的输入 KV 在上面用完后丢弃, 不进入 memory

        if exists(memory):
            mk, mv = memory
            next_k = cat((mk, next_k), dim = -2)
            next_v = cat((mv, next_v), dim = -2)

        return attn_out, (next_k, next_v)

    def forward_naive_recurrent(
        self,
        tokens,
        return_memory = False,
    ):
        """
        沿序列维度按 block 串行执行递归 attention.

        每个 block 中的 query 能看到两类 KV:
        - 之前所有 block 的持久 KV, 由各位置的 norm(attn_out + residual) 投影得到
        - 本 block 内当前及更早位置的输入 KV (kv^temp)

        block_size = 1 时, 位置 t 看到持久 KV_{<t} 和 kv^temp_t, 可见结构与论文逐 token 的递归定义一致.
        block_size > 1 时, 同一 block 内的位置之间只通过输入 KV 交互, 计算结果与 block_size = 1 不同.
        这是一种近似, 不同于论文 Fig.3 中保持计算结果不变的 tiling 算法.

        串行步数为 ceil(n / block_size). memory 没有 detach, 梯度沿整条 memory 链反向传播.

        tokens: (b, n, dim)
        返回 (b, n, dim); return_memory 时额外返回全部位置的持久 KV, 形状均为 (b, h, n, d)
        """
        outs = []

        memory = None

        # 最后一个 block 可能不足 block_size 个 token
        for block in tokens.split(self.block_size, dim = -2):
            out, memory = self(
                block,
                memory = memory,
                return_recurr_memory = True
            )

            outs.append(out)

        outs = cat(outs, dim = -2)

        if not return_memory:
            return outs

        return outs, memory

# feedforward

class GEGLU(Module):
    # 最后一维对半切分: 一半作为值, 另一半经 GELU 后作为 gate
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(
    dim,
    expansion = 4.
):
    # GEGLU 有值和 gate 两路投影, 隐藏维度乘以 2/3 后, 参数量与隐藏维度为 dim * expansion 的普通 MLP 相同 (不计 bias)
    dim_inner = int(dim * expansion * 2 / 3)

    return Sequential(
        nn.RMSNorm(dim),              # pre-norm
        Linear(dim, dim_inner * 2),   # 值和 gate 在一次投影中同时计算
        GEGLU(),
        Linear(dim_inner, dim),       # 输出投影, 在 RecurrentTransformer 中按层深缩放初始化
    )

# classes

class RecurrentTransformer(Module):
    """
    Decoder-only 语言模型: token embedding -> depth 个 (Attention, FeedForward) 层 -> RMSNorm + 输出投影.

    recurrent = False: 每层 attention 一次并行处理全部位置, 相当于带 value mlp, attn gate 和相对位置 bias 的标准 Transformer.
    recurrent = True: 每层 attention 调用 forward_naive_recurrent, 在层内沿时间串行递归.
    两种模式下层与层之间都是逐层计算: 第 l 层处理完整个序列后, 第 l + 1 层才开始.

    参数:
        num_tokens                          词表大小
        dim                                 residual stream 维度
        depth                               层数
        dim_head, heads                     每个 head 的维度和 head 数, attention 内部维度为 dim_head * heads
        ff_expansion                        FeedForward 的扩展倍数
        recurrent                           是否启用层内递归
        block_size                          递归模式下每个 block 的 token 数
        attn_gate, gate_low_rank            是否使用 attention gate, 以及 gate 的 LoRA 秩 (须小于 dim 和 dim_head * heads)
        use_value_mlp                       是否对 value 使用残差 MLP
        rel_pos_bias, rel_pos_bias_kwargs   相对位置 bias 的开关和参数, 见 RelativePositionBias
    """

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion = 4.,
        recurrent = False,
        block_size = 1,
        attn_gate = True,
        gate_low_rank = 32,
        use_value_mlp = True,
        rel_pos_bias = True,
        rel_pos_bias_kwargs: dict = dict()
    ):
        super().__init__()

        # 递归模式下每个 block 至少包含 1 个 token
        assert not recurrent or block_size >= 1

        self.recurrent = recurrent

        # embed

        self.token_emb = nn.Embedding(num_tokens, dim)

        # layers

        layers = ModuleList([])

        for layer_index in range(depth):
            # 从 1 开始的层号, 用于下面的初始化缩放
            layer_depth = layer_index + 1

            attn = Attention(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                block_size = block_size,
                attn_gate = attn_gate,
                gate_low_rank = gate_low_rank,
                use_value_mlp = use_value_mlp,
                rel_pos_bias = rel_pos_bias,
                rel_pos_bias_kwargs = rel_pos_bias_kwargs
            )

            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([attn, ff]))

            # depth residual scaling - Yang et al., following Wortsman init scheme
            # 两个子层输出投影的初始化标准差为 1 / sqrt(2 * fan_in * layer_depth),
            # 层越深, 初始时写入 residual stream 的增量越小

            attn_out, ff_out = attn.to_out, ff[-1]

            nn.init.normal_(attn_out.weight, std = inv_sqrt(2 * attn_out.in_features * layer_depth))
            nn.init.normal_(ff_out.weight, std = inv_sqrt(2 * ff_out.in_features * layer_depth))

        self.layers = layers

        # unembed

        self.to_logits = Sequential(
            nn.RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

    def forward(
        self,
        ids,
        return_loss = False,
        labels = None
    ):
        """
        ids: (b, n) token id
        return_loss: 未给 labels 时按语言模型目标训练, 用 ids[:, :-1] 预测 ids[:, 1:]
        labels: 可选 (b, n) 逐位置目标. 给出时直接与 logits 逐位置对齐计算交叉熵, 不做平移

        返回 logits (b, n, num_tokens), 或标量 loss
        """

        # 显式给出 labels 时总是返回 loss
        if exists(labels):
            return_loss = True

        # 语言模型目标: 输入和目标错开一位
        if return_loss and not exists(labels):
            ids, labels = ids[:, :-1], ids[:, 1:]

        # embed
        # (b, n) -> (b, n, dim), 不加绝对位置编码

        tokens = self.token_emb(ids)

        # layers

        for attn, ff in self.layers:

            # 递归模式按 block 串行; 否则一次并行处理全部位置
            forward_attn = attn.forward_naive_recurrent if self.recurrent else attn

            # 两个子层都是 pre-norm 残差结构, norm 在子层内部完成
            tokens = forward_attn(tokens) + tokens
            tokens = ff(tokens) + tokens

        # unembed

        logits = self.to_logits(tokens)

        if not return_loss:
            return logits

        # cross_entropy 要求类别维在第 1 维: (b, n, l) -> (b, l, n)
        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels
        )

        return loss
