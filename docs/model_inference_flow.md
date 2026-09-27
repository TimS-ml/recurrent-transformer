# 模型推理流程

仓库的推理入口是 `RecurrentTransformer.forward`. 它接收整条 Token 序列, 一次前向返回所有位置的 logits. 模型层面没有逐 Token 生成接口, 也不管理 KV cache. 这份文档说明一次前向中数据如何流动, 递归模式下每个 block 看到哪些 KV, 以及如何用 `Attention` 已有的 `memory` 参数实现自回归解码. 组件内部的公式见 [core_components.md](./core_components.md).

## 调用入口

`forward(ids, return_loss = False, labels = None)` 根据参数决定输入, 目标和返回值:

| 调用方式 | 实际输入 | 目标 | 返回 |
|---|---|---|---|
| `model(ids)` | `ids` | 无 | logits, 形状 `(b, n, V)` |
| `model(ids, return_loss = True)` | `ids[:, :-1]` | `ids[:, 1:]` | 标量 loss |
| `model(ids, labels = y)` | `ids` | `y`, 与输入逐位置对齐 | 标量 loss |

给出 `labels` 时, `return_loss` 自动置为 `True`, 且不做平移. parity 脚本使用第三种方式. 它的目标是包含当前位置的前缀奇偶性, 不是下一个 Token.

## 并行模式的前向 (recurrent = False)

下表以测试用例的配置为例: `b = 2`, `n = 16`, `dim = 128`, `heads = 2`, `dim_head = 64`, `num_tokens = 256`. 此时 $\text{dim\_inner} = \lfloor 128 \cdot 4 \cdot 2/3 \rfloor = 341$.

| 步骤 | 代码 | 输出形状 |
|---|---|---|
| Token embedding | `token_emb(ids)` | `(2, 16, 128)` |
| Attention: pre-norm 与投影 | `norm`, `to_queries`, `to_keys_values`, `split_heads` | q, k, v 均为 `(2, 2, 16, 64)` |
| Attention: 打分 | `einsum(q, k)`, 加 bias, causal mask | `(2, 2, 16, 16)` |
| Attention: 聚合 | `softmax`, `einsum(attn, v)`, `merge_heads` | `(2, 16, 128)` |
| Attention: gate 与输出投影 | `sigmoid(attn_gate(tokens))`, `to_out` | `(2, 16, 128)` |
| 加 residual | `tokens = attn_out + tokens` | `(2, 16, 128)` |
| FeedForward | `RMSNorm`, `Linear`, `GEGLU`, `Linear` | `(2, 16, 682)`, `(2, 16, 341)`, `(2, 16, 128)` |
| 加 residual | `tokens = ff(tokens) + tokens` | `(2, 16, 128)` |
| 重复上述层 | 共 `depth` 层 | `(2, 16, 128)` |
| 输出头 | `to_logits` | `(2, 16, 256)` |

这个模式与标准 Transformer 的 prefill 相同. 每层 attention 只调用一次, 所有位置同时计算.

## 递归模式的前向 (recurrent = True)

递归模式只改变每层 attention 的执行方式. 每层调用 `attn.forward_naive_recurrent(tokens)`, 其核心循环是:

```python
memory = None
for block in tokens.split(block_size, dim = -2):    # block: (b, n_k, dim), n_k <= block_size
    out, memory = attn(block, memory = memory, return_recurr_memory = True)
    outs.append(out)
```

设当前 block 之前已有 $m$ 个位置, block 长度为 $n_k$. 一次循环按以下顺序计算:

1. 由 block 计算 q 和临时 KV, 形状均为 `(b, h, n_k, d)`.
2. 把 memory 中的持久 KV 拼在临时 KV 之前, key 的形状变为 `(b, h, m + n_k, d)`.
3. 打分矩阵为 `(b, h, n_k, m + n_k)`. 相对位置 bias 以 `rel_pos_bias(n_k, m + n_k)` 计算, query 的绝对位置从 $m$ 开始.
4. causal mask 为 `ones(n_k, m + n_k).triu(m + 1)`. memory 部分全部可见, block 内部下三角可见.
5. 得到 `attn_out`, 形状为 `(b, n_k, dim)`.
6. 由 `norm(attn_out + block)` 投影出本 block 的持久 KV, 拼到 memory 之后. memory 变为 `(b, h, m + n_k, d)`, 进入下一次循环.

所有 block 的输出沿序列维拼接为 `(b, n, dim)`, 加 residual 后交给 FeedForward. FeedForward 仍然对所有位置并行计算. 整个模型的 attention 调用次数为 $\text{depth} \times \lceil n / B \rceil$, 其中 $B$ 为 `block_size`.

以 $m = 2$, $n_k = 2$ 为例, 第 4 步的 mask 为 `ones(2, 4).triu(3)`. 第一行只屏蔽第 4 列, 第二行不屏蔽. 即位置 3 看到 memory 中的位置 1, 2 和自己的临时 KV. 位置 4 看到 memory 中的位置 1, 2 和位置 3, 4 的临时 KV.

### block_size 对可见性的影响

下表列出长度为 4 的序列中, 每个 query 能看到的 KV. 位置从 1 开始编号. 持久 KV 由 $\mathrm{norm}(a_s + x_s)$ 投影得到, 临时 KV 由本层输入投影得到.

| query 位置 | B = 1: 持久 KV | B = 1: 临时 KV | B = 2: 持久 KV | B = 2: 临时 KV |
|---|---|---|---|---|
| 1 | 无 | 1 | 无 | 1 |
| 2 | 1 | 2 | 无 | 1, 2 |
| 3 | 1, 2 | 3 | 1, 2 | 3 |
| 4 | 1, 2, 3 | 4 | 1, 2 | 3, 4 |

$B = 1$ 时, 每个位置只看到自己的临时 KV, 更早位置都通过持久 KV 可见. 这与 [architecture_overview.md](./architecture_overview.md) 中逐 Token 递归的定义一致. $B = 2$ 时, 位置 2 看到的是位置 1 的临时 KV, 位置 4 同理. 本地验证显示, 同一组权重下两种设置的输出不同, 只有第一个位置完全一致.

## 损失计算

`to_logits` 输出 `(b, n, V)` 后, 代码把它重排为 `(b, V, n)`, 以满足 `F.cross_entropy` 要求类别维在第 1 维的约定. `cross_entropy` 使用默认的 `reduction = 'mean'`, 即对 $b \times n$ 个位置的交叉熵取平均. 递归模式下 memory 没有 `detach`, 反向传播沿整条 memory 链进行.

## parity 评估中的推理

`train_parity_extrapolation.py` 训练结束后调用 `model.eval()`, 在 `torch.no_grad()` 下对长度 16, 32, 64, 128 的序列各生成 100 条样本. 每条样本只做一次完整前向 `model(tokens)`, 然后对每个位置取 `argmax` 作为预测. 脚本报告两个指标. token acc 是全部位置中预测正确的比例. seq acc 是所有位置都预测正确的序列所占比例.

长序列与训练长度 16 走同一条代码路径. 模型没有绝对位置编码, 默认的 learned ALiBi 对任意距离都有定义, 所以前向不受长度限制. 递归模式默认 `block_size = 1`, 长度为 128 的序列在每层需要 128 次串行的 attention 调用.

一次本地运行记录: 环境为 torch 2.14.0 (CPU), einx 0.4.3, einops 0.8.2, x-mlps-pytorch 0.6.2. 第 1000 步的训练 loss 显示为 0.0000. 四个评估长度上的 token acc 和 seq acc 均为 100.00%. 整个脚本耗时约 1 分 35 秒. 脚本固定了随机种子 42, 不同硬件或库版本下结果可能不同.

## 自回归解码

模型层面没有生成函数. 最简单的做法是每生成一个 Token 就对整个前缀重新调用 `model(seq)`, 取最后一个位置的 logits. 由于 attention 是因果的, 并且 block 总是从位置 0 开始每 $B$ 个位置切一次, 前缀的 logits 与完整序列前向中对应位置的 logits 相同. 这种做法对所有模式都正确. 代价是每步都重算整个前缀.

`Attention` 的 `memory` 参数可以避免重算. 本节的代码不在仓库中. 这些代码在本地用随机权重验证过: 逐步得到的 logits 与完整前向一致 (`atol = 1e-5`), 贪心生成的结果与上面的重算做法逐 Token 相同.

### 标准模式的 KV cache

`recurrent = False` 时, 每层调用 `attn(x, memory = mem, return_memory = True)`. 返回的 memory 是拼接后的输入 KV, 用法与普通 Transformer 的 KV cache 相同.

### 递归模式的解码

递归模式把 `return_memory` 换成 `return_recurr_memory`. 每层保存的是持久 KV. 下面的辅助函数把一段 Token 送过所有层, 并返回每层更新后的 memory:

```python
import torch
from torch import cat

@torch.no_grad()
def forward_with_memory(model, ids, memories):
    # ids: (b, n) 本次处理的 token; memories: 每层一个持久 (k, v), 或 None
    tokens = model.token_emb(ids)
    new_memories = []

    for (attn, ff), memory in zip(model.layers, memories):
        out, memory = attn(tokens, memory = memory, return_recurr_memory = True)
        tokens = out + tokens
        tokens = ff(tokens) + tokens
        new_memories.append(memory)

    return model.to_logits(tokens), new_memories
```

`block_size = 1` 时, 每次传入一个 Token, 并把返回的 memory 直接用于下一步. `block_size > 1` 时不能这样做. 逐 Token 传入会让每个 Token 自成一个 block, 与训练时的可见性不一致. 本地验证中, $B = 2$ 和 $B = 4$ 的逐 Token 结果都与完整前向不同. 正确做法是: 当前 block 未满时, 每步重算该 block 中截至当前位置的全部 Token; block 满了以后再提交它的持久 KV. 下面的生成函数同时覆盖 $B = 1$ 和 $B > 1$:

```python
@torch.no_grad()
def generate(model, prompt, num_new_tokens):
    assert model.recurrent
    block_size = model.layers[0][0].block_size

    seq = prompt                                    # (b, n0)
    committed = [None] * len(model.layers)          # 已完成 block 的持久 KV

    for t in range(prompt.shape[1] + num_new_tokens - 1):
        # 重算当前 block 中截至位置 t 的全部 token
        start = (t // block_size) * block_size
        logits, memories = forward_with_memory(model, seq[:, start:t + 1], committed)

        # block 已满时提交它的持久 KV, 之后不再重算
        if (t + 1) % block_size == 0:
            committed = memories

        # prompt 处理完后, 用位置 t 的 logits 生成位置 t + 1 的 token
        if t + 1 == seq.shape[1]:
            next_id = logits[:, -1].argmax(dim = -1, keepdim = True)
            seq = cat((seq, next_id), dim = -1)

    return seq
```

$B = 1$ 时, 每步只处理 1 个 Token, 每层增加一对持久 KV. memory 的大小与标准 KV cache 相同, 这与论文摘要中保持标准解码代价的说法一致. 额外开销是每个 Token 在每层多一次 KV 投影和一次 value MLP. $B > 1$ 时, 每步最多重算 $B$ 个 Token, 单步计算量最多约为 $B = 1$ 时的 $B$ 倍. 示例中的 prompt 也是逐步处理的. 前缀较长时, 可以按完整 block 分段处理 prompt, 以减少调用次数.

## 训练与 prefill 的代价

`forward_naive_recurrent` 是朴素实现, 与并行模式相比有三类额外代价.

第一类是串行步数. 并行模式每层调用一次 attention. 递归模式每层调用 $\lceil n / B \rceil$ 次, 每次只处理 $B$ 个 query, 矩阵乘法的规模小, 难以充分利用 GPU. 论文摘要把朴素做法描述为 bandwidth-bound, 有效算术强度接近 1.

第二类是 memory 拼接. 每个 block 都用 `cat` 生成新的 memory 张量, 复制已有的全部持久 KV. 第 $k$ 个 block 复制的元素数与 $kB \cdot h \cdot d$ 成正比. 对 $k = 1, \dots, n/B$ 求和, 总复制量为

$$
\sum_{k=1}^{n/B} kB \cdot h d = B \cdot h d \cdot \frac{(n/B)(n/B + 1)}{2} \approx \frac{n^2}{2B} \cdot h d
$$

K 和 V 各有一份, 用于 attention 的拼接和持久 memory 的拼接又各有一次, 所以实际总量是上式的常数倍, 量级为 $O(n^2 h d / B)$.

第三类是训练时保存的激活. 每个 block 拼接后的 K 和 V 会作为矩阵乘法的输入保存下来, 供反向传播使用. 这部分总量同样是 $O(n^2 h d / B)$. 并行模式保存的主要是 $O(n^2 h)$ 的 attention 矩阵. 下表是本地测量结果. 测量对象是单个 `Attention` 层 (`dim = 64`, `dim_head = 32`, `heads = 2`, `gate_low_rank = 16`, batch 为 1). 用 `torch.autograd.graph.saved_tensors_hooks` 统计 autograd 保存的张量元素总数, 同一张量只计一次.

| 序列长度 $n$ | 并行模式 | 递归 B = 1 | 递归 B = 4 | 递归 B = 16 |
|---|---|---|---|---|
| 64 | 126,400 | 411,488 | 215,552 | 169,088 |
| 128 | 287,488 | 1,353,280 | 568,192 | 376,960 |
| 256 | 781,696 | 4,895,744 | 1,752,704 | 977,024 |

$B = 1$ 时, 长度每翻一倍, 保存量分别增长 3.3 倍和 3.6 倍, 接近平方增长的 4 倍. 在 $n = 256$ 时, $B = 1$ 的保存量约为并行模式的 6.3 倍. 增大 $B$ 可以降低这部分开销, 代价是 block 内部不再逐 Token 递归. 论文的 tiling 算法针对的是保持计算结果不变的 prefill 访存问题. 本仓库没有实现这个算法.
