# parity 长度外推实验: 在长度 16 的序列上训练, 在长度 16 到 128 的序列上测试
# 任务要求模型逐位置累积 XOR 状态, 用来检验层内递归能否学到可外推的串行计算

import torch
from torch.optim import AdamW

from recurrent_transformer import RecurrentTransformer

# seed

torch.manual_seed(42)

# model
# 词表只有 {0, 1}; 启用递归模式, block_size 取默认值 1, 即逐 token 递归
# gate_low_rank 必须小于 dim = 32 和 dim_head * heads = 32, 所以从默认的 32 降到 16

model = RecurrentTransformer(
    num_tokens = 2,
    dim = 32,
    depth = 2,
    dim_head = 16,
    heads = 2,
    recurrent = True,
    gate_low_rank = 16
)

optimizer = AdamW(model.parameters(), lr = 3e-3, weight_decay = 1e-4)

# parity task
# 输入是随机 0/1 序列; 位置 t 的目标是前 t 个 token (含第 t 个) 之和的奇偶性

def generate_batch(batch_size, seq_len):
    tokens = torch.randint(0, 2, (batch_size, seq_len))
    return tokens, tokens.cumsum(dim = -1) % 2

# train on short sequences

train_seq_len, batch_size, num_steps = 16, 64, 1000

print(f'training parity on seq len {train_seq_len}...\n')

for step in range(1, num_steps + 1):
    tokens, labels = generate_batch(batch_size, train_seq_len)

    # 显式传入 labels: 逐位置对齐计算交叉熵, 不做自回归平移
    loss = model(tokens, labels = labels)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

    optimizer.step()
    optimizer.zero_grad()

    if step % 250 == 0:
        print(f'step {step:4d} | loss: {loss.item():.4f}')

# length extrapolation
# 在训练时未见过的更长序列上评估. 模型没有绝对位置编码, 相对位置 bias 对任意长度都有定义

print('\nevaluating length extrapolation...\n')

model.eval()

with torch.no_grad():
    for seq_len in (16, 32, 64, 128):
        tokens, labels = generate_batch(100, seq_len)

        # 一次完整前向, 每个位置取 argmax 作为预测
        preds = model(tokens).argmax(dim = -1)

        # token acc: 逐位置准确率; seq acc: 整条序列全部位置都正确的比例
        token_acc = (preds == labels).float().mean().item()
        seq_acc = (preds == labels).all(dim = -1).float().mean().item()

        print(f'seq len {seq_len:3d} | token acc {token_acc:6.2%} | seq acc {seq_acc:6.2%}')
