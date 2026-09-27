import torch
import pytest
param = pytest.mark.parametrize

from recurrent_transformer.recurrent_transformer import Attention, RecurrentTransformer

# 冒烟测试: 覆盖 并行/递归 x 3 种 block_size x 3 种相对位置 bias 配置, 共 18 组
# 只检查输出形状, loss 为标量, 以及反向传播能运行; 不检查数值正确性

@param('recurrent', (False, True))
@param('block_size', (1, 2, 4))    # 并行模式下 block_size 不生效
@param('rel_pos_bias_kwargs', (
    dict(),                                             # 默认: 只用 learned alibi
    dict(learned_alibi = False, distance_basis = True), # 只用距离基曲线
    dict(learned_alibi = True, distance_basis = True)   # 两者叠加
))
def test_recurrent_transformer(recurrent, block_size, rel_pos_bias_kwargs):

    model = RecurrentTransformer(
        num_tokens = 256,
        dim = 128,
        depth = 2,
        dim_head = 64,
        heads = 2,
        recurrent = recurrent,
        block_size = block_size,
        rel_pos_bias_kwargs = rel_pos_bias_kwargs
    )

    ids = torch.randint(0, 256, (2, 16))

    # 推理: (b, n) -> (b, n, num_tokens)
    logits = model(ids)

    assert logits.shape == (2, 16, 256)

    # 训练: 未给 labels, 内部把 ids 错开一位作为目标
    loss = model(ids, return_loss = True)

    assert loss.numel() == 1

    loss.backward()
