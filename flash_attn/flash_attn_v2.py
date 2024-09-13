import torch


torch.manual_seed(456)

N, d = 1024, 512

Q = torch.rand((N, d))
K = torch.rand((N, d))
V = torch.rand((N, d))

expected_softmax = torch.softmax(Q @ K.T, dim=1)
expected_attention = expected_softmax @ V

# 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 8
Bc = d

O = torch.zeros((N, d))

# 算法流程第3步，执行外循环
for block_start_Br in range(0, N, Br):
    block_end_Br = block_start_Br + Br

    # 算法流程第4步，从HBM中load Qi 的一个block到SRAM
    Qi = Q[block_start_Br:block_end_Br, :]

    # 算法流程第5步，初始化每个block的值
    Oi = torch.zeros((Br, d))  # shape Br x d
    li = torch.zeros((Br, 1))  # shape Br x 1
    mi = torch.full((Br, 1), -torch.inf)  # shape Br x 1

    # 算法流程第6步，执行内循环
    for block_start_Bc in range(0, N, Bc):
        block_end_Bc = block_start_Bc + Bc

        # 算法流程第7步，load Kj, Vj到SRAM
        Kj = K[block_start_Bc:block_end_Bc, :]
        Vj = V[block_start_Bc:block_end_Bc, :]

        # 算法流程第8步
        Sij = Qi @ Kj.T

        # 算法流程第9步
        mij_hat = torch.max(Sij, dim=1, keepdim=True).values
        mi_new = torch.maximum(mi, mij_hat)
        Pij_hat = torch.exp(Sij - mi_new)
        lij_hat = torch.sum(Pij_hat, dim=1, keepdim=True)
        li_new = torch.exp(mi - mi_new) * li + lij_hat

        # 算法流程第10步
        Oi_new = Oi * torch.exp(mi - mi_new) + Pij_hat @ Vj

        mi = mi_new
        li = li_new
        Oi = Oi_new

    # 第12步
    Oi = Oi / li

    # 第14步
    O[block_start_Br:block_end_Br, :] = Oi

    
assert torch.allclose(O, expected_attention, atol=1e-4)