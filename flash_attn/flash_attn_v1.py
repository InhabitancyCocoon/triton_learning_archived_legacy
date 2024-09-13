import torch

torch.manual_seed(456)

N, d = 1024, 512

Q = torch.rand((N, d))
K = torch.rand((N, d))
V = torch.rand((N, d))

# 执行标准的pytorch softmax和attention计算
expected_softmax = torch.softmax(Q @ K.T, dim=1)
expected_attention = expected_softmax @ V


# 算法流程第一行: 分块（tiling）尺寸，以SRAM的大小计算得到
Br = 8
Bc = d

# 算法流程第二行，首先在HBM中创建用于存储输出结果的O，全部初始化为0
O = torch.zeros((N, d))
# 算法流程第二行，用来存储softmax的分母值，在HBM中创建
l = torch.zeros((N, 1))
# 算法流程第二行，用来存储每个block的最大值，在HBM中创建
m = torch.full((N, 1), -torch.inf)

# 算法流程第五行，执行外循环
for block_start_Bc in range(0, N, Bc):
    block_end_Bc = block_start_Bc + Bc
    # 算法流程第六行，从HBM中load Kj, Vj的一个block到SRAM
    Kj = K[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    Vj = V[block_start_Bc:block_end_Bc, :]  # shape Bc x d
    # 算法流程第七行，执行内循环
    for block_start_Br in range(0, N, Br):
        block_end_Br = block_start_Br + Br
    # 算法流程第八行，从HBM中分别load以下几项到SRAM中
        mi = m[block_start_Br:block_end_Br, :]  # shape Br x 1
        li = l[block_start_Br:block_end_Br, :]  # shape Br x 1
        Oi = O[block_start_Br:block_end_Br, :]  # shape Br x d
        Qi = Q[block_start_Br:block_end_Br, :]  # shape Br x d

        # 算法流程第九行
        Sij = Qi @ Kj.T  # shape Br x Bc

        # 算法流程第十行，计算当前block每行的最大值
        mij_hat = torch.max(Sij, dim=1, keepdim=True).values

        # 算法流程第十行，计算softmax的分母
        Pij_hat = torch.exp(Sij - mij_hat)
        lij_hat = torch.sum(Pij_hat, dim=1, keepdim=True)

        # 算法流程第十一行，找到当前block的每行最大值以及之前的最大值
        mi_new = torch.maximum(mi, mij_hat)

        # 算法流程第十一行，计算softmax的分母，但是带了online计算的校正，此公式与前面说的online safe softmax不一致，但是是同样的数学表达式，只是从针对标量的逐个计算扩展到了针对逐个向量的计算
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

        # 算法流程第十二行，计算每个block的输出值
        Oi_new = ((li * torch.exp(mi - mi_new) * Oi) + (torch.exp(mij_hat - mi_new) * Pij_hat) @ Vj) / li_new
        # 算法流程第十二行，将Oi_new再写回到HBM
        O[block_start_Br:block_end_Br, :] = Oi_new

        # 算法流程第十三行, 将mi_new, li_new写入HBM
        m[block_start_Br:block_end_Br, :] = mi_new  # row max
        l[block_start_Br:block_end_Br, :] = li_new  # softmax denominator

assert torch.allclose(O, expected_attention, atol=1e-4)