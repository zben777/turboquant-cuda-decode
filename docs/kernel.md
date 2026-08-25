12. 一个 block 和一个 tile 在这里刚好完全对应

这也是这份代码设计得非常舒服的地方。

固定：

KV cache block size = 16 tokens
TILE                = 16 tokens

所以：

1 attention tile
=
1 physical KV block



加载 query → 初始化 p_s → 循环 8 个 tile：
    ├─ 读 block table + 元数据
    ├─ 反量化 K/V 到共享内存
    ├─ warp0 计算 QK → scratch.qk
    ├─ 在线 softmax → p_s, alpha
    ├─ 缩放旧 out0/out1
    ├─ P@V → out0/out1 累加
    └─ 同步
→ 写回 mid_o（输出 + LSE）











加载 query
├─ 参与线程：全部 128 线程
├─ 数据来源：全局内存 q_rot（fp32）
├─ 操作：将 4 个 query head 的 128 维数据转为 half，写入共享内存 q_s[16][128]
│         （仅前 4 行有效，后 12 行未初始化但后续忽略）
└─ 说明：GQA=4，当前 KV head 对应 qh=0,1,2,3

初始化 p_s
├─ 参与线程：全部 128 线程
├─ 操作：将共享内存 p_s[16][16] 所有元素置为 0
└─ 目的：后续 softmax 只更新前 4 行，其余行保持 0，避免污染

__syncthreads()  ◄── 确保 query 和 p_s 就绪

循环 8 个 tile（每个 tile 处理 16 tokens）
│
├─ 读 block table + 元数据
│   ├─ 参与线程：仅 warp0
│   ├─ lane0 从 block_table[b][logical_block] 读物理 block id
│   ├─ __shfl_sync 广播 block id 给 warp0 全部 lane
│   ├─ lane<16 计算 data_base[lane]（token 在 cache 中的数据偏移）
│   ├─ lane<16 读取 k_norm/lane, v_scale/lane, v_zero/lane（元数据区，half→float）
│   ├─ 写入共享内存 data_base[], k_norm[], v_scale[], v_zero[]
│   └─ __syncthreads()  ◄── 元数据对所有线程可见

├─ 反量化 K/V 到共享内存
│   ├─ 参与线程：全部 128 线程
│   ├─ 数据来源：全局内存 cache（uint8，4-bit packed）
│   ├─ K：读 byte → 两个 4bit 索引 → 查 centroids 码本（通过 __shfl_sync 广播）→ × k_norm → half
│   ├─ V：读 byte → 两个 4bit 整数 → × v_scale + v_zero → half
│   ├─ 结果写入共享内存 k_s[16][128] 和 v_s[16][128]
│   └─ __syncthreads()  ◄── K/V 就绪

├─ warp0 计算 QK → scratch.qk
│   ├─ 参与线程：仅 warp0（32 线程）
│   ├─ 使用 WMMA（16×16×16）分 8 次累加 Q @ K^T
│   ├─ Q 来自 q_s（16×128，行主序），K 来自 k_s（16×128，用 col_major 加载实现转置）
│   ├─ 结果：16×16 logits 矩阵（行=query head，列=token）
│   ├─ 存入共享内存 scratch.qk[16][16]
│   └─ __syncthreads()  ◄── logits 对所有 warp 可见

├─ 在线 softmax → p_s, alpha
│   ├─ 参与线程：每个 warp 独立处理一个 query head（4 个 warp 并行）
│   ├─ lane<16 读取 scratch.qk[warp][lane] 作为 logit
│   ├─ log2 域转换：score = logit * ATTN_SCALE * RCP_LN2
│   ├─ 求当前 tile 最大值 tile_m（warp 内归约）
│   ├─ 更新全局最大值 new_m = max(running_m, tile_m)
│   ├─ 计算旧输出缩放因子 alpha = exp2(running_m - new_m)（首个 tile 时 alpha=0）
│   ├─ 计算当前 tile 概率 p = exp2(score - new_m)（仅 lane<16）
│   ├─ 求 tile 概率和 tile_l（warp 内归约）
│   ├─ 更新全局统计量：running_l = running_l * alpha + tile_l, running_m = new_m
│   ├─ 将 p 写入 p_s[warp][lane]（half），tile_alpha[warp] = alpha
│   └─ __syncthreads()  ◄── p_s 和 alpha 就绪

├─ 缩放旧 out0/out1
│   ├─ 参与线程：每个 warp 独立操作自己的 WMMA 累加器（寄存器）
│   ├─ 遍历 out0/out1 中每个元素，确定其对应矩阵行 row
│   ├─ 若 row < 4，乘以 tile_alpha[row]；否则乘以 0（无效行）
│   ├─ 目的：在线 softmax 校正，避免保存旧概率
│   └─ 无同步（仅操作私有寄存器）

├─ P@V → out0/out1 累加
│   ├─ 参与线程：每个 warp 独立执行，负责输出维度 col = warp*32 ~ warp*32+31
│   ├─ 加载 p_s[16][16] 作为矩阵 A（前4行有效，其余为0）
│   ├─ 第一次加载 V 列块 col~col+15 → WMMA 累加 out0
│   ├─ 第二次加载 V 列块 col+16~col+31 → WMMA 累加 out1
│   ├─ 累加器 out0/out1 保存该 warp 负责的 32 个输出维度在 4 个 query head 上的部分和
│   └─ 若未定义 TQ4_FUSED_TILE_BARRIER，则 __syncthreads()  ◄── 防止下一轮写 V 冲突

└─ （循环回到“读 block table”，处理下一个 16 tokens，共 8 次）

写回 mid_o（输出 + LSE）
├─ 每个 warp 的 lane0：
│   ├─ split_inv_l[warp] = 1.0f / running_l
│   └─ split_lse[warp] = running_m * LN2 + logf(running_l)   // 自然对数域 LSE
├─ 将 out0/out1 通过 WMMA store 写入共享内存 scratch.output[16][128]
│   （每个 warp 写入自己负责的 32 列，形成完整 16×128 矩阵）
├─ __syncthreads()  ◄── 输出矩阵就绪
├─ 循环 q=0..3（4 个 query head）：
│   ├─ 所有 128 线程协作：mid_o[out + tid] = scratch.output[q][tid] * split_inv_l[q]
│   └─ tid==0 线程：mid_o[out + 128] = split_lse[q]
└─ 完成：mid_o[0, qh, 0, 0:128] = 归一化注意力输出，mid_o[0, qh, 0, 128] = LSE