# TurboQuant CUDA Decode 面试问答

这份问答面向秋招 CUDA、GPU Kernel、大模型推理和高性能计算岗位，对应本项目
可以写在简历上的描述：

> 针对 Qwen3-4B 形状的 GQA Decode，设计并优化 TurboQuant 4-bit KV Cache
> CUDA Kernel；融合 K/V 反量化、Tensor Core QK/PV 和 online softmax，使用
> 向量化 packed-cache load、WMMA fragment 直接写回及原生
> `mma.sync.m16n8k16` 改善 GQA-4 利用率，并借鉴 FlashInfer 的
> register-resident attention state 消除 QK shared-memory 往返。RTX 4090
> 上 Stage1 从 CUDA V1 的 2.074 ms 降至 V9 的 0.485 ms，约 4.28x；V9
> 相比修复后的 Triton baseline 快 2.22x。另实现 CUDA Stage2 和 Full Decode，
> 完整链路相比 Triton 快 1.66x。

面试时应根据自己的实际贡献调整上面这段描述，不要把仓库中保留的上游 vLLM
代码、尚未完成的 production 集成或未采集的 profiler 指标说成自己的成果。

## 一分钟项目介绍

这个项目研究的是大模型自回归 Decode 阶段的量化 KV Cache Attention。Decode
每生成一个 token，都要读取此前所有 token 的 K/V，因此长上下文下通常受
KV Cache 显存流量限制。TurboQuant 将 K 和 V 压缩到 4 bit：K 使用旋转、
Lloyd-Max centroid index 和 norm，V 使用 uniform index、scale 和 zero。

我的工作不是先把完整 K/V 解压到显存再调用 Attention，而是在同一个 Stage1
Kernel 中读取 packed cache、即时反量化，并直接完成 QK、online softmax 和
PV。固定 workload 是 batch 64、context 4096、32 个 Q head、8 个 KV head、
head dimension 128，并将序列分成 32 个 split。

优化从线程映射和 K/V 复用开始，逐步加入单遍 tiled online softmax、Tensor
Core、固定 workload 特化、WMMA fragment 直接写回、`uint32`/`half2`
向量化和 barrier 融合。V8 通过原生 `m16n8k16` 将 GQA-4 的有效 Tensor Core
槽位从 `4/16` 提高到 `4/8`；V9 再让 QK score 留在寄存器中更新 softmax
state，将静态 barrier 从 34 降到 26。

在 RTX 4090 上，CUDA V1 是 2.074 ms，V9 是 0.485 ms，约 4.28x；V9
相比修复后的 Triton V2-fixed 快 2.22x。项目还通过未经修改的 vLLM SoA
Store 生成真实压缩 cache，验证 CUDA 与 Triton Decode 读取的是同一字节布局。

## 核心问题

### 1. 这个项目解决什么问题？

它解决的是 Decode Attention 读取 KV Cache 时显存流量大、低比特 cache 又需要
额外解码的问题。项目把 4-bit K/V 解码与 Attention Stage1 融合，避免生成一份
完整的 FP16 K/V 临时张量，从而同时减少存储容量、全局内存读取和 Kernel 边界。

它目前是固定 workload 的 CUDA 研究框架，不是可以直接替换 vLLM backend 的
通用实现。

面试时可以分四层回答：

1. **容量问题**：FP16 K/V 每个 `(token,kv_head)` 是 512 B，`4bit_nc` 是
   134 B，逻辑压缩比为 3.82x；
2. **带宽问题**：Decode 每生成一个 token 都要扫描历史 K/V，context 越长，
   KV traffic 越大；
3. **融合问题**：若先恢复完整 FP16 K/V，会重新引入 global-memory 中间结果；
   本项目在 tile 内直接完成 lookup、反量化、QK、softmax 和 PV；
4. **项目结果**：V7 Full Decode 为 0.664842 ms，相比同语义 Triton
   V2-fixed 的 1.104148 ms 提速 1.661x。

这个项目同时解决“cache 存得小”和“压缩格式能否被高效消费”两个问题。只做
量化格式而没有融合 Decode Kernel，容量收益不一定能转化为 latency 收益。

### 2. 为什么 Decode 阶段特别关注 KV Cache？

Prefill 可以通过较大的矩阵乘法获得较高计算强度，而 Decode 每一步通常只有
一个新 query，却必须读取整个历史 K/V。随着 context length 增长，读取 K/V
的字节数线性增长，计算相对较少，因此容易成为 memory-bound 路径。低比特
KV Cache 的直接收益是减少每个历史 token 必须搬运的数据。

更具体地说，单步 Decode 的 Q 数量通常是每个 sequence 一个，而历史 K/V 数量
是 $L$。QK 和 PV 的 useful FLOPs 都随 $L$ 线性增长，但同一个历史 K/V 在该步
通常只消费一次，缺少 Prefill 中多 Query 复用，因此算术强度较低。

当前 workload 只按 compressed cache 计算的 useful AI 约为：

$$
AI_{cache}=\frac{4GD}{D+6}
=\frac{4\times4\times128}{134}=15.28\ \text{FLOP/B}.
$$

计入 Query 重读和 `mid_o` 写出后约为 12.31 useful FLOP/B，远低于 RTX 4090
dense FP16 Tensor ridge point。因此静态 Roofline 判断它偏 memory side，但是否
由 DRAM 饱和、L2、lookup dependency 或同步限制，还要结合当前版本 NCU。

### 3. 为什么不能直接先反量化，再调用普通 FlashAttention？

这样会先读 INT4 cache，再把完整 FP16 K/V 写回显存，随后 Attention 又重新
读取 FP16 K/V。额外的中间张量、写回流量和 Kernel launch 会抵消压缩收益。
本项目选择在 tile 内解码到 Shared Memory，然后立即用于 Tensor Core QK/PV，
解码结果不落到全局内存。

可以直接做一笔流量账。每个 `(token,kv_head)`：

- compressed input 是 134 B；
- 若物化完整 FP16 K/V，需要额外写 512 B；
- 后续 attention 还要再次读取这 512 B；
- 还需要保存临时 tensor、增加至少一个 Kernel 边界。

融合路径读取 134 B 后只在 Shared Memory/寄存器中形成 tile 级 FP16 数据，使用
完即覆盖。这里不是说 FlashAttention 算法不适用，而是普通 FP16 FlashAttention
接口不直接理解 TurboQuant 的 nibble、centroid 和 metadata；正确方向是把量化
Decode 融入 attention tile load，而不是先全量反量化。

### 4. 项目的固定 workload 是什么？

```text
GPU                 RTX 4090, sm_89
Batch               64
Context length      4096
Q heads / KV heads  32 / 8
GQA group size      4
Head dimension      128
KV block size       16
KV splits           32
Tokens per split    128
K / V               4-bit / 4-bit
```

固定形状允许展开八个 16-token tile，并简化 split boundary、对齐和 page
mapping；代价是当前性能结论不能直接推广到任意模型和序列长度。

## TurboQuant 量化原理

### 5. TurboQuant 与普通 per-tensor INT4 有什么区别？

TurboQuant 的 K 路径不是简单线性量化。它先对 K 向量归一化并做正交旋转，
让各坐标分布更稳定，然后使用预先计算的 Lloyd-Max centroid codebook 做
非均匀标量量化。Cache 保存每个坐标的 4-bit centroid index，以及每个 token、
每个 KV head 的 K norm。

V 不参与 QK 内积，本项目对 V 使用更常见的 per-token/per-KV-head uniform
4-bit 量化，保存 index、scale 和 zero。

| 对比项 | K 路径 | V 路径 |
|---|---|---|
| 量化前处理 | 向量归一化 + 正交/Hadamard 旋转 | 直接对原 V 向量求 min/max |
| 4-bit 含义 | 16 个非均匀 Lloyd-Max centroid 的 index | `0..15` 均匀整数 index |
| metadata 粒度 | 每个 `(token,kv_head)` 一个 corrected norm | 每个 `(token,kv_head)` 一组 scale/zero |
| Decode 重建 | `centroid[index] * norm` | `index * scale + zero` |
| 是否模型校准 | codebook 由理论分布和 `(D,bits)` 决定 | scale/zero 由当前 V 向量决定 |

因此“TurboQuant 就是普通 INT4”不准确：存储位宽相同，但 K 的坐标变换、非均匀
codebook 和 Query matching rotation 都与普通线性 INT4 不同。

### 6. 为什么要对 K 做旋转？

旋转用于重新分配向量各坐标的能量，减弱异常值和坐标分布不均对标量量化的
影响。对于归一化 K，正交旋转保持二范数和内积结构：

$$
K_r = K\Pi^T, \qquad Q_r = Q\Pi^T
$$

只要 Q 和 K 使用一致的正交变换，就有：

$$
Q_rK_r^T = Q\Pi^T\Pi K^T = QK^T.
$$

具体左右乘形式取决于代码中向量采用行向量还是列向量，面试时重点应说明
“Q 与 K 使用匹配的正交旋转，理论内积保持不变”。

### 7. 为什么 Query 也要旋转？

Cache 中保存的是旋转坐标系下的 K index。如果 Q 仍在原坐标系，QK 就不再
对应原始 Attention score。Store 阶段旋转 K，Decode 前旋转 Q，使二者处在
同一个正交坐标系。项目 Stage1 benchmark 从预先生成的 `q_rot` 开始，因此
Query rotation 不计入 Stage1 时间。

若按行向量记法，Store 保存的是 $K_r=K\Pi^T$，Decode 应使用
$Q_r=Q\Pi^T$：

$$
Q_rK_r^T=Q\Pi^T(K\Pi^T)^T
=Q\Pi^T\Pi K^T=QK^T.
$$

必须同时满足两个条件：$\Pi$ 正交，即 $\Pi^T\Pi=I$；Q 和 K 使用同一变换及
一致的左右乘约定。V 不参与 QK score，不需要为了保持该内积而做同样旋转。

项目中 `q_rot` 是 `[B,Hq,D]` FP32 输入，说明 rotation 已在计时区间外完成；
回答性能时必须主动指出这一点。

### 8. Lloyd-Max codebook 是什么？

4 bit 对应 16 个 centroid。Lloyd-Max 根据目标概率分布迭代更新量化区间边界
和区间条件均值，使标量均方误差降低。Store 用 15 个 midpoint 对旋转后的
K 坐标做 bucketize，最终只保存 0 到 15 的 index；Decode 根据 index 查
16-entry FP32 centroid table。

Lloyd-Max 反复执行两个条件：

1. 固定 centroid 时，最小平方误差边界为相邻 centroid 中点：

   $$
   b_i=\frac{c_i+c_{i+1}}{2};
   $$

2. 固定区间 $[b_{i-1},b_i]$ 时，新 centroid 是该区间的条件均值：

   $$
   c_i=\frac{\int_{b_{i-1}}^{b_i}x f(x)\,dx}
   {\int_{b_{i-1}}^{b_i}f(x)\,dx}.
   $$

vLLM 对 $d\ge64$ 使用 $N(0,1/d)$ 近似，通过数值积分迭代求解，不需要先采样
真实模型 KV 数据。Store 使用 boundary，Decode 使用 centroid；cache 只保存 index。

### 9. K 是怎样量化和恢复的？

Store 的主要过程是：

1. 计算每个 K 向量的二范数；
2. 归一化并通过 GEMM 完成旋转；
3. 根据 midpoint 对每个坐标做二分 bucketize；
4. 两个 4-bit index 打包成一个 byte；
5. 保存校正后的 K norm。

Decode 对每个 nibble 查 centroid，再乘对应 `(token,kv_head)` 的 norm，得到
tile 内用于 QK 的 FP16 K。完整 K 不写回 Global Memory。

### 10. norm correction 是什么？

量化后的 centroid 向量范数不一定恰好为 1。Store 在开启 norm correction
时，将 centroid 向量的逆范数折叠进保存的标量：

$$
\gamma_{stored} = \frac{\lVert K\rVert_2}{\lVert c\rVert_2}.
$$

Decode 只需要计算 `centroid[index] * gamma_stored`，不必在每个 tile 重新求
centroid 向量范数。这是用 Store 阶段一次计算换 Decode 热路径更少的操作。

这里的粒度必须说清：对每个 token、每个 KV head 的 128 维 K 向量分别计算
$\gamma_{t,h}$，不是每层一个，也不是整个 KV head 跨 token 共用。若一次 Store
有 $N$ 个 token、$H_{kv}$ 个 head，就保存 $N\times H_{kv}$ 个 FP16 corrected
norm。

它同时承担两个作用：

- 恢复归一化前原始 K 的幅值 $\lVert K\rVert_2$；
- 修正 centroid reconstruction 后方向向量范数不再恰好为 1 的误差。

因此 `nc` 是 norm correction，不是 V affine quantization 的 scale。

### 11. V 为什么不用同样的 centroid 量化？

K 直接决定 QK score，对内积误差敏感；V 在 softmax 权重确定后参与加权求和，
工程实现选择更简单的 per-token/per-KV-head uniform quantization。4-bit V 使用：

$$
v_{recon} = index \times scale + zero, \qquad index\in[0,15].
$$

这样 Decode 只需 nibble unpack、整数转浮点和一次乘加，不需要 centroid lookup。

具体公式是：

$$
v_{min}=\min_dV_d,\qquad
scale=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right),
$$

$$
q_d=\mathrm{clip}\left(\mathrm{round}
\left(\frac{V_d-v_{min}}{scale}\right),0,15\right).
$$

cache 中 `zero` 保存的是 FP16 `v_min`，不是整数 zero-point。K 与 V 采用不同
量化器，是因为 K 误差先进入指数敏感的 QK/softmax，V 误差在线性加权路径中
传播；这是算法与工程开销的折中，不表示 V 精度不重要。

### 12. 当前实现使用 QJL residual 吗？

没有。当前研究的是 vLLM `turboquant_4bit_nc` 路径，使用 rotation、centroid、
norm、V scale/zero，不保存 QJL residual。不能把论文中更广泛的 TurboQuant
变体全部说成当前 Kernel 已实现的功能。

需要区分三层：

- **TurboQuant-MSE**：旋转后用 Lloyd-Max centroid 最小化坐标重建 MSE；
- **TurboQuant-Prod**：在 MSE reconstruction 外保留 residual，并用 QJL 估计
  residual 与 Query 的内积修正；
- **当前 vLLM `4bit_nc` 与本项目**：使用 centroid index + corrected norm，
  不保存 QJL projection/sign 或 residual norm。

所以论文设计 QJL 不代表所有部署 preset 必须使用。当前 Kernel 的参数列表和
134 B slot 中都没有 QJL payload；如果面试官追问，应从代码数据契约回答，而不是
把论文所有分支混成一个实现。

### 13. 4-bit 打包具体节省多少空间？

head dimension 是 128。K 的 128 个 4-bit index 占 64 B，V 也占 64 B；
另有三个 FP16 metadata：K norm、V scale、V zero，共 6 B。因此一个
token/KV-head 的逻辑 slot 是：

```text
K index   64 B
V index   64 B
metadata   6 B
total     134 B
```

相比只计算 K/V payload 的 FP16 `128*2*2 = 512 B`，payload 大约缩小 4 倍；
实际压缩率还应计入 metadata 和 block/page 管理开销。

把 metadata 纳入逻辑 slot 后，准确压缩比是：

$$
\frac{512}{64+64+2+2+2}=\frac{512}{134}=3.8209\times.
$$

固定 workload 下共有 $64\times4096\times8=2{,}097{,}152$ 个逻辑 slot：

- packed K/V payload：256 MiB；
- K norm、V scale、V zero：12 MiB；
- 合计：268 MiB。

这个数字不含 block table、allocator 对齐、空 physical page 和其他模型张量；因此
可以说“KV slot 逻辑压缩 3.82x”，不能说“整模型显存下降 3.82x”。

## Cache 布局与 Paged Attention

### 14. AoS 与 SoA 在这里分别是什么？

AoS 把一个 token/head 的 packed K/V 和 metadata 更紧密地放在一起。SoA 将
大块数据区和 metadata 区分开：

```text
data:     [token][kv_head][K64 | V64]
metadata: [kv_head][field][token]
```

Decode 对连续 token 批量读取相同 field 时，SoA 的 metadata 地址更规则；
实测仅从 AoS Triton V1 改为 SoA Triton V1，就从 1.692 ms 降到 1.284 ms，
约 1.318x。

当前 SoA 不是简单把整个 cache 变成 `[field][all tokens]`，而是混合布局：

- payload：`[position][kv_head][K64 | V64]`，让一个 token/head 的 K/V 紧邻；
- metadata：`[kv_head][field][position]`，让同一 head 的 16 个 norm/scale/zero
  可由连续 lane 合并读取。

因此 AoS→SoA 的收益主要来自 metadata transaction 和地址规则性，不意味着
payload 总字节数发生变化。AoS V1 与 SoA V1 的 1.318x 是 layout ablation；
不能把它算作 CUDA V1→V9 的优化收益。

### 15. 一个 physical block 的字节布局是什么？

一个 block 有 16 token 和 8 个 KV head。Data region 是：

```text
16 * 8 * 128 B = 16384 B
```

Metadata region 是：

```text
8 heads * 3 fields * 16 tokens * 2 B = 768 B
```

总计 `17152 B`，等于 `16 * 8 * 134 B`。CUDA Kernel 根据 block table 取得
physical block，再分别计算 data 和 metadata 地址。

地址计算可以展开为：

```text
block_base = physical_block * 17152
data_base  = block_base + position * (8 * 128) + kv_head * 128
meta_base  = block_base + 16384
meta_index = (kv_head * 3 + field) * 16 + position
```

`field=0/1/2` 分别对应 K norm、V scale、V zero。data region 的起点和 K/V
offset 都满足 4 B 对齐，这是 V6–V9 使用 `uint32_t` load 的前提。

### 16. 为什么 metadata 使用 FP16？

每个 token/KV-head 只需三个标量。FP16 将 metadata 控制在 6 B，同时其精度对
当前 4-bit 量化路径足够。Decode 按 `uint16` 位模式读取后转换为 half/float。
centroid table 本身仍是 16 个 FP32 值。

选择 FP16 的具体权衡是：

- 三个 FP16 metadata 共 6 B，slot 为 134 B，压缩比 3.8209x；
- 若都改为 FP32，共 12 B，slot 为 140 B，压缩比降到 3.657x；
- Decode 会将 FP16 metadata 转成 float 参与重建，减少存储并不等于全程 half
  算术；
- 代价是 norm/scale/zero 的舍入误差，必须通过 output、LSE 和模型质量验证。

centroid 只有 16 个全局共享常量，容量不是主要问题，所以保留 FP32；metadata
数量随 token 数增长，使用 FP16 的容量和带宽收益更明显。

### 17. block table 在 Kernel 中做什么？

Paged KV Cache 的逻辑 token 不保证位于连续 physical page。`block_table[b,
logical_block]` 将序列的逻辑 block 映射到 cache 中的 physical block。固定
workload 每个 tile 恰好 16 token，因此 V4 之后每个 tile 只需读取一次
block-table entry。

对逻辑 token $t$：

$$
logical\_block=\lfloor t/16\rfloor,\qquad pos=t\bmod16.
$$

Kernel 用 `block_table[b, logical_block]` 得到 physical block。当前 16-token tile
与 page block 完全对齐，所以 tile 内 16 个位置共享同一 physical block。若
block size 改变、split start 未对齐或 tile 跨页，就必须读取多个 entry 并处理
边界，当前固定 Kernel 会直接不适用。

Store 使用的是 `slot_mapping`，它告诉新 token 写入哪个 physical slot；Decode
使用 `block_table`，它从历史逻辑位置找到 physical page。两者职责不同。

### 18. 为什么向量化 `uint32` load 是安全的？

每个 packed K 或 V 区域是 64 B，slot 的 data 部分是 128 B，固定布局保证
读取地址满足四字节对齐。V6 每次读取四个 packed byte，对应八个 4-bit
dimension；随后通过 `half2` 写两个重建值，减少 load、地址计算和 shared
store 指令。对齐约束是 V6-V9 的显式限制，不能假定任意 cache layout 都安全。

精确地说，每线程一次 `uint32_t` load 得到 4 个 byte，也就是 8 个 nibble，而
不是每线程 128-bit `uint4`。安全性来自：

1. physical block base 按 cache allocation 对齐；
2. 每 position 的 payload stride 是 `8 * 128 = 1024 B`；
3. 每 head 的 data stride 是 128 B；
4. K 起点为 0、V 起点为 64 B；
5. `word * 4` 保持 4 B 对齐且不会越过各自 64 B 区域。

若 head dimension、bit width、metadata placement 或 allocator 对齐发生变化，应
重新证明这些条件并增加 launcher check，不能只保留 reinterpret cast。

### 19. 如何证明 CUDA 读取的是真实 vLLM Store 布局？

`validation.store_decode` 从原始 FP16 Q/K/V 开始，调用未经修改的 vLLM SoA
Triton Store，真实执行旋转、bucketize、packing 和 metadata 写入，然后将
同一个 cache tensor 直接交给 Triton Decode 与 CUDA V7，中间没有 byte
rearrangement。CUDA 与 Triton output 最大差约 `5.06e-06`，说明二者对 layout
的解释一致。

验证链路是：

```text
FP16 K/V
 -> 未修改 vLLM SoA Store
 -> packed cache + norm/scale/zero
 -> Triton Decode 与 CUDA Decode
 -> 比较完整 output 和 LSE
```

它能发现 nibble 顺序、field offset、physical-page 地址和 metadata dtype 等契约
错误。还要说明局限：该测试证明 Store→Decode 字节兼容和 kernel-level 数值一致，
不等于已经覆盖 production continuous batching、所有 ragged tail 或模型 PPL。

## Attention 与 Split-KV

### 20. Stage1 到底计算什么？

一个 CTA 对应 `(batch, KV head, split)`，处理该 KV head 对应的四个 Q head
和当前 split 的 128 个 token。它循环处理八个 16-token tile，完成：

```text
packed K/V load -> dequant -> QK -> online softmax -> PV
```

最后为每个 Q head/split 输出 128 维 partial output 和一个 split LSE。

固定 workload 的 launch 和 CTA 内工作量是：

```text
grid             = (64, 8, 32) = 16,384 CTAs
threads / CTA    = 128 = 4 warps
tokens / split   = 128
tiles / split    = 128 / 16 = 8
Q heads / CTA    = 4
```

每个 tile 依次完成 page lookup/metadata load、packed K/V cooperative load、K
centroid reconstruction、V affine reconstruction、QK MMA、online-softmax update
和 PV MMA。Stage1 不写完整 score matrix或 FP16 K/V，只写 split partial state。

### 21. 为什么要把 4096 token 分成 32 个 split？

如果一个 CTA 处理完整 4096 token，并行 CTA 数量会不足，单 CTA 生命周期也
很长。拆成 32 个 128-token split 后，可以在 batch、KV head 和 split 三个
维度产生更多 CTA，提高 GPU 并行度。代价是需要 Stage2 合并各 split state。

`num_splits` 是并行度与归并开销的权衡：

- split 太少：CTA 数不足，单 CTA 循环过长，尤其低 batch 难以占满 128 个 SM；
- split 太多：Q 被更多 CTA 重读，`mid_o` 线性增大，Stage2 工作增加，单 CTA
  初始化和 launch 占比上升；
- 当前 32 splits 让每份正好 128 token，匹配固定展开的 8 个 16-token tile。

它是该固定 workload 的选择，不是 TurboQuant 算法规定。生产实现应根据 batch、
context、Hkv 和 GPU 动态选择或 autotune。

### 22. Stage1 输出为什么是 129 个 float？

前 128 个是归一化后的 partial output，最后一个是该 split 的 log-sum-exp：

```text
mid_o shape = [B, Hq, 32, 128 + 1]
```

Stage2 使用每个 split 的 LSE 对 partial output 做数值稳定的重新加权，不能
简单对 32 份 partial output 求平均。

完整 shape 和大小是：

$$
mid\_o\in\mathbb{R}^{64\times32\times32\times129},
$$

$$
64\times32\times32\times129\times4
=33{,}816{,}576\ \text{B}=32.25\ \text{MiB}.
$$

前 128 项是该 split 内已经归一化的 output，最后一项是 split LSE。采用 FP32
是为了让跨 32 split 的重加权和 output accumulation 保持数值稳定。

### 23. Online softmax 的递推公式是什么？

对新 tile 的 score，先求 tile 最大值 `m_t` 和指数和 `l_t`。已有 state 为
`(m,l,o)`，合并时：

$$
m' = \max(m,m_t),
$$

$$
l' = l e^{m-m'} + \sum_j e^{s_j-m'},
$$

$$
o' = o e^{m-m'} + \sum_j e^{s_j-m'}v_j.
$$

最终输出 `o'/l'`，LSE 为 `m' + log(l')`。V8/V9 使用 `exp2f`，因此 score
先乘 `log2(e)`，最后再换回自然对数语义。

### 24. 为什么 online softmax 比两遍算法更适合？

两遍算法先算完所有 score、求全局 max/sum，再重新读取 score 和 V；需要更大
scratch 或额外 global/shared traffic。Online softmax 在每个 tile 到达时更新
state，V3 之后能在一次 tile traversal 中完成 QK 和 PV，并让 output
accumulator 长时间保留在寄存器中。

两遍方法通常需要：

1. 第一遍产生或重算所有 score，求全局 max；
2. 第二遍重新读取 score/K 或保存 score scratch，求指数和及 PV；
3. 在长 context 下保存更大的中间 score 或增加数据遍历。

Online softmax 每个 16-token tile 只保留 $(m,l,o)$，状态大小与 context 无关。
代价是旧 output accumulator 每轮都要乘 $\alpha=e^{m_{old}-m_{new}}$，并有 warp
reduction/指数指令；所以它降低存储与遍历，不代表 softmax 本身没有计算成本。

### 25. Stage2 怎样合并 split？

设第 $i$ 个 split 的 normalized partial output 为 $o_i$，LSE 为 $L_i$。先求：

$$
M=\max_iL_i,\qquad w_i=e^{L_i-M}.
$$

再得到：

$$
o=\frac{\sum_iw_io_i}{\sum_iw_i},\qquad
LSE=M+\log\sum_iw_i.
$$

减去 $M$ 避免指数溢出。CUDA Stage2 时间为 0.008868 ms，只占 V7 Full
Decode 的约 1.33%，但它是数学语义上必需的，不能因耗时小就省略。

### 26. 为什么 Stage1 与 Stage2 使用两个 Kernel？

Stage2 必须等所有 split CTA 写完 `mid_o`。普通 CUDA Kernel 内没有通用的
grid-wide barrier，因此使用 Kernel launch 边界表达全局同步最直接，也避免
cooperative launch 的额外约束。

替代方案各有明显代价：

- atomic counter + last-CTA reduction：需要严格内存顺序，调度和复用复杂；
- cooperative launch：要求整 grid 满足 cooperative residency，限制并行规模；
- persistent kernel：需要重写任务调度，并可能降低不同 workload 的适应性。

由于 Stage2 当前只占 Full 的 1.33%，两 launch 的清晰同步边界是更合理的工程
选择。只有在小 batch 下 launch latency 成为主要部分时，才值得重新评估融合。

### 27. Stage1 时间和 Full Decode 时间为什么不能混为一谈？

Stage1 benchmark 不含 Stage2、Query rotation、Store、输入构造和 JIT。
Full Decode 也只定义为预先旋转的 Q 和压缩 cache 经过 Stage1+Stage2，不包含
Store。简历和面试必须明确计时边界，否则 `0.485 ms` 不能被描述成完整端到端
请求延迟。

项目里至少有三组必须分开的数字：

| 口径 | CUDA 时间 | 对比含义 |
|---|---:|---|
| V9 Stage1 | 0.484516 ms | 最新 Stage1 candidate |
| V7 Stage1 | 0.631255 ms | Full benchmark 中单独测量 |
| V7 Full | 0.664842 ms | V7 Stage1 + CUDA Stage2 的独立 runner |

Full 也不等于完整模型 Decode：它不含 QKV projection、RoPE、Query rotation、
Store、其他 layer、采样、allocation 和 JIT。简历中的“完整 Decode Kernel 链路”
应解释为本项目定义的 attention Stage1+Stage2 链路。

## GQA、WMMA 与 Tensor Core

### 28. GQA-4 在这个项目中是什么意思？

32 个 Q head 共享 8 个 KV head，因此每个 KV head 对应四个 Q head。一个 CTA
以 KV group 为单位，解码一份 K/V tile，并为四个 Q head 复用它。这样避免
每个 Q head 都重复读取和反量化同一份 K/V。

头映射是：

$$
G=H_q/H_{kv}=32/8=4,
$$

$$
qh=kvh\times4+q_{local},\qquad q_{local}=0,1,2,3.
$$

例如 KV head 3 服务 Q head 12–15。共享的是同一历史 K/V 向量、K norm 和 V
scale/zero；不共享的是四个 Query、QK score、softmax $(m,l)$ 和最终 output。
这一区分决定了哪些数据适合 CTA 共享、哪些状态必须按 Q head 独立保存。

### 29. 为什么普通 `m16n16k16` 会浪费 Tensor Core 槽位？

V3-V7 把四个 Q head 放到 WMMA 的 M 维。硬件 tile 要求 M=16，但只有四行
真实数据，其余 12 行是 padding，有效行比例只有 `4/16=25%`。虽然 Tensor
Core 很快，这种结构仍执行了无效 HMMA 工作，并扩大 accumulator/scratch。

可以把 QK 看成：

```text
Q tile:  [16 rows, 128 dims]，只有 row 0..3 是真实 Q head
K tile:  [16 tokens, 128 dims]
output:  [16 Q rows, 16 token columns]
```

每个 K=16 的 MMA step 都会计算 16 行，padding 行不会因为输入为 0 而自动免除
Tensor Core 指令。25% 是**矩阵槽位有效率**，不是 NCU 实测 Tensor Core active
百分比，也不代表整个 Kernel 只有 25% 利用率。

### 30. 为什么不能把四个 KV head 和 16 个 Q head 直接拼成一次 dense MMA？

因为四个 GQA group 使用四份不同的 K/V 矩阵。普通 dense GEMM 的同一次
矩阵乘法要求所有输出行共享同一个右操作数；直接堆叠会产生跨 group 的错误
Q-K 配对。除非构造 block-diagonal K，这又会引入更多零和复杂布局，因此
不能只凭“16 个 Q head 正好填满 M=16”判断数学上成立。

若把四个 group 的 Query 堆成 16 行，而 K 只放某一个 group，则另外 12 行乘错
K；若把四组 K 也拼在普通 dense 右操作数中，会产生所有 Q-K group 的交叉项。
数学上可构造 block-diagonal multiplication 屏蔽交叉项，但需要更大的零填充矩阵，
数据布局和 output 选择成本通常抵消收益。

真正可行的跨 group 合并需要支持独立 batch/group operand 的 MMA 组织，或者让
不同 warp 执行各自 MMA；不能仅以 WMMA 的 M=16 容量作为正确性依据。

### 31. V8 怎样提高 GQA-4 的 MMA 利用率？

V8 在每个合法 KV group 内转置两次乘法：

```text
QK: K(16x16)   * Q^T(16x8)
PV: V^T(16x16) * P(16x8)
```

它使用原生 `mma.sync.aligned.m16n8k16`，让四个 Q head 占 N=8 的四列，有效
槽位比例从 `4/16=25%` 提高到 `4/8=50%`。静态 HMMA site 从 V7 的 160
降到 V8 的 80。

关键是把 Query head 放到 `m16n8k16` 的 N 维：

- M=16 对应 16 个 token 或 output dimension 行；
- N=8 中前 4 列对应 4 个真实 Query head；
- 后 4 列仍为空，因此不是 100%；
- K=16 沿 head dimension 分块，D=128 需要 8 个 K step。

PV 也做匹配转置，让相同 N=8 的四列继续代表四个 Query head。QK 和 PV 必须
成对调整数据布局，否则只优化其中一侧会引入额外转置或错误 output mapping。

### 32. 为什么 V8 不继续使用 C++ WMMA API？

CUDA C++ WMMA 常用接口提供的是 `m16n16k16` fragment，而 V8 需要明确的
`m16n8k16` register contract。Inline PTX 能直接指定 MMA shape 和 operand
register，并手工实现 lane 到矩阵元素的映射。代价是代码与 `sm_89` 架构和
fragment layout 更紧密耦合。

实际 PTX 指令是：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

含义是 A/B 输入 FP16、accumulator/output FP32、矩阵 shape 为 16×8×16。代码
必须手工将每个 lane 的 half2 打包成 PTX 要求的 A/B register，并把四个 FP32
accumulator register 映射回 token、Q head 和 output dimension。

这提高了控制力，但可移植性更差：换 `sm_86`、其他 MMA shape 或 CUDA toolchain
时，需要重新检查 PTX 支持、lane mapping、寄存器数和 correctness。

### 33. V8 节省了哪些资源？

相对 V7：

```text
                    V7          V8
register/thread     49          51
shared/CTA          14224 B     10336 B
static HMMA         160         80
static BAR.SYNC     34          34
```

V8 用两个额外 register 换取更小的 Q、QK scratch 和 PV fragment，Stage1
从 0.631 ms 降到 0.513 ms，约 1.230x。

这些数字的含义需要分开：

- register/thread 和 shared/CTA 是编译后资源用量，会影响 resident CTA 上限；
- static HMMA/BAR 是 SASS 中静态指令 site，受循环展开影响；
- static site 减半不等于运行时间减半，也不等于动态 Tensor 指令必然减半；
- V8 实测 1.230x 才是最终性能证据。

V8 shared memory 减少 3,888 B，主要来自更紧凑的 Q/QK/P layout；register 从 49
升到 51，说明优化不是所有资源都同时下降，而是用少量 register 换取更少 Shared
Memory 和无效 MMA。

### 34. V9 借鉴 FlashInfer 的地方是什么？

借鉴的是 register-resident attention state，而不是直接调用 FlashInfer API
或复制它的 Kernel。V8 会把 QK accumulator 写入 `qk_s`，同步后再由四个 warp
读取并更新 softmax。V9 让 warp 0 直接按 lane class 对四个 Q-head score 列
做 max/sum reduction，在寄存器中维护 `(m,l)`，只将 FP16 probability 写给
后续 PV。

由于 compressed K/V 必须执行 nibble unpack、centroid lookup 和 scale/zero
重建，普通 `cp.async` 不能直接完成这段变换；因此这里优先迁移 FlashInfer
最适合当前数据路径的 state-fusion 思路。

V8 的数据路径是：

```text
QK MMA accumulator
 -> qk_s Shared Memory
 -> CTA barrier
 -> 四个 warp 读取 score
 -> 更新各自 online-softmax state
```

V9 改为 warp 0 直接按 `lane % 4`/lane group 持有四个 Query head 的 score 与
`running_m/running_l`，在寄存器内完成 max/sum，再只把 FP16 probability 写入
`p_s` 供 PV。这样删除 512 B `qk_s` 及其 producer-consumer barrier。

借鉴的是“attention state 尽量寄存器常驻”的原则；项目没有新增 FlashInfer
依赖，也没有宣称直接采用 FlashInfer Kernel。

### 35. V9 的结果和资源变化是什么？

```text
                    V8          V9
time                0.513208    0.484516 ms
register/thread     51          50
shared/CTA          10336 B     9824 B
static HMMA         80          80
static BAR.SYNC     34          26
```

V9 删除 512 B `qk_s`，每个 tile 少一次 CTA barrier，并略降寄存器。相对 V8
提升 1.059x，相对 Triton V2-fixed 提升 2.221x。

还应补充三个限制：

1. 这是 Stage1 时间，不含 Stage2；
2. 50% MMA slot utilization 没有改变，因为仍是 4 个真实 Q head 填 N=8；
3. 当前没有正式采集 V9 在 4090 上的动态 NCU 报告，不能仅凭资源和时间断言
   某个硬件 counter 已达到多少。

从 useful work 角度，V9 在 0.484516 ms 内完成约 4.295 GFLOPs，即 8.86 useful
TFLOP/s；考虑 50% 槽位后估算 executed Tensor work 约 8.59 GFLOPs，即 17.73
TFLOP/s。它远低于 4090 dense Tensor 峰值，符合 memory/data-movement-side 的
静态判断，但最终瓶颈仍需 NCU 证实。

## CUDA V1-V9 演进

### 36. CUDA V1 的设计和问题是什么？

V1 一个 CTA 负责一个 `(batch, KV head, split)`，四个 Q head 复用解码后的
K/V。问题是每个 token 都要为四个 head 做多轮 warp reduction，并频繁通过
Shared Memory 做 CTA 协作。它是正确、可复用 K/V 的起点，但 2.074 ms 比
Triton baseline 慢。

V1 建立了后续优化必须保持的三个契约：

- grid 使用 `(batch,kv_head,split)`，四个 GQA Query 共享一份 K/V；
- 直接消费 134 B SoA compressed slot，不物化 global FP16 K/V；
- 输出 `[B,Hq,S,D+1]` 的 partial output/LSE，能接同一 Stage2 语义。

它的价值不是性能，而是先做出可验证 CUDA baseline。V1 慢说明“从 Triton 改写
成 CUDA”本身不会自动加速；warp reduction、Shared Memory 往返和同步必须逐项
分析。

### 37. CUDA V2 为什么每个 Q head 一个 warp？

这样 QK reduction、softmax state 和 output 都可以保留在 warp 内，移除 CTA
barrier，并把静态 `SHFL.DOWN` site 从 20 降到 5。代价是四个 warp 重复读取
并反量化同一 K/V。最终从 2.074 ms 降到 1.748 ms，说明同步减少有收益，
但重复 decode 限制了进一步提升。

这是典型的资源交换：

| 收益 | 代价 |
|---|---|
| 每 warp 独立维护一个 Q 的 max/sum/output | 同一 KV group 被四个 warp 重复读取 |
| warp shuffle 代替部分 CTA Shared Memory 通信 | nibble unpack、centroid lookup、V dequant 重复四次 |
| CTA barrier 减少 | cache traffic 和整数/lookup 指令增加 |

V2 的实验结论不是“warp-per-Q 永远更优”，而是当前 V1 的同步成本高于重复解码
代价；后续 V3 又通过 CTA 共享 K/V 和 Tensor Core 重新寻找更好的平衡。

### 38. CUDA V3 的关键变化是什么？

V3 建立单遍 tiled Tensor Core 执行图：一次解码 16 个 token，在 WMMA 上完成
QK，更新 online softmax，再做 PV；八个 tile 的 output accumulator 保留在
fragment registers。它移除两遍 score/weight staging，从 V2 的 1.748 ms
降到 1.381 ms，并在 SASS 中确认生成 HMMA 指令。

每个 split 的执行顺序是：

```text
8 × {
  decode 16-token K/V tile
  QK WMMA over D=128
  update four online-softmax states
  rescale old output accumulators
  probability × V WMMA
}
write normalized partial output + LSE
```

这里最关键的算法变化是从“先生成全部 score，再处理 softmax/PV”改成单遍流式
状态，而不只是把标量乘法替换成 Tensor Core。验证 SASS 出现 HMMA 只能证明
Tensor 指令生成，最终收益仍由 CUDA Event 结果确认。

### 39. CUDA V4 为什么要做固定 workload 特化？

V4 利用每个 split 固定 128 token 且按 16 对齐的条件，只加载一次 tile 的
block-table entry，只初始化四个有效 Q 行，将 centroid table放到 warp
register，并完全展开八个 tile。它用通用性换取更少的分支、地址计算和循环
控制，从 1.381 ms 降到 1.121 ms。

固定特化具体依赖：`D=128`、GQA=4、block size=16、split=128 token、32 splits，
以及 split start 16-token 对齐。由此可以：

- 完全展开 8 个 tile 和 D 方向 8 个 K-step；
- 删除通用 tail mask和动态循环控制；
- 每 tile 只查一次 page；
- 只初始化 4 个真实 Q 行；
- 将 16-entry centroid 分布到 warp lane register。

代价是 shape 不满足时当前 Kernel 会 return，而不是自动走通用 tail。面试时应把
它称为 fixed-workload specialization，不应说成支持任意 vLLM 请求。

### 40. CUDA V5 的 fragment 直接写回是什么？

V4 把完整 `16x128` output accumulator 先写到 Shared Memory，再只读取四个
有效行。V5 根据在 `sm_89` 上探测出的 WMMA lane-to-row mapping，直接从
fragment register 将四行写到 `mid_o`，移除约 7 KB scratch 和一次大规模
shared round trip，时间降到 0.845 ms。

优化前路径是：

```text
fragment registers -> store_matrix_sync -> shared scratch
CTA barrier -> scalar threads读取有效四行 -> mid_o
```

优化后直接根据 lane/index 映射把四行写入 `mid_o`。风险是 WMMA fragment 内部
布局不是跨架构稳定 API；项目用 `wmma_fragment_probe.cu` 在 `sm_89` 验证映射。
因此 V5 的 1.326x 收益带有架构特化成本，迁移时必须重新 probe 和回归。

### 41. CUDA V6 的向量化为什么收益大？

INT4 decode 涉及大量细粒度 byte load、nibble 提取、地址计算和 half store。
V6 用一个对齐 `uint32` 同时读取四个 byte，再用 `half2` 写八个重建维度，
减少指令和 shared-store 数量。它没有改变 Tensor Core 工作量，却从 0.845 ms
降到 0.639 ms，说明 decode 数据路径此前占比很高。

准确的数据粒度是每线程 32-bit load：

```text
uint32 load = 4 packed bytes = 8 INT4 coordinates
```

每个 byte 拆 low/high nibble，K 通过 register-shuffle centroid lookup 后乘 norm，
V 执行 `index*scale+zero`，最后用 `half2` 一次写两个 Shared Memory 元素。收益来自
减少 load/store/address 指令和提高合并访问效率，不是显存 payload 再减少。

该版本不能表述成“每线程 128-bit vector load”；128-bit load 是简历其他项目的
技术点，不属于这个 Kernel。

### 42. CUDA V7 怎样减少 barrier？

V6 每个 tile 末尾有一个 barrier。V7 发现下一个 tile 开头原本就有 metadata
发布 barrier，而 metadata storage 与 K/V storage 独立，因此可以让这个开头
barrier 同时承担“等待上一个 PV 完成”和“发布新 metadata”两个作用。静态
barrier 从 42 降到 34，收益约 1.012x。

删除同步前的正确性证明是：

1. 上一轮 PV 读取的是 `p_s/v_s`，output accumulator 已进入各 warp register；
2. 下一轮 warp 0 先写的是独立的 metadata/data-base arrays；
3. 下一轮已有 barrier 会等待所有 warp 完成上一轮 PV；
4. barrier 之后才允许线程覆盖下一轮 `k_s/v_s/p_s`。

所以删除的是冗余生命周期边界，不是依赖“warp 恰好同步”的冒险优化。时间只提升
1.2% 也应保留，因为结果稳定、实现变化单一，并为后续 barrier 分析提供基线。

### 43. 为什么 V7 到 V8 的收益比 V6 到 V7 大？

V7 只消除一部分同步，主计算形状仍有 75% 无效 WMMA 行；V8 直接改变 MMA
shape，将 HMMA 数量减半，并缩小 Shared Memory 和 accumulator。前者是局部
调度优化，后者减少了核心 Tensor Core 工作量，所以 V8 获得约 23% 提升，
而 V7 只有约 1.2%。

用数据表达：

$$
V6\rightarrow V7:\quad0.638863/0.631122=1.012\times,
$$

$$
V7\rightarrow V8:\quad0.631122/0.513208=1.230\times.
$$

V7 优化的是同步等待这一局部开销；V8 同时降低无效 MMA 槽位、静态 HMMA site
和 Shared Memory footprint，作用于每个 tile 的主计算图。性能差异符合修改的
覆盖范围，但最终不能仅靠“改得更大”解释，仍需同一 harness 的测量。

### 44. 为什么要保留所有历史版本？

每个版本只引入一类主要变化，构成可复现的 ablation：线程映射、单遍算法、
固定特化、写回、向量化、同步、MMA shape 和 softmax fusion。这样可以解释
性能来自哪里，也能避免把多个变化一起提交后无法归因。面试时这比只展示
最终 V9 更能体现性能工程方法。

版本保留还承担回归定位：若 V9 在新 CUDA 版本上错误，可以二分判断问题首次出现
在 fragment mapping、vector load、barrier 还是 inline PTX；若某 GPU 上 V8 比
V7 慢，也能识别是 MMA shape 还是其他资源变化。

每版的正确讲法是“瓶颈假设 → 单一主要修改 → correctness → 资源/SASS → 实测
时间 → 新限制”，而不是把 V1–V9 当成九个没有因果关系的文件。

## Correctness 与实验设计

### 45. 上游 Triton V2 曾经有什么 correctness 问题？

复制的 V2 使用 `tl.interleave(v_lo, v_hi)` 重建 V，随后直接交给 `tl.dot`。
在 CUDA 上该 layout 与 dot 期望不兼容，造成 V 列置换。QK 和 softmax 未受
影响，所以 LSE 看起来正确，但 output 最大误差约 0.325。

修复版根据 `d//2` 和 nibble shift 直接构造最终 `[TILE, D]` V layout，输出
误差恢复到约 `1e-4`。这说明只检查 LSE 不足以证明 Attention 正确。

这个 bug 的诊断逻辑是：

- LSE 只依赖 QK score，LSE 正确说明 K decode、QK 和 softmax statistic 大概率正确；
- output 还依赖 probability 与 V 各列的对应关系；
- output 大错但 LSE 正确，将怀疑范围缩到 V layout/PV；
- `tl.interleave` 产生的内部 layout 与 `tl.dot` operand 解释不一致，最终导致列置换。

修复后需要同时比较完整 `mid_o[...,0:128]` 和 `mid_o[...,128]`，并更新 baseline
性能；不能继续引用错误版本较快或较慢的数字。

### 46. CUDA V9 的正确性怎样验证？

正式 harness 为所有实现构造同一逻辑 cache，并与 SoA Triton V1 比较完整
`mid_o` 的 partial output 和 LSE，同时检查所有值 finite。V9 的结果是：

```text
output max/mean  9.6827745e-05 / 1.2782620e-05
LSE max/mean     2.4318695e-05 / 3.7480786e-06
```

此外，V7 Full Decode 与 Triton Full Decode 的最终 output 最大差约
`5.66e-07`，Store 兼容性测试也覆盖真实压缩 cache。

验证应分三层：

1. **Stage1 同 cache 对照**：比较所有 split 的 partial output、LSE 和 finite；
2. **Stage2/Full 对照**：比较最终 `[B,Hq,D]` output 和 `[B,Hq]` LSE；
3. **Store→Decode 契约**：用未修改 vLLM Store 生成 cache，排除 synthetic layout
   自己写错但双方同时读错的风险。

还应跑多随机种子、极端 norm、常量 V、随机 page table 和 tail shape。当前固定
harness 完成了核心链路，但后几类 production case 仍属于待扩展范围。

### 47. 为什么 CUDA V3-V9 与 Triton V1 有约 `1e-4` 误差？

这些版本使用 FP16 Tensor Core operand、不同的求和顺序、online softmax 和
fast-math 指数近似。浮点加法不满足结合律，因此与逐元素 FP32 路径不会逐位
一致。误差应结合 reference、最终输出和量化误差判断，不能把非零差异直接
当成 Kernel 错误。

判断误差是否可接受不能只设一个绝对阈值，应同时看：

- max absolute error：捕获最坏元素；
- mean absolute error：观察整体偏差；
- relative error：避免不同量级被同一绝对阈值掩盖；
- LSE 与 output：分别覆盖 QK/softmax 和 PV；
- 与同 Tensor Core/fast-math reference 的差异：隔离实现顺序误差。

若误差随 context、split 数或输入幅度持续放大，则不能简单归因于浮点结合律，
需要检查 online-softmax rescale 和 Stage2 合并。

### 48. 如何区分 Kernel 数值误差和 4-bit 量化误差？

Store 验证中，CUDA 与 Triton 读取同一量化 cache，output 最大差约
`5.06e-06`；而量化 Decode 与原始 FP32 Attention 的 output 最大差约
`1.87e-02`。前者远小于后者，说明较大的差异来自预期量化损失，不是 CUDA
layout 或 Attention 算法错误。

隔离方法的核心是固定变量：

```text
CUDA compressed decode vs Triton compressed decode
  -> 测 Kernel/layout/执行顺序误差

Triton or CUDA compressed decode vs canonical FP32 attention
  -> 包含 4-bit quantization error
```

如果第一组已经很大，应先修 Kernel；只有第一组足够小，第二组才能用于评估量化
误差。模型 PPL/下游任务又是第三层，不能由单个 synthetic output max error代替。

### 49. Benchmark 怎样保证版本比较公平？

所有版本使用同一逻辑输入、相同 shape 和输出语义。Harness 在 AoS/SoA 间做
无损转换，转换不在计时区；每个实现先 warmup，再用 CUDA Event 对 100 次
launch 计时，进行五轮并轮换测量顺序，最终报告五轮中位数。JIT、分配和输入
构造均不计时。

公平性还包括：

- AoS/SoA 来源于同一逻辑 cache，转换是 lossless 且在计时外；
- 所有输出预分配，避免 allocator 进入计时；
- runner 顺序每轮旋转，降低固定顺序带来的 boost/cache 偏差；
- 每轮 100 次 launch，5 轮取中位数；
- 性能比较前先检查完整输出语义相同。

剩余风险包括 GPU 温度/功耗、后台进程、cache 热度和固定 shape 过拟合。正式报告
最好保存每轮原始样本、GPU clock/power，并在独立进程重复。

### 50. 为什么使用中位数而不是只报最快一次？

GPU Boost、温度、后台负载和首次 cache 状态会造成波动。最快值可能只是偶然
高频状态，平均值也容易受离群点影响。多轮交错测试加中位数更稳健；小于几个
百分点的优化还应重复验证，并结合资源与 SASS 证据。

中位数回答的是“典型一轮性能”，不是置信区间。严谨报告还应给出最小/最大值、
标准差或分位数。若 V6→V7 只有约 1.2%，必须确认收益大于运行波动；若只是某一
轮最快，不应作为稳定优化写入简历。

五轮轮换顺序也比连续跑完某版本再跑下一版本更好，因为它让温度和 boost 随时间
变化更均匀地影响所有 candidate。

### 51. 你使用了哪些 profiling 证据？

当前可复现的 V3-V9 证据主要来自 `cuobjdump`：register/thread、Shared
Memory/CTA、HMMA 和 `BAR.SYNC` 静态 site，并通过 CUDA Event 测量性能。
仓库中的旧 NCU 报告生成于 Triton V2 correctness 修复前，CUDA V1 的 NCU
采集还遇到 `ERR_NVGPUCTRPERM`。

因此面试时不能声称 V9 已经通过 NCU 证明达到某个 DRAM 峰值百分比；可以说
静态资源和 SASS 支持优化归因，动态瓶颈仍需要重新采集 NCU。

当前可以可靠陈述的证据是：

| 证据 | 能说明什么 | 不能说明什么 |
|---|---|---|
| CUDA Event | 同 workload latency | 具体 stall 原因 |
| ptxas/cubin resource | registers/shared/local stack | achieved occupancy |
| SASS HMMA/BAR site | 指令是否生成、静态结构变化 | 动态执行次数和利用率 |
| correctness harness | 数值与 layout 一致 | 模型质量 |
| 历史 NCU | 提供旧版本诊断假设 | 当前 4090 V9 瓶颈 |

下一次 NCU 应针对 V7/V8/V9 同一 commit 和输入采集 DRAM/L2、Tensor pipe、warp
stall、occupancy、MIO、bank conflict 和 local spill，形成可比较证据。

### 52. 静态 HMMA 或 barrier 数量能直接等价为运行时间吗？

不能。完全展开的八个 tile 会让同一逻辑操作出现多个静态 site；实际性能还
取决于指令吞吐、依赖、occupancy、memory latency 和调度。V8 的 HMMA 减半
且实测明显加速，V9 的 barrier 减少也有稳定收益，但必须以同环境 CUDA Event
结果确认，不能只看反汇编计数。

例如一个循环完全展开八次，会把逻辑上一处 barrier 展开成多个 static site；反之，
未展开循环只有一个 site，却可能动态执行八次。HMMA 也受 predicate、loop trip、
warp 数和输入 shape 影响。

正确归因顺序是：先比较代码路径和动态工作量，再看 SASS 是否符合预期，随后看
资源与 NCU counter，最后以稳定 wall/kernel time 判断收益。任何单一静态数字都
不能替代这条证据链。

## 性能口径与设计取舍

### 53. 4.28x、2.22x 和 1.66x 分别是什么？

三者比较对象不同：

```text
CUDA V1 -> CUDA V9 Stage1       2.074204 / 0.484516 = 4.281x
Triton V2-fixed -> CUDA V9      1.075988 / 0.484516 = 2.221x
Triton -> CUDA V7 Full Decode   1.104148 / 0.664842 = 1.661x
```

不能把 V9 的 Stage1 速度与 V7 的 Full Decode 速度混合，也不能把 4.28x 描述
成完整 vLLM 请求端到端加速。

建议面试中只主动强调与简历一致的 `1.66x Full Decode`，其余作为追问展开：

- 4.28x 是项目内部 CUDA V1→V9 的优化演进，证明迭代收益；
- 2.22x 是最新 Stage1 对强 Triton Stage1 baseline；
- 1.66x 是可完整比较的 V7 Stage1+Stage2 对 Triton Full，也是简历数字；
- 三者 workload 相同，但入口和版本不同，不能拼接成“V9 Full 2.22x”。

### 54. “加速 4.28x”和“耗时降低多少”有什么区别？

```text
speedup = 2.074204 / 0.484516 = 4.281x
time reduction = 1 - 0.484516 / 2.074204 = 76.64%
```

4.28x 不是“耗时降低 428%”，也不宜简单说“性能提升 328%”而不说明计算口径。

一般公式是：

$$
speedup=\frac{T_{old}}{T_{new}},\qquad
\mathrm{latency\ reduction}=1-\frac{T_{new}}{T_{old}}.
$$

对简历 Full 数据，1.661x 对应 latency 降低 39.79%；对 V1→V9，4.281x 对应
降低 76.64%。回答“快了多少”前先确认面试官问 throughput multiplier 还是
latency percentage。

### 55. Shared Memory 越少、occupancy 就一定越高吗？

不一定。CTA residency 同时受 registers、Shared Memory、threads、warp slots
和架构限制。减少 Shared Memory 可能解除某个限制，也可能根本不是当前 limiting
resource。Occupancy 只是 latency hiding 的条件，不是最终性能指标；降低资源
却增加指令或 spill 仍可能变慢。

应按资源上限逐项判断 CTA residency：

```text
register limit  = SM register file / registers per CTA
shared limit    = SM shared capacity / shared bytes per CTA
thread limit    = max resident threads / threads per CTA
block limit     = architecture max resident CTAs
```

最小值决定理论 active CTA，再结合实际 eligible warps、stall 和 spill。V7→V8
shared 从 14,224 B 降到 10,336 B，但 register 从 49 增到 51；是否提高 occupancy
必须用 occupancy calculator/NCU 验证。即使 occupancy 不变，更少 Shared Memory
traffic 和 MMA 工作仍可能加速。

### 56. 为什么不用 double buffering 或 `cp.async` 做完整流水？

压缩 cache 不是可以直接异步复制成最终 FP16 tile的数据。K 需要 unpack、
centroid lookup 和 norm，V 需要 unpack、scale/zero；`cp.async` 只能搬字节，
不能执行这些变换。双缓冲还会增加 Shared Memory，并可能降低 resident CTA。

它仍然是可实验方向，例如异步预取 packed byte 或 metadata，但必须测量变换、
额外同步和 occupancy 的综合成本，不能因为 FlashInfer 使用 pipeline 就假定
本项目照搬一定更快。

采用 `cp.async` 前应先回答：

1. NCU 是否显示 global-memory long scoreboard 是主要 stall？
2. 当前 tile 计算是否足够长，能覆盖下一 tile packed-byte load？
3. 双份 packed/shared buffer 会不会降低 resident CTA？
4. metadata、block-table 和 nibble transformation 如何与异步 copy 排程？
5. 新 pipeline 增加的 commit/wait/barrier 是否小于被隐藏的 latency？

可先只双缓冲原始 packed K/V 和 metadata，再在消费前 unpack，而不是试图让
`cp.async` 直接完成反量化。优化后必须比较 Full 时间和 NCU stall，不应只展示
source 中出现了异步指令。

### 57. 为什么 V9 仍然只有 50% 的 N 维有效槽位？

`m16n8k16` 的 N 固定为 8，而 GQA group 只有四个 Q head，所以仍有四列为空。
它已比 `m16n16k16` 的四行有效更好，但不是 100%。进一步填充需要让一次 MMA
同时处理更多合法输出，同时保证不同 KV group 的 K/V 不交叉；这通常要求
更复杂的 block-diagonal、稀疏或多-MMA 调度，收益需覆盖布局成本。

不同 GQA ratio 的结果也不同：

- GQA=8：刚好填满 N=8，槽位可达 100%，但 Q/softmax/output state 翻倍；
- GQA=4：当前 4/8=50%；
- GQA=2：只有 2/8=25%，可能需要不同 MMA shape 或 SIMT 路径；
- MHA/GQA=1：量化 decode 访存仍有价值，但当前 Tensor mapping 很浪费。

不能跨 KV group 简单填空列，因为每组使用不同 K/V。通用 backend 应按 GQA ratio
选择模板，而不是强制所有模型使用 V9 mapping。

### 58. 为什么 V9 没有直接替换 Full Decode 中的 V7 Stage1？

当前版本演进把 V8/V9 保持为独立 Stage1 candidate，V7 则提供经过验证的
Stage1、Stage2 和 Full launcher。将 V9 接入 Full Decode 在工程上可做，但
需要新增稳定导出、完整回归和 Store 路径验证。当前文档明确区分这两个范围，
避免把 Stage1 实验误报成已完成的 production chain。

正式接入至少需要：

1. 确认 V9 `mid_o` layout、normalization 和 LSE 与 V7 Stage2 契约逐元素一致；
2. 导出 V9 Stage1 和 Full launcher，复用预分配 output；
3. 跑 Stage1、Stage2、Full 三层 correctness；
4. 跑未修改 vLLM Store→V9 Full compatibility；
5. 在同一 harness 独立测 V9 Full，不能用 V9 Stage1 + V7 Stage2 的两个中位数
   手工相加；
6. 更新文档后才能把 V9 称作完整链路。

### 59. 这个项目目前最大的局限是什么？

主要局限包括：固定 `B=64/context=4096/Hq=32/Hkv=8/D=128`；每个 split
必须是对齐的 128 token；V8/V9 的 inline PTX 和 fragment mapping 面向
`sm_89`；合成 benchmark 使用连续 page 和统一 sequence length；尚未覆盖
随机 block table、ragged batch、其他 GQA ratio 和完整 vLLM backend 注册。

可按四类归纳：

| 类别 | 当前限制 |
|---|---|
| Shape | 固定 D=128、GQA=4、128 token/split、32 splits |
| Layout | block size=16、4 B aligned SoA payload、固定 metadata fields |
| Architecture | `sm_89` PTX/fragment mapping，未验证其他 GPU |
| System | 未注册 production backend，未覆盖 continuous batching、tail、TP/多 GPU |
| Quality | 有 kernel correctness，没有完整模型 PPL/downstream 报告 |

这些限制不否定固定 workload 的优化结果，但决定了简历必须写“固定 workload”，
不能写“通用 vLLM Decode 已全面提速”。

### 60. 下一步最值得做什么？

优先级较高的是：

1. 将 V9 Stage1 接入 CUDA Full Decode 并重复 Store compatibility；
2. 支持 variable sequence length、tail split 和随机 block table；
3. 为不同 GQA ratio、head dimension 和 batch 做模板化/autotune；
4. 重新采集 V8/V9 NCU，确认 memory、Tensor Core、barrier 和 scheduler stall；
5. 尝试 packed-byte/metadata 预取、warp specialization 与 Stage1/Stage2 融合边界；
6. 解决上游 cache shape 声明与 launcher 实际 position-major 语义的集成契约。

优化优先级应由 profiler 和端到端占比决定，而不是继续无目标减少几条指令。

建议按风险和收益排序：

- **第一优先级**：把 V9 接入 Full，消除简历最新 Stage1 与 Full 版本不一致；
- **第二优先级**：支持 tail/ragged/random page，证明不是只对连续 synthetic case；
- **第三优先级**：采集当前 4090 NCU，确认下一项优化针对真实瓶颈；
- **第四优先级**：扩展 D/GQA/split 模板和 dispatch；
- **第五优先级**：模型 PPL/任务质量及 production vLLM 集成。

每一步都应有 acceptance criteria：correctness threshold、支持 shape、Full latency、
资源上限和回归测试，而不是只列技术名词。

## 项目贡献与行为问题

### 61. 你个人的核心贡献可以怎样回答？

可以概括为：

1. 搭建统一输入、correctness、五轮 CUDA Event benchmark 和真实 Store 验证；
2. 将反量化、Tensor Core Attention 与 online softmax 融合，完成 V1-V9
   逐版本可归因优化；
3. 针对 GQA-4 设计 `m16n8k16` 转置映射，并借鉴 FlashInfer state fusion
   删除 QK shared-memory 往返；
4. 用 cubin/SASS 资源、HMMA/barrier 计数和同轮性能结果验证优化，而不是只看
   CUDA 源码猜测收益。

这段必须根据自己真正完成的部分调整。如果某些版本由他人实现，应明确自己的
设计、实现、测试或分析边界。

更适合面试的完整回答是：

> 上游提供 TurboQuant 算法、Store/Decode Triton 路径和 cache 语义。我先建立
> 同一 compressed cache 下的 CUDA/Triton correctness 与 benchmark，然后从
> V1 到 V9 逐步重构 CTA/warp 映射、单遍 online softmax、Tensor Core、fragment
> 写回、packed load、barrier 和 register-resident state；最后补 CUDA Stage2
> 与 Store compatibility。简历中的 3.82x 是格式容量，0.664842 ms/1.66x 是
> V7 Full 的实测结果。

回答时明确三个边界：TurboQuant/Lloyd-Max 是上游算法；`reference/`、`vllm/`
是上游快照；个人贡献主要落在 `cuda/`、`benchmarks/`、`validation/` 及对应
分析。这样既能说明工作量，也不会把上游代码归为原创。

### 62. 这个项目最大的技术难点是什么？

一是数学正确性：四个 KV group 不能为了填满 WMMA 而错误拼成 dense GEMM；
二是数值正确性：layout 错误可能保持 LSE 正常却破坏最终 output；三是性能
归因：反量化、Tensor Core、Shared Memory、barrier 和寄存器互相制约。

真正困难的是同时保证量化语义、Attention 语义和 GPU 映射正确，再通过公平
实验判断哪一项优化确实转化为性能。

可以选择 V8 作为最有代表性的难点展开：

1. **数学约束**：不同 KV group 不能为了填满 16 行而混算；
2. **映射设计**：把合法 group 内 QK/PV 转置到 `m16n8k16` 的 N 维；
3. **底层实现**：按 PTX lane contract 手工打包 half2/register；
4. **正确性**：验证四个 Q 列及 128 output dimension 映射；
5. **性能证据**：V7→V8 从 0.631122 降到 0.513208 ms，同时 shared 和 static
   HMMA 下降；
6. **限制**：仍只有 50% slot，并且绑定 `sm_89`。

这样的回答同时展示数学、CUDA 和实验能力，比只说“Tensor Core 映射比较难”
更可核验。

### 63. 如果面试官问“为什么不用现成 FlashInfer”，怎么回答？

FlashInfer 是标准 KV Cache Attention 的高性能实现和重要参考，但当前输入是
TurboQuant 特定的 4-bit index、centroid/norm 和 scale/zero layout，不能直接
当作普通 FP16 paged KV Cache 传入。先完整反量化再调用 FlashInfer 会增加
Global Memory traffic。

因此项目保留量化专用 decode，并迁移适合的执行思想，例如 register-resident
softmax state。未来也可以把 TurboQuant dequant iterator 接入更通用的
FlashInfer 调度框架，但需要处理数据布局和模板接口。

可以从接口不匹配解释，而不是贬低现有库：

- FlashInfer 的成熟调度、paged attention 和 register-state 思路值得复用；
- 当前 cache 是 K centroid index/norm 与 V affine index/scale/zero，不是普通
  FP16/BF16 K/V；
- 直接调用前若必须全量解压，会增加每 slot 512 B 写回和后续重读；
- 本项目验证的是量化专用 load/decode 与 attention compute 的融合方式；
- 长期可以给 FlashInfer 增加 quantized iterator/epilogue，而不是重复实现整个
  调度框架。

V9 的准确表述是“借鉴 FlashInfer 的 register-resident state 思路”，不是“基于
FlashInfer Kernel”或“调用 FlashInfer API”。

### 64. 如果换成 RTX 3090，结果会一样吗？

不会。3090 是 `sm_86`，4090 是 `sm_89`，二者的 SM 数量、时钟、缓存、显存
带宽和调度行为不同。V8/V9 还显式依赖 `sm_89` inline PTX/fragment contract。
移植时需要重新编译、验证 lane mapping、检查 SASS、重跑 correctness 和性能，
不能只修改编译架构字符串后沿用 4090 数字。

迁移检查表包括：

1. 改为 `sm_86` 编译并确认 `m16n8k16` PTX/SASS 支持；
2. 重新验证 WMMA/inline PTX lane mapping；
3. 检查 ptxas registers、Shared Memory、spill 和 theoretical occupancy；
4. 根据 3090 的 SM、L2、带宽重调 split 和 CTA 参数；
5. 重跑 Store、Stage1、Stage2 和 Full correctness；
6. 重采 CUDA Event 与 NCU，不能拿旧 3090/错误 Triton V2 报告代替。

可移植不等于性能可移植：即使结果正确，最优 tile、occupancy 和瓶颈仍可能改变。

### 65. 面试时如何证明这是工程优化，不是只调参数？

应展示完整证据链：先固定 workload 和 reference；用版本隔离单一变化；每版
验证 partial output、LSE 和最终 output；再用 CUDA Event 看稳定收益，用
cuobjdump 检查 register/shared/HMMA/barrier；对 V8 还要解释为什么新的 MMA
映射数学上成立，对 V9 解释为何减少一次 shared round trip。

比起背诵“用了 Shared Memory、Tensor Core、FlashInfer”，这种从问题、假设、
实现、证据到限制的闭环更能体现性能工程能力。

可以准备一张固定证据表：

| 优化 | 瓶颈假设 | 主要修改 | 证据 |
|---|---|---|---|
| V5 direct write | output shared round trip 昂贵 | fragment 直接写 `mid_o` | 0.845486 ms，shared下降 |
| V6 packed load | byte decode 指令多 | `uint32_t` + `half2` | 0.638863 ms |
| V7 barrier | tile 边界同步冗余 | 合并生命周期 barrier | 0.631122 ms，BAR下降 |
| V8 MMA shape | GQA-4 padding浪费 | `m16n8k16` 转置 | 0.513208 ms，HMMA/shared下降 |
| V9 state fusion | `qk_s` 往返昂贵 | register softmax state | 0.484516 ms，shared/BAR下降 |

每一行还必须配同 cache correctness。若面试官要求复现，应能指出 source diff、
benchmark command 和 output/LSE 结果，而不是只展示最终时间表。

## 随机面试深挖题

这一组故意不按知识章节排列，用于模拟面试官连续追问。回答时先给结论，再根据
面试官反应补数学推导或 CUDA 细节。

### 66. TurboQuant 为什么要在量化前对 K 做正交旋转？不旋转会有什么问题？

原始 K 的能量可能集中在少数坐标，不同坐标的尺度和尾部也可能相差很大。如果
直接用一套 4-bit scalar quantizer，少量异常值会迫使量化范围变宽，使大多数
普通坐标只能使用很少的有效量化级，内积误差也会集中在这些高能量方向。

TurboQuant 先归一化 K，再用正交变换把能量分散到各个坐标。正交变换保持范数，
并且只要 Q 使用匹配变换，就保持精确内积。这样旋转后坐标具有统一、可预测的
边缘分布，适合使用一套固定的 Lloyd-Max codebook。旋转并没有消灭量化误差，
而是让有限的 16 个量化级被更均衡地使用。

### 67. “向量在单位球面上均匀分布”是否表示每个坐标服从 Uniform distribution？

不是。“球面均匀”指向量的方向相对于旋转不偏向任何方向，不是每个坐标都在
`[-1, 1]` 上均匀分布。若 $Y$ 均匀分布在 $d$ 维单位球面 $S^{d-1}$ 上，单个
坐标 $Y_i$ 的密度为：

$$
f(t)=\frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)}
     (1-t^2)^{(d-3)/2},\qquad -1\le t\le 1.
$$

高维时密度强烈集中在 0 附近，显然不是平坦的 Uniform density。各坐标也不
严格独立，因为始终满足 $\sum_i Y_i^2=1$。

### 68. TurboQuant 中单个坐标的精确分布是什么？为什么可近似为 $N(0,1/d)$？

对 Haar 随机正交旋转后的单位向量，有：

$$
Y_i^2\sim \mathrm{Beta}\left(\frac12,\frac{d-1}{2}\right).
$$

等价地，平移后的变量满足：

$$
\frac{Y_i+1}{2}\sim
\mathrm{Beta}\left(\frac{d-1}{2},\frac{d-1}{2}\right).
$$

由球面对称性，$E[Y_i]=0$；又因为 $\sum_iY_i^2=1$ 且所有坐标地位相同，
$E[Y_i^2]=1/d$。当 $d$ 增大时，$\sqrt dY_i$ 依分布趋近 $N(0,1)$，所以
$Y_i\approx N(0,1/d)$。这里说的是边缘分布近似；有限维坐标之间仍受单位范数
约束，不能说成严格独立。

### 69. vLLM 会从真实 KV Cache 采样数据来生成 codebook 吗？

不会。vLLM 的 centroid 生成脚本直接把旋转后坐标近似为
$N(0,1/d)$，根据这个理论概率密度做数值积分并迭代 Lloyd-Max 方程。它不是先
收集几十万条真实 K，再运行 K-means，也不需要为每个模型做数据 calibration。

具体过程是：给定 bit 数得到 $2^b$ 个 centroid，反复用相邻 centroid 中点更新
decision boundary，再用每个区间中的条件期望更新 centroid，直到收敛。需要
区分的是，论文的分布论证基于随机正交旋转；当前 vLLM 工程路径使用归一化
Sylvester Hadamard 变换，并利用对称量化器省略随机符号翻转。

### 70. `head_dim = 128` 时 Gaussian approximation 的方差和标准差是多少？

$$
\mathrm{Var}(Y_i)=\frac1{128}=0.0078125,
\qquad
\sigma=\frac1{\sqrt{128}}\approx0.0883883.
$$

不要把这里的坐标标准差与 Attention 的缩放因子混为一个 metadata。二者数值
都可能出现 $1/\sqrt{128}$，但前者用于构造理论分布，后者用于缩放 QK logits。

### 71. 为什么 4-bit 正好需要 16 个 centroid？运行时保存什么？

一个 4-bit 无符号 index 有 $2^4=16$ 种取值，因此 codebook 包含 16 个
centroid。Store 时，每个旋转后坐标通过 15 个 decision boundary 落入一个
区间，最终保存的是 `0..15` 的 centroid index，而不是 centroid 浮点值。

两个 index 打包进一个 byte，所以 128 个坐标占 64 B。Decode 时取出 nibble，
执行 `centroid[index]`，再乘 K 的校正 norm，恢复用于 QK 累加的近似坐标。

运行时涉及三个不同存储层次：

- cache：每个坐标只存 4-bit index；
- global/device 参数：整个 launch 传入一张 `[16]` FP32 centroid table，共 64 B；
- warp register：CUDA V4–V9 让 lane 0–15 各持有一个 centroid，通过 shuffle lookup。

15 个 boundary 只在 Store bucketize 时使用，不写进每个 cache slot，也不在 Decode
再次比较。4-bit 决定的是 index 信息量，不表示 centroid 本身也以 4 bit 保存。

### 72. Lloyd-Max 中相邻 centroid 的 decision boundary 怎么计算？

在标量平方误差准则下，第 $i$ 和第 $i+1$ 个 centroid 之间的边界是二者中点：

$$
b_i=\frac{c_i+c_{i+1}}{2},\qquad i=0,\ldots,14.
$$

因为在这个位置有 $(x-c_i)^2=(x-c_{i+1})^2$。运行时 bucketize 只需将输入和
这 15 个 midpoint 比较，就能得到 4-bit index。

推导时展开平方项：

$$
(x-c_i)^2=(x-c_{i+1})^2
\Longrightarrow 2x(c_{i+1}-c_i)=c_{i+1}^2-c_i^2,
$$

在 $c_i\ne c_{i+1}$ 时得到 $x=(c_i+c_{i+1})/2$。这个中点结论依赖平方误差和
相同误差权重；若目标改成绝对误差或带权 distortion，最优 boundary/representative
条件会变化。

### 73. Lloyd-Max 的新 centroid 为什么是区间条件均值而不是区间中点？

固定量化区间 $[a,b]$ 后，要选择重建值 $c$ 最小化区间内期望平方误差：

$$
J(c)=\int_a^b(x-c)^2f(x)dx.
$$

令导数为零：

$$
\frac{dJ}{dc}=-2\int_a^b(x-c)f(x)dx=0,
$$

得到：

$$
c=\frac{\int_a^bxf(x)dx}{\int_a^bf(x)dx}=E[X\mid a\le X\le b].
$$

只有当区间内概率密度关于中点对称或近似常数时，它才等于 $(a+b)/2$。Gaussian
在尾部区间明显不均匀，因此简单取几何中点通常不是 MSE 最优重建值。

### 74. TurboQuant 为什么可以使用固定 codebook，而不需要模型级 calibration？

归一化去除了每个 K 向量的整体尺度，正交混合又使坐标边缘分布接近只由维度
$d$ 决定的球面坐标分布。于是 codebook 可以针对理论近似
$N(0,1/d)$ 离线求解，而不是针对某层、某模型的经验直方图求解。

这是 TurboQuant 的设计优势，不代表所有真实数据都精确服从 Gaussian，也不
代表固定 codebook 在任何任务上都必然优于校准量化。工程上仍需用模型质量和
下游任务评估验证这种理论近似。

固定 codebook 成立依赖几个前提：

1. 每个 K 先按自己的向量 norm 归一化；
2. 使用匹配的正交/Hadamard 变换充分混合坐标；
3. head dimension 足够高，使 Gaussian approximation 可用；
4. 量化目标主要是旋转坐标的 MSE；
5. 模型实际分布没有严重偏离理论假设。

优势是无需 per-model calibration，部署简单且各层共享数值 codebook；风险是理论
近似不能自动保证 PPL。应把“无需 calibration”与“无需质量验证”明确区分。

### 75. 一个 FP16 Key 从输入到写入 4-bit KV Cache 经历什么？

以 $K\in R^{128}$ 为例，主要数据流是：

1. 以 FP32 累加计算原始二范数 $s=\lVert K\rVert_2$；
2. 归一化得到 $u=K/s$，并处理极小范数的数值边界；
3. 用正交矩阵或归一化 Hadamard 变换得到 $z=\Pi u$；
4. 用 15 个 midpoint 对每个 $z_i$ bucketize，得到 128 个 4-bit index；
5. 将 index 查回的 centroid 组成 $c$，计算其量化后范数 $\lVert c\rVert_2$；
6. 开启 norm correction 时保存 $\gamma=s/\lVert c\rVert_2$；
7. 每两个 index 打包为一个 byte，按 paged KV Cache 布局写入 64 B K payload；
8. 将 FP16 `gamma` 写入该 token/KV-head 的 metadata。

Decode 读出的近似旋转 K 是 $\hat K_r=\gamma c$。本项目随后直接在 tile 内参与
QK，不生成完整的 Global Memory FP16 K buffer。

### 76. K 的 `norm` 何时计算？它与 centroid 是什么关系？

原始 norm $s=\lVert K\rVert_2$ 在 Store/量化阶段、归一化之前计算。Lloyd-Max
centroid 是离线根据目标分布生成的一套全局常量，不由这个 norm 生成。

开启 norm correction 后，实际保存的标量还会结合量化 centroid 向量的范数：

$$
\gamma_{stored}=\frac{\lVert K\rVert_2}{\lVert c\rVert_2}.
$$

所以 centroid index 描述方向，保存的 norm 类 metadata 恢复幅值并补偿量化后
方向向量的范数偏差。面试时不能把它说成 Lloyd-Max 的 scale 参数。

### 77. `norm`、`scale` 和 `QJL` 是不是同一个东西？

不是，它们处在不同路径并解决不同问题：

| 名称 | 所在路径 | 作用 |
|---|---|---|
| K norm / corrected norm | K centroid quantization | 恢复 K 的整体幅值，并可补偿 centroid 向量范数 |
| V scale/zero | V affine quantization | 将 `0..15` 的 uniform index 映射回 V 的动态范围 |
| QJL residual channel | 论文 TurboQuant-Prod | 用随机投影符号和 residual norm 估计残差内积 |

QJL 不是一个浮点 scale。当前本项目不含 QJL 的 4-bit 路径只保存 K norm、
V scale 和 V zero，没有 QJL residual payload。

### 78. TurboQuant-MSE 优化什么？TurboQuant-Prod 为什么引入 residual？

TurboQuant-MSE 选择标量量化器来最小化旋转坐标的重建均方误差，目标可写成：

$$
E\left[\lVert x-\hat x_{mse}\rVert_2^2\right].
$$

但 Attention 真正关心的是 query 与 key 的内积。即使 $\hat x_{mse}$ 已很好地
重建 x，残差 $r=x-\hat x_{mse}$ 仍会产生 $q^Tr$，从而扰动 logits。
TurboQuant-Prod 因此通常让主 MSE 通道使用 $b-1$ bit，并用额外 1 bit 的 QJL
通道编码残差信息，目标更直接地降低或校正内积估计误差。

### 79. TurboQuant-Prod 的 residual 如何处理？QJL 起什么作用？

先计算主量化结果和残差：

$$
r=x-\hat x_{mse}.
$$

QJL 使用随机投影得到 residual 的符号信息，并配合 residual norm 等 metadata，
在查询时构造 $q^Tr$ 的低成本、无偏估计。最终内积由主通道和残差估计相加：

$$
q^Tx\approx q^T\hat x_{mse}+\widehat{q^Tr}.
$$

因此 QJL 是 residual inner-product estimator，不是对 K 或 V 乘一次的普通
scale，也不是 Lloyd-Max codebook 本身。

### 80. 论文有 QJL，为什么当前 vLLM decode 可以不用？

论文给出了包括 TurboQuant-Prod/QJL 在内的算法设计，但工程实现可以选择不同
质量、显存和吞吐折中。当前 vLLM 的 TurboQuant backend 使用旋转、K centroid
index/norm 与 V uniform scale/zero，不把 QJL residual 放进当前 decode cache
contract。

vLLM 文档还说明 QJL 估计方差会影响 softmax attention 的质量，因此当前实现
有意省略该通道。这样 cache 布局和 decode 更简单、确定性更强，但不能把当前
vLLM 路径描述为完整实现了论文所有变体。本项目跟随的也是不含 QJL 的路径。

### 81. K 和 V 是否使用完全相同的量化方法？

不是。K 路径是：归一化、正交/Hadamard 旋转、Lloyd-Max 非均匀 centroid
quantization，并保存 4-bit index 和 corrected norm。V 路径通常不做这套
centroid rotation，而是按向量求 `vmin/vmax`：

$$
scale=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right),
\qquad zero=v_{min},
$$

$$
q=\mathrm{clip}\left(\mathrm{round}
\frac{v-zero}{scale},0,15\right),
\qquad \hat v=q\cdot scale+zero.
$$

本项目每个 token/KV-head 保存 64 B K index、64 B V index，以及三个 FP16
metadata：K corrected norm、V scale、V zero，总计 134 B。

### 82. 为什么 K 适合 centroid quantization，而 V 可用 affine quantization？

K 进入 QK 内积并进一步进入 softmax，logit 误差可能改变整行 attention weight；
归一化和旋转后，K 坐标又具有可利用的稳定、近 Gaussian 分布，因此用针对该
分布优化的非均匀 centroid 有明确动机。

V 在 softmax 权重确定后参加加权和。工程上可以按每个 V 向量的实际
`min/max` 使用简单 affine quantization，Decode 只需一次乘加，成本较低。
这是一项算法与实现折中，不应表述为“V 对误差不敏感”或“V 永远不值得做
非均匀量化”；最终仍要通过输出质量评估决定。

### 83. K 被旋转后，为什么 Query 也必须做相应旋转？

使用行向量约定，令：

$$
K_r=K\Pi^T,\qquad Q_r=Q\Pi^T,
$$

其中 $\Pi$ 为正交矩阵，即 $\Pi^T\Pi=I$。那么：

$$
Q_rK_r^T
=Q\Pi^T(K\Pi^T)^T
=Q\Pi^T\Pi K^T
=QK^T.
$$

如果只旋转 K 而不旋转 Q，计算的是 $Q\Pi K^T$ 或其对应约定形式，不再等于
原始 attention score。代码采用列向量时左右乘形式会变化，但“Q/K 必须进入
同一个正交坐标系”这一结论不变。

### 84. 为什么 decode 可以直接从 index lookup 进入 QK accumulation？

QK 只需要逐坐标使用近似 K，并不要求先拥有一个完整、连续、长期存在的 FP16
K tensor。因此 Kernel 可以在 tile 内执行：nibble unpack、centroid lookup、
乘 corrected norm，然后立即送入 Tensor Core 或普通 FMA 累加。

最大的收益是避免把完整 FP16 K 写回 Global Memory 后再读一次，同时避免额外
dequant Kernel、临时显存和 launch 边界。Shared Memory 仍可作为 tile staging，
但解码后的数据只在 CTA 内短暂存在。V 也可类似地用 `index*scale+zero` 后直接
进入 PV。

### 85. 4096 个历史 token 后生成新 token，Q 是否和自己的 K 做 attention？

要先明确“4096 个历史 token”是否包含当前 decode position。标准 causal
self-attention 对位置 $t$ 允许访问所有 $j\le t$，因此当前位置自己的 K/V
应当参与 attention；只屏蔽未来位置 $j>t$。

工程上常在该层计算当前 token 的 Q/K/V，将当前 K/V 写入 cache，再让 Q 读取
长度为 `context_len` 的有效 cache。如果 `4096` 表示写入当前 K 后的有效长度，
QK 有 4096 个 K；如果严格表示此前已有 4096 个旧 K，随后又追加当前 K，则有
4097 个。不要只凭“正在生成第几个输出 token”判断，应该检查 API 中
`context_len/seq_len` 的定义和 cache append 顺序。

### 86. KV Cache 从 FP16 降到 4-bit，为什么 decode 不一定严格加速 4 倍？

4 倍主要是 K/V payload 字节数的理论缩减，不是整个 Kernel 时间的缩减。实际
路径还包含 metadata、page-table 访问、nibble unpack、centroid lookup、V
反量化、QK/PV、online softmax、同步和 Stage2。固定 launch、调度与计算开销
不会按 cache bit 数同步缩小。

此外，压缩后瓶颈可能从 DRAM 转向 Tensor Core、MIO、整数流水线、dependency
stall 或 occupancy。最终 speedup 受 Amdahl 定律和新瓶颈约束，必须用同 workload
实测，不能从 `16/4` 直接宣布 4x 端到端加速。

### 87. centroid lookup 增加指令，为什么仍可能比 FP16 KV Cache 快？

Decode 长上下文通常需要搬运大量 KV 数据，而 16-entry codebook 很小、可被缓存
或放入适合的只读存储。用少量 unpack、lookup 和乘法换取约 4 倍 payload 流量
下降，在 memory-bound 场景中通常是有利的典型“用计算换带宽”。

是否获益取决于 lookup 的实现、cache 命中、指令依赖和原 Kernel 的瓶颈。如果
上下文很短或实现造成严重 serialization，额外指令可能超过流量收益，所以仍需
benchmark，而不是把这种权衡当作无条件结论。

本项目进一步避免了真正的随机 global gather：每个 warp 的 lane 0–15 先各持有
一个 centroid，nibble index 作为 `__shfl_sync` 的 source lane。于是 hot loop
的 lookup 主要消耗 shuffle、整数位运算和依赖延迟，而不是 128 次独立 table
memory load。

因此判断收益要同时比较：减少的 KV bytes、增加的 integer/shuffle 指令、register
压力、eligible warps 和 long/short-scoreboard。codebook 只有 64 B 只能说明容量
很小，不能说明 lookup 链路必然免费。

### 88. DRAM Throughput 不高但 L1/TEX 或 MIO 很高，应该如何解释？

这说明 Kernel 不一定受 DRAM 带宽上限约束。4-bit payload 已减少 DRAM 流量，
但 unpack、类型转换、centroid table 读取、Shared Memory load/store 或特殊数据
通路可能让 L1/TEX、MIO 相关流水线变成更紧的瓶颈。大量短依赖链还可能让吞吐
指标高但 eligible warp 不足。

应联合检查 NCU 的 Speed-of-Light、Memory Workload Analysis、Scheduler Stats
和 Warp Stall Sampling，例如 L1/TEX sectors、MIO throttle、long scoreboard、
short scoreboard、barrier、issue active 和 achieved occupancy。不能因为 DRAM
百分比低就简单得出“内存完全不是问题”，也不能只凭一个 MIO 指标下结论。

### 89. `centroid[index]` 这种数据相关 lookup 会给 GPU 哪些部分带来压力？

可能的压力包括：

- nibble 提取、位运算和地址计算占用 integer pipeline；
- index 到地址再到数据形成 load-use dependency，增加 scoreboard latency；
- table 放置不当时产生 constant-memory divergence 或额外 L1/TEX load；
- 解码值、metadata、MMA fragment 和 softmax state 同时存活，抬高 register 压力；
- staging 到 Shared Memory 时增加 MIO 指令、bank conflict 风险和同步；
- register 增长可能降低 resident CTA，严重时还会 spill 到 local memory。

16-entry table本身很小，不代表 lookup 一定免费。应比较 constant memory、
read-only cache、Shared Memory 或寄存器化方案，并用 SASS 与 NCU 判断真实瓶颈。

### 90. “TurboQuant 不就是 INT4 KV Cache 吗？”如何用 30 秒回答？

可以回答：

> 它确实把主要 K/V payload 存成 4-bit index，但不是普通的线性 INT4 KV
> quantization。K 会先按向量归一化并做正交或 Hadamard 旋转，使坐标分布稳定，
> 再用针对该分布离线求出的 16 个 Lloyd-Max centroid 做非均匀量化；cache
> 保存 centroid index 和 corrected norm。Query 也做匹配旋转，从而保持 QK
> 内积语义。V 才使用更常见的 per-vector affine 4-bit scale/zero。Decode 时
> 把 lookup/反量化直接融合进 attention，避免恢复完整 FP16 KV Cache。

若继续追问，再补充：论文还有 TurboQuant-Prod/QJL residual 变体，但当前 vLLM
和本项目采用的不含 QJL 路径，不能把论文全部能力混到当前 Kernel 描述里。

## 简历逐句拷问与硬件 Roofline

### 91. 请用一分钟介绍简历中的 TurboQuant CUDA Decode 项目

可以回答：

> 这个项目针对 LLM decode 阶段 KV Cache 访存开销大、TurboQuant 原始 Triton
> 实现存在额外中间张量和融合空间的问题。我基于 vLLM 的 `4bit_nc` cache 格式，
> 为 RTX 4090、Qwen3-4B 形状实现了 CUDA Tensor Core Stage1 Kernel，把 4-bit
> K/V unpack、K 的 Lloyd-Max centroid lookup、GQA、online softmax 和 V
> 反量化融合在一个 launch 中，并保留 split-KV Stage2 做跨 split 归并。
> 优化过程从 CTA 数据复用、warp-per-Q、WMMA、固定形状特化，一直推进到
> fragment 寄存器直写、32-bit 对齐加载、barrier 消减和寄存器常驻 softmax
> 状态。固定工作负载下，V7 的完整 Decode 链路是 `0.664842 ms`，相对 SoA
> Triton V2 的 `1.104148 ms` 提速约 `1.66x`；V9 的 Stage1 单独达到
> `0.484516 ms`。KV slot 从 FP16 K/V 的 512 B 降到 134 B，压缩约 `3.82x`。

回答结束后主动限定口径：这是固定 shape 的研究型 benchmark，不包含 Query
rotation、Store/QKV projection/RoPE，也没有宣称已经替换生产 vLLM backend。

### 92. RTX 4090 的 FLOPS、显存带宽和与你 Kernel 相关的规格是多少？

NVIDIA 官方 Ada 白皮书给出的 RTX 4090 关键规格是：

| 指标 | 数值 | 在本项目中的含义 |
|---|---:|---|
| 架构 / Compute Capability | Ada / `sm_89` | 编译目标和可用指令集 |
| SM | 128 | 决定并行 CTA 的硬件承载能力 |
| CUDA Core | 16,384 | 普通 FP32/整数等流水线资源 |
| Tensor Core | 512 | WMMA/HMMA 的核心执行资源 |
| Boost Clock | 2.52 GHz | 峰值计算能力的理论时钟基础 |
| FP32 非 Tensor 峰值 | 82.6 TFLOPS | 普通 FP32 FMA 的理论上限 |
| FP16 Tensor，FP32 accumulate，dense | 165.2 TFLOPS | 本项目 WMMA 更相关的理论峰值 |
| FP16 Tensor，sparse | 330.4 TFLOPS | 需要结构化稀疏，本项目不能使用 |
| 显存 | 24 GB GDDR6X | KV Cache 容量上限的一部分 |
| 显存位宽 / 速率 | 384 bit / 21 Gbps | 理论带宽来源 |
| 显存理论带宽 | 1008 GB/s | Roofline 的 DRAM 带宽上限 |
| L2 Cache | 72 MiB | 重复读取和工作集驻留的重要层级 |

面试时最重要的是说明：本 Kernel 使用 dense FP16 Tensor Core MMA，所以相关
峰值是约 `165.2 TFLOPS`，不能拿 sparse 峰值或营销中的 PetaOPS 当分母。

### 93. 为什么 RTX 4090 会同时出现 82.6、165.2、330.4 TFLOPS 和 1.321 PetaOPS？

因为它们对应不同数据类型和稀疏条件：

- `82.6 TFLOPS`：普通 CUDA Core 的 FP32 dense 吞吐；
- `165.2 TFLOPS`：Tensor Core FP16 输入、FP32 accumulate 的 dense 吞吐；
- `330.4 TFLOPS`：同类 Tensor 运算开启 2:4 structured sparsity 后的标称吞吐；
- `1.321 PetaOPS`：更低精度并叠加稀疏条件的营销指标。

项目的 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 属于 dense FP16
Tensor 运算。因此 Roofline 应使用 165.2 TFLOPS。混用峰值会让计算利用率和
ridge point 错一倍甚至更多。

### 94. `sm_89` 是什么意思？为什么不是说“专门使用 4090 指令”？

`sm_89` 是 CUDA 对 Ada GeForce GPU Compute Capability 8.9 的编译目标。它定义
可生成的 SASS/PTX 能力、资源约束和架构特性，但不是 RTX 4090 独占标识；其他
Compute Capability 8.9 GPU 也可能运行同一 cubin。项目进一步针对 4090 的固定
SM 数量、带宽、cache 和 benchmark shape 调优，因此“为 4090 调优”和“代码只
能在 4090 执行”是两回事。

### 95. `4bit_nc` 中的 `nc` 是什么？

`nc` 是 **norm correction**，不是 non-contiguous，也不是某种 CUDA Core 模式。
对一个 Key 向量，先记原向量范数为

$$
n = \lVert K \rVert_2.
$$

将单位化并旋转后的坐标量化到 centroid 后，重建向量的范数一般不再严格为 1。
因此保存修正系数

$$
\gamma = \frac{\lVert K \rVert_2}
{\lVert \hat{u} \rVert_2},
$$

其中 $\hat{u}$ 是 centroid lookup 后的量化方向。Decode 中重建的是
$\hat{K}=\gamma\hat{u}$。每个 token/KV head 保存一个 FP16 corrected norm；这
既恢复原始幅值，也修正 centroid 量化造成的范数漂移。

### 96. 简历中的 KV Cache 压缩 `3.82x` 是怎么计算的？

固定 `head_dim=128`，每个 token、每个 KV head 的 FP16 K/V payload 是：

$$
128\times 2\ \text{B} + 128\times 2\ \text{B}=512\ \text{B}.
$$

`4bit_nc` slot 包含：

| 字段 | 字节数 |
|---|---:|
| K 的 128 个 4-bit index | 64 B |
| V 的 128 个 4-bit index | 64 B |
| K corrected norm，FP16 | 2 B |
| V scale，FP16 | 2 B |
| V zero，FP16 | 2 B |
| 合计 | 134 B |

所以

$$
\text{compression ratio}=\frac{512}{134}=3.8209\times.
$$

它不是严格 4 倍，因为每个 slot 还有 6 B metadata。这个数字只计算 K/V cache
主体，不代表模型全部显存占用也下降 3.82 倍。

### 97. 为什么说工作负载是“Qwen3-4B 形状”，而不是完整跑了 Qwen3-4B？

官方 Qwen3-4B 配置中的 attention 形状是 `32` 个 Q head、`8` 个 KV head、
`head_dim=128`，即 GQA ratio 为 4。本 benchmark 采用相同的 head shape，但
`batch=64`、`context=4096`、`num_splits=32` 是项目自行固定的测试参数，并未
加载完整模型权重或执行全部 Transformer layer。

严谨说法是“Qwen3-4B-shaped fixed workload”。如果说“Qwen3-4B 端到端推理
提速 1.66x”，会把单层 attention decode Kernel 结果错误外推到整个模型。

### 98. `0.664842 ms` 的 Full Decode 到底包含和不包含什么？

它包含：

1. V7 Stage1：读取压缩 K/V、centroid lookup/反量化、QK、online softmax、
   每个 split 的 partial output 和 LSE；
2. Stage2：按 split LSE 做稳定归一化并归并 partial output。

它不包含 Store Kernel、Query rotation、QKV projection、RoPE、采样、模型其他
层、内存分配和 JIT 编译。所谓 `Full Decode` 是本项目 benchmark 中 Stage1 +
Stage2 的完整 attention decode 链，不是完整 token generation latency。

### 99. “融合至单次 Stage1 launch”具体融合了什么？为什么有价值？

单个 Stage1 CTA 内完成：

```text
加载 packed K/V
  -> 提取 4-bit nibble
  -> K centroid lookup + corrected norm
  -> V affine 反量化
  -> GQA QK Tensor Core MMA
  -> online softmax 状态更新
  -> probability × V 累加
  -> 写出 split partial output 和 LSE
```

价值不是单纯“少几个 Kernel 名字”，而是避免把完整 FP16 K/V、score 或
probability 中间张量写回 global memory，降低 launch 开销并让解码结果尽可能
停留在寄存器/Shared Memory 中。Stage2 仍是另一个 launch，因为不同 CTA 的
split 结果需要全局同步后才能归并。

### 100. 固定工作负载下会启动多少 CTA？每个 CTA 做什么？

Stage1 grid 是

$$
(B,H_{kv},S)=(64,8,32),
$$

因此 CTA 数量是

$$
64\times 8\times 32=16{,}384.
$$

每个 CTA 有 128 threads，即 4 个 warp，负责一个 batch、一个 KV head、一个
split，以及该 KV head 对应的全部 4 个 Q head。每个 split 有 128 token，按
16-token tile 处理 8 轮。这个映射让同一组 K/V 被 4 个 Q head 复用。

### 101. Qwen3-4B 固定 workload 的有效计算量是多少？

只统计算法上必要的 QK 和 PV FMA，并按一次 FMA 等于 2 FLOPs：

$$
\begin{aligned}
F_{QK} &= 2BH_qLD,\\
F_{PV} &= 2BH_qLD,\\
F_{useful} &= 4BH_qLD.
\end{aligned}
$$

代入 $B=64,H_q=32,L=4096,D=128$：

$$
F_{useful}=4\times64\times32\times4096\times128
=4{,}294{,}967{,}296\ \text{FLOPs},
$$

即约 `4.295 GFLOPs`。这不包含 unpack、lookup、地址计算、softmax exp、归约等
非 FMA 指令，因此只是 useful FLOPs，不是完整指令成本。

### 102. Stage1 逻辑访存量大约是多少？

按每个逻辑数据只计算一次、不推断 cache hit/miss：

| 数据 | 估算 |
|---|---:|
| 压缩 K/V cache | 281,018,368 B = 268 MiB |
| Q 被每个 split CTA 重读 | 33,554,432 B = 32 MiB |
| Stage1 `mid_o` 写出 | 33,816,576 B = 32.25 MiB |
| block table | 524,288 B = 0.5 MiB |
| seq-len 等小 metadata | 约 0.06 MiB |
| 合计 | 348,979,200 B = 332.812 MiB |

这里的 Q 重读来自 `(B,Hkv,split)` CTA 各自加载对应 4 个 query head；`mid_o`
是 32 个 split 的 FP32 partial output。该估算称为 logical traffic，实际 DRAM
bytes 会被 L2 hit、write policy、transaction 放大和重放改变。

### 103. 这个 Kernel 的算术强度大约是多少？

有两个有用口径。

只看每个 token/KV-head 的压缩 cache，4 个 GQA query head 对 K 做 QK、再对 V
做 PV，useful FLOPs 是 $4GD=4\times4\times128=2048$，读取 134 B，因此：

$$
AI_{cache}=\frac{2048}{134}=15.28\ \text{FLOP/B}.
$$

若把 Q 重读、`mid_o` 写出、block table 等完整 Stage1 逻辑流量计入：

$$
AI_{stage1}=\frac{4.294967\ \text{GFLOP}}
{348.9792\ \text{MB}}=12.31\ \text{FLOP/B}.
$$

面试时先声明分子是 useful FLOPs、分母是 logical bytes。不同统计口径不可直接
与 NCU 的硬件 counter 混用。

### 104. V8/V9 的 useful FLOPs 与 Tensor Core 实际执行 FLOPs为什么不同？

V8/V9 使用 `m16n8k16` MMA，但当前 GQA group 只有 4 个 Q head，因此 N 方向
8 个槽位中只有 4 个有数学意义，slot utilization 为 50%。硬件仍执行完整 MMA，
所以估算的 Tensor FLOPs 是：

$$
F_{executed}\approx \frac{F_{useful}}{0.5}=8.590\ \text{GFLOPs}.
$$

报告算法性能时用 useful FLOPs；对比 165.2 TFLOPS Tensor 峰值时应使用
executed FLOPs。若拿 4.295 GFLOPs 除以 Tensor 峰值，会把空槽成本隐藏掉。

### 105. V9 的有效吞吐和逻辑带宽是多少？应该怎么表述？

V9 Stage1 时间为 `0.484516 ms`：

$$
\text{useful throughput}=\frac{4.295\ \text{GFLOP}}{0.484516\ \text{ms}}
=8.86\ \text{TFLOP/s},
$$

$$
\text{executed Tensor throughput}\approx17.73\ \text{TFLOP/s},
$$

$$
\text{logical bandwidth}=\frac{348.9792\ \text{MB}}{0.484516\ \text{ms}}
=720.3\ \text{GB/s}.
$$

最后一个数字只能叫“根据逻辑字节估算的有效/逻辑带宽”，不能说 NCU 实测
DRAM bandwidth 是 720 GB/s。两者分母相同，但分子含义不同。

### 106. 怎么用 Roofline 判断它偏 compute-bound 还是 memory-bound？

RTX 4090 对本 Kernel 的理论 ridge point 约为：

$$
AI_{ridge}=\frac{165.2\ \text{TFLOP/s}}{1008\ \text{GB/s}}
=163.9\ \text{FLOP/B}.
$$

V9 executed 口径的 AI 约 $2\times12.31=24.62$ FLOP/B，仍远低于 163.9。
等价地，理想内存下界约为

$$
348.98\ \text{MB}/1008\ \text{GB/s}\approx0.346\ \text{ms},
$$

理想 Tensor 计算下界约为

$$
8.59\ \text{GFLOP}/165.2\ \text{TFLOP/s}\approx0.052\ \text{ms}.
$$

因此 Roofline 预测它是 memory-side candidate，而不是 Tensor compute-bound。
但实际 `0.4845 ms` 高于内存下界，差值还可能来自 unpack/lookup、地址依赖、
Shared Memory、barrier、低 occupancy 或 transaction efficiency。没有当前 V9
的 NCU counter 时，应说“memory-dominant candidate”，不要断言“纯 DRAM bound”。

### 107. 面试官只给 20 秒，如何严谨回答 compute-bound 还是 memory-bound？

可以回答：

> 我先做静态 Roofline。V9 的完整逻辑算术强度约 12.3 useful FLOP/B；即使按
> 50% MMA 有效槽位折成约 24.6 executed FLOP/B，也显著低于 4090 dense FP16
> Tensor 的 ridge point 163.9 FLOP/B，所以它明显偏 memory side。进一步是否
> 真由 HBM 饱和限制，要看当前版本 NCU 的 DRAM throughput、L2 hit、stall 和
> Tensor utilization；仅凭 Roofline 我不会宣称它是纯 DRAM bandwidth-bound。

### 108. 用 NCU 最终确认瓶颈时，你会看哪些指标？

建议按证据链回答：

| 问题 | 重点指标/现象 |
|---|---|
| DRAM 是否接近上限 | `dram__bytes`、DRAM throughput、read/write sectors |
| L2 是否吸收流量 | L2 hit rate、L2 sectors、L2 throughput |
| Tensor Core 是否忙 | HMMA pipe utilization、Tensor active cycles |
| 是否 latency-bound | long/short scoreboard stall、eligible warps per cycle |
| Shared Memory 是否受限 | bank conflicts、MIO throttle、shared transactions |
| 同步是否过多 | barrier stall、每 CTA barrier 动态次数 |
| occupancy 是否不足 | achieved occupancy、active warps、register/shared limit |
| 是否发生 spill | local load/store、stack frame、ptxas register 信息 |
| load 是否合并 | global load efficiency、sector/request、replay |

先用 Speed-of-Light/Memory Workload Analysis 定位大类，再下钻到 Scheduler、Warp
State 和 Source/SASS。不能只凭单个“DRAM Throughput %”做结论。

### 109. 为什么 logical bandwidth 不等于 NCU 的 DRAM bandwidth？

`logical bytes / time` 是算法视角：按照张量语义推算应读取和写入多少数据。
NCU DRAM bytes 是硬件视角：只统计真正越过 L2 与显存控制器的数据。二者差异
来自 L2 hit、cache line/sector 粒度、未合并访问、重复 transaction、writeback、
ECC/计数口径等。

逻辑带宽适合跨实现比较“单位有效数据处理速度”；DRAM 带宽用于判断硬件显存
是否饱和。项目当前的 `720.3 GB/s` 属于前者。

### 110. 如果 NCU 显示 DRAM 只有峰值的 55%，还能说它偏 memory-bound 吗？

可以，但必须说明是哪一种 memory-side 限制。低于峰值不自动等于 compute-bound：

- 数据相关 lookup 可能形成 load-use dependency，warp 数不足以隐藏 latency；
- 随机 block-table 映射可能降低 coalescing 或 L2 locality；
- Shared Memory/MIO、bank conflict 或 barrier 可能限制发射；
- unpack 的整数流水线和地址计算可能阻止持续提交 memory request；
- register/shared 限制可能降低 occupancy，造成带宽喂不满。

若 Tensor pipe 同时很低、scoreboard/MIO/barrier stall 很高，结论应是“memory
subsystem 或数据搬运路径受限，但不是纯 DRAM bandwidth saturation”。若 Tensor
pipe 很高且计算下界更接近实测，才转向 compute-bound 判断。

## 性能诊断与场景追问

### 111. context 从 4096 改成 8192，仍设 32 个 split，V9 能直接运行吗？

不能。此时每个 split 是 256 token，而当前 V9 固定要求
`tokens_per_split == 128`，否则 guard 会直接 return。即使删除 guard，循环边界、
tile 数、Shared Memory 生命周期和寄存器状态也需要重新验证，不能把原时间线性
外推。

可选方案是把 split 增至 64 维持每 split 128 token，或者把 Kernel 泛化成每个
split 循环多个 128-token chunk。前者会增加 CTA 数、Q 重读、`mid_o` 和 Stage2
成本；后者增加单 CTA 工作量和状态驻留时间，需要实测选择。

### 112. batch 从 64 降到 1，性能会怎样？

固定其他参数时 CTA 从 16,384 降到

$$
1\times8\times32=256.
$$

4090 有 128 个 SM，平均只有约 2 CTA/SM 的总工作量，Kernel 很快进入并行度和
launch-latency 敏感区。吞吐时间不会简单按 64 倍缩短；如果每个 CTA 又受寄存器、
Shared Memory 或 latency 限制，硬件更难用其他 CTA 隐藏延迟。

优化方向包括增加 split 并行、让一个 CTA 的 warp 更充分、批量合并请求，或为
低 batch 单独设计 persistent/更细粒度映射。但增加 split 会放大 Stage2 和中间
结果成本，所以要针对延迟目标调参。

### 113. `num_splits` 应该怎么选？越多越好吗？

不是。增加 split 的收益是增加 CTA 数、缩短每个 CTA 的序列循环，有利于小
batch/长 context 的并行度。成本是：

- Query 被更多 split CTA 重复读取；
- `mid_o` 和 `mid_lse` 线性增大；
- Stage2 要归并更多 partial state；
- CTA 过短后 launch、初始化和尾部成本占比上升。

调优时应对 `(batch, context, Hkv, D)` 建立候选 split 集合，同时测 Stage1、
Stage2 和 Full，不可只选 Stage1 最快值。生产实现通常需要启发式或 autotune，
而不是固定 32。

### 114. GQA ratio 从 4 改成 8 或 2，会怎样？

对 V8/V9 的 `m16n8k16` 映射：

- `GQA=8` 可以填满 N=8 的 Q-head 槽位，理论有效槽位从 50% 到 100%；但 Q、
  score、softmax 和 output state 翻倍，register/shared 压力也上升；
- `GQA=2` 只有 2/8 槽有效，slot utilization 降到 25%，Tensor 浪费更严重；
- 直接把不同 KV head 的 Q group 拼到同一个 MMA 不正确，因为各组必须乘不同 K，
  后续也对应不同 V。

因此 GQA=8 不保证恰好 2 倍快，GQA=2 也可能需要改用 SIMT、不同 MMA shape 或
一次处理多个独立 tile 的布局。

### 115. `head_dim` 改成 64 或 256，对算术强度和实现有什么影响？

仅按压缩 cache 估算，slot 字节为 $D+6$，useful FLOPs 为 $4GD$，其中 $G=4$：

| D | cache-only AI |
|---:|---:|
| 64 | $1024/70=14.63$ FLOP/B |
| 128 | $2048/134=15.28$ FLOP/B |
| 256 | $4096/262=15.63$ FLOP/B |

维度增大后 metadata 占比降低，所以 AI 略增并趋近 16 FLOP/B，但仍远低于 4090
Tensor ridge point。代码层面当前 V4-V9 对 D=128 有硬编码假设；D=64 会改变 K
循环和线程利用，D=256 会增加 MMA 数、寄存器 output state 和 Shared Memory，
必须重新设计并验证，不能只改常量。

### 116. block size 不是 16，或者 block table 很随机，会发生什么？

当前 block size 16 与 token tile 16 对齐，一个 tile 可自然映射到一个 paged-cache
block，地址计算和协作加载简单。若 block size 不是 16，一个 MMA tile 可能跨
物理 block，需要额外边界判断和两段加载。

即使 block size 仍为 16，随机的 logical-to-physical block table 也会破坏跨 CTA
或连续 tile 的空间局部性，降低 L2 命中和预取效果。单 CTA 内访问仍可 coalesced，
但“coalesced”不等于“具有跨 tile cache locality”。应使用连续映射与随机映射
两组 benchmark，结合 L2/DRAM counters 判断。

### 117. 为什么这个 Decode Kernel 的设计不能直接用于 Prefill？

Decode 通常每个 sequence 只有一个新 Query，QK/PV 形状接近 GEMV，KV 读取占比
高，因此 KV 压缩和 split-KV 很有价值。Prefill 有大量 Query token，QK 是更规则
的大矩阵乘，计算复用和 Tensor Core occupancy 更高，而且需要完整 causal mask、
不同 tiling 与更复杂的 softmax 数据流。

把 decode 的 one-query、warp-per-Q、固定 split 映射直接搬到 prefill，会丢失 Q
维度复用并启动过多 CTA。Prefill 应使用 FlashAttention 类二维 Q/K tiling，再把
TurboQuant 解码融合到 K/V tile load 路径中。

### 118. 项目能否声称比未量化 FP16 FlashAttention 更快？

不能。当前主要公平对比是同一压缩 cache 语义下的 Triton TurboQuant baseline
与 CUDA 实现；仓库没有固定相同 workload、相同输入输出范围并经过验证的 FP16
FlashAttention baseline。

理论上 4-bit cache 显著减少 KV bytes，但引入 unpack、lookup、metadata 和可能
较低的 Tensor slot utilization。最终是否快于成熟 FP16 kernel 必须新增端到端
benchmark 后回答，不能由压缩比推导性能比。

### 119. `3.82x` 压缩是否证明模型精度几乎不下降？

不证明。压缩比只描述存储；Kernel correctness 只证明 CUDA 与参考实现对同一
量化 cache 的数值一致。要证明模型质量，还需要：

1. 在真实模型上执行完整 K/V Store 与 Decode；
2. 测量 attention output、logit 的误差分布；
3. 跑 perplexity 数据集；
4. 跑下游生成/推理任务并与 FP16、其他 KV quant baseline 比较；
5. 覆盖不同 layer、context length 和 outlier 情况。

当前仓库没有完整模型 PPL 或 downstream quality 报告，因此只能陈述 kernel-level
数值验证，不能陈述无损或模型精度结论。

### 120. 向量化加载后性能没有提升，你会怎么排查？

先确认项目实际使用的是每线程一个 32-bit `uint32_t` aligned load，不把它误称为
每线程 `uint4` 128-bit load。然后检查：

- 地址是否真正 4-byte 对齐，warp request 是否合并；
- 编译后 SASS 是否生成预期宽度 load，还是被拆分；
- 总瓶颈是否在 DRAM，若在 lookup/barrier/MMA，加载变宽可能无收益；
- 新实现是否增加寄存器、地址计算或 unpack 指令；
- transaction 数、sector/request、replay 和 long-scoreboard 是否改善；
- cache hit 已很高时，global load 宽度是否不是关键路径。

判断依据是版本间时间与 NCU counter 的共同变化，不是源代码类型名。

### 121. 是否应该继续用 `cp.async` 和双缓冲优化？

它可能有效，但不是默认答案。`cp.async` 适合把 global-to-shared 的下一 tile 加载
与当前 tile 计算重叠；需要有足够连续计算窗口、规则拷贝以及可承受的双份 Shared
Memory。TurboQuant 还有 nibble unpack、centroid lookup 和 metadata scaling，
并非所有处理都能由纯异步 copy 覆盖。

实施前先确认 long-scoreboard/global-memory latency 是主要 stall，并估算双缓冲
是否降低 occupancy。实现后比较 async pipeline active、barrier、register/shared
用量和 Full 时间。若主要瓶颈是 MIO、lookup dependency 或低 batch 并行度，
`cp.async` 可能只增加复杂度。

### 122. 为什么不把 Stage2 也融合进同一个 Kernel？

Stage1 的不同 split 由不同 CTA 计算。Stage2 必须等同一 `(batch, q_head)` 的全部
split 写完 partial output/LSE 后才能稳定归并，而普通 CUDA kernel 内没有跨 CTA
的 grid-wide barrier。让最后一个 CTA 通过 atomic counter 做归并会引入内存顺序、
调度死锁风险和复杂同步；cooperative groups 又限制 grid launch/residency。

当前两次 launch 是清晰且可靠的全局同步边界。Stage2 只占 V7 Full 的约 1.33%，
优化优先级低于 Stage1。除非 profiler 证明 launch 对小 workload 占比很高，否则
不值得以复杂持久化 Kernel 换取很小收益。

### 123. V9 只有 Stage1，怎样得到公平的 V9 Full Decode 结果？

需要正式把 V9 Stage1 接入与 V7 相同的 Stage2 和 benchmark harness：

1. 确认 V9 输出的 `mid_o`、`mid_lse` layout、dtype 和语义与 Stage2 契约一致；
2. 编写/导出 V9 Full launcher，而不是手工相加两个独立历史数字；
3. 对 compressed reference 验证 final output 和 LSE；
4. 与 Triton Full 使用同一输入、预分配、warmup、轮换顺序和统计方法；
5. 同时报告 Stage1、Stage2、Full，避免把 `0.484516 ms` 称为完整 decode。

在这条链路完成前，简历保留 V7 `0.664842 ms / 1.66x` 作为完整结果最严谨。

### 124. 要把当前研究 Kernel 接进 production vLLM，还缺什么？

主要缺口包括：

- 支持 variable sequence length、tail split、非 128-token split；
- 支持动态 batch、不同 GQA ratio、head dimension、block size 和 dtype；
- 正确处理 paged KV cache 的任意 block table、continuous batching 和请求结束；
- 与 vLLM dispatcher、CUDA stream、graph capture 和 tensor layout 正式集成；
- 完整 Store + rotation + Decode 数据契约和多 GPU/TP 场景；
- 系统性数值测试、随机/极端输入、模型 PPL/任务质量测试；
- 多 shape 性能回归、fallback 路径、错误检查与可维护构建系统。

当前代码更准确的定位是固定 workload 的 CUDA optimization research prototype。

### 125. 如果把 Kernel 从 4090 移到 3090，需要重点重做什么？

3090 是 Ampere `sm_86`，SM 数、Tensor Core 代际、时钟、L2、内存子系统和资源
调度与 Ada 不同。首先重新编译 `sm_86` 并确认 PTX/SASS 指令可用；然后重新测：

- ptxas registers、Shared Memory、occupancy 和 active CTA；
- MMA、load、barrier 的实际吞吐和 stall；
- split 数、threads/CTA、tile 数等参数；
- L2/DRAM 行为以及完整 benchmark；
- 数值一致性。

不能只把架构 flag 从 89 改成 86，也不能把项目中旧 3090 NCU 报告当作当前 V9
在 4090 上的证据。历史报告可以提供假设，最终结论必须按 GPU 和代码版本重采。

## 代码证据与学习路线

### 126. 如何用两分钟讲清 V1 到 V9 的优化演进？

可以按“每一版只解决一个主要瓶颈”回答：

| 版本 | 主要变化 | Stage1 时间 |
|---|---|---:|
| V1 | 首个 CTA 内复用 K/V 的 CUDA candidate | 2.074204 ms |
| V2 | warp-per-Q，让 4 个 warp 对应 4 个 GQA Q head | 1.748470 ms |
| V3 | 16-token tiled online softmax，引入 WMMA | 1.380587 ms |
| V4 | 固定 Qwen3-4B shape 特化，消除动态分支与通用开销 | 1.121208 ms |
| V5 | WMMA accumulator fragment 直接写回，减少标量搬运 | 0.845486 ms |
| V6 | K/V 使用对齐 `uint32_t` load，配合 `half2` 重建 | 0.638863 ms |
| V7 | 合并 barrier，并补齐 Stage2/Full benchmark | 0.631122 ms |
| V8 | inline PTX `m16n8k16`，有效 Q 槽从 4/16 提高到 4/8 | 0.513208 ms |
| V9 | QK score 与 online-softmax state 寄存器常驻 | 0.484516 ms |

总体 V1→V9 Stage1 提速约 `4.28x`。但 Full Decode 的正式比较仍采用 V7：
`1.104148 / 0.664842 = 1.66x`。优化故事应同时讲清收益来源与版本口径，不能把
V9 Stage1 时间包装成 V9 Full。

### 127. “WMMA fragment 寄存器直写”解决了什么问题？

普通写法可能先把 accumulator fragment `store_matrix_sync` 到 Shared Memory，
同步后再由线程标量读取、转置或重排，形成 `register -> shared -> register` 往返。
V5 利用 fragment 元素在线程 lane 中的实际映射，直接从 accumulator fragment
提取目标值写入后续状态，减少 Shared Memory traffic、同步和标量 load。

风险是 fragment 内部布局并非 CUDA C++ API 保证的跨架构稳定 ABI，因此项目用
`wmma_fragment_probe.cu` 探测映射，并将优化绑定到目标架构。迁移架构或编译器时
必须重新验证，不应把经验映射当成通用标准。

### 128. V6 的“向量化访存”准确来说是多少位？

源码中每个线程通过 `reinterpret_cast<const uint32_t *>` 各加载一个 32-bit word，
一次得到 8 个 4-bit index；warp 层面连续线程协作形成合并访问。它不是每线程
加载 `uint4` 的 128-bit vector load。

面试中应说“每线程 32-bit 对齐加载，warp 合并读取，再使用位运算和 `half2`
并行重建”。简历其他项目若有 128-bit vectorized load，也不能移植成这个 Kernel
的实现细节。

### 129. V7 消减 barrier 时，如何证明没有引入 race condition？

删除 `__syncthreads()` 前要证明 producer-consumer 范围：

1. 若数据只在同一 warp 内通过寄存器/shuffle 传递，可依赖 warp-synchronous
   执行，但应在独立线程调度语义下使用明确的 `__syncwarp()`；
2. 若 Shared Memory 被其他 warp 读取，仍需要 CTA 级 barrier；
3. 双缓冲复用同一 Shared Memory 区域前，必须保证上一轮所有 consumer 完成；
4. barrier 不能只因“当前测试没错”就删除，要从索引和 happens-before 关系证明。

验证层面应运行 `compute-sanitizer --tool racecheck`，再做随机输入、多次运行、不同
block mapping 和 output/LSE 对照。静态证明与动态工具缺一不可。

### 130. 项目到底如何借鉴 FlashInfer？

借鉴的是 decode attention 的数据流思想：让 QK score、online softmax 的
`m/s` 状态和 output accumulator 尽可能 register-resident，以减少 Shared Memory
往返和 CTA barrier。V9 在这一方向上把 score/state 留在寄存器中。

项目没有链接或调用 FlashInfer runtime，也不是复制某个 FlashInfer Kernel。
准确表述是“参考 FlashInfer 的寄存器常驻 decode 状态设计后，在本项目固定
TurboQuant 数据布局上自行实现”。若面试官要求代码证据，应指向 `cuda/tq4_cuda_v9.cu`
及其与 V8 的 diff。

### 131. 你如何分层验证 CUDA Kernel 的正确性？

建议按三层回答：

1. **Stage1 语义一致性**：CUDA 与 Triton 对相同 compressed cache 比较完整
   `mid_o` 和 `mid_lse`，不是只比 LSE；
2. **Full Decode 一致性**：Stage1 + Stage2 比较最终 output 和 LSE，检查 split
   归并公式；
3. **量化误差基线**：另行与 synthetic FP32 attention 比较，用于观察 4-bit
   量化误差，但不能把该误差判为 CUDA bug。

还要覆盖 NaN/Inf、最大绝对误差、相对误差和多随机种子。Triton V2 曾出现 V
column layout 错误，表现为 LSE 正确但 output 错，这正说明只验证 softmax statistic
不够。

### 132. synthetic benchmark 与 Store→Decode 验证分别证明什么？

synthetic benchmark 直接构造满足布局的 cache，适合隔离 Decode Kernel、重复测量
和定位性能；它证明特定输入和固定 layout 下 CUDA 与参考实现一致。

`validation/store_decode.py` 使用未修改的 vLLM SoA Store 生成真实 compressed
cache，再经过布局转换进入 Decode，用于证明 codebook index、norm correction、
V scale/zero 和 cache layout 契约能衔接。它仍不是 production continuous batching
全覆盖：随机/ragged sequence、任意 page table、tail 和模型质量还需额外测试。

### 133. benchmark 如何保证公平？还存在哪些误差来源？

项目采用同一逻辑 compressed cache，AoS/SoA 转换保持语义一致；输出预分配；先
warmup；用 CUDA Event 计时；每轮 100 次、共 5 轮；轮换 runner 顺序并报告中位数；
性能测试前后做 correctness check。

仍需警惕 GPU boost/温度、后台任务、cache 热度、编译 flags、首次加载、不同 layout
转换是否计时、CUDA stream 同步以及固定 shape 过拟合。更严格的报告应记录 GPU
功耗/时钟，随机化运行顺序，给出各轮分布，并在独立进程复现。

### 134. 简历中的每条表述，分别能在仓库哪里找到证据？

| 简历表述 | 主要证据 |
|---|---|
| vLLM TurboQuant `4bit_nc` 格式 | `reference/config.py`、`reference/turboquant_attn.py` |
| Lloyd-Max centroid 与 norm correction | `reference/centroids.py`、Store/Decode reference |
| 4-bit unpack、GQA、WMMA、online softmax、V 反量化融合 | `cuda/tq4_cuda_stage1_template.cuh`、`cuda/tq4_cuda_v*.cu` |
| fragment 寄存器直写 | `cuda/tq4_cuda_v5.cu`、`cuda/wmma_fragment_probe.cu` |
| 对齐加载和 `half2` | `cuda/tq4_cuda_stage1_template.cuh`、V6 diff |
| barrier 消减 | V6/V7 source diff 与 CUDA README |
| V8 `m16n8k16` | `cuda/tq4_cuda_v8.cu` |
| V9 register-resident state | `cuda/tq4_cuda_v9.cu` |
| split-KV Stage2/Full | `cuda/tq4_cuda_v7.cu`、`benchmarks/full_decode.py` |
| Store→Decode 契约验证 | `validation/store_decode.py` |
| 各版本时间与压缩比 | `cuda/README.md`、`benchmarks/README.md` |
| NCU 历史分析 | `results/v2_stage1.md`、`results/*.ncu-rep` |

其中 `reference/` 和 `vllm/` 是上游快照，用于说明接口和算法来源；原创优化证据
主要在 `cuda/`、`benchmarks/`、`validation/` 和文档中。面试中要区分“阅读并
复现上游逻辑”与“自己设计 CUDA 优化”。

### 135. 如果要在七天内把这部分准备到能扛住追问，怎么学？

| 天数 | 学习重点 | 当天必须完成的自测 |
|---:|---|---|
| Day 1 | TurboQuant 数学：旋转、坐标分布、Lloyd-Max、norm correction | 不看文档推导 K Store→Decode 数据流和 16-centroid codebook |
| Day 2 | GQA、online softmax、split-KV | 手写 online softmax 与 Stage2 合并公式，解释数值稳定性 |
| Day 3 | CUDA 映射：grid/CTA/warp、paged cache、Shared Memory | 给定 shape 算出 CTA 数、每 CTA token/Q head 和 cache bytes |
| Day 4 | WMMA/PTX：`m16n16k16`、`m16n8k16`、fragment | 画出 V7/V8 数据布局，解释 25%→50% 而非 100% |
| Day 5 | 性能：Roofline、occupancy、register、barrier、NCU | 独立算出 4.295 GFLOPs、12.31 FLOP/B、720.3 GB/s及限制 |
| Day 6 | 代码证据：逐个 diff V1→V9、跑 correctness/benchmark | 每版用一句“瓶颈→修改→证据→代价”讲清楚 |
| Day 7 | 模拟面试与边界 | 随机抽 30 题录音回答，任何数字都能现场推导且不越界 |

每道题建议按四句话组织：**结论、原理、项目证据、限制/下一步**。遇到没有 NCU、
模型 PPL 或 production 集成证据的问题，直接说明缺口并给出实验方案，比猜测一个
漂亮结论更可靠。

### 136. K norm 按什么粒度计算和保存？是每个 `num_head` 共用一个吗？

不是整个 KV head 在所有 token 上共用一个 norm。准确粒度是：**每个 token、
每个 KV head、每个 K 向量保存一个独立 norm**。

Store 输入的 Key 逻辑形状是：

$$
K\in\mathbb{R}^{N\times H_{kv}\times D}.
$$

代码将前两维展平成 `N * Hkv` 个长度为 `D` 的向量：

$$
K_{flat}\in\mathbb{R}^{(N H_{kv})\times D},
$$

然后沿 `D` 维分别计算二范数：

$$
s_{t,h}=\sqrt{\sum_{d=0}^{D-1}K_{t,h,d}^{2}}.
$$

开启 norm correction 后，最终保存的是：

$$
\gamma_{t,h}=\frac{s_{t,h}}{\lVert c_{t,h}\rVert_2}.
$$

因此，如果一次 Store 有 $N$ 个 token、$H_{kv}$ 个 KV head，就有
$N\times H_{kv}$ 个 K corrected norm。对当前 `Hkv=8` 的 workload，每个 token
有 8 个 K norm，而不是 32 个 Q-head norm，也不是全层只保存 8 个 norm。

在 paged SoA cache 中，其逻辑 metadata 布局可理解为：

```text
[physical_block, kv_head, metadata_field, token_offset_in_block]
```

其中 `metadata_field=K_NORM`。Decode CTA 根据 token 的 physical block、KV head
和 block 内 offset 读取对应的 `gamma[t,h]`，并让该 KV head 对应的 4 个 GQA
Query head 共同使用这一个 K 向量及其 norm。

V 的 `scale` 和 `zero` 也是每个 token、每个 KV head 各一组，但它们沿 `D` 维
描述该 V 向量的 affine quantization 范围。K norm、V scale/zero 都不是按单个
coordinate 保存，否则 metadata 开销会抵消 4-bit cache 的压缩收益。

## 简历原子级实现追问

### 137. K norm 为什么沿 `head_dim` 计算，而不是沿 token 或 head 维计算？

因为 TurboQuant 的基本量化对象是一个 Key head vector $K_{t,h,:}\in\mathbb{R}^D$。
它需要先把这个向量归一化到单位球面，再旋转和逐坐标量化，因此归约维度必须是
最后一维 $D$：

$$
s_{t,h}=\left\lVert K_{t,h,:}\right\rVert_2
=\sqrt{\sum_{d=0}^{D-1}K_{t,h,d}^2}.
$$

沿 token 维求 norm 会把不同历史位置混合，沿 head 维求 norm 会把不同 KV head
混合，两者都会改变原始 attention 中每个 Key 的方向和幅值语义。源码中
`key.float().reshape(NH, D)` 后执行 `norm(dim=1)`，就是对每个 `(token, kv_head)`
独立沿 $D$ 归约。

### 138. 固定 workload 一共有多少个 K norm？metadata 占多少显存？

若按完整逻辑 cache 的 $B=64,L=4096,H_{kv}=8$ 计算，K 向量数量是：

$$
64\times4096\times8=2{,}097{,}152.
$$

每个 K corrected norm 是 FP16 2 B，因此 K norm 共 `4 MiB`。V scale 和 V zero
也各有同样数量，各占 `4 MiB`，三种 metadata 合计 `12 MiB`。

K/V packed payload 每个 `(token,kv_head)` 是 128 B，总计 `256 MiB`；所以完整
compressed cache 是 `268 MiB`。这也可以从
$2{,}097{,}152\times134\ \text{B}=268\ \text{MiB}$ 复算。实际 paged cache
按已分配 physical block 容量占用，不一定恰好等于当前有效 token 数。

### 139. 为什么不为整个 KV head 只保存一个 norm？

同一个 KV head 在不同 token 位置产生的 Key 向量范数不同。若整条序列共用一个
norm，相当于强迫所有 $K_{t,h,:}$ 使用相同幅值，会直接扭曲不同 token 的 QK
logit。per-token/per-head norm 保留了每个 Key vector 的幅值，只把它的归一化方向
交给 centroid index 表示。

更细到 per-coordinate 保存 scale 又没有必要：K 每个 coordinate 已由非均匀
centroid 表示，额外 128 个 scale 会使 metadata 大幅膨胀。一个向量一个 norm 是
数学语义和压缩开销之间的设计点。

### 140. corrected norm 是在 Store 还是 Decode 计算？为什么？

在 Store 阶段计算。Store 已经拥有当前向量的全部 128 个 quantized index，可以
查 centroid 并归约得到 $\lVert c_{t,h}\rVert_2$，然后一次性保存：

$$
\gamma_{t,h}=\frac{\lVert K_{t,h,:}\rVert_2}
{\lVert c_{t,h,:}\rVert_2}.
$$

Decode 热路径只执行 `centroid[index] * gamma`。如果只保存原始 norm，Decode
每次读取历史 K 都要重新计算 128 个 centroid 平方和、开方和除法；同一个历史
token 会在每一步生成中反复付费。把修正折叠到 Store 是典型的“一次写入计算，
多次 Decode 复用”。

### 141. Lloyd-Max codebook 是每个 token、每个 head、每层各一套吗？

从数学参数看都不是。centroid 只由 `head_dim=d` 和量化位数 `bits` 决定；对当前
`d=128,bits=4`，所有 token、KV head 使用同一组 16 个数。它不是从某一层真实
KV 数据校准出来的，所以数值上也不需要 per-layer codebook。

实现上 `get_centroids(d,bits)` 在 CPU 侧有缓存，但 vLLM 的 `_ensure_on_device`
会把相同数值的 centroid tensor 挂到各 attention layer 上。要区分“数学上是否
每层不同”和“工程上是否每层持有一个 device tensor”。本 CUDA benchmark 接收
一个 `[16]` FP32 centroid tensor，整个 launch 共享。

### 142. centroid 和 midpoint 分别在 Store、Decode 的哪个阶段使用？

两者不能混为一张表：

- 15 个 midpoint 是相邻 centroid 的 decision boundary；Store bucketize 时使用，
  将旋转坐标映射成 `0..15` index；
- 16 个 centroid 是重建值；Store 计算 norm correction 时会查一次，Decode 对
  每个 4-bit index 做 reconstruction 时也会查；
- cache 只保存 index 和 corrected norm，不保存 midpoint 或逐 token centroid。

因此 Decode 不需要重新做二分查找。它已经拿到离散 index，只需直接 lookup。

### 143. 4-bit K/V 的两个 index 在一个 byte 中如何排列？

Store 将相邻 coordinate 两两分组：

$$
\text{byte}_j=(index_{2j}\ \&\ 0xF)
\;|\;(index_{2j+1}\ll4).
$$

所以偶数维在低 4 bit，奇数维在高 4 bit。Decode 对应执行：

```cpp
lo = packed & 15;
hi = packed >> 4;
```

`D=128` 形成 64 B K index，V 同样形成 64 B。这里的“高低 nibble 顺序”必须与
Store 完全一致；顺序颠倒可能仍产生有限数值，却会悄悄置换相邻维度。

### 144. V 的 scale 和 zero 按什么粒度计算？公式是什么？

与 K norm 一样，是每个 token、每个 KV head 的整个 $D$ 维 V 向量一组。Store
沿 $D$ 计算：

$$
v_{min}=\min_d V_{t,h,d},\qquad
v_{max}=\max_d V_{t,h,d},
$$

$$
scale_{t,h}=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right),
$$

$$
q_{t,h,d}=\mathrm{clip}
\left(\mathrm{round}\left(\frac{V_{t,h,d}-v_{min}}{scale_{t,h}}\right),0,15\right).
$$

cache 中的 `zero` 实际保存 FP16 `v_min`，Decode 使用
$\hat V=q\times scale+zero$。它不是常见整数 affine quantization 中需要再参与
减法的 integer zero-point，面试时应说明这个命名差异。

### 145. 如果一个 V 向量的所有元素都相等，scale 会不会为 0？

FP32 量化计算阶段不会。此时 $v_{max}-v_{min}=0$，代码把 scale clamp 到
$10^{-8}$，避免求 index 时除零。所有 index 会量化为 0，Decode 得到：

$$
0\times10^{-8}+v_{min}=v_{min},
$$

仍能恢复常量向量。随后 scale 转成 FP16 时，$10^{-8}$ 可能下溢为 0；但 index
已经在 FP32 scale 下算成 0，故 Decode 仍是 $0\times0+v_{min}=v_{min}$。所以要
区分“量化计算使用的 FP32 scale”和“cache 保存的 FP16 scale”。

### 146. K norm、V scale 和 V zero 为什么保存成 FP16？

每个 `(token,kv_head)` 只有三个标量，但长序列下数量仍很大。FP16 将 metadata
控制为 6 B，使 slot 保持 134 B；如果改成 FP32，slot 会变成 140 B，压缩比从
$512/134=3.82x$ 降到 $512/140=3.66x$，metadata 流量也翻倍。

代价是 corrected norm 和 V affine 参数有 FP16 舍入误差。项目选择 FP16 是容量、
带宽和精度的工程折中，是否可改成 BF16/FP32 应通过 attention output 和模型质量
测试决定，而不是假设 metadata 精度不重要。

### 147. 一个 physical block 内部的 SoA cache 精确布局是什么？

当前固定 `block_size=16,Hkv=8,D=128`。每个 `(position,kv_head)` 的 packed data
只有 K 64 B + V 64 B，不把 metadata 夹在其中。一个 physical block 是：

```text
data region: [16 positions, 8 KV heads, 128 data bytes]
metadata:    [8 KV heads, 3 fields, 16 positions] × FP16
```

字节数分别为：

$$
16\times8\times128=16{,}384\ \text{B},
$$

$$
8\times3\times16\times2=768\ \text{B},
$$

合计 `17,152 B`，等于 $16\times8\times134$。metadata region 从
`META_OFFSET=16384` 开始，三个 field 依次是 K norm、V scale、V zero。

### 148. 为什么 payload 和 metadata 要采用这种“数据区 + SoA metadata”布局？

Decode 对一个 tile 的同一 KV head 连续读取 16 个 token。把 metadata 排成
`[kv_head, field, position]` 后，warp 0 的连续 lane 可分别连续读取 16 个 norm、
16 个 scale 或 16 个 zero，便于合并访问；如果每个 2 B metadata 都夹在 128 B
payload 后，字段访问会形成 134 B stride。

payload 仍按 `[position,kv_head,data]` 排列，使某 token/head 的 K/V 128 B 紧邻，
便于 cooperative packed load。这里所谓 SoA 主要指 metadata 字段拆开，不应笼统
说成整个 cache 每一部分都是纯 field-major。

### 149. Store 的 `slot_mapping` 和 Decode 的 `block_table` 分别解决什么问题？

Store 面对新 token，`slot_mapping[token]` 直接给出写入的全局 physical slot：

$$
block=slot/16,\qquad offset=slot\bmod16.
$$

Decode 从 sequence 的逻辑 token 位置出发，通过
`block_table[batch, logical_block]` 找到 physical block，再加 block 内 offset。
前者是“当前 token 写到哪里”，后者是“历史逻辑位置存在哪个物理页”。二者共同
实现 paged KV cache，但不是同一个数组，也不能互换。

### 150. Stage1 的 grid 三个维度分别是什么？为什么这样分？

grid 是 `(batch, kv_head, split)`：

```text
blockIdx.x -> batch sequence
blockIdx.y -> KV head
blockIdx.z -> KV split
```

一个 CTA 负责该 KV head 对应的全部 4 个 Query head。这样 K/V tile 解码一次后
能被 4 个 GQA Query 复用。若改成 `(batch,q_head,split)`，CTA 更多，但同一 K/V
会被四个 CTA 重复读取和反量化；若一个 CTA 负责全部 8 个 KV head，Shared
Memory、寄存器和单 CTA 工作量又过大。

### 151. `kvh` 如何映射到 4 个 Query head？

当前 Qwen3-4B shape 的 GQA ratio 是：

$$
G=H_q/H_{kv}=32/8=4.
$$

代码使用：

$$
qh=kvh\times4+q_{local},\qquad q_{local}\in\{0,1,2,3\}.
$$

例如 `kvh=3` 对应 Q head 12、13、14、15。这依赖 head 按连续 group 排列的 layout
契约；若模型或框架采用不同 head mapping，不能继续使用该公式而不做转换。

### 152. 128-thread CTA 中四个 warp 的职责完全相同吗？

不完全相同。V8/V9 中：

- 128 threads 协作加载 Q、解码 K/V；
- warp 0 负责 QK 的 `m16n8k16` MMA；
- V8 的四个 warp 各维护一个 Query head 的 online-softmax 标量状态；
- V9 把四个 Query head 的 QK/softmax 状态重新映射到 warp 0 的 lane classes；
- PV 阶段四个 warp 分别负责 output 的 32 个维度切片，共覆盖 D=128。

因此“一个 warp 对应一个 Query head”对 V2/V7 的解释较直观，但不能不加限定地
套到 V9 的所有阶段。V9 的 QK/softmax 所有权与 PV output 切片所有权不同。

### 153. 一个 16-token tile 的 norm/scale/zero 由谁加载，谁使用？

warp 0 先查一次 block table；lane 0 得到 physical block 后用 shuffle 广播。
随后 warp 0 的 lane 0–15 各负责一个 token position，分别加载该 `kvh` 的
`k_norm[t]`、`v_scale[t]` 和 `v_zero[t]` 到 Shared Memory。

CTA barrier 后，全部 128 threads 在 cooperative unpack 中使用这 16 组 metadata。
同一 token 的 K norm 和 V scale/zero 会服务该 KV group 的 4 个 Query head；它们
不是每个 warp、每个 Query head重复从 global memory 加载一份。

### 154. 16-entry centroid lookup 在 CUDA 中怎样实现？每次都访问 global memory 吗？

每个 warp 开始时让 lane 0–15 各自加载一个 FP32 centroid 到寄存器
`centroid_lane`。解码 nibble 后，以 index 作为 source lane 调用
`__shfl_sync`，把对应 centroid 广播给需要它的 lane。

所以热循环中的 lookup 主要是 register shuffle，不是每个 coordinate 都发起一次
global gather。代价仍包括 nibble 位运算、shuffle 和数据依赖；而且每个 warp
都有自己的 16-entry 分布式寄存器表，不是整个 CTA 只有一份物理寄存器。

### 155. `q_rot` 的形状是什么？Query rotation 是否在 `0.6648 ms` 内？

Kernel 输入 `q_rot` 的逻辑形状是 `[B,Hq,D]`，当前 dtype 为 FP32。对每个 CTA，
只加载当前 `kvh` 对应的 4 个 Query head，即 `[4,128]`。

Query 在进入 benchmark 前已经完成与 K 相匹配的旋转；`0.664842 ms` 只测预旋转
Query 和 compressed KV 经过 Stage1 + Stage2，不包含 Query rotation。简历中的
“完整 Decode Kernel 链路”必须按这个项目定义解释，不能说成完整 attention backend
或完整 token latency。

### 156. 为什么 score 要乘 `1/sqrt(128)`，代码又为什么出现 `RCP_LN2`？

scaled dot-product attention 使用：

$$
score=QK^T/\sqrt{D}.
$$

当前 $D=128$，所以 `ATTN_SCALE=1/sqrt(128)=0.088388...`。Kernel 用更适合 GPU
的 `exp2f` 实现指数，因此利用

$$
e^x=2^{x/\ln2},
$$

先把 score 乘 `RCP_LN2=1/ln(2)`。最终 LSE 再用
`running_m * LN2 + log(running_l)` 转回自然对数语义。`RCP_LN2` 不是另一个
attention scale，也不是量化 scale。

### 157. Online softmax 的 `m`、`l` 和 output accumulator 按什么粒度分配？

逻辑上每个 `(batch, q_head, split)` 有一套独立状态：

- $m$：该 split 已处理 score 的运行最大值；
- $l$：以 $m$ 为基准的指数和；
- $o\in\mathbb{R}^{128}$：该 Query head 的 Value 加权累加器。

四个 GQA Query head 共享 K/V tile，但不能共享 softmax 状态，因为它们的 Query
不同，QK score 也不同。V9 只是改变状态在 warp/lane 寄存器中的物理映射，没有
改变“一 Query head、一 split、一套逻辑 softmax state”的数学语义。

### 158. `mid_o` 为什么最后一维是 129，而不是 128？总大小是多少？

每个 split 要输出 128 维 partial attention output，并额外输出一个 split LSE，
所以 shape 是：

$$
[B,H_q,S,D+1]=[64,32,32,129].
$$

FP32 总字节数为：

$$
64\times32\times32\times129\times4
=33{,}816{,}576\ \text{B}=32.25\ \text{MiB}.
$$

前 128 个元素已经除以当前 split 的 softmax denominator，是 normalized partial
output；第 129 个元素是该 split 的 LSE。Stage2 用 LSE 在全局尺度重新加权，
不能直接对 32 个 partial output 求平均。

### 159. Stage2 为什么必须同时读取 partial output 和 split LSE？

设第 $s$ 个 split 的 normalized output 为 $o_s$，LSE 为 $L_s$。全局最大值
$M=\max_s L_s$，则合并权重是：

$$
w_s=e^{L_s-M},
$$

$$
o=\frac{\sum_s w_so_s}{\sum_s w_s},\qquad
L=M+\log\sum_s w_s.
$$

若只保留 $o_s$ 而没有 $L_s$，就不知道每个 split 的 softmax denominator 相对
大小，无法恢复全序列 attention。LSE 同时保证指数计算数值稳定。

### 160. `0.664842 ms` 是否等于表中的 Stage1 中位数加 Stage2 中位数？

不是严格相加。完整数据是：

| 实现 | Stage1 | Stage2 | Full |
|---|---:|---:|---:|
| Triton V2-fixed | 1.066916 ms | 0.024852 ms | 1.104148 ms |
| CUDA V7 | 0.631255 ms | 0.008868 ms | 0.664842 ms |

Full 是把两个 launch 放进同一个 runner 后独立计时并取五轮中位数；Stage1 和
Stage2 也各自独立测量、独立取中位数。不同样本集合的中位数不满足可加性，并且
Full 还包含两个 launch 之间的实际提交顺序，所以
`0.631255 + 0.008868 != 0.664842` 是正常的。

### 161. 为什么用 SoA Triton V2-fixed 作为主要 baseline？

因为它与 CUDA candidate 使用相同的 TurboQuant 算法语义、相同 4bit_nc 数据、
相同固定 workload，并采用更适合 metadata 访问的 SoA layout；相比 AoS V1 和
SoA V1，它是仓库中更强的 Triton Stage1 baseline。用弱 baseline 会夸大 CUDA
优化收益。

`fixed` 还很重要：原 V2 的 V column layout 有 correctness bug，可能出现 LSE
正确但 output 错误。性能比较使用修复后的 V2；历史修复前 NCU 报告只能辅助定位，
不能作为当前公平性能结论。

### 162. `1.66x` 是怎样得到的？它等于快了 66% 吗？

Full Decode 加速比是：

$$
speedup=\frac{1.104148}{0.664842}=1.661\times.
$$

“吞吐能力约为原来的 1.661 倍”可以口语化为“1.66x faster”。但 latency reduction
是：

$$
1-\frac{0.664842}{1.104148}=39.79\%.
$$

因此不能说延迟降低 66%。`1.66x` 和 `39.8% latency reduction` 是同一组时间的
两种表达，分母不同。

### 163. “单次 Stage1 launch”是否表示量化、Store 和整个 Decode 都只有一次 launch？

不是。它只表示 Stage1 attention 内部没有把以下步骤拆成多个 global Kernel：
packed K/V load、nibble unpack、K lookup、V affine reconstruction、QK、online
softmax 和 PV accumulation 在一个 CUDA Stage1 launch 内完成。

KV Store 在历史 token 写入 cache 时单独执行；Query rotation 在 benchmark 之前；
跨 split 归并由 Stage2 第二个 launch 完成。简历中的“融合至单次 Stage1 launch”
已经限定了范围，回答时不能省略 `Stage1` 这个限定词。

### 164. WMMA fragment 直接写回依赖什么非通用假设？如何验证？

它依赖 `sm_89` 下 accumulator fragment 元素到 lane/fragment index 的实际映射。
CUDA WMMA C++ API允许 load、mma 和 store，但 fragment 内部元素布局不是稳定的
跨架构 ABI。V5 绕过 `store_matrix_sync -> Shared Memory -> scalar reload` 时，
必须知道哪些 lane 持有哪一行、哪一列。

项目使用 `cuda/wmma_fragment_probe.cu` 恢复并验证映射，再把直接写回限制在目标
架构。更换 GPU 架构、CUDA 编译器或 MMA shape 后应重新跑 probe 和完整 output
correctness，不能仅凭旧映射编译通过就认为正确。

### 165. “同步 barrier 消减”具体删除了什么？为什么可以删？

V6 在每个 tile 的 PV MMA 之后还有一个 CTA barrier。V7 发现该 barrier 与下一
轮开头 metadata 准备后的 barrier 在生命周期上可以合并：PV 完成后各 warp 的
accumulator 在寄存器中，下一轮 warp 0 更新的是 `data_base/k_norm/v_scale/v_zero`，
随后已有 CTA barrier 会在任何线程重写并消费下一轮 K/V Shared Memory 前汇合。

因此 V7 通过 `TQ4_FUSED_TILE_BARRIER` 去掉每 tile 一个冗余同步，而不是删除所有
barrier。V9 进一步不再物化 `qk_s`，每 tile 的 CTA barrier 从 V8 的 4 次降为
3 次。安全性来自 Shared Memory producer-consumer 生命周期分析，并需要 racecheck
和数值测试共同验证。

### 166. 哪些是上游 vLLM 内容，哪些是这个项目自己的工作？

上游或参考内容包括 TurboQuant 数学、centroid 生成、4bit_nc 配置、Triton
Store/Decode 和 cache layout；仓库的 `reference/`、`vllm/` 是对应快照，不应
声称为原创。

本项目工作的主要证据在：CUDA V1-V9 的设计演进、固定 workload 的 Tensor Core
映射、fragment 写回、packed load、barrier/寄存器状态优化、CUDA Stage2、benchmark
harness、Store→Decode 验证以及文档化分析。面试回答应把“基于 vLLM 接口实现与
优化”说清楚，既不贬低工程工作，也不把上游算法归到个人贡献。

### 167. 将简历 TurboQuant 段落逐短语拆开后，是否每项都有独立问题？

按当前文档审计如下：

| 简历原短语 | 独立问题 |
|---|---|
| 基于 vLLM TurboQuant `4bit_nc` | 5、12、77、80、95、166 |
| CUDA Tensor Core 融合 Decode Kernel | 20、29、31–33、91、99 |
| 4-bit K/V unpack | 18、41、120、128、143、153–154 |
| Lloyd-Max 查表 | 8、69–74、89、141–142、154 |
| Grouped Attention | 28–31、100、114、150–152 |
| online softmax | 23–25、34、99、156–159 |
| Value 反量化 | 11、81–82、144–146、153 |
| 单次 Stage1 launch | 20、27、98–99、163 |
| split-KV Stage2 归并 | 21–26、98、122–123、158–160 |
| WMMA fragment 寄存器直写 | 40、127、164 |
| 向量化访存 | 18、41、120、128、143 |
| barrier 消减 | 42、129、165 |
| KV Cache 压缩约 `3.82x` | 13、96、138、146 |
| RTX 4090 | 64、92–94、106–110、125 |
| Qwen3-4B 固定 workload | 4、39、97、100–105、111–117 |
| Full Decode `0.6648 ms` | 27、53、98、123、160 |
| SoA Triton V2 baseline 提速 `1.66x` | 14、45、49、53、161–162 |
| norm/scale 等参数粒度 | 136–146 |
| 个人贡献与边界 | 59–65、124、134、166 |

这里的验收标准不再是“某个综合答案顺带提过”，而是关键实现假设必须能被单独
抽问、给出公式或代码映射，并说明适用边界。

## 容易被追问的口径

- 不要说 V9 已经是 production vLLM backend；它目前是固定 workload 的
  Stage1 research extension。
- 不要把 `0.484516 ms` 说成包含 Store、Query rotation 和 Stage2 的端到端时间。
- 不要把 CUDA V1→V9 的 4.28x、V9 对 Triton 的 2.22x 和 V7 Full Decode 的
  1.66x 混为一个加速比。
- 不要说四个 KV head 可以无条件拼成 16 个 Q head 填满一次 dense WMMA；
  不同 group 使用不同 K/V，直接拼接在数学上错误。
- 不要说 V8 把 Tensor Core 利用率提高到 100%；有效槽位只是从 25% 提高到 50%。
- 不要说 V9 直接使用了 FlashInfer Kernel；它借鉴 register-resident state
  update 思路，项目没有新增 FlashInfer 运行时依赖。
- 不要把 `cuobjdump` 的静态 site 当成 NCU 动态执行次数或硬件吞吐率。
- 不要声称 V9 已通过 NCU 达到某个显存带宽百分比；当前新版本动态 NCU 数据
  尚未正式采集。
- 不要只检查 LSE；Triton V2 的 V-layout bug 证明 LSE 正确时 output 仍可能错。
- 不要把量化相对 FP32 的误差归咎于 CUDA；应先比较 CUDA 与 Triton 对同一
  compressed cache 的结果。
- 不要说 occupancy 越高越好；最终应以无 spill、足够 latency hiding 和实测
  Kernel 时间判断。
- 不要把固定 4090 数据直接外推到 3090、其他 head dimension、GQA ratio 或
  variable-length workload。
- 不要用 `1.321 PetaOPS` 计算本 Kernel 的 Tensor 利用率；相关 dense FP16
  Tensor 峰值约为 `165.2 TFLOPS`。
- 不要把 `720.3 GB/s` 称为 NCU 实测 DRAM 带宽；它是 logical bytes/time。
- 没有当前 V9 的动态 NCU 证据时，不要断言“纯 DRAM bandwidth-bound”；应说
  Roofline 判断为 memory-dominant candidate。
- 不要把 V6 每线程 `uint32_t` 读取说成每线程 128-bit `uint4` 读取。
- 不要声称已超过 FP16 FlashAttention；仓库没有对应的公平 baseline。
- 不要把 synthetic kernel correctness 说成完整模型精度或 perplexity 已验证。
