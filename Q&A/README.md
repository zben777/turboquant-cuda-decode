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

### 2. 为什么 Decode 阶段特别关注 KV Cache？

Prefill 可以通过较大的矩阵乘法获得较高计算强度，而 Decode 每一步通常只有
一个新 query，却必须读取整个历史 K/V。随着 context length 增长，读取 K/V
的字节数线性增长，计算相对较少，因此容易成为 memory-bound 路径。低比特
KV Cache 的直接收益是减少每个历史 token 必须搬运的数据。

### 3. 为什么不能直接先反量化，再调用普通 FlashAttention？

这样会先读 INT4 cache，再把完整 FP16 K/V 写回显存，随后 Attention 又重新
读取 FP16 K/V。额外的中间张量、写回流量和 Kernel launch 会抵消压缩收益。
本项目选择在 tile 内解码到 Shared Memory，然后立即用于 Tensor Core QK/PV，
解码结果不落到全局内存。

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
非均匀标量量化。Cache 保存每个坐标的 4-bit centroid index 和每个 token 的
K norm。

V 不参与 QK 内积，本项目对 V 使用更常见的 per-token uniform 4-bit 量化，
保存 index、scale 和 zero。

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

### 8. Lloyd-Max codebook 是什么？

4 bit 对应 16 个 centroid。Lloyd-Max 根据目标概率分布迭代更新量化区间边界
和区间条件均值，使标量均方误差降低。Store 用 15 个 midpoint 对旋转后的
K 坐标做 bucketize，最终只保存 0 到 15 的 index；Decode 根据 index 查
16-entry FP32 centroid table。

### 9. K 是怎样量化和恢复的？

Store 的主要过程是：

1. 计算每个 K 向量的二范数；
2. 归一化并通过 GEMM 完成旋转；
3. 根据 midpoint 对每个坐标做二分 bucketize；
4. 两个 4-bit index 打包成一个 byte；
5. 保存校正后的 K norm。

Decode 对每个 nibble 查 centroid，再乘对应 token 的 norm，得到 tile 内用于
QK 的 FP16 K。完整 K 不写回 Global Memory。

### 10. norm correction 是什么？

量化后的 centroid 向量范数不一定恰好为 1。Store 在开启 norm correction
时，将 centroid 向量的逆范数折叠进保存的标量：

$$
\gamma_{stored} = \frac{\lVert K\rVert_2}{\lVert c\rVert_2}.
$$

Decode 只需要计算 `centroid[index] * gamma_stored`，不必在每个 tile 重新求
centroid 向量范数。这是用 Store 阶段一次计算换 Decode 热路径更少的操作。

### 11. V 为什么不用同样的 centroid 量化？

K 直接决定 QK score，对内积误差敏感；V 在 softmax 权重确定后参与加权求和，
工程实现选择更简单的 per-token uniform quantization。4-bit V 使用：

$$
v_{recon} = index \times scale + zero, \qquad index\in[0,15].
$$

这样 Decode 只需 nibble unpack、整数转浮点和一次乘加，不需要 centroid lookup。

### 12. 当前实现使用 QJL residual 吗？

没有。当前研究的是 vLLM `turboquant_4bit_nc` 路径，使用 rotation、centroid、
norm、V scale/zero，不保存 QJL residual。不能把论文中更广泛的 TurboQuant
变体全部说成当前 Kernel 已实现的功能。

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

### 16. 为什么 metadata 使用 FP16？

每个 token/head 只需三个标量。FP16 将 metadata 控制在 6 B，同时其精度对
当前 4-bit 量化路径足够。Decode 按 `uint16` 位模式读取后转换为 half/float。
centroid table 本身仍是 16 个 FP32 值。

### 17. block table 在 Kernel 中做什么？

Paged KV Cache 的逻辑 token 不保证位于连续 physical page。`block_table[b,
logical_block]` 将序列的逻辑 block 映射到 cache 中的 physical block。固定
workload 每个 tile 恰好 16 token，因此 V4 之后每个 tile 只需读取一次
block-table entry。

### 18. 为什么向量化 `uint32` load 是安全的？

每个 packed K 或 V 区域是 64 B，slot 的 data 部分是 128 B，固定布局保证
读取地址满足四字节对齐。V6 每次读取四个 packed byte，对应八个 4-bit
dimension；随后通过 `half2` 写两个重建值，减少 load、地址计算和 shared
store 指令。对齐约束是 V6-V9 的显式限制，不能假定任意 cache layout 都安全。

### 19. 如何证明 CUDA 读取的是真实 vLLM Store 布局？

`validation.store_decode` 从原始 FP16 Q/K/V 开始，调用未经修改的 vLLM SoA
Triton Store，真实执行旋转、bucketize、packing 和 metadata 写入，然后将
同一个 cache tensor 直接交给 Triton Decode 与 CUDA V7，中间没有 byte
rearrangement。CUDA 与 Triton output 最大差约 `5.06e-06`，说明二者对 layout
的解释一致。

## Attention 与 Split-KV

### 20. Stage1 到底计算什么？

一个 CTA 对应 `(batch, KV head, split)`，处理该 KV head 对应的四个 Q head
和当前 split 的 128 个 token。它循环处理八个 16-token tile，完成：

```text
packed K/V load -> dequant -> QK -> online softmax -> PV
```

最后为每个 Q head/split 输出 128 维 partial output 和一个 split LSE。

### 21. 为什么要把 4096 token 分成 32 个 split？

如果一个 CTA 处理完整 4096 token，并行 CTA 数量会不足，单 CTA 生命周期也
很长。拆成 32 个 128-token split 后，可以在 batch、KV head 和 split 三个
维度产生更多 CTA，提高 GPU 并行度。代价是需要 Stage2 合并各 split state。

### 22. Stage1 输出为什么是 129 个 float？

前 128 个是归一化后的 partial output，最后一个是该 split 的 log-sum-exp：

```text
mid_o shape = [B, Hq, 32, 128 + 1]
```

Stage2 使用每个 split 的 LSE 对 partial output 做数值稳定的重新加权，不能
简单对 32 份 partial output 求平均。

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

### 25. Stage2 怎样合并 split？

先求所有 split LSE 的最大值，再计算每份权重：

$$
w_i = e^{LSE_i-LSE_{global}},
$$

然后归一化加权 partial output，并得到全局 LSE。CUDA Stage2 只占 V7 Full
Decode 的约 1.33%，但它是语义上必需的，不能因耗时小就省略。

### 26. 为什么 Stage1 与 Stage2 使用两个 Kernel？

Stage2 必须等所有 split CTA 写完 `mid_o`。普通 CUDA Kernel 内没有通用的
grid-wide barrier，因此使用 Kernel launch 边界表达全局同步最直接，也避免
cooperative launch 的额外约束。

### 27. Stage1 时间和 Full Decode 时间为什么不能混为一谈？

Stage1 benchmark 不含 Stage2、Query rotation、Store、输入构造和 JIT。
Full Decode 也只定义为预先旋转的 Q 和压缩 cache 经过 Stage1+Stage2，不包含
Store。简历和面试必须明确计时边界，否则 `0.485 ms` 不能被描述成完整端到端
请求延迟。

## GQA、WMMA 与 Tensor Core

### 28. GQA-4 在这个项目中是什么意思？

32 个 Q head 共享 8 个 KV head，因此每个 KV head 对应四个 Q head。一个 CTA
以 KV group 为单位，解码一份 K/V tile，并为四个 Q head 复用它。这样避免
每个 Q head 都重复读取和反量化同一份 K/V。

### 29. 为什么普通 `m16n16k16` 会浪费 Tensor Core 槽位？

V3-V7 把四个 Q head 放到 WMMA 的 M 维。硬件 tile 要求 M=16，但只有四行
真实数据，其余 12 行是 padding，有效行比例只有 `4/16=25%`。虽然 Tensor
Core 很快，这种结构仍执行了无效 HMMA 工作，并扩大 accumulator/scratch。

### 30. 为什么不能把四个 KV head 和 16 个 Q head 直接拼成一次 dense MMA？

因为四个 GQA group 使用四份不同的 K/V 矩阵。普通 dense GEMM 的同一次
矩阵乘法要求所有输出行共享同一个右操作数；直接堆叠会产生跨 group 的错误
Q-K 配对。除非构造 block-diagonal K，这又会引入更多零和复杂布局，因此
不能只凭“16 个 Q head 正好填满 M=16”判断数学上成立。

### 31. V8 怎样提高 GQA-4 的 MMA 利用率？

V8 在每个合法 KV group 内转置两次乘法：

```text
QK: K(16x16)   * Q^T(16x8)
PV: V^T(16x16) * P(16x8)
```

它使用原生 `mma.sync.aligned.m16n8k16`，让四个 Q head 占 N=8 的四列，有效
槽位比例从 `4/16=25%` 提高到 `4/8=50%`。静态 HMMA site 从 V7 的 160
降到 V8 的 80。

### 32. 为什么 V8 不继续使用 C++ WMMA API？

CUDA C++ WMMA 常用接口提供的是 `m16n16k16` fragment，而 V8 需要明确的
`m16n8k16` register contract。Inline PTX 能直接指定 MMA shape 和 operand
register，并手工实现 lane 到矩阵元素的映射。代价是代码与 `sm_89` 架构和
fragment layout 更紧密耦合。

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

### 34. V9 借鉴 FlashInfer 的地方是什么？

借鉴的是 register-resident attention state，而不是直接调用 FlashInfer API
或复制它的 Kernel。V8 会把 QK accumulator 写入 `qk_s`，同步后再由四个 warp
读取并更新 softmax。V9 让 warp 0 直接按 lane class 对四个 Q-head score 列
做 max/sum reduction，在寄存器中维护 `(m,l)`，只将 FP16 probability 写给
后续 PV。

由于 compressed K/V 必须执行 nibble unpack、centroid lookup 和 scale/zero
重建，普通 `cp.async` 不能直接完成这段变换；因此这里优先迁移 FlashInfer
最适合当前数据路径的 state-fusion 思路。

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

## CUDA V1-V9 演进

### 36. CUDA V1 的设计和问题是什么？

V1 一个 CTA 负责一个 `(batch, KV head, split)`，四个 Q head 复用解码后的
K/V。问题是每个 token 都要为四个 head 做多轮 warp reduction，并频繁通过
Shared Memory 做 CTA 协作。它是正确、可复用 K/V 的起点，但 2.074 ms 比
Triton baseline 慢。

### 37. CUDA V2 为什么每个 Q head 一个 warp？

这样 QK reduction、softmax state 和 output 都可以保留在 warp 内，移除 CTA
barrier，并把静态 `SHFL.DOWN` site 从 20 降到 5。代价是四个 warp 重复读取
并反量化同一 K/V。最终从 2.074 ms 降到 1.748 ms，说明同步减少有收益，
但重复 decode 限制了进一步提升。

### 38. CUDA V3 的关键变化是什么？

V3 建立单遍 tiled Tensor Core 执行图：一次解码 16 个 token，在 WMMA 上完成
QK，更新 online softmax，再做 PV；八个 tile 的 output accumulator 保留在
fragment registers。它移除两遍 score/weight staging，从 V2 的 1.748 ms
降到 1.381 ms，并在 SASS 中确认生成 HMMA 指令。

### 39. CUDA V4 为什么要做固定 workload 特化？

V4 利用每个 split 固定 128 token 且按 16 对齐的条件，只加载一次 tile 的
block-table entry，只初始化四个有效 Q 行，将 centroid table放到 warp
register，并完全展开八个 tile。它用通用性换取更少的分支、地址计算和循环
控制，从 1.381 ms 降到 1.121 ms。

### 40. CUDA V5 的 fragment 直接写回是什么？

V4 把完整 `16x128` output accumulator 先写到 Shared Memory，再只读取四个
有效行。V5 根据在 `sm_89` 上探测出的 WMMA lane-to-row mapping，直接从
fragment register 将四行写到 `mid_o`，移除约 7 KB scratch 和一次大规模
shared round trip，时间降到 0.845 ms。

### 41. CUDA V6 的向量化为什么收益大？

INT4 decode 涉及大量细粒度 byte load、nibble 提取、地址计算和 half store。
V6 用一个对齐 `uint32` 同时读取四个 byte，再用 `half2` 写八个重建维度，
减少指令和 shared-store 数量。它没有改变 Tensor Core 工作量，却从 0.845 ms
降到 0.639 ms，说明 decode 数据路径此前占比很高。

### 42. CUDA V7 怎样减少 barrier？

V6 每个 tile 末尾有一个 barrier。V7 发现下一个 tile 开头原本就有 metadata
发布 barrier，而 metadata storage 与 K/V storage 独立，因此可以让这个开头
barrier 同时承担“等待上一个 PV 完成”和“发布新 metadata”两个作用。静态
barrier 从 42 降到 34，收益约 1.012x。

### 43. 为什么 V7 到 V8 的收益比 V6 到 V7 大？

V7 只消除一部分同步，主计算形状仍有 75% 无效 WMMA 行；V8 直接改变 MMA
shape，将 HMMA 数量减半，并缩小 Shared Memory 和 accumulator。前者是局部
调度优化，后者减少了核心 Tensor Core 工作量，所以 V8 获得约 23% 提升，
而 V7 只有约 1.2%。

### 44. 为什么要保留所有历史版本？

每个版本只引入一类主要变化，构成可复现的 ablation：线程映射、单遍算法、
固定特化、写回、向量化、同步、MMA shape 和 softmax fusion。这样可以解释
性能来自哪里，也能避免把多个变化一起提交后无法归因。面试时这比只展示
最终 V9 更能体现性能工程方法。

## Correctness 与实验设计

### 45. 上游 Triton V2 曾经有什么 correctness 问题？

复制的 V2 使用 `tl.interleave(v_lo, v_hi)` 重建 V，随后直接交给 `tl.dot`。
在 CUDA 上该 layout 与 dot 期望不兼容，造成 V 列置换。QK 和 softmax 未受
影响，所以 LSE 看起来正确，但 output 最大误差约 0.325。

修复版根据 `d//2` 和 nibble shift 直接构造最终 `[TILE, D]` V layout，输出
误差恢复到约 `1e-4`。这说明只检查 LSE 不足以证明 Attention 正确。

### 46. CUDA V9 的正确性怎样验证？

正式 harness 为所有实现构造同一逻辑 cache，并与 SoA Triton V1 比较完整
`mid_o` 的 partial output 和 LSE，同时检查所有值 finite。V9 的结果是：

```text
output max/mean  9.6827745e-05 / 1.2782620e-05
LSE max/mean     2.4318695e-05 / 3.7480786e-06
```

此外，V7 Full Decode 与 Triton Full Decode 的最终 output 最大差约
`5.66e-07`，Store 兼容性测试也覆盖真实压缩 cache。

### 47. 为什么 CUDA V3-V9 与 Triton V1 有约 `1e-4` 误差？

这些版本使用 FP16 Tensor Core operand、不同的求和顺序、online softmax 和
fast-math 指数近似。浮点加法不满足结合律，因此与逐元素 FP32 路径不会逐位
一致。误差应结合 reference、最终输出和量化误差判断，不能把非零差异直接
当成 Kernel 错误。

### 48. 如何区分 Kernel 数值误差和 4-bit 量化误差？

Store 验证中，CUDA 与 Triton 读取同一量化 cache，output 最大差约
`5.06e-06`；而量化 Decode 与原始 FP32 Attention 的 output 最大差约
`1.87e-02`。前者远小于后者，说明较大的差异来自预期量化损失，不是 CUDA
layout 或 Attention 算法错误。

### 49. Benchmark 怎样保证版本比较公平？

所有版本使用同一逻辑输入、相同 shape 和输出语义。Harness 在 AoS/SoA 间做
无损转换，转换不在计时区；每个实现先 warmup，再用 CUDA Event 对 100 次
launch 计时，进行五轮并轮换测量顺序，最终报告五轮中位数。JIT、分配和输入
构造均不计时。

### 50. 为什么使用中位数而不是只报最快一次？

GPU Boost、温度、后台负载和首次 cache 状态会造成波动。最快值可能只是偶然
高频状态，平均值也容易受离群点影响。多轮交错测试加中位数更稳健；小于几个
百分点的优化还应重复验证，并结合资源与 SASS 证据。

### 51. 你使用了哪些 profiling 证据？

当前可复现的 V3-V9 证据主要来自 `cuobjdump`：register/thread、Shared
Memory/CTA、HMMA 和 `BAR.SYNC` 静态 site，并通过 CUDA Event 测量性能。
仓库中的旧 NCU 报告生成于 Triton V2 correctness 修复前，CUDA V1 的 NCU
采集还遇到 `ERR_NVGPUCTRPERM`。

因此面试时不能声称 V9 已经通过 NCU 证明达到某个 DRAM 峰值百分比；可以说
静态资源和 SASS 支持优化归因，动态瓶颈仍需要重新采集 NCU。

### 52. 静态 HMMA 或 barrier 数量能直接等价为运行时间吗？

不能。完全展开的八个 tile 会让同一逻辑操作出现多个静态 site；实际性能还
取决于指令吞吐、依赖、occupancy、memory latency 和调度。V8 的 HMMA 减半
且实测明显加速，V9 的 barrier 减少也有稳定收益，但必须以同环境 CUDA Event
结果确认，不能只看反汇编计数。

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

### 54. “加速 4.28x”和“耗时降低多少”有什么区别？

```text
speedup = 2.074204 / 0.484516 = 4.281x
time reduction = 1 - 0.484516 / 2.074204 = 76.64%
```

4.28x 不是“耗时降低 428%”，也不宜简单说“性能提升 328%”而不说明计算口径。

### 55. Shared Memory 越少、occupancy 就一定越高吗？

不一定。CTA residency 同时受 registers、Shared Memory、threads、warp slots
和架构限制。减少 Shared Memory 可能解除某个限制，也可能根本不是当前 limiting
resource。Occupancy 只是 latency hiding 的条件，不是最终性能指标；降低资源
却增加指令或 spill 仍可能变慢。

### 56. 为什么不用 double buffering 或 `cp.async` 做完整流水？

压缩 cache 不是可以直接异步复制成最终 FP16 tile的数据。K 需要 unpack、
centroid lookup 和 norm，V 需要 unpack、scale/zero；`cp.async` 只能搬字节，
不能执行这些变换。双缓冲还会增加 Shared Memory，并可能降低 resident CTA。

它仍然是可实验方向，例如异步预取 packed byte 或 metadata，但必须测量变换、
额外同步和 occupancy 的综合成本，不能因为 FlashInfer 使用 pipeline 就假定
本项目照搬一定更快。

### 57. 为什么 V9 仍然只有 50% 的 N 维有效槽位？

`m16n8k16` 的 N 固定为 8，而 GQA group 只有四个 Q head，所以仍有四列为空。
它已比 `m16n16k16` 的四行有效更好，但不是 100%。进一步填充需要让一次 MMA
同时处理更多合法输出，同时保证不同 KV group 的 K/V 不交叉；这通常要求
更复杂的 block-diagonal、稀疏或多-MMA 调度，收益需覆盖布局成本。

### 58. 为什么 V9 没有直接替换 Full Decode 中的 V7 Stage1？

当前版本演进把 V8/V9 保持为独立 Stage1 candidate，V7 则提供经过验证的
Stage1、Stage2 和 Full launcher。将 V9 接入 Full Decode 在工程上可做，但
需要新增稳定导出、完整回归和 Store 路径验证。当前文档明确区分这两个范围，
避免把 Stage1 实验误报成已完成的 production chain。

### 59. 这个项目目前最大的局限是什么？

主要局限包括：固定 `B=64/context=4096/Hq=32/Hkv=8/D=128`；每个 split
必须是对齐的 128 token；V8/V9 的 inline PTX 和 fragment mapping 面向
`sm_89`；合成 benchmark 使用连续 page 和统一 sequence length；尚未覆盖
随机 block table、ragged batch、其他 GQA ratio 和完整 vLLM backend 注册。

### 60. 下一步最值得做什么？

优先级较高的是：

1. 将 V9 Stage1 接入 CUDA Full Decode 并重复 Store compatibility；
2. 支持 variable sequence length、tail split 和随机 block table；
3. 为不同 GQA ratio、head dimension 和 batch 做模板化/autotune；
4. 重新采集 V8/V9 NCU，确认 memory、Tensor Core、barrier 和 scheduler stall；
5. 尝试 packed-byte/metadata 预取、warp specialization 与 Stage1/Stage2 融合边界；
6. 解决上游 cache shape 声明与 launcher 实际 position-major 语义的集成契约。

优化优先级应由 profiler 和端到端占比决定，而不是继续无目标减少几条指令。

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

### 62. 这个项目最大的技术难点是什么？

一是数学正确性：四个 KV group 不能为了填满 WMMA 而错误拼成 dense GEMM；
二是数值正确性：layout 错误可能保持 LSE 正常却破坏最终 output；三是性能
归因：反量化、Tensor Core、Shared Memory、barrier 和寄存器互相制约。

真正困难的是同时保证量化语义、Attention 语义和 GPU 映射正确，再通过公平
实验判断哪一项优化确实转化为性能。

### 63. 如果面试官问“为什么不用现成 FlashInfer”，怎么回答？

FlashInfer 是标准 KV Cache Attention 的高性能实现和重要参考，但当前输入是
TurboQuant 特定的 4-bit index、centroid/norm 和 scale/zero layout，不能直接
当作普通 FP16 paged KV Cache 传入。先完整反量化再调用 FlashInfer 会增加
Global Memory traffic。

因此项目保留量化专用 decode，并迁移适合的执行思想，例如 register-resident
softmax state。未来也可以把 TurboQuant dequant iterator 接入更通用的
FlashInfer 调度框架，但需要处理数据布局和模板接口。

### 64. 如果换成 RTX 3090，结果会一样吗？

不会。3090 是 `sm_86`，4090 是 `sm_89`，二者的 SM 数量、时钟、缓存、显存
带宽和调度行为不同。V8/V9 还显式依赖 `sm_89` inline PTX/fragment contract。
移植时需要重新编译、验证 lane mapping、检查 SASS、重跑 correctness 和性能，
不能只修改编译架构字符串后沿用 4090 数字。

### 65. 面试时如何证明这是工程优化，不是只调参数？

应展示完整证据链：先固定 workload 和 reference；用版本隔离单一变化；每版
验证 partial output、LSE 和最终 output；再用 CUDA Event 看稳定收益，用
cuobjdump 检查 register/shared/HMMA/barrier；对 V8 还要解释为什么新的 MMA
映射数学上成立，对 V9 解释为何减少一次 shared round trip。

比起背诵“用了 Shared Memory、Tensor Core、FlashInfer”，这种从问题、假设、
实现、证据到限制的闭环更能体现性能工程能力。

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

### 72. Lloyd-Max 中相邻 centroid 的 decision boundary 怎么计算？

在标量平方误差准则下，第 $i$ 和第 $i+1$ 个 centroid 之间的边界是二者中点：

$$
b_i=\frac{c_i+c_{i+1}}{2},\qquad i=0,\ldots,14.
$$

因为在这个位置有 $(x-c_i)^2=(x-c_{i+1})^2$。运行时 bucketize 只需将输入和
这 15 个 midpoint 比较，就能得到 4-bit index。

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
