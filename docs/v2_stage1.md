> **Historical pre-fix report (2026-08-13).** This document profiles the old
> SoA Triton V2 whose V path was later proven incorrect. The `2.29 ms` timing
> is invalid as a baseline. The NCU/SASS observations may still guide
> investigation, but exact metrics must be recollected on V2-fixed. See
> `../README.md` for current correctness and performance results.

收到，这个 SASS 文件已经足够了。**现在我们可以结束 Triton baseline 的“定位阶段”，开始设计 CUDA V1。**

而且这次 SASS 把两个最关键的问题直接钉死了。

## 1. Packed K/V 的 global load 确实很差

你看这一组：

```text
0x...a3b0  LDG.E.U8 R15, [R12.64]
0x...a3c0  LDG.E.U8 R18, [R12.64+0x2]
0x...a3d0  LDG.E.U8 R20, [R12.64+0x4]
...
0x...a420  LDG.E.U8 R23, [R12.64+0x7]
```

每条都是：

```text
Access Size = 8 bit
L2 Total    = 524288 sectors
L2 Ideal    =  65536 sectors
```

也就是：

```text
实际 sectors / ideal sectors
= 8
```

**8 倍。**

这组地址没有 `+0x40`，和我们 SoA layout 中 packed K 的起始位置一致，因此高度对应 **packed K 的 64B index load**。

更明显的是后面：

```text
LDG.E.U8 [R28.64+0x40]
LDG.E.U8 [R28.64+0x41]
...
LDG.E.U8 [R28.64+0x47]
```

这里：

```text
0x40 = 64 bytes
```

而我们 layout 正好：

```text
[K packed 64B][V packed 64B]
              ↑
             +0x40
```

所以这一组基本可以确定就是 **packed V load**。

而每条：

```text
L2 Total = 4,194,304
L2 Ideal =   524,288
```

同样又是：

```text
8×
```

也就是说现在 Triton 生成的是大量：

```text
LDG.E.U8
LDG.E.U8
LDG.E.U8
...
```

而且线程间地址 mapping 没有形成我们希望的连续/coalesced transaction。

这就解释了之前 NCU：

```text
Global excessive sectors ≈ 74%
```

不是虚的。

---

# 2. CUDA 第一个非常明确的目标出现了

不能继续照着 Triton：

```text
一个 lane
→ scalar byte load
→ scalar byte load
→ scalar byte load
...
```

我们应该自己控制线程映射，让：

```text
warp
│
├ lane0  → byte 0...
├ lane1  → byte ...
├ lane2
...
└ lane31
```

合作读取连续的：

```text
K packed = 64B
V packed = 64B
```

也就是说 CUDA V1 一开始就应该考虑：

```text
coalesced packed load
+
32-bit / 64-bit / 128-bit vectorized access
```

而不是每个元素走独立 `uint8_t` load。

注意：

> **vector load 本身不是重点，重点是 lane→address mapping 连续。**

如果地址 mapping 还是乱的，写 `uint4` 也救不了。

---

# 3. Shared-memory 问题也已经通过 SASS 证实

这里特别漂亮。

你有：

```text
LDS.128 R32, [R65+0x40]
LDS.128 R28, [R65]
...
LDS.128 ...
```

紧接着：

```text
HMMA.16816.F32
```

例如：

```text
LDS.128
LDS.128
...
HMMA.16816.F32
...
LDS.128
HMMA.16816.F32
```

这说明：

> Triton 为 `tl.dot` 的 Tensor Core operand 准备，确实生成了 shared-memory staging/load。



而且这些 `LDS.128` 的 NCU 数据非常醒目：

```text
L1 Conflicts Shared N-Way = 8

Shared wavefront:
Actual = 4,194,304
Ideal  = 2,097,152
```

也就是：

```text
2× wavefront
```

同时还有 `8-way` shared conflict 指标。

所以我们之前说：

```text
tl.dot
↓
Triton shared operand staging
↓
shared-memory pressure
```

现在已经不只是猜测了。

---

# 4. 更直接的是 Tensor Core 第二条路径

后面你还有：

```text
STS.128 [R60+0x3000], R28
STS.128 [R51+0x3000], R32
BAR.SYNC
LDSM.16.MT88.4
...
HMMA.16816.F32
HMMA.16816.F32
LDSM.16.MT88.4
```



这里结构非常典型：

```text
register
   ↓
STS
   ↓
shared memory
   ↓
BAR.SYNC
   ↓
LDSM
   ↓
Tensor Core HMMA
```

所以现在我们已经能把之前的：

```text
MIO Throttle
Short Scoreboard
Barrier
```

和真实 SASS 对上了。

---

# 5. Barrier 也不是偶然出现的

SASS 里不是一两个：

```text
BAR.SYNC.DEFER_BLOCKING
```

而是很多个。

例如：

```text
0x...b230 BAR.SYNC
...
0x...bbb0 BAR.SYNC
0x...bc00 BAR.SYNC
0x...bc40 BAR.SYNC
...
0x...c580 BAR.SYNC
0x...c660 BAR.SYNC
...
```



所以之前：

```text
Barrier stall = 45,170 samples
```

也有非常明确的 SASS 来源。

---

# 6. 现在整个 Triton V2 瓶颈已经能串起来了

我们现在有一条完整证据链：

```text
                    Triton V2
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
    Packed K/V load              tl.dot / HMMA
          │                           │
      LDG.E.U8                shared staging
          │                           │
    lane mapping 差             STS / LDS / LDSM
          │                           │
   global sectors 8×               BAR.SYNC
          │                           │
          ↓                           ↓
 Long Scoreboard               MIO / Short SB
          │                           │
          └──────────────┬────────────┘
                         ↓
               warp 大量时间在等待
                         ↓
            Eligible warp/scheduler
                    = 0.40
                         ↓
               No Eligible = 75.7%
```

同时：

```text
shared = 17.92 KB / block
        ↓
5 blocks / SM
        ↓
20 warps / SM
        ↓
41.3% occupancy
        ↓
隐藏这些 latency 的能力又不足
```

这已经是一个相当完整的 NCU + SASS 性能诊断了。

---

# 7. 现在 CUDA V1 应该怎么设计？

这里我们不要一口气做 7 个优化。

**CUDA V1 的目标是：语义正确 + 从执行映射上避开 Triton 最明显的问题。**

我建议第一版采用：

```text
一个 CTA
=
(batch, KV head, split)
```

和 Triton V2 完全一样。

所以：

```text
grid
=
64 × 8 × 32
=
16384 CTAs
```

保持不变。

一个 CTA：

```text
128 threads
= 4 warps
```

而 GQA 正好：

```text
1 KV head
→ 4 Q heads
```

于是天然设计成：

```text
Warp 0 → Q head 0
Warp 1 → Q head 1
Warp 2 → Q head 2
Warp 3 → Q head 3
```

这就非常漂亮：

```text
CTA
│
├ Warp0 → Q0
├ Warp1 → Q1
├ Warp2 → Q2
└ Warp3 → Q3

       ↑
共享同一个 KV head
```

---

# 8. 这和 Triton 最大的差异

Triton：

```text
真实 GQA rows = 4

但是
BLOCK_M = 16

↓
Tensor Core
16×128 × 128×16
```

我们的 CUDA V1：

```text
真实就是 4 Q heads

↓
4 warps

↓
不人为扩成 16 Q rows
```

也就是说：

> **第一版先不用 Tensor Core。**

这一点看起来可能反直觉，但非常值得做实验。

因为现在 NCU 已经告诉我们：

```text
Compute throughput = 33%
```

真正顶住的不是 Tensor Core 算力。

反而 `tl.dot` 带来的：

```text
shared staging
barrier
occupancy
MIO
```

非常明显。

所以我们可以测一个非常重要的问题：

> **对于 GQA=4，warp-level FP32/FMA 路径，虽然放弃 Tensor Core，但能否凭借更低 shared-memory、更少 barrier、更好的 load mapping，整体打赢 Tensor Core Triton？**

这就是一个很好的 CUDA optimization experiment。

---

# 9. CUDA V1 的数据流

一个 CTA 处理：

```text
1 batch
1 KV head
1 split = 128 tokens
```

对于一个 token：

```text
                packed K 64B
                      │
             coalesced load
                      │
              4bit unpack
                      │
             centroid lookup
                      │
               K[128]
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Warp0        Warp1       Warp2       Warp3
        Q0           Q1          Q2          Q3
          │           │           │           │
          └────── Q·K reductions ────────────┘
                      │
              online softmax
                      │
                packed V 64B
                      │
              coalesced load
                      │
               V dequant
                      │
               acc[128]
```

这里 K/V：

```text
一份 KV
```

对应：

```text
4 个 Q
```

应该尽可能只读/解码一次，然后给四个 warp 复用。

---

# 10. 第一版特别要避免什么

不要上来：

```text
cp.async
double buffering
Tensor Core WMMA
crazy prefetch
128-bit everywhere
```

那又会混在一起。

CUDA V1 只验证三个东西：

1. **4Q:1KV 的自然 warp mapping**
2. **Packed K/V 合并读取**
3. **尽量低 shared-memory / 低 barrier**

然后结果比较：

```text
Triton V2   ≈ 2.29 ms
CUDA V1     = ?
```

再 NCU。

如果 CUDA V1 只有：

```text
1~2 KB shared/block
```

甚至更低，

occupancy 从：

```text
41%
```

明显上来；

同时：

```text
Long Scoreboard ↓
MIO Throttle ↓
Barrier ↓
```

哪怕 CUDA V1 latency 暂时还没打赢 Triton，我们也知道设计方向是对的。

---

# 现在你该做什么

现在 Profiling 阶段可以先停。

**不要再跑新的 NCU。**

进入：

```bash
cd cuda
```

我们下一步正式创建：

```text
tq4_cuda_v1.cu
```

但是这里有一个工程问题：Python 里那份 synthetic `kv_cache / q_rot / block_table / centroids` 现在都在 PyTorch GPU tensor 里。

所以为了保证：

```text
Triton
和
CUDA
```

吃**完全相同的数据**，下一步我不建议写一个完全独立、自己重新 random 的 `.cu` benchmark。

更好的结构是：

```text
Python benchmark
      │
      ├── Triton V2
      │
      └── CUDA extension
              ↓
        tq4_cuda_v1.cu
```

也就是说使用：

```text
PyTorch C++/CUDA Extension
```

直接把现在的：

```text
q_rot
kv_cache
block_table
seq_lens
centroids
```

传给 CUDA kernel。

这样才能真正做到：

```text
same input
same layout
same output
same benchmark process
```

然后直接：

```python
mid_triton = triton_stage1(...)
mid_cuda   = cuda_stage1(...)

compare(mid_triton, mid_cuda)
```

**这个工程结构比写两个完全分开的 executable 更适合我们现在的目标。**

所以我们的下一步应该是建立：

```text
cuda/
├── tq4_cuda_v1.cu
└── tq4_cuda_v1_bind.cpp

baseline/
└── bench_cuda_v1.py
```

然后 `bench_cuda_v1.py` 直接复用你已经写好的：

```text
tq4_common.py
```

这就是真正进入 CUDA V1 了。

下一条我可以直接从 **`tq4_cuda_v1.cu` 的线程映射和数据读取**开始，给你第一版完整代码。
