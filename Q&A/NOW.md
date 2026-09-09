# TurboQuant CUDA Decode 面试问答｜完整版

覆盖所上传题库的全部 **167 道问题**，保留原题顺序；每题包含完整回答和一个针对性追问及答案。内容依据本次压缩包的源码、README 和实验记录核对。文中的 vLLM 行为指上传快照，性能数字为仓库已有记录，本次整理没有重新运行 GPU benchmark。

## 项目介绍

### 一分钟版本

> 我的项目围绕大模型 Decode 阶段的 KV Cache 压缩与 Attention Kernel 优化展开。我基于上传快照中的 TurboQuant 4bit_nc 格式，为 RTX 4090 上 Qwen3-4B 形状的 GQA Decode 实现和优化 CUDA 路径，让 GPU 直接读取压缩 cache，在同一个 Stage1 中完成解包、K 查表、V 反量化、QK、online softmax 和 PV，再用 Stage2 合并 split。
>
> 优化从 CTA 复用和 warp 映射推进到 tiled WMMA、固定形状特化、fragment 直接写回、32-bit packed load 和 barrier 合并。V8 用原生 m16n8k16 把四个 Query 的有效槽位比例从 25% 提高到 50%，V9 再消除 QK score 的 shared 往返。
>
> 仓库记录中，KV slot 从 512 B 降到 134 B，约压缩 3.82 倍；V7 Full 从 Triton 的 1.104148 ms 降到 0.664842 ms，约 1.66 倍，V9 Stage1 为 0.484516 ms。完整计时范围是预旋转 Query 下的 Stage1+Stage2，不包含 Store、rotation 和整模型推理。

---

## 核心问题

### 1. 这个项目解决什么问题？

> 我这个项目解决的是大模型 Decode 阶段压缩 KV Cache 的高效读取和计算问题。长上下文下，每生成一步都要读取大量历史 K/V；把它们压缩到 4 bit 可以节省容量，但也引入了解包、查表和反量化。如果先恢复完整 FP16 K/V 再计算 Attention，又会产生中间张量的显存写回与读取。
>
> 所以我基于上传项目中的 TurboQuant 4bit_nc 格式，把 packed K/V load、K 的 centroid lookup、V 的 affine reconstruction，以及 QK、online softmax、PV 融合进一个 Stage1 Kernel。一个 CTA 复用同一 KV head 的数据服务四个 Query head，序列分片后再由 Stage2 归并。
>
> 当前每个 token/KV-head 的缓存从 FP16 的 512 B 变成 134 B，逻辑压缩约 3.82 倍。仓库记录中，RTX 4090 上 V7 Full 为 0.664842 ms，相对 Triton Full 的 1.104148 ms 加速约 1.66 倍；V9 的 Stage1 达到 0.484516 ms。这里研究的是固定 workload 的 Decode 计算路径，完整模型和生产 backend 集成还需要另外验证。

**可能追问：压缩率和 Kernel 加速比为什么要分开报？**

> 压缩率只由存储格式决定，本项目是 512/134≈3.82 倍；加速比还受反量化、同步、访存层级和归并开销影响。实际报告我会分别给出格式容量、V9 Stage1 时间以及 V7 Full 时间，避免用容量收益替代性能测量。

---

### 2. 为什么 Decode 阶段特别关注 KV Cache？

> Prefill 可以通过较大的矩阵乘法获得较高计算强度，而 Decode 每一步通常只有一个新 query，却必须读取整个历史 K/V。随着 context length 增长，读取 K/V 的字节数线性增长，计算相对较少，因此容易成为 memory-bound 路径。低比特 KV Cache 的直接收益是减少每个历史 token 必须搬运的数据。
>
> 更具体地说，单步 Decode 的 Q 数量通常是每个 sequence 一个，而历史 K/V 数量是 $L$。QK 和 PV 的 useful FLOPs 都随 $L$ 线性增长，但同一个历史 K/V 在该步通常只消费一次，缺少 Prefill 中多 Query 复用，因此算术强度较低。
>
> 当前 workload 只按 compressed cache 计算的 useful AI 约为：
>
> $$
> AI_{cache}=\frac{4GD}{D+6}
> =\frac{4\times4\times128}{134}=15.28\ \text{FLOP/B}.
> $$
>
> 计入 Query 重读和 `mid_o` 写出后约为 12.31 useful FLOP/B，远低于 RTX 4090 dense FP16 Tensor ridge point。因此静态 Roofline 判断它偏 memory side，但是否由 DRAM 饱和、L2、lookup dependency 或同步限制，还要结合当前版本 NCU。

**可能追问：GQA 已经复用 K/V，为什么 Decode 仍然值得压缩？**

> GQA 把同一份 K/V 复用于多个 Query head，但每步仍要扫描长序列的历史 cache。本项目四个 Query 共用一份 K/V 后，每个 token/head 仍要读 134 B，计算强度也不高，所以复用与压缩是可以叠加的收益。

---

### 3. 为什么不能直接先反量化，再调用普通 FlashAttention？

> 这样会先读 INT4 cache，再把完整 FP16 K/V 写回显存，随后 Attention 又重新读取 FP16 K/V。额外的中间张量、写回流量和 Kernel launch 会抵消压缩收益。本项目选择在 tile 内解码到 Shared Memory，然后立即用于 Tensor Core QK/PV，解码结果不落到全局内存。
>
> 可以直接做一笔流量账。每个 `(token,kv_head)`：
>
> - compressed input 是 134 B；
> - 若物化完整 FP16 K/V，需要额外写 512 B；
> - 后续 attention 还要再次读取这 512 B；
> - 还需要保存临时 tensor、增加至少一个 Kernel 边界。
>
> 融合路径读取 134 B 后只在 Shared Memory/寄存器中形成 tile 级 FP16 数据，使用完即覆盖。这里不是说 FlashAttention 算法不适用，而是普通 FP16 FlashAttention 接口不直接理解 TurboQuant 的 nibble、centroid 和 metadata；正确方向是把量化 Decode 融入 attention tile load，而不是先全量反量化。

**可能追问：先全量反量化的路径还有没有价值？**

> 有，它适合做正确性参考和性能对照，能把压缩格式解析与 Attention 计算分开排错。但作为热路径，要把反量化写回、临时显存和后续读取都计入时间，才能判断融合是否确实更有收益。

---

### 4. 项目的固定 workload 是什么？

> 我用来比较各版本的是一组 Qwen3-4B Attention 形状的固定 workload：RTX 4090、batch 64、context 4096、32 个 Query head、8 个 KV head、head_dim 128，GQA ratio 为 4。KV physical block 包含 16 个 token，序列分成 32 个 split，所以每个 split 正好 128 个 token，也就是八个 16-token tile。
>
> K 使用 4-bit Lloyd-Max index 和 corrected norm，V 使用 4-bit uniform index、scale、zero。Stage1 的 grid 为 (64,8,32)，共 16,384 个 CTA，每个 CTA 128 个线程。它输出 FP32 的 mid_o，形状为 [64,32,32,129]。
>
> 需要区分测试参数和代码约束：B=64 是报告中的实验规模，V9 launcher 实际从输入读取 batch；D、head 数、32 splits、block size 和每 split 128 token 则进入了固定特化逻辑。因此换 batch 可以构造另一组实验，换 context 或 GQA 则不能默认直接支持。

**可能追问：Batch=64 是源码强制要求吗？**

> 它是报告中的测试配置。V9 host launcher 从 q_rot 读取实际 batch 并据此启动 grid，没有把 B 写死为 64；但 Hq=32、D=128、32 splits 和每 split 128 token 等约束仍存在。支持另一种 batch 的形状不等于已经验证它的性能。

---

## TurboQuant 量化原理

### 5. TurboQuant 与普通 per-tensor INT4 有什么区别？

> TurboQuant 的 K 路径不是简单线性量化。它先对 K 向量归一化并做正交旋转，让各坐标分布更稳定，然后使用预先计算的 Lloyd-Max centroid codebook 做非均匀标量量化。Cache 保存每个坐标的 4-bit centroid index，以及每个 token、每个 KV head 的 K norm。
>
> V 不参与 QK 内积，本项目对 V 使用更常见的 per-token/per-KV-head uniform 4-bit 量化，保存 index、scale 和 zero。
>
> | 对比项 | K 路径 | V 路径 |
> |---|---|---|
> | 量化前处理 | 向量归一化 + 正交/Hadamard 旋转 | 直接对原 V 向量求 min/max |
> | 4-bit 含义 | 16 个非均匀 Lloyd-Max centroid 的 index | `0..15` 均匀整数 index |
> | metadata 粒度 | 每个 `(token,kv_head)` 一个 corrected norm | 每个 `(token,kv_head)` 一组 scale/zero |
> | Decode 重建 | `centroid[index] * norm` | `index * scale + zero` |
> | 是否模型校准 | codebook 由理论分布和 `(D,bits)` 决定 | scale/zero 由当前 V 向量决定 |
>
> 因此“TurboQuant 就是普通 INT4”不准确：存储位宽相同，但 K 的坐标变换、非均匀 codebook 和 Query matching rotation 都与普通线性 INT4 不同。

**可能追问：K 的 4-bit 能直接送进 INT4 Tensor Core 吗？**

> 不能按当前语义直接送入。K nibble 是非均匀浮点 centroid 的编号，两个编号的整数乘积不等于重建坐标的乘积。源码先查表并乘 corrected norm，转成 FP16，再使用 FP16 输入、FP32 累加的 Tensor Core MMA。

---

### 6. 为什么要对 K 做旋转？

> 旋转的目的是混合 Key 的坐标，减弱原始坐标尺度不均对一套标量量化器的影响。Store 先按向量归一化，再做正交变换；在理想随机旋转假设下，坐标边缘分布可用高维球面分布及其 Gaussian 近似描述，适合离线生成统一 codebook。
>
> 正交变换本身保持范数，Q/K 使用匹配变换时也保持未量化的内积。按行向量约定，K_r=KΠᵀ、Q_r=QΠᵀ，因此 Q_r K_rᵀ=QΠᵀΠKᵀ=QKᵀ。真正引入误差的是之后的量化与有限精度重建，旋转不会自动使量化无损。
>
> 当前快照使用归一化 Sylvester Hadamard，属于固定的结构化正交变换。它具有正交性，但不能据此断言任意输入都变成 Haar 随机方向或精确 Gaussian；理论分布假设和工程路径要分别说明。

**可能追问：固定 Hadamard 是否保证任何 K 都变成 Gaussian？**

> 不保证。正交性保证范数与匹配内积保持，Gaussian 边缘分布论证还依赖随机旋转或数据分布假设。当前快照用固定归一化 Hadamard，不能把它当作对任意输入都成立的 Haar 随机旋转。

---

### 7. 为什么 Query 也要旋转？

> Cache 中保存的是旋转坐标系下的 K index。如果 Q 仍在原坐标系，QK 就不再对应原始 Attention score。Store 阶段旋转 K，Decode 前旋转 Q，使二者处在同一个正交坐标系。项目 Stage1 benchmark 从预先生成的 `q_rot` 开始，因此 Query rotation 不计入 Stage1 时间。
>
> 若按行向量记法，Store 保存的是 $K_r=K\Pi^T$，Decode 应使用$Q_r=Q\Pi^T$：
>
> $$
> Q_rK_r^T=Q\Pi^T(K\Pi^T)^T
> =Q\Pi^T\Pi K^T=QK^T.
> $$
>
> 必须同时满足两个条件：$\Pi$ 正交，即 $\Pi^T\Pi=I$；Q 和 K 使用同一变换及一致的左右乘约定。V 不参与 QK score，不需要为了保持该内积而做同样旋转。
>
> 项目中 `q_rot` 是 `[B,Hq,D]` FP32 输入，说明 rotation 已在计时区间外完成；回答性能时必须主动指出这一点。

**可能追问：能否在 Decode 时把 K 旋转回去，从而省掉 Query rotation？**

> 数学上可以重建原坐标系的 K，但那会对大量历史 Key 重复做逆变换。每步只旋转少量 Query，再直接消费旋转域 K，更符合 Decode 的复用关系。当前计时从预旋转 Query 开始，旋转的系统成本要另测。

---

### 8. Lloyd-Max codebook 是什么？

> 4 bit 对应 16 个 centroid。Lloyd-Max 根据目标概率分布迭代更新量化区间边界和区间条件均值，使标量均方误差降低。Store 用 15 个 midpoint 对旋转后的 K 坐标做 bucketize，最终只保存 0 到 15 的 index；Decode 根据 index 查 16-entry FP32 centroid table。
>
> Lloyd-Max 反复执行两个条件：
>
> 1. 固定 centroid 时，最小平方误差边界为相邻 centroid 中点：
>
>    $$
>    b_i=\frac{c_i+c_{i+1}}{2};
>    $$
>
> 2. 固定区间 $[b_{i-1},b_i]$ 时，新 centroid 是该区间的条件均值：
>
>    $$
>    c_i=\frac{\int_{b_{i-1}}^{b_i}x f(x)\,dx}
>    {\int_{b_{i-1}}^{b_i}f(x)\,dx}.
>    $$
>
> vLLM 对 $d\ge64$ 使用 $N(0,1/d)$ 近似，通过数值积分迭代求解，不需要先采样真实模型 KV 数据。Store 使用 boundary，Decode 使用 centroid；cache 只保存 index。
>
> 当前 solve_lloyd_max 的代码始终使用 Gaussian PDF，注释中的 d≥64 描述近似适用依据；函数没有为低维自动切换精确球面分布的分支。

**可能追问：它和对模型数据做 K-means 有什么不同？**

> 这里用一维理论概率密度求 Lloyd-Max 条件，输入是维度和位宽，不是模型 KV 样本。虽然都更新代表值，但当前 codebook 没有训练数据采集过程；每个向量的幅值变化由单独的 norm metadata 表示。

---

### 9. K 是怎样量化和恢复的？

> Store 的主要过程是：
>
> 1. 计算每个 K 向量的二范数；
> 2. 归一化并通过 GEMM 完成旋转；
> 3. 根据 midpoint 对每个坐标做二分 bucketize；
> 4. 两个 4-bit index 打包成一个 byte；
> 5. 保存校正后的 K norm。
>
> Decode 对每个 nibble 查 centroid，再乘对应 `(token,kv_head)` 的 norm，得到 tile 内用于 QK 的 FP16 K。完整 K 不写回 Global Memory。

**可能追问：K 全为零时，归一化和重建会怎样？**

> Store 用 norm 加小量避免归一化除零，量化索引仍会生成，但保存的 corrected norm 为零，因此重建 K 为零。现有随机输入验证还要求 K norm 严格为正，这个辅助检查没有覆盖零向量，扩展测试时应单独处理该合法边界。

---

### 10. norm correction 是什么？

> 量化后的 centroid 向量范数不一定恰好为 1。Store 在开启 norm correction 时，将 centroid 向量的逆范数折叠进保存的标量：
>
> $$
> \gamma_{stored} = \frac{\lVert K\rVert_2}{\lVert c\rVert_2}.
> $$
>
> Decode 只需要计算 `centroid[index] * gamma_stored`，不必在每个 tile 重新求 centroid 向量范数。这是用 Store 阶段一次计算换 Decode 热路径更少的操作。
>
> 这里的粒度必须说清：对每个 token、每个 KV head 的 128 维 K 向量分别计算$\gamma_{t,h}$，不是每层一个，也不是整个 KV head 跨 token 共用。若一次 Store 有 $N$ 个 token、$H_{kv}$ 个 head，就保存 $N\times H_{kv}$ 个 FP16 corrected norm。
>
> 它同时承担两个作用：
>
> - 恢复归一化前原始 K 的幅值 $\lVert K\rVert_2$；
> - 修正 centroid reconstruction 后方向向量范数不再恰好为 1 的误差。
>
> 因此 `nc` 是 norm correction，不是 V affine quantization 的 scale。

**可能追问：norm correction 能否恢复原始 Key 的全部信息？**

> 不能。它主要校正量化向量的长度，无法恢复被 4-bit 索引丢掉的方向信息。忽略 metadata 舍入和稳定项时，重建向量的范数可接近原始范数，但 QK 内积和最终 Attention 仍有量化误差。

---

### 11. V 为什么不用同样的 centroid 量化？

> K 直接决定 QK score，对内积误差敏感；V 在 softmax 权重确定后参与加权求和，工程实现选择更简单的 per-token/per-KV-head uniform quantization。4-bit V 使用：
>
> $$
> v_{recon} = index \times scale + zero, \qquad index\in[0,15].
> $$
>
> 这样 Decode 只需 nibble unpack、整数转浮点和一次乘加，不需要 centroid lookup。
>
> 具体公式是：
>
> $$
> v_{min}=\min_dV_d,\qquad
> scale=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right),
> $$
>
> $$
> q_d=\mathrm{clip}\left(\mathrm{round}
> \left(\frac{V_d-v_{min}}{scale}\right),0,15\right).
> $$
>
> cache 中 `zero` 保存的是 FP16 `v_min`，不是整数 zero-point。K 与 V 采用不同量化器，是因为 K 误差先进入指数敏感的 QK/softmax，V 误差在线性加权路径中传播；这是算法与工程开销的折中，不表示 V 精度不重要。

**可能追问：V 的 zero 是整数 zero-point 吗？**

> 这里不是。字段名虽然叫 zero，实际保存的是 FP16 的向量最小值 v_min，重建为 index×scale+v_min。若误写成 (index-zero)×scale，就改变了 cache 契约，输出会错而 QK/LSE 可能仍然正常。

---

### 12. 当前实现使用 QJL residual 吗？

> 没有。当前研究的是 vLLM `turboquant_4bit_nc` 路径，使用 rotation、centroid、 norm、V scale/zero，不保存 QJL residual。不能把论文中更广泛的 TurboQuant 变体全部说成当前 Kernel 已实现的功能。
>
> 需要区分三层：
>
> - **TurboQuant-MSE**：旋转后用 Lloyd-Max centroid 最小化坐标重建 MSE；
> - **TurboQuant-Prod**：在 MSE reconstruction 外保留 residual，并用 QJL 估计
>   residual 与 Query 的内积修正；
> - **当前 vLLM `4bit_nc` 与本项目**：使用 centroid index + corrected norm，
>   不保存 QJL projection/sign 或 residual norm。
>
> 所以论文设计 QJL 不代表所有部署 preset 必须使用。当前 Kernel 的参数列表和 134 B slot 中都没有 QJL payload；如果面试官追问，应从代码数据契约回答，而不是把论文所有分支混成一个实现。

**可能追问：怎么从 cache 布局确认没有 QJL？**

> 当前 K64 B、V64 B，再加 K corrected norm、V scale、V zero 三个 FP16 标量，合计 134 B。源码没有为残差符号和 residual norm 预留字段，Decode 也没有残差内积估计通道，因此不能把这条路径叫 TurboQuant-Prod 的完整实现。

---

### 13. 4-bit 打包具体节省多少空间？

> head dimension 是 128。K 的 128 个 4-bit index 占 64 B，V 也占 64 B；另有三个 FP16 metadata：K norm、V scale、V zero，共 6 B。因此一个 token/KV-head 的逻辑 slot 是：
>
> ```text
> K index   64 B
> V index   64 B
> metadata   6 B
> total     134 B
> ```
>
> 相比只计算 K/V payload 的 FP16 `128*2*2 = 512 B`，payload 大约缩小 4 倍；实际压缩率还应计入 metadata 和 block/page 管理开销。
>
> 把 metadata 纳入逻辑 slot 后，准确压缩比是：
>
> $$
> \frac{512}{64+64+2+2+2}=\frac{512}{134}=3.8209\times.
> $$
>
> 固定 workload 下共有 $64\times4096\times8=2{,}097{,}152$ 个逻辑 slot：
>
> - packed K/V payload：256 MiB；
> - K norm、V scale、V zero：12 MiB；
> - 合计：268 MiB。
>
> 这个数字不含 block table、allocator 对齐、空 physical page 和其他模型张量；因此可以说“KV slot 逻辑压缩 3.82x”，不能说“整模型显存下降 3.82x”。

**可能追问：3.82 倍是否等于整个模型显存降低 3.82 倍？**

> 不是，它只针对这里的 K/V cache payload 与 metadata。模型权重、激活、Query、split 中间结果、allocator 预留和其他状态都不在这个分母里。报告模型级显存收益需要在完整推理进程中单独测峰值。

---

## Cache 布局与 Paged Attention

### 14. AoS 与 SoA 在这里分别是什么？

> AoS 把一个 token/head 的 packed K/V 和 metadata 更紧密地放在一起。SoA 将大块数据区和 metadata 区分开：
>
> ```text
> data:     [token][kv_head][K64 | V64]
> metadata: [kv_head][field][token]
> ```
>
> Decode 对连续 token 批量读取相同 field 时，SoA 的 metadata 地址更规则；实测仅从 AoS Triton V1 改为 SoA Triton V1，就从 1.692 ms 降到 1.284 ms，约 1.318x。
>
> 当前 SoA 不是简单把整个 cache 变成 `[field][all tokens]`，而是混合布局：
>
> - payload：`[position][kv_head][K64 | V64]`，让一个 token/head 的 K/V 紧邻；
> - metadata：`[kv_head][field][position]`，让同一 head 的 16 个 norm/scale/zero
>   可由连续 lane 合并读取。
>
> 因此 AoS→SoA 的收益主要来自 metadata transaction 和地址规则性，不意味着 payload 总字节数发生变化。AoS V1 与 SoA V1 的 1.318x 是 layout ablation；不能把它算作 CUDA V1→V9 的优化收益。

**可能追问：SoA 表示 K 和 V 在整个显存中完全分开吗？**

> 不是。本项目主要把 metadata 按字段拆到 block 尾部；payload 仍按 position、KV head 排列，每个 token/head 内 K64 B 和 V64 B 相邻。解释布局时必须区分数据区和 metadata 区，不能只背 SoA 的通用定义。

---

### 15. 一个 physical block 的字节布局是什么？

> 一个 physical block 存 16 个 token、每个 token 有 8 个 KV head。它先存所有 K/V packed payload，再存 SoA metadata，而不是每隔 134 B 就放一组完整字段。
>
> | 区域 | 物理排列 | 字节数 |
> |---|---|---:|
> | 数据区 | [position=16, kv_head=8, K64 B与V64 B] | 16,384 B |
> | metadata 区 | [kv_head=8, field=3, position=16]，FP16 | 768 B |
> | 整个 block | 数据区之后紧接 metadata | 17,152 B |
>
> 设 block_base=physical_block×cache_stride，则 data_base=block_base+position×1024+kv_head×128。K 从 data_base 开始，V 从 data_base+64 开始。metadata 地址为 block_base+16384+((kv_head×3+field)×16+position)×2，其中 field 依次表示 K corrected norm、V scale 和 V zero。
>
> 134 B 是每个逻辑 slot 的等效容量，便于计算压缩率；真实 payload 的 head stride 是 128 B。这个区别直接决定 CUDA 的地址计算是否正确，也解释了为什么不能按普通 AoS 的 134 B stride 去读取 SoA payload。

**可能追问：既然 tensor shape 最后一维是 134，能否直接按 slot 的 134 B stride 读 K？**

> 对当前物理 SoA 格式不能这样读。134 是等效容量，实际 payload stride 是每 head 128 B，metadata 另放在 block 尾部。CUDA 按字节偏移解释缓存，张量外观形状并不表示每个逻辑 slot 的三个字段紧随 payload。

---

### 16. 为什么 metadata 使用 FP16？

> 我使用 FP16 metadata，是为了把每个 token/KV-head 的额外开销控制在 6 B。三个字段分别是 K corrected norm、V scale 和 V zero，加上 K64 B 与 V64 B，slot 为 134 B。若三个字段都用 FP32，就变成 140 B，metadata 流量也翻倍。
>
> Decode 将这些 FP16 标量转换成 FP32，完成 centroid×norm 或 index×scale+zero，再转成 Tensor Core 所需的 FP16 operand。metadata 因此既影响容量，也会影响重建精度。
>
> 仓库随机样本验证支持了当前计算路径的一致性，但不能证明 FP16 对所有模型和极端值都足够。特别小的 scale 可能下溢，大的 norm 可能溢出；是否改用更高精度，应以输入范围、Attention 输出和模型质量对照决定。

**可能追问：为什么不统一把 metadata 改成 FP32？**

> FP32 能减少 metadata 舍入，但会把每 slot 从 134 B 增到 140 B，并让 metadata 流量翻倍。是否值得，需要对照输出误差和模型质量；当前 4-bit 量化本身的误差也可能比这部分更大，不能只凭 dtype 下结论。

---

### 17. block table 在 Kernel 中做什么？

> Paged KV Cache 的逻辑 token 不保证位于连续 physical page。`block_table[b, logical_block]` 将序列的逻辑 block 映射到 cache 中的 physical block。固定 workload 每个 tile 恰好 16 token，因此 V4 之后每个 tile 只需读取一次 block-table entry。
>
> 对逻辑 token $t$：
>
> $$
> logical\_block=\lfloor t/16\rfloor,\qquad pos=t\bmod16.
> $$
>
> Kernel 用 `block_table[b, logical_block]` 得到 physical block。当前 16-token tile 与 page block 完全对齐，所以 tile 内 16 个位置共享同一 physical block。若 block size 改变、split start 未对齐或 tile 跨页，就必须读取多个 entry 并处理边界，当前固定 Kernel 会直接不适用。
>
> Store 使用的是 `slot_mapping`，它告诉新 token 写入哪个 physical slot；Decode 使用 `block_table`，它从历史逻辑位置找到 physical page。两者职责不同。

**可能追问：block table 随机后，一个 tile 的访问就一定不合并了吗？**

> 不一定。它改变物理页的位置和跨页局部性，但当前 tile 对齐一个 physical block，页内各 lane 的地址关系仍可保持合并。应分别观察页内 sector 效率和跨 tile 的 L2 行为，不能把随机页映射直接等同于随机逐元素加载。

---

### 18. 为什么向量化 `uint32` load 是安全的？

> 每个 packed K 或 V 区域是 64 B，slot 的 data 部分是 128 B，固定布局保证读取地址满足四字节对齐。V6 每次读取四个 packed byte，对应八个 4-bit dimension；随后通过 `half2` 写两个重建值，减少 load、地址计算和 shared store 指令。对齐约束是 V6-V9 的显式限制，不能假定任意 cache layout 都安全。
>
> 精确地说，每线程一次 `uint32_t` load 得到 4 个 byte，也就是 8 个 nibble，而不是每线程 128-bit `uint4`。安全性来自：
>
> 1. physical block base 按 cache allocation 对齐；
> 2. 每 position 的 payload stride 是 `8 * 128 = 1024 B`；
> 3. 每 head 的 data stride 是 128 B；
> 4. K 起点为 0、V 起点为 64 B；
> 5. `word * 4` 保持 4 B 对齐且不会越过各自 64 B 区域。
>
> 若 head dimension、bit width、metadata placement 或 allocator 对齐发生变化，应重新证明这些条件并增加 launcher check，不能只保留 reinterpret cast。

**可能追问：整个 block 的 17152 B stride 会破坏 4 B 对齐吗？**

> 不会，17152、每 token 的 1024 B、每 head 的 128 B，以及 V 的 64 B 偏移都能被 4 整除。再加上基地址对齐，uint32 地址成立。这个推导依赖当前布局，换成 metadata 混排的 AoS 不能直接沿用。

---

### 19. 如何证明 CUDA 读取的是真实 vLLM Store 布局？

> `validation.store_decode` 从原始 FP16 Q/K/V 开始，调用未经修改的 vLLM SoA Store 路径，先归一化和旋转，再由 Triton 执行 bucketize、packing 和 metadata 写入，然后将同一个 cache tensor 直接交给 Triton Decode 与 CUDA V7，中间没有 byte rearrangement。CUDA 与 Triton output 最大差约 `5.06e-06`，说明二者对 layout 的解释一致。
>
> 验证链路是：
>
> 数据流为：FP16 K/V → 未修改 vLLM SoA Store → packed cache + norm/scale/zero → Triton Decode 与 CUDA Decode → 比较完整 output 和 LSE。
>
> 它能发现 nibble 顺序、field offset、physical-page 地址和 metadata dtype 等契约错误。还要说明局限：该测试证明 Store→Decode 字节兼容和 kernel-level 数值一致，不等于已经覆盖 production continuous batching、所有 ragged tail 或模型 PPL。

**可能追问：现有 Store 验证也证明 V9 的 Full 链路了吗？**

> 还没有。validation/store_decode.py 当前调用 CUDA V7 Full，对同一份 Store 输出与 Triton 比较；V9 的现有验证主要在 Stage1 benchmark。要声称 V9 Full 已验证，需要接入 Stage2 后重跑真实 Store 输入与最终 output/LSE 对照。

---

## Attention 与 Split-KV

### 20. Stage1 到底计算什么？

> 一个 CTA 对应 `(batch, KV head, split)`，处理该 KV head 对应的四个 Q head 和当前 split 的 128 个 token。它循环处理八个 16-token tile，完成：
>
> 数据流为：packed K/V load -> dequant -> QK -> online softmax -> PV。
>
> 最后为每个 Q head/split 输出 128 维 partial output 和一个 split LSE。
>
> 固定 workload 的 launch 和 CTA 内工作量是：
>
> ```text
> grid             = (64, 8, 32) = 16,384 CTAs
> threads / CTA    = 128 = 4 warps
> tokens / split   = 128
> tiles / split    = 128 / 16 = 8
> Q heads / CTA    = 4
> ```
>
> 每个 tile 依次完成 page lookup/metadata load、packed K/V cooperative load、K centroid reconstruction、V affine reconstruction、QK MMA、online-softmax update 和 PV MMA。Stage1 不写完整 score matrix或 FP16 K/V，只写 split partial state。

**可能追问：Stage1 结果可以直接作为 Attention 最终输出吗？**

> 每个 split 单独归一化，只有序列的一部分信息，不能直接作为全序列结果。当前 Stage1 写出 128 维 partial output 和一个 LSE；Stage2 用 LSE 恢复各分片的相对权重，才能得到完整 Attention。

---

### 21. 为什么要把 4096 token 分成 32 个 split？

> split-KV 的目的，是沿历史序列再增加一层并行度。当前每个 sequence 有 8 个 KV head，如果不分 split，batch 64 时只有 512 个 CTA，而且每个 CTA 要顺序扫描 4096 个 token。分成 32 份后，grid 变成 16,384 个 CTA，每个只处理 128 个 token，有更多任务用于调度和延迟隐藏。
>
> 每个 CTA 仍服务同一 KV head 对应的四个 Query，因此 K/V 反量化可以组内复用。代价是 Query 被不同 split 重读、需要写出 32 份 partial output/LSE，并新增 Stage2 归并。并行度收益和中间结果成本必须一起衡量。
>
> 32 是当前固定实验的选择，不是已经证明对所有 batch/context 最优的常数。选择 split 时，我会比较不同方案的 Full 时间；源码特化也要随 split 契约一起修改，不能只改 Python 侧参数。

**可能追问：32 个 split 是通过搜索证明的全局最优吗？**

> 仓库把它作为固定 workload 的设计点，没有给出覆盖所有 split 数的最优性证明。它增加了 CTA 并行度，也增加 Query 重读和归并成本。扩展调优时应同时测 Stage1、Stage2、Full，不能只看 Stage1 最快值。

---

### 22. Stage1 输出为什么是 129 个 float？

> 前 128 个是归一化后的 partial output，最后一个是该 split 的 log-sum-exp：
>
> ```text
> mid_o shape = [B, Hq, 32, 128 + 1]
> ```
>
> Stage2 使用每个 split 的 LSE 对 partial output 做数值稳定的重新加权，不能简单对 32 份 partial output 求平均。
>
> 完整 shape 和大小是：
>
> $$
> mid\_o\in\mathbb{R}^{64\times32\times32\times129},
> $$
>
> $$
> 64\times32\times32\times129\times4
> =33{,}816{,}576\ \text{B}=32.25\ \text{MiB}.
> $$
>
> 前 128 项是该 split 内已经归一化的 output，最后一项是 split LSE。采用 FP32 是为了让跨 32 split 的重加权和 output accumulation 保持数值稳定。

**可能追问：这 129 个 float 有没有额外保存未归一化分母？**

> 没有独立分母字段。前 128 个值已经除以该 split 的指数和，第 129 个值用 LSE 编码该分片的归一化常数。Stage2 通过不同 LSE 的指数差建立相对权重，所以不需要再存一份原始 denominator。

---

### 23. Online softmax 的递推公式是什么？

> 对新 tile 的 score，先求 tile 最大值 `m_t` 和指数和 `l_t`。已有 state 为`(m,l,o)`，合并时：
>
> $$
> m' = \max(m,m_t),
> $$
>
> $$
> l' = l e^{m-m'} + \sum_j e^{s_j-m'},
> $$
>
> $$
> o' = o e^{m-m'} + \sum_j e^{s_j-m'}v_j.
> $$
>
> 最终输出 `o'/l'`，LSE 为 `m' + log(l')`。V8/V9 使用 `exp2f`，因此 score 先乘 `log2(e)`，最后再换回自然对数语义。

**可能追问：新 tile 的最大值更大时，为什么旧 output 也要缩放？**

> 旧 output 与旧指数和使用旧最大值作为数值基准。最大值变成新值后，两者都必须乘同一个 alpha，才能和新 tile 的指数权重处于相同尺度。只缩放 denominator 而不缩放 output，会直接改变 Attention 的数学结果。

---

### 24. 为什么 online softmax 比两遍算法更适合？

> online softmax 可以在只保存运行最大值 m、指数和 l、以及输出累加器 o 的情况下，逐 tile 完成 Attention。每次读入一个 K/V tile，先得到 score，更新 m/l，再把旧 o 缩放到新的指数基准，最后累加当前 tile 的 PV。这样不用先保存整个 split 的全部 score 或 probability。
>
> 它与压缩 KV 的融合很合适：K/V 解码后只在 CTA 内短暂存在，消费后即可覆盖。相比“先求全部分数，再处理权重和输出”的设计，可以减少中间状态物化或重复读取，但仍需保留跨 warp 的必要通信。
>
> 上传源码中 CUDA V1 和 V2 已经采用 online softmax。V3 的提升主要是 16-token tiled 处理、共享反量化和 WMMA QK/PV，而不是第一次从两遍算法改成 online softmax。

**可能追问：V3 是项目第一个使用 online softmax 的版本吗？**

> 不是，上传源码中的 CUDA V1、V2 已维护 running_m/running_l。V3 的关键增量是 16-token tiled、CTA 共享反量化和 WMMA QK/PV，不能把 V2 描述成先物化全部分数的两遍算法。

---

### 25. Stage2 怎样合并 split？

> 设第 $i$ 个 split 的 normalized partial output 为 $o_i$，LSE 为 $L_i$。先求：
>
> $$
> M=\max_iL_i,\qquad w_i=e^{L_i-M}.
> $$
>
> 再得到：
>
> $$
> o=\frac{\sum_iw_io_i}{\sum_iw_i},\qquad
> LSE=M+\log\sum_iw_i.
> $$
>
> 减去 $M$ 避免指数溢出。CUDA Stage2 时间为 0.008868 ms，其独立中位时间与 V7 Full 中位时间的比值约为 1.33%，但它是数学语义上必需的，不能因耗时小就省略。
>
> 这个比值用于近似判断优化优先级，不是从 Full 的同一条时间线测出的严格阶段占比。

**可能追问：如果两个 split 的 LSE 相差很大，合并会不会溢出？**

> Stage2 先减去所有 split LSE 的最大值，再计算指数，所以权重不大于 1，大的正指数溢出得到控制。较弱 split 的权重可能下溢到接近零，这是稳定 softmax 的常见行为；空 split 仍需额外定义有效性处理。

---

### 26. 为什么 Stage1 与 Stage2 使用两个 Kernel？

> Stage2 必须等所有 split CTA 写完 `mid_o`。普通 CUDA Kernel 内没有通用的 grid-wide barrier，因此使用 Kernel launch 边界表达全局同步最直接，也避免 cooperative launch 的额外约束。
>
> 替代方案各有明显代价：
>
> - atomic counter + last-CTA reduction：需要严格内存顺序，调度和复用复杂；
> - cooperative launch：要求整 grid 满足 cooperative residency，限制并行规模；
> - persistent kernel：需要重写任务调度，并可能降低不同 workload 的适应性。
>
> 由于 单独 Stage2 与 Full 中位时间的比值约为 1.33%，两 launch 的清晰同步边界是更合理的工程选择。只有在小 batch 下 launch latency 成为主要部分时，才值得重新评估融合。
>
> 这个比值用于近似判断优化优先级，不是从 Full 的同一条时间线测出的严格阶段占比。

**可能追问：同一 CUDA stream 上两个 Kernel 之间需要 CPU synchronize 吗？**

> 不需要为了数据依赖单独做 CPU 同步。同一 stream 的 launch 顺序保证 Stage2 在 Stage1 之后消费结果，源码也使用 current stream。跨 stream 消费才需要事件或其他明确依赖，计时同步与计算依赖是不同问题。

---

### 27. Stage1 时间和 Full Decode 时间为什么不能混为一谈？

> Stage1 benchmark 不含 Stage2、Query rotation、Store、输入构造和 JIT。 Full Decode 也只定义为预先旋转的 Q 和压缩 cache 经过 Stage1+Stage2，不包含 Store。简历和面试必须明确计时边界，否则 `0.485 ms` 不能被描述成完整端到端请求延迟。
>
> 项目里至少有三组必须分开的数字：
>
> | 口径 | CUDA 时间 | 对比含义 |
> |---|---:|---|
> | V9 Stage1 | 0.484516 ms | 最新 Stage1 candidate |
> | V7 Stage1 | 0.631255 ms | Full benchmark 中单独测量 |
> | V7 Full | 0.664842 ms | V7 Stage1 + CUDA Stage2 的独立 runner |
>
> Full 也不等于完整模型 Decode：它不含 QKV projection、RoPE、Query rotation、 Store、其他 layer、采样、allocation 和 JIT。简历中的“完整 Decode Kernel 链路”应解释为本项目定义的 attention Stage1+Stage2 链路。

**可能追问：能不能用 V9 Stage1 加 V7 Stage2 的独立时间得到 V9 Full？**

> 只能做粗略预估，不能当作实测成绩。两阶段中间数据兼容是必要条件，还要在同一 runner 中真实串联、验证并独立计时，才能报告 V9 Full。不同实验的中位数也不具有可加性。

---

## GQA、WMMA 与 Tensor Core

### 28. GQA-4 在这个项目中是什么意思？

> 32 个 Q head 共享 8 个 KV head，因此每个 KV head 对应四个 Q head。一个 CTA 以 KV group 为单位，解码一份 K/V tile，并为四个 Q head 复用它。这样避免每个 Q head 都重复读取和反量化同一份 K/V。
>
> 头映射是：
>
> $$
> G=H_q/H_{kv}=32/8=4,
> $$
>
> $$
> qh=kvh\times4+q_{local},\qquad q_{local}=0,1,2,3.
> $$
>
> 例如 KV head 3 服务 Q head 12–15。共享的是同一历史 K/V 向量、K norm 和 V scale/zero；不共享的是四个 Query、QK score、softmax $(m,l)$ 和最终 output。这一区分决定了哪些数据适合 CTA 共享、哪些状态必须按 Q head 独立保存。

**可能追问：GQA 是不是把四个 Query head 合并成一个 head？**

> 不是。四个 Query 共享同一组 K/V，但各有自己的 score、softmax 状态和 128 维输出。CUDA CTA 复用的是输入与反量化工作，不会平均或合并四个 Query 的语义。

---

### 29. 为什么普通 `m16n16k16` 会浪费 Tensor Core 槽位？

> V3-V7 把四个 Q head 放到 WMMA 的 M 维。硬件 tile 要求 M=16，但只有四行真实数据，其余 12 行是 padding，有效行比例只有 `4/16=25%`。虽然 Tensor Core 很快，这种结构仍执行了无效 HMMA 工作，并扩大 accumulator/scratch。
>
> 可以把 QK 看成：
>
> ```text
> Q tile:  [16 rows, 128 dims]，只有 row 0..3 是真实 Q head
> K tile:  [16 tokens, 128 dims]
> output:  [16 Q rows, 16 token columns]
> ```
>
> 每个 K=16 的 MMA step 都会计算 16 行，padding 行不会因为输入为 0 而自动免除 Tensor Core 指令。25% 是**矩阵槽位有效率**，不是 NCU 实测 Tensor Core active 百分比，也不代表整个 Kernel 只有 25% 利用率。

**可能追问：25% 的有效槽位是否等于 NCU Tensor Core utilization 25%？**

> 不是。4/16 是矩阵中真实 Query 行占比，属于算法映射；NCU utilization 描述硬件执行管线的活跃或吞吐。即使空槽很多，硬件仍可能忙于执行完整 MMA，这两个比例的定义不同。

---

### 30. 为什么不能把四个 KV head 和 16 个 Q head 直接拼成一次 dense MMA？

> 因为四个 GQA group 使用四份不同的 K/V 矩阵。普通 dense GEMM 的同一次矩阵乘法要求所有输出行共享同一个右操作数；直接堆叠会产生跨 group 的错误 Q-K 配对。除非构造 block-diagonal K，这又会引入更多零和复杂布局，因此不能只凭“16 个 Q head 正好填满 M=16”判断数学上成立。
>
> 若把四个 group 的 Query 堆成 16 行，而 K 只放某一个 group，则另外 12 行乘错 K；若把四组 K 也拼在普通 dense 右操作数中，会产生所有 Q-K group 的交叉项。数学上可构造 block-diagonal multiplication 屏蔽交叉项，但需要更大的零填充矩阵，数据布局和 output 选择成本通常抵消收益。
>
> 真正可行的跨 group 合并需要支持独立 batch/group operand 的 MMA 组织，或者让不同 warp 执行各自 MMA；不能仅以 WMMA 的 M=16 容量作为正确性依据。

**可能追问：用 block-diagonal 大矩阵装不同 KV group 可以吗？**

> 理论上可以构造带零块的大矩阵，使跨组乘积被屏蔽，但 dense MMA 仍会计算许多无用元素，还增加数据组织成本。它不是免费把四个 group 填满；当前项目选择组内转置，保持 K/V 复用和正确的 Attention 语义。

---

### 31. V8 怎样提高 GQA-4 的 MMA 利用率？

> V8 在每个合法 KV group 内转置两次乘法：
>
> ```text
> QK: K(16x16)   * Q^T(16x8)
> PV: V^T(16x16) * P(16x8)
> ```
>
> 它使用原生 `mma.sync.aligned.m16n8k16`，让四个 Q head 占 N=8 的四列，有效槽位比例从 `4/16=25%` 提高到 `4/8=50%`。静态 HMMA site 从 V7 的 160 降到 V8 的 80。
>
> 关键是把 Query head 放到 `m16n8k16` 的 N 维：
>
> - M=16 对应 16 个 token 或 output dimension 行；
> - N=8 中前 4 列对应 4 个真实 Query head；
> - 后 4 列仍为空，因此不是 100%；
> - K=16 沿 head dimension 分块，D=128 需要 8 个 K step。
>
> PV 也做匹配转置，让相同 N=8 的四列继续代表四个 Query head。QK 和 PV 必须成对调整数据布局，否则只优化其中一侧会引入额外转置或错误 output mapping。

**可能追问：m16n8k16 中的 K=16 是历史 token 数吗？**

> 要看乘法阶段。QK 的收缩维是 head_dim，D=128 要做八个 K-step；PV 的收缩维是当前 tile 的 16 个 token。相同指令形状在两阶段对应不同逻辑维度，不能仅凭字母 K 把它理解为 Key tensor。

---

### 32. 为什么 V8 不继续使用 C++ WMMA API？

> V8 需要让四个 Query head 占原生 MMA 的 N=8 维，而当前使用的 C++ WMMA 路径是 m16n16k16。为了明确控制 m16n8k16 的输入寄存器和输出列映射，源码使用 inline PTX 指令 mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32，即 FP16 输入、FP32 累加。
>
> 实现上，每个 lane 按矩阵位置打包 A 的四个 32-bit 寄存器和 B 的两个 32-bit 寄存器，接收四个 FP32 accumulator。QK 将 token 放在 M 维、Query head 放在 N 维；PV 使用匹配转置，让 Query 列的含义保持一致。
>
> 这里的寄存器布局由 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-f16) 规定，不能与 C++ WMMA fragment 未公开的内部 ABI 混为一谈。代价是代码更底层，迁移时仍需确认指令支持、编译结果、资源和正确性；它不是 RTX 4090 独占的数学映射。

**可能追问：PTX 的 fragment 映射和 WMMA fragment 内部映射是一回事吗？**

> 不是。原生 mma 指令的寄存器布局由 PTX ISA 给出，代码按该契约打包 operand；C++ WMMA fragment 的内部元素布局不属于稳定公开 ABI。两者都需要正确映射，但 V5 的经验 probe 和 V8/V9 的 PTX 规范映射应分开解释。

---

### 33. V8 节省了哪些资源？

> 相对 V7：
>
> ```text
>                     V7          V8
> register/thread     49          51
> shared/CTA          14224 B     10336 B
> static HMMA         160         80
> static BAR.SYNC     34          34
> ```
>
> V8 用两个额外 register 换取更小的 Q、QK scratch 和 PV fragment，Stage1 从 0.631 ms 降到 0.513 ms，约 1.230x。
>
> 这些数字的含义需要分开：
>
> - register/thread 和 shared/CTA 是编译后资源用量，会影响 resident CTA 上限；
> - static HMMA/BAR 是 SASS 中静态指令 site，受循环展开影响；
> - static site 减半不等于运行时间减半，也不等于动态 Tensor 指令必然减半；
> - V8 实测 1.230x 才是最终性能证据。
>
> V8 shared memory 减少 3,888 B，主要来自更紧凑的 Q/QK/P layout；register 从 49 升到 51，说明优化不是所有资源都同时下降，而是用少量 register 换取更少 Shared Memory 和无效 MMA。

**可能追问：寄存器从 49 增到 51，为什么性能反而提高？**

> 寄存器数只是约束之一。V8 同时减少无效 MMA 和 shared footprint，少量寄存器增加未必跨越 occupancy 阶梯，也可能换来更短的数据路径。最后要看编译资源、是否 spill 和同条件时间，不能按单个资源数字排名。

---

### 34. V9 借鉴 FlashInfer 的地方是什么？

> 借鉴的是 register-resident attention state，而不是直接调用 FlashInfer API 或复制它的 Kernel。V8 会把 QK accumulator 写入 `qk_s`，同步后再由四个 warp 读取并更新 softmax。V9 让 warp 0 直接按 lane class 对四个 Q-head score 列做 max/sum reduction，在寄存器中维护 `(m,l)`，只将 FP16 probability 写给后续 PV。
>
> 由于 compressed K/V 必须执行 nibble unpack、centroid lookup 和 scale/zero 重建，普通 `cp.async` 不能直接完成这段变换；因此这里优先迁移 FlashInfer 最适合当前数据路径的 state-fusion 思路。
>
> V8 的数据路径是：
>
> 数据流为：QK MMA accumulator → qk_s Shared Memory → CTA barrier → 四个 warp 读取 score → 更新各自 online-softmax state。
>
> V9 改为 warp 0 直接按 `lane % 4`/lane group 持有四个 Query head 的 score 与`running_m/running_l`，在寄存器内完成 max/sum，再只把 FP16 probability 写入`p_s` 供 PV。这样删除 512 B `qk_s` 及其 producer-consumer barrier。
>
> 借鉴的是“attention state 尽量寄存器常驻”的原则；项目没有新增 FlashInfer 依赖，也没有宣称直接采用 FlashInfer Kernel。

**可能追问：V9 完全不使用 Shared Memory 了吗？**

> 没有。Q/K/V tile、PV 需要的概率以及跨 warp 的缩放参数仍在 Shared Memory。删除的是 QK score 的一次物化和相关同步；QK score 与 softmax 标量状态留在 warp 0 寄存器中，不代表整个 Attention 都寄存器化。

---

### 35. V9 的结果和资源变化是什么？

> ```text
>                     V8          V9
> time / ms           0.513208    0.484516
> register/thread     51          50
> shared/CTA          10336 B     9824 B
> static HMMA         80          80
> static BAR.SYNC     34          26
> ```
>
> V9 删除 512 B `qk_s`，每个 tile 少一次 CTA barrier，并略降寄存器。相对 V8 提升 1.059x，相对 Triton V2-fixed 提升 2.221x。
>
> 还应补充三个限制：
>
> 1. 这是 Stage1 时间，不含 Stage2；
> 2. 50% MMA slot utilization 没有改变，因为仍是 4 个真实 Q head 填 N=8；
> 3. 当前没有正式采集 V9 在 4090 上的动态 NCU 报告，不能仅凭资源和时间断言
>    某个硬件 counter 已达到多少。
>
> 从 useful work 角度，V9 在 0.484516 ms 内完成约 4.295 GFLOPs，即 8.86 useful TFLOP/s；考虑 50% 槽位后估算 按映射推算的 dense MMA work 约 8.59 GFLOPs，即 17.73 TFLOP/s。它远低于 4090 dense Tensor 峰值，符合 memory/data-movement-side 的静态判断，但最终瓶颈仍需 NCU 证实。

**可能追问：V9 比 V8 快约 5.9%，是否说明 barrier 占总时间 5.9%？**

> 不能这样反推。改动同时改变 shared 往返、依赖、寄存器和调度，时间差是这组改动的综合效果。可以说消除 score 物化的版本取得约 1.059 倍收益，但不能把差值直接当作某类指令的独立耗时占比。

---

## CUDA V1-V9 演进

### 36. CUDA V1 的设计和问题是什么？

> V1 一个 CTA 负责一个 `(batch, KV head, split)`，四个 Q head 复用解码后的 K/V。问题是每个 token 都要为四个 head 做多轮 warp reduction，并频繁通过 Shared Memory 做 CTA 协作。它是正确、可复用 K/V 的起点，但 2.074 ms 比 Triton baseline 慢。
>
> V1 建立了后续优化必须保持的三个契约：
>
> - grid 使用 `(batch,kv_head,split)`，四个 GQA Query 共享一份 K/V；
> - 直接消费 134 B SoA compressed slot，不物化 global FP16 K/V；
> - 输出 `[B,Hq,S,D+1]` 的 partial output/LSE，能接同一 Stage2 语义。
>
> 它的价值不是性能，而是先做出可验证 CUDA baseline。V1 慢说明“从 Triton 改写成 CUDA”本身不会自动加速；warp reduction、Shared Memory 往返和同步必须逐项分析。

**可能追问：为什么第一个 CUDA 版本比 Triton 慢也值得保留？**

> 它建立了布局、精度和接口都能对照的基线。后续可以用相同输入逐步替换线程映射和数据路径，判断收益来自哪里。直接只保留最快版本，会丢掉“为什么优化有效”的实验证据。

---

### 37. CUDA V2 为什么每个 Q head 一个 warp？

> 这样 QK reduction、softmax state 和 output 都可以保留在 warp 内，移除 CTA barrier，并把静态 `SHFL.DOWN` site 从 20 降到 5。代价是四个 warp 重复读取并反量化同一 K/V。最终从 2.074 ms 降到 1.748 ms，说明同步减少有收益，但重复 decode 限制了进一步提升。
>
> 这是典型的资源交换：
>
> | 收益 | 代价 |
> |---|---|
> | 每 warp 独立维护一个 Q 的 max/sum/output | 同一 KV group 被四个 warp 重复读取 |
> | warp shuffle 代替部分 CTA Shared Memory 通信 | nibble unpack、centroid lookup、V dequant 重复四次 |
> | CTA barrier 减少 | cache traffic 和整数/lookup 指令增加 |
>
> V2 的实验结论不是“warp-per-Q 永远更优”，而是当前 V1 的同步成本高于重复解码代价；后续 V3 又通过 CTA 共享 K/V 和 Tensor Core 重新寻找更好的平衡。

**可能追问：四个 warp 重复加载 K/V，就一定产生四倍 DRAM 流量吗？**

> 不一定，重复请求可能被缓存吸收。但四份 unpack、centroid lookup 和 affine reconstruction 仍然消耗执行资源，访问指令也会增加。因此 V2 的代价既包括逻辑读取重复，也包括确定存在的反量化计算重复。

---

### 38. CUDA V3 的关键变化是什么？

> V3 的关键变化，是把 warp-per-Q 的标量路径改成 CTA 共享的 16-token tiled Tensor Core 路径。四个 Query 共用解码后的 K/V tile，QK 与 PV 都通过 WMMA 完成，输出累加器跨 tile 保存在 fragment registers 中，直到 split 结束才归一化写出。
>
> 一轮 tile 的顺序是：加载并反量化 16 个 token 的 K/V，沿 D=128 完成 QK，更新四个 Query 的 online softmax 状态，按 alpha 缩放旧输出，然后计算当前 probability×V。固定实验中每个 split 有八轮这样的处理。
>
> 源码中的 CUDA V1/V2 已有 running_m 和 running_l，所以不能把 V3 说成首次引入 online softmax，或说 V2 先物化全部 score 再进行第二遍扫描。V3 的主要增量是 tiled 协作和 Tensor Core 映射，仓库记录时间从 V2 的 1.748470 ms 降到 1.380587 ms；HMMA 记录支持其确实生成了 Tensor 指令。

**可能追问：如何确认 V3 真正生成了 Tensor Core 指令？**

> 源码中的 wmma::mma_sync 表明实现意图，编译后的 HMMA 才能确认目标机器指令。仓库记录了这类静态结果；复现实验时应保存对应二进制、编译参数和 SASS，再用同 workload 时间判断是否获得实际收益。

---

### 39. CUDA V4 为什么要做固定 workload 特化？

> V4 利用每个 split 固定 128 token 且按 16 对齐的条件，只加载一次 tile 的 block-table entry，只初始化四个有效 Q 行，将 centroid table放到 warp register，并完全展开八个 tile。它用通用性换取更少的分支、地址计算和循环控制，从 1.381 ms 降到 1.121 ms。
>
> 固定特化具体依赖：`D=128`、GQA=4、block size=16、split=128 token、32 splits，以及 split start 16-token 对齐。由此可以：
>
> - 完全展开 8 个 tile 和 D 方向 8 个 K-step；
> - 删除通用 tail mask和动态循环控制；
> - 每 tile 只查一次 page；
> - 只初始化 4 个真实 Q 行；
> - 将 16-entry centroid 分布到 warp lane register。
>
> 代价是 shape 不满足时当前 Kernel 会 return，而不是自动走通用 tail。面试时应把它称为 fixed-workload specialization，不应说成支持任意 vLLM 请求。

**可能追问：不支持的序列长度被 guard return 后有什么风险？**

> 当前 guard 返回时不会写对应 mid_o，调用者可能继续读取未初始化或旧数据。面向通用调用时应在 dispatch 层明确拒绝不支持的请求，或走有 tail mask 的通用实现，不能把静默返回当成正常的形状支持。

---

### 40. CUDA V5 的 fragment 直接写回是什么？

> V4 把完整 `16x128` output accumulator 先写到 Shared Memory，再只读取四个有效行。V5 根据在 `sm_89` 上探测出的 WMMA lane-to-row mapping，直接从 fragment register 将四行写到 `mid_o`，移除约 7 KB scratch 和一次大规模 shared round trip，时间降到 0.845 ms。
>
> 优化前路径是：
>
> 数据流为：fragment registers -> store_matrix_sync -> shared scratch → CTA barrier -> scalar threads读取有效四行 -> mid_o。
>
> 优化后直接根据 lane/index 映射把四行写入 `mid_o`。风险是 WMMA fragment 内部布局不是跨架构稳定 API；项目用 `wmma_fragment_probe.cu` 在 `sm_89` 验证映射。因此 V5 的 1.326x 收益带有架构特化成本，迁移时必须重新 probe 和回归。

**可能追问：为什么不能直接把 fragment 的所有元素逐个写出去？**

> 每个 lane 持有的是矩阵的一部分，而且包含 padding 行。必须先推导 lane、fragment index 与逻辑行列的对应，只写四个有效 Query 行，并避免重复写或漏写。完整 output 对照和映射 probe 都是必要证据。

---

### 41. CUDA V6 的向量化为什么收益大？

> V6 优化的是反量化的数据通路。它让每个线程分别用一次对齐 uint32 加载 K 和 V 的四个 packed byte，每个 word 含八个 4-bit index，然后逐 byte 拆出低、高 nibble。K 查 centroid 后乘 norm，V 做 index×scale+zero，最后用 half2 向 Shared Memory 写入成对的 FP16 值。
>
> 因此一个 word 的八个重建值对应四次 half2 写，而不是一次 half2 写八个元素。当前地址中的 block stride、token stride、head stride 和 V offset 都满足 4 B 对齐，这是 uint32 加载成立的前提。
>
> 它减少了细粒度 load/store 与地址相关指令，packed payload 总字节数并没有再次缩小。仓库记录从 V5 的 0.845486 ms 降到 V6 的 0.638863 ms，约 1.323 倍，说明优化解码数据路径很有价值；具体硬件瓶颈仍需动态 profiler 进一步归因。

**可能追问：一次 uint32 load 后，half2 store 要做几次？**

> 一个 uint32 包含四个 byte，也就是八个 4-bit 坐标；每个 half2 写两个重建值，所以每个 K word 对应四次 half2 写，V word 也一样。不能把 half2 说成一次写八个 FP16 元素。

---

### 42. CUDA V7 怎样减少 barrier？

> V6 每个 tile 末尾有一个 barrier。V7 发现下一个 tile 开头原本就有 metadata 发布 barrier，而 metadata storage 与 K/V storage 独立，因此可以让这个开头 barrier 同时承担“等待上一个 PV 完成”和“发布新 metadata”两个作用。静态 barrier 从 42 降到 34，收益约 1.012x。
>
> 删除同步前的正确性证明是：
>
> 1. 上一轮 PV 读取的是 `p_s/v_s`，output accumulator 已进入各 warp register；
> 2. 下一轮 warp 0 先写的是独立的 metadata/data-base arrays；
> 3. 下一轮已有 barrier 会等待所有 warp 完成上一轮 PV；
> 4. barrier 之后才允许线程覆盖下一轮 `k_s/v_s/p_s`。
>
> 所以删除的是冗余生命周期边界，不是依赖“warp 恰好同步”的冒险优化。时间只提升 1.2% 也应保留，因为结果稳定、实现变化单一，并为后续 barrier 分析提供基线。

**可能追问：最后一个 tile 没有下一轮 barrier，是否会出问题？**

> 要看最后的数据消费者。最后一次 PV 后不再重写 K/V tile，后续只处理寄存器输出和最终归一化状态；仍保留必要的最终状态发布同步。能否省同步取决于是否有覆盖和跨 warp 依赖，不取决于循环形式本身。

---

### 43. 为什么 V7 到 V8 的收益比 V6 到 V7 大？

> V7 只消除一部分同步，主计算形状仍有 75% 无效 WMMA 行；V8 直接改变 MMA shape，将 HMMA 数量减半，并缩小 Shared Memory 和 accumulator。前者是局部调度优化，后者减少了核心 Tensor Core 工作量，所以 V8 获得约 23% 提升，而 V7 只有约 1.2%。
>
> 用数据表达：
>
> $$
> V6\rightarrow V7:\quad0.638863/0.631122=1.012\times,
> $$
>
> $$
> V7\rightarrow V8:\quad0.631122/0.513208=1.230\times.
> $$
>
> V7 优化的是同步等待这一局部开销；V8 同时降低无效 MMA 槽位、静态 HMMA site 和 Shared Memory footprint，作用于每个 tile 的主计算图。性能差异符合修改的覆盖范围，但最终不能仅靠“改得更大”解释，仍需同一 harness 的测量。

**可能追问：为什么没有收益更大的优化就删除小收益版本？**

> 小收益版本仍能说明单一改动的效果，并帮助定位回归。V7 是 Full Decode 的实际入口，其同步结构也成为 V8/V9 的基础。不过微小收益要结合各轮分布判断稳定性，不能只凭一个最小值认定有效。

---

### 44. 为什么要保留所有历史版本？

> 每个版本只引入一类主要变化，构成可复现的 ablation：线程映射、单遍算法、固定特化、写回、向量化、同步、MMA shape 和 softmax fusion。这样可以解释性能来自哪里，也能避免把多个变化一起提交后无法归因。面试时这比只展示最终 V9 更能体现性能工程方法。
>
> 版本保留还承担回归定位：若 V9 在新 CUDA 版本上错误，可以二分判断问题首次出现在 fragment mapping、vector load、barrier 还是 inline PTX；若某 GPU 上 V8 比 V7 慢，也能识别是 MMA shape 还是其他资源变化。
>
> 每版的正确讲法是“瓶颈假设 → 单一主要修改 → correctness → 资源/SASS → 实测时间 → 新限制”，而不是把 V1–V9 当成九个没有因果关系的文件。

**可能追问：如何避免历史版本把面试叙述弄乱？**

> 我会固定三条主线：算法与数据契约、每版唯一主要改动、同口径时间。结果只引用指定版本和指定范围，例如 V9 Stage1 与 V7 Full；旧错误 baseline 和旧 NCU 单独标明历史用途。

---

## Correctness 与实验设计

### 45. 上游 Triton V2 曾经有什么 correctness 问题？

> 复制的 V2 使用 `tl.interleave(v_lo, v_hi)` 重建 V，随后直接交给 `tl.dot`。在 CUDA 上该 layout 与 dot 期望不兼容，造成 V 列置换。QK 和 softmax 未受影响，所以 LSE 看起来正确，但 output 最大误差约 0.325。
>
> 修复版根据 `d//2` 和 nibble shift 直接构造最终 `[TILE, D]` V layout，输出误差恢复到约 `1e-4`。这说明只检查 LSE 不足以证明 Attention 正确。
>
> 这个 bug 的诊断逻辑是：
>
> - LSE 只依赖 QK score，LSE 正确说明 K decode、QK 和 softmax statistic 大概率正确；
> - output 还依赖 probability 与 V 各列的对应关系；
> - output 大错但 LSE 正确，将怀疑范围缩到 V layout/PV；
> - `tl.interleave` 产生的内部 layout 与 `tl.dot` operand 解释不一致，最终导致列置换。
>
> 修复后需要同时比较完整 `mid_o[...,0:128]` 和 `mid_o[...,128]`，并更新 baseline 性能；不能继续引用错误版本较快或较慢的数字。

**可能追问：为什么 LSE 正确而 output 错，能帮助定位到 V 路径？**

> LSE 只取决于 QK score 和 softmax 归一化，V 只参与后面的加权求和。LSE 对齐而输出明显偏离，会让我优先查 V unpack、列映射和 PV，而不是首先怀疑 K 的 codebook 或 Query rotation。

---

### 46. CUDA V9 的正确性怎样验证？

> 正式 harness 为所有实现构造同一逻辑 cache，并与 SoA Triton V1 比较完整`mid_o` 的 partial output 和 LSE，同时检查所有值 finite。V9 的结果是：
>
> ```text
> output max/mean  9.6827745e-05 / 1.2782620e-05
> LSE max/mean     2.4318695e-05 / 3.7480786e-06
> ```
>
> 此外，V7 Full Decode 与 Triton Full Decode 的最终 output 最大差约`5.66e-07`，Store 兼容性测试也覆盖真实压缩 cache。
>
> 验证应分三层：
>
> 1. **Stage1 同 cache 对照**：比较所有 split 的 partial output、LSE 和 finite；
> 2. **Stage2/Full 对照**：比较最终 `[B,Hq,D]` output 和 `[B,Hq]` LSE；
> 3. **Store→Decode 契约**：用未修改 vLLM Store 生成 cache，排除 synthetic layout
>    自己写错但双方同时读错的风险。
>
> 还应跑多随机种子、极端 norm、常量 V、随机 page table 和 tail shape。当前固定 harness 完成了核心链路，但后几类 production case 仍属于待扩展范围。

**可能追问：max abs 接近 1e-4 就能证明所有输入都正确吗？**

> 不能，它只证明当前样本在该阈值内。还应覆盖不同随机种子、结构化输入、元数据极值以及受支持的页映射，并检查有限值和误差位置。当前文档中的误差是仓库记录，不能扩展成任意输入保证。

---

### 47. 为什么 CUDA V3-V9 与 Triton V1 有约 `1e-4` 误差？

> 这些版本使用 FP16 Tensor Core operand、不同的求和顺序、online softmax 和 fast-math 指数近似。浮点加法不满足结合律，因此与逐元素 FP32 路径不会逐位一致。误差应结合 reference、最终输出和量化误差判断，不能把非零差异直接当成 Kernel 错误。
>
> 判断误差是否可接受不能只设一个绝对阈值，应同时看：
>
> - max absolute error：捕获最坏元素；
> - mean absolute error：观察整体偏差；
> - relative error：避免不同量级被同一绝对阈值掩盖；
> - LSE 与 output：分别覆盖 QK/softmax 和 PV；
> - 与同 Tensor Core/fast-math reference 的差异：隔离实现顺序误差。
>
> 若误差随 context、split 数或输入幅度持续放大，则不能简单归因于浮点结合律，需要检查 online-softmax rescale 和 Stage2 合并。

**可能追问：FP32 accumulation 为什么仍会有 FP16 相关误差？**

> 累加精度与乘法输入精度是两层。Query、重建 K/V 和概率进入 MMA 前会转换成 FP16，舍入已经发生；FP32 累加能减轻求和误差，却恢复不了转换前的信息。不同归约顺序也会改变最后几位。

---

### 48. 如何区分 Kernel 数值误差和 4-bit 量化误差？

> Store 验证中，CUDA 与 Triton 读取同一量化 cache，output 最大差约`5.06e-06`；而量化 Decode 与原始 FP32 Attention 的 output 最大差约`1.87e-02`。前者远小于后者，说明较大的差异来自预期量化损失，不是 CUDA layout 或 Attention 算法错误。
>
> 隔离方法的核心是固定变量：
>
> 数据流为：CUDA compressed decode vs Triton compressed decode → 测 Kernel/layout/执行顺序误差 → Triton or CUDA compressed decode vs canonical FP32 attention → 包含 4-bit quantization error。
>
> 如果第一组已经很大，应先修 Kernel；只有第一组足够小，第二组才能用于评估量化误差。模型 PPL/下游任务又是第三层，不能由单个 synthetic output max error代替。

**可能追问：如果量化参考与 FP32 attention 差得多，先改 CUDA 吗？**

> 先区分误差来源。如果 CUDA 与量化 Triton 对齐，而两者都偏离原始 FP32 Attention，应先检查量化方案及 metadata 精度。只有 CUDA 在相同 compressed cache 上额外偏离，才优先查 Kernel 的布局、数学和同步。

---

### 49. Benchmark 怎样保证版本比较公平？

> 我会先保证比较对象处理同一份逻辑 compressed cache，AoS/SoA 转换只改变布局，不改变数值语义，而且转换、输入构造、内存分配和 JIT 都放在计时外。CUDA 与 Triton 的输出范围一致，先检查完整 partial output 和 LSE，再进行性能测量。
>
> 脚本默认先 warmup 20 次；每轮在一对 CUDA Event 之间运行 100 次，把总耗时除以 100 得到平均单次时间。总共五轮，每轮轮换 runner 顺序，最后取五个轮均值的中位数。它不是对 100 个单独 latency 求中位数。
>
> Stage1 和 Full 各自独立测量，Full 的两个 launch 放进同一个 runner。复现还应固定设备、输入、编译参数与软件环境，查看各轮分布，避免温度、频率和缓存热度造成系统性偏差。

**可能追问：一百次计时是取一百个延迟的中位数吗？**

> 不是。event_time_ms 在一对 CUDA Event 之间连续执行 100 次，用总时间除以次数，得到一轮的平均单次时间；再对五轮的结果取中位数。明确这两层统计，才能准确解释报告数字。

---

### 50. 为什么使用中位数而不是只报最快一次？

> GPU Boost、温度、后台负载和首次 cache 状态会造成波动。最快值可能只是偶然高频状态，平均值也容易受离群点影响。多轮交错测试加中位数更稳健；小于几个百分点的优化还应重复验证，并结合资源与 SASS 证据。
>
> 中位数回答的是“典型一轮性能”，不是置信区间。严谨报告还应给出最小/最大值、标准差或分位数。若 V6→V7 只有约 1.2%，必须确认收益大于运行波动；若只是某一轮最快，不应作为稳定优化写入简历。
>
> 五轮轮换顺序也比连续跑完某版本再跑下一版本更好，因为它让温度和 boost 随时间变化更均匀地影响所有 candidate。

**可能追问：中位数能完全排除温度和频率变化吗？**

> 不能。它主要减少少数离群样本的影响，对系统性降频或先后顺序偏差无能为力。项目轮换实现顺序是补充措施；严谨复现还应记录设备、软件版本和各轮分布，检查是否有持续漂移。

---

### 51. 你使用了哪些 profiling 证据？

> 我使用三类不同证据：CUDA Event 给出同 workload 的延迟；ptxas/cubin 和 SASS 用于核对寄存器、Shared Memory、HMMA 与 barrier 的静态结构；correctness harness 用于保证优化保持数值和布局契约。三类证据要互相配合，不能由单个资源数字直接宣布性能原因。
>
> 上传仓库的 README 记录了 V1–V9 的时间与静态资源。results 里的 NCU 属于历史 Triton V2 分析，且对应 correctness 修复前的路径；CUDA V1 的采集还遇到 ERR_NVGPUCTRPERM。因此没有当前 V9 动态报告支持“DRAM 已达峰值某个百分比”之类结论。
>
> 如果继续 profiling，我会在相同源码和输入下采集 V7/V8/V9 的 DRAM/L2 流量、Tensor pipe、warp stall、eligible warps、occupancy、shared bank conflict 和 local spill。静态证据说明改动落到了机器代码，动态证据才帮助确认实际限制。

**可能追问：当前压缩包是否包含 V9 的动态 NCU 报告？**

> 没有可用于确认 V9 当前瓶颈的正式动态报告。README 保存了时间与静态资源记录，results 中是历史 Triton 分析，不能移用为 V9 counter。后续采集应锁定源码版本、输入和编译条件。

---

### 52. 静态 HMMA 或 barrier 数量能直接等价为运行时间吗？

> 不能。完全展开的八个 tile 会让同一逻辑操作出现多个静态 site；实际性能还取决于指令吞吐、依赖、occupancy、memory latency 和调度。V8 的 HMMA 减半且实测明显加速，V9 的 barrier 减少也有稳定收益，但必须以同环境 CUDA Event 结果确认，不能只看反汇编计数。
>
> 例如一个循环完全展开八次，会把逻辑上一处 barrier 展开成多个 static site；反之，未展开循环只有一个 site，却可能动态执行八次。HMMA 也受 predicate、loop trip、 warp 数和输入 shape 影响。
>
> 正确归因顺序是：先比较代码路径和动态工作量，再看 SASS 是否符合预期，随后看资源与 NCU counter，最后以稳定 wall/kernel time 判断收益。任何单一静态数字都不能替代这条证据链。

**可能追问：静态 barrier 少八处，是否代表每个 warp 少执行八次？**

> 需要结合控制流解释。在固定展开的八个 tile 中，每 tile 删除一次 CTA barrier，源码可推导对应动态路径的减少；但单看 SASS site 数还不能忽略谓词、循环和 warp 分支，更不能直接换算时延。

---

## 性能口径与设计取舍

### 53. 4.28x、2.22x 和 1.66x 分别是什么？

> 三者比较对象不同：
>
> ```text
> CUDA V1 -> CUDA V9 Stage1       2.074204 / 0.484516 = 4.281x
> Triton V2-fixed -> CUDA V9      1.075988 / 0.484516 = 2.221x
> Triton -> CUDA V7 Full Decode   1.104148 / 0.664842 = 1.661x
> ```
>
> 不能把 V9 的 Stage1 速度与 V7 的 Full Decode 速度混合，也不能把 4.28x 描述成完整 vLLM 请求端到端加速。
>
> 建议面试中只主动强调与简历一致的 `1.66x Full Decode`，其余作为追问展开：
>
> - 4.28x 是项目内部 CUDA V1→V9 的优化演进，证明迭代收益；
> - 2.22x 是最新 Stage1 对强 Triton Stage1 baseline；
> - 1.66x 是可完整比较的 V7 Stage1+Stage2 对 Triton Full，也是简历数字；
> - 三者 workload 相同，但入口和版本不同，不能拼接成“V9 Full 2.22x”。

**可能追问：简历上应该优先写哪一个加速比？**

> 如果强调完整 split Decode 链路，我会写 V7 Full 对 Triton Full 的 1.66 倍，并给出 0.6648 ms。V9 的 2.22 倍可以作为 Stage1 优化补充。CUDA V1 到 V9 的 4.28 倍适合说明内部演进，需要明确起点。

---

### 54. “加速 4.28x”和“耗时降低多少”有什么区别？

> ```text
> speedup = 2.074204 / 0.484516 = 4.281x
> time reduction = 1 - 0.484516 / 2.074204 = 76.64%
> ```
>
> 4.28x 不是“耗时降低 428%”，也不宜简单说“性能提升 328%”而不说明计算口径。
>
> 一般公式是：
>
> $$
> speedup=\frac{T_{old}}{T_{new}},\qquad
> \mathrm{latency\ reduction}=1-\frac{T_{new}}{T_{old}}.
> $$
>
> 对简历 Full 数据，1.661x 对应 latency 降低 39.79%；对 V1→V9，4.281x 对应降低 76.64%。回答“快了多少”前先确认面试官问 throughput multiplier 还是 latency percentage。

**可能追问：4.28 倍加速对应多少延迟下降？**

> 用 1−新时间/旧时间计算，而不是用 4.28−1。代入 CUDA V1 的 2.074204 ms 与 V9 的 0.484516 ms，延迟约下降 76.6%。这与吞吐约为原来的 4.28 倍是同一组数据的不同表达。

---

### 55. Shared Memory 越少、occupancy 就一定越高吗？

> 不一定。CTA residency 同时受 registers、Shared Memory、threads、warp slots 和架构限制。减少 Shared Memory 可能解除某个限制，也可能根本不是当前 limiting resource。Occupancy 只是 latency hiding 的条件，不是最终性能指标；降低资源却增加指令或 spill 仍可能变慢。
>
> 应按资源上限逐项判断 CTA residency：
>
> ```text
> register limit  = SM register file / registers per CTA
> shared limit    = SM shared capacity / shared bytes per CTA
> thread limit    = max resident threads / threads per CTA
> block limit     = architecture max resident CTAs
> ```
>
> 最小值决定理论 active CTA，再结合实际 eligible warps、stall 和 spill。V7→V8 shared 从 14,224 B 降到 10,336 B，但 register 从 49 增到 51；是否提高 occupancy 必须用 occupancy calculator/NCU 验证。即使 occupancy 不变，更少 Shared Memory traffic 和 MMA 工作仍可能加速。

**可能追问：寄存器下降一个就一定多驻留一个 CTA 吗？**

> 不一定。硬件按一定粒度分配寄存器和 shared memory，resident CTA 上限还受线程、warp、block 数等共同约束。资源只有跨过限制阶梯才可能改变理论 occupancy，而更高 occupancy 也未必降低时间。

---

### 56. 为什么不用 double buffering 或 `cp.async` 做完整流水？

> 压缩 cache 不是可以直接异步复制成最终 FP16 tile的数据。K 需要 unpack、 centroid lookup 和 norm，V 需要 unpack、scale/zero；`cp.async` 只能搬字节，不能执行这些变换。双缓冲还会增加 Shared Memory，并可能降低 resident CTA。
>
> 它仍然是可实验方向，例如异步预取 packed byte 或 metadata，但必须测量变换、额外同步和 occupancy 的综合成本，不能因为 FlashInfer 使用 pipeline 就假定本项目照搬一定更快。
>
> 采用 `cp.async` 前应先回答：
>
> 1. NCU 是否显示 global-memory long scoreboard 是主要 stall？
> 2. 当前 tile 计算是否足够长，能覆盖下一 tile packed-byte load？
> 3. 双份 packed/shared buffer 会不会降低 resident CTA？
> 4. metadata、block-table 和 nibble transformation 如何与异步 copy 排程？
> 5. 新 pipeline 增加的 commit/wait/barrier 是否小于被隐藏的 latency？
>
> 可先只双缓冲原始 packed K/V 和 metadata，再在消费前 unpack，而不是试图让`cp.async` 直接完成反量化。优化后必须比较 Full 时间和 NCU stall，不应只展示 source 中出现了异步指令。

**可能追问：cp.async 可以直接把 4-bit 变成 FP16 吗？**

> 不可以，它搬运字节，不执行 nibble unpack、查表或 affine reconstruction。若引入异步拷贝，还要给 packed 数据分配 staging buffer，再安排解码与计算重叠；因此收益取决于被隐藏的等待能否覆盖新增资源和同步成本。

---

### 57. 为什么 V9 仍然只有 50% 的 N 维有效槽位？

> `m16n8k16` 的 N 固定为 8，而 GQA group 只有四个 Q head，所以仍有四列为空。它已比 `m16n16k16` 的四行有效更好，但不是 100%。进一步填充需要让一次 MMA 同时处理更多合法输出，同时保证不同 KV group 的 K/V 不交叉；这通常要求更复杂的 block-diagonal、稀疏或多-MMA 调度，收益需覆盖布局成本。
>
> 不同 GQA ratio 的结果也不同：
>
> - GQA=8：刚好填满 N=8，槽位可达 100%，但 Q/softmax/output state 翻倍；
> - GQA=4：当前 4/8=50%；
> - GQA=2：只有 2/8=25%，可能需要不同 MMA shape 或 SIMT 路径；
> - MHA/GQA=1：量化 decode 访存仍有价值，但当前 Tensor mapping 很浪费。
>
> 不能跨 KV group 简单填空列，因为每组使用不同 K/V。通用 backend 应按 GQA ratio 选择模板，而不是强制所有模型使用 V9 mapping。

**可能追问：能不能用一个更小的 N=4 MMA 完全匹配 GQA-4？**

> 不能假设硬件提供任意矩阵形状。当前选用的 FP16 原生指令是 m16n8k16；若想进一步减少空槽，需要重新考虑指令集合、SIMT 路径或合法的多任务映射，并把布局转换成本一起计入。

---

### 58. 为什么 V9 没有直接替换 Full Decode 中的 V7 Stage1？

> 当前版本演进把 V8/V9 保持为独立 Stage1 candidate，V7 则提供经过验证的 Stage1、Stage2 和 Full launcher。将 V9 接入 Full Decode 在工程上可做，但需要新增稳定导出、完整回归和 Store 路径验证。当前文档明确区分这两个范围，避免把 Stage1 实验误报成已完成的 production chain。
>
> 正式接入至少需要：
>
> 1. 确认 V9 `mid_o` layout、normalization 和 LSE 与 V7 Stage2 契约逐元素一致；
> 2. 导出 V9 Stage1 和 Full launcher，复用预分配 output；
> 3. 跑 Stage1、Stage2、Full 三层 correctness；
> 4. 跑未修改 vLLM Store→V9 Full compatibility；
> 5. 在同一 harness 独立测 V9 Full，不能用 V9 Stage1 + V7 Stage2 的两个中位数
>    手工相加；
> 6. 更新文档后才能把 V9 称作完整链路。

**可能追问：V9 和 V7 的 Stage1 输出接口一致，为什么仍要重验 Full？**

> 接口一致只说明形状和统计量约定匹配，不代表数值误差、写回映射及集成路径已经验证。接入后还要比较最终 output/LSE，并让 V9 读取 Store 产生的 cache，再独立测完整两个 launch 的时间。

---

### 59. 这个项目目前最大的局限是什么？

> 这个项目最大的边界是固定 workload 的研究实现，还不是完整生产 backend。当前报告在 B=64、L=4096、Hq=32、Hkv=8、D=128 上测得；V9 launcher 可以读取动态 batch，但计算路径仍要求固定 head 数、32 splits、16-token block 和每 split 128 token。
>
> 架构方面，实验针对 RTX 4090 的 sm_89，WMMA 直接写回依赖探测出的内部映射，原生 PTX 版本也需要在其他设备重新编译验证。输入方面，现有验证主要采用等长序列和连续页，尚未全面覆盖 ragged batch、随机 page、尾部和不同 GQA。
>
> 系统和质量方面，V9 目前只作为 Stage1 candidate，经过记录的 Full 与 Store 验证主要是 V7；没有完整模型 perplexity、服务端 token latency 或生产 backend 注册的结果。面试中我会把这些支持范围说清楚，同时展示固定范围内可复现的优化和验证。

**可能追问：固定形状研究项目如何体现工程价值？**

> 它能展示从真实压缩格式出发，建立对照基线、设计线程和矩阵映射、处理同步与精度，并用可复现数据验证取舍。价值在可解释的优化过程；适用范围则应限定在已经支持和测量的 workload。

---

### 60. 下一步最值得做什么？

> 我会先把 V9 Stage1 接入 Full runner，复用明确的 mid_o/LSE 契约，重跑 Stage1、Stage2、Full 和真实 Store 输入对照，再独立测 V9 Full。这样能确认最新 Stage1 的收益是否传到完整 split Decode，也统一结果口径。
>
> 随后补不支持形状的明确 dispatch、tail/ragged 序列与随机页验证，避免 guard 静默返回造成未写输出。再采集当前 V9 的 NCU，根据实际限制选择预取、双缓冲或其他数据路径优化，而不是无目标继续减少指令。
>
> 更长期再做 D/GQA/split 模板、低 batch 路径、模型质量和完整 vLLM 集成。每一阶段都要明确支持的输入范围、correctness 标准和 Full 时间，只有数据证明有效，才把它写成已经完成的收益。

**可能追问：为什么优先补 Full，而不是继续追求 Stage1 最快？**

> Full 能确认最新 Stage1 的收益真正传到完整 split Decode，并统一简历的性能范围。它也可能暴露误差和接口问题。只有先量化整体收益，才知道继续优化 Stage1、归并或其他环节哪个更值得。

---

## 项目贡献与行为问题

### 61. 你个人的核心贡献可以怎样回答？

> 我在这个项目里的工作重点，是把 TurboQuant 的压缩格式与 CUDA Attention 的计算组织衔接起来，并建立可以逐版比较的验证和性能流程。上游提供量化思想、codebook 生成、Store/Decode 参考和 cache 语义；项目中的工程优化集中在 CUDA 版本、benchmark 与兼容性验证。
>
> 实现主线从 CTA/warp 映射开始，逐步采用 tiled WMMA、固定 workload 特化、fragment 直接写回、uint32/half2 解码和 barrier 合并。V8 针对 GQA-4 调整原生 m16n8k16 映射，V9 进一步让 QK score 和 softmax 状态留在寄存器，减少 shared 往返。
>
> 验证上，通过同 cache 的 CUDA/Triton 对照、Stage2 和 Full 检查，以及未修改 SoA Store 到 Decode 的链路来说明结果可信。个人贡献应按实际承担的设计、编码、测试和分析范围表述；我不会把 TurboQuant 算法或上游快照都归为个人原创。

**可能追问：如果实现过程中使用了代码生成工具，面试怎么回答？**

> 我会如实说明辅助工具参与的部分，把自己完成的需求分析、方案取舍、源码核对、实验设计和问题定位讲清楚。能够解释一条地址公式、一个 barrier 的必要性和一次 correctness 修复，比把所有代码都笼统称为手写更有说服力。

---

### 62. 这个项目最大的技术难点是什么？

> 一是数学正确性：四个 KV group 不能为了填满 WMMA 而错误拼成 dense GEMM；二是数值正确性：layout 错误可能保持 LSE 正常却破坏最终 output；三是性能归因：反量化、Tensor Core、Shared Memory、barrier 和寄存器互相制约。
>
> 真正困难的是同时保证量化语义、Attention 语义和 GPU 映射正确，再通过公平实验判断哪一项优化确实转化为性能。
>
> 以 V8 为例，我会具体说明：
>
> 1. **数学约束**：不同 KV group 不能为了填满 16 行而混算；
> 2. **映射设计**：把合法 group 内 QK/PV 转置到 `m16n8k16` 的 N 维；
> 3. **底层实现**：按 PTX lane contract 手工打包 half2/register；
> 4. **正确性**：验证四个 Q 列及 128 output dimension 映射；
> 5. **性能证据**：V7→V8 从 0.631122 降到 0.513208 ms，同时 shared 和 static
>    HMMA 下降；
> 6. **限制**：仍只有 50% slot，并且当前实验在 `sm_89` 上验证。

**可能追问：能举一个同时涉及性能与正确性的难点吗？**

> V8 的组内转置映射很典型：它让四个 Query 只占 N=8 而非 M=16，但同时改变 QK operand、score 归属、PV operand 和输出写回。只改指令形状不会自动正确，必须把四段索引连成一致的数学关系。

---

### 63. 如果面试官问“为什么不用现成 FlashInfer”，怎么回答？

> 项目将 FlashInfer 的 Attention 数据流作为设计参考，但当前输入是 TurboQuant 特定的 4-bit index、centroid/norm 和 scale/zero layout，不能直接当作普通 FP16 paged KV Cache 传入。先完整反量化再调用 FlashInfer 会增加 Global Memory traffic。
>
> 因此项目保留量化专用 decode，并迁移适合的执行思想，例如 register-resident softmax state。未来也可以把 TurboQuant dequant iterator 接入更通用的 FlashInfer 调度框架，但需要处理数据布局和模板接口。
>
> 结合当前工程接口，具体取舍是：
>
> - FlashInfer 的成熟调度、paged attention 和 register-state 思路值得复用；
> - 当前 cache 是 K centroid index/norm 与 V affine index/scale/zero，不是普通
>   FP16/BF16 K/V；
> - 直接调用前若必须全量解压，会增加每 slot 512 B 写回和后续重读；
> - 本项目验证的是量化专用 load/decode 与 attention compute 的融合方式；
> - 长期可以给 FlashInfer 增加 quantized iterator/epilogue，而不是重复实现整个
>   调度框架。
>
> V9 的准确表述是“借鉴 FlashInfer 的 register-resident state 思路”，不是“基于 FlashInfer Kernel”或“调用 FlashInfer API”。

**可能追问：使用现成库和研究专用 Kernel 如何选择？**

> 如果目标是上线，应先评估库是否支持当前量化语义、页布局和目标平台；如果已经支持且表现足够好，集成通常更省维护成本。这个项目的研究价值在于固定 TurboQuant 数据契约下的融合和 GQA 映射，并不预设自研一定胜过所有库。

---

### 64. 如果换成 RTX 3090，结果会一样吗？

> 不会。3090 是 `sm_86`，4090 是 `sm_89`，二者的 SM 数量、时钟、缓存、显存带宽和调度行为不同。V8/V9 已在 `sm_89` 上验证，迁移还要按 PTX 契约和目标架构重新核对。移植时需要重新编译、验证 lane mapping、检查 SASS、重跑 correctness 和性能，不能只修改编译架构字符串后沿用 4090 数字。
>
> 迁移检查表包括：
>
> 1. 改为 `sm_86` 编译并确认 `m16n8k16` PTX/SASS 支持；
> 2. 重新验证 WMMA/inline PTX lane mapping；
> 3. 检查 ptxas registers、Shared Memory、spill 和 theoretical occupancy；
> 4. 根据 3090 的 SM、L2、带宽重调 split 和 CTA 参数；
> 5. 重跑 Store、Stage1、Stage2 和 Full correctness；
> 6. 重采 CUDA Event 与 NCU，不能拿旧 3090/错误 Triton V2 报告代替。
>
> 可移植不等于性能可移植：即使结果正确，最优 tile、occupancy 和瓶颈仍可能改变。

**可能追问：迁移到 3090 时能只改编译参数吗？**

> 重新编译只是第一步。还应验证指令支持与输出映射、检查寄存器和 shared 资源，再测相同 workload。不同架构的 cache、带宽、调度和 Tensor 吞吐会改变最优取舍，4090 的数值不能直接作为 3090 结论。

---

### 65. 面试时如何证明这是工程优化，不是只调参数？

> 我会展示完整证据链：先固定 workload 和 reference；用版本隔离单一变化；逐版验证 Stage1 的 partial output/LSE，并对已接入的 Full 路径验证最终 output；再用 CUDA Event 看稳定收益，用 cuobjdump 检查 register/shared/HMMA/barrier；对 V8 还要解释为什么新的 MMA 映射数学上成立，对 V9 解释为何减少一次 shared round trip。
>
> 比起背诵“用了 Shared Memory、Tensor Core、FlashInfer”，这种从问题、假设、实现、证据到限制的闭环更能体现性能工程能力。
>
> 对应的优化证据可以列为：
>
> | 优化 | 瓶颈假设 | 主要修改 | 证据 |
> |---|---|---|---|
> | V5 direct write | output shared round trip 昂贵 | fragment 直接写 `mid_o` | 0.845486 ms，shared下降 |
> | V6 packed load | byte decode 指令多 | `uint32_t` + `half2` | 0.638863 ms |
> | V7 barrier | tile 边界同步冗余 | 合并生命周期 barrier | 0.631122 ms，BAR下降 |
> | V8 MMA shape | GQA-4 padding浪费 | `m16n8k16` 转置 | 0.513208 ms，HMMA/shared下降 |
> | V9 state fusion | `qk_s` 往返昂贵 | register softmax state | 0.484516 ms，shared/BAR下降 |
>
> 每一行还必须配同 cache correctness。若面试官要求复现，应能指出 source diff、 benchmark command 和 output/LSE 结果，而不是只展示最终时间表。

**可能追问：最能证明你理解项目的一段代码是什么？**

> 我会选择 V9 的一轮 tile：由 block table 算 data_base，uint32 解包成 K/V，执行 QK MMA，在 warp 0 更新 softmax，再由四个 warp 做 PV。能解释数据归属和每个同步点，就能把数学、布局和性能联系起来。

---

## 随机面试深挖题

### 66. TurboQuant 为什么要在量化前对 K 做正交旋转？不旋转会有什么问题？

> 不旋转时，Key 的能量可能集中在少数坐标，直接用统一的低比特标量量化器可能难以同时照顾异常值和大量小值。归一化先分离整体幅值，正交混合再改变坐标表示，使理论分布量化器更有机会有效利用有限的 16 个重建值。
>
> 理想的 Haar 随机旋转会把固定单位向量变成球面均匀方向，高维单坐标近似 N(0,1/d)，这为固定 Lloyd-Max codebook 提供依据。Q 与 K 做相同的正交变换时，量化前内积不变。
>
> 当前源码使用固定的归一化 Hadamard。它保持范数和匹配内积，但不能保证任意 Key 都满足上述随机分布，因此我会把理论动机、实际变换和最终质量验证分别说明，而不会说“旋转后必然没有异常值”。

**可能追问：遇到与 Hadamard 基向量对齐的输入会怎样？**

> 固定 Hadamard 可能把这类特殊方向变成能量集中的坐标，说明它并不保证所有输入都被均匀摊开。理论分布适合解释设计动机，工程效果仍应通过实际分布和质量评估验证，不能把经验有效说成逐向量定理。

---

### 67. “向量在单位球面上均匀分布”是否表示每个坐标服从 Uniform distribution？

> 不是。“球面均匀”指向量的方向相对于旋转不偏向任何方向，不是每个坐标都在`[-1, 1]` 上均匀分布。若 $Y$ 均匀分布在 $d$ 维单位球面 $S^{d-1}$ 上，单个坐标 $Y_i$ 的密度为：
>
> $$
> f(t)=\frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)}
>      (1-t^2)^{(d-3)/2},\qquad -1\le t\le 1.
> $$
>
> 高维时密度强烈集中在 0 附近，显然不是平坦的 Uniform density。各坐标也不严格独立，因为始终满足 $\sum_i Y_i^2=1$。

**可能追问：为什么球面坐标不独立？**

> 单位范数要求所有坐标平方和恒等于 1。某些坐标的绝对值变大，留给其他坐标的平方和就会变小，因此存在约束。高维下有限个坐标可近似 Gaussian，并不表示整个向量的全部坐标严格独立。

---

### 68. TurboQuant 中单个坐标的精确分布是什么？为什么可近似为 $N(0,1/d)$？

> 对 Haar 随机正交旋转后的单位向量，有：
>
> $$
> Y_i^2\sim \mathrm{Beta}\left(\frac12,\frac{d-1}{2}\right).
> $$
>
> 等价地，平移后的变量满足：
>
> $$
> \frac{Y_i+1}{2}\sim
> \mathrm{Beta}\left(\frac{d-1}{2},\frac{d-1}{2}\right).
> $$
>
> 由球面对称性，$E[Y_i]=0$；又因为 $\sum_iY_i^2=1$ 且所有坐标地位相同，$E[Y_i^2]=1/d$。当 $d$ 增大时，$\sqrt dY_i$ 依分布趋近 $N(0,1)$，所以$Y_i\approx N(0,1/d)$。这里说的是边缘分布近似；有限维坐标之间仍受单位范数约束，不能说成严格独立。

**可能追问：怎样不背密度公式也能说明方差是 1/d？**

> 利用对称性，各坐标平方的期望相同，而平方和恒为 1。因此 d 个相同的二阶矩相加等于 1，每个就是 1/d；均值为零，所以二阶矩也就是方差。这个推导依赖球面均匀方向假设。

---

### 69. vLLM 会从真实 KV Cache 采样数据来生成 codebook 吗？

> 这份快照不会先从真实 KV Cache 采样来生成 codebook。reference/centroids.py 的 solve_lloyd_max 根据 d 和 bits 构造 Gaussian PDF，用数值积分更新区间条件均值，并用相邻 centroid 的中点更新边界；get_centroids 缓存结果。
>
> 当前 D=128、bits=4，所以得到 16 个 FP32 centroid，供全部 token 和 KV head 使用。每根 Key 的幅值通过独立 corrected norm 保存，不需要因此生成一套新 codebook。
>
> 注释说明 d≥64 时 Gaussian 近似的依据，但函数本身始终使用 Gaussian PDF，没有为小维度切换精确球面分布的实现。当前 Hadamard 工程路径也不能等同于 Haar 随机旋转；无需模型采样是已实现的行为，分布匹配程度仍需质量评估。

**可能追问：当前 centroids.py 对小维度会自动切换精确球面分布吗？**

> 不会。这个快照的 solve_lloyd_max 始终调用 Gaussian PDF，只是在注释中说明 d≥64 的近似依据，没有小维度分支。说明仓库行为时要按实际函数回答，不能把理论上的另一种求解方式说成已实现。

---

### 70. `head_dim = 128` 时 Gaussian approximation 的方差和标准差是多少？

> 在球面坐标的 Gaussian 近似下，D=128 时单坐标均值为零、方差为 1/128=0.0078125，标准差为 1/√128≈0.0883883。原因是单位向量各坐标对称、平方和为 1，所以每个坐标的二阶矩为 1/D。
>
> 这里描述的是归一化并按理论随机旋转假设处理后的坐标，用于生成 Lloyd-Max codebook。它不是原始 K 的标准差，也不是某个 token 的 metadata；原始 Key 的幅值由 corrected norm 恢复。
>
> Attention 中也使用 1/√D 缩放 QK，两者在当前维度下数值相同，但作用不同：一个定义量化理论分布，一个控制 logit 尺度，不能当成同一个参数。

**可能追问：K 原始 norm 很大时，codebook 的标准差也会变大吗？**

> 不会。codebook 针对归一化后的坐标，D=128 时理论标准差仍是 1/√128。原始 K 的幅值由 per-vector corrected norm 恢复，因此不同 token 的 norm 不会改变共享 centroid 的数值。

---

### 71. 为什么 4-bit 正好需要 16 个 centroid？运行时保存什么？

> 一个 4-bit 无符号 index 有 $2^4=16$ 种取值，因此 codebook 包含 16 个 centroid。Store 时，每个旋转后坐标通过 15 个 decision boundary 落入一个区间，最终保存的是 `0..15` 的 centroid index，而不是 centroid 浮点值。
>
> 两个 index 打包进一个 byte，所以 128 个坐标占 64 B。Decode 时取出 nibble，执行 `centroid[index]`，再乘 K 的校正 norm，恢复用于 QK 累加的近似坐标。
>
> 运行时涉及三个不同存储层次：
>
> - cache：每个坐标只存 4-bit index；
> - global/device 参数：整个 launch 传入一张 `[16]` FP32 centroid table，共 64 B；
> - warp register：CUDA V4–V9 让 lane 0–15 各持有一个 centroid，通过 shuffle lookup。
>
> 15 个 boundary 只在 Store bucketize 时使用，不写进每个 cache slot，也不在 Decode 再次比较。4-bit 决定的是 index 信息量，不表示 centroid 本身也以 4 bit 保存。

**可能追问：16 个 centroid 为什么不计入每个 slot 的 134 B？**

> 它们是整个 launch 共享的参数，不是每个 token/head 独立保存一套。134 B 统计每个 slot 的 K/V index 和三个 metadata。完整设备占用仍应包含那张 64 B 的表，但不能把它重复算到每个 slot。

---

### 72. Lloyd-Max 中相邻 centroid 的 decision boundary 怎么计算？

> 固定一组有序 centroid，在等权标量平方误差下，x 应选择距离最近的重建值。相邻 c_i 和 c_{i+1} 的误差相等时，(x−c_i)²=(x−c_{i+1})²，解得边界 b_i=(c_i+c_{i+1})/2。4-bit 的 16 个 centroid 因此对应 15 个 midpoint。
>
> 当前 Store 对 midpoint 做二分 bucketize，四轮搜索得到 0..15 的 index；源码用 y_vec>=mid_val 更新右侧，所以刚好等于边界时选择较高的区间。两侧误差在边界处相同，但参考实现要遵守同一比较规则。
>
> 边界规则与代表值更新要分开。平方误差下代表值是区间条件均值；改成绝对误差时代表值变为条件中位数，但对等权相邻值的最近邻边界仍可在中点，不能笼统说换损失后 midpoint 必然失效。

**可能追问：坐标刚好等于 midpoint 时会落在哪一边？**

> 数学上两侧平方误差相同；工程上要遵守 Store 的比较约定。当前二分 bucketize 使用 y_vec>=mid_val 时进入右侧，相等时选择较高的区间。Decode 只消费 index，因此判断规则一致比选择哪一侧更重要。

---

### 73. Lloyd-Max 的新 centroid 为什么是区间条件均值而不是区间中点？

> 固定量化区间 $[a,b]$ 后，要选择重建值 $c$ 最小化区间内期望平方误差：
>
> $$
> J(c)=\int_a^b(x-c)^2f(x)dx.
> $$
>
> 令导数为零：
>
> $$
> \frac{dJ}{dc}=-2\int_a^b(x-c)f(x)dx=0,
> $$
>
> 得到：
>
> $$
> c=\frac{\int_a^bxf(x)dx}{\int_a^bf(x)dx}=E[X\mid a\le X\le b].
> $$
>
> 只有当区间内概率密度关于中点对称或近似常数时，它才等于 $(a+b)/2$。Gaussian 在尾部区间明显不均匀，因此简单取几何中点通常不是 MSE 最优重建值。

**可能追问：把平方误差改为绝对误差，新代表值是什么？**

> 固定区间后，最小化绝对误差的代表值是条件中位数，平方误差对应条件均值。需要区分代表值更新与区间边界：对等权相邻重建值，最近邻边界仍可在中点，不能笼统说换损失后二者都必然改变。

---

### 74. TurboQuant 为什么可以使用固定 codebook，而不需要模型级 calibration？

> 归一化去除了每个 K 向量的整体尺度，正交混合又使坐标边缘分布接近只由维度$d$ 决定的球面坐标分布。于是 codebook 可以针对理论近似$N(0,1/d)$ 离线求解，而不是针对某层、某模型的经验直方图求解。
>
> 这是 TurboQuant 的设计优势，不代表所有真实数据都精确服从 Gaussian，也不代表固定 codebook 在任何任务上都必然优于校准量化。工程上仍需用模型质量和下游任务评估验证这种理论近似。
>
> 固定 codebook 成立依赖几个前提：
>
> 1. 每个 K 先按自己的向量 norm 归一化；
> 2. 使用匹配的正交/Hadamard 变换充分混合坐标；
> 3. head dimension 足够高，使 Gaussian approximation 可用；
> 4. 量化目标主要是旋转坐标的 MSE；
> 5. 模型实际分布没有严重偏离理论假设。
>
> 优势是无需 per-model calibration，部署简单且各层共享数值 codebook；风险是理论近似不能自动保证 PPL。应把“无需 calibration”与“无需质量验证”明确区分。
>
> 其中随机正交旋转的分布结论不能无条件套到固定 Hadamard；当前实现采用理论 codebook，是具体工程选择，不是对所有输入分布的保证。

**可能追问：无需 calibration 是否意味着不需要任何模型验证？**

> 不是。无需 calibration 只省去了生成 codebook 的模型采样步骤，真实数据与理论近似仍可能有偏差。最终还要测长上下文、任务指标和生成稳定性；当前仓库的 synthetic output 误差不能替代这些测试。

---

### 75. 一个 FP16 Key 从输入到写入 4-bit KV Cache 经历什么？

> 以 $K\in R^{128}$ 为例，主要数据流是：
>
> 1. 以 FP32 累加计算原始二范数 $s=\lVert K\rVert_2$；
> 2. 归一化得到 $u=K/s$，并处理极小范数的数值边界；
> 3. 用正交矩阵或归一化 Hadamard 变换得到 $z=\Pi u$；
> 4. 用 15 个 midpoint 对每个 $z_i$ bucketize，得到 128 个 4-bit index；
> 5. 将 index 查回的 centroid 组成 $c$，计算其量化后范数 $\lVert c\rVert_2$；
> 6. 开启 norm correction 时保存 $\gamma=s/\lVert c\rVert_2$；
> 7. 每两个 index 打包为一个 byte，按 paged KV Cache 布局写入 64 B K payload；
> 8. 将 FP16 `gamma` 写入该 token/KV-head 的 metadata。
>
> Decode 读出的近似旋转 K 是 $\hat K_r=\gamma c$。本项目随后直接在 tile 内参与 QK，不生成完整的 Global Memory FP16 K buffer。

**可能追问：这些 Store 步骤全部融合在一个 CUDA launch 中吗？**

> 不是当前快照的做法。归一化和旋转在 Python/PyTorch 路径中计算，再交给 Triton Store 做 bucketize、pack、norm correction 和 V 量化写入。项目的“单次 launch 融合”主要限定于 CUDA Decode Stage1。

---

### 76. K 的 `norm` 何时计算？它与 centroid 是什么关系？

> 原始 norm $s=\lVert K\rVert_2$ 在 Store/量化阶段、归一化之前计算。Lloyd-Max centroid 是离线根据目标分布生成的一套全局常量，不由这个 norm 生成。
>
> 开启 norm correction 后，实际保存的标量还会结合量化 centroid 向量的范数：
>
> $$
> \gamma_{stored}=\frac{\lVert K\rVert_2}{\lVert c\rVert_2}.
> $$
>
> 所以 centroid index 描述方向，保存的 norm 类 metadata 恢复幅值并补偿量化后方向向量的范数偏差。面试时不能把它说成 Lloyd-Max 的 scale 参数。

**可能追问：更换 codebook 却复用旧 index 和 norm，会发生什么？**

> 旧 index 对应的重建值改变，原来折叠进 norm 的 centroid-vector 长度也失配。即使 byte 布局完全一致，语义仍然错。codebook、midpoint、rotation 与缓存数据必须作为同一套契约管理。

---

### 77. `norm`、`scale` 和 `QJL` 是不是同一个东西？

> 不是，它们处在不同路径并解决不同问题：
>
> | 名称 | 所在路径 | 作用 |
> |---|---|---|
> | K norm / corrected norm | K centroid quantization | 恢复 K 的整体幅值，并可补偿 centroid 向量范数 |
> | V scale/zero | V affine quantization | 将 `0..15` 的 uniform index 映射回 V 的动态范围 |
> | QJL residual channel | 论文 TurboQuant-Prod | 用随机投影符号和 residual norm 估计残差内积 |
>
> QJL 不是一个浮点 scale。当前本项目不含 QJL 的 4-bit 路径只保存 K norm、 V scale 和 V zero，没有 QJL residual payload。

**可能追问：norm correction 是否等于 QJL 的低成本替代？**

> 不能说数学等价。norm correction 只调整重建向量的整体长度，QJL 则为残差内积构造随机估计，目标与保存的信息不同。当前选择不含 QJL 的格式，是另一种工程取舍，不是把 QJL 压缩成一个 norm。

---

### 78. TurboQuant-MSE 优化什么？TurboQuant-Prod 为什么引入 residual？

> TurboQuant-MSE 选择标量量化器来最小化旋转坐标的重建均方误差，目标可写成：
>
> $$
> E\left[\lVert x-\hat x_{mse}\rVert_2^2\right].
> $$
>
> 但 Attention 真正关心的是 query 与 key 的内积。即使 $\hat x_{mse}$ 已很好地重建 x，残差 $r=x-\hat x_{mse}$ 仍会产生 $q^Tr$，从而扰动 logits。 TurboQuant-Prod 因此通常让主 MSE 通道使用 $b-1$ bit，并用额外 1 bit 的 QJL 通道编码残差信息，目标更直接地降低或校正内积估计误差。

**可能追问：MSE 更小是否保证所有 Query 的内积误差都更小？**

> 不保证逐个 Query 都更小。内积误差是 qᵀr，与残差方向和 Query 方向都有关；较小的残差范数提供上界控制，但不能确定特定方向上的误差排序。这也是区分重建目标与内积目标的原因。

---

### 79. TurboQuant-Prod 的 residual 如何处理？QJL 起什么作用？

> 先计算主量化结果和残差：
>
> $$
> r=x-\hat x_{mse}.
> $$
>
> QJL 使用随机投影得到 residual 的符号信息，并配合 residual norm 等 metadata，在查询时构造 $q^Tr$ 的低成本、无偏估计。最终内积由主通道和残差估计相加：
>
> $$
> q^Tx\approx q^T\hat x_{mse}+\widehat{q^Tr}.
> $$
>
> 因此 QJL 是 residual inner-product estimator，不是对 K 或 V 乘一次的普通 scale，也不是 Lloyd-Max codebook 本身。
>
> 这里的无偏性质需要相应随机投影、归一化与估计器假设，并不是对一次固定估计保证误差为零；当前 CUDA 4bit_nc 热路径没有实现这一 residual 通道。

**可能追问：残差内积估计无偏，能保证 softmax 输出无偏吗？**

> 不能。softmax 是非线性函数，对 logit 的无偏扰动经过 softmax 后一般不再保持无偏，而且估计方差会影响概率分布。因此内积估计性质不能直接变成 Attention 或模型质量保证。

---

### 80. 论文有 QJL，为什么当前 vLLM decode 可以不用？

> 需要区分论文的算法变体与上传项目采用的具体格式。当前 preset 是 turboquant_4bit_nc：K 走旋转、Lloyd-Max index 和 norm correction，V 走 uniform index、scale、zero。cache 中没有 QJL residual 的 payload 或 residual norm，CUDA QK 也没有对应估计通道。
>
> QJL 的目标是对残差内积提供随机估计，理论内积性质并不自动变成 softmax 或模型质量保证。工程上可以选择不含该通道的实现来保持当前布局和解码路径简单，但这不是说 norm correction 与 QJL 数学等价。
>
> 因此我的项目介绍会明确“基于此快照的 4bit_nc 路径”，不会声称完整实现了 TurboQuant-Prod 的所有能力，也不会把上传的 vLLM 片段扩展成对所有版本上游行为的结论。

**可能追问：未来补 QJL 能否只在当前解码后加一次乘法？**

> 不能。Store 要计算并保存残差信息，布局和 metadata 会改变，Query 侧也要匹配随机投影，QK 路径要加入残差估计。还必须重新比较容量、时延与质量，已经超出当前 134 B slot 的实现范围。

---

### 81. K 和 V 是否使用完全相同的量化方法？

> 不是。K 路径是：归一化、正交/Hadamard 旋转、Lloyd-Max 非均匀 centroid quantization，并保存 4-bit index 和 corrected norm。V 路径通常不做这套 centroid rotation，而是按向量求 `vmin/vmax`：
>
> $$
> scale=\max\left(\frac{v_{max}-v_{min}}{15},10^{-8}\right),
> \qquad zero=v_{min},
> $$
>
> $$
> q=\mathrm{clip}\left(\mathrm{round}
> \frac{v-zero}{scale},0,15\right),
> \qquad \hat v=q\cdot scale+zero.
> $$
>
> 本项目每个 token/KV-head 保存 64 B K index、64 B V index，以及三个 FP16 metadata：K corrected norm、V scale、V zero，总计 134 B。

**可能追问：为什么 V 不需要为了 QK 内积而做匹配旋转？**

> QK 的坐标系一致性涉及 Query 和 Key；Value 参与的是按 token 权重的加权和。当前直接重建原坐标系 V，输出也是原 Value 坐标系。如果另行旋转 V，输出侧还要处理对应逆变换。

---

### 82. 为什么 K 适合 centroid quantization，而 V 可用 affine quantization？

> K 进入 QK 内积并进一步进入 softmax，logit 误差可能改变整行 attention weight；归一化和旋转后，K 坐标又具有可利用的稳定、近 Gaussian 分布，因此用针对该分布优化的非均匀 centroid 有明确动机。
>
> V 在 softmax 权重确定后参加加权和。工程上可以按每个 V 向量的实际`min/max` 使用简单 affine quantization，Decode 只需一次乘加，成本较低。这是一项算法与实现折中，不应表述为“V 对误差不敏感”或“V 永远不值得做非均匀量化”；最终仍要通过输出质量评估决定。

**可能追问：K 的量化误差与 V 的量化误差分别怎样影响结果？**

> K 误差先扰动 logits，再通过 softmax 改变所有 token 的权重；V 误差在给定权重下直接进入加权和。两条路径的敏感性不同，但不能仅凭角色就断言某种量化永远最优，仍要以输出和模型质量评估。

---

### 83. K 被旋转后，为什么 Query 也必须做相应旋转？

> 使用行向量约定，令：
>
> $$
> K_r=K\Pi^T,\qquad Q_r=Q\Pi^T,
> $$
>
> 其中 $\Pi$ 为正交矩阵，即 $\Pi^T\Pi=I$。那么：
>
> $$
> Q_rK_r^T
> =Q\Pi^T(K\Pi^T)^T
> =Q\Pi^T\Pi K^T
> =QK^T.
> $$
>
> 如果只旋转 K 而不旋转 Q，计算的是 $Q\Pi K^T$ 或其对应约定形式，不再等于原始 attention score。代码采用列向量时左右乘形式会变化，但“Q/K 必须进入同一个正交坐标系”这一结论不变。

**可能追问：rotation 与 RoPE 是同一个变换吗？**

> 不是。RoPE 编码位置信息，TurboQuant rotation 用于改善量化坐标分布。它们的应用顺序和 Q/K 匹配都必须与框架契约一致；本 benchmark 从预旋转 Query 与压缩 cache 开始，没有测整个 RoPE 和投影链路。

---

### 84. 为什么 decode 可以直接从 index lookup 进入 QK accumulation？

> QK 只需要逐坐标使用近似 K，并不要求先拥有一个完整、连续、长期存在的 FP16 K tensor。因此 Kernel 可以在 tile 内执行：nibble unpack、centroid lookup、乘 corrected norm，然后立即送入 Tensor Core 或普通 FMA 累加。
>
> 最大的收益是避免把完整 FP16 K 写回 Global Memory 后再读一次，同时避免额外 dequant Kernel、临时显存和 launch 边界。Shared Memory 仍可作为 tile staging，但解码后的数据只在 CTA 内短暂存在。V 也可类似地用 `index*scale+zero` 后直接进入 PV。

**可能追问：Decode 为什么不再需要 midpoint 二分搜索？**

> Store 已经把连续坐标归到具体量化区间，cache 中的 nibble 就是答案。Decode 做的是从 index 到 centroid 的直接重建，不再决定区间。把 midpoint 搜索放进 Decode 会重复无用工作。

---

### 85. 4096 个历史 token 后生成新 token，Q 是否和自己的 K 做 attention？

> 要先明确“4096 个历史 token”是否包含当前 decode position。标准 causal self-attention 对位置 $t$ 允许访问所有 $j\le t$，因此当前位置自己的 K/V 应当参与 attention；只屏蔽未来位置 $j>t$。
>
> 工程上常在该层计算当前 token 的 Q/K/V，将当前 K/V 写入 cache，再让 Q 读取长度为 `context_len` 的有效 cache。如果 `4096` 表示写入当前 K 后的有效长度， QK 有 4096 个 K；如果严格表示此前已有 4096 个旧 K，随后又追加当前 K，则有 4097 个。不要只凭“正在生成第几个输出 token”判断，应该检查 API 中`context_len/seq_len` 的定义和 cache append 顺序。

**可能追问：固定 L=4096 的 benchmark 是否模拟了从 4096 增长到 4097？**

> 没有，它消费构造好的 4096-token cache，不包含完整生成步的写入和长度更新。真实自回归中当前 token 是否已写入，取决于调用时序和 seq_lens 契约；不能把静态 benchmark 当成动态 append 流程证据。

---

### 86. KV Cache 从 FP16 降到 4-bit，为什么 decode 不一定严格加速 4 倍？

> 4 倍主要是 K/V payload 字节数的理论缩减，不是整个 Kernel 时间的缩减。实际路径还包含 metadata、page-table 访问、nibble unpack、centroid lookup、V 反量化、QK/PV、online softmax、同步和 Stage2。固定 launch、调度与计算开销不会按 cache bit 数同步缩小。
>
> 此外，压缩后瓶颈可能从 DRAM 转向 Tensor Core、MIO、整数流水线、dependency stall 或 occupancy。最终 speedup 受 Amdahl 定律和新瓶颈约束，必须用同 workload 实测，不能从 `16/4` 直接宣布 4x 端到端加速。

**可能追问：量化后 Kernel 甚至变慢，是否说明压缩毫无价值？**

> 也不能。它可能仍显著节约 cache 容量，支持更大 batch 或上下文，只是当前单步时延未受益。应该分别衡量容量、单请求延迟和服务吞吐，再判断格式与 Kernel 是否满足目标。

---

### 87. centroid lookup 增加指令，为什么仍可能比 FP16 KV Cache 快？

> Decode 长上下文通常需要搬运大量 KV 数据，而 16-entry codebook 很小、可被缓存或放入适合的只读存储。用少量 unpack、lookup 和乘法换取约 4 倍 payload 流量下降，在 memory-bound 场景中通常是有利的典型“用计算换带宽”。
>
> 是否获益取决于 lookup 的实现、cache 命中、指令依赖和原 Kernel 的瓶颈。如果上下文很短或实现造成严重 serialization，额外指令可能超过流量收益，所以仍需 benchmark，而不是把这种权衡当作无条件结论。
>
> 本项目进一步避免了真正的随机 global gather：每个 warp 的 lane 0–15 先各持有一个 centroid，nibble index 作为 `__shfl_sync` 的 source lane。于是 hot loop 的 lookup 主要消耗 shuffle、整数位运算和依赖延迟，而不是 128 次独立 table memory load。
>
> 因此判断收益要同时比较：减少的 KV bytes、增加的 integer/shuffle 指令、register 压力、eligible warps 和 long/short-scoreboard。codebook 只有 64 B 只能说明容量很小，不能说明 lookup 链路必然免费。

**可能追问：怎么验证减少的字节没有被解码指令成本抵消？**

> 建立同范围、同 shape 的 FP16 对照，同时观察时间和当前版本 profiler。既要看 DRAM/L2 流量下降，也要看解包、shuffle 和 shared 路径的代价。当前仓库尚无这组成熟 FP16 对比，所以只能解释潜在收益。

---

### 88. DRAM Throughput 不高但 L1/TEX 或 MIO 很高，应该如何解释？

> 如果 DRAM Throughput 不高，而 L1/TEX、MIO 或某类 stall 很高，我会先判断是否受片上数据通路、依赖或同步限制，不能直接说显存带宽已经饱和。低比特减少了部分字节，但 shared load/store、细粒度操作和不足的延迟隐藏仍可能阻止持续发出请求。
>
> 指标归因必须落到具体指令。V9 热循环中的 centroid lookup 是 register shuffle；unpack 和类型转换也不能仅因为位于解码路径，就全部归到 L1/TEX 或 MIO。需要结合 Source/SASS、shared transactions、bank conflicts、eligible warps 和 scoreboard/barrier stall 分别判断。
>
> 此外，当前压缩包的旧 NCU 不能代表 V9。这里说明的是拿到新报告后的诊断方法，实际瓶颈必须以当前版本动态数据为准。

**可能追问：L1/TEX 高能否直接证明 centroid lookup 是瓶颈？**

> 不能。该层还服务其他访问，汇总高值不能独立归因。V9 的热循环 centroid lookup 使用 register shuffle，因此应结合具体 SASS、source attribution、访存请求与 stall，先确认真正繁忙的操作再下结论。

---

### 89. `centroid[index]` 这种数据相关 lookup 会给 GPU 哪些部分带来压力？

> 我会先区分 lookup 的实现方式。若直接访问内存表，数据相关索引可能带来额外加载、缓存行为和 load-use 依赖；若放在 constant memory，不同 lane 的不同地址也可能影响访问效率。但这些属于备选设计，不能全部说成当前 V9 已发生的问题。
>
> V9 让每个 warp 的 lane 0–15 各持有一个 centroid，用 nibble 作为 __shfl_sync 的源 lane，所以热循环不为每个坐标发起 global table gather。它仍需要 nibble 提取、shuffle、乘 norm、FP16 转换和 shared 写入，也会与 MMA fragment、metadata 和 softmax 状态竞争寄存器。
>
> 因此 64 B 小表解决的是表容量与重复读取问题，不等于 lookup 免费。下一步应结合当前 SASS 和 NCU 看指令依赖、寄存器压力、shared 路径与延迟隐藏，再决定是否值得更换查表方式。

**可能追问：V9 的 codebook 很小，为什么 lookup 仍不是零成本？**

> 表容量只有 64 B，不代表索引解包和查询免费。每个 nibble 要经历位操作、数据相关 shuffle、乘 norm 和精度转换，形成执行与依赖成本。小表主要降低存储问题，不能消除所有计算路径开销。

---

### 90. “TurboQuant 不就是 INT4 KV Cache 吗？”如何用 30 秒回答？

> 它确实把 K/V 的主要数据编码成 4-bit，但 K 不是普通线性 INT4。Key 先按向量归一化并旋转，再用 16 个非均匀 Lloyd-Max centroid 表示坐标；cache 保存 centroid index 和 corrected norm。Query 使用匹配旋转，Value 则使用 per-vector affine 4-bit 量化。
>
> 我优化的是如何直接消费这种压缩格式：在 Stage1 内解包和重建 K/V，马上进入 GQA 的 QK、online softmax 和 PV，避免把完整 FP16 KV 临时张量写回显存。这里的 INT4 是存储编码，实际矩阵乘使用 FP16 输入、FP32 累加的 Tensor Core。
>
> 当前路径不含 QJL residual，3.82 倍是格式容量收益；实际时延收益按指定版本和 Stage1/Full 范围分别报告。

**可能追问：一句话概括项目与“只把 dtype 改成 INT4”的差别？**

> 我处理的是低比特格式的完整消费链：按真实 cache 布局解包、恢复非均匀 K 与 affine V，并与 GQA Attention 融合计算。INT4 是存储编码，实际 QK/PV 仍使用 FP16 Tensor Core。

---

## 简历逐句拷问与硬件 Roofline

### 91. 请用一分钟介绍简历中的 TurboQuant CUDA Decode 项目

> 我的项目围绕大模型 Decode 阶段的 KV Cache 压缩与 Attention Kernel 优化展开。我基于上传快照中的 TurboQuant 4bit_nc 格式，为 RTX 4090 上 Qwen3-4B 形状的 GQA Decode 实现和优化 CUDA 路径，让 GPU 直接读取压缩 cache，在同一个 Stage1 中完成解包、K 查表、V 反量化、QK、online softmax 和 PV，再用 Stage2 合并 split。
>
> 优化从 CTA 复用和 warp 映射推进到 tiled WMMA、固定形状特化、fragment 直接写回、32-bit packed load 和 barrier 合并。V8 用原生 m16n8k16 把四个 Query 的有效槽位比例从 25% 提高到 50%，V9 再消除 QK score 的 shared 往返。
>
> 仓库记录中，KV slot 从 512 B 降到 134 B，约压缩 3.82 倍；V7 Full 从 Triton 的 1.104148 ms 降到 0.664842 ms，约 1.66 倍，V9 Stage1 为 0.484516 ms。完整计时范围是预旋转 Query 下的 Stage1+Stage2，不包含 Store、rotation 和整模型推理。

**可能追问：面试官只追问一个结果，你会选哪个？**

> 我会优先给出 V7 Full 从 Triton 的 1.104148 ms 降到 0.664842 ms，约 1.66 倍，并立即说明 Full 指预旋转 Query 下的 Stage1+Stage2。它比单说压缩率或内部最慢版本加速更能说明完整计算链路的收益。

---

### 92. RTX 4090 的 FLOPS、显存带宽和与你 Kernel 相关的规格是多少？

> 我会优先记与本 Kernel 相关的规格：RTX 4090 是 Ada、Compute Capability 8.9，有 128 个 SM、512 个 Tensor Core，24 GB GDDR6X，理论显存带宽 1008 GB/s，L2 为 72 MiB。普通 FP32 峰值约 82.6 TFLOPS；当前 dense FP16 输入、FP32 accumulation 的 Tensor 路径约 165.2 TFLOPS。[NVIDIA Ada 白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
>
> 这里必须按指令的数据类型和累加类型选峰值：V8/V9 使用 f32.f16.f16.f32 的 dense MMA，所以不能拿 sparse 峰值或更低精度的 PetaOPS 作分母。理论带宽也不是每个 Kernel 都能达到的实际带宽。
>
> 分析性能时，我会将这些规格和当前计算量、逻辑流量及编译资源结合，用于提出瓶颈假设；实际时钟、缓存命中、DRAM bytes 和硬件利用率仍需要实验记录或 profiler。

**可能追问：能用官方 Boost 峰值直接当作实验中的实际频率吗？**

> 不能。峰值规格用于理论参照，实际频率受功耗、温度和运行状态影响。复现实验应记录设备状态；Roofline 用理论峰值时，也要标明它是理想上限，不能据此声称 Kernel 已达到某个实测利用率。

---

### 93. 为什么 RTX 4090 会同时出现 82.6、165.2、330.4 TFLOPS 和 1.321 PetaOPS？

> 这些数字不是互相矛盾，而是对应不同运算路径。约 82.6 TFLOPS 是普通 FP32；165.2 是 Tensor Core 的 dense FP16 输入、FP32 累加；330.4 是同一 FP32 累加路径在结构化稀疏条件下的标称值。白皮书还列出约 330.3 的 dense FP16 累加路径，以及低精度加稀疏条件下约 1.321 PetaOPS 的指标。[NVIDIA Ada 白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
>
> 所以只看到“FP16 Tensor 峰值”还不够，必须同时问输入类型、accumulator 类型、dense/sparse 和频率条件。当前指令是 mma.sync 的 dense FP16 输入、FP32 累加，不使用 2:4 sparsity，因此相关理论峰值约为 165.2 TFLOPS。
>
> 我会先按这条具体路径建立 Roofline，再区分 useful FLOPs 与 padding 后的 MMA 工作量。混用峰值和计算量，会让算术强度拐点及所谓利用率失去比较意义。

**可能追问：为什么只看到“FP16 330.4 TFLOPS”还不能直接引用？**

> 还缺少 accumulation dtype 和 sparsity 条件。白皮书分别列出约 330.3 的 dense FP16 accumulate，以及 330.4 的 sparse FP32 accumulate；当前指令是 dense FP16 输入、FP32 accumulate，应按这条具体路径选择约 165.2 TFLOPS。

---

### 94. `sm_89` 是什么意思？为什么不是说“专门使用 4090 指令”？

> sm_89 表示 CUDA 针对 Compute Capability 8.9 的目标架构，RTX 4090 属于这一类。它决定目标代码可使用的架构能力和资源约束，并不是某一个显卡型号独占的指令名字。
>
> 项目还根据 4090 上测得的固定 workload 来选择 tile、warp 分工和资源取舍，所以“按 sm_89 编译”与“为 4090 的性能调优”是两个层次。原生 m16n8k16 的矩阵与寄存器语义也不是为 4090 临时发明的，代码需要遵守 PTX 指令契约。
>
> 迁移到其他设备时，我会分别检查二进制兼容、所用指令支持、fragment 写回及数值正确性，再重测性能。相同 Compute Capability 不保证时间相同，不同 Compute Capability 也不意味着算法必然无法迁移。

**可能追问：同为 sm_89 的其他显卡，性能能按 SM 数直接缩放吗？**

> 只能作粗略假设，不能作结论。不同卡的时钟、带宽、cache 容量和工作集驻留都会改变瓶颈；相同指令兼容性不代表性能比例固定。需要在目标设备重新测同一 workload。

---

### 95. `4bit_nc` 中的 `nc` 是什么？

> `nc` 是 **norm correction**，不是 non-contiguous，也不是某种 CUDA Core 模式。对一个 Key 向量，先记原向量范数为
>
> $$
> n = \lVert K \rVert_2.
> $$
>
> 将单位化并旋转后的坐标量化到 centroid 后，重建向量的范数一般不再严格为 1。因此保存修正系数
>
> $$
> \gamma = \frac{\lVert K \rVert_2}
> {\lVert \hat{u} \rVert_2},
> $$
>
> 其中 $\hat{u}$ 是 centroid lookup 后的量化方向。Decode 中重建的是$\hat{K}=\gamma\hat{u}$。每个 token/KV head 保存一个 FP16 corrected norm；这既恢复原始幅值，也修正 centroid 量化造成的范数漂移。

**可能追问：不开启 nc 时，stored norm 的含义有什么变化？**

> 不开启校正时保存原始 K 范数，用它放大量化 centroid 向量；开启后额外除以 centroid 向量范数，补偿重建长度偏差。因此读取字段名相同，也要确认 Store 使用的是哪一个 preset。

---

### 96. 简历中的 KV Cache 压缩 `3.82x` 是怎么计算的？

> 固定 `head_dim=128`，每个 token、每个 KV head 的 FP16 K/V payload 是：
>
> $$
> 128\times 2\ \text{B} + 128\times 2\ \text{B}=512\ \text{B}.
> $$
>
> `4bit_nc` slot 包含：
>
> | 字段 | 字节数 |
> |---|---:|
> | K 的 128 个 4-bit index | 64 B |
> | V 的 128 个 4-bit index | 64 B |
> | K corrected norm，FP16 | 2 B |
> | V scale，FP16 | 2 B |
> | V zero，FP16 | 2 B |
> | 合计 | 134 B |
>
> 所以
>
> $$
> \text{compression ratio}=\frac{512}{134}=3.8209\times.
> $$
>
> 它不是严格 4 倍，因为每个 slot 还有 6 B metadata。这个数字只计算 K/V cache 主体，不代表模型全部显存占用也下降 3.82 倍。

**可能追问：metadata 占比到底有多大？**

> 每 slot 中 metadata 是 6/134≈4.48%，payload 占其余部分。当前 workload 的 metadata 共 12 MiB，压缩 cache 共 268 MiB。比例不大，但数量随 token 和 KV head 增长，因此仍值得设计合并访问。

---

### 97. 为什么说工作负载是“Qwen3-4B 形状”，而不是完整跑了 Qwen3-4B？

> 官方 Qwen3-4B 配置中的 attention 形状是 `32` 个 Q head、`8` 个 KV head、`head_dim=128`，即 GQA ratio 为 4。本 benchmark 采用相同的 head shape，但`batch=64`、`context=4096`、`num_splits=32` 是项目自行固定的测试参数，并未加载完整模型权重或执行全部 Transformer layer。
>
> 严谨说法是“Qwen3-4B-shaped fixed workload”。如果说“Qwen3-4B 端到端推理提速 1.66x”，会把单层 attention decode Kernel 结果错误外推到整个模型。
>
> 仓库输入由脚本构造，未执行完整 QKV projection、全部模型层和自回归生成，因此这些时间不能当作 Qwen3-4B 的每 token 延迟。真正模型级结论要在集成后另外测量。

**可能追问：完整模型验证需要补哪些不同于 Kernel benchmark 的内容？**

> 至少要接入真实层调用和 cache 生命周期，检查 projection、RoPE、rotation、调度与 Decode 的衔接，再测试生成质量和 token latency。当前固定随机张量只匹配 Attention 维度，不能证明所有模型层已经运行。

---

### 98. `0.664842 ms` 的 Full Decode 到底包含和不包含什么？

> 它包含：
>
> 1. V7 Stage1：读取压缩 K/V、centroid lookup/反量化、QK、online softmax、
>    每个 split 的 partial output 和 LSE；
> 2. Stage2：按 split LSE 做稳定归一化并归并 partial output。
>
> 它不包含 Store Kernel、Query rotation、QKV projection、RoPE、采样、模型其他层、内存分配和 JIT 编译。所谓 `Full Decode` 是本项目 benchmark 中 Stage1 + Stage2 的完整 attention decode 链，不是完整 token generation latency。

**可能追问：Full 中间缓冲分配也算在 0.664842 ms 内吗？**

> 不算，benchmark 预分配 mid_o、output 和 LSE，再重复运行两阶段计算。这个测量适合比较稳定执行的 Kernel 链路；若评估服务端延迟，还要另外统计调度、内存管理与其他算子。

---

### 99. “融合至单次 Stage1 launch”具体融合了什么？为什么有价值？

> 单个 Stage1 CTA 内完成：
>
> 数据流为：加载 packed K/V → 提取 4-bit nibble → K centroid lookup + corrected norm → V affine 反量化 → GQA QK Tensor Core MMA → online softmax 状态更新 → probability × V 累加 → 写出 split partial output 和 LSE。
>
> 价值不是单纯“少几个 Kernel 名字”，而是避免把完整 FP16 K/V、score 或 probability 中间张量写回 global memory，降低 launch 开销并让解码结果尽可能停留在寄存器/Shared Memory 中。Stage2 仍是另一个 launch，因为不同 CTA 的 split 结果需要全局同步后才能归并。

**可能追问：融合越多是否一定越快？**

> 不一定。融合减少中间全局流量和 launch，但可能增大寄存器、shared footprint 和同步范围。当前把 tile 级解码与 Attention 放在 Stage1，而把跨 CTA split 合并单独执行，就是在数据复用与并行组织之间取舍。

---

### 100. 固定工作负载下会启动多少 CTA？每个 CTA 做什么？

> Stage1 grid 是
>
> $$
> (B,H_{kv},S)=(64,8,32),
> $$
>
> 因此 CTA 数量是
>
> $$
> 64\times 8\times 32=16{,}384.
> $$
>
> 每个 CTA 有 128 threads，即 4 个 warp，负责一个 batch、一个 KV head、一个 split，以及该 KV head 对应的全部 4 个 Q head。每个 split 有 128 token，按 16-token tile 处理 8 轮。这个映射让同一组 K/V 被 4 个 Q head 复用。

**可能追问：16,384 个 CTA 是否会同时驻留在 GPU？**

> 不会，这是整个 grid 的总 CTA 数。每个 SM 同时能驻留多少 CTA 由线程、寄存器、Shared Memory 和硬件 block 上限共同决定，剩余 CTA 分批调度。不能用总 grid 数直接计算 achieved occupancy。

---

### 101. Qwen3-4B 固定 workload 的有效计算量是多少？

> 只统计算法上必要的 QK 和 PV FMA，并按一次 FMA 等于 2 FLOPs：
>
> $$
> \begin{aligned}
> F_{QK} &= 2BH_qLD,\\
> F_{PV} &= 2BH_qLD,\\
> F_{useful} &= 4BH_qLD.
> \end{aligned}
> $$
>
> 代入 $B=64,H_q=32,L=4096,D=128$：
>
> $$
> F_{useful}=4\times64\times32\times4096\times128
> =4{,}294{,}967{,}296\ \text{FLOPs},
> $$
>
> 即约 `4.295 GFLOPs`。这不包含 unpack、lookup、地址计算、softmax exp、归约等非 FMA 指令，因此只是 useful FLOPs，不是完整指令成本。

**可能追问：为什么计算量公式前面是 4，而不是 2？**

> QK 每次乘加按两个 FLOP 计算，得到 2BHqLD；PV 还有同量级的乘加，再加一份 2BHqLD，所以合计 4BHqLD。这不统计解包、指数、归约和地址指令，也没有统计 padding 的无效 MMA 工作。

---

### 102. Stage1 逻辑访存量大约是多少？

> 我按固定 V9 Stage1 的逻辑访问来估算，而不预先假设 cache hit/miss。主要项如下：
>
> | 数据 | 逻辑字节数 |
> |---|---:|
> | K/V payload 与三个 metadata | 281,018,368 B，268 MiB |
> | 各 split 对 Query 的重复加载 | 33,554,432 B，32 MiB |
> | mid_o 写出 | 33,816,576 B，32.25 MiB |
> | 每 tile 的 block table entry | 524,288 B，0.5 MiB |
> | 每 CTA 的 seq_len 读取 | 65,536 B，0.0625 MiB |
> | 以上主要项合计 | 348,979,200 B，332.8125 MiB |
>
> 这个模型忽略了 centroid 表等较小项；若按每 CTA 四个 warp 各读 64 B 表计算，还会有约 4 MiB 的逻辑表读取，多数可被缓存吸收。因此 348.98 MB 是主要项近似，不是逐条机器加载的精确清单。
>
> 同样，Query 原始 tensor 只有 1 MiB，但 32 个 split 产生 32 MiB 逻辑读取。实际 DRAM 流量还取决于 L2 命中、事务粒度和写回行为，不能直接把本表当成 profiler 的 DRAM bytes。

**可能追问：Q 原始只有 1 MiB，为什么表里计了 32 MiB？**

> 每个 split CTA 都重新读取自己的四个 Query head，32 个 split 形成 32 倍逻辑读取。硬件上这些重复请求可能命中 L2，所以该项不能直接等同于 32 MiB DRAM 读取，这正是逻辑流量与物理流量的区别。

---

### 103. 这个 Kernel 的算术强度大约是多少？

> 有两个有用口径。
>
> 只看每个 token/KV-head 的压缩 cache，4 个 GQA query head 对 K 做 QK、再对 V 做 PV，useful FLOPs 是 $4GD=4\times4\times128=2048$，读取 134 B，因此：
>
> $$
> AI_{cache}=\frac{2048}{134}=15.28\ \text{FLOP/B}.
> $$
>
> 若把 Q 重读、`mid_o` 写出、block table 等完整 Stage1 逻辑流量计入：
>
> $$
> AI_{stage1}=\frac{4.294967\ \text{GFLOP}}
> {348.9792\ \text{MB}}=12.31\ \text{FLOP/B}.
> $$
>
> 面试时先声明分子是 useful FLOPs、分母是 logical bytes。不同统计口径不可直接与 NCU 的硬件 counter 混用。
>
> 上述字节与 MMA 工作量属于固定形状下的模型估算：逻辑流量只统计主要项，MMA 工作量由 tile 与 padding 推导，均不是当前 V9 的动态 NCU counter。

**可能追问：增加 GQA ratio 会怎样影响 cache-only 算术强度？**

> 在 D 和单个 KV 向量格式不变时，同一份 K/V 被更多 Query 复用，useful FLOPs 随 G 增长，而该份 cache 字节基本不变，因此 AI 近似随 G 增长。实际实现还要处理更多 Query 与 output 状态，资源成本不会消失。

---

### 104. V8/V9 的 useful FLOPs 与 Tensor Core 实际执行 FLOPs为什么不同？

> V8/V9 使用 `m16n8k16` MMA，但当前 GQA group 只有 4 个 Q head，因此 N 方向 8 个槽位中只有 4 个有数学意义，slot utilization 为 50%。硬件仍执行完整 MMA，所以估算的 Tensor FLOPs 是：
>
> $$
> F_{executed}\approx \frac{F_{useful}}{0.5}=8.590\ \text{GFLOPs}.
> $$
>
> 报告算法性能时用 useful FLOPs；对比 165.2 TFLOPS Tensor 峰值时应使用 executed FLOPs。若拿 4.295 GFLOPs 除以 Tensor 峰值，会把空槽成本隐藏掉。
>
> 上述字节与 MMA 工作量属于固定形状下的模型估算：逻辑流量只统计主要项，MMA 工作量由 tile 与 padding 推导，均不是当前 V9 的动态 NCU counter。

**可能追问：8.59 GFLOPs 是 NCU 测出来的动态指令量吗？**

> 不是，它由固定 tile 数和半满 N 维推算，是当前映射下的 dense MMA 工作量估计。要报告动态硬件执行量，仍需当前二进制与 profiler counter；不能把这个估计直接命名为实测 Tensor 吞吐。

---

### 105. V9 的有效吞吐和逻辑带宽是多少？应该怎么表述？

> V9 Stage1 时间为 `0.484516 ms`：
>
> $$
> \text{useful throughput}=\frac{4.295\ \text{GFLOP}}{0.484516\ \text{ms}}
> =8.86\ \text{TFLOP/s},
> $$
>
> $$
> \text{executed Tensor throughput}\approx17.73\ \text{TFLOP/s},
> $$
>
> $$
> \text{logical bandwidth}=\frac{348.9792\ \text{MB}}{0.484516\ \text{ms}}
> =720.3\ \text{GB/s}.
> $$
>
> 最后一个数字只能叫“根据逻辑字节估算的有效/逻辑带宽”，不能说 NCU 实测 DRAM bandwidth 是 720 GB/s。两者分母相同，但分子含义不同。
>
> 上述字节与 MMA 工作量属于固定形状下的模型估算：逻辑流量只统计主要项，MMA 工作量由 tile 与 padding 推导，均不是当前 V9 的动态 NCU counter。

**可能追问：720.3 GB/s 能不能除以 1008 得到 DRAM 利用率？**

> 不能作为实测 DRAM 利用率。分子是逻辑字节除时间，包含可能命中缓存的重复访问，和 NCU 统计的 DRAM bytes 不同。最多把它当作一种模型化参考比例，正式 DRAM 饱和结论需要硬件计数。

---

### 106. 怎么用 Roofline 判断它偏 compute-bound 还是 memory-bound？

> 我会先做一个简化 Roofline 判断。用约 165.2 TFLOPS 的相关 dense Tensor 峰值除以 1008 GB/s，拐点约为 163.9 FLOP/B。当前主要逻辑流量模型下，useful AI 约 12.31 FLOP/B；考虑 N 维半满，估算的 MMA 工作量口径约 24.62 FLOP/B，仍明显低于这个拐点。
>
> 对应地，8.59 GFLOPs 的 dense MMA 工作量除以峰值，得到约 0.052 ms 的理想计算时间；若假设约 348.98 MB 的主要逻辑流量全部经过 DRAM，带宽时间约 0.346 ms。后者只是一个假设场景，因为 Query 重读等可能命中缓存，不能称为严格的运行时间下界。
>
> 因此我的初步判断是数据搬运侧更值得关注，而不是已经证明 Tensor 算力饱和。要确认是 DRAM、片上存储、解包指令、依赖还是同步主导，需要当前 V9 的 NCU。逻辑 Roofline 用于提出假设，不替代分层流量和动态瓶颈分析。

**可能追问：0.346 ms 为什么不是严格的运行时间下界？**

> 它把估算的所有逻辑流量都当成 DRAM 访问，再除以理论带宽。但 Query 重读等可能命中缓存，实际 DRAM bytes 不一定等于这个分子，所以这是简化场景的带宽时间估计；严格分析应使用分层流量或实测 DRAM bytes。

---

### 107. 面试官只给 20 秒，如何严谨回答 compute-bound 还是 memory-bound？

> 我会说：从静态模型看，它更偏数据搬运侧。V9 的主要逻辑算术强度约 12.3 useful FLOP/B，即使计入半满 MMA 的无效槽位，也只有约 24.6 FLOP/B，显著低于当前 Tensor 路径的理论拐点约 164 FLOP/B。
>
> 但我不会仅凭这个模型说它已经被 DRAM 带宽饱和限制。当前还缺 V9 的正式动态 NCU，需要看真实 DRAM/L2 流量、Tensor pipe、eligible warps 和 stall，区分带宽、依赖、同步或指令执行问题。4090 的显存是 GDDR6X，表述时用显存或 DRAM，不把它叫 HBM。

**可能追问：4090 的显存应该叫 HBM 吗？**

> 不应该，RTX 4090 使用 GDDR6X。讨论这张卡时我会说显存或 DRAM 带宽，避免把数据中心 GPU 常见的 HBM 名称套过来；这不改变 Roofline 方法，但关系到硬件口径是否准确。

---

### 108. 用 NCU 最终确认瓶颈时，你会看哪些指标？

> 我会按下面这些问题检查：
>
> | 问题 | 重点指标/现象 |
> |---|---|
> | DRAM 是否接近上限 | `dram__bytes`、DRAM throughput、read/write sectors |
> | L2 是否吸收流量 | L2 hit rate、L2 sectors、L2 throughput |
> | Tensor Core 是否忙 | HMMA pipe utilization、Tensor active cycles |
> | 是否 latency-bound | long/short scoreboard stall、eligible warps per cycle |
> | Shared Memory 是否受限 | bank conflicts、MIO throttle、shared transactions |
> | 同步是否过多 | barrier stall、每 CTA barrier 动态次数 |
> | occupancy 是否不足 | achieved occupancy、active warps、register/shared limit |
> | 是否发生 spill | local load/store、stack frame、ptxas register 信息 |
> | load 是否合并 | global load efficiency、sector/request、replay |
>
> 先用 Speed-of-Light/Memory Workload Analysis 定位大类，再下钻到 Scheduler、Warp State 和 Source/SASS。不能只凭单个“DRAM Throughput %”做结论。

**可能追问：看见 barrier stall 高，就直接删 barrier 吗？**

> 不能。先查等待的是哪段生产者工作、是否存在负载不均，以及 barrier 是否承担真实数据依赖。只有能证明覆盖前所有消费者已完成，才可以合并或删除；否则需要调整工作分配，而不是冒险去掉同步。

---

### 109. 为什么 logical bandwidth 不等于 NCU 的 DRAM bandwidth？

> `logical bytes / time` 是算法视角：按照张量语义推算应读取和写入多少数据。 NCU DRAM bytes 是硬件视角：只统计真正越过 L2 与显存控制器的数据。二者差异来自 L2 hit、cache line/sector 粒度、未合并访问、重复 transaction、writeback、 ECC/计数口径等。
>
> 逻辑带宽适合跨实现比较“单位有效数据处理速度”；DRAM 带宽用于判断硬件显存是否饱和。项目当前的 `720.3 GB/s` 属于前者。

**可能追问：两版 Kernel 的 logical bytes 相同，DRAM bytes 也会相同吗？**

> 不一定。线程映射、访问顺序、缓存命中和事务粒度都可能改变实际流量。向量化还可能主要减少指令而不减少有效字节。因此性能对照要同时看逻辑工作、物理流量和执行路径。

---

### 110. 如果 NCU 显示 DRAM 只有峰值的 55%，还能说它偏 memory-bound 吗？

> DRAM 只有峰值 55%，说明还没有直接看到显存带宽饱和，不能据此认定 compute-bound，也不能继续笼统坚持“纯 DRAM bound”。我会分别检查 L2 和 shared 的访问情况、scoreboard 等待、eligible warps、Tensor pipe，以及整数/解包和 barrier 的执行成本。
>
> 例如页映射改变可能影响跨 tile 局部性，但不必然破坏页内合并；V9 的 centroid lookup 是寄存器 shuffle，不能归因成大量 global table gather。如果主要是未隐藏的加载延迟或片上存储压力，可以说受对应存储路径限制；如果 barrier 或整数指令主导，则应直接指出同步或指令瓶颈。
>
> 所以“memory side”只是初步方向。最终回答要落到可观测的硬件层级或依赖链，并用针对性改动与同条件时间验证，而不是用一个百分比给整个 Kernel 贴标签。

**可能追问：barrier 或整数指令受限，也应该笼统叫 memory-bound 吗？**

> 最好直接说具体限制。DRAM 未饱和而 barrier 或整数管线很忙，可能是同步或指令吞吐主导，不能只因代码在搬数据就统一称为 memory-bound。Roofline 提供候选方向，动态证据负责区分真正瓶颈。

---

## 性能诊断与场景追问

### 111. context 从 4096 改成 8192，仍设 32 个 split，V9 能直接运行吗？

> 当前 V9 不能直接正确处理 L=8192、32 splits。它会得到每 split 256 token，而 device guard 要求 split_end−split_start 等于 128 且起点按 16 对齐；不满足就返回，对应 mid_o 不会被写出。调用者若继续执行 Stage2，可能读取未初始化或旧内容。
>
> 一种改造是采用 64 splits 维持每 split 128 token，但当前 NUM_SPLITS=32 是 CUDA 编译期常量，grid、mid_o shape、地址公式和 Stage2 都依赖它，所以必须一起修改或模板化。另一种是保留 32 splits，扩展单 CTA 的 tile 循环处理 256 token。
>
> 两者的并行度、Query 重读、中间输出和归并成本不同，需要以 Full 时间选择。不能只删除 guard 或改一个 Python 参数，也不能把 4096 的成绩简单线性外推为新成绩。

**可能追问：把 Python 的 split 参数改成 64 就能支持 L=8192 吗？**

> 不够。当前 CUDA 的 NUM_SPLITS=32 是编译期常量，Stage1 的形状检查、偏移与 grid、Stage2 的归并组织都依赖它。采用 64 splits 必须同步修改或模板化两阶段，并重新验证数值与完整时间。

---

### 112. batch 从 64 降到 1，性能会怎样？

> 固定其他参数时 CTA 从 16,384 降到
>
> $$
> 1\times8\times32=256.
> $$
>
> 4090 有 128 个 SM，平均只有约 2 CTA/SM 的总工作量，Kernel 很快进入并行度和 launch-latency 敏感区。吞吐时间不会简单按 64 倍缩短；如果每个 CTA 又受寄存器、 Shared Memory 或 latency 限制，硬件更难用其他 CTA 隐藏延迟。
>
> 优化方向包括增加 split 并行、让一个 CTA 的 warp 更充分、批量合并请求，或为低 batch 单独设计 persistent/更细粒度映射。但增加 split 会放大 Stage2 和中间结果成本，所以要针对延迟目标调参。
>
> 这些是改造和调优方向。当前固定 32 splits 与每 split 128 token 的 CUDA 契约，不支持不经修改就任意增加 split；也不能保证 persistent 一定适合低 batch。

**可能追问：小 batch 是否应该改成 persistent Kernel？**

> 它只是候选方案。当前 B=1 的总工作量本来就少，persistent 不能凭空增加任务，也可能带来调度成本。应先比较 launch 开销、CTA 粒度和延迟隐藏是否不足，再决定细分 split、Graph 或专门的低 batch 路径。

---

### 113. `num_splits` 应该怎么选？越多越好吗？

> split 数是在并行度和额外成本之间取舍。更多 split 让 CTA 更多、单 CTA 的序列循环更短，可能帮助低 batch 或长 context 的延迟隐藏；但 Query 重读、初始化、partial output 写出和 Stage2 归并都会增加，过短的 CTA 也可能效率更低。
>
> 本项目的中间状态统一放在 mid_o=[B,Hq,S,D+1]，LSE 就在最后一维，没有另一份独立 mid_lse tensor。S 增大时这整个缓冲线性增长，选择参数不能只看 Stage1。
>
> 我会按 batch、context、Hkv、D 和目标延迟建立候选配置，独立测 Stage1、Stage2 和 Full。当前 CUDA 的 32 splits 是固定编译契约，扩展搜索前必须先让两阶段支持候选形状，再谈 autotune。

**可能追问：调 split 时应该优化吞吐还是单请求延迟？**

> 取决于目标场景。在线低 batch 更关注单步延迟，高 batch 服务还要关注总体吞吐和显存。无论目标如何，都应以真实 Full 路径评估，避免通过增加中间结果和归并开销换来漂亮的 Stage1 单项成绩。

---

### 114. GQA ratio 从 4 改成 8 或 2，会怎样？

> 对 V8/V9 的 `m16n8k16` 映射：
>
> - `GQA=8` 可以填满 N=8 的 Q-head 槽位，理论有效槽位从 50% 到 100%；但 Q、
>   score、softmax 和 output state 翻倍，register/shared 压力也上升；
> - `GQA=2` 只有 2/8 槽有效，slot utilization 降到 25%，Tensor 浪费更严重；
> - 直接把不同 KV head 的 Q group 拼到同一个 MMA 不正确，因为各组必须乘不同 K，
>   后续也对应不同 V。
>
> 因此 GQA=8 不保证恰好 2 倍快，GQA=2 也可能需要改用 SIMT、不同 MMA shape 或一次处理多个独立 tile 的布局。
>
> 这是新映射的潜在槽位比例，不代表当前源码只改 GQA 常量就能支持。Query 加载、lane class、概率与 alpha、输出写回，以及两阶段 head 数检查，都要随新形状一起调整。

**可能追问：把 GQA 常量改成 8 就能填满 V9 的 N=8 吗？**

> 不能只改一个常量。当前 warp 0 的 lane class、概率写入、alpha 数组和最终写回都有四个 Query 的硬编码映射，Hq/Hkv 检查也要匹配。填满 N 维是潜在设计收益，需要先完成整个状态映射的重构。

---

### 115. `head_dim` 改成 64 或 256，对算术强度和实现有什么影响？

> 仅按压缩 cache 估算，slot 字节为 $D+6$，useful FLOPs 为 $4GD$，其中 $G=4$：
>
> | D | cache-only AI |
> |---:|---:|
> | 64 | $1024/70=14.63$ FLOP/B |
> | 128 | $2048/134=15.28$ FLOP/B |
> | 256 | $4096/262=15.63$ FLOP/B |
>
> 维度增大后 metadata 占比降低，所以 AI 略增并趋近 16 FLOP/B，但仍远低于 4090 Tensor ridge point。代码层面当前 V4-V9 对 D=128 有硬编码假设；D=64 会改变 K 循环和线程利用，D=256 会增加 MMA 数、寄存器 output state 和 Shared Memory，必须重新设计并验证，不能只改常量。

**可能追问：D 改变后 codebook 需要跟着改变吗？**

> 需要，理论坐标方差为 1/D，get_centroids 的缓存键也包含 D。除了 MMA 循环、shared 数组和输出维度，Store/Decode 使用的 centroid 与 midpoint 必须同时更新，不能复用 D=128 的旧缓存数据。

---

### 116. block size 不是 16，或者 block table 很随机，会发生什么？

> 当前 block size=16 与 tile size=16 匹配，每个对齐 tile 正好属于一个物理页，所以只需查一次 block table，并让 lane 0–15 直接对应页内位置。metadata 的地址和 payload stride 都基于这一布局。
>
> 改成其他 block size 后，要重新计算页号、页内 offset、block stride 和 metadata 偏移。tile 可能跨页，但不是必然：例如 block size=32 且 tile 起点对齐时，一个 16-token tile 仍可完整落在页内，只是不能继续把页内位置固定解释成 lane。
>
> 随机 block table 主要改变物理页之间的局部性和缓存行为，页内连续 word 访问仍可合并。现有连续页结果不足以预测随机页性能，应新增同语义随机映射对照，再看 L2、DRAM 和 sector/request。

**可能追问：block size=32 时，每个 16-token tile 一定跨页吗？**

> 不一定。只要起点适当对齐，16-token tile 可以完整落在 32-token 页内，但 block 内 offset 和 metadata stride 都变了。当前源码按 block size=16 特化，仍需改地址契约；“不是 16 就一定跨页”并不成立。

---

### 117. 为什么这个 Decode Kernel 的设计不能直接用于 Prefill？

> Decode 通常每个 sequence 只有一个新 Query，QK/PV 形状接近 GEMV，KV 读取占比高，因此 KV 压缩和 split-KV 很有价值。Prefill 有大量 Query token，QK 是更规则的大矩阵乘，计算复用和 Tensor Core occupancy 更高，而且需要完整 causal mask、不同 tiling 与更复杂的 softmax 数据流。
>
> 把 decode 的 one-query、warp-per-Q、固定 split 映射直接搬到 prefill，会丢失 Q 维度复用并启动过多 CTA。Prefill 应使用 FlashAttention 类二维 Q/K tiling，再把 TurboQuant 解码融合到 K/V tile load 路径中。

**可能追问：Prefill 还需要 KV Cache 量化吗？**

> 可以需要，Prefill 生成的 K/V 要供后续 Decode 长期读取，压缩写入仍有容量价值。但 Prefill Attention 本身更强调多 Query 的矩阵复用，是否在 Prefill 计算时读取量化 K/V 是另一个设计问题。

---

### 118. 项目能否声称比未量化 FP16 FlashAttention 更快？

> 不能。当前主要公平对比是同一压缩 cache 语义下的 Triton TurboQuant baseline 与 CUDA 实现；仓库没有固定相同 workload、相同输入输出范围并经过验证的 FP16 FlashAttention baseline。
>
> 理论上 4-bit cache 显著减少 KV bytes，但引入 unpack、lookup、metadata 和可能较低的 Tensor slot utilization。最终是否快于成熟 FP16 kernel 必须新增端到端 benchmark 后回答，不能由压缩比推导性能比。

**可能追问：补 FP16 baseline 时最容易不公平的地方是什么？**

> 是否包含反量化和 rotation、是否同样执行 Stage2、输出精度是否一致，以及 page layout 和输入分布是否相同。比较应先明确两条路径各自的输入输出边界，再统一计时，不能拿单 Kernel 对完整链路。

---

### 119. `3.82x` 压缩是否证明模型精度几乎不下降？

> 不证明。压缩比只描述存储；Kernel correctness 只证明 CUDA 与参考实现对同一量化 cache 的数值一致。要证明模型质量，还需要：
>
> 1. 在真实模型上执行完整 K/V Store 与 Decode；
> 2. 测量 attention output、logit 的误差分布；
> 3. 跑 perplexity 数据集；
> 4. 跑下游生成/推理任务并与 FP16、其他 KV quant baseline 比较；
> 5. 覆盖不同 layer、context length 和 outlier 情况。
>
> 当前仓库没有完整模型 PPL 或 downstream quality 报告，因此只能陈述 kernel-level 数值验证，不能陈述无损或模型精度结论。

**可能追问：目前的量化误差实验覆盖了整个模型吗？**

> 没有。store_decode.py 保留并对比的是第一条 synthetic sequence 的原始 FP32 Attention，用于观察量化误差。它没有覆盖完整模型层、长程生成或 perplexity，因此不能由此写“模型精度无损”。

---

### 120. 向量化加载后性能没有提升，你会怎么排查？

> 先确认项目实际使用的是每线程一个 32-bit `uint32_t` aligned load，不把它误称为每线程 `uint4` 128-bit load。然后检查：
>
> - 地址是否真正 4-byte 对齐，warp request 是否合并；
> - 编译后 SASS 是否生成预期宽度 load，还是被拆分；
> - 总瓶颈是否在 DRAM，若在 lookup/barrier/MMA，加载变宽可能无收益；
> - 新实现是否增加寄存器、地址计算或 unpack 指令；
> - transaction 数、sector/request、replay 和 long-scoreboard 是否改善；
> - cache hit 已很高时，global load 宽度是否不是关键路径。
>
> 判断依据是版本间时间与 NCU counter 的共同变化，不是源代码类型名。

**可能追问：C++ 写了 uint32，怎样确认编译器没有拆成 byte load？**

> 检查对应 SASS 中全局加载的实际宽度与地址路径，并结合 ptxas 资源、local load/store 判断是否引入 spill。源码类型只表达意图；机器指令和最终时间共同证明向量化是否按预期落地。

---

### 121. 是否应该继续用 `cp.async` 和双缓冲优化？

> 它可能有效，但不是默认答案。`cp.async` 适合把 global-to-shared 的下一 tile 加载与当前 tile 计算重叠；需要有足够连续计算窗口、规则拷贝以及可承受的双份 Shared Memory。TurboQuant 还有 nibble unpack、centroid lookup 和 metadata scaling，并非所有处理都能由纯异步 copy 覆盖。
>
> 实施前先确认 long-scoreboard/global-memory latency 是主要 stall，并估算双缓冲是否降低 occupancy。实现后比较 async pipeline active、barrier、register/shared 用量和 Full 时间。若主要瓶颈是 MIO、lookup dependency 或低 batch 并行度，`cp.async` 可能只增加复杂度。

**可能追问：双缓冲需要保护哪两个方向的数据依赖？**

> 既要保证消费者读取前，下一块数据已经搬运和解码完成；也要保证生产者覆盖缓冲前，上一轮消费者已经用完。异步 copy 的完成等待与跨 warp 的 shared 生命周期都要处理，不能只加一个 wait 就认为流水安全。

---

### 122. 为什么不把 Stage2 也融合进同一个 Kernel？

> Stage1 的不同 split 由不同 CTA 计算。Stage2 必须等同一 `(batch, q_head)` 的全部 split 写完 partial output/LSE 后才能稳定归并，而普通 CUDA kernel 内没有跨 CTA 的 grid-wide barrier。采用最后完成 CTA 归并，需要正确设计计数器和内存发布顺序；它不必然死锁，但错误地让未完成 CTA 自旋等待其他 CTA 可能阻塞调度。cooperative grid 同步则要求相应的启动与驻留条件。
>
> 当前两次 launch 是清晰且可靠的全局同步边界。单独 Stage2 中位时间与 V7 Full 中位时间的比值约为 1.33%，这只是近似占比参考，优化优先级低于 Stage1。除非 profiler 证明 launch 对小 workload 占比很高，否则不值得以复杂持久化 Kernel 换取很小收益。

**可能追问：为什么普通 CTA barrier 不能完成跨 split 合并？**

> 不同 split 是不同 CTA，__syncthreads 只能同步当前 CTA。跨 CTA 合并要通过第二次 launch、受约束的 cooperative grid synchronization 或其他严格设计。当前两阶段方案让 Kernel 边界提供清晰的全局执行顺序。

---

### 123. V9 只有 Stage1，怎样得到公平的 V9 Full Decode 结果？

> 需要正式把 V9 Stage1 接入与 V7 相同的 Stage2 和 benchmark harness：
>
> 1. 确认 V9 输出的 `mid_o`（含最后一维 LSE） layout、dtype 和语义与 Stage2 契约一致；
> 2. 编写/导出 V9 Full launcher，而不是手工相加两个独立历史数字；
> 3. 对 compressed reference 验证 final output 和 LSE；
> 4. 与 Triton Full 使用同一输入、预分配、warmup、轮换顺序和统计方法；
> 5. 同时报告 Stage1、Stage2、Full，避免把 `0.484516 ms` 称为完整 decode。
>
> 在这条链路完成前，简历保留 V7 `0.664842 ms / 1.66x` 作为完整结果最严谨。

**可能追问：可以先给出 V9 Full 的预估值写到简历吗？**

> 不应把预估写成实测。可以在计划中说明预期下降，但简历中的精确毫秒和加速比应来自已验证、已计时的 runner。目前可引用的完整链路成绩仍是 V7 Full，V9 只引用 Stage1 成绩。

---

### 124. 要把当前研究 Kernel 接进 production vLLM，还缺什么？

> 我会先补完整的输入和 dispatch 契约：当前 Hq/D、cache 外观 shape、dtype 和 mid_o 有检查，但 unsupported seq_len 仍可能在 device guard 中静默返回；还应处理各 tensor 的设备一致性、合法维度、页表范围、tail 和不支持形状的明确 fallback。
>
> 系统集成还包括缓存的真实字节布局、slot_mapping/block_table、有效长度与写入时序、框架 backend 注册，以及 continuous batching、Graph capture 和多设备场景。上传的 vllm 目录只是相关源文件快照，不能等同于完成了整套框架验证。
>
> 当前 CUDA 已使用 CUDAGuard 和 getCurrentCUDAStream，不能把 stream 支持说成从零缺失。接下来应将 V9 纳入 Full 和 Store 验证，补真实模型质量、端到端时延和多 workload 性能，才有依据讨论生产替换。

**可能追问：当前 CUDA 已经处理了 PyTorch current stream 吗？**

> 是的，源码使用 CUDAGuard 和 getCurrentCUDAStream 发射 Kernel。生产集成仍需核对多 tensor 同设备、维度和长度合法性、Graph capture 以及缓存生命周期，但不能把已经实现的 current-stream 支持说成完全缺失。

---

### 125. 如果把 Kernel 从 4090 移到 3090，需要重点重做什么？

> 3090 是 Ampere `sm_86`，SM 数、Tensor Core 代际、时钟、L2、内存子系统和资源调度与 Ada 不同。首先重新编译 `sm_86` 并确认 PTX/SASS 指令可用；然后重新测：
>
> - ptxas registers、Shared Memory、occupancy 和 active CTA；
> - MMA、load、barrier 的实际吞吐和 stall；
> - split 数、threads/CTA、tile 数等参数；
> - L2/DRAM 行为以及完整 benchmark；
> - 数值一致性。
>
> 不能只把架构 flag 从 89 改成 86，也不能把项目中旧 3090 NCU 报告当作当前 V9 在 4090 上的证据。历史报告可以提供假设，最终结论必须按 GPU 和代码版本重采。

**可能追问：WMMA 版本和原生 PTX 版本的迁移检查有什么不同？**

> WMMA 直接写 fragment 的版本要重新验证内部映射 probe；原生 PTX 版本先依据 ISA 确认指令与 operand 契约，再验证编译生成和输出。二者都要重测资源与时间，但依赖的布局证据不同。

---

## 代码证据与学习路线

### 126. 如何用两分钟讲清 V1 到 V9 的优化演进？

> 我会按三段讲。第一段是建立正确的 CUDA 基线并调整协作：V1 用 CTA 协作复用 K/V，V2 采用 warp-per-Q，以重复解码换减少 CTA 通信；V3 再以 16-token tile 共享反量化，并用 WMMA 完成 QK/PV。V1/V2 本身已有 online softmax，V3 的增量是 tiled Tensor 组织。
>
> 第二段是固定 workload 下的数据路径优化：V4 利用每 split 128 token 与页对齐，展开循环、减少地址和查表开销；V5 从 fragment 直接写有效输出，V6 用 uint32 加载与 half2 写入，V7 合并 tile 边界同步。
>
> 第三段是 GQA 映射与状态通信：V8 用原生 m16n8k16 组内转置，将有效槽位从 25% 提到 50%；V9 把 score/softmax 状态留在 warp 0 寄存器，删除 qk_s 往返。Stage1 从 2.074204 ms 到 0.484516 ms，约 4.28 倍；经过记录的完整链路成绩另外是 V7 Full 的 1.66 倍。

**可能追问：V1 到 V9 哪一步属于算法变化，哪一步属于数据路径优化？**

> V2/V3 主要重构 CTA/warp 协作与 tiled Tensor 计算，V8 改变合法的矩阵映射；V5/V6/V7/V9 更多优化写回、解码访存和同步。各版保持 Attention 语义，而且 V1/V2 已有 online softmax，不能把 V3 说成首次引入它。

---

### 127. “WMMA fragment 寄存器直写”解决了什么问题？

> 普通写法可能先把 accumulator fragment `store_matrix_sync` 到 Shared Memory，同步后再由线程标量读取、转置或重排，形成 `register -> shared -> register` 往返。 V5 利用 fragment 元素在线程 lane 中的实际映射，直接从 accumulator fragment 提取目标值写入后续状态，减少 Shared Memory traffic、同步和标量 load。
>
> 风险是 fragment 内部布局并非 CUDA C++ API 保证的跨架构稳定 ABI，因此项目用`wmma_fragment_probe.cu` 探测映射，并将优化绑定到目标架构。迁移架构或编译器时必须重新验证，不应把经验映射当成通用标准。

**可能追问：直接写回主要消除了计算还是搬运？**

> 主要消除完整 accumulator 写到 shared 后再读有效行的搬运，以及相关同步和地址操作。QK/PV 所需数学工作没有因此减少；这说明优化 Attention 还要关注矩阵乘之外的数据流。

---

### 128. V6 的“向量化访存”准确来说是多少位？

> 源码中每个线程通过 `reinterpret_cast<const uint32_t *>` 各加载一个 32-bit word，一次得到 8 个 4-bit index；warp 层面连续线程协作形成合并访问。它不是每线程加载 `uint4` 的 128-bit vector load。
>
> 面试中应说“每线程 32-bit 对齐加载，warp 合并读取，再使用位运算和 `half2`并行重建”。简历其他项目若有 128-bit vectorized load，也不能移植成这个 Kernel 的实现细节。
>
> 在同一次循环迭代中分别加载 K word 与 V word，合计 8 B，但它们是两个相隔 64 B 的地址，不能因此说成一条 64-bit load。每个 word 再对应四次 half2 写，重建八个坐标。

**可能追问：每线程分别读 K 和 V 两个 uint32，是否可称为一次 64-bit load？**

> 不能。源码是两次独立 32-bit 加载，地址分别落在 K 和 V 区域，彼此相隔 64 B。可以说一次迭代每线程读取 8 B packed 数据，但不能把总字节数等同于一条 64-bit 向量加载指令。

---

### 129. V7 消减 barrier 时，如何证明没有引入 race condition？

> 我会直接围绕 V7 被删掉的同步解释。上一轮 PV 使用 p_s/v_s，输出留在 warp 的 accumulator registers；下一轮 warp 0 首先更新独立的 data_base、norm、scale、zero 数组，这不会覆盖上一轮 PV 正在消费的数据。随后原有 metadata 发布 barrier 等待所有 warp 汇合，之后才允许写下一轮 K/V tile。
>
> 因此该 barrier 同时承担“上一轮消费结束”和“新 metadata 已就绪”两个条件，tile 末尾的另一个等待点可以合并。安全性来自具体 shared 数组的读写生命周期，不是凭“warp 大致同时运行”推测。
>
> 验证时应配合 racecheck、随机/结构化输入和完整 output/LSE 对照；上传包没有给出可以据以宣称所有这类动态测试都通过的报告。静态推导与动态验证是不同层次，使用 shuffle/syncwarp 也要遵守参与线程与掩码语义。

**可能追问：源码推导无 race，是否说明已经跑过 racecheck？**

> 不是。静态依赖分析与动态工具结果是不同证据。当前压缩包没有可据以宣称 V7/V9 racecheck 全部通过的完整报告；回答时可以解释安全理由，并把 racecheck、多输入回归作为后续验证要求。

---

### 130. 项目到底如何借鉴 FlashInfer？

> 借鉴的是 decode attention 的数据流思想：让 QK score、online softmax 的`m/l` 状态和 output accumulator 尽可能 register-resident，以减少 Shared Memory 往返和 CTA barrier。V9 在这一方向上把 score/state 留在寄存器中。
>
> 项目没有链接或调用 FlashInfer runtime，也不是复制某个 FlashInfer Kernel。准确表述是“参考 FlashInfer 的寄存器常驻 decode 状态设计后，在本项目固定 TurboQuant 数据布局上自行实现”。若面试官要求代码证据，应指向 `cuda/tq4_cuda_v9.cu`及其与 V8 的 diff。

**可能追问：为什么 V9 仍要把概率写到 shared？**

> QK 与 softmax 由 warp 0 处理，而 PV 的不同输出维度由四个 warp 承担，概率需要跨 warp 共享。保留这一次必要通信，消除不必要的 score 中间落地，是根据消费者归属作出的取舍。

---

### 131. 你如何分层验证 CUDA Kernel 的正确性？

> 我会分层验证。第一层比较同一 compressed cache 上的 Stage1：完整 mid_o 的前 128 维 partial output 与最后一维 LSE都要比较，并检查 NaN/Inf。第二层把同一 mid_o 输入两种 Stage2，隔离归并实现；再串起各自 Stage1+Stage2，比较最终 output/LSE。
>
> 第三层使用上传的未修改 SoA Store，从原始 FP16 K/V 真正生成 cache，再直接交给 CUDA 与 Triton，验证 packing、norm correction、metadata 和页地址契约。当前这条完整验证使用 V7，V9 仍需要接入后重跑。
>
> 量化误差另外对照原始 FP32 Attention，不能与 CUDA 的额外数值误差混为一谈。现有记录主要是固定 synthetic 输入，还应补多种子、结构化数据、极端 metadata 和合法边界。Triton V2 的历史问题正说明，LSE 对齐不能替代 PV/output 验证。

**可能追问：只比较 Stage2 最终输出，会遗漏什么问题？**

> 不同 split 的误差可能部分抵消或被较小权重掩盖，最终输出看起来正常。直接比较完整 mid_o 能定位到 batch、head、split 和维度，随后再单测 Stage2，排错范围更清楚。

---

### 132. synthetic benchmark 与 Store→Decode 验证分别证明什么？

> synthetic benchmark 直接构造满足格式的压缩 cache，适合控制输入、隔离 Decode 并公平比较不同版本。为比较 AoS/SoA，它在计时外做无损布局转换；这能说明同一逻辑数据的 Decode 一致性，但构造器与读取器也可能共同误解真实 Store 契约。
>
> Store→Decode 验证补上这一层：validation/store_decode.py 从原始 FP16 K/V 出发，调用未修改的上游 SoA Store，完成真实 packing 和 metadata 写入，然后把同一个 cache tensor 直接交给 Triton 与 CUDA V7 Full，中间没有 AoS/SoA 转换或 byte rearrangement。
>
> 仓库记录中这条路径的 output 最大差约 5.06e-6，说明测试范围内两者对 Store 字节语义一致。它仍采用合成 Q/K/V，不等于完整模型、ragged sequence、任意页映射和服务调度都已经验证。

**可能追问：Store→Decode 中有没有 AoS/SoA 转换？**

> 这条验证路径没有。它把未修改的上游 SoA Store 写出的 cache 直接传给两种 Decode。AoS/SoA 无损转换发生在另一套 synthetic benchmark 中，用于比较布局，不应混写到 Store 兼容性验证。

---

### 133. benchmark 如何保证公平？还存在哪些误差来源？

> 公平性的基础是同一逻辑输入、同一输出语义和明确的计时范围。项目在计时外准备 cache、AoS/SoA 转换、参数与输出缓冲，先做 correctness，再默认 warmup 20 次；每轮 CUDA Event 包围 100 次执行，取平均单次时间，五轮轮换 runner 顺序，最后报告轮均值的中位数。
>
> Stage1 与 Full 分开测量，Full 在同一个 runner 内串联两次 launch。当前脚本主要在正式性能循环前验证结果，不是每轮计时前后都重复一次 correctness。
>
> 仍可能有 boost、温度、后台任务、缓存热度、编译配置和输入分布带来的误差。更严格复现可以保存全部轮次、记录设备与软件环境，在独立进程重复，并检查结果是否稳定；仅选择最快一次或混用不同实验的时间不够可靠。

**可能追问：现有脚本是在 correctness 前后都反复验证吗？**

> 主要是在正式计时前计算并比较各实现结果，然后进行 warmup 和多轮计时。不能笼统说成每轮前后都有检查。若担心状态污染，可以补计时后的验证，但应把已有行为与新增验证建议分开。

---

### 134. 简历中的每条表述，分别能在仓库哪里找到证据？

> 我会把简历中的每个技术点落实到可检查的文件，尤其区分源码事实、仓库记录和后续计划。
>
> | 表述 | 上传仓库中的主要依据 |
> |---|---|
> | 4bit_nc 与 centroid | reference/config.py、reference/centroids.py |
> | Store packing 与 corrected norm | reference/soa_store.py |
> | SoA 字节布局与固定 workload | baseline/common.py、CUDA load_meta 和 data_base |
> | V1–V3 线程映射与 WMMA | cuda/tq4_cuda_v1.cu 至 tq4_cuda_v3.cu |
> | V4–V7 特化、直接写回、向量化、同步 | cuda/tq4_cuda_stage1_template.cuh 与对应开关文件 |
> | WMMA 内部映射探测 | cuda/wmma_fragment_probe.cu |
> | V8/V9 原生 MMA 与状态融合 | cuda/tq4_cuda_v8.cu、cuda/tq4_cuda_v9.cu |
> | Stage2 与 Full | cuda/tq4_cuda_v7.cu、benchmarks/full_decode.py |
> | 强 Triton baseline 的 V-layout 修复 | baseline/triton_v2.py |
> | Store 字节兼容验证 | validation/soa_store.py、validation/store_decode.py |
> | 性能、误差和静态资源记录 | README.md、README_CN.md、cuda/README.md、benchmarks/README.md |
> | 历史 NCU 分析 | results/v2_stage1.md 与 ncu-rep |
>
> 其中 reference 与 vllm 是上游快照；个人设计、实现和分析按实际贡献说明。README 的毫秒与资源数字是历史实验记录，阅读这些文件本身不等于已经重新运行了 benchmark。

**可能追问：只有 README 的数字，算不算已经现场复现？**

> 不算。README 是上传项目保存的实验记录；现场复现还需要匹配环境、编译运行并保存结果。整理面试材料时我会明确数字来源，不把阅读源码和历史记录写成刚刚在 GPU 上重新测量。

---

### 135. 如果要在七天内把这部分准备到能扛住追问，怎么学？

> 如果只有七天，我会围绕“能解释、能手推、能定位证据”来安排。前两天吃透输入形状、134 B slot、norm/scale/zero、rotation 和 codebook，能从 token/head 算出实际 byte address。第三天手推 online softmax 与 Stage2 的 LSE 合并，知道 mid_o 每一维的含义。
>
> 第四、五天沿源码看 V1–V9，重点比较 V3 的 tiled WMMA、V5 的 fragment 写回、V6 的 packed load、V7 的 barrier 生命周期，以及 V8/V9 的 Query 列映射。每一步都回答“改了什么、代价是什么、为什么正确”。
>
> 第六天整理计时范围、三个加速比、量化误差与 Kernel 误差，以及历史 NCU 的使用边界；有目标 GPU 时按仓库 runner 复现。第七天模拟变形题：换 batch/context/GQA、常量 V、随机页和不支持形状，再用输入契约与依赖关系推导答案。

**可能追问：时间不够时先准备哪三段？**

> 先讲清 134 B cache 契约，再手推一轮 tile 的解包、QK、online softmax 和 PV，最后说明三种加速比与验证层次。这三段覆盖输入、实现和证据，比孤立背诵所有版本数字更能应对追问。

---

### 136. K norm 按什么粒度计算和保存？是每个 `num_head` 共用一个吗？

> K norm 的逻辑粒度是每个 token、每个 KV head 的一根 D 维 Key 向量。对 K[b,t,h,:]，沿 head_dim 计算二范数 s[b,t,h]=√Σ_d K[b,t,h,d]²，因此不同 token、不同 KV head 都有自己的值。
>
> 当前 Store 把输入展平为 [num_tokens×Hkv,D]，逐行计算 norm；开启 nc 后再折叠 centroid-vector 的长度，保存 corrected norm γ=s/√(Σ_d c_d²+1e-16)，最终写成 FP16。缓存字段名叫 norm，但在 4bit_nc 下实际不是未经修正的原始范数。
>
> 同一个 token/head 的 K 被四个 GQA Query 复用，所以这四个 Query 也使用同一份 K metadata；它们各自的 score、softmax 和 output 仍独立。metadata 区的排列是 [kv_head,field,position]，并不是“每个 head 只有一个 norm 服务整条序列”。

**可能追问：四个 Query 共享一个 KV head，也共享它的 K norm 吗？**

> 对于同一个 token 和 KV head，是共享同一个 K corrected norm，因为读取的是同一根 Key 向量。但四个 Query 有独立的 QK score 与 softmax 状态；输入 metadata 共享不表示输出状态共享。

---

## 简历原子级实现追问

### 137. K norm 为什么沿 `head_dim` 计算，而不是沿 token 或 head 维计算？

> 因为 TurboQuant 的基本量化对象是一个 Key head vector $K_{t,h,:}\in\mathbb{R}^D$。它需要先把这个向量归一化到单位球面，再旋转和逐坐标量化，因此归约维度必须是最后一维 $D$：
>
> $$
> s_{t,h}=\left\lVert K_{t,h,:}\right\rVert_2
> =\sqrt{\sum_{d=0}^{D-1}K_{t,h,d}^2}.
> $$
>
> 沿 token 维求 norm 会把不同历史位置混合，沿 head 维求 norm 会把不同 KV head 混合，两者都会改变原始 attention 中每个 Key 的方向和幅值语义。源码中`key.float().reshape(NH, D)` 后执行 `norm(dim=1)`，就是对每个 `(token, kv_head)`独立沿 $D$ 归约。

**可能追问：沿 token 维求 norm 会破坏什么？**

> 它会把不同历史位置的幅值耦合起来，无法用一个标量恢复每根 Key 的长度。Attention 需要比较不同 token 的 logits，这种错误归一化会改变它们的相对幅值，而不只是改变一个可抵消的公共常数。

---

### 138. 固定 workload 一共有多少个 K norm？metadata 占多少显存？

> 若按完整逻辑 cache 的 $B=64,L=4096,H_{kv}=8$ 计算，K 向量数量是：
>
> $$
> 64\times4096\times8=2{,}097{,}152.
> $$
>
> 每个 K corrected norm 是 FP16 2 B，因此 K norm 共 `4 MiB`。V scale 和 V zero 也各有同样数量，各占 `4 MiB`，三种 metadata 合计 `12 MiB`。
>
> K/V packed payload 每个 `(token,kv_head)` 是 128 B，总计 `256 MiB`；所以完整 compressed cache 是 `268 MiB`。这也可以从$2{,}097{,}152\times134\ \text{B}=268\ \text{MiB}$ 复算。实际 paged cache 按已分配 physical block 容量占用，不一定恰好等于当前有效 token 数。

**可能追问：有效 token 少于已分配 slot，显存是否立即等比例减少？**

> 不一定。paged cache 按 physical block 容量分配，未使用的 slot 和 allocator 预留仍可能占显存。268 MiB 是当前给定完整容量的计算结果，实际服务要区分有效数据量、已分配 cache 和进程峰值。

---

### 139. 为什么不为整个 KV head 只保存一个 norm？

> 同一个 KV head 在不同 token 位置产生的 Key 向量范数不同。若整条序列共用一个 norm，相当于强迫所有 $K_{t,h,:}$ 使用相同幅值，会直接扭曲不同 token 的 QK logit。per-token/per-head norm 保留了每个 Key vector 的幅值，只把它的归一化方向交给 centroid index 表示。
>
> 更细到 per-coordinate 保存 scale 又没有必要：K 每个 coordinate 已由非均匀 centroid 表示，额外 128 个 scale 会使 metadata 大幅膨胀。一个向量一个 norm 是数学语义和压缩开销之间的设计点。

**可能追问：per-head scale 在权重量化里常见，为什么这里不能直接类比？**

> 量化对象不同。这里每个历史 token 的 K 都是一根独立动态向量，幅值直接影响其 Attention logit；权重量化的通道分组是另一种统计和复用结构。粒度选择应服从当前张量语义，不能只照搬术语。

---

### 140. corrected norm 是在 Store 还是 Decode 计算？为什么？

> 在 Store 阶段计算。Store 已经拥有当前向量的全部 128 个 quantized index，可以查 centroid 并归约得到 $\lVert c_{t,h}\rVert_2$，然后一次性保存：
>
> $$
> \gamma_{t,h}=\frac{\lVert K_{t,h,:}\rVert_2}
> {\lVert c_{t,h,:}\rVert_2}.
> $$
>
> Decode 热路径只执行 `centroid[index] * gamma`。如果只保存原始 norm，Decode 每次读取历史 K 都要重新计算 128 个 centroid 平方和、开方和除法；同一个历史 token 会在每一步生成中反复付费。把修正折叠到 Store 是典型的“一次写入计算，多次 Decode 复用”。
>
> 源码实际使用 1/√(Σc²+1e-16) 处理数值边界，gamma 最后转成 FP16。因此范数保持的推导是理想表达，真实重建仍包含稳定项和 metadata 舍入。

**可能追问：norm correction 的公式与代码是否完全没有稳定项？**

> 代码计算 centroid 平方和后使用 1/√(sum+1e-16)，再乘原始 norm。面试推导可写 γ=||K||/||c||，但解释零值与极小值时应补充稳定项，以及最终 FP16 metadata 舍入的影响。

---

### 141. Lloyd-Max codebook 是每个 token、每个 head、每层各一套吗？

> 从数学参数看都不是。centroid 只由 `head_dim=d` 和量化位数 `bits` 决定；对当前`d=128,bits=4`，所有 token、KV head 使用同一组 16 个数。它不是从某一层真实 KV 数据校准出来的，所以数值上也不需要 per-layer codebook。
>
> 实现上 `get_centroids(d,bits)` 在 CPU 侧有缓存，但 vLLM 的 `_ensure_on_device`会把相同数值的 centroid tensor 挂到各 attention layer 上。要区分“数学上是否每层不同”和“工程上是否每层持有一个 device tensor”。本 CUDA benchmark 接收一个 `[16]` FP32 centroid tensor，整个 launch 共享。

**可能追问：每层各持有一个相同 centroid tensor，会改变算法吗？**

> 不会，数值相同就表示相同 codebook，只是设备参数的管理方式不同。数学共享不一定等于只有一个物理 tensor；判断语义应看生成参数和数值，判断显存则看实际对象是否复用。

---

### 142. centroid 和 midpoint 分别在 Store、Decode 的哪个阶段使用？

> 两者不能混为一张表：
>
> - 15 个 midpoint 是相邻 centroid 的 decision boundary；Store bucketize 时使用，
>   将旋转坐标映射成 `0..15` index；
> - 16 个 centroid 是重建值；Store 计算 norm correction 时会查一次，Decode 对
>   每个 4-bit index 做 reconstruction 时也会查；
> - cache 只保存 index 和 corrected norm，不保存 midpoint 或逐 token centroid。
>
> 因此 Decode 不需要重新做二分查找。它已经拿到离散 index，只需直接 lookup。

**可能追问：Store 查 centroid 后，为什么还要保留 midpoint 表？**

> 两者服务不同步骤。midpoint 把连续输入映射为 index；centroid 在得到 index 后重建量化向量，用于 norm correction。Decode 只有重建需求，所以只需要 centroid，不再需要边界搜索。

---

### 143. 4-bit K/V 的两个 index 在一个 byte 中如何排列？

> Store 将相邻 coordinate 两两分组：
>
> $$
> \text{byte}_j=(index_{2j}\ \&\ 0xF)
> \;|\;(index_{2j+1}\ll4).
> $$
>
> 所以偶数维在低 4 bit，奇数维在高 4 bit。Decode 对应执行：
>
> ```cpp
> lo = packed & 15;
> hi = packed >> 4;
> ```
>
> `D=128` 形成 64 B K index，V 同样形成 64 B。这里的“高低 nibble 顺序”必须与 Store 完全一致；顺序颠倒可能仍产生有限数值，却会悄悄置换相邻维度。

**可能追问：怎么用一个 byte 快速检查 nibble 顺序？**

> 例如 byte=0xA3，应解为偶数维 index=3、奇数维 index=10，再按各自 centroid 或 V affine 参数重建。使用这种可手算输入比只看随机误差更容易发现相邻维度对调。

---

### 144. V 的 scale 和 zero 按什么粒度计算？公式是什么？

> V 的参数按每个 token、每个 KV head 的整根 D 维向量计算。Store 先求 v_min 和 v_max，再令 scale=max((v_max−v_min)/15,1e-8)，zero=v_min。它对非负的归一化坐标 (v−v_min)/scale 加 0.5 后转整数，再截断到 0..15。
>
> 这相当于当前输入范围内的就近量化，但恰好半整数时的规则要按源码复现，不能默认与所有语言的 round 一致。两个 index 挤进一个 byte，128 维占 64 B；scale 与 v_min 最终各存一个 FP16 标量。
>
> Decode 使用 v_hat=index×scale+zero。这里 zero 是浮点最小值，不是整数 zero-point；也不需要再对 V 做 K 的 centroid lookup 或 Query matching rotation。

**可能追问：Store 的 round 与 Python round 是否总是一致？**

> 不能直接假设一致。源码对非负归一化值采用加 0.5 后转整数，再截断到 0..15；Python 默认 round 在恰好半整数时可能采用不同的舍入规则。写逐字节参考时应复现源码的具体规则。

---

### 145. 如果一个 V 向量的所有元素都相等，scale 会不会为 0？

> 要区分量化计算时的 scale 和 cache 保存的 scale。常量 V 的范围为零，Store 在 FP32 中把 scale 限制为至少 1e-8，因此计算 index 时不会除零；因为所有 v−v_min 都为零，所有 index 都是零。
>
> scale 转成 FP16 后，1e-8 可能下溢成零，但 Decode 仍得到 index×scale+v_min=0×0+v_min，能够恢复存储精度下的常量。这个结论不代表任意小范围向量都完全无误差，FP16 参数舍入仍需要考虑。
>
> 当前 validation/store_decode.py 的辅助检查要求 V scale 严格大于零，适用于脚本的随机输入，却会拒绝这类 scale 下溢的合法常量边界。扩展验证时应把这个检查与实际重建语义分开，不能把未覆盖边界说成已测试通过。

**可能追问：为什么常量 V 合法，现有验证却可能拒绝？**

> 常量 V 量化后的 index 为零，FP16 scale 即使下溢为零也可正确重建常量。但 check_store_metadata 要求 scale 严格大于零，适合当前随机输入，却不覆盖该边界。扩展测试时应调整检查，不能把它当成量化数学失败。

---

### 146. K norm、V scale 和 V zero 为什么保存成 FP16？

> 每个 `(token,kv_head)` 只有三个标量，但长序列下数量仍很大。FP16 将 metadata 控制为 6 B，使 slot 保持 134 B；如果改成 FP32，slot 会变成 140 B，压缩比从$512/134=3.82x$ 降到 $512/140=3.66x$，metadata 流量也翻倍。
>
> 代价是 corrected norm 和 V affine 参数有 FP16 舍入误差。项目选择 FP16 是容量、带宽和精度的工程折中，是否可改成 BF16/FP32 应通过 attention output 和模型质量测试决定，而不是假设 metadata 精度不重要。

**可能追问：FP16 metadata 除了舍入还可能有什么边界？**

> 很小的 scale 可能下溢，很大的 norm 或范围可能溢出成 Inf。当前随机样本不代表所有真实输入都安全，扩展支持时要检查输入范围、有限值和必要的高精度或 fallback 路径。

---

### 147. 一个 physical block 内部的 SoA cache 精确布局是什么？

> 当前固定 `block_size=16,Hkv=8,D=128`。每个 `(position,kv_head)` 的 packed data 只有 K 64 B + V 64 B，不把 metadata 夹在其中。一个 physical block 是：
>
> ```text
> data region: [16 positions, 8 KV heads, 128 data bytes]
> metadata:    [8 KV heads, 3 fields, 16 positions] × FP16
> ```
>
> 字节数分别为：
>
> $$
> 16\times8\times128=16{,}384\ \text{B},
> $$
>
> $$
> 8\times3\times16\times2=768\ \text{B},
> $$
>
> 合计 `17,152 B`，等于 $16\times8\times134$。metadata region 从`META_OFFSET=16384` 开始，三个 field 依次是 K norm、V scale、V zero。

**可能追问：metadata 中 field=1、position=5 的地址怎样算？**

> 先取 physical block 基地址，再加 16384 B 数据区偏移，接着加 ((kvh×3+1)×16+5)×2 B。末尾乘 2 是因为 metadata 为 FP16；不能把 u16 元素索引直接当成 byte offset。

---

### 148. 为什么 payload 和 metadata 要采用这种“数据区 + SoA metadata”布局？

> Decode 对一个 tile 的同一 KV head 连续读取 16 个 token。把 metadata 排成`[kv_head, field, position]` 后，warp 0 的连续 lane 可分别连续读取 16 个 norm、 16 个 scale 或 16 个 zero，便于合并访问；如果每个 2 B metadata 都夹在 128 B payload 后，字段访问会形成 134 B stride。
>
> payload 仍按 `[position,kv_head,data]` 排列，使某 token/head 的 K/V 128 B 紧邻，便于 cooperative packed load。这里所谓 SoA 主要指 metadata 字段拆开，不应笼统说成整个 cache 每一部分都是纯 field-major。

**可能追问：metadata 连续，但 payload 按 token 的 stride 很大，会不会抵消收益？**

> 要看 warp 的实际分工。当前 payload 由连续线程读取同一 token/head 的连续 packed word，metadata 则由 warp 0 的 lane 0–15 连续读取一个字段。两个区域分别匹配各自访问方式，不能仅用单一 stride 判断整个布局。

---

### 149. Store 的 `slot_mapping` 和 Decode 的 `block_table` 分别解决什么问题？

> Store 面对新 token，`slot_mapping[token]` 直接给出写入的全局 physical slot：
>
> $$
> block=slot/16,\qquad offset=slot\bmod16.
> $$
>
> Decode 从 sequence 的逻辑 token 位置出发，通过`block_table[batch, logical_block]` 找到 physical block，再加 block 内 offset。前者是“当前 token 写到哪里”，后者是“历史逻辑位置存在哪个物理页”。二者共同实现 paged KV cache，但不是同一个数组，也不能互换。

**可能追问：slot_mapping 和 block_table 不一致会怎样？**

> Store 会把新 K/V 写到一页，而 Decode 去另一页读，结果可能是旧值或其他 sequence 的数据。错误通常不会表现为简单越界，所以集成验证要同时检查写入映射、长度更新与读取页表的对应关系。

---

### 150. Stage1 的 grid 三个维度分别是什么？为什么这样分？

> grid 是 `(batch, kv_head, split)`：
>
> ```text
> blockIdx.x -> batch sequence
> blockIdx.y -> KV head
> blockIdx.z -> KV split
> ```
>
> 一个 CTA 负责该 KV head 对应的全部 4 个 Query head。这样 K/V tile 解码一次后能被 4 个 GQA Query 复用。若改成 `(batch,q_head,split)`，CTA 更多，但同一 K/V 会被四个 CTA 重复读取和反量化；若一个 CTA 负责全部 8 个 KV head，Shared Memory、寄存器和单 CTA 工作量又过大。

**可能追问：为什么不一个 CTA 处理整个 sequence 的全部 KV head？**

> 那会增加单 CTA 的 K/V tile、Query 与输出状态，也减少可并行调度的 CTA 数。当前按 KV head 和 split 划分，既让四个 GQA Query 复用 K/V，又控制资源规模，是固定 workload 的平衡点。

---

### 151. `kvh` 如何映射到 4 个 Query head？

> 当前 Qwen3-4B shape 的 GQA ratio 是：
>
> $$
> G=H_q/H_{kv}=32/8=4.
> $$
>
> 代码使用：
>
> $$
> qh=kvh\times4+q_{local},\qquad q_{local}\in\{0,1,2,3\}.
> $$
>
> 例如 `kvh=3` 对应 Q head 12、13、14、15。这依赖 head 按连续 group 排列的 layout 契约；若模型或框架采用不同 head mapping，不能继续使用该公式而不做转换。

**可能追问：Query head 的内存顺序变化后，需要改哪些地方？**

> 不仅是输入 qh 索引，输出 mid_o、最终 output 以及参考实现的 group 映射都要一致。现有公式假设同组 Query 连续排列；如果框架采用不同排列，应显式转换或传入映射，而不是只改一个加载地址。

---

### 152. 128-thread CTA 中四个 warp 的职责完全相同吗？

> 不完全相同。V8/V9 中：
>
> - 128 threads 协作加载 Q、解码 K/V；
> - warp 0 负责 QK 的 `m16n8k16` MMA；
> - V8 的四个 warp 各维护一个 Query head 的 online-softmax 标量状态；
> - V9 把四个 Query head 的 QK/softmax 状态重新映射到 warp 0 的 lane classes；
> - PV 阶段四个 warp 分别负责 output 的 32 个维度切片，共覆盖 D=128。
>
> 因此“一个 warp 对应一个 Query head”对 V2/V7 的解释较直观，但不能不加限定地套到 V9 的所有阶段。V9 的 QK/softmax 所有权与 PV output 切片所有权不同。

**可能追问：V9 四个 warp 都执行 PV，会不会重复计算同一输出？**

> 它们分别负责不同的 32 维输出切片，四个 warp 合起来覆盖 D=128。每个 warp 的 fragment 中仍同时包含 Query 列，最终按 lane/index 写到正确 head 和维度。分工是按输出维度切片，不是四份相同 PV。

---

### 153. 一个 16-token tile 的 norm/scale/zero 由谁加载，谁使用？

> warp 0 先查一次 block table；lane 0 得到 physical block 后用 shuffle 广播。随后 warp 0 的 lane 0–15 各负责一个 token position，分别加载该 `kvh` 的`k_norm[t]`、`v_scale[t]` 和 `v_zero[t]` 到 Shared Memory。
>
> CTA barrier 后，全部 128 threads 在 cooperative unpack 中使用这 16 组 metadata。同一 token 的 K norm 和 V scale/zero 会服务该 KV group 的 4 个 Query head；它们不是每个 warp、每个 Query head重复从 global memory 加载一份。

**可能追问：同一 tile 的 block table 是 16 个 lane 各查一次吗？**

> V9 中先由 warp 0 的 lane 0 查一次，再用 shuffle 广播 physical block；lane 0–15 根据各自 position 读取 norm、scale、zero。这样利用 tile 与物理页对齐，避免对同一页号重复查询。

---

### 154. 16-entry centroid lookup 在 CUDA 中怎样实现？每次都访问 global memory 吗？

> 每个 warp 开始时让 lane 0–15 各自加载一个 FP32 centroid 到寄存器`centroid_lane`。解码 nibble 后，以 index 作为 source lane 调用`__shfl_sync`，把对应 centroid 广播给需要它的 lane。
>
> 所以热循环中的 lookup 主要是 register shuffle，不是每个 coordinate 都发起一次 global gather。代价仍包括 nibble 位运算、shuffle 和数据依赖；而且每个 warp 都有自己的 16-entry 分布式寄存器表，不是整个 CTA 只有一份物理寄存器。

**可能追问：为什么用 FULL_MASK 做 centroid shuffle 是可行的？**

> 当前 decode 循环让参与 warp 的所有 lane 按相同控制路径执行 shuffle，且 index 在 0..15，源 lane 都在 mask 中。若未来加入分歧 tail 或部分 lane 提前退出，就要重新保证参与掩码和源 lane 的有效性。

---

### 155. `q_rot` 的形状是什么？Query rotation 是否在 `0.6648 ms` 内？

> Kernel 输入 `q_rot` 的逻辑形状是 `[B,Hq,D]`，当前 dtype 为 FP32。对每个 CTA，只加载当前 `kvh` 对应的 4 个 Query head，即 `[4,128]`。
>
> Query 在进入 benchmark 前已经完成与 K 相匹配的旋转；`0.664842 ms` 只测预旋转 Query 和 compressed KV 经过 Stage1 + Stage2，不包含 Query rotation。简历中的“完整 Decode Kernel 链路”必须按这个项目定义解释，不能说成完整 attention backend 或完整 token latency。

**可能追问：q_rot 是 FP32，为什么 Tensor Core operand 是 FP16？**

> FP32 是入口 tensor 和旋转结果的存储精度，CUDA 加载后把参与 QK 的 Query tile 转为 FP16。接口 dtype 与计算 operand dtype不同，这也是与 FP32/SIMT 参考出现微小数值差异的来源之一。

---

### 156. 为什么 score 要乘 `1/sqrt(128)`，代码又为什么出现 `RCP_LN2`？

> scaled dot-product attention 使用：
>
> $$
> score=QK^T/\sqrt{D}.
> $$
>
> 当前 $D=128$，所以 `ATTN_SCALE=1/sqrt(128)=0.088388...`。Kernel 用更适合 GPU 的 `exp2f` 实现指数，因此利用
>
> $$
> e^x=2^{x/\ln2},
> $$
>
> 先把 score 乘 `RCP_LN2=1/ln(2)`。最终 LSE 再用`running_m * LN2 + log(running_l)` 转回自然对数语义。`RCP_LN2` 不是另一个 attention scale，也不是量化 scale。

**可能追问：Stage2 能直接把 base-2 的 running_m 当自然对数 LSE 吗？**

> 不能。Stage1 要用 running_m×ln2+ln(running_l) 得到自然对数 LSE，再交给使用 expf/logf 的 Stage2。如果漏掉 ln2 转换，跨 split 权重会错，即使局部 softmax 看起来正常。

---

### 157. Online softmax 的 `m`、`l` 和 output accumulator 按什么粒度分配？

> 逻辑上每个 `(batch, q_head, split)` 有一套独立状态：
>
> - $m$：该 split 已处理 score 的运行最大值；
> - $l$：以 $m$ 为基准的指数和；
> - $o\in\mathbb{R}^{128}$：该 Query head 的 Value 加权累加器。
>
> 四个 GQA Query head 共享 K/V tile，但不能共享 softmax 状态，因为它们的 Query 不同，QK score 也不同。V9 只是改变状态在 warp/lane 寄存器中的物理映射，没有改变“一 Query head、一 split、一套逻辑 softmax state”的数学语义。

**可能追问：V9 lane 中有两份 m/l，是否意味着一个 head 有两套独立状态？**

> 那是 MMA fragment 和 lane class 的物理分布方式，部分值在协作 lane 间复制或分片。逻辑上仍是每个 Query head、每个 split 一套状态，最终归约与写回必须恢复这一对应，不能按寄存器数组长度推断 head 数。

---

### 158. `mid_o` 为什么最后一维是 129，而不是 128？总大小是多少？

> 每个 split 要输出 128 维 partial attention output，并额外输出一个 split LSE，所以 shape 是：
>
> $$
> [B,H_q,S,D+1]=[64,32,32,129].
> $$
>
> FP32 总字节数为：
>
> $$
> 64\times32\times32\times129\times4
> =33{,}816{,}576\ \text{B}=32.25\ \text{MiB}.
> $$
>
> 前 128 个元素已经除以当前 split 的 softmax denominator，是 normalized partial output；第 129 个元素是该 split 的 LSE。Stage2 用 LSE 在全局尺度重新加权，不能直接对 32 个 partial output 求平均。

**可能追问：mid_o 改成 FP16 能节省一半中间流量吗？**

> 容量上可以减少，但接口和 Stage2 必须一起改，LSE 与 partial output 的精度影响也要分别评估。32 个 split 的重新加权可能放大某些舍入效应，所以不能只按字节收益就断言性能和精度都更优。

---

### 159. Stage2 为什么必须同时读取 partial output 和 split LSE？

> 设第 $s$ 个 split 的 normalized output 为 $o_s$，LSE 为 $L_s$。全局最大值$M=\max_s L_s$，则合并权重是：
>
> $$
> w_s=e^{L_s-M},
> $$
>
> $$
> o=\frac{\sum_s w_so_s}{\sum_s w_s},\qquad
> L=M+\log\sum_s w_s.
> $$
>
> 若只保留 $o_s$ 而没有 $L_s$，就不知道每个 split 的 softmax denominator 相对大小，无法恢复全序列 attention。LSE 同时保证指数计算数值稳定。

**可能追问：为什么不能把每个 split 的 output 简单求平均？**

> 各 split 的归一化常数通常不同。一个分片即使 token 数相同，也可能拥有更大的 logits 和总概率质量，应该得到更大权重。只有极特殊的相同归一化常数情形，平均才等价；一般情况必须使用 LSE 权重。

---

### 160. `0.664842 ms` 是否等于表中的 Stage1 中位数加 Stage2 中位数？

> 不严格相等，因为 Full 与单独两阶段来自独立计时。Full runner 把 Stage1、Stage2 按顺序放在同一个被测函数中；Stage1-only 和 Stage2-only 则分别测量，各自得到五轮均值，再分别取中位数。
>
> | 路径 | Stage1 | Stage2 | Full |
> |---|---:|---:|---:|
> | Triton V2-fixed | 1.066916 ms | 0.024852 ms | 1.104148 ms |
> | CUDA V7 | 0.631255 ms | 0.008868 ms | 0.664842 ms |
>
> 中位数不满足可加性，不同 runner 的缓存状态与执行过程也可能不同。因此不能把 Full 减去两个独立中位数的差值直接归因成固定 launch overhead，更不能拿 V9 Stage1 与 V7 Stage2 相加就宣布获得 V9 Full 的实测时间。

**可能追问：Full 与两段时间之和的差值，能直接叫 launch overhead 吗？**

> 不能直接这样归因。三组数来自独立测量和独立中位数，cache 状态、执行顺序及样本变化都会影响差值。要量化 launch 或提交间隙，需要专门的测量或时间线工具，不能从表格相减得到唯一解释。

---

### 161. 为什么用 SoA Triton V2-fixed 作为主要 baseline？

> 因为它与 CUDA candidate 使用相同的 TurboQuant 算法语义、相同 4bit_nc 数据、相同固定 workload，并采用更适合 metadata 访问的 SoA layout；相比 AoS V1 和 SoA V1，它是仓库中更强的 Triton Stage1 baseline。用弱 baseline 会夸大 CUDA 优化收益。
>
> `fixed` 还很重要：原 V2 的 V column layout 有 correctness bug，可能出现 LSE 正确但 output 错误。性能比较使用修复后的 V2；历史修复前 NCU 报告只能辅助定位，不能作为当前公平性能结论。

**可能追问：既然 SoA V1 更慢，为什么仍然用于正确性参考？**

> 性能基线和正确性参考可以承担不同角色。SoA V1 的独立计算路径有助于发现 V2/Tensor 路径的共同错误，而性能比较应选已经修正、同语义且更强的 V2-fixed。不能只因某实现更快就让它成为唯一真值。

---

### 162. `1.66x` 是怎样得到的？它等于快了 66% 吗？

> Full Decode 加速比是：
>
> $$
> speedup=\frac{1.104148}{0.664842}=1.661\times.
> $$
>
> “吞吐能力约为原来的 1.661 倍”可以口语化为“1.66x faster”。但 latency reduction 是：
>
> $$
> 1-\frac{0.664842}{1.104148}=39.79\%.
> $$
>
> 因此不能说延迟降低 66%。`1.66x` 和 `39.8% latency reduction` 是同一组时间的两种表达，分母不同。

**可能追问：固定 batch 下 1.66 倍能否直接说服务吞吐提高 66%？**

> 只能说该 Full Decode 计算路径按倒数时间换算有约 66% 的处理能力提升。完整服务还包含其他算子、调度和通信，整体 tokens/s 需要系统测试，不能把局部 Kernel 比例直接外推。

---

### 163. “单次 Stage1 launch”是否表示量化、Store 和整个 Decode 都只有一次 launch？

> 不是。它只表示 Stage1 attention 内部没有把以下步骤拆成多个 global Kernel： packed K/V load、nibble unpack、K lookup、V affine reconstruction、QK、online softmax 和 PV accumulation 在一个 CUDA Stage1 launch 内完成。
>
> KV Store 在历史 token 写入 cache 时单独执行；Query rotation 在 benchmark 之前；跨 split 归并由 Stage2 第二个 launch 完成。简历中的“融合至单次 Stage1 launch”已经限定了范围，回答时不能省略 `Stage1` 这个限定词。

**可能追问：KV Store 的成本为什么没有包含在 Stage1？**

> 历史 K/V 在写入时量化一次，后续生成会被多次 Decode 读取，因此 Store 与 Decode 是不同的复用阶段。单测 Stage1 便于隔离读取计算路径；评估真正每步时延时，新 token 的 Store 与 rotation 仍应纳入完整范围。

---

### 164. WMMA fragment 直接写回依赖什么非通用假设？如何验证？

> 它依赖 `sm_89` 下 accumulator fragment 元素到 lane/fragment index 的实际映射。 CUDA WMMA C++ API允许 load、mma 和 store，但 fragment 内部元素布局不是稳定的跨架构 ABI。V5 绕过 `store_matrix_sync -> Shared Memory -> scalar reload` 时，必须知道哪些 lane 持有哪一行、哪一列。
>
> 项目使用 `cuda/wmma_fragment_probe.cu` 恢复并验证映射，再把直接写回限制在目标架构。更换 GPU 架构、CUDA 编译器或 MMA shape 后应重新跑 probe 和完整 output correctness，不能仅凭旧映射编译通过就认为正确。

**可能追问：仅在一种随机输入上 output 对齐，足以证明 fragment 映射吗？**

> 不够。随机输入可能掩盖对称或重复值造成的问题，probe 应让矩阵位置携带可辨识的值，再覆盖有效行、列与 padding。完成映射验证后，还要在实际 Attention 数据上做整体回归。

---

### 165. “同步 barrier 消减”具体删除了什么？为什么可以删？

> V7 删除的是每 tile 的 PV 之后一个冗余等待点。下一轮开头 warp 0 更新独立的 metadata/data_base 数组，已有的 metadata 发布 barrier 会等待上一轮所有 warp 完成 PV；只有越过它，线程才覆盖下一轮 K/V tile。因此这两个同步边界可以合并，宏开关为 TQ4_FUSED_TILE_BARRIER。
>
> V9 删除的是另一类通信：QK accumulator 不再先写 qk_s 再由其他 warp 读出，warp 0 直接在寄存器里完成 score reduction 和 online state 更新，只把后续 PV 必须共享的概率及参数写出，所以 qk_s 的 producer-consumer barrier 也不再需要。
>
> 仓库记录中 V7 静态 BAR.SYNC 为 34，V9 为 26；这是固定展开代码的静态 site，不等同于硬件 stall 比例。安全性要按 shared 生命周期推导，并辅以 racecheck 和数值验证；当前资料不能用来声称所有动态 race 检查均已完成。

**可能追问：V7 和 V9 删除的是同一种 barrier 吗？**

> 不是同一个位置。V7 把 tile 末尾等待与下一轮 metadata 发布同步合并；V9 取消 QK score 的 shared 物化，使对应 producer-consumer barrier 不再需要。两者都依赖生命周期分析，但触发条件和消除的数据路径不同。

---

### 166. 哪些是上游 vLLM 内容，哪些是这个项目自己的工作？

> 上游或参考内容包括 TurboQuant 数学、centroid 生成、4bit_nc 配置、Triton Store/Decode 和 cache layout；仓库的 `reference/`、`vllm/` 是对应快照，不应声称为原创。
>
> 本项目工作的主要证据在：CUDA V1-V9 的设计演进、固定 workload 的 Tensor Core 映射、fragment 写回、packed load、barrier/寄存器状态优化、CUDA Stage2、benchmark harness、Store→Decode 验证以及文档化分析。面试回答应把“基于 vLLM 接口实现与优化”说清楚，既不贬低工程工作，也不把上游算法归到个人贡献。

**可能追问：如何避免把上游算法和个人优化混为一谈？**

> 我会先说明 TurboQuant、Lloyd-Max、参考 Store 与布局来自上传的上游快照，再把自己的设计、实现和分析落实到 CUDA 版本、benchmark 和验证脚本。个人完成范围按实际经历描述，代码来源和优化贡献分别讲清。

---

### 167. 将简历 TurboQuant 段落逐短语拆开后，是否每项都有独立问题？

> 是的。我把简历中的关键短语分别映射到独立问题；面试时既能给整体介绍，也能沿某个实现细节继续展开：
>
> | 简历原短语 | 独立问题 |
> |---|---|
> | 基于 vLLM TurboQuant `4bit_nc` | 5、12、77、80、95、166 |
> | CUDA Tensor Core 融合 Decode Kernel | 20、29、31–33、91、99 |
> | 4-bit K/V unpack | 18、41、120、128、143、153–154 |
> | Lloyd-Max 查表 | 8、69–74、89、141–142、154 |
> | Grouped Attention | 28–31、100、114、150–152 |
> | online softmax | 23–25、34、99、156–159 |
> | Value 反量化 | 11、81–82、144–146、153 |
> | 单次 Stage1 launch | 20、27、98–99、163 |
> | split-KV Stage2 归并 | 21–26、98、122–123、158–160 |
> | WMMA fragment 寄存器直写 | 40、127、164 |
> | 向量化访存 | 18、41、120、128、143 |
> | barrier 消减 | 42、129、165 |
> | KV Cache 压缩约 `3.82x` | 13、96、138、146 |
> | RTX 4090 | 64、92–94、106–110、125 |
> | Qwen3-4B 固定 workload | 4、39、97、100–105、111–117 |
> | Full Decode `0.6648 ms` | 27、53、98、123、160 |
> | SoA Triton V2 baseline 提速 `1.66x` | 14、45、49、53、161–162 |
> | norm/scale 等参数粒度 | 136–146 |
> | 个人贡献与边界 | 59–65、124、134、166 |
>
> 这里的验收标准不再是“某个综合答案顺带提过”，而是关键实现假设必须能被单独抽问、给出公式或代码映射，并说明适用边界。

**可能追问：面对没有准备过的变形题，怎样用这份题库回答？**

> 先回到输入形状和 cache 契约，再判断 Query/KV 的复用、矩阵维度、状态归属和同步是否仍成立，最后说明需要什么验证。这样可以从已理解的实现推出答案，而不是把固定数字套到新场景。

---

## 性能数字速查

以下为上传仓库 README 的实验记录；Stage1 与 Full 两张表来自各自独立测量。

| 实现 | Stage1 中位时间 |
|---|---:|
| AoS Triton V1 | 1.692314 ms |
| SoA Triton V1 | 1.284270 ms |
| SoA Triton V2-fixed | 1.075988 ms |
| CUDA V1 | 2.074204 ms |
| CUDA V2 | 1.748470 ms |
| CUDA V3 | 1.380587 ms |
| CUDA V4 | 1.121208 ms |
| CUDA V5 | 0.845486 ms |
| CUDA V6 | 0.638863 ms |
| CUDA V7 | 0.631122 ms |
| CUDA V8 | 0.513208 ms |
| CUDA V9 | 0.484516 ms |

| Full 实验 | Stage1 | Stage2 | Full |
|---|---:|---:|---:|
| Triton V2-fixed | 1.066916 ms | 0.024852 ms | 1.104148 ms |
| CUDA V7 | 0.631255 ms | 0.008868 ms | 0.664842 ms |

**统一口径：**3.82 倍是 KV 格式的逻辑压缩比；4.28 倍是 CUDA V1→V9 的 Stage1 演进；2.22 倍是 V9 Stage1 对 SoA Triton V2-fixed；1.66 倍是 V7 Full 对 Triton Full。Full 指预旋转 Query 下的 Stage1+Stage2，不含 Store、rotation 或整模型推理。

## 源码阅读入口

路径均相对于上传项目根目录。

| 要核对的内容 | 阅读入口 |
|---|---|
| 固定 shape、字节容量、输入构造 | baseline/common.py |
| codebook 与 Store 量化 | reference/centroids.py、reference/soa_store.py |
| Hadamard 与上游接口快照 | reference/turboquant_attn.py、vllm/README.md |
| V4–V7 的共同计算主线 | cuda/tq4_cuda_stage1_template.cuh |
| V7 Stage2 与 Full 两次 launch | cuda/tq4_cuda_v7.cu |
| 原生 m16n8k16 与寄存器状态 | cuda/tq4_cuda_v8.cu、cuda/tq4_cuda_v9.cu |
| WMMA 内部映射探测 | cuda/wmma_fragment_probe.cu |
| Triton V 的维度解包修复 | baseline/triton_v2.py |
| 计时、correctness 与版本入口 | benchmarks/stage1.py、benchmarks/full_decode.py |
| 未修改 Store 到 Decode 的验证 | validation/soa_store.py、validation/store_decode.py |
| 时间、误差与静态资源记录 | README.md、README_CN.md、cuda/README.md |
| 历史 profiler 的背景与限制 | results/v2_stage1.md |

上传仓库的快速验证入口是 `./run.sh smoke`、`./run.sh store` 与 `./run.sh benchmark`，需在匹配的 CUDA 环境中运行。README 记录的实验环境为 RTX 4090、Python 3.11.15、PyTorch 2.5.1+cu121、Triton 3.7.1、CUDA compiler 12.2；复现时以实际环境和新输出为准。
