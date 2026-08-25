TurboQuant/
|
├── 1. 背景：为什么KV Cache需要量化
|
├── 2. Attention Decode数学过程
|
├── 3. TurboQuant论文算法
│   ├── Random Rotation
│   ├── Beta distribution
│   ├── Gaussian approximation
│   ├── Scalar Quantization
│   └── Lloyd-Max
|
├── 4. Codebook生成全过程
│   ├── vLLM centroids.py源码解析
│   ├── N(0,1/d)
│   ├── PDF积分
│   ├── boundary更新
│   └── centroid收敛
|
├── 5. KV Cache存储格式
│   ├── K layout
│   ├── V layout
│   └── 134 Bytes slot
|
├── 6. TurboQuant_mse vs TurboQuant_prod
│   ├── residual
│   ├── QJL
│   └── 为什么vLLM不用
|
├── 7. vLLM实现
│   ├── backend
│   ├── quantization
│   └── Triton decode
|
├── 8. CUDA kernel对应
│   ├── Stage1
│   ├── Stage2
│   └── thread mapping
|
└── 9. 面试回答模板









# TurboQuant 原理、vLLM 实现与 CUDA Decode 详解

> 学习目标：理解 TurboQuant 从论文算法到 vLLM 工程实现，再到 CUDA decode
> kernel 的完整链路。

## 1. TurboQuant解决的问题

LLM decode阶段的主要瓶颈是KV Cache显存访问。

Attention:

\[ Attention(Q,K,V)=softmax(QK\^T)V \]

生成一个token时，需要读取历史KV Cache，因此随着context长度增长，KV
Cache成为memory bandwidth瓶颈。

TurboQuant目标：

    FP16 KV Cache
          |
          v
    低bit KV Cache
          |
          v
    直接参与Attention计算

避免：

    INT4 KV
     |
    dequant
     |
    FP16 KV
     |
    Attention

产生额外memory traffic。

------------------------------------------------------------------------

# 2. TurboQuant整体思想

TurboQuant不是简单的：

    FP16 -> INT4

而是：

    vector x

     |
     v

    norm normalization

     |
     v

    orthogonal rotation

     |
     v

    scalar quantization

     |
     v

    4bit index

核心思想：

通过rotation让高维vector的coordinate满足稳定分布，然后使用固定codebook进行scalar
quantization。

------------------------------------------------------------------------

# 3. Rotation为什么有效

对于归一化向量：

\[ \|\|x\|\|\_2=1 \]

进行：

\[ y=`\Pi `{=tex}x \]

其中：

\[ `\Pi`{=tex}\^T`\Pi`{=tex}=I \]

旋转保持长度不变，但是重新分配能量。

随机旋转后：

-   coordinate服从Beta型分布
-   高维情况下近似Gaussian
-   coordinate之间近似独立

因此：

高维vector quantization

可以转换为：

多个独立scalar quantization。

------------------------------------------------------------------------

# 4. coordinate分布到底是什么？

论文理论：

随机旋转后的vector在单位球面：

\[ S\^{d-1} \]

均匀分布。

单个coordinate的边缘分布：

Beta distribution。

但是：

当：

\[ d\>=64 \]

时：

coordinate近似：

\[ N(0,1/d) \]

因此vLLM工程实现直接使用：

\[ X`\sim `{=tex}N(0,1/d) \]

作为Lloyd-Max优化分布。

------------------------------------------------------------------------

# 5. vLLM codebook构造

重要：

vLLM不是：

    真实KV Cache采样

    ↓

    训练codebook

也不是：

    Gaussian采样

    ↓

    K-means

而是：

    Gaussian PDF

    ↓

    Lloyd-Max积分优化

    ↓

    centroid table

------------------------------------------------------------------------

## Lloyd-Max过程

假设4bit：

需要：

\[ 2\^4=16 \]

个centroid。

初始化：

    c0,c1,...c15

然后循环：

### 1. 根据centroid确定boundary

\[ b_i=(c_i+c\_{i+1})/2 \]

### 2. 更新centroid

对于区间：

\[ \[b\_{i-1},b_i\] \]

新的中心：

\[ c_i= `\frac{\int x f(x)dx}`{=tex} {`\int `{=tex}f(x)dx} \]

不断迭代直到收敛。

最终：

得到固定：

    centroid[16]

部署时查表。

------------------------------------------------------------------------

# 6. K量化流程

输入：

    K vector
    dim=128

## Step 1 norm

计算：

\[ `\gamma`{=tex}=\|\|K\|\|\_2 \]

保存：

    gamma(float)

归一化：

\[ K/`\gamma`{=tex} \]

------------------------------------------------------------------------

## Step 2 rotation

\[ K_r=`\Pi`{=tex}(K/`\gamma`{=tex}) \]

------------------------------------------------------------------------

## Step 3 centroid lookup

每个coordinate：

找到最近centroid：

    Kr[i]

    ↓

    nearest centroid

    ↓

    index

保存：

    4bit index

    +
    gamma

------------------------------------------------------------------------

# 7. Decode恢复K

不是：

    4bit

    ↓

    FP16 K

    ↓

    Attention

而是：

    4bit index

    ↓

    centroid[index]

    ↓

    multiply norm

    ↓

    QK dot

直接fusion。

------------------------------------------------------------------------

# 8. 为什么Query也需要rotation

因为：

\[ QK\^T \]

如果：

\[ K_r=`\Pi `{=tex}K \]

那么：

\[ Q_r=Q`\Pi`{=tex}\^T \]

保持：

\[ QK^T=Q_rK_r^T \]

------------------------------------------------------------------------

# 9. TurboQuant_prod与QJL

TurboQuant_mse:

目标：

降低：

\[ \|\|x-`\hat{x}`{=tex}\|\|\^2 \]

TurboQuant_prod:

目标：

优化：

\[ q\^Tx \]

因为attention依赖inner product。

流程：

    MSE quant

    ↓

    得到x_hat

    ↓

    residual:

    r=x-x_hat

    ↓

    QJL residual encoding

QJL不是scale。

QJL作用：

补偿quantization error。

scale/norm作用：

恢复幅度。

二者完全不同。

------------------------------------------------------------------------

# 10. vLLM是否使用QJL

当前vLLM TurboQuant decode路径：

使用：

    rotation

    +
    centroid

    +
    norm/scale

没有：

    QJL residual

原因：

QJL增加：

-   storage
-   decode计算
-   memory访问

工程上选择更简单的路径。

------------------------------------------------------------------------

# 11. K和V区别

K:

用于：

\[ QK\^T \]

需要保持inner product。

因此：

    4bit index

    ↓

    centroid lookup

    ↓

    norm correction

------------------------------------------------------------------------

V:

用于：

\[ PV \]

不需要inner product。

使用：

    4bit index

    +

    scale

    +

    zero

普通uniform quantization。

------------------------------------------------------------------------

# 12. 对你的CUDA项目对应

你的项目：

    turboquant-cuda-decode

对应：

vLLM TurboQuant路径。

K:

    4bit index

    +

    centroid table

    +

    float norm

V:

    4bit index

    +

    scale

    +

    zero

decode:

    compressed K

    ↓

    unpack

    ↓

    centroid lookup

    ↓

    QK

    ↓

    softmax

    ↓

    compressed V

    ↓

    dequant

    ↓

    PV

------------------------------------------------------------------------

# 13. 总结

TurboQuant本质：

不是普通INT4。

它利用：

    rotation
    +
    稳定coordinate分布
    +
    Lloyd-Max codebook
    +
    online lookup

实现：

低bit KV Cache。

其中：

-   centroid来自离线数学优化
-   norm保存向量尺度
-   QJL属于TurboQuant_prod，不是scale
-   vLLM当前实现没有QJL
-   K和V采用不同量化策略
