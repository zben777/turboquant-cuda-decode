# 完整链路验证

`soa_store.py` 让未经修改的 vLLM SoA Store 可以独立运行。`store_decode.py`
先让原始 Q/K/V 经过该 Store，再把同一个压缩 cache 交给 Triton V2-fixed 与
CUDA V7，并将最终结果和 FP32 attention 比较。

在仓库根目录运行 `./run.sh store`。
