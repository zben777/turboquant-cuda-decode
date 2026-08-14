# 历史 Profiling 结果

本目录保存分析原始 SoA Triton V2 kernel 时生成的 Nsight Compute 报告和
SASS 导出。

这些文件生成于 **V2 correctness 修复之前**。当时被 profile 的 kernel
因为 `tl.interleave` 路径存在静默的 V 列置换，因此旧的约 `2.29 ms` 时间
不能作为有效性能 baseline。其访存和指令观察仍可作为历史诊断材料，但精确
性能指标需要在 V2-fixed 上重新采集。

当前 correctness 与 benchmark 结果见 `../README_CN.md`，详细历史分析见
`../docs/v2_stage1.md`。
