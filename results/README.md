# Historical Profiling Results

This directory contains Nsight Compute reports and a SASS export captured
while investigating the original SoA Triton V2 kernel.

These artifacts **predate the V2 correctness fix**. The profiled kernel had a
silent V-column permutation caused by its `tl.interleave` path, so the old
timing near `2.29 ms` is invalid as a performance baseline. The memory-access
and instruction observations remain useful as historical diagnostic material,
but exact performance metrics should be recollected on V2-fixed.

Current correctness and benchmark results are recorded in `../README.md`.
The detailed historical analysis is in `../docs/v2_stage1.md`.
