#!/usr/bin/env bash
set -euo pipefail


# Stage1 V1-V9 benchmark
cd /home/bzhang/code/codex/turboquant-cuda-decode
/home/bzhang/miniconda3/envs/tqdecode/bin/python -B -m benchmarks.stage1 \
  --include-cuda --include-cuda-v2 --include-cuda-v3 \
  --include-cuda-v4 --include-cuda-v5 --include-cuda-v6 \
  --include-cuda-v7 --include-cuda-v8 --include-cuda-v9 \
  --rounds 5


# Full Decode benchmark
cd /home/bzhang/code/codex/turboquant-cuda-decode
/home/bzhang/miniconda3/envs/tqdecode/bin/python -B -m benchmarks.full_decode \
  --rounds 5


# Commit and push
cd /home/bzhang/code/codex/turboquant-cuda-decode
git add -A
git diff --cached --quiet || git commit -m "Update project"
git push origin main

