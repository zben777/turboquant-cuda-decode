#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

stage1_cuda_flags=(
  --include-cuda
  --include-cuda-v2
  --include-cuda-v3
  --include-cuda-v4
  --include-cuda-v5
  --include-cuda-v6
  --include-cuda-v7
  --include-cuda-v8
)

usage() {
  cat <<'EOF'
Usage: ./run.sh MODE

Modes:
  check       Parse every Python source file; no GPU work.
  layout      Build and validate the fixed synthetic SoA inputs.
  smoke       Short Triton/CUDA V7-V8 Stage1 and V7 Full Decode regression.
  store       Raw Q/K/V -> vLLM SoA Store -> Triton/CUDA Decode check.
  benchmark   Official five-round Stage1 V1-V8 and Full Decode benchmark.
  all         Run check, layout, benchmark, and store in sequence.
  clean       Remove generated PyTorch CUDA extension build directories.
  help        Show this message.

Environment:
  PYTHON=/path/to/python    Python interpreter (default: python)
  CUDA_HOME=/path/to/cuda  Optional CUDA toolkit override
EOF
}

run_check() {
  echo "== Python syntax check =="
  PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/turboquant_cuda_pycache" \
    "${PYTHON_BIN}" -m compileall -q \
    "${PROJECT_ROOT}/baseline" \
    "${PROJECT_ROOT}/benchmarks" \
    "${PROJECT_ROOT}/validation" \
    "${PROJECT_ROOT}/tools" \
    "${PROJECT_ROOT}/reference" \
    "${PROJECT_ROOT}/vllm"
}

run_layout() {
  echo "== Fixed-layout self-test =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m baseline.common)
}

run_smoke() {
  run_check
  echo "== Short Stage1 regression =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m benchmarks.stage1 \
    --include-cuda-v7 --include-cuda-v8 --warmup 3 --iters 10 --rounds 1
  )
  echo "== Short Full Decode regression =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m benchmarks.full_decode \
    --warmup 3 --iters 10 --rounds 1
  )
}

run_store() {
  echo "== Real SoA Store compatibility =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m validation.store_decode)
}

run_benchmark() {
  echo "== Official Stage1 benchmark =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m benchmarks.stage1 \
    "${stage1_cuda_flags[@]}"
  )
  echo "== Official Full Decode benchmark =="
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -B -m benchmarks.full_decode)
}

run_clean() {
  echo "== Removing generated CUDA extension builds =="
  rm -rf "${PROJECT_ROOT}"/cuda/build_v*
}

mode="${1:-help}"
case "${mode}" in
  check)
    run_check
    ;;
  layout)
    run_layout
    ;;
  smoke)
    run_smoke
    ;;
  store)
    run_store
    ;;
  benchmark)
    run_benchmark
    ;;
  all)
    run_check
    run_layout
    run_benchmark
    run_store
    ;;
  clean)
    run_clean
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    usage >&2
    exit 2
    ;;
esac
