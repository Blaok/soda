#!/bin/bash
. "$(dirname $0)/util.sh"
prepare

for file in "${base_dir}/tests/src/"*.soda; do
  basename="$(basename "${file}")"
  iocl_kernel="${tmp_dir}/${basename/%soda/cl}"
  xhls_kernel="${tmp_dir}/${basename/%soda/cpp}"

  log "+ sodac <- ${basename} ... "
  "${base_dir}/src/sodac" "${file}" \
    --iocl-kernel "${iocl_kernel}" \
    --xocl-kernel "${xhls_kernel}" &&
    pass || fail

  log "  - g++ <- xhls ... "
  g++ -std=c++11 -fsyntax-only "-I${XILINX_VIVADO}/include" \
    -c "${xhls_kernel}" >&2 &&
    pass || fail

  log "  - aoc <- iocl ... "
  aoc "-I${INTELFPGAOCLSDKROOT}/include/kernel_headers" \
    -c "${iocl_kernel}" >&2 &&
    pass || fail
done 2>&${log_fd}

cleanup
