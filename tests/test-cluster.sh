#!/bin/bash
. "${0%/*}"/util.sh
prepare

granularity="${1:-full}"

for file in "${base_dir}"/tests/src/*.soda; do
  basename="${file##*/}"
  frt_host="${tmp_dir}/${basename/%soda/host.cpp}"
  iocl_exe="${tmp_dir}/${basename/%soda/frt.exe}"
  iocl_kernel="${tmp_dir}/${basename/%soda/cl}"
  iocl_bitstream="${tmp_dir}/${basename/%soda/aocx}"
  xhls_exe="${tmp_dir}/${basename/%soda/cpp.exe}"
  xhls_kernel="${tmp_dir}/${basename/%soda/cpp}"

  log "+ sodac --cluster=none <- ${basename} ... "
  "${base_dir}/src/sodac" "${file}" \
    --cluster=none \
    --frt-host "${frt_host}" &&
    pass || fail

  log "+ sodac --cluster=${granularity} <- ${basename} ... "
  "${base_dir}/src/sodac" "${file}" \
    --cluster="${granularity}" \
    --iocl-kernel="${iocl_kernel}" \
    --xocl-platform "${XCL_PLATFORM}" \
    --xocl-kernel="${xhls_kernel}" &&
    pass || fail

  log "  - g++ <- xhls ... "
  g++ \
    "${xhls_kernel}" \
    "${frt_host}" \
    -o "${xhls_exe}" \
    -g -fsanitize=address \
    -DSODA_TEST_MAIN -DSODA_CPP_BINDING \
    -I"${XILINX_VIVADO}"/include \
    -lOpenCL &&
    pass || fail

  log "  - exe <- xhls ... "
  "${xhls_exe}" '' >&2 && pass || fail

  which aoc >/dev/null || continue

  log "  - aoc <- iocl ... "
  aoc "${iocl_kernel}" -g -o "${iocl_bitstream}" \
    -march=emulator -legacy-emulator -v \
    "-I${INTELFPGAOCLSDKROOT}/include/kernel_headers" \
    >&2 &&
    pass || fail

  log "  - g++ <- frt  ... "
  g++ \
    "${frt_host}" \
    -o "${iocl_exe}" \
    -DSODA_TEST_MAIN \
    -O2 \
    "-I${XILINX_VIVADO}/include" \
    -lfrt -lOpenCL &&
    pass || fail

  log "  - exe <- iocl ... "
  "${iocl_exe}" "${iocl_bitstream}" >&2 && pass || fail
done 2>&${log_fd}

cleanup
