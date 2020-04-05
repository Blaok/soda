#!/bin/bash
. "$(dirname $0)/util.sh"
prepare

for file in "${base_dir}/tests/src/"*.soda; do
  basename="$(basename "${file}")"
  kernel_name="${basename/%.soda/_kernel}"
  frt_host="${tmp_dir}/${basename/%soda/host.cpp}"
  frt_exe="${tmp_dir}/${basename/%soda/exe}"
  iocl_kernel="${tmp_dir}/${basename/%soda/cl}"
  iocl_bitstream="${tmp_dir}/${basename/%soda/aocx}"
  xhls_kernel="${tmp_dir}/${basename/%soda/cpp}"
  xhls_object="${tmp_dir}/${basename/%soda/xo}"
  xhls_bitstream="${tmp_dir}/${basename/%soda/xclbin}"

  log -n "+ sodac <- ${basename} ... "
  "${base_dir}/src/sodac" "${file}" \
    --frt-host "${frt_host}" \
    --iocl-kernel "${iocl_kernel}" \
    --xocl-kernel "${xhls_kernel}" &&
    log "PASS" || log "FAIL"

  log -n "  - g++ <- frt ... "
  g++ "${frt_host}" -g -o "${frt_exe}" \
    "-I${XILINX_VIVADO}/include" -DSODA_TEST_MAIN -lfrt -lOpenCL &&
    log "PASS" || log "FAIL"

  log -n "  - v++ <- xhls ... "
  v++ "${xhls_kernel}" -g -o "${xhls_object}" \
    --compile \
    --target sw_emu \
    --kernel "${kernel_name}" \
    --platform "${XCL_PLATFORM}" \
    --advanced.prop "kernel.${kernel_name}.kernel_flags=-std=c++11" \
    >&2 && test "${xhls_object}" -nt "${xhls_kernel}" &&
    log "PASS" || log "FAIL"

  log -n "  - v++ <- xo ... "
  v++ "${xhls_object}" -g -o "${xhls_bitstream}" \
    --link \
    --target sw_emu \
    --platform "${XCL_PLATFORM}" \
    >&2 && test "${xhls_bitstream}" -nt "${xhls_object}" &&
    log "PASS" || log "FAIL"

  log -n "  - exe <- xclbin ... "
  ${frt_exe} ${xhls_bitstream} >&2 && log "PASS" || log "FAIL"

  log -n "  - aoc <- iocl ... "
  aoc "${iocl_kernel}" -g -o "${iocl_bitstream}" \
    -march=emulator -legacy-emulator -v \
    "-I${INTELFPGAOCLSDKROOT}/include/kernel_headers" \
    >&2 &&
    log "PASS" || log "FAIL"

  log -n "  - exe <- aocx ... "
  ${frt_exe} ${iocl_bitstream} >&2 && log "PASS" || log "FAIL"
done 2>&${log_fd}

cleanup
