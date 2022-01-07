#!/bin/bash
. "$(dirname $0)/util.sh"
prepare

for file in "${base_dir}/tests/src/"*.soda; do
  basename="$(basename "${file}")"
  xrtl_object="${tmp_dir}/${basename/%soda/hw.xo}"
  xrtl_bitstream="${tmp_dir}/${basename/%soda/hw_emu.xclbin}"
  xrtl_connectivity="${tmp_dir}/${basename/%soda/connectivity.ini}"
  frt_host="${tmp_dir}/${basename/%soda/host.cpp}"
  frt_exe="${tmp_dir}/${basename/%soda/exe}"

  log "+ sodac <- ${basename} ... "
  python3 "${base_dir}/src/sodac" "${file}" \
    --frt-host "${frt_host}" \
    --xocl-platform "${XCL_PLATFORM}" \
    --xocl-connectivity="${xrtl_connectivity}" \
    --xocl-hw-xo "${xrtl_object}" &&
    pass || fail

  log "  - v++ <- xo ... "
  v++ "${xrtl_object}" -g -o "${xrtl_bitstream}" \
    --link \
    --target hw_emu \
    --platform "${XCL_PLATFORM}" \
    --config="${xrtl_connectivity}" \
    >&2 && test "${xrtl_bitstream}" -nt "${xrtl_object}" &&
    pass || fail

  log "  - g++ <- frt ... "
  g++ "${frt_host}" -g -o "${frt_exe}" \
    "-I${XILINX_VIVADO}/include" -DSODA_TEST_MAIN -lfrt -lOpenCL &&
    pass || fail

  log "  - exe <- xclbin ... "
  ${frt_exe} ${xrtl_bitstream} >&2 && pass || fail

done 2>&${log_fd}

cleanup
