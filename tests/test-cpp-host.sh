#!/bin/bash
. "${0%/*}"/util.sh
prepare

for file in "${base_dir}"/tests/src/*.soda; do
  basename="${file##*/}"
  kernel_name="${basename/%.soda/_kernel}"
  cpp_host="${tmp_dir}/${basename/%soda/host.cpp}"
  cpp_exe="${tmp_dir}/${basename/%soda/exe}"
  cpp_kernel="${tmp_dir}/${basename/%soda/cpp}"

  log "+ sodac <- ${basename} ... "
  "${base_dir}/src/sodac" "${file}" \
    --xocl-kernel "${cpp_kernel}" \
    --frt-host "${cpp_host}" &&
    pass || fail

  log "  - g++ <- cpp ... "
  g++ \
    "${cpp_kernel}" \
    "${cpp_host}" \
    -o "${cpp_exe}" \
    -g -fsanitize=address \
    -DSODA_TEST_MAIN -DSODA_CPP_BINDING \
    -I"${XILINX_VIVADO}"/include \
    -lOpenCL &&
    pass || fail

  log "  - exe <- exe ... "
  "${cpp_exe}" '' >&2 && pass || fail
done 2>&${log_fd}

cleanup
