#!/bin/bash
test -z "${BENCHMARK}" && echo 'Set BENCHMARK please!' >&2 && exit 1
test -z "${NPIXELS}" && echo 'Set NPIXELS please!' >&2 && exit 1
test -z "${PIXEL_WIDTH}" && echo 'Set PIXEL_WIDTH please!' >&2 && exit 1
test -z "${UNROLL_FACTOR}" && echo 'Set UNROLL_FACTOR please!' >&2 && exit 1
test -z "${TILE_SIZE}" && echo 'Set TILE_SIZE please!' >&2 && exit 1

test -z "${HLS_RPT}" && HLS_RPT="rpt/hw/${BENCHMARK}_kernel_csynth.rpt"
UTIL="$(grep '|Utilization (%)' ${HLS_RPT})"
IFS='|'
UTIL_ARRAY=(${UTIL})
BRAM="${UTIL_ARRAY[2]// /}"
DSP="${UTIL_ARRAY[3]// /}"
FF="${UTIL_ARRAY[4]// /}"
LUT="${UTIL_ARRAY[5]// /}"
unset IFS

HLS_LATENCY_TILE="$(grep -E '\|- Loop.*\| *\?\| *\?\|' ${HLS_RPT})"
HLS_LATENCY_TILE_ARRAY=(${HLS_LATENCY_TILE})
HLS_LATENCY_TILE="${HLS_LATENCY_TILE_ARRAY[8]}"

test -z "${COSIM_LOG}" && COSIM_LOG="log/${BENCHMARK}-cosim.log"
COSIM_LATENCY="$(grep 'Emulation time:' ${COSIM_LOG}|tail -n 1)"
COSIM_LATENCY_ARRAY=(${COSIM_LATENCY})
COSIM_LATENCY=$(zsh -c "eval echo \$\(\(${COSIM_LATENCY_ARRAY[9]} \* 1000\)\)")

test -z "${ONBOARD_LOG}" && ONBOARD_LOG="log/${BENCHMARK}-hw.log"
ONBOARD_LATENCY="$(grep '^Kernel run time: ' ${ONBOARD_LOG})"
ONBOARD_LATENCY_ARRAY=(${ONBOARD_LATENCY})
ONBOARD_LATENCY=${ONBOARD_LATENCY_ARRAY[3]}

test -z "${HW_XCLBIN}" && HW_XCLBIN="bit/${BENCHMARK}-hw.xclbin"
HW_FREQ="$(grep 'frequency=' ${HW_XCLBIN} -a)"
IFS='"'
HW_FREQ_ARRAY=(${HW_FREQ})
HW_FREQ="${HW_FREQ_ARRAY[3]//MHz/}"

echo ${BENCHMARK}, ${NPIXELS}, ${PIXEL_WIDTH}, ${UNROLL_FACTOR}, ${TILE_SIZE}, ${BRAM}, ${DSP}, ${FF}, ${LUT}, ${HLS_LATENCY_TILE}, ${COSIM_LATENCY}, ${ONBOARD_LATENCY}, ${HW_FREQ}

