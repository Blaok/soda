#!/bin/sh
TARGET=$1
BENCHMARK=gaussian
DIM0=1998
DIM1=998

INPUT_DEPTH=$(( ( (${DIM0}+${TILE_SIZE_DIM0}-5) / (${TILE_SIZE_DIM0}-4) ) * ( (${DIM1}+${TILE_SIZE_DIM1}-5) / (${TILE_SIZE_DIM1}-4) ) * ${TILE_SIZE_DIM0} * ${TILE_SIZE_DIM1} / 32 ))

sed -i 's/#pragma HLS INTERFACE m_axi port=var_gaussian_y offset=slave depth=.* bundle=gmem1 latency=120/#pragma HLS INTERFACE m_axi port=var_gaussian_y offset=slave depth='${INPUT_DEPTH}' bundle=gmem1 latency=120/' src/${KERNEL_SRCS}
sed -i 's/#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth=.* bundle=gmem2 latency=120/#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth='${INPUT_DEPTH}' bundle=gmem2 latency=120/' src/${KERNEL_SRCS}

if echo $0|grep '\-hls.sh$' >/dev/null 2>/dev/null
then
    # HLS
    rm -f bit/${BENCHMARK}-hls-cosim.xclbin bit/${BENCHMARK}-hls-hw.xclbin
    make bit/${BENCHMARK}-hls-cosim.xclbin bit/${BENCHMARK}-hls-hw.xclbin
    mv bit/${BENCHMARK}-hls-cosim.xclbin ${TARGET}/bit/${BENCHMARK}-hls-cosim-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.xclbin
    mv bit/${BENCHMARK}-hls-hw.xclbin ${TARGET}/bit/${BENCHMARK}-hls-hw-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.xclbin
    RPT_BASE="_xocc_${BENCHMARK}_hls_${BENCHMARK}-hls-hw.dir/impl/kernels/${BENCHMARK}_kernel/${BENCHMARK}_kernel/solution_OCL_REGION_0/syn/report"
    cp ${RPT_BASE}/${BENCHMARK}_kernel_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-hls-kernel-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/compute_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-hls-compute-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/load_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-hls-load-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/store_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-hls-store-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
else
    # UNROLL
    rm -f bit/${BENCHMARK}-cosim.xclbin bit/${BENCHMARK}-hw.xclbin
    make bit/${BENCHMARK}-cosim.xclbin bit/${BENCHMARK}-hw.xclbin
    mv bit/${BENCHMARK}-cosim.xclbin ${TARGET}/bit/${BENCHMARK}-cosim-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.xclbin
    mv bit/${BENCHMARK}-hw.xclbin ${TARGET}/bit/${BENCHMARK}-hw-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.xclbin
    RPT_BASE="_xocc_${BENCHMARK}_kernel_${BENCHMARK}-hw.dir/impl/kernels/${BENCHMARK}_kernel/${BENCHMARK}_kernel/solution_OCL_REGION_0/syn/report"
    cp ${RPT_BASE}/${BENCHMARK}_kernel_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-kernel-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/compute_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-compute-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/load_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-load-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
    cp ${RPT_BASE}/store_csynth.rpt ${TARGET}/rpt/${BENCHMARK}-store-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.rpt
fi

