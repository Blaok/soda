#!/bin/sh
BENCHMARK=curved
export RM=":"

export CSIM_XCLBIN="${BENCHMARK}-csim.xclbin"
export COSIM_XCLBIN="${BENCHMARK}-cosim.xclbin"
export HW_XCLBIN="${BENCHMARK}-hw.xclbin"
export KERNEL_SRCS="${BENCHMARK}_kernel.cpp"
export KERNEL_NAME="${BENCHMARK}_kernel"
export HOST_SRCS="${BENCHMARK}_run.cpp ${BENCHMARK}.cpp"
export HOST_ARGS=" "
export HOST_BIN="${BENCHMARK}"

(for TILE in 128:128 64:64 256:256 64:128 128:64 64:256 256:64 128:256 256:128
do
    export TILE_SIZE_DIM0=${TILE//:*/}
    export TILE_SIZE_DIM1=${TILE//*:/}
    sed -i 's/#pragma HLS INTERFACE m_axi port=var_processed offset=slave depth=.* bundle=gmem1 latency=120/#pragma HLS INTERFACE m_axi port=var_processed offset=slave depth='$(( ( (2560+${TILE_SIZE_DIM0}-23) / (TILE_SIZE_DIM0-22) ) * ( (1920+${TILE_SIZE_DIM1}-19) / (TILE_SIZE_DIM1-18) ) * ( ${TILE_SIZE_DIM0} * ${TILE_SIZE_DIM1} / 21 ) ))' bundle=gmem1 latency=120/' src/${KERNEL_SRCS}
    sed -i 's/#pragma HLS INTERFACE m_axi port=var_input offset=slave depth=.* bundle=gmem2 latency=120/#pragma HLS INTERFACE m_axi port=var_input offset=slave depth='$(( ( (2560+${TILE_SIZE_DIM0}-23) / (TILE_SIZE_DIM0-22) ) * ( (1920+${TILE_SIZE_DIM1}-19) / (TILE_SIZE_DIM1-18) ) * ( ${TILE_SIZE_DIM0} * ${TILE_SIZE_DIM1} / 64 ) ))' bundle=gmem2 latency=120/' src/${KERNEL_SRCS}
    for UNROLL in 1 2 4 8 16 20 24 28 32
    do
        export UNROLL_FACTOR=${UNROLL}

        rm -f bin/${BENCHMARK}
        make bin/${BENCHMARK}
        mv bin/${BENCHMARK}{,-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}


        rm -f bit/${BENCHMARK}-cosim.xclbin bit/${BENCHMARK}-hw.xclbin
        make -j2 bit/${BENCHMARK}-cosim.xclbin bit/${BENCHMARK}-hw.xclbin
        mv bit/${BENCHMARK}-cosim{,-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}.xclbin
        mv bit/${BENCHMARK}-hw{,-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}.xclbin
        RPT_BASE="_xocc_${BENCHMARK}_kernel_${BENCHMARK}-hw.dir/impl/kernels/${BENCHMARK}_kernel/${BENCHMARK}_kernel/solution_OCL_REGION_0/syn/report"
        cp ${RPT_BASE}/${BENCHMARK}_kernel_csynth.rpt rpt/${BENCHMARK}-kernel-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
        cp ${RPT_BASE}/compute_csynth.rpt rpt/${BENCHMARK}-compute-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
        cp ${RPT_BASE}/load_csynth.rpt rpt/${BENCHMARK}-load-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
        cp ${RPT_BASE}/store_csynth.rpt rpt/${BENCHMARK}-store-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
    done
done) &

export TILE_SIZE_DIM0=128
export TILE_SIZE_DIM1=128
export UNROLL_FACTOR=32
export CSIM_XCLBIN="${BENCHMARK}-hls-csim.xclbin"
export COSIM_XCLBIN="${BENCHMARK}-hls-cosim.xclbin"
export HW_XCLBIN="${BENCHMARK}-hls-hw.xclbin"
export KERNEL_SRCS="${BENCHMARK}_hls.cpp"

sed -i 's/#pragma HLS INTERFACE m_axi port=var_processed offset=slave depth=.* bundle=gmem1 latency=120/#pragma HLS INTERFACE m_axi port=var_processed offset=slave depth='$(( ( (2560+${TILE_SIZE_DIM0}-23) / (TILE_SIZE_DIM0-22) ) * ( (1920+${TILE_SIZE_DIM1}-19) / (TILE_SIZE_DIM1-18) ) * ( ${TILE_SIZE_DIM0} * ${TILE_SIZE_DIM1} / 21 ) ))' bundle=gmem1 latency=120/' src/${KERNEL_SRCS}
sed -i 's/#pragma HLS INTERFACE m_axi port=var_input offset=slave depth=.* bundle=gmem2 latency=120/#pragma HLS INTERFACE m_axi port=var_input offset=slave depth='$(( ( (2560+${TILE_SIZE_DIM0}-23) / (TILE_SIZE_DIM0-22) ) * ( (1920+${TILE_SIZE_DIM1}-19) / (TILE_SIZE_DIM1-18) ) * ( ${TILE_SIZE_DIM0} * ${TILE_SIZE_DIM1} / 64 ) ))' bundle=gmem2 latency=120/' src/${KERNEL_SRCS}

rm -f bit/${BENCHMARK}-hls-cosim.xclbin bit/${BENCHMARK}-hls-hw.xclbin
(make -j2 bit/${BENCHMARK}-hls-cosim.xclbin bit/${BENCHMARK}-hls-hw.xclbin
mv bit/${BENCHMARK}-hls-cosim{,-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}.xclbin
mv bit/${BENCHMARK}-hls-hw{,-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}.xclbin
RPT_BASE="_xocc_${BENCHMARK}_hls_${BENCHMARK}-hls-hw.dir/impl/kernels/${BENCHMARK}_kernel/${BENCHMARK}_kernel/solution_OCL_REGION_0/syn/report"
cp ${RPT_BASE}/${BENCHMARK}_kernel_csynth.rpt rpt/${BENCHMARK}-hls-kernel-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
cp ${RPT_BASE}/compute_csynth.rpt rpt/${BENCHMARK}-hls-compute-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
cp ${RPT_BASE}/load_csynth.rpt rpt/${BENCHMARK}-hls-load-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt
cp ${RPT_BASE}/store_csynth.rpt rpt/${BENCHMARK}-hls-store-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.rpt) &

function ctrl_c()
{
    echo "Ctrl-C pressed"
    jobs -p
    kill $(jobs -p)
}


trap ctrl_c INT

wait

