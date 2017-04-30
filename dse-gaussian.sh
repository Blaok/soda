#!/bin/sh
BENCHMARK=gaussian

export CSIM_XCLBIN="${BENCHMARK}-csim.xclbin"
export COSIM_XCLBIN="${BENCHMARK}-cosim.xclbin"
export HW_XCLBIN="${BENCHMARK}-hw.xclbin"
export KERNEL_SRCS="${BENCHMARK}_kernel.cpp"
export KERNEL_NAME="${BENCHMARK}_kernel"
export HOST_SRCS="${BENCHMARK}_run.cpp ${BENCHMARK}.cpp"
export HOST_ARGS=" "
export HOST_BIN="${BENCHMARK}"

#for TILE in 128:128 64:64 256:256 64:128 128:64 64:256 256:64 128:256 256:128
#do
#    export TILE_SIZE_DIM0=${TILE//:*/}
#    export TILE_SIZE_DIM1=${TILE//*:/}
#    rm -f bin/${BENCHMARK}
#    make -B bin/${BENCHMARK} >&2
#    mv bin/${BENCHMARK}{,-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}
#    for UNROLL in 1 8 16 32 64
#    do
#        export UNROLL_FACTOR=${UNROLL}
#        make mktemp
#        mv mktemp{,-${BENCHMARK}-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}}.sh
#        echo "./mktemp-${BENCHMARK}-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}.sh -c 'export CSIM_XCLBIN=\"${CSIM_XCLBIN}\"; export COSIM_XCLBIN=\"${COSIM_XCLBIN}\"; export HW_XCLBIN=\"${HW_XCLBIN}\"; export KERNEL_SRCS=\"${KERNEL_SRCS}\"; export KERNEL_NAME=\"${KERNEL_NAME}\"; export HOST_SRCS=\"${HOST_SRCS}\"; export HOST_ARGS=\"${HOST_ARGS}\"; export HOST_BIN=\"${HOST_BIN}\"HOST_BIN; export TILE_SIZE_DIM0=\"${TILE_SIZE_DIM0}\"; export TILE_SIZE_DIM1=\"${TILE_SIZE_DIM1}\"; export UNROLL_FACTOR=${UNROLL_FACTOR}; cp ${PWD}/make-${BENCHMARK}.sh .;./make-${BENCHMARK}.sh ${PWD}'"
#    done
#done

export TILE_SIZE_DIM0=128
export TILE_SIZE_DIM1=128
export UNROLL_FACTOR=32
export CSIM_XCLBIN="${BENCHMARK}-hls-csim.xclbin"
export COSIM_XCLBIN="${BENCHMARK}-hls-cosim.xclbin"
export HW_XCLBIN="${BENCHMARK}-hls-hw.xclbin"
export KERNEL_SRCS="${BENCHMARK}_hls.cpp"

echo "make mktemp;./mktemp.sh -c 'export CSIM_XCLBIN=\"${CSIM_XCLBIN}\"; export COSIM_XCLBIN=\"${COSIM_XCLBIN}\"; export HW_XCLBIN=\"${HW_XCLBIN}\"; export KERNEL_SRCS=\"${KERNEL_SRCS}\"; export KERNEL_NAME=\"${KERNEL_NAME}\"; export HOST_SRCS=\"${HOST_SRCS}\"; export HOST_ARGS=\"${HOST_ARGS}\"; export HOST_BIN=\"${HOST_BIN}\"HOST_BIN; export TILE_SIZE_DIM0=\"${TILE_SIZE_DIM0}\"; export TILE_SIZE_DIM1=\"${TILE_SIZE_DIM1}\"; export UNROLL_FACTOR=${UNROLL_FACTOR}; cp ${PWD}/make-${BENCHMARK}-hls.sh .;./make-${BENCHMARK}-hls.sh ${PWD}'"

