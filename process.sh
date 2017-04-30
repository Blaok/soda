#!/bin/bash
# Goal: read lines from stdin, echo command lines that can be executed by parallel
# Input format: <benchmark>-unroll<k>-tile<dim0>x<dim1>-ki<ki>-ko<ko>
# Output
log()
{
    echo "$@" >&2
}
while read line
do
    BENCHMARK=${line//-unroll*/}
    line=${line//${BENCHMARK}-unroll/}
    K=${line//-tile*/}
    line=${line//${K}-tile/}
    TILE0=${line//x*/}
    line=${line//${TILE0}x/}
    TILE1=${line//-ki*/}
    line=${line//${TILE1}-ki/}
    KI=${line//-ko*/}
    KO=${line//${KI}-ko/}

    log $BENCHMARK $K $TILE0 $TILE1 $KI $KO

    export RM=:
    export CSIM_XCLBIN="${BENCHMARK}-csim.xclbin"
    export COSIM_XCLBIN="${BENCHMARK}-cosim.xclbin"
    export HW_XCLBIN="${BENCHMARK}-hw.xclbin"
    export KERNEL_SRCS="${BENCHMARK}_kernel.cpp"
    export KERNEL_NAME="${BENCHMARK}_kernel"
    export HOST_SRCS="${BENCHMARK}_run.cpp ${BENCHMARK}.cpp"
    export HOST_ARGS=" "
    export HOST_BIN="${BENCHMARK}"

    export TILE_SIZE_DIM0=${TILE0}
    export TILE_SIZE_DIM1=${TILE1}
    export UNROLL_FACTOR=${K}
    make mktemp
    mv mktemp{,-${BENCHMARK}-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}}.sh
    echo "./mktemp-${BENCHMARK}-unroll${UNROLL_FACTOR}-tile${TILE_SIZE_DIM0}x${TILE_SIZE_DIM1}-ki${KI}-ko${KO}.sh -c 'export CSIM_XCLBIN=\"${CSIM_XCLBIN}\"; export COSIM_XCLBIN=\"${COSIM_XCLBIN}\"; export HW_XCLBIN=\"${HW_XCLBIN}\"; export KERNEL_SRCS=\"${KERNEL_SRCS}\"; export KERNEL_NAME=\"${KERNEL_NAME}\"; export HOST_SRCS=\"${HOST_SRCS}\"; export HOST_ARGS=\"${HOST_ARGS}\"; export HOST_BIN=\"${HOST_BIN}\"; export TILE_SIZE_DIM0=\"${TILE_SIZE_DIM0}\"; export TILE_SIZE_DIM1=\"${TILE_SIZE_DIM1}\"; export UNROLL_FACTOR=${UNROLL_FACTOR};export KI=\"${KI}\";export KO=\"${KO}\"; cp ${PWD}/make-${BENCHMARK}.sh .;./make-${BENCHMARK}.sh ${PWD}'"

done
