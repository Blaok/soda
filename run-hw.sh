#!/bin/bash

for BITSTREAM in bit/$1*-hw-*.xclbin
do
    TMP=${BITSTREAM//-hw-*/}
    BENCHMARK=${TMP//bit\//}
    TMP=${BITSTREAM//*-unroll/}
    UNROLL=${TMP//-tile*/}
    TMP=${BITSTREAM//*-tile/}
    TILE=${TMP//.xclbin/}
    LOG=${PWD}/log
    with-sdaccel bin/${BENCHMARK//-hls/}-tile${TILE} bit/${BENCHMARK}-hw-unroll${UNROLL}-tile${TILE}.xclbin |tee ${LOG}/${BENCHMARK}-hw-unroll${UNROLL}-tile${TILE}.log
done

