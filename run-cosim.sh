#!/bin/bash
ALLTMPFILE=$(mktemp --suffix=-alltmps-$$)

TASKS=$(for BITSTREAM in bit/$1*-cosim-*.xclbin
do
    TMP=${BITSTREAM//-cosim-*/}
    BENCHMARK=${TMP//bit\//}
    TMP=${BITSTREAM//*-unroll/}
    UNROLL=${TMP//-tile*/}
    TMP=${BITSTREAM//*-tile/}
    TILE=${TMP//.xclbin/}
    TMP=$(mktemp -d --suffix=-cosim-$$)
    echo -n "${TMP} " >> ${ALLTMPFILE}
    cp bin/${BENCHMARK//-hls/}-tile${TILE} bit/${BENCHMARK}-cosim-unroll${UNROLL}-tile${TILE}.xclbin bin/emconfig.json ${TMP}
    LOG=${PWD}/log
    echo "(cd ${TMP};XCL_EMULATION_MODE=true SDA_VER=2016.3 with-sdaccel ./${BENCHMARK//-hls/}-tile${TILE} ${BENCHMARK}-cosim-unroll${UNROLL}-tile${TILE}.xclbin |tee ${LOG}/${BENCHMARK}-cosim-unroll${UNROLL}-tile${TILE}.log;rm -rf ${TMP})"
done)

ALLTMPS=$(cat ${ALLTMPFILE})
rm -f ${ALLTMPFILE}

function ctrl_c()
{
    echo
    echo "Ctrl-C pressed."
    rm -rf ${ALLTMPS}
    exit 0
}

trap ctrl_c INT
trap ctrl_c TERM

echo "${TASKS}"|parallel --no-notice $2

