#!/bin/bash
RUN_ID=iccad18-unroll-factor-ku3-heat3d
LOG_DIR=log/${RUN_ID}
mkdir -p ${LOG_DIR}
DRAM_BANK=2
DRAM_SEPARATE=yes
CLUSTER=none
BORDER=ignore
XDEVICE='xilinx:adm-pcie-ku3:2ddr:3.3'
LABEL=iccad18
APP=heat3d

. ./make_supo.sh

for tile_size in 128
do
    TILE_SIZE_DIM_0=${tile_size}
    TILE_SIZE_DIM_1=${tile_size}
    HOST_ARGS="${tile_size} ${tile_size} 128"
    for iterate in 1 8
    do
        ITERATE=${iterate}
        make_supo exe
        for unroll_factor in 1 2 4 8 16
        do
            (
                UNROLL_FACTOR=${unroll_factor}
                for target in csim hls bitstream
                do
                    make_supo ${target}
                    if [ "${unroll_factor}" -ne "1" ]
                    then
                        REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
                    fi
                done
            )&
        done
        wait
    done
done

