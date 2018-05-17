#!/bin/bash
RUN_ID=iccad18-unroll-factor
LOG_DIR=log/${RUN_ID}
mkdir -p ${LOG_DIR}
APP=jacobi2d
UNROLL_FACTOR=8
DRAM_BANK=2
DRAM_SEPARATE=yes
TILE_SIZE_DIM_0=32
CLUSTER=none
BORDER=ignore
XDEVICE='xilinx:adm-pcie-ku3:2ddr:3.3'
ITERATE=1

. ./make_supo.sh

for app in sobel2d
do
    for target in exe csim hls bitstream
    do
        for tile_size in 16384
        do
            for unroll_factor in 1 2 4 8 16 32 64
            do
            (
                APP=${app} TILE_SIZE_DIM_0=${tile_size}      UNROLL_FACTOR=${unroll_factor} make_supo ${target}
                APP=${app} TILE_SIZE_DIM_0=${tile_size} REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
            )&
            done
            wait
        done
    done
done

for app in denoise2d
do
    for target in exe csim hls bitstream
    do
        for tile_size in 16384
        do
            for unroll_factor in 1 2 4 8 16 32 64
            do
            (
                unset DRAM_SEPARATE
                APP=${app} TILE_SIZE_DIM_0=${tile_size}      UNROLL_FACTOR=${unroll_factor} make_supo ${target}
                APP=${app} TILE_SIZE_DIM_0=${tile_size} REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
            )&
            done
            wait
        done
    done
done

for app in denoise3d 
do
    for target in exe csim hls bitstream
    do
        for tile_size in 128
        do
            for unroll_factor in 1 2 4 8 16 32 64
            do
            (
                unset DRAM_SEPARATE
                APP=${app} TILE_SIZE_DIM_0=${tile_size} TILE_SIZE_DIM_1=${tile_size}      UNROLL_FACTOR=${unroll_factor} make_supo ${target}
                APP=${app} TILE_SIZE_DIM_0=${tile_size} TILE_SIZE_DIM_1=${tile_size} REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
            )&
            done
            wait
        done
    done
done

for app in jacobi2d seidel2d
do
    for target in exe csim hls bitstream
    do
        for tile_size in 16384
        do
            for iterate in 1 8
            do
                for unroll_factor in 1 2 4 8 16 32 64
                do
                (
                    APP=${app} TILE_SIZE_DIM_0=${tile_size}      UNROLL_FACTOR=${unroll_factor} make_supo ${target}
                    APP=${app} TILE_SIZE_DIM_0=${tile_size} REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
                )&
                done
                wait
            done
        done
    done
done

for app in jacobi3d heat3d
do
    for target in exe csim hls bitstream
    do
        for tile_size in 128
        do
            for iterate in 1 8
            do
                for unroll_factor in 1 2 4 8 16 32 64
                do
                (
                    APP=${app} TILE_SIZE_DIM_0=${tile_size} TILE_SIZE_DIM_1=${tile_size}      UNROLL_FACTOR=${unroll_factor} make_supo ${target}
                    APP=${app} TILE_SIZE_DIM_0=${tile_size} TILE_SIZE_DIM_1=${tile_size} REPLICATION_FACTOR=${unroll_factor} make_supo ${target}
                )&
                done
                wait
            done
        done
    done
done

