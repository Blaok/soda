#!/bin/bash
for XDEVICE in xilinx:adm-pcie-7v3:1ddr:3.0 xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0
do
    if [ "${XDEVICE}" = "xilinx:adm-pcie-7v3:1ddr:3.0" ]
    then
        export SDA_VER=2017.2
        export DRAM_BANK=1
        unset DRAM_SEPARATE
    elif [ "${XDEVICE}" = "xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0" ]
    then
        export SDA_VER=2017.1
        export DRAM_BANK=4
        export DRAM_SEPARATE=yes
    fi
    for target in exe csim cosim bitstream
    do
        # 2d benchmarks
        for app in sobel2d denoise2d jacobi2d seidel2d
        do
            (
                make APP=${app}      UNROLL_FACTOR=2 ${target}
                make APP=${app} REPLICATION_FACTOR=2 ${target}
            )&
        done

        # 3d benchmarks
        for app in denoise3d jacobi3d heat3d
        do
            (
                make APP=${app}      UNROLL_FACTOR=2 TILE_SIZE_DIM_1=32 HOST_ARGS='32 32 32' ${target}
                make APP=${app} REPLICATION_FACTOR=2 TILE_SIZE_DIM_1=32 HOST_ARGS='32 32 32' ${target}
            )&
        done

        wait
    done
done
