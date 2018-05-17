#!/bin/bash
RUN_ID=dse-jacobi2d
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

make_supo() {
    (
        export APP
        export UNROLL_FACTOR
        export REPLICATION_FACTOR
        export DRAM_BANK
        export DRAM_SEPARATE
        export TILE_SIZE_DIM_0
        export TILE_SIZE_DIM_1
        export CLUSTER
        export BORDER
        export ITERATE
        export AWS_BUCKET
        export XDEVICE
        task_id="$(awk -F ':' '{print $2}' <<< "${XDEVICE}"|awk -F '-' '{print $3}')-${APP}-${target}-tile${TILE_SIZE_DIM_0}"
        if [[ ! -z "${TILE_SIZE_DIM_1}" ]]
        then
            task_id="${task_id}x${TILE_SIZE_DIM_1}"
        fi
        if [[ -z "${REPLICATION_FACTOR}" ]]
        then
            task_id="${task_id}-unroll${UNROLL_FACTOR}"
        else
            task_id="${task_id}-replicate${REPLICATION_FACTOR}"
        fi
        task_id="${task_id}-ddr${DRAM_BANK}"
        if [[ ! -z "${DRAM_SEPARATE}" ]]
        then
            task_id="${task_id}-separated"
        fi
        task_id="${task_id}-iterate${ITERATE}-border-${BORDER}-${CLUSTER}-clustered"

        i=0
        log_base=${LOG_DIR}/${task_id}
        while test -f ${log_base}.time | test -f ${log_base}.stdout | test -f ${log_base}.stderr
        do
            log_base=${LOG_DIR}/${task_id}.${i}
            i=$((i+1))
        done

        exec /usr/bin/time --verbose --output=${log_base}.time -- make $@ > >(exec tee ${log_base}.stdout /dev/stdout) 2> >(exec tee ${log_base}.stderr /dev/stderr)
    )
}

for target in bitstream #exe csim hls
do
    for tile_size in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536
    do
        (
            TILE_SIZE_DIM_0=${tile_size} make_supo ${target}
            TILE_SIZE_DIM_0=${tile_size} REPLICATION_FACTOR=${UNROLL_FACTOR} make_supo ${target}
        )&
    done
    wait
done

exit 0

for target in exe csim hls
do
    num_pe=1
    for iterate in 1 2 4 8 16 32 64
    do
        (
            ITERATE=${iterate}      UNROLL_FACTOR=${num_pe} make_supo ${target}
            ITERATE=${iterate} REPLICATION_FACTOR=${num_pe} make_supo ${target}
        )&
    done

    num_pe=2
    for iterate in 1 2 4 8 16 32
    do
        (
            ITERATE=${iterate}      UNROLL_FACTOR=${num_pe} make_supo ${target}
            ITERATE=${iterate} REPLICATION_FACTOR=${num_pe} make_supo ${target}
        )&
    done

    num_pe=4
    for iterate in 1 2 4 8 16
    do
        (
            ITERATE=${iterate}      UNROLL_FACTOR=${num_pe} make_supo ${target}
            ITERATE=${iterate} REPLICATION_FACTOR=${num_pe} make_supo ${target}
        )&
    done

    num_pe=8
    for iterate in 1 2 3 4 5 6 7 8
    do
        (
            ITERATE=${iterate}      UNROLL_FACTOR=${num_pe} make_supo ${target}
            ITERATE=${iterate} REPLICATION_FACTOR=${num_pe} make_supo ${target}
        )&
    done

    num_pe=16
    for iterate in 1 2 3 4
    do
        (
            ITERATE=${iterate}      UNROLL_FACTOR=${num_pe} make_supo ${target}
            ITERATE=${iterate} REPLICATION_FACTOR=${num_pe} make_supo ${target}
        )&
    done
    wait
done

