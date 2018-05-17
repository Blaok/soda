#!/bin/bash
make_supo() {
    (
        export LABEL
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
        export HOST_ARGS
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

        exec /usr/bin/time --verbose --output=${log_base}.time -- make $@ > >(exec tee ${log_base}.stdout) 2> >(exec tee ${log_base}.stderr 1>&2)
    )
}
