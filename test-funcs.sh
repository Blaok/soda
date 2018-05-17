run_exec_time() {
    for num_tile in 128 256 512 1024
    do
        for t in $(DRAM_BANK=2 with-sdaccel bin/${LABEL}/${XDEVICE}/${app}-tile${tile_size}-iterate${iterate}-border-ignored bit/${LABEL}/${XDEVICE}/${app}-hw-tile${tile_size}-unroll${unroll}-2ddr-${separate}iterate${iterate}-border-ignored-none-clustered.xclbin ${tile_size//x/ } ${num_tile}|&grep 'Kernel execution time:'|awk '{print $4}')
        do
            echo -n "$t "
        done
    done
    echo
}

run_post_syn_util() {
    rpt_file=rpt/${LABEL}/${XDEVICE}/${app}-hw-tile${tile_size}-unroll${unroll}-2ddr-${separate}iterate${iterate}-border-ignored-none-clustered/kernel_util_routed.rpt 
    if test -f ${rpt_file}
    then
        grep '_kernel ' ${rpt_file}|&awk '{print $4"\t"$8"\t"$12"\t"$16"\t"$20}'
    else
        echo
    fi
}

run_freq() {
    bit_file=bit/${LABEL}/${XDEVICE}/${app}-hw-tile${tile_size}-unroll${unroll}-2ddr-${separate}iterate${iterate}-border-ignored-none-clustered.xclbin
    if test -f ${bit_file}
    then
        grep -aoe 'frequency="\(.*\)MHz"' ${bit_file}|grep -oP '[\d.]*'
    else
        echo
    fi
}

run_estimate() {
    src/supoc src/${app}.supo --estimation-file - --model-file src/ku3_model.json --unroll-factor ${unroll} --iterate ${iterate} --dram-bank ${bank} --dram-separate ${DRAM_SEPARATE} --tile-size ${tile_size//x/ }|jq '"\(.resource_routed.BRAM) \(.resource_routed.DSP) \(.resource_routed.LUT) \(.resource_routed.REG) \(.performance)"' -r
}

