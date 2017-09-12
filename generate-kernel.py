#!/usr/bin/python3.6
import functools
import json
import math
import operator
import os
import sys

# input
#input_type = 'uint16_t'
#output_type = 'uint16_t'
#app_name = 'blur'
#St = [2000, 0]
#A = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
##A = [[0,-1],[-1,0],[0,0],[1,0],[0,1]]
#k = 16

# constants
coords_in_tile = 'ijkl'
coords_in_orig = 'pqrs'

# serialize
def Serialize(vec, St):
    # convert vec to scalar coordinates
    result = vec[0]
    for i in range(1, len(St)):
        result += vec[i]*functools.reduce(operator.mul, St[0:i])
    return result

def GetChains(St, A, k):
    A_dag = set()
    for i in range(0, k):
        A_dag = A_dag.union([x+i for x in [Serialize(x, St) for x in A]])
    A_dag = list(A_dag)
    A_dag.sort()
    A_dag = [x for x in A_dag]
    chains = []
    for i in range(0, k):
        chains += [[x for x in A_dag if x%k == i]]
    return chains

def GetPoints(A, k):
    all_points = {} # {offset:{unroll_index:point_index}}
    for i in range(0, k):
        points = [Serialize(x, St) for x in A]
        for idx, j in enumerate(points):
            all_points[j+i] = all_points.get(j+i, {})
            all_points[j+i][i] = idx
    return all_points

def GetBuffer(St, A, k):
    FFs = []    # [outputs]
    FIFOs = {}  # {length:[outputs]}
    inputs = []
    for chain in GetChains(St, A, k):
        inputs += [chain[-1]]
        for j in range(0, len(chain)-1):
            interval_length = chain[j+1]-chain[j]
            if interval_length == k:
                FFs += [chain[j]]
            else:
                FIFOs[interval_length] = FIFOs.get(interval_length, []) + [chain[j]]
    FFs.sort()
    inputs.sort()
    for FIFO in FIFOs.values():
        FIFO.sort()
    return {'FFs':FFs, 'FIFOs':FIFOs, 'k':k, 'inputs':inputs}

def GetConsumeMap(buf):
    mapping = {'k':buf['k']}    # maps an offset to what consumes it
    for idx, item in enumerate(buf['FFs']):
        mapping[item+k] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item+fifo_length] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def GetProduceMap(buf):
    mapping = {'k':buf['k']}    # maps an offset to what produces it
    for idx, item in enumerate(buf['inputs']):
        mapping[item] = {'inputs':{'index':idx, 'produce':item}}
    for idx, item in enumerate(buf['FFs']):
        mapping[item] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def PrintLine(line = '', local_indent = -1):
    global indent
    if local_indent < 0:
        local_indent = indent
    if line:
        print('%s%s' % (' '*local_indent*4,line))
    else:
        print()

def PrintUpdate(producer_map, consumer_map, src):
    if 'FIFOs' in src:
        dst = consumer_map.get(src['FIFOs']['produce'])
        if not dst:
            return
        fifo_length = src['FIFOs']['length']/producer_map['k']
        fifo_index = src['FIFOs']['index']
        src_str = 'FIFO_%d[c][%d][FIFO_%d_ptr]' % (fifo_length, fifo_index, fifo_length)
    elif 'FFs' in src:
        dst = consumer_map.get(src['FFs']['produce'])
        if not dst:
            return
        src_str = 'FF[c][%d]' % (src['FFs']['index'])
    else:
        dst = consumer_map.get(src['inputs']['produce'])
        if not dst:
            return
        src_str = 'input_buffer[c][%d]' % (src['inputs']['index'])

    PrintUpdate(producer_map, consumer_map, dst)

    if 'FIFOs' in dst:
        fifo_length = dst['FIFOs']['length']/consumer_map['k']
        fifo_index = dst['FIFOs']['index']
        dst_str = 'FIFO_%d[c][%d][FIFO_%d_ptr]' % (fifo_length, fifo_index, fifo_length)
    elif 'FFs' in dst:
        dst_str = 'FF[c][%d]' % (dst['FFs']['index'])
    else:
        return

    PrintLine('%s = %s;' % (dst_str, src_str))

def PrintCompute(St, A, k, compute_content, input_partition, output_partition, extra_params):
    global indent
    buf = GetBuffer(St, A, k)
    points = GetPoints(A, k)
#    indent_save = indent
#    indent = 0
#    PrintLine('#if %d!=(UNROLL_FACTOR)' % k)
#    PrintLine('#error UNROLL_FACTOR != %d' % k)
#    PrintLine('#endif')
#    PrintLine()
#    for dim, size in enumerate(St):
#        PrintLine('#if %d!=(TILE_SIZE_DIM%d)' % (size, dim))
#        PrintLine('#error TILE_SIZE_DIM%d != %d' % (dim, size))
#        PrintLine('#endif')
#        PrintLine()
#    PrintLine('#if ((%s*(PIXEL_WIDTH))%%(BURST_WIDTH))!=0' % '*'.join(['(TILE_SIZE_DIM%d)'%x for x in range(0, len(St))]))
#    PrintLine('#error TILE_SIZE%BURST_WIDTH != 0')
#    PrintLine('#endif')
#    PrintLine()
#    indent = indent_save
    PrintLine('void compute(bool compute_flag, output_type output[CHANNEL_NUM_O][BURST_LENGTH],')
    indent += 1
    PrintLine('input_type input[CHANNEL_NUM_I][BURST_LENGTH],')
    if extra_params:
        for param_name, param in extra_params.items():
            if 'dup' in param:
                dup = ('[%d]' % param['dup'])
            else:
                dup = ''
            PrintLine('%s %s%s[UNROLL_FACTOR][%d],' % (param['type'], param_name, dup, param['length']))
    PrintLine('input_type FF[CHANNEL_NUM_I][%d],' % len(buf['FFs']))
    for fifo_length, fifo_list in buf['FIFOs'].items():
        PrintLine('input_type FIFO_%d[CHANNEL_NUM_I][%d][%d],' % (fifo_length/k, len(fifo_list), fifo_length/k))
    PrintLine('int32_t FIFO_ptrs[%d],' % len(buf['FIFOs']))
    for i in range(0, len(St)):
        PrintLine('int32_t %s_base[UNROLL_FACTOR],' % coords_in_tile[i])
    for i in range(0, len(St)-1):
        PrintLine('int32_t %s_base,' % coords_in_orig[i])
    PrintLine('int32_t input_index_base)')
    indent -= 1
    PrintLine('{')
    indent += 1
    PrintLine('if(compute_flag)')
    PrintLine('{')
    indent += 1
    i = 0
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('int32_t& FIFO_%d_ptr = FIFO_ptrs[%d];' % (fifo_length/k, i))
        i += 1
    PrintLine()

    PrintLine('input_type input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d];' % len(A))
    for idx, item in enumerate(A):
        PrintLine('//         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d] <=> (%s)' % (idx, str(item)[1:-1]))
    PrintLine('input_type input_buffer[CHANNEL_NUM_I][UNROLL_FACTOR];')
    PrintLine('#pragma HLS array_partition variable=input_points complete dim=0', 0)
    PrintLine('#pragma HLS array_partition variable=input_buffer complete dim=0', 0)
    PrintLine()

    PrintLine('// produce output')
    PrintLine('compute_epoch:', 0)
    PrintLine('for(int32_t epoch = 0; epoch < BURST_LENGTH/UNROLL_FACTOR; ++epoch)')
    PrintLine('{')
    indent += 1
    PrintLine('#pragma HLS dependence variable=FF inter false', 0)
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('#pragma HLS dependence variable=FIFO_%d inter false' % (fifo_length/k), 0)
#    indent_save = indent
#    indent = 0
#    PrintLine('#pragma HLS array_partition variable=FF complete')
#    for fifo_length in buf['FIFOs'].keys():
#        PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=1' % (fifo_length/k))
#    indent = indent_save
    PrintLine('int32_t input_index = epoch + input_index_base;')
    PrintLine('#pragma HLS pipeline II=1', 0)
    PrintLine('compute_load_channel:', 0)
    PrintLine('for(int32_t c = 0; c<CHANNEL_NUM_I; ++c)')
    PrintLine('{')
    indent += 1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('compute_load_unrolled:', 0)
    PrintLine('for(int32_t unroll_index = 0; unroll_index<UNROLL_FACTOR; ++unroll_index)')
    PrintLine('{')
    indent += 1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('input_buffer[c][unroll_index] = input[c][epoch*UNROLL_FACTOR+unroll_index];')
    indent -= 1
    PrintLine('}')
    indent -= 1
    PrintLine('}')
    PrintLine()

    PrintLine('compute_chain_channel:', 0)
    PrintLine('for(int32_t c = 0; c<CHANNEL_NUM_I; ++c)')
    PrintLine('{')
    indent += 1
    PrintLine('#pragma HLS unroll', 0)
    for idx, item in enumerate(buf['inputs']):
        for unroll_index in points[item].keys():
            PrintLine("input_points[c][%d][%d] = input_buffer[c][%d]; // (%s)" % (unroll_index, points[item][unroll_index], idx, str(A[points[item][unroll_index]])[1:-1]))
    for idx, item in enumerate(buf['FFs']):
        for unroll_index in points[item].keys():
            PrintLine("input_points[c][%d][%d] = FF[c][%d]; // (%s)" % (unroll_index, points[item][unroll_index], idx, str(A[points[item][unroll_index]])[1:-1]))
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            for unroll_index in points[item].keys():
                PrintLine("input_points[c][%d][%d] = FIFO_%d[c][%d][FIFO_%d_ptr]; // (%s)" % (unroll_index, points[item][unroll_index], fifo_length/k, idx, fifo_length/k, str(A[points[item][unroll_index]])[1:-1]))

    indent -= 1
    PrintLine('}')
    PrintLine()

    PrintLine('compute_unrolled:', 0)
    PrintLine('for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS unroll', 0)
    for i in range(0, len(St)):
        PrintLine('int32_t& %s = %s_base[unroll_index];' % ((coords_in_tile[i],)*2))
    for i in range(0, len(St)-1):
        PrintLine('int32_t %s = %s_base+%s;' % ((coords_in_orig[i],)*2 + (coords_in_tile[i],)))
    PrintLine('int32_t %s = %s;' % (coords_in_orig[len(St)-1],coords_in_tile[len(St)-1]))
    PrintLine('int32_t output_index_offset = epoch*UNROLL_FACTOR+unroll_index;')
    PrintLine()

    for line in compute_content:
        if len(line) > 0:
            PrintLine(line.replace('\n','').replace('\r',''))
    PrintLine()

    PrintLine('%s += UNROLL_FACTOR;' % coords_in_tile[0])
    if(len(St)>1):
        PrintLine('if(%s>=TILE_SIZE_DIM_0)' % coords_in_tile[0])
        PrintLine('{');indent += 1
        PrintLine('%s -= TILE_SIZE_DIM_0;' % coords_in_tile[0])
        PrintLine('++%s;' % coords_in_tile[1])
        if(len(St)>2):
            PrintLine('if(%s>=TILE_SIZE_DIM_1)' % coords_in_tile[1])
            PrintLine('{');indent += 1
            PrintLine('%s -= TILE_SIZE_DIM_1;' % coords_in_tile[1])
            PrintLine('++%s;' % coords_in_tile[2])
            if(len(St)>3):
                PrintLine('if(%s>=TILE_SIZE_DIM_2)' % coords_in_tile[2])
                PrintLine('{');indent += 1
                PrintLine('%s -= TILE_SIZE_DIM_2;' % coords_in_tile[2])
                PrintLine('++%s;' % coords_in_tile[3])
                indent -= 1;PrintLine('}')
            indent -= 1;PrintLine('}')
        indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('} // for unroll_index')
    PrintLine()

    PrintLine('compute_finalize_channel:', 0)
    PrintLine('for(int32_t c = 0; c<CHANNEL_NUM_I; ++c)')
    PrintLine('{')
    indent += 1
    PrintLine('#pragma HLS unroll', 0)
    first = True
    for idx, item in enumerate(buf['inputs']):
        if first:
            first = False
        else:
            PrintLine()
        PrintUpdate(GetProduceMap(buf), GetConsumeMap(buf), {'inputs':{'index':idx, 'produce':item}})
    indent -= 1
    PrintLine('}')
    PrintLine()
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('FIFO_%d_ptr = FIFO_%d_ptr==(%d-1) ? 0 : FIFO_%d_ptr+1;' % (fifo_length/k, fifo_length/k, fifo_length/k, fifo_length/k))
    indent -= 1
    PrintLine('}')
    indent -= 1
    PrintLine('}')
    indent -= 1
    PrintLine('}')

def PrintKernel(St, A, k, app_name, extra_params):
    global indent
    buf = GetBuffer(St, A, k)
    points = GetPoints(A, k)
    PrintLine('extern "C"')
    PrintLine('{')
    PrintLine()
    PrintLine('void %s_kernel(ap_uint<BURST_WIDTH>* var_output,' % app_name)
    indent += 1
    PrintLine('ap_uint<BURST_WIDTH>* var_input,')
    if extra_params:
        for param_name, param in extra_params.items():
            PrintLine('%s* var_%s,' % (param['type'], param_name))
    for i in range(0, len(St)-1):
        PrintLine('int32_t tile_num_dim_%d,' % i)
    PrintLine('int32_t input_size_dim_%d,' % (len(St)-1))
    PrintLine('int64_t tile_burst_num,')
    PrintLine('int64_t extra_space_i_coalesed,')
    PrintLine('int64_t extra_space_o_coalesed,')
    PrintLine('int32_t total_burst_num)')
    indent -= 1
    PrintLine('{')
    indent += 1

    PrintLine('#pragma HLS INTERFACE m_axi port=var_output offset=slave depth=65536 bundle=gmem1 latency=120', 0)
    PrintLine('#pragma HLS INTERFACE m_axi port=var_input offset=slave depth=65536 bundle=gmem2 latency=120', 0)
    if extra_params:
        for param_name, param in extra_params.items():
            PrintLine('#pragma HLS INTERFACE m_axi port=var_%s offset=slave depth=%d bundle=gmem3 latency=120' % (param_name, param['length']), 0)
    PrintLine()
    PrintLine('#pragma HLS INTERFACE s_axilite port=var_output bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=var_input bundle=control', 0)
    if extra_params:
        for param_name in extra_params.keys():
            PrintLine('#pragma HLS INTERFACE s_axilite port=var_%s bundle=control' % param_name, 0)
    for i in range(0, len(St)-1):
        PrintLine('#pragma HLS INTERFACE s_axilite port=tile_num_dim_%d bundle=control' % i, 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=input_size_dim_1 bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=tile_burst_num bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=extra_space_i_coalesed bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=extra_space_o_coalesed bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=total_burst_num bundle=control', 0)
    PrintLine('#pragma HLS INTERFACE s_axilite port=return bundle=control', 0)
    PrintLine()

    PrintLine('input_type  input_0[CHANNEL_NUM_I][BURST_LENGTH];')
    PrintLine('input_type  input_1[CHANNEL_NUM_I][BURST_LENGTH];')
    PrintLine('output_type output_0[CHANNEL_NUM_O][BURST_LENGTH];')
    PrintLine('output_type output_1[CHANNEL_NUM_O][BURST_LENGTH];')
    PrintLine('input_type FF[CHANNEL_NUM_I][%d];' % len(buf['FFs']))
    for fifo_length, fifo_list in buf['FIFOs'].items():
        PrintLine('input_type FIFO_%d[CHANNEL_NUM_I][%d][%d];' % (fifo_length/k, len(fifo_list), fifo_length/k) )
    PrintLine('#pragma HLS array_partition variable=input_0 complete dim=1', 0)
    PrintLine('#pragma HLS array_partition variable=input_0 cyclic factor=%d dim=2' % input_partition, 0)
    PrintLine('#pragma HLS array_partition variable=input_1 complete dim=1', 0)
    PrintLine('#pragma HLS array_partition variable=input_1 cyclic factor=%d dim=2' % input_partition, 0)
    PrintLine('#pragma HLS array_partition variable=output_0 complete dim=1', 0)
    PrintLine('#pragma HLS array_partition variable=output_0 cyclic factor=%d dim=2' % output_partition, 0)
    PrintLine('#pragma HLS array_partition variable=output_1 complete dim=1', 0)
    PrintLine('#pragma HLS array_partition variable=output_1 cyclic factor=%d dim=2' % output_partition, 0)
    PrintLine('#pragma HLS array_partition variable=FF complete dim=0', 0)
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=1' % (fifo_length/k), 0)
        PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=2' % (fifo_length/k), 0)
    PrintLine()

    for i in range(0, len(St)-1):
        PrintLine('int32_t tile_index_dim_%d = 0;' % i)
    PrintLine('bool    load_flag;')
    PrintLine('bool compute_flag;')
    PrintLine('bool   store_flag;')
    PrintLine('int32_t burst_index_load = 0;')
    PrintLine('int32_t burst_index_compute = 0;')
    PrintLine('int32_t burst_index_store = 0;')
    PrintLine('int32_t FIFO_ptrs[%d] = {0};' % len(buf['FIFOs']))
    PrintLine('int32_t input_index_base = 0;')
    PrintLine()

    for i in range(0, len(St)):
        PrintLine('int32_t %s_base[UNROLL_FACTOR];' % coords_in_tile[i])
    for i in range(0, len(St)-1):
        PrintLine('int32_t %s_base = 0;' % coords_in_orig[i])
    PrintLine()

    PrintLine('#pragma HLS array_partition variable=FIFO_ptrs complete', 0)
    for i in range(0, len(St)):
        PrintLine('#pragma HLS array_partition variable=%s_base complete' % coords_in_tile[i], 0)
    PrintLine()

    PrintLine('bases_init:', 0)
    PrintLine('for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
    PrintLine('{');indent+=1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('i_base[unroll_index] = unroll_index - STENCIL_DISTANCE;')
    for i in range(1, len(St)):
        PrintLine('%s_base[unroll_index] = 0;' % coords_in_tile[i])
    indent-=1;PrintLine('}')
    PrintLine()

    if extra_params:
        for param_name, param in extra_params.items():
            if 'dup' in param:
                dup = ('[%d]' % param['dup'])
            else:
                dup = ''
            PrintLine('%s %s%s[UNROLL_FACTOR][%d];' % (param['type'], param_name, dup, param['length']))
        PrintLine()

        for param_name, param in extra_params.items():
            if 'dup' in param:
                partition_dim = 3
                PrintLine('#pragma HLS array_partition variable=%s complete dim=1' % param_name, 0)
            else:
                partition_dim = 2
            PrintLine('#pragma HLS array_partition variable=%s complete dim=%d' % (param_name, partition_dim-1), 0)
            if 'partition' in param:
                PrintLine('#pragma HLS array_partition variable=%s %s dim=%d' % (param_name, param['partition'], partition_dim), 0)
        PrintLine()

        PrintLine('extra_params_unrolled:', 0)
        PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
        PrintLine('{');indent += 1
        PrintLine('#pragma HLS unroll',0)
        for param_name, param in extra_params.items():
            PrintLine('%s_init:' % param_name, 0)
            PrintLine('for(int %s_index = 0; %s_index < %d; ++%s_index)' % (param_name, param_name, param['length'], param_name))
            PrintLine('{');indent += 1
            PrintLine('#pragma HLS pipeline II=1', 0)
            if 'dup' in param:
                for i in range(0, param['dup']):
                    PrintLine('%s[%d][unroll_index][%s_index] = var_%s[%s_index];' % ((param_name, i)+(param_name,)*3))
            else:
                PrintLine('%s[unroll_index][%s_index] = var_%s[%s_index];' % ((param_name,)*4))
            indent -= 1;PrintLine('}')
        indent -= 1;PrintLine('}')
        PrintLine()

    extra_params_str = ''.join([var+', ' for var in extra_params.keys()]) if extra_params else ''
    coords_str = ''.join([coords_in_tile[i]+'_base, ' for i in range(0, len(St))]) + ''.join([coords_in_orig[i]+'_base, ' for i in range(0, len(St)-1)])

    PrintLine('burst:', 0)
    PrintLine('for(int32_t burst_index_in_total = 0; burst_index_in_total < total_burst_num+2; ++burst_index_in_total)')
    PrintLine('{');indent += 1
    PrintLine('load_flag = burst_index_in_total < total_burst_num;')
    PrintLine('compute_flag = burst_index_in_total > 0 && burst_index_in_total < total_burst_num+1;')
    PrintLine('store_flag = burst_index_in_total > 1;')
    PrintLine('if(burst_index_in_total%2==0)')
    PrintLine('{');indent += 1
    PrintLine('load(load_flag, input_0, var_input);')
    PrintLine('compute(compute_flag, output_1, input_1, '+extra_params_str+'FF, '+''.join([('FIFO_%d, ' % (fifo_length/k)) for fifo_length in buf['FIFOs'].keys()])+'FIFO_ptrs, '+coords_str+'input_index_base);')
    PrintLine('store(store_flag, var_output, output_0);')
    indent -= 1;PrintLine('}')
    PrintLine('else')
    PrintLine('{');indent += 1
    PrintLine('load(load_flag, input_1, var_input);')
    PrintLine('compute(compute_flag, output_0, input_0, '+extra_params_str+'FF, '+''.join([('FIFO_%d, ' % (fifo_length/k)) for fifo_length in buf['FIFOs'].keys()])+'FIFO_ptrs, '+coords_str+'input_index_base);')
    PrintLine('store(store_flag, var_output, output_1);')
    indent -= 1;PrintLine('}')

    PrintLine('if(load_flag)')
    PrintLine('{');indent += 1
    PrintLine('var_input += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I)*CHANNEL_NUM_I;')
    PrintLine('burst_index_load += 1;')
    PrintLine('if(burst_index_load == tile_burst_num)')
    PrintLine('{');indent += 1
    PrintLine('burst_index_load = 0;')
    PrintLine('var_input -= extra_space_i_coalesed;')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

    PrintLine('if(compute_flag)')
    PrintLine('{');indent += 1
    PrintLine('burst_index_compute += 1;')
    PrintLine('input_index_base  += BURST_LENGTH/UNROLL_FACTOR;')
    PrintLine('if(burst_index_compute == tile_burst_num)')
    PrintLine('{');indent += 1
    PrintLine('burst_index_compute = 0;')
    PrintLine('input_index_base = 0;')
    PrintLine('tile_index_dim_0 += 1;')
    PrintLine('%s_base += (TILE_SIZE_DIM_0-STENCIL_DIM_0+1);' % coords_in_orig[0])
    PrintLine('if(tile_index_dim_0==tile_num_dim_0)')
    PrintLine('{');indent += 1
    PrintLine('tile_index_dim_0 = 0;')
    if(len(St)>2):
        PrintLine('tile_index_dim_1 += 1;')
        PrintLine('if(tile_index_dim_1==tile_num_dim_1)')
        PrintLine('{');indent += 1
        PrintLine('tile_index_dim_1 = 0;')
        if(len(St)>3):
            PrintLine('tile_index_dim_2 += 2;')
            PrintLine('if(tile_index_dim_2==tile_num_dim_2)')
            PrintLine('{');indent += 1
            PrintLine('tile_index_dim_2 = 0;')
            indent -= 1;PrintLine('}')
        indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

    PrintLine('if(store_flag)')
    PrintLine('{');indent += 1
    PrintLine('var_output += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O)*CHANNEL_NUM_O;')
    PrintLine('burst_index_store += 1;')
    PrintLine('if(burst_index_store == tile_burst_num)')
    PrintLine('{');indent += 1
    PrintLine('burst_index_store = 0;')
    PrintLine('var_output -= extra_space_o_coalesed;')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    PrintLine()
    PrintLine('}//extern "C"')

def PrintHeader(app_name):
    for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int']:
        PrintLine('#include<%s.h>' % header)
    PrintLine()
    PrintLine('#include"%s_params.h"' % app_name)
    PrintLine()

def PrintLoad():
    global indent
    PrintLine('void load(bool load_flag, input_type to[CHANNEL_NUM_I][BURST_LENGTH], ap_uint<BURST_WIDTH>* from)')
    PrintLine('{');indent += 1
    PrintLine('if(load_flag)')
    PrintLine('{');indent += 1
    PrintLine('load_channel:', 0)
    PrintLine('for(int c = 0; c < CHANNEL_NUM_I; ++c)')
    PrintLine('{');indent += 1
    PrintLine('load_epoch:', 0)
    PrintLine('for(int i = 0; i < BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I); ++i)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS pipeline II=1', 0)
    PrintLine('ap_uint<BURST_WIDTH> tmp(from[c*(BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O))+i]);')
    PrintLine('load_coalesced:', 0)
    PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_I; ++j)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('to[c][i*BURST_WIDTH/PIXEL_WIDTH_I+j] = tmp((j+1)*PIXEL_WIDTH_I-1, j*PIXEL_WIDTH_I);')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

def PrintStore():
    global indent
    PrintLine('void store(bool store_flag, ap_uint<BURST_WIDTH>* to, output_type from[CHANNEL_NUM_O][BURST_LENGTH])')
    PrintLine('{');indent += 1
    PrintLine('if(store_flag)')
    PrintLine('{');indent += 1
    PrintLine('store_channel:', 0)
    PrintLine('for(int c = 0; c < CHANNEL_NUM_O; ++c)')
    PrintLine('{');indent += 1
    PrintLine('store_epoch:', 0)
    PrintLine('for(int i = 0; i < BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O); ++i)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS pipeline II=1', 0)
    PrintLine('ap_uint<BURST_WIDTH> tmp;')
    PrintLine('store_coalesced:', 0)
    PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_O; ++j)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('tmp((j+1)*PIXEL_WIDTH_O-1, j*PIXEL_WIDTH_O) = from[c][i*BURST_WIDTH/PIXEL_WIDTH_O+j];')
    indent -= 1;PrintLine('}')
    PrintLine('to[c*(BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O))+i] = tmp;')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

def PrintGuard(var, val):
    PrintLine('#if %s != %d' % (var, val))
    PrintLine('#error %s != %d' % (var, val))
    PrintLine('#endif//%s != %d' % (var, val))

type_width = {'uint8_t':8, 'uint16_t':16, 'uint32_t':32, 'uint64_t':64, 'int8_t':8, 'int16_t':16, 'int32_t':32, 'int64_t':64}
config = json.loads(sys.stdin.read())
input_type = config['input_type']
output_type = config['output_type']
app_name = config['app_name']
St = config.get('St', [0]*config['dim'])
A = config['A']
k = int(os.environ.get('UNROLL_FACTOR', config['k']))
burst_width = config['burst_width']
pixel_width_i = type_width[input_type]
pixel_width_o = type_width[output_type]
input_partition = burst_width/pixel_width_i if burst_width/pixel_width_i > k else k
output_partition = burst_width/pixel_width_o if burst_width/pixel_width_o > k else k
compute_content = config['compute_content']
extra_params = config.get('extra_params', None)
i = 0
while (('TILE_SIZE_DIM_%d' % i) in os.environ):
    St[i] = int(os.environ[('TILE_SIZE_DIM_%d' % i)])
    i += 1

indent = 0
PrintHeader(app_name)
PrintLine('typedef %s input_type;' % input_type)
PrintLine('typedef %s output_type;' % output_type)
PrintGuard('UNROLL_FACTOR', k)
for i in range(0, len(St)-1):
    PrintGuard('TILE_SIZE_DIM_%d' % i, St[i])
PrintLine()
PrintLoad()
PrintLine()
PrintStore()
PrintLine()
PrintCompute(St, A, k, compute_content, input_partition, output_partition, extra_params)
PrintLine()
PrintKernel(St, A, k, app_name, extra_params)

