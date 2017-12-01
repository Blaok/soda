#!/usr/bin/python3.6
import functools
import json
import math
import operator
import os
import sys
from fractions import Fraction
from functools import reduce

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

def PrintUpdate(producer_map, consumer_map, src, input_type):
    for c in range(0, input_type[1]):
        if 'FIFOs' in src:
            dst = consumer_map.get(src['FIFOs']['produce'])
            if not dst:
                return
            fifo_length = src['FIFOs']['length']/producer_map['k']
            fifo_index = src['FIFOs']['index']
            src_str = 'FIFO_%d_%d_%d' % (fifo_length, c, fifo_index)
            #src_str = 'FIFO_%d[%d][%d][FIFO_%d_ptr]' % (fifo_length, c, fifo_index, fifo_length)
        elif 'FFs' in src:
            dst = consumer_map.get(src['FFs']['produce'])
            if not dst:
                return
            src_str = 'FF[%d][%d]' % (c, src['FFs']['index'])
        else:
            dst = consumer_map.get(src['inputs']['produce'])
            if not dst:
                return
            src_str = 'input_buffer[%d][%d]' % (c, src['inputs']['index'])

        PrintUpdate(producer_map, consumer_map, dst, input_type)

        if 'FIFOs' in dst:
            fifo_length = dst['FIFOs']['length']/consumer_map['k']
            fifo_index = dst['FIFOs']['index']
            dst_str = 'FIFO_%d[%d][%d][FIFO_%d_ptr]' % (fifo_length, c, fifo_index, fifo_length)
        elif 'FFs' in dst:
            dst_str = 'FF[%d][%d]' % (c, dst['FFs']['index'])
        else:
            return

        PrintLine('%s = %s;' % (dst_str, src_str))

def PrintCompute(St, A, k, compute_content, input_partition, output_partition, extra_params, input_type, dram_chan):
    global indent
    buf = GetBuffer(St, A, k)
    points = GetPoints(A, k)

    PrintLine('void compute(')
    indent += 1
    for i in range(0, dram_chan):
        PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& to_%d,' % i)
    for i in range(0, dram_chan):
        PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& from_%d,' % i)
    if extra_params:
        for param_name, param in extra_params.items():
            if 'dup' in param:
                dup = ('[%d]' % param['dup'])
            else:
                dup = ''
            PrintLine('%s %s%s[UNROLL_FACTOR][%d],' % (param['type'], param_name, dup, param['length']))
    PrintLine('int64_t coalesced_data_num,')
    PrintLine('int64_t tile_data_num,')
    for i in range(0, len(St)-1):
        PrintLine('int32_t tile_num_dim_%d,' % i)
    PrintLine('int32_t input_size_dim_%d)' % (len(St)-1))
    indent -= 1
    PrintLine('{')
    indent += 1

    for i in range(0, len(St)-1):
        PrintLine('int32_t tile_index_dim_%d = 0;' % i)
    PrintLine()

    PrintLine('int32_t i_base[UNROLL_FACTOR] = {%s};' % (', '.join(['%d-STENCIL_DISTANCE' % x for x in range(0, k)])))
    for i in range(1, len(St)):
        PrintLine('int32_t %c_base[UNROLL_FACTOR] = {0};' % coords_in_tile[i])
    for i in range(0, len(St)-1):
        PrintLine('int32_t %c_base = 0;' % coords_in_orig[i])
    for i in range(0, len(St)):
        PrintLine('#pragma HLS array_partition variable=%s_base complete' % coords_in_tile[i], 0)
    PrintLine()

    # array declaration
    if len(buf['FFs'])>0:
        PrintLine('input_type FF[CHANNEL_NUM_I][%d];' % len(buf['FFs']))

    for fifo_length, fifo_list in buf['FIFOs'].items():
        PrintLine('input_type FIFO_%d[CHANNEL_NUM_I][%d][%d];' % (fifo_length/k, len(fifo_list), fifo_length/k))

    if len(buf['FFs'])>0:
        PrintLine('#pragma HLS array_partition variable=FF complete dim=0', 0)

    for fifo_length in buf['FIFOs'].keys():
        PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=1' % (fifo_length/k), 0)
        PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=2' % (fifo_length/k), 0)
    PrintLine()

    for fifo_length in buf['FIFOs'].keys():
        PrintLine('int%d_t FIFO_%d_ptr = 0;' % (2**math.ceil(math.log2(math.log2(fifo_length/k))), fifo_length/k))
    PrintLine()

    PrintLine('input_type input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d];' % len(A))
    for idx, item in enumerate(A):
        PrintLine('//         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d] <=> (%s)' % (idx, str(item)[1:-1]))
    PrintLine('input_type input_buffer[CHANNEL_NUM_I][UNROLL_FACTOR*(BURST_WIDTH*%d/PIXEL_WIDTH_I/UNROLL_FACTOR)];' % dram_chan)
    PrintLine('output_type output_buffer[CHANNEL_NUM_O][UNROLL_FACTOR*(BURST_WIDTH*%d/PIXEL_WIDTH_O/UNROLL_FACTOR)];' % dram_chan)
    PrintLine('#pragma HLS array_partition variable=input_points complete dim=0', 0)
    PrintLine('#pragma HLS array_partition variable=input_buffer complete dim=0', 0)
    PrintLine('#pragma HLS array_partition variable=output_buffer complete dim=0', 0)
    PrintLine()

    for i in range(0, len(St)-1):
        PrintLine('int32_t %c_base_counter = 0;' % coords_in_orig[i])
    PrintLine()

    PrintLine('// produce output')
    PrintLine('compute_epoch:', 0)
    PrintLine('for(int32_t epoch = 0; epoch < coalesced_data_num*(BURST_WIDTH*%d/PIXEL_WIDTH_I/UNROLL_FACTOR); ++epoch)' % dram_chan)
    PrintLine('{')
    indent += 1
    if len(buf['FFs'])>0:
        PrintLine('#pragma HLS dependence variable=FF inter false', 0)
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('#pragma HLS dependence variable=FIFO_%d inter false' % (fifo_length/k), 0)
    PrintLine('#pragma HLS pipeline II=1', 0)

    PrintLine('switch(epoch&%d)' % (burst_width*dram_chan/pixel_width_i/k-1))
    PrintLine('{'); indent += 1
    PrintLine('case 0:')
    PrintLine('{'); indent += 1
    for i in range(0, dram_chan):
        PrintLine('ap_uint<BURST_WIDTH> tmp_%d;' % i)
    for i in range(0, dram_chan):
        PrintLine('from_%d>>tmp_%d;' % (i, i))
    PrintLine('load_coalesced:', 0)
    PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_I; ++j)')
    PrintLine('{'); indent += 1
    PrintLine('#pragma HLS unroll', 0)
    for i in range(0, dram_chan):
        PrintLine('input_buffer[0][j*%d+%d] = tmp_%d((j+1)*PIXEL_WIDTH_I-1, j*PIXEL_WIDTH_I);' % (dram_chan, i, i))
    indent -= 1; PrintLine('}')
    PrintLine('break;')
    indent -= 1; PrintLine('}')
    PrintLine('default:')
    PrintLine('{'); indent += 1
    PrintLine('load_shift:', 0)
    PrintLine('for(int j = 0; j < UNROLL_FACTOR; ++j)')
    PrintLine('{'); indent += 1
    PrintLine('#pragma HLS unroll', 0)
    for i in range(0, int(burst_width/pixel_width_i*dram_chan/k-1)):
        PrintLine('input_buffer[0][j+UNROLL_FACTOR*%d] = input_buffer[0][j+UNROLL_FACTOR*%d];' % (i, i+1))
    indent -= 1; PrintLine('}')
    indent -= 1; PrintLine('}')
    indent -= 1; PrintLine('}')
    PrintLine()

    for c in range(0, input_type[1]):
        for fifo_length in buf['FIFOs'].keys():
            for idx, item in enumerate(buf['FIFOs'][fifo_length]):
                PrintLine('input_type FIFO_%d_%d_%d = FIFO_%d[%d][%d][FIFO_%d_ptr];'% (fifo_length/k, c, idx, fifo_length/k, c, idx, fifo_length/k))
    PrintLine()

    for c in range(0, input_type[1]):
        for idx, item in enumerate(buf['inputs']):
            for unroll_index in points[item].keys():
                PrintLine("input_points[%d][%d][%d] = input_buffer[%d][%d]; // (%s)" % (c, unroll_index, points[item][unroll_index], c, idx, str(A[points[item][unroll_index]])[1:-1]))
        for idx, item in enumerate(buf['FFs']):
            for unroll_index in points[item].keys():
                PrintLine("input_points[%d][%d][%d] = FF[%d][%d]; // (%s)" % (c, unroll_index, points[item][unroll_index], c, idx, str(A[points[item][unroll_index]])[1:-1]))
        for fifo_length in buf['FIFOs'].keys():
            for idx, item in enumerate(buf['FIFOs'][fifo_length]):
                for unroll_index in points[item].keys():
                    PrintLine("input_points[%d][%d][%d] = FIFO_%d_%d_%d; // (%s)" % (c, unroll_index, points[item][unroll_index], fifo_length/k, c, idx, str(A[points[item][unroll_index]])[1:-1]))

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
    PrintLine()
    for idx, item in enumerate(A):
        PrintLine('input_type input_%s = input_points[0][unroll_index][%d];' % ('_'.join([str(x) for x in item]), idx))
    PrintLine()

    for line in compute_content:
        if len(line) > 0:
            PrintLine(line.replace('\n','').replace('\r',''))
    PrintLine('switch(epoch&%d)' % (burst_width*dram_chan/pixel_width_i/k-1))
    PrintLine('{'); indent += 1
    for i in range(0, int(burst_width*dram_chan/pixel_width_i/k)):
        PrintLine('case %d:' % i)
        PrintLine('{'); indent += 1
        PrintLine('output_buffer[0][unroll_index+UNROLL_FACTOR*%d] = result;' % i)
        indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')
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

    for c in range(0, input_type[1]):
        first = True
        for idx, item in enumerate(buf['inputs']):
            if first:
                first = False
            else:
                PrintLine()
            PrintUpdate(GetProduceMap(buf), GetConsumeMap(buf), {'inputs':{'index':idx, 'produce':item}}, input_type)
    PrintLine()

    for fifo_length in buf['FIFOs'].keys():
        PrintLine('FIFO_%d_ptr = FIFO_%d_ptr==(%d-1) ? 0 : FIFO_%d_ptr+1;' % (fifo_length/k, fifo_length/k, fifo_length/k, fifo_length/k))
    PrintLine()

    PrintLine('if((epoch&%d)==%d)' % ((burst_width*dram_chan/pixel_width_i/k-1), burst_width*dram_chan/pixel_width_i/k-1))
    PrintLine('{'); indent += 1
    PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join(['tmp_%d' % x for x in range(0, dram_chan)])))
    PrintLine('store_coalesced:', 0)
    PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_O; ++j)')
    PrintLine('{'); indent += 1
    PrintLine('#pragma HLS unroll', 0)
    for i in range(0, dram_chan):
        PrintLine('tmp_%d((j+1)*PIXEL_WIDTH_O-1, j*PIXEL_WIDTH_O) = output_buffer[0][j*%d+%d];' % (i, dram_chan, i))
    indent -= 1; PrintLine('}')
    for i in range(0, dram_chan):
        PrintLine('to_%d<<tmp_%d;' % (i, i))
    indent -= 1; PrintLine('}')
    PrintLine()

    PrintLine('++p_base_counter;')
    PrintLine('if(p_base_counter == tile_data_num)')
    PrintLine('{'); indent += 1
    PrintLine('p_base_counter = 0;')
    PrintLine('reset_bases:', 0)
    PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
    PrintLine('{'); indent += 1
    PrintLine('#pragma HLS unroll', 0)
    PrintLine('i_base[unroll_index] = unroll_index-STENCIL_DISTANCE;')
    PrintLine('j_base[unroll_index] = 0;')
    indent -= 1; PrintLine('}')
    PrintLine('p_base += TILE_SIZE_DIM_0-STENCIL_DIM_0+1;')
    indent -= 1; PrintLine('}')
    indent -= 1; PrintLine('}')
    indent -= 1; PrintLine('}')

def PrintKernel(St, A, k, app_name, extra_params, dram_chan, dram_separate):
    global indent
    buf = GetBuffer(St, A, k)
    points = GetPoints(A, k)
    PrintLine('extern "C"')
    PrintLine('{')
    PrintLine()
    PrintLine('void %s_kernel(' % app_name)
    indent += 1
    for i in range(0, dram_chan):
        PrintLine('ap_uint<BURST_WIDTH>* var_output_%d,' % i)
    for i in range(0, dram_chan):
        PrintLine('ap_uint<BURST_WIDTH>* var_input_%d,' % i)
    if extra_params:
        for param_name, param in extra_params.items():
            PrintLine('%s* var_%s,' % (param['type'], param_name))
    PrintLine('int64_t coalesced_data_num,')
    PrintLine('int64_t tile_data_num,')
    for i in range(0, len(St)-1):
        PrintLine('int32_t tile_num_dim_%d,' % i)
    PrintLine('int32_t input_size_dim_%d)' % (len(St)-1))
    indent -= 1
    PrintLine('{')
    indent += 1

    chan = 0
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS interface m_axi port=var_output_%d offset=slave depth=65536 bundle=gmem%d latency=120' % (i, chan), 0)
        chan += 1
    if not dram_separate:
        chan = 0
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS interface m_axi port=var_input_%d offset=slave depth=65536 bundle=gmem%d latency=120' % (i, chan), 0)
        chan += 1
    if extra_params:
        for param_name, param in extra_params.items():
            PrintLine('#pragma HLS interface m_axi port=var_%s offset=slave depth=%d bundle=gmem0 latency=120' % (param_name, param['length']), 0)
    PrintLine()
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS interface s_axilite port=var_output_%d bundle=control' % i, 0)
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS interface s_axilite port=var_input_%d bundle=control' % i, 0)
    if extra_params:
        for param_name in extra_params.keys():
            PrintLine('#pragma HLS interface s_axilite port=var_%s bundle=control' % param_name, 0)
    PrintLine('#pragma HLS interface s_axilite port=coalesced_data_num bundle=control', 0)
    PrintLine('#pragma HLS interface s_axilite port=tile_data_num bundle=control', 0)
    for i in range(0, len(St)-1):
        PrintLine('#pragma HLS interface s_axilite port=tile_num_dim_%d bundle=control' % i, 0)
    PrintLine('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % (len(St)-1), 0)
    PrintLine('#pragma HLS interface s_axilite port=return bundle=control', 0)
    PrintLine()

    for i in range(0, dram_chan):
        PrintLine('hls::stream<ap_uint<BURST_WIDTH> >  input_stream_%d( "input_stream_%d");' % (i, i))
    for i in range(0, dram_chan):
        PrintLine('hls::stream<ap_uint<BURST_WIDTH> > output_stream_%d("output_stream_%d");' % (i, i))
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS stream variable=input_stream_%d depth=32' % i, 0)
    for i in range(0, dram_chan):
        PrintLine('#pragma HLS stream variable=output_stream_%d depth=32' % i, 0)
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

    PrintLine('#pragma HLS dataflow', 0)
    for i in range(0, dram_chan):
        PrintLine('load(input_stream_%d, var_input_%d, coalesced_data_num);' % (i, i))
    output_stream = ', '.join(['output_stream_%d' % x for x in range(0, dram_chan)])
    input_stream = ', '.join(['input_stream_%d' % x for x in range(0, dram_chan)])
    tile_num_dim = ', '.join(['tile_num_dim_%d' % x for x in range(0, len(St)-1)])
    PrintLine('compute(%s, %s, coalesced_data_num, tile_data_num, %s, input_size_dim_%d);' % (output_stream, input_stream, tile_num_dim, len(St)-1))
    for i in range(0, dram_chan):
        PrintLine('store(var_output_%d, output_stream_%d, coalesced_data_num);' % (i, i))

    indent -= 1;PrintLine('}')
    PrintLine()
    PrintLine('}//extern "C"')

def PrintHeader(app_name):
    for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int', 'hls_stream']:
        PrintLine('#include<%s.h>' % header)
    PrintLine()
    PrintLine('#include"%s_params.h"' % app_name)
    PrintLine()

def PrintLoad(input_type):
    global indent
    PrintLine('void load(hls::stream<ap_uint<BURST_WIDTH> > &to, ap_uint<BURST_WIDTH>* from, int data_num)')
    PrintLine('{');indent += 1
    PrintLine('load_epoch:', 0)
    PrintLine('for(int i = 0; i < data_num; ++i)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS pipeline II=1', 0)
    PrintLine('to<<from[i];')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

def PrintStore(output_type):
    global indent
    PrintLine('void store(ap_uint<BURST_WIDTH>* to, hls::stream<ap_uint<BURST_WIDTH> > &from, int data_num)')
    PrintLine('{');indent += 1
    PrintLine('store_epoch:', 0)
    PrintLine('for(int i = 0; i < data_num; ++i)')
    PrintLine('{');indent += 1
    PrintLine('#pragma HLS pipeline II=1', 0)
    PrintLine('from>>to[i];')
    indent -= 1;PrintLine('}')
    indent -= 1;PrintLine('}')

def PrintGuard(var, val):
    PrintLine('#if %s != %d' % (var, val))
    PrintLine('#error %s != %d' % (var, val))
    PrintLine('#endif//%s != %d' % (var, val))

# for generating iterative
def GetA(A,  num_iter):
    if num_iter < 1:
        return None
    if num_iter == 1:
        return A
    A1 = GetA(A, num_iter-1)
    A2 = {}
    for a1 in A1.values():
        for a in A.values():
            a2 = [o1+o2 for o1, o2 in zip(a[0], a1[0])]
            A2[str(a2)] = (a2, A2.get(str(a2), (a2, 0))[1]+a[1]*a1[1])
    return A2

def gcd(*numbers):
    from fractions import gcd
    return reduce(gcd, numbers)

def lcm(*numbers):
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

def GetAdderTree(variables, count):
    global compute_content
    if len(variables)>1:
        r1 = GetAdderTree(variables[0:int(len(variables)/2)], count)
        r2 = GetAdderTree(variables[int(len(variables)/2):], r1[1])
        compute_content += ['float tmp_adder_%d = %s+%s;' % (r2[1], r1[0], r2[0])]
        return ('tmp_adder_%d' % r2[1], r2[1]+1)
    return (variables[0], count)

type_width = {'uint8_t':8, 'uint16_t':16, 'uint32_t':32, 'uint64_t':64, 'int8_t':8, 'int16_t':16, 'int32_t':32, 'int64_t':64, 'float':32, 'double':64}
config = json.loads(sys.stdin.read())
input_type = config['input_type']
output_type = config['output_type']
dram_chan = int(os.environ.get('DRAM_CHAN', config['dram_chan']))
dram_separate = ('DRAM_SEPARATE' in os.environ) or ('dram_separate' in config and config['dram_separate'])
if dram_separate:
    dram_chan = int(dram_chan/2)
app_name = config['app_name']
St = config.get('St', [0]*config['dim'])
A = config['A']
k = int(os.environ.get('UNROLL_FACTOR', config['k']))
burst_width = config['burst_width']
pixel_width_i = type_width[input_type[0]]
pixel_width_o = type_width[output_type[0]]
input_partition = burst_width/pixel_width_i*dram_chan/2 if burst_width/pixel_width_i*dram_chan/2 > k/2 else k/2
output_partition = burst_width/pixel_width_o*dram_chan/2 if burst_width/pixel_width_o*dram_chan/2 > k/2 else k/2

# generate compute content if it is not present in the config but iter and coeff are
compute_content = []
if 'compute_content' in config:
    compute_content = config['compute_content']
elif ('iter' in config or 'ITER' in os.environ) and 'coeff' in config:
    num_iter = int(os.environ.get('ITER', config['iter']))
    coeff = [Fraction(x) for x in config['coeff']]
    if len(A) != len(coeff):
        sys.stderr.write('A and coeff must have the same length\n')
        sys.exit(1)
    A = {str(a):(a, c) for a, c in zip(A, coeff)}
    A = (GetA(A, num_iter))
    final_var = GetAdderTree(['input_%s*%ff' % ('_'.join([str(x) for x in a[0]]), (1.*a[1].numerator)/a[1].denominator) for a in A.values()], 0)
    compute_content += ['output_type output_point = %s;' % (final_var[0],), 'output[0][output_index_offset] = output_point;' ,'']
    A = [a[0] for a in A.values()]

extra_params = config.get('extra_params', None)
i = 0
while (('TILE_SIZE_DIM_%d' % i) in os.environ):
    St[i] = int(os.environ[('TILE_SIZE_DIM_%d' % i)])
    i += 1

indent = 0
PrintHeader(app_name)
PrintLine('typedef %s input_type;' % input_type[0])
PrintLine('typedef %s output_type;' % output_type[0])
PrintGuard('UNROLL_FACTOR', k)
for i in range(0, len(St)-1):
    PrintGuard('TILE_SIZE_DIM_%d' % i, St[i])
PrintLine()
PrintLoad(input_type[0])
PrintLine()
PrintStore(output_type[0])
PrintLine()
PrintCompute(St, A, k, compute_content, input_partition, output_partition, extra_params, input_type, dram_chan)
PrintLine()
PrintKernel(St, A, k, app_name, extra_params, dram_chan, dram_separate)

