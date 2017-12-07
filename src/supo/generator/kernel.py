#!/usr/bin/python3.6
import json
import math
import operator
import os
import sys
from fractions import Fraction
from functools import reduce
from supo.generator.utils import coords_in_tile, coords_in_orig, type_width, Stencil, Printer

# serialize
def Serialize(vec, tile_size):
    # convert vec to scalar coordinates
    result = vec[0]
    for i in range(1, len(tile_size)):
        result += vec[i]*reduce(operator.mul, tile_size[0:i])
    return result

def GetChains(tile_size, A, k):
    A_dag = set()
    for i in range(0, k):
        A_dag = A_dag.union([x+i for x in [Serialize(x, tile_size) for x in A]])
    A_dag = list(A_dag)
    A_dag.sort()
    A_dag = [x for x in A_dag]
    chains = []
    for i in range(0, k):
        chains += [[x for x in A_dag if x%k == i]]
    return chains

def GetPoints(tile_size, A, k):
    all_points = {} # {offset:{unroll_index:point_index}}
    for i in range(0, k):
        points = [Serialize(x, tile_size) for x in A]
        for idx, j in enumerate(points):
            all_points[j+i] = all_points.get(j+i, {})
            all_points[j+i][i] = idx
    return all_points

def Gettile_sizeencilDistance(A, tile_size):
    A_serialized = [Serialize(x, tile_size) for x in A]
    return max(A_serialized) - min(A_serialized)

def Gettile_sizeencilDim(A):
    min_in_dims = [min([point[dim] for point in A]) for dim in range(0, len(A[0]))]
    max_in_dims = [max([point[dim] for point in A]) for dim in range(0, len(A[0]))]
    return [max_index-min_index+1 for max_index, min_index in zip(max_in_dims, min_in_dims)]

def GetBuffer(tile_size, A, k):
    FFs = []    # [outputs]
    FIFOs = {}  # {length:[outputs]}
    inputs = []
    for chain in GetChains(tile_size, A, k):
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
    k = buf['k']
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

def PrintUpdate(printer, producer_map, consumer_map, src, input_chan):
    for c in range(0, input_chan):
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

        PrintUpdate(printer, producer_map, consumer_map, dst, input_chan)

        if 'FIFOs' in dst:
            fifo_length = dst['FIFOs']['length']/consumer_map['k']
            fifo_index = dst['FIFOs']['index']
            dst_str = 'FIFO_%d[%d][%d][FIFO_%d_ptr]' % (fifo_length, c, fifo_index, fifo_length)
        elif 'FFs' in dst:
            dst_str = 'FF[%d][%d]' % (c, dst['FFs']['index'])
        else:
            return

        printer.PrintLine('%s = %s;' % (dst_str, src_str))

def PrintCompute(printer, tile_size, A, k, compute_content, input_partition, output_partition, extra_params, input_name, input_chan, burst_width, pixel_width_i, pixel_width_o, dram_chan):
    buf = GetBuffer(tile_size, A, k)
    points = GetPoints(tile_size, A, k)

    printer.PrintLine('void compute(')
    printer.DoIndent()
    for i in range(0, dram_chan):
        printer.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& to_%d,' % i)
    for i in range(0, dram_chan):
        printer.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& from_%d,' % i)
    if extra_params:
        for param_name, param in extra_params.items():
            if 'dup' in param:
                dup = ('[%d]' % param['dup'])
            else:
                dup = ''
            printer.PrintLine('%s %s%s[UNROLL_FACTOR][%d],' % (param['type'], param_name, dup, param['length']))
    printer.PrintLine('int64_t coalesced_data_num,')
    printer.PrintLine('int64_t tile_data_num,')
    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t tile_num_dim_%d,' % i)
    printer.PrintLine('int32_t input_size_dim_%d)' % (len(tile_size)-1))
    printer.UnIndent()
    printer.PrintLine('{')
    printer.DoIndent()

    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t tile_index_dim_%d = 0;' % i)
    printer.PrintLine()

    printer.PrintLine('int32_t i_base[UNROLL_FACTOR] = {%s};' % (', '.join(['%d-STENCIL_DISTANCE' % x for x in range(0, k)])))
    for i in range(1, len(tile_size)):
        printer.PrintLine('int32_t %c_base[UNROLL_FACTOR] = {0};' % coords_in_tile[i])
    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t %c_base = 0;' % coords_in_orig[i])
    for i in range(0, len(tile_size)):
        printer.PrintLine('#pragma HLS array_partition variable=%s_base complete' % coords_in_tile[i], 0)
    printer.PrintLine()

    # array declaration
    if len(buf['FFs'])>0:
        printer.PrintLine('input_type FF[CHANNEL_NUM_I][%d];' % len(buf['FFs']))

    for fifo_length, fifo_list in buf['FIFOs'].items():
        printer.PrintLine('input_type FIFO_%d[CHANNEL_NUM_I][%d][%d];' % (fifo_length/k, len(fifo_list), fifo_length/k))

    if len(buf['FFs'])>0:
        printer.PrintLine('#pragma HLS array_partition variable=FF complete dim=0', 0)

    for fifo_length in buf['FIFOs'].keys():
        printer.PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=1' % (fifo_length/k), 0)
        printer.PrintLine('#pragma HLS array_partition variable=FIFO_%d complete dim=2' % (fifo_length/k), 0)
    printer.PrintLine()

    for fifo_length in buf['FIFOs'].keys():
        printer.PrintLine('int%d_t FIFO_%d_ptr = 0;' % (2**math.ceil(math.log2(math.log2(fifo_length/k))), fifo_length/k))
    printer.PrintLine()

    printer.PrintLine('input_type input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d];' % len(A))
    for idx, item in enumerate(A):
        printer.PrintLine('//         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][%d] <=> (%s)' % (idx, str(item)[1:-1]))
    if burst_width*dram_chan/pixel_width_i/k <= 1:
        printer.PrintLine('input_type input_buffer[CHANNEL_NUM_I][UNROLL_FACTOR];')
    else:
        printer.PrintLine('input_type input_buffer[CHANNEL_NUM_I][UNROLL_FACTOR*(BURST_WIDTH*%d/PIXEL_WIDTH_I/UNROLL_FACTOR)];' % dram_chan)
    if burst_width*dram_chan/pixel_width_o/k <= 1:
        printer.PrintLine('output_type output_buffer[CHANNEL_NUM_O][UNROLL_FACTOR];')
    else:
        printer.PrintLine('output_type output_buffer[CHANNEL_NUM_O][UNROLL_FACTOR*(BURST_WIDTH*%d/PIXEL_WIDTH_O/UNROLL_FACTOR)];' % dram_chan)
    printer.PrintLine('#pragma HLS array_partition variable=input_points complete dim=0', 0)
    printer.PrintLine('#pragma HLS array_partition variable=input_buffer complete dim=0', 0)
    printer.PrintLine('#pragma HLS array_partition variable=output_buffer complete dim=0', 0)
    printer.PrintLine()

    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t %c_base_counter = 0;' % coords_in_orig[i])
    printer.PrintLine()

    printer.PrintLine('// produce output')
    printer.PrintLine('compute_epoch:', 0)
    printer.PrintLine('for(int32_t epoch = 0; epoch < coalesced_data_num*BURST_WIDTH*%d/PIXEL_WIDTH_I/UNROLL_FACTOR; ++epoch)' % dram_chan)
    printer.PrintLine('{')
    printer.DoIndent()
    if len(buf['FFs'])>0:
        printer.PrintLine('#pragma HLS dependence variable=FF inter false', 0)
    for fifo_length in buf['FIFOs'].keys():
        printer.PrintLine('#pragma HLS dependence variable=FIFO_%d inter false' % (fifo_length/k), 0)
    printer.PrintLine('#pragma HLS pipeline II=1', 0)

    if burst_width*dram_chan/pixel_width_i/k <= 1:
        ratio_i = int(pixel_width_i*k/dram_chan/burst_width)
        printer.DoScope()
        printer.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join([('tmp_%d' % x) for x in range(0, dram_chan*ratio_i)])))
        for i in range(0, ratio_i):
            for j in range(0, dram_chan):
                printer.PrintLine('from_%d>>tmp_%d;' % (j, i*dram_chan+j))
        printer.PrintLine('load_coalesced:', 0)
        printer.PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_I; ++j)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS unroll', 0)
        for i in range(0, ratio_i):
            for j in range(0, dram_chan):
                printer.PrintLine('input_buffer[0][BURST_WIDTH/PIXEL_WIDTH_I*%d*%d+j*%d+%d] = tmp_%d((j+1)*PIXEL_WIDTH_I-1, j*PIXEL_WIDTH_I);' % (dram_chan, i, dram_chan, j, i*dram_chan+j))
        printer.UnScope()
        printer.UnScope()
    else:
        printer.PrintLine('switch(epoch&%d)' % (burst_width*dram_chan/pixel_width_i/k-1))
        printer.DoScope()
        printer.PrintLine('case 0:')
        printer.DoScope()
        for i in range(0, dram_chan):
            printer.PrintLine('ap_uint<BURST_WIDTH> tmp_%d;' % i)
        for i in range(0, dram_chan):
            printer.PrintLine('from_%d>>tmp_%d;' % (i, i))
        printer.PrintLine('load_coalesced:', 0)
        printer.PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_I; ++j)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS unroll', 0)
        for i in range(0, dram_chan):
            printer.PrintLine('input_buffer[0][j*%d+%d] = tmp_%d((j+1)*PIXEL_WIDTH_I-1, j*PIXEL_WIDTH_I);' % (dram_chan, i, i))
        printer.UnScope()
        printer.PrintLine('break;')
        printer.UnScope()
        printer.PrintLine('default:')
        printer.DoScope()
        printer.PrintLine('load_shift:', 0)
        printer.PrintLine('for(int j = 0; j < UNROLL_FACTOR; ++j)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS unroll', 0)
        for i in range(0, int(burst_width/pixel_width_i*dram_chan/k-1)):
            printer.PrintLine('input_buffer[0][j+UNROLL_FACTOR*%d] = input_buffer[0][j+UNROLL_FACTOR*%d];' % (i, i+1))
        printer.UnScope()
        printer.UnScope()
        printer.UnScope()
    printer.PrintLine()

    for c in range(0, input_chan):
        for fifo_length in buf['FIFOs'].keys():
            for idx, item in enumerate(buf['FIFOs'][fifo_length]):
                printer.PrintLine('input_type FIFO_%d_%d_%d = FIFO_%d[%d][%d][FIFO_%d_ptr];'% (fifo_length/k, c, idx, fifo_length/k, c, idx, fifo_length/k))
    printer.PrintLine()

    for c in range(0, input_chan):
        for idx, item in enumerate(buf['inputs']):
            for unroll_index in points[item].keys():
                printer.PrintLine("input_points[%d][%d][%d] = input_buffer[%d][%d]; // (%s)" % (c, unroll_index, points[item][unroll_index], c, idx, str(A[points[item][unroll_index]])[1:-1]))
        for idx, item in enumerate(buf['FFs']):
            for unroll_index in points[item].keys():
                printer.PrintLine("input_points[%d][%d][%d] = FF[%d][%d]; // (%s)" % (c, unroll_index, points[item][unroll_index], c, idx, str(A[points[item][unroll_index]])[1:-1]))
        for fifo_length in buf['FIFOs'].keys():
            for idx, item in enumerate(buf['FIFOs'][fifo_length]):
                for unroll_index in points[item].keys():
                    printer.PrintLine("input_points[%d][%d][%d] = FIFO_%d_%d_%d; // (%s)" % (c, unroll_index, points[item][unroll_index], fifo_length/k, c, idx, str(A[points[item][unroll_index]])[1:-1]))

    printer.PrintLine()

    printer.PrintLine('compute_unrolled:', 0)
    printer.PrintLine('for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
    printer.PrintLine('{');printer.DoIndent()
    printer.PrintLine('#pragma HLS unroll', 0)
    for i in range(0, len(tile_size)):
        printer.PrintLine('int32_t& %s = %s_base[unroll_index];' % ((coords_in_tile[i],)*2))
    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t %s = %s_base+%s;' % ((coords_in_orig[i],)*2 + (coords_in_tile[i],)))
    printer.PrintLine('int32_t %s = %s;' % (coords_in_orig[len(tile_size)-1],coords_in_tile[len(tile_size)-1]))
    printer.PrintLine()
    for idx, item in enumerate(A):
        printer.PrintLine('input_type load_%s_at_%s = input_points[0][unroll_index][%d];' % (input_name, '_'.join([str(x).replace('-', 'm') for x in item]), idx))
    printer.PrintLine()

    for line in compute_content:
        if len(line) > 0:
            printer.PrintLine(line.replace('\n','').replace('\r',''))
    if burst_width*dram_chan/pixel_width_o/k <= 1:
        printer.PrintLine('output_buffer[0][unroll_index] = result;')
    else:
        printer.PrintLine('switch(epoch&%d)' % (burst_width*dram_chan/pixel_width_o/k-1))
        printer.DoScope()
        for i in range(0, int(burst_width*dram_chan/pixel_width_o/k)):
            printer.PrintLine('case %d:' % i)
            printer.DoScope()
            printer.PrintLine('output_buffer[0][unroll_index+UNROLL_FACTOR*%d] = result;' % i)
            printer.UnScope()
        printer.UnScope()
    printer.PrintLine()

    printer.PrintLine('%s += UNROLL_FACTOR;' % coords_in_tile[0])
    if(len(tile_size)>1):
        printer.PrintLine('if(%s>=TILE_SIZE_DIM_0)' % coords_in_tile[0])
        printer.PrintLine('{');printer.DoIndent()
        printer.PrintLine('%s -= TILE_SIZE_DIM_0;' % coords_in_tile[0])
        printer.PrintLine('++%s;' % coords_in_tile[1])
        if(len(tile_size)>2):
            printer.PrintLine('if(%s>=TILE_SIZE_DIM_1)' % coords_in_tile[1])
            printer.PrintLine('{');printer.DoIndent()
            printer.PrintLine('%s -= TILE_SIZE_DIM_1;' % coords_in_tile[1])
            printer.PrintLine('++%s;' % coords_in_tile[2])
            if(len(tile_size)>3):
                printer.PrintLine('if(%s>=TILE_SIZE_DIM_2)' % coords_in_tile[2])
                printer.PrintLine('{');printer.DoIndent()
                printer.PrintLine('%s -= TILE_SIZE_DIM_2;' % coords_in_tile[2])
                printer.PrintLine('++%s;' % coords_in_tile[3])
                printer.UnScope()
            printer.UnScope()
        printer.UnScope()
    printer.UnIndent()
    printer.PrintLine('} // for unroll_index')
    printer.PrintLine()

    for c in range(0, input_chan):
        first = True
        for idx, item in enumerate(buf['inputs']):
            if first:
                first = False
            else:
                printer.PrintLine()
            PrintUpdate(printer, GetProduceMap(buf), GetConsumeMap(buf), {'inputs':{'index':idx, 'produce':item}}, input_chan)
    printer.PrintLine()

    for fifo_length in buf['FIFOs'].keys():
        printer.PrintLine('FIFO_%d_ptr = FIFO_%d_ptr==(%d-1) ? 0 : FIFO_%d_ptr+1;' % (fifo_length/k, fifo_length/k, fifo_length/k, fifo_length/k))
    printer.PrintLine()

    if burst_width*dram_chan/pixel_width_o/k <= 1:
        ratio_o = int(pixel_width_o*k/dram_chan/burst_width)
        printer.DoScope()
        printer.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join([('tmp_%d' % x) for x in range(0, dram_chan*ratio_o)])))
        printer.PrintLine('store_coalesced:', 0)
        printer.PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_O; ++j)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS unroll', 0)
        for i in range(0, ratio_i):
            for j in range(0, dram_chan):
                printer.PrintLine('tmp_%d((j+1)*PIXEL_WIDTH_O-1, j*PIXEL_WIDTH_O) = output_buffer[0][BURST_WIDTH/PIXEL_WIDTH_O*%d*%d+j*%d+%d];' % (i*dram_chan+j, dram_chan, i, dram_chan, j))
        printer.UnScope()
        for i in range(0, ratio_o):
            for j in range(0, dram_chan):
                printer.PrintLine('to_%d<<tmp_%d;' % (j, i*dram_chan+j))
        printer.UnScope()
    else:
        printer.PrintLine('if((epoch&%d)==%d)' % ((burst_width*dram_chan/pixel_width_i/k-1), burst_width*dram_chan/pixel_width_i/k-1))
        printer.DoScope()
        printer.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join(['tmp_%d' % x for x in range(0, dram_chan)])))
        printer.PrintLine('store_coalesced:', 0)
        printer.PrintLine('for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_O; ++j)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS unroll', 0)
        for i in range(0, dram_chan):
            printer.PrintLine('tmp_%d((j+1)*PIXEL_WIDTH_O-1, j*PIXEL_WIDTH_O) = output_buffer[0][j*%d+%d];' % (i, dram_chan, i))
        printer.UnScope()
        for i in range(0, dram_chan):
            printer.PrintLine('to_%d<<tmp_%d;' % (i, i))
        printer.UnScope()
    printer.PrintLine()

    printer.PrintLine('++p_base_counter;')
    printer.PrintLine('if(p_base_counter == tile_data_num)')
    printer.DoScope()
    printer.PrintLine('p_base_counter = 0;')
    printer.PrintLine('reset_bases:', 0)
    printer.PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS unroll', 0)
    printer.PrintLine('i_base[unroll_index] = unroll_index-STENCIL_DISTANCE;')
    printer.PrintLine('j_base[unroll_index] = 0;')
    printer.UnScope()
    printer.PrintLine('p_base += TILE_SIZE_DIM_0-STENCIL_DIM_0+1;')
    printer.UnScope()
    printer.UnScope()
    printer.UnScope()

def PrintKernel(printer, tile_size, A, k, app_name, extra_params, dram_chan, dram_separate):
    buf = GetBuffer(tile_size, A, k)
    points = GetPoints(tile_size, A, k)
    printer.PrintLine('extern "C"')
    printer.PrintLine('{')
    printer.PrintLine()
    printer.PrintLine('void %s_kernel(' % app_name)
    printer.DoIndent()
    for i in range(0, dram_chan):
        printer.PrintLine('ap_uint<BURST_WIDTH>* var_output_%d,' % i)
    for i in range(0, dram_chan):
        printer.PrintLine('ap_uint<BURST_WIDTH>* var_input_%d,' % i)
    if extra_params:
        for param_name, param in extra_params.items():
            printer.PrintLine('%s* var_%s,' % (param['type'], param_name))
    printer.PrintLine('int64_t coalesced_data_num,')
    printer.PrintLine('int64_t tile_data_num,')
    for i in range(0, len(tile_size)-1):
        printer.PrintLine('int32_t tile_num_dim_%d,' % i)
    printer.PrintLine('int32_t input_size_dim_%d)' % (len(tile_size)-1))
    printer.UnIndent()
    printer.PrintLine('{')
    printer.DoIndent()

    chan = 0
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS interface m_axi port=var_output_%d offset=slave depth=65536 bundle=gmem%d latency=120' % (i, chan), 0)
        chan += 1
    if not dram_separate:
        chan = 0
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS interface m_axi port=var_input_%d offset=slave depth=65536 bundle=gmem%d latency=120' % (i, chan), 0)
        chan += 1
    if extra_params:
        for param_name, param in extra_params.items():
            printer.PrintLine('#pragma HLS interface m_axi port=var_%s offset=slave depth=%d bundle=gmem0 latency=120' % (param_name, param['length']), 0)
    printer.PrintLine()
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS interface s_axilite port=var_output_%d bundle=control' % i, 0)
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS interface s_axilite port=var_input_%d bundle=control' % i, 0)
    if extra_params:
        for param_name in extra_params.keys():
            printer.PrintLine('#pragma HLS interface s_axilite port=var_%s bundle=control' % param_name, 0)
    printer.PrintLine('#pragma HLS interface s_axilite port=coalesced_data_num bundle=control', 0)
    printer.PrintLine('#pragma HLS interface s_axilite port=tile_data_num bundle=control', 0)
    for i in range(0, len(tile_size)-1):
        printer.PrintLine('#pragma HLS interface s_axilite port=tile_num_dim_%d bundle=control' % i, 0)
    printer.PrintLine('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % (len(tile_size)-1), 0)
    printer.PrintLine('#pragma HLS interface s_axilite port=return bundle=control', 0)
    printer.PrintLine()

    for i in range(0, dram_chan):
        printer.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >  input_stream_%d( "input_stream_%d");' % (i, i))
    for i in range(0, dram_chan):
        printer.PrintLine('hls::stream<ap_uint<BURST_WIDTH> > output_stream_%d("output_stream_%d");' % (i, i))
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS stream variable=input_stream_%d depth=32' % i, 0)
    for i in range(0, dram_chan):
        printer.PrintLine('#pragma HLS stream variable=output_stream_%d depth=32' % i, 0)
    printer.PrintLine()

    if extra_params:
        for param_name, param in extra_params.items():
            if 'dup' in param:
                dup = ('[%d]' % param['dup'])
            else:
                dup = ''
            printer.PrintLine('%s %s%s[UNROLL_FACTOR][%d];' % (param['type'], param_name, dup, param['length']))
        printer.PrintLine()

        for param_name, param in extra_params.items():
            if 'dup' in param:
                partition_dim = 3
                printer.PrintLine('#pragma HLS array_partition variable=%s complete dim=1' % param_name, 0)
            else:
                partition_dim = 2
            printer.PrintLine('#pragma HLS array_partition variable=%s complete dim=%d' % (param_name, partition_dim-1), 0)
            if 'partition' in param:
                printer.PrintLine('#pragma HLS array_partition variable=%s %s dim=%d' % (param_name, param['partition'], partition_dim), 0)
        printer.PrintLine()

        printer.PrintLine('extra_params_unrolled:', 0)
        printer.PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
        printer.PrintLine('{');printer.DoIndent()
        printer.PrintLine('#pragma HLS unroll',0)
        for param_name, param in extra_params.items():
            printer.PrintLine('%s_init:' % param_name, 0)
            printer.PrintLine('for(int %s_index = 0; %s_index < %d; ++%s_index)' % (param_name, param_name, param['length'], param_name))
            printer.PrintLine('{');printer.DoIndent()
            printer.PrintLine('#pragma HLS pipeline II=1', 0)
            if 'dup' in param:
                for i in range(0, param['dup']):
                    printer.PrintLine('%s[%d][unroll_index][%s_index] = var_%s[%s_index];' % ((param_name, i)+(param_name,)*3))
            else:
                printer.PrintLine('%s[unroll_index][%s_index] = var_%s[%s_index];' % ((param_name,)*4))
            printer.UnScope()
        printer.UnScope()
        printer.PrintLine()

    extra_params_str = ''.join([var+', ' for var in extra_params.keys()]) if extra_params else ''
    coords_str = ''.join([coords_in_tile[i]+'_base, ' for i in range(0, len(tile_size))]) + ''.join([coords_in_orig[i]+'_base, ' for i in range(0, len(tile_size)-1)])

    printer.PrintLine('#pragma HLS dataflow', 0)
    for i in range(0, dram_chan):
        printer.PrintLine('load(input_stream_%d, var_input_%d, coalesced_data_num);' % (i, i))
    output_stream = ', '.join(['output_stream_%d' % x for x in range(0, dram_chan)])
    input_stream = ', '.join(['input_stream_%d' % x for x in range(0, dram_chan)])
    tile_num_dim = ', '.join(['tile_num_dim_%d' % x for x in range(0, len(tile_size)-1)])
    printer.PrintLine('compute(%s, %s, coalesced_data_num, tile_data_num, %s, input_size_dim_%d);' % (output_stream, input_stream, tile_num_dim, len(tile_size)-1))
    for i in range(0, dram_chan):
        printer.PrintLine('store(var_output_%d, output_stream_%d, coalesced_data_num);' % (i, i))

    printer.UnScope()
    printer.PrintLine()
    printer.PrintLine('}//extern "C"')

def PrintHeader(printer, app_name):
    for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int', 'hls_stream']:
        printer.PrintLine('#include<%s.h>' % header)
    printer.PrintLine()

def PrintLoad(printer):
    printer.PrintLine('void load(hls::stream<ap_uint<BURST_WIDTH> > &to, ap_uint<BURST_WIDTH>* from, int data_num)')
    printer.PrintLine('{');printer.DoIndent()
    printer.PrintLine('load_epoch:', 0)
    printer.PrintLine('for(int i = 0; i < data_num; ++i)')
    printer.PrintLine('{');printer.DoIndent()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('to<<from[i];')
    printer.UnScope()
    printer.UnScope()

def PrintStore(printer):
    printer.PrintLine('void store(ap_uint<BURST_WIDTH>* to, hls::stream<ap_uint<BURST_WIDTH> > &from, int data_num)')
    printer.PrintLine('{');printer.DoIndent()
    printer.PrintLine('store_epoch:', 0)
    printer.PrintLine('for(int i = 0; i < data_num; ++i)')
    printer.PrintLine('{');printer.DoIndent()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('from>>to[i];')
    printer.UnScope()
    printer.UnScope()

def PrintGuard(printer, var, val):
    printer.PrintLine('#if %s != %d' % (var, val))
    printer.PrintLine('#error %s != %d' % (var, val))
    printer.PrintLine('#endif//%s != %d' % (var, val))

def PrintDefine(printer, var, val):
    printer.PrintLine('#ifndef %s' % var)
    printer.PrintLine('#define %s %d' % (var, val))
    printer.PrintLine('#endif//%s' % var)

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

def PrintCode(stencil, output_file):
    printer = Printer(output_file)
    burst_width = stencil.burst_width
    dram_chan = stencil.dram_chan

    app_name = stencil.app_name
    input_name = stencil.input_name
    input_type = stencil.input_type
    input_chan = stencil.input_chan
    output_type = stencil.output_type
    output_chan = stencil.output_chan
    A = stencil.A
    dim = stencil.dim
    extra_params = stencil.extra_params
    compute_content = stencil.compute_content

    tile_size = stencil.tile_size
    k = stencil.k
    dram_separate = stencil.dram_separate

    pixel_width_i = type_width[input_type]
    pixel_width_o = type_width[output_type]
    input_partition = burst_width/pixel_width_i*dram_chan/2 if burst_width/pixel_width_i*dram_chan/2 > k/2 else k/2
    output_partition = burst_width/pixel_width_o*dram_chan/2 if burst_width/pixel_width_o*dram_chan/2 > k/2 else k/2

    PrintHeader(printer, app_name)

    printer.PrintLine('typedef %s input_type;' % input_type)
    printer.PrintLine('typedef %s output_type;' % output_type)
    printer.PrintLine()

    PrintDefine(printer, 'BURST_WIDTH', burst_width)
    PrintDefine(printer, 'PIXEL_WIDTH_I', pixel_width_i)
    PrintDefine(printer, 'PIXEL_WIDTH_O', pixel_width_o)
    for i, dim in enumerate(Gettile_sizeencilDim(A)):
        PrintDefine(printer, 'STENCIL_DIM_%d' % i, dim)
    PrintDefine(printer, 'STENCIL_DISTANCE', Gettile_sizeencilDistance(A, tile_size))
    PrintDefine(printer, 'CHANNEL_NUM_I', input_chan)
    PrintDefine(printer, 'CHANNEL_NUM_O', output_chan)
    printer.PrintLine()

    PrintGuard(printer, 'UNROLL_FACTOR', k)
    for i in range(0, len(tile_size)-1):
        PrintGuard(printer, 'TILE_SIZE_DIM_%d' % i, tile_size[i])
    PrintGuard(printer, 'BURST_WIDTH', burst_width)
    PrintGuard(printer, 'PIXEL_WIDTH_I', pixel_width_i)
    PrintGuard(printer, 'PIXEL_WIDTH_O', pixel_width_o)
    for i, dim in enumerate(Gettile_sizeencilDim(A)):
        PrintGuard(printer, 'STENCIL_DIM_%d' % i, dim)
    PrintGuard(printer, 'STENCIL_DISTANCE', Gettile_sizeencilDistance(A, tile_size))
    printer.PrintLine()

    PrintLoad(printer)
    printer.PrintLine()
    PrintStore(printer)
    printer.PrintLine()
    PrintCompute(printer, tile_size, A, k, compute_content, input_partition, output_partition, extra_params, input_name, input_chan, burst_width, pixel_width_i, pixel_width_o, dram_chan)
    printer.PrintLine()
    PrintKernel(printer, tile_size, A, k, app_name, extra_params, dram_chan, dram_separate)

def main():
    config = json.loads(sys.stdin.read())
    input_type = config['input_type']
    output_type = config['output_type']
    dram_chan = int(os.environ.get('DRAM_CHAN', config['dram_chan']))
    dram_separate = ('DRAM_SEPARATE' in os.environ) or ('dram_separate' in config and config['dram_separate'])
    if dram_separate:
        dram_chan = int(dram_chan/2)
    app_name = config['app_name']
    tile_size = config.get('St', [0]*config['dim'])
    A = config['A']
    k = int(os.environ.get('UNROLL_FACTOR', config['k']))
    burst_width = config['burst_width']

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
    while (i < config['dim']-1 and ('TILE_SIZE_DIM_%d' % i) in os.environ):
        tile_size[i] = int(os.environ[('TILE_SIZE_DIM_%d' % i)])
        i += 1

    stencil = Stencil(
        burst_width = burst_width,
        app_name = app_name,
        input_name = 'input',
        input_type = input_type[0],
        input_chan = input_type[1],
        output_name = 'output',
        output_type = output_type[0],
        output_chan = output_type[1],
        A = A,
        dim = config['dim'],
        extra_params = extra_params,
        compute_content = compute_content,
        dram_chan = dram_chan,
        tile_size = tile_size,
        k = k,
        dram_separate = dram_separate)

    PrintCode(stencil, sys.stdout)

if __name__ == '__main__':
    main()
