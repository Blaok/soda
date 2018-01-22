#!/usr/bin/python3.6
from collections import deque
from fractions import Fraction
from functools import reduce
import json
import logging
import math
import operator
import os
import sys

from supo.generator.utils import coords_in_tile, coords_in_orig, type_width, IsFloat, Stencil, Printer, PrintDefine, PrintGuard, SerializeIterative, GetStencilDistance, GetStencilDim, GetOverallStencilWindow
from supo.grammar import ExtraParam

logger = logging.getLogger('__main__').getChild(__name__)

def GetChains(tile_size, b, unroll_factor):
    logger.debug('get reuse chains of buffer %s' % b.name)
    A_dag = set.union(*[(lambda offsets: set.union(*[{max(offsets)+i-x+s.delay[b.name] for x in offsets} for i in range(unroll_factor)]))(SerializeIterative(s.window[b.name], tile_size)) for s in b.children])
    logger.debug('A† of buffer %s: %s' % (b.name, A_dag))
    chains = sum([tuple([tuple(sorted([x for x in A_dag if x%unroll_factor == i]))]) for i in range(unroll_factor)], ())
    for idx, chain in enumerate(chains):
        logger.debug('reuse chain %d of buffer %s: %s' % (idx, b.name, chain))
    return chains

def GetPoints(tile_size, b, unroll_factor):
    all_points = {} # {name:{offset:{unroll_index:point_index}}}
    for s in b.children:
        all_points[s.output.name] = {}
        offsets = SerializeIterative(s.window[b.name], tile_size)
        max_offset = max(offsets)
        for unroll_index in range(unroll_factor):
            for idx, offset in enumerate(offsets):
                all_points[s.output.name].setdefault(max_offset-offset+s.delay[b.name]+unroll_index, {})[unroll_factor-1-unroll_index] = idx
    for s in b.children:
        for offset, points in all_points[s.output.name].items():
            for unroll_index, point in points.items():
                logger.debug('%s <- %s @ offset=%d <=> (%s) @ unroll_index=%d' % (s.output.name, b.name, offset, ', '.join(map(str, s.window[b.name][point])), unroll_index))
    return all_points

def GetBuffer(tile_size, b, unroll_factor):
    FFs = []    # [outputs]
    FIFOs = {}  # {length:[outputs]}
    inputs = []
    for chain in GetChains(tile_size, b, unroll_factor):
        inputs.append(chain[0])
        for j in range(len(chain)-1):
            interval_length = chain[j+1]-chain[j]
            if interval_length == unroll_factor:
                FFs.append(chain[j+1])
            else:
                FIFOs.setdefault(interval_length, []).append(chain[j+1])
    reuse_buffer = {'FFs':tuple(sorted(FFs)), 'FIFOs':{k:tuple(sorted(v)) for k, v in FIFOs.items()}, 'inputs':tuple(reversed(sorted(inputs)))}
    logger.debug('   FFs in reuse chains of buffer %s: %s' % (b.name, reuse_buffer['FFs']))
    logger.debug(' FIFOs in reuse chains of buffer %s: %s' % (b.name, reuse_buffer['FIFOs']))
    logger.debug('inputs in reuse chains of buffer %s: %s' % (b.name, reuse_buffer['inputs']))
    return reuse_buffer

def GetConsumeMap(buf, unroll_factor):
    mapping = {}    # maps an offset to what consumes it
    for idx, item in enumerate(buf['FFs']):
        mapping[item-unroll_factor] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item-fifo_length] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def GetProduceMap(buf):
    mapping = {}    # maps an offset to what produces it
    for idx, item in enumerate(buf['inputs']):
        mapping[item] = {'inputs':{'index':idx, 'produce':item}}
    for idx, item in enumerate(buf['FFs']):
        mapping[item] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def PrintUpdate(p, unroll_factor, producer_map, consumer_map, src, chan, variable_name):
    if 'FIFOs' in src:
        dst = consumer_map.get(src['FIFOs']['produce'])
        if not dst:
            return
        fifo_length = src['FIFOs']['length']/unroll_factor
        fifo_index = src['FIFOs']['index']
        src_str = ['FIFO_%d_%s_chan_%d_fifo_%d' % (fifo_length, variable_name, c, fifo_index) for c in range(chan)]
    elif 'FFs' in src:
        dst = consumer_map.get(src['FFs']['produce'])
        if not dst:
            return
        src_str = ['FF_%s_chan_%d[%d]' % (variable_name, c, src['FFs']['index']) for c in range(chan)]
    else:
        dst = consumer_map.get(src['inputs']['produce'])
        if not dst:
            return
        src_str = ['buffer_%s_chan_%d[%d]' % (variable_name, c, src['inputs']['index']) for c in range(chan)]

    PrintUpdate(p, unroll_factor, producer_map, consumer_map, dst, chan, variable_name)

    if 'FIFOs' in dst:
        fifo_length = dst['FIFOs']['length']/unroll_factor
        fifo_index = dst['FIFOs']['index']
        dst_str = ['FIFO_%d_%s_chan_%d[%d][FIFO_%d_%s_ptr]' % (fifo_length, variable_name, c, fifo_index, fifo_length, variable_name) for c in range(chan)]
    elif 'FFs' in dst:
        dst_str = ['FF_%s_chan_%d[%d]' % (variable_name, c, dst['FFs']['index']) for c in range(chan)]
    else:
        return

    for c in range(chan):
        p.PrintLine('%s = %s;' % (dst_str[c], src_str[c]))

def PrintCompute(p, stencil):
    app_name = stencil.app_name
    buffers = stencil.buffers
    burst_width = stencil.burst_width
    dim = stencil.dim
    dram_bank = stencil.dram_bank
    dram_separate = stencil.dram_separate
    extra_params = stencil.extra_params
    input_chan = stencil.input.chan
    input_name = stencil.input.name
    input_type = stencil.input.type
    output_chan = stencil.output.chan
    output_name = stencil.output.name
    output_type = stencil.output.type
    pixel_width_i = stencil.pixel_width_i
    pixel_width_o = stencil.pixel_width_o
    stages = stencil.stages
    tile_size = stencil.tile_size
    unroll_factor = stencil.unroll_factor

    produce_consume_ratio_i = int((burst_width*dram_bank)/(pixel_width_i*unroll_factor))
    consume_produce_ratio_i = int((pixel_width_i*unroll_factor)/(burst_width*dram_bank))
    produce_consume_ratio_o = int((burst_width*dram_bank)/(pixel_width_o*unroll_factor))
    consume_produce_ratio_o = int((pixel_width_o*unroll_factor)/(burst_width*dram_bank))

    print_aux = False

    logger.debug('generate compute function')

    p.PrintLine('void compute(')
    p.DoIndent()
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& to_chan_%d_bank_%d,' % (c, i))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >& from_chan_%d_bank_%d,' % (c, i))
    if extra_params:
        for param in extra_params.values():
            if param.dup:
                dup_str = ('[%d]' % param.dup)
            else:
                dup_str = ''
            p.PrintLine('%s param_%s%s[UNROLL_FACTOR][%s],' % (param.type, param.name, dup_str, ']['.join(map(str, param.size))))
    p.PrintLine('int64_t coalesced_data_num,')
    p.PrintLine('int64_t tile_data_num,')
    for i in range(len(tile_size)-1):
        p.PrintLine('int32_t tile_num_dim_%d,' % i)
    p.PrintLine('int32_t input_size_dim_%d)' % (len(tile_size)-1))
    p.UnIndent()
    p.DoScope()

    for i in range(len(tile_size)-1):
        p.PrintLine('int32_t tile_index_dim_%d = 0;' % i)
    p.PrintLine()

    # generate reuse buffers and map offsets to points
    logger.info('generate reuse buffers')
    reuse_buffers = {}
    all_points = {}
    for b in stencil.GetProducerBuffers():
        reuse_buffers[b.name] = GetBuffer(tile_size, b, unroll_factor)
        all_points[b.name] = GetPoints(tile_size, b, unroll_factor)
    overall_stencil_window = GetOverallStencilWindow(stencil.input, stencil.output)

    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        points_from_b = all_points[b.name]

        if print_aux:
            msg = 'aux parameters for %s' % b.name
            logger.debug('generate '+msg)
            p.PrintLine('// '+msg)
            stencil_distance = stencil.output.offset.get() - b.offset.get()
            p.PrintLine('int32_t i_base_%s[UNROLL_FACTOR] = {%s};' % (b.name, ', '.join([str(x-stencil_distance) for x in range(unroll_factor)])))
            for i in range(1, len(tile_size)):
                p.PrintLine('int32_t %c_base_%s[UNROLL_FACTOR] = {0};' % (coords_in_tile[i], b.name))
            for i in range(len(tile_size)-1):
                p.PrintLine('int32_t %c_base_%s = 0;' % (coords_in_orig[i], b.name))
                p.PrintLine('int32_t %c_base_%s_counter = 0;' % (coords_in_orig[i], b.name))
            for i in range(len(tile_size)):
                p.PrintLine('#pragma HLS array_partition variable=%s_base_%s complete' % (coords_in_tile[i], b.name), 0)
            p.PrintLine()

        # reuse chains
        msg = 'reuse chains for %s' % b.name
        logger.debug('generate '+msg)
        p.PrintLine('// '+msg)
        for c in range(b.chan):
            if len(buf['FFs'])>0:
                p.PrintLine('%s FF_%s_chan_%d[%d];' % (b.type, b.name, c, len(buf['FFs'])))

        for c in range(b.chan):
            for fifo_length, fifo_list in buf['FIFOs'].items():
                p.PrintLine('%s FIFO_%d_%s_chan_%d[%d][%d];' % (b.type, fifo_length/unroll_factor, b.name, c, len(fifo_list), fifo_length/unroll_factor))

        for c in range(b.chan):
            if len(buf['FFs'])>0:
                p.PrintLine('#pragma HLS array_partition variable=FF_%s_chan_%d complete' % (b.name, c), 0)

        for c in range(b.chan):
            for fifo_length in buf['FIFOs'].keys():
                p.PrintLine('#pragma HLS array_partition variable=FIFO_%d_%s_chan_%d complete dim=1' % (fifo_length/unroll_factor, b.name, c), 0)
        p.PrintLine()

        for fifo_length in buf['FIFOs'].keys():
            p.PrintLine('uint%d_t FIFO_%d_%s_ptr = 0;' % (2**math.ceil(math.log2(math.log2(fifo_length/unroll_factor))), fifo_length/unroll_factor, b.name))
        p.PrintLine()

        msg = 'points aliases for %s' % b.name
        logger.debug('generate '+msg)
        p.PrintLine('// '+msg)
        for s in b.children:
            for c in range(b.chan):
                p.PrintLine('%s points_from_%s_to_%s_chan_%d[UNROLL_FACTOR][%d];' % (b.type, b.name, s.output.name, c, len(s.window[b.name])))
            for idx, point in enumerate(s.window[b.name]):
                p.PrintLine('//%s points_from_%s_to_%s_chan_x[UNROLL_FACTOR][%d] <=> %s[x](%s)' % (' '*(len(b.type)-2), b.name, s.output.name, idx, b.name, ', '.join(map(str, point))))
            for c in range(b.chan):
                p.PrintLine('#pragma HLS array_partition variable=points_from_%s_to_%s_chan_%d complete dim=0' % (b.name, s.output.name, c), 0)
        p.PrintLine()

    msg = 'input buffer'
    logger.debug('generate '+msg)
    p.PrintLine('// '+msg)
    for c in range(input_chan):
        if produce_consume_ratio_i <= 1:
            p.PrintLine('%s buffer_%s_chan_%d[UNROLL_FACTOR];' % (input_type, input_name, c))
        else:
            p.PrintLine('%s buffer_%s_chan_%d[UNROLL_FACTOR*%d];' % (input_type, input_name, c, produce_consume_ratio_i))
    for c in range(input_chan):
        p.PrintLine('#pragma HLS array_partition variable=buffer_%s_chan_%d complete dim=0' % (input_name, c), 0)
    p.PrintLine()

    msg = 'intermediate buffer for '+b.name
    logger.debug('generate '+msg)
    p.PrintLine('// '+msg)
    for b in buffers.values():
        if b.name in (input_name, output_name):
            continue
        for c in range(b.chan):
            p.PrintLine('%s buffer_%s_chan_%d[UNROLL_FACTOR];' % (b.type, b.name, c))
        for c in range(b.chan):
            p.PrintLine('#pragma HLS array_partition variable=buffer_%s_chan_%d complete dim=0' % (b.name, c), 0)
        p.PrintLine()

    msg = 'output buffer'
    logger.debug('generate '+msg)
    p.PrintLine('// '+msg)
    for c in range(output_chan):
        if produce_consume_ratio_o <= 1:
            p.PrintLine('%s buffer_%s_chan_%d[UNROLL_FACTOR];' % (output_type, output_name, c))
        else:
            p.PrintLine('%s buffer_%s_chan_%d[UNROLL_FACTOR*%d];' % (output_type, output_name, c, produce_consume_ratio_o))
    for c in range(output_chan):
        p.PrintLine('#pragma HLS array_partition variable=buffer_%s_chan_%d complete dim=0' % (output_name, c), 0)
    p.PrintLine()

    p.PrintLine('// produce output')
    p.PrintLine('compute_epoch:', 0)
    p.PrintLine('for(int32_t epoch = 0; epoch < coalesced_data_num%c%d; ++epoch)' % (('*', produce_consume_ratio_i) if produce_consume_ratio_i>=1 else ('/', consume_produce_ratio_i)))
    p.DoScope()

    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        for c in range(b.chan):
            if len(buf['FFs'])>0:
                p.PrintLine('#pragma HLS dependence variable=FF_%s_chan_%d inter false' % (b.name, c), 0)

        for c in range(b.chan):
            for fifo_length in buf['FIFOs'].keys():
                p.PrintLine('#pragma HLS dependence variable=FIFO_%d_%s_chan_%d inter false' % (fifo_length/unroll_factor, b.name, c), 0)
    p.PrintLine('#pragma HLS pipeline II=1', 0)

    if produce_consume_ratio_i <= 1:
        p.DoScope()
        for c in range(input_chan):
            p.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join([('tmp_chan_%d_bank_%d' % (c, x)) for x in range(dram_bank*consume_produce_ratio_i)])))
        for c in range(input_chan):
            for i in range(consume_produce_ratio_i):
                for j in range(dram_bank):
                    p.PrintLine('from_chan_%d_bank_%d>>tmp_chan_%d_bank_%d;' % (c, j, c, i*dram_bank+j))
        p.PrintLine('load_coalesced:', 0)
        p.PrintLine('for(int j = 0; j < BURST_WIDTH/%d; ++j)' % pixel_width_i)
        p.DoScope()
        p.PrintLine('#pragma HLS unroll', 0)
        for c in range(input_chan):
            for i in range(consume_produce_ratio_i):
                for j in range(dram_bank):
                    if IsFloat(input_type):
                        p.PrintLine('uint%d_t raw_bits_chan_%d_bank_%d = tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d);' % (type_width[input_type], c, i*dram_bank+j, c, i*dram_bank+j, pixel_width_i, pixel_width_i))
                        p.PrintLine('buffer_%s_chan_%d[BURST_WIDTH/%d*%d*%d+j*%d+%d] = *(%s*)(&raw_bits_chan_%d_bank_%d);' % (input_name, c, pixel_width_i, dram_bank, i, dram_bank, j, input_type, c, i*dram_bank+j))
                    else:
                        p.PrintLine('buffer_%s_chan_%d[BURST_WIDTH/%d*%d*%d+j*%d+%d] = tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d);' % (input_name, c, pixel_width_i, dram_bank, i, dram_bank, j, c, i*dram_bank+j, pixel_width_i, pixel_width_i))
        p.UnScope()
        p.UnScope()
    else:
        p.PrintLine('switch(epoch&%d)' % (produce_consume_ratio_i-1))
        p.DoScope()
        p.PrintLine('case 0:')
        p.DoScope()
        for c in range(input_chan):
            for i in range(dram_bank):
                p.PrintLine('ap_uint<BURST_WIDTH> tmp_chan_%d_bank_%d;' % (c, i))
        for c in range(input_chan):
            for i in range(dram_bank):
                p.PrintLine('from_chan_%d_bank_%d>>tmp_chan_%d_bank_%d;' % (c, i, c, i))
        p.PrintLine('load_coalesced:', 0)
        p.PrintLine('for(int j = 0; j < BURST_WIDTH/%d; ++j)' % pixel_width_i)
        p.DoScope()
        p.PrintLine('#pragma HLS unroll', 0)
        for c in range(input_chan):
            for i in range(dram_bank):
                if IsFloat(input_type):
                    p.PrintLine('uint%d_t raw_bits_chan_%d_bank_%d = tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d);' % (type_width[input_type], c, i, c, i, pixel_width_i, pixel_width_i))
                    p.PrintLine('buffer_%s_chan_%d[j*%d+%d] = *(%s*)(&raw_bits_chan_%d_bank_%d);' % (input_name, c, dram_bank, i, input_type, c, i))
                else:
                    p.PrintLine('buffer_%s_chan_%d[j*%d+%d] = tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d);' % (input_name, c, dram_bank, i, c, i, pixel_width_i, pixel_width_i))
        p.UnScope()
        p.PrintLine('break;')
        p.UnScope()
        p.PrintLine('default:')
        p.DoScope()
        p.PrintLine('load_shift:', 0)
        p.PrintLine('for(int j = 0; j < UNROLL_FACTOR; ++j)')
        p.DoScope()
        p.PrintLine('#pragma HLS unroll', 0)
        for c in range(input_chan):
            for i in range(produce_consume_ratio_i-1):
                p.PrintLine('buffer_%s_chan_%d[j+UNROLL_FACTOR*%d] = buffer_%s_chan_%d[j+UNROLL_FACTOR*%d];' % (input_name, c, i, input_name, c, i+1))
        p.UnScope()
        p.UnScope()
        p.UnScope()
    p.PrintLine()

    # aliases for FIFO outputs
    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        for c in range(b.chan):
            for fifo_length in buf['FIFOs'].keys():
                for idx, item in enumerate(buf['FIFOs'][fifo_length]):
                    p.PrintLine('%s& FIFO_%d_%s_chan_%d_fifo_%d = FIFO_%d_%s_chan_%d[%d][FIFO_%d_%s_ptr];' % (b.type, fifo_length/unroll_factor, b.name, c, idx, fifo_length/unroll_factor, b.name, c, idx, fifo_length/unroll_factor, b.name))
    p.PrintLine()

    # computational kernel must be scheduled in order
    logger.info('generate computational kernel')
    processing_queue = deque([stencil.input.name])
    processed_buffers = {stencil.input.name}
    while len(processing_queue)>0:
        b = stencil.buffers[processing_queue.popleft()]
        logger.debug('inspecting buffer %s\'s children' % b.name)
        for s in b.children:
            if {x.name for x in s.inputs.values()} <= processed_buffers and s.output.name not in processed_buffers:
                # good, all inputs are processed, can emit code to produce current buffer
                logger.debug('input%s for buffer %s (i.e. %s) %s processed' % ('' if len(s.inputs)==1 else 's', s.output.name, ', '.join([x.name for x in s.inputs.values()]), 'is' if len(s.inputs)==1 else 'are'))

                # start emitting code for stage %s
                logger.debug('emit code for stage %s' % s.output.name)
                # connect points from previous buffer/FIFO/FF
                for bb in s.inputs.values():
                    buf = reuse_buffers[bb.name]
                    points = all_points[bb.name][s.output.name]
                    logger.debug('%s <- %s points: %s' % (s.output.name, bb.name, points))
                    logger.debug('%s <- %s buf: %s' % (s.output.name, bb.name, buf))
                    for idx, offset in enumerate(buf['inputs']):
                        for unroll_index, point in points.get(offset, {}).items():
                            for c in range(bb.chan):
                                p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = buffer_%s_chan_%d[%d]; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.output.name, c, unroll_index, point, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))
                    for idx, offset in enumerate(buf['FFs']):
                        for unroll_index, point in points.get(offset, {}).items():
                            for c in range(bb.chan):
                                p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = FF_%s_chan_%d[%d]; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.output.name, c, unroll_index, point, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))
                    for fifo_length in buf['FIFOs'].keys():
                        for idx, offset in enumerate(buf['FIFOs'][fifo_length]):
                            for unroll_index, point in points.get(offset, {}).items():
                                for c in range(bb.chan):
                                    p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = FIFO_%d_%s_chan_%d_fifo_%d; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.output.name, c, unroll_index, point, fifo_length/unroll_factor, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))

                p.PrintLine()

                p.PrintLine('compute_%s_unrolled:' % s.output.name, 0)
                p.PrintLine('for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
                p.DoScope('unroll_index')
                p.PrintLine('#pragma HLS unroll', 0)

                if print_aux:
                    for i in range(len(tile_size)):
                        p.PrintLine('int32_t& %c_%s = %c_base_%s[unroll_index];' % ((coords_in_tile[i], s.output.name)*2))
                    for i in range(len(tile_size)-1):
                        p.PrintLine('int32_t %c_%s = %c_base_%s+%c_%s;' % ((coords_in_orig[i], s.output.name)*2 + (coords_in_tile[i], s.output.name)))
                    p.PrintLine('int32_t %c_%s = %c_%s;' % (coords_in_orig[len(tile_size)-1], s.output.name, coords_in_tile[len(tile_size)-1], s.output.name))
                    p.PrintLine()

                for bb in s.inputs.values():
                    for c in range(bb.chan):
                        for idx, point in enumerate(s.window[bb.name]):
                            p.PrintLine('%s& load_%s_for_%s_chan_%d_at_%s = points_from_%s_to_%s_chan_%d[unroll_index][%d];' % (bb.type, bb.name, s.output.name, c, '_'.join([str(x).replace('-', 'm') for x in point]), bb.name, s.output.name, c, idx))
                p.PrintLine()

                LoadPrinter = lambda node: 'param_%s%s[unroll_index]%s' % (node.name, '' if extra_params[node.name].dup is None else '[%d]' % node.chan, ''.join(['[%d]'%x for x in node.idx])) if node.name in extra_params else 'load_%s_for_%s_chan_%d_at_%s' % (node.name, s.output.name, node.chan, '_'.join([str(x).replace('-', 'm') for x in node.idx]))
                StorePrinter = lambda node: '%s result_chan_%d' % (output_type, node.chan) if node.name == output_name else 'buffer_%s_chan_%d[unroll_index]' % (node.name, node.chan)

                for e in s.expr:
                    p.PrintLine(e.GetCode(LoadPrinter, StorePrinter))

                if len(s.output.children)==0:
                    p.PrintLine()
                    if produce_consume_ratio_o <= 1:
                        for c in range(output_chan):
                            p.PrintLine('buffer_%s_chan_%d[unroll_index] = result_chan_%d;' % (output_name, c, c))
                    else:
                        p.PrintLine('switch(epoch&%d)' % (produce_consume_ratio_o-1))
                        p.DoScope()
                        for i in range(produce_consume_ratio_o):
                            p.PrintLine('case %d:' % i)
                            p.DoScope()
                            for c in range(output_chan):
                                p.PrintLine('buffer_%s_chan_%d[unroll_index+UNROLL_FACTOR*%d] = result_chan_%d;' % (output_name, c, i, c))
                            p.UnScope()
                        p.UnScope()

                if print_aux:
                    p.PrintLine()
                    p.PrintLine('%c_%s += UNROLL_FACTOR;' % (coords_in_tile[0], s.output.name))
                    if len(tile_size)>1:
                        p.PrintLine('if(%c_%s>=TILE_SIZE_DIM_0)' % (coords_in_tile[0], s.output.name))
                        p.DoScope()
                        p.PrintLine('%c_%s -= TILE_SIZE_DIM_0;' % (coords_in_tile[0], s.output.name))
                        p.PrintLine('++%c_%s;' % (coords_in_tile[1], s.output.name))
                        if len(tile_size)>2:
                            p.PrintLine('if(%c_%s>=TILE_SIZE_DIM_1)' % (coords_in_tile[1], s.output.name))
                            p.DoScope()
                            p.PrintLine('%c_%s -= TILE_SIZE_DIM_1;' % (coords_in_tile[1], s.output.name))
                            p.PrintLine('++%c_%s;' % (coords_in_tile[2], s.output.name))
                            if len(tile_size)>3:
                                p.PrintLine('if(%c_%s>=TILE_SIZE_DIM_2)' % (coords_in_tile[2], s.output.name))
                                p.DoScope()
                                p.PrintLine('%c_%s -= TILE_SIZE_DIM_2;' % (coords_in_tile[2], s.output.name))
                                p.PrintLine('++%c_%s;' % (coords_in_tile[3], s.output.name))
                                p.UnScope()
                            p.UnScope()
                        p.UnScope()

                p.UnScope()
                p.PrintLine()
                # finish emitting code

                processing_queue.append(s.output.name)
                processed_buffers.add(s.output.name)
            else:
                for bb in s.inputs.values():
                    if bb.name not in processed_buffers:
                        logger.debug('buffer %s requires buffer %s as an input' % (s.output.name, bb.name))
                        logger.debug('but buffer %s isn\'t produced yet' % bb.name)
                        logger.debug('add %s to scheduling queue' % bb.name)
                        processing_queue.append(bb.name)

    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        for idx, item in enumerate(buf['inputs']):
            msg = 'move reuse chain %d for buffer %s' % (idx, b.name)
            logger.debug(msg)
            p.PrintLine('// '+msg)
            PrintUpdate(p, unroll_factor, GetProduceMap(buf), GetConsumeMap(buf, unroll_factor), {'inputs':{'index':idx, 'produce':item}}, b.chan, b.name)
            p.PrintLine()

        if len(buf['FIFOs'])>0:
            msg = 'move FIFO ptrs for buffer %s' % b.name
            logger.debug(msg)
            p.PrintLine('// '+msg)
            for fifo_length in buf['FIFOs'].keys():
                p.PrintLine('FIFO_%d_%s_ptr = FIFO_%d_%s_ptr==uint%d_t(%d-1) ? 0 : FIFO_%d_%s_ptr+1;' % (fifo_length/unroll_factor, b.name, fifo_length/unroll_factor, b.name, 2**math.ceil(math.log2(math.log2(fifo_length/unroll_factor))) ,fifo_length/unroll_factor, fifo_length/unroll_factor, b.name))
            p.PrintLine()

    if produce_consume_ratio_o <= 1:
        p.DoScope()
        for c in range(output_chan):
            p.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join([('tmp_chan_%d_bank_%d' % (c, x)) for x in range(dram_bank*consume_produce_ratio_o)])))
        p.PrintLine('store_coalesced:', 0)
        p.PrintLine('for(int j = 0; j < BURST_WIDTH/%d; ++j)' % pixel_width_o)
        p.DoScope()
        p.PrintLine('#pragma HLS unroll', 0)
        for c in range(output_chan):
            for i in range(consume_produce_ratio_o):
                for j in range(dram_bank):
                    if IsFloat(output_type):
                        p.PrintLine('%s raw_bits_chan_%d_bank_%d = buffer_%s_chan_%d[BURST_WIDTH/%d*%d*%d+j*%d+%d];' % (output_type, c, i*dram_bank+j, output_name, c, pixel_width_o, dram_bank, i, dram_bank, j))
                        p.PrintLine('tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d) = *(uint%d_t*)(&raw_bits_chan_%d_bank_%d);' % (c, i*dram_bank+j, pixel_width_o, pixel_width_o, type_width[output_type], c, i*dram_bank+j))
                    else:
                        p.PrintLine('tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d) = buffer_%s_chan_%d[BURST_WIDTH/%d*%d*%d+j*%d+%d];' % (c, i*dram_bank+j, pixel_width_o, pixel_width_o, output_name, c, pixel_width_o, dram_bank, i, dram_bank, j))
        p.UnScope()
        for c in range(output_chan):
            for i in range(consume_produce_ratio_o):
                for j in range(dram_bank):
                    p.PrintLine('to_chan_%d_bank_%d<<tmp_chan_%d_bank_%d;' % (c, j, c, i*dram_bank+j))
        p.UnScope()
    else:
        p.PrintLine('if((epoch&%d)==%d)' % ((produce_consume_ratio_o-1), produce_consume_ratio_o-1))
        p.DoScope()
        for c in range(output_chan):
            p.PrintLine('ap_uint<BURST_WIDTH> %s;' % (', '.join(['tmp_chan_%d_bank_%d' % (c, x) for x in range(dram_bank)])))
        p.PrintLine('store_coalesced:', 0)
        p.PrintLine('for(int j = 0; j < BURST_WIDTH/%d; ++j)' % pixel_width_o)
        p.DoScope()
        p.PrintLine('#pragma HLS unroll', 0)
        for c in range(output_chan):
            for i in range(dram_bank):
                if IsFloat(output_type):
                    p.PrintLine('%s raw_bits_chan_%d_bank_%d = buffer_%s_chan_%d[j*%d+%d];' % (output_type, c, i, output_name, c, dram_bank, i))
                    p.PrintLine('tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d) = *(uint%d_t*)(&raw_bits_chan_%d_bank_%d);' % (c, i, pixel_width_o, pixel_width_o, type_width[output_type], c, i))
                else:
                    p.PrintLine('tmp_chan_%d_bank_%d((j+1)*%d-1, j*%d) = buffer_%s_chan_%d[j*%d+%d];' % (c, i, pixel_width_o, pixel_width_o, output_name, c, dram_bank, i))
        p.UnScope()
        for c in range(output_chan):
            for i in range(dram_bank):
                p.PrintLine('to_chan_%d_bank_%d<<tmp_chan_%d_bank_%d;' % (c, i, c, i))
        p.UnScope()

    if print_aux:
        p.PrintLine()
        for b in stencil.GetProducerBuffers():
            stencil_distance = stencil.output.offset.get() - b.offset.get()
            # TODO: calculate for more than 2 dims
            p.PrintLine('p_base_%s_counter++;' % b.name)
            if len(tile_size)>1:
                p.PrintLine('if(p_base_%s_counter == tile_data_num)' % b.name)
                p.DoScope()
                p.PrintLine('p_base_%s_counter = 0;' % b.name)
                p.PrintLine('reset_bases_%s:' % b.name, 0)
                p.PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
                p.DoScope()
                p.PrintLine('#pragma HLS unroll', 0)
                p.PrintLine('i_base_%s[unroll_index] = unroll_index-%d; // STENCIL_DISTANCE' % (b.name, stencil_distance))
                p.PrintLine('j_base_%s[unroll_index] = 0;' % b.name)
                p.UnScope()
                p.PrintLine('p_base_%s += TILE_SIZE_DIM_0-%d+1; // TILE_SIZE_DIM_0-STENCIL_DIM_0+1' % (b.name, GetStencilDim(overall_stencil_window)[0]))
                p.UnScope()

    p.UnScope()
    p.UnScope()

def PrintInterface(p, stencil):
    tile_size = stencil.tile_size
    unroll_factor = stencil.unroll_factor
    app_name = stencil.app_name
    extra_params = stencil.extra_params
    dram_bank = stencil.dram_bank
    dram_separate = stencil.dram_separate
    input_chan = stencil.input.chan
    output_chan = stencil.output.chan

    p.PrintLine('extern "C"')
    p.PrintLine('{')
    p.PrintLine()
    p.PrintLine('void %s_kernel(' % app_name)
    p.DoIndent()
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('ap_uint<BURST_WIDTH>* var_output_%d_%d,' % (c, i))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('ap_uint<BURST_WIDTH>* var_input_%d_%d,' % (c, i))
    if extra_params:
        for param in extra_params.values():
            p.PrintLine('%s* var_%s,' % (param.type, param.name))
    p.PrintLine('int64_t coalesced_data_num,')
    p.PrintLine('int64_t tile_data_num,')
    for i in range(len(tile_size)-1):
        p.PrintLine('int32_t tile_num_dim_%d,' % i)
    p.PrintLine('int32_t input_size_dim_%d)' % (len(tile_size)-1))
    p.UnIndent()
    p.DoScope()

    bank = 0
    for i in range(dram_bank):
        for c in range(output_chan):
            p.PrintLine('#pragma HLS interface m_axi port=var_output_%d_%d offset=slave depth=65536 bundle=chan%dbank%d latency=120' % (c, i, c, bank), 0)
        bank += 1
    if not dram_separate:
        bank = 0
    for i in range(dram_bank):
        for c in range(input_chan):
            p.PrintLine('#pragma HLS interface m_axi port=var_input_%d_%d offset=slave depth=65536 bundle=chan%dbank%d latency=120' % (c, i, c, bank), 0)
        bank += 1
    if extra_params:
        for idx, param in enumerate(extra_params.values()):
            p.PrintLine('#pragma HLS interface m_axi port=var_%s offset=slave depth=%d bundle=gmem%d latency=120' % (param.name, reduce(operator.mul, param.size), idx), 0)
    p.PrintLine()
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS interface s_axilite port=var_output_%d_%d bundle=control' % (c, i), 0)
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS interface s_axilite port=var_input_%d_%d bundle=control' % (c, i), 0)
    if extra_params:
        for param in extra_params.values():
            p.PrintLine('#pragma HLS interface s_axilite port=var_%s bundle=control' % param.name, 0)
    p.PrintLine('#pragma HLS interface s_axilite port=coalesced_data_num bundle=control', 0)
    p.PrintLine('#pragma HLS interface s_axilite port=tile_data_num bundle=control', 0)
    for i in range(len(tile_size)-1):
        p.PrintLine('#pragma HLS interface s_axilite port=tile_num_dim_%d bundle=control' % i, 0)
    p.PrintLine('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % (len(tile_size)-1), 0)
    p.PrintLine('#pragma HLS interface s_axilite port=return bundle=control', 0)
    p.PrintLine()

    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >  input_stream_%d_%d( "input_stream_%d_%d");' % ((c, i)*2))
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> > output_stream_%d_%d("output_stream_%d_%d");' % ((c, i)*2))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS stream variable=input_stream_%d_%d depth=32' % (c, i), 0)
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS stream variable=output_stream_%d_%d depth=32' % (c, i), 0)
    p.PrintLine()

    if extra_params:
        for param in extra_params.values():
            if param.dup:
                dup = ('[%d]' % param.dup)
            else:
                dup = ''
            p.PrintLine('%s %s%s[UNROLL_FACTOR][%s];' % (param.type, param.name, dup, ']['.join([str(x) for x in param.size])))
        p.PrintLine()

        for param in extra_params.values():
            p.PrintLine('#pragma HLS array_partition variable=%s complete dim=1' % param.name, 0)
            dim_offset = 1
            if param.dup:
                p.PrintLine('#pragma HLS array_partition variable=%s complete dim=2' % param.name, 0)
                dim_offset = 2
            for partitioning in param.partitioning:
                p.PrintLine('#pragma HLS array_partition variable=%s %s dim=%d%s' % (
                    param.name,
                    partitioning.partition_type,
                    dim_offset+1 if partitioning.dim is None else partitioning.dim+dim_offset,
                    '' if partitioning.factor is None else ' factor=%d' % partitioning.factor,
                ), 0)
        p.PrintLine()

        for param in extra_params.values():
            if len(param.size) > 1:
                for dim, size in enumerate(param.size):
                    p.PrintLine('uint32_t %s_index_dim_%d = 0;' % (param.name, dim))
            p.PrintLine('%s_init:' % param.name, 0)
            p.PrintLine('for(int %s_index = 0; %s_index < %d; ++%s_index)' % (param.name, param.name, reduce(operator.mul, param.size), param.name))
            p.DoScope()
            p.PrintLine('#pragma HLS pipeline II=1', 0)
            p.PrintLine('%s& %s_tmp = var_%s[%s_index];' % (param.type, param.name, param.name, param.name))
            p.PrintLine('%s_unrolled:' % param.name, 0)
            p.PrintLine('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
            p.DoScope()
            p.PrintLine('#pragma HLS unroll',0)
            if param.dup is None:
                p.PrintLine('%s[unroll_index]%s = %s_tmp;' % ((param.name, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name)))
            else:
                for i in range(param.dup):
                    p.PrintLine('%s[%d][unroll_index]%s = %s_tmp;' % (param.name, i, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name))
            p.UnScope()
            if len(param.size) > 1:
                for dim in range(len(param.size)):
                    p.PrintLine('++%s_index_dim_%d;' % (param.name, dim))
                    if dim<len(param.size)-1:
                        p.PrintLine('if(%s_index_dim_%d==%d)' % (param.name, dim, param.size[len(param.size)-1-dim]))
                        p.DoScope()
                        p.PrintLine('%s_index_dim_%d = 0;' % (param.name, dim))
            for size in param.size[:-1]:
                p.UnScope()
            p.UnScope()
        p.PrintLine()

    extra_params_str = ''.join([param.name+', ' for param in extra_params.values()])

    p.PrintLine('#pragma HLS dataflow', 0)
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('load(input_stream_%d_%d, var_input_%d_%d, coalesced_data_num);' % ((c, i)*2))
    output_stream = ', '.join([', '.join(['output_stream_%d_%d' % (c, x) for x in range(dram_bank)]) for c in range(output_chan)])
    input_stream = ', '.join([', '.join(['input_stream_%d_%d' % (c, x) for x in range(dram_bank)]) for c in range(input_chan)])
    tile_num_dim = ', '.join(['tile_num_dim_%d' % x for x in range(len(tile_size)-1)])
    p.PrintLine('compute(%s, %s, %scoalesced_data_num, tile_data_num, %s, input_size_dim_%d);' % (output_stream, input_stream, extra_params_str, tile_num_dim, len(tile_size)-1))
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('store(var_output_%d_%d, output_stream_%d_%d, coalesced_data_num);' % ((c, i)*2))

    p.UnScope()
    p.PrintLine()
    p.PrintLine('}//extern "C"')

def PrintHeader(p):
    for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int', 'hls_stream']:
        p.PrintLine('#include<%s.h>' % header)
    p.PrintLine()

def PrintLoad(p):
    p.PrintLine('void load(hls::stream<ap_uint<BURST_WIDTH> > &to, ap_uint<BURST_WIDTH>* from, int data_num)')
    p.DoScope()
    p.PrintLine('load_epoch:', 0)
    p.PrintLine('for(int i = 0; i < data_num; ++i)')
    p.DoScope()
    p.PrintLine('#pragma HLS pipeline II=1', 0)
    p.PrintLine('to<<from[i];')
    p.UnScope()
    p.UnScope()

def PrintStore(p):
    p.PrintLine('void store(ap_uint<BURST_WIDTH>* to, hls::stream<ap_uint<BURST_WIDTH> > &from, int data_num)')
    p.DoScope()
    p.PrintLine('store_epoch:', 0)
    p.PrintLine('for(int i = 0; i < data_num; ++i)')
    p.DoScope()
    p.PrintLine('#pragma HLS pipeline II=1', 0)
    p.PrintLine('from>>to[i];')
    p.UnScope()
    p.UnScope()

def PrintCode(s, output_file):
    logger.info('Generate kernel code as %s' % output_file.name)
    p = Printer(output_file)

    PrintHeader(p)

    p.PrintLine()

    PrintDefine(p, 'BURST_WIDTH', s.burst_width)
    p.PrintLine()

    PrintGuard(p, 'UNROLL_FACTOR', s.unroll_factor)
    for i in range(len(s.tile_size)-1):
        PrintGuard(p, 'TILE_SIZE_DIM_%d' % i, s.tile_size[i])
    PrintGuard(p, 'BURST_WIDTH', s.burst_width)
    p.PrintLine()

    PrintLoad(p)
    p.PrintLine()
    PrintStore(p)
    p.PrintLine()
    PrintCompute(p, s)
    p.PrintLine()
    PrintInterface(p, s)

