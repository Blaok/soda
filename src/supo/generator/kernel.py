#!/usr/bin/python3.6
from collections import deque
from fractions import Fraction
from functools import reduce
import itertools
import json
import logging
import math
import operator
import os
import sys

from supo.generator.utils import *
from supo.grammar import ExtraParam

logger = logging.getLogger('__main__').getChild(__name__)

def PrintComputeStage(printer, stencil, stage):
    reuse_buffers = stencil.GetReuseBuffers()
    printer.PrintLine('template<uint32_t pe_id>')
    printer.PrintLine('void compute_%s(%s,' % (stage.name, ', '.join('hls::stream<%s>& %s_chan_%d' % (stage.output.type, stage.name, c) for c in range(stencil.buffers[stage.name].chan))))
    printer.DoIndent()
    for param in [(stencil.buffers[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, GetIndicesId(indices))) for input_name, input_window in stage.window.items() for indices in input_window for c in range(stencil.buffers[input_name].chan)]:
        printer.PrintLine('hls::stream<%s>& %s,' % param)
    for d in range(stencil.dim):
        printer.PrintLine('uint32_t input_size_dim_%d,' % d)
    printer.PrintLine('uint64_t epoch_num)')
    printer.UnIndent()
    printer.DoScope()

    stencil_window = GetOverallStencilWindow(stencil.input, stage.output)
    overall_idx = GetStencilWindowOffset(stencil_window)
    delay = GetStencilDistance(stencil_window, stencil.tile_size) - Serialize(overall_idx, stencil.tile_size)
    if stage.PreserveBorderFrom():
        msg = 'aux parameters for %s' % stage.name
        logger.debug('generate '+msg)
        printer.PrintLine('// '+msg)
        printer.PrintLine('int32_t i = pe_id-%d;' % delay)
        for i in range(1, len(stencil.tile_size)):
            printer.PrintLine('uint16_t %c = 0;' % coords_in_tile[i])
        for i in range(len(stencil.tile_size)-1):
            printer.PrintLine('uint16_t %c_base = 0;' % coords_in_orig[i])
        printer.PrintLine()

    printer.PrintLine('compute_%s_epoch:' % stage.name, 0)
    printer.PrintLine('for(uint64_t epoch = 0; epoch < epoch_num; ++epoch)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)

    if stage.PreserveBorderFrom():
        for i in range(len(stencil.tile_size)-1):
            printer.PrintLine('uint16_t  %c = %c_base+%c;' % (coords_in_orig[i], coords_in_orig[i], coords_in_tile[i]))
        printer.PrintLine('uint16_t& %c = %c;' % (coords_in_orig[len(stencil.tile_size)-1], coords_in_tile[len(stencil.tile_size)-1]))

        IndexTile = lambda d: '%c' % (coords_in_tile[d])
        IndexOrig = lambda d: '%c' % (coords_in_orig[d])
        output_idx = GetStencilWindowOffset(stencil_window)
        stencil_dim = GetStencilDim(stencil_window)
        MarginCondition = lambda d: ('%s<%d || ' % (IndexOrig(d), output_idx[d]) if output_idx[d]>0 else '') + '%s>input_size_dim_%d-%d+%d' % (IndexOrig(d), d, stencil_dim[d], output_idx[d])
        for d in range(stencil.dim):
            printer.PrintLine('bool margin_condition_dim_%d[1];' % d)
            printer.PrintLine('#pragma HLS resource variable=margin_condition_dim_%d latency=1 core=RAM_2P_LUTRAM' % d, 0)
            printer.DoScope()
            printer.PrintLine('#pragma HLS latency min=1', 0)
            printer.PrintLine('margin_condition_dim_%d[0] = %s;' % (d, MarginCondition(d)))
            printer.UnScope()
        printer.PrintLine()

    for input_name, input_window in stage.window.items():
        params = []
        for indices in input_window:
            for c in range(stencil.buffers[input_name].chan):
                params.append((stencil.buffers[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, GetIndicesId(indices))))

        for param in params:
            printer.PrintLine('%s load_%s = %s.read();' % (param[0], param[1], param[1]))
        printer.PrintLine()

    if stage.PreserveBorderFrom():
        printer.PrintLine('if(%s)' % (' || '.join('margin_condition_dim_%d[0]' % d for d in range(stencil.dim))))
        printer.DoScope()
        preserve_border_from = stage.PreserveBorderFrom()
        printer.PrintLine('%s_chan_%d<<load_%s_chan_%d_at_%s;' % (stage.name, c, preserve_border_from.name, c, GetIndicesId(stage.idx)))
        printer.UnScope()
        printer.PrintLine('else')
        printer.DoScope()

    LoadPrinter = lambda node: 'param_%s%s[unroll_index]%s' % (node.name, '' if stencil.extra_params[node.name].dup is None else '[%d]' % node.chan, ''.join(['[%d]'%x for x in node.idx])) if node.name in stencil.extra_params else 'load_%s_chan_%d_at_%s' % (node.name, node.chan, GetIndicesId(node.idx))
    StorePrinter = lambda node: '%s store_%s_chan_%d' % (stage.output.type, node.name, node.chan)

    for expr in stage.expr:
        expr.PrintCode(printer, stencil.buffers, LoadPrinter, StorePrinter, add_latency=True)

    for c in range(stage.output.chan):
        printer.PrintLine('%s_chan_%d<<store_%s_chan_%d;' % ((stage.name, c)*2))
        #printer.PrintLine('printf("epoch%%d pe%%d %%d %s %%d\\n", epoch, pe_id, epoch*%d+pe_id-%d, %s, store_%s_chan_%d);' % (' '.join(['%d']*len(params)), stencil.unroll_factor, delay, ', '.join('load_%s'%p[1] for p in params), stage.name, c))

    if stage.PreserveBorderFrom():
        printer.UnScope()
        printer.PrintLine()
        overall_stencil_window = GetOverallStencilWindow(stencil.input, stencil.output)
        overall_stencil_dim = GetStencilDim(overall_stencil_window)
        PrintIfTile = lambda d: printer.PrintLine('if(%c>=TILE_SIZE_DIM_%d)' % (coords_in_tile[d], d))
        PrintIfTileLastDim = lambda d: printer.PrintLine('if(%c >= input_size_dim_%d)' % (coords_in_tile[d], d))
        PrintIfTensor = lambda d: printer.PrintLine('if(%c >= input_size_dim_%d)' % (coords_in_orig[d], d))
        PrintIncrementTile = lambda d: printer.PrintLine('++%c;' % (coords_in_tile[d]))
        PrintDecrementTile = lambda d: printer.PrintLine('%c -= TILE_SIZE_DIM_%d;' % (coords_in_tile[d], d))
        PrintIncrementOrig = lambda d: printer.PrintLine('%c_base += TILE_SIZE_DIM_%d - %s + 1;' % (coords_in_orig[d], d, overall_stencil_dim[d]))
        PrintDecrementOrig = lambda d: printer.PrintLine('%c = 0;' % (coords_in_orig[d], s.name))
        PrintDecrementTileLastDim = lambda d: printer.PrintLine('%c -= input_size_dim_%d;' % (coords_in_tile[d], d))

        printer.PrintLine('i+=%d;' % stencil.unroll_factor)
        if len(stencil.tile_size)>1:
            PrintIfTile(0)
            printer.DoScope()
            printer.PrintLine('#pragma HLS latency min=1', 0)
            PrintDecrementTile(0)
            PrintIncrementTile(1)
            if len(stencil.tile_size)>2:
                PrintIfTile(1)
                printer.DoScope()
                printer.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTile(1)
                PrintIncrementTile(2)
                if len(stencil.tile_size)>3:
                    PrintIfTile(2)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTile(2)
                    PrintIncrementTile(3)

                    PrintIfTileLastDim(3)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTileLastDim(3)
                    PrintIncrementOrig(0)
                    PrintIfTensor(0)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementOrig(0)
                    PrintIncrementOrig(1)
                    PrintIfTensor(1)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementOrig(1)
                    PrintIncrementOrig(2)
                    printer.UnScope()
                    printer.UnScope()
                    printer.UnScope()

                    printer.UnScope()
                else:
                    PrintIfTileLastDim(2)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTileLastDim(2)
                    PrintIncrementOrig(0)

                    PrintIfTensor(0)
                    printer.DoScope()
                    printer.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementOrig(0)
                    PrintIncrementOrig(1)
                    printer.UnScope()

                    printer.UnScope()
                printer.UnScope()
            else:
                PrintIfTileLastDim(1)
                printer.DoScope()
                printer.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTileLastDim(1)
                PrintIncrementOrig(0)
                printer.UnScope()
            printer.UnScope()
        else:
            PrintIfTileLastDim(0)
            printer.DoScope()
            printer.PrintLine('#pragma HLS latency min=1', 0)
            PrintDecrementTileLastDim(0)
            printer.UnScope()


    printer.UnScope()
    printer.UnScope()
    printer.PrintLine()

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
    for d in range(stencil.dim-1):
        p.PrintLine('int32_t tile_num_dim_%d,' % d)
    for d in range(stencil.dim-1):
        p.PrintLine('int32_t input_size_dim_%d,' % d)
    p.PrintLine('int32_t input_size_dim_%d)' % (stencil.dim-1))
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

    for b in stencil.GetConsumerBuffers():
        if b.parent.PreserveBorderFrom():
            msg = 'aux parameters for %s' % b.name
            logger.debug('generate '+msg)
            p.PrintLine('// '+msg)
            stencil_window = GetOverallStencilWindow(stencil.input, b)
            overall_idx = GetStencilWindowOffset(stencil_window)
            delay = GetStencilDistance(stencil_window, stencil.tile_size) - Serialize(overall_idx, stencil.tile_size)
            p.PrintLine('int32_t i_base_%s = %d;' % (b.name, -delay))
            for i in range(1, len(tile_size)):
                p.PrintLine('int32_t %c_base_%s = 0;' % (coords_in_tile[i], b.name))
            for i in range(len(tile_size)-1):
                p.PrintLine('int32_t %c_base_%s = 0;' % (coords_in_orig[i], b.name))
            p.PrintLine()

    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        points_from_b = all_points[b.name]

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
                p.PrintLine('#pragma HLS resource variable=FF_%s_chan_%d latency=1' % (b.name, c), 0)

        for c in range(b.chan):
            for fifo_length in buf['FIFOs'].keys():
                p.PrintLine('#pragma HLS array_partition variable=FIFO_%d_%s_chan_%d complete dim=1' % (fifo_length/unroll_factor, b.name, c), 0)
        p.PrintLine()

        for fifo_length in buf['FIFOs'].keys():
            ptr_width = 2**math.ceil(math.log2(math.log2(fifo_length/unroll_factor)))
            if ptr_width < 8:
                ptr_width = 8
            elif ptr_width > 64:
                ptr_width = 64
            p.PrintLine('uint%d_t FIFO_%d_%s_ptr = 0;' % (ptr_width, fifo_length/unroll_factor, b.name))
        p.PrintLine()

        msg = 'points aliases for %s' % b.name
        logger.debug('generate '+msg)
        p.PrintLine('// '+msg)
        for s in b.children:
            for c in range(b.chan):
                p.PrintLine('%s points_from_%s_to_%s_chan_%d[UNROLL_FACTOR][%d];' % (b.type, b.name, s.name, c, len(s.window[b.name])))
            for idx, point in enumerate(s.window[b.name]):
                p.PrintLine('//%s points_from_%s_to_%s_chan_x[UNROLL_FACTOR][%d] <=> %s[x](%s)' % (' '*(len(b.type)-2), b.name, s.name, idx, b.name, ', '.join(map(str, point))))
            for c in range(b.chan):
                p.PrintLine('#pragma HLS array_partition variable=points_from_%s_to_%s_chan_%d complete dim=0' % (b.name, s.name, c), 0)
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
        p.PrintLine('#pragma HLS resource variable=buffer_%s_chan_%d latency=1' % (input_name, c), 0)
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
            p.PrintLine('#pragma HLS resource variable=buffer_%s_chan_%d latency=1' % (b.name, c), 0)
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
        p.PrintLine('#pragma HLS resource variable=buffer_%s_chan_%d latency=1' % (output_name, c), 0)
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
    for s in stencil.GetStagesChronologically():
        # start emitting code for stage s
        logger.debug('emit code for stage %s' % s.name)
        # connect points from previous buffer/FIFO/FF
        for bb in s.inputs.values():
            buf = reuse_buffers[bb.name]
            points = all_points[bb.name][s.name]
            logger.debug('%s <- %s points: %s' % (s.name, bb.name, points))
            logger.debug('%s <- %s buf: %s' % (s.name, bb.name, buf))
            for idx, offset in enumerate(buf['inputs']):
                for unroll_index, point in points.get(offset, {}).items():
                    for c in range(bb.chan):
                        p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = buffer_%s_chan_%d[%d]; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.name, c, unroll_index, point, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))
            for idx, offset in enumerate(buf['FFs']):
                for unroll_index, point in points.get(offset, {}).items():
                    for c in range(bb.chan):
                        p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = FF_%s_chan_%d[%d]; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.name, c, unroll_index, point, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))
            for fifo_length in buf['FIFOs'].keys():
                for idx, offset in enumerate(buf['FIFOs'][fifo_length]):
                    for unroll_index, point in points.get(offset, {}).items():
                        for c in range(bb.chan):
                            p.PrintLine("points_from_%s_to_%s_chan_%d[%d][%d] = FIFO_%d_%s_chan_%d_fifo_%d; // %s[%d](%s) @ unroll_index=%d" % (bb.name, s.name, c, unroll_index, point, fifo_length/unroll_factor, bb.name, c, idx, bb.name, c, ', '.join([str(x) for x in s.window[bb.name][point]]), unroll_index))

        p.PrintLine()

        p.PrintLine('compute_%s_unrolled:' % s.name, 0)
        p.PrintLine('for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
        p.DoScope('unroll_index')
        p.PrintLine('#pragma HLS unroll', 0)
        p.PrintLine('#pragma HLS latency min=1', 0)

        if s.PreserveBorderFrom():
            p.PrintLine('int32_t %c_%s = %c_base_%s+unroll_index;' % ((coords_in_tile[0], s.name)*2))
            for i in range(1,len(tile_size)):
                p.PrintLine('int32_t %c_%s = %c_base_%s;' % ((coords_in_tile[i], s.name)*2))
            for i in range(len(tile_size)-1):
                p.PrintLine('int32_t %c_%s = %c_base_%s+%c_%s;' % ((coords_in_orig[i], s.name)*2+(coords_in_tile[i], s.name)))
            p.PrintLine('int32_t %c_%s = %c_%s;' % (coords_in_orig[len(tile_size)-1], s.name, coords_in_tile[len(tile_size)-1], s.name))
            overall_stencil_dim = GetStencilDim(overall_stencil_window)
            PrintIfTile = lambda d: p.PrintLine('if(%c_%s>=TILE_SIZE_DIM_%d)' % (coords_in_tile[d], s.name, d))
            PrintIfTileLastDim = lambda d: p.PrintLine('if(%c_%s >= input_size_dim_%d)' % (coords_in_tile[d], s.name, d))
            PrintIfTensor = lambda d: p.PrintLine('if(%c_%s >= input_size_dim_%d)' % (coords_in_orig[d], s.name, d))
            PrintIncrementTile = lambda d: p.PrintLine('++%c_%s;' % (coords_in_tile[d], s.name))
            PrintDecrementTile = lambda d: p.PrintLine('%c_%s -= TILE_SIZE_DIM_%d;' % (coords_in_tile[d], s.name, d))
            PrintIncrementOrig = lambda d: p.PrintLine('%c_%s += TILE_SIZE_DIM_%d - %s + 1;' % (coords_in_orig[d], s.name, d, overall_stencil_dim[d]))
            PrintDecrementOrig = lambda d: p.PrintLine('%c_%s = 0;' % (coords_in_orig[d], s.name))
            PrintDecrementTileLastDim = lambda d: p.PrintLine('%c_%s -= input_size_dim_%d;' % (coords_in_tile[d], s.name, d))

            if len(tile_size)>1:
                PrintIfTile(0)
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTile(0)
                PrintIncrementTile(1)
                if len(tile_size)>2:
                    PrintIfTile(1)
                    p.DoScope()
                    p.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTile(1)
                    PrintIncrementTile(2)
                    if len(tile_size)>3:
                        PrintIfTile(2)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTile(2)
                        PrintIncrementTile(3)

                        PrintIfTileLastDim(3)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTileLastDim(3)
                        PrintIncrementOrig(0)
                        PrintIfTensor(0)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(0)
                        PrintIncrementOrig(1)
                        PrintIfTensor(1)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(1)
                        PrintIncrementOrig(2)
                        p.UnScope()
                        p.UnScope()
                        p.UnScope()

                        p.UnScope()
                    else:
                        PrintIfTileLastDim(2)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTileLastDim(2)
                        PrintIncrementOrig(0)

                        PrintIfTensor(0)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(0)
                        PrintIncrementOrig(1)
                        p.UnScope()

                        p.UnScope()
                    p.UnScope()
                else:
                    PrintIfTileLastDim(1)
                    p.DoScope()
                    p.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTileLastDim(1)
                    PrintIncrementOrig(0)
                    p.UnScope()
                p.UnScope()
            else:
                PrintIfTileLastDim(0)
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTileLastDim(0)
                p.UnScope()
            p.PrintLine()

        for bb in s.inputs.values():
            for c in range(bb.chan):
                for idx, point in enumerate(s.window[bb.name]):
                    p.PrintLine('%s& load_%s_for_%s_chan_%d_at_%s = points_from_%s_to_%s_chan_%d[unroll_index][%d];' % (bb.type, bb.name, s.name, c, '_'.join([str(x).replace('-', 'm') for x in point]), bb.name, s.name, c, idx))
        p.PrintLine()

        LoadPrinter = lambda node: 'param_%s%s[unroll_index]%s' % (node.name, '' if extra_params[node.name].dup is None else '[%d]' % node.chan, ''.join(['[%d]'%x for x in node.idx])) if node.name in extra_params else 'load_%s_for_%s_chan_%d_at_%s' % (node.name, s.name, node.chan, '_'.join([str(x).replace('-', 'm') for x in node.idx]))
        StorePrinter = lambda node: 'result_chan_%d' % node.chan if node.name == output_name else 'buffer_%s_chan_%d[unroll_index]' % (node.name, node.chan)

        if s.name == output_name:
            for c in range(s.output.chan):
                p.PrintLine('%s result_chan_%d;' % (s.output.type, c))

        if s.PreserveBorderFrom():
            p.PrintLine()
            bb = s.PreserveBorderFrom()
            stencil_window = GetOverallStencilWindow(bb, s.output)
            stencil_dim = GetStencilDim(stencil_window)
            output_idx = GetStencilWindowOffset(stencil_window)
            IndexTile = lambda d: '%c_%s' % (coords_in_tile[d], s.name)
            IndexOrig = lambda d: '%c_%s' % (coords_in_orig[d], s.name)
            MarginCondition = lambda d: ('(0<=%s && %s<%d) || ' % (IndexOrig(d), IndexOrig(d), output_idx[d]) if output_idx[d]>0 else '') + '%s>input_size_dim_%d-%d+%d' % (IndexOrig(d), d, stencil_dim[d], output_idx[d])
            for d in range(stencil.dim):
                p.PrintLine('bool margin_condition_dim_%d[1];' % d)
                p.PrintLine('#pragma HLS resource variable=margin_condition_dim_%d latency=1 core=RAM_2P_LUTRAM' % d, 0)
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                p.PrintLine('margin_condition_dim_%d[0] = %s;' % (d, MarginCondition(d)))
                p.UnScope()
            p.PrintLine('if(%s>=0 && (%s))' % (IndexTile(0), ' || '.join('margin_condition_dim_%d[0]' % d for d in range(stencil.dim))))
            p.DoScope()
            p.PrintLine('// forward input to output directly and preserve tensor border')
            for c in range(s.output.chan):
                if s.name == output_name:
                    dst = 'result_chan_%d' % c
                else:
                    dst = 'buffer_%s_chan_%d[unroll_index]' % (s.name, c)
                p.PrintLine('#pragma HLS latency min=1', 0)
                p.PrintLine('%s = load_%s_for_%s_chan_%d_at_%s;' % (dst, bb.name, s.name, c, '_'.join([str(x).replace('-', 'm') for x in s.idx])))
            p.UnScope()
            p.PrintLine('else')
            p.DoScope()
            for e in s.expr:
                p.PrintLine('#pragma HLS latency min=1', 0)
                e.PrintCode(p, stencil.buffers, LoadPrinter, StorePrinter, add_latency=True)
            p.UnScope()
            #for d in range(stencil.dim-1):
            #    if stencil_dim[d] < 2:
            #        continue
            #    p.PrintLine('if(%s >= %d-1 && %s < input_size_dim_0-%d+1)' % (IndexOrig(d), stencil_dim[d], IndexOrig(d), stencil_dim[d]))
            #    p.DoScope()
            #    p.PrintLine('switch(%s)' % IndexTile(d))
            #    p.DoScope()
            #    for i in itertools.chain(range(output_idx[d], stencil_dim[d]-1), range(stencil.tile_size[d]-stencil_dim[d]+1, stencil.tile_size[d]-stencil_dim[d]+output_idx[d]+1)):
            #        p.PrintLine('case %d:' % i)
            #        p.DoScope()
            #        p.PrintLine('// duplicate output to border buffer')
            #        p.PrintLine('break;')
            #        p.UnScope()
            #    p.UnScope()
            #    p.UnScope()
        else:
            for e in s.expr:
                e.PrintCode(p, stencil.buffers, LoadPrinter, StorePrinter, add_latency=True)

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
                    p.PrintLine('break;')
                    p.UnScope()
                p.UnScope()

        p.UnScope()
        p.PrintLine()
        # finish emitting code

    for b in stencil.GetProducerBuffers():
        buf = reuse_buffers[b.name]
        first = True
        for idx, item in enumerate(buf['inputs']):
            if first:
                first = False
            else:
                p.PrintLine()
            msg = 'move reuse chain %d for buffer %s' % (idx, b.name)
            logger.debug(msg)
            p.PrintLine('// '+msg)
            PrintUpdate(p, unroll_factor, GetProduceMap(buf), GetConsumeMap(buf, unroll_factor), {'inputs':{'index':idx, 'produce':item}}, b.chan, b.name)

        if len(buf['FIFOs'])>0:
            p.PrintLine()
            msg = 'move FIFO ptrs for buffer %s' % b.name
            logger.debug(msg)
            p.PrintLine('// '+msg)
            for fifo_length in buf['FIFOs'].keys():
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                ptr_width = 2**math.ceil(math.log2(math.log2(fifo_length/unroll_factor)))
                if ptr_width < 8:
                    ptr_width = 8
                elif ptr_width > 64:
                    ptr_width = 64
                p.PrintLine('FIFO_%d_%s_ptr = FIFO_%d_%s_ptr==uint%d_t(%d-1) ? 0 : FIFO_%d_%s_ptr+1;' % (fifo_length/unroll_factor, b.name, fifo_length/unroll_factor, b.name, ptr_width ,fifo_length/unroll_factor, fifo_length/unroll_factor, b.name))
                p.UnScope()


    if stencil.iterate:
        for b in stencil.GetConsumerBuffers():
            s = b.parent
            if s.PreserveBorderFrom() is None:
                continue
            p.DoScope()
            p.PrintLine('#pragma HLS latency min=1', 0)

            overall_stencil_dim = GetStencilDim(overall_stencil_window)
            PrintIfTile = lambda d: p.PrintLine('if(%c_%s>=TILE_SIZE_DIM_%d)' % (coords_in_tile[d], s.name, d))
            PrintIfTileLastDim = lambda d: p.PrintLine('if(%c_%s >= input_size_dim_%d)' % (coords_in_tile[d], s.name, d))
            PrintIfTensor = lambda d: p.PrintLine('if(%c_%s >= input_size_dim_%d)' % (coords_in_orig[d], s.name, d))
            PrintIncrementTile = lambda d: p.PrintLine('++%c_%s;' % (coords_in_tile[d], s.name))
            PrintDecrementTile = lambda d: p.PrintLine('%c_%s -= TILE_SIZE_DIM_%d;' % (coords_in_tile[d], s.name, d))
            PrintIncrementOrig = lambda d: p.PrintLine('%c_%s += TILE_SIZE_DIM_%d - %s + 1;' % (coords_in_orig[d], s.name, d, overall_stencil_dim[d]))
            PrintDecrementOrig = lambda d: p.PrintLine('%c_%s = 0;' % (coords_in_orig[d], s.name))
            PrintDecrementTileLastDim = lambda d: p.PrintLine('%c_%s -= input_size_dim_%d;' % (coords_in_tile[d], s.name, d))

            for i in range(len(tile_size)):
                p.PrintLine('int32_t& %c_%s = %c_base_%s;' % ((coords_in_tile[i], s.name)*2))
            for i in range(len(tile_size)-1):
                p.PrintLine('int32_t& %c_%s = %c_base_%s;' % ((coords_in_orig[i], s.name)*2))
            p.PrintLine()

            p.PrintLine('%c_%s += UNROLL_FACTOR;' % (coords_in_tile[0], s.name))
            if len(tile_size)>1:
                PrintIfTile(0)
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTile(0)
                PrintIncrementTile(1)
                if len(tile_size)>2:
                    PrintIfTile(1)
                    p.DoScope()
                    p.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTile(1)
                    PrintIncrementTile(2)
                    if len(tile_size)>3:
                        PrintIfTile(2)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTile(2)
                        PrintIncrementTile(3)

                        PrintIfTileLastDim(3)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTileLastDim(3)
                        PrintIncrementOrig(0)
                        PrintIfTensor(0)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(0)
                        PrintIncrementOrig(1)
                        PrintIfTensor(1)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(1)
                        PrintIncrementOrig(2)
                        p.UnScope()
                        p.UnScope()
                        p.UnScope()

                        p.UnScope()
                    else:
                        PrintIfTileLastDim(2)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementTileLastDim(2)
                        PrintIncrementOrig(0)

                        PrintIfTensor(0)
                        p.DoScope()
                        p.PrintLine('#pragma HLS latency min=1', 0)
                        PrintDecrementOrig(0)
                        PrintIncrementOrig(1)
                        p.UnScope()

                        p.UnScope()
                    p.UnScope()
                else:
                    PrintIfTileLastDim(1)
                    p.DoScope()
                    p.PrintLine('#pragma HLS latency min=1', 0)
                    PrintDecrementTileLastDim(1)
                    PrintIncrementOrig(0)
                    p.UnScope()
                p.UnScope()
            else:
                PrintIfTileLastDim(0)
                p.DoScope()
                p.PrintLine('#pragma HLS latency min=1', 0)
                PrintDecrementTileLastDim(0)
                p.UnScope()
            p.UnScope()
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

    logger.info('generate reuse buffers')
    reuse_buffers = stencil.GetReuseBuffers()
    all_points = stencil.GetAllPoints()
    overall_stencil_window = GetOverallStencilWindow(stencil.input, stencil.output)

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
    p.PrintLine('uint64_t coalesced_data_num,')
    p.PrintLine('uint64_t tile_data_num,')
    for i in range(stencil.dim-1):
        p.PrintLine('uint32_t tile_num_dim_%d,' % i)
    for d in range(stencil.dim-1):
        p.PrintLine('uint32_t input_size_dim_%d,' % d)
    p.PrintLine('uint32_t input_size_dim_%d)' % (stencil.dim-1))
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
    for d in range(stencil.dim-1):
        p.PrintLine('#pragma HLS interface s_axilite port=tile_num_dim_%d bundle=control' % d, 0)
    for d in range(stencil.dim):
        p.PrintLine('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % d, 0)
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
            p.PrintLine('%s %s%s[UNROLL_FACTOR][%s];' % (param.type, param.name, dup, ']['.join(map(str, param.size))))
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

    GetTensorAt = lambda n, o, c: ('%s_offset_%d_chan_%d') % (n, o, c)

    # reuse buffers
    next_fifo = stencil.GetNextFIFO()
    for name, reuse_buffer in reuse_buffers.items():
        pragmas = []
        msg = 'reuse buffers for %s' % name
        logger.debug('generate %s' % msg)
        p.PrintLine('// %s' % msg)
        for start, end in reuse_buffer[1:]:
            for c in range(stencil.buffers[name].chan):
                p.PrintLine('hls::stream<%s> %s("%s");' % ((stencil.buffers[name].type,)+(GetTensorAt(name, end, c),)*2))
                if (end-start)//unroll_factor > 1:
                    pragmas.append((GetTensorAt(name, end, c), (end-start)//unroll_factor))
        for pragma in pragmas:
            p.PrintLine('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
        p.PrintLine()

    p.PrintLine('// %s' % stencil.output.name)
    pragmas = []
    for unroll_index in range(unroll_factor):
        for c in range(stencil.output.chan):
            p.PrintLine('hls::stream<%s> %s("%s");' % ((stencil.buffers[stencil.output.name].type,)+(GetTensorAt(stencil.output.name, unroll_index, c),)*2))
    p.PrintLine()

    # params
    msg = 'params'
    logger.debug('generate %s' % msg)
    p.PrintLine('// %s' % msg)
    for stage in stencil.GetStagesChronologically():
        for unroll_index in range(unroll_factor):
            for param in [(stencil.buffers[input_name].type, 'from_%s_to_%s_param_%d_chan_%d_pe_%d' % (input_name, stage.name, i, c, unroll_index)) for input_name, input_window in stage.window.items() for i in range(len(input_window)) for c in range(stencil.buffers[input_name].chan)]:
                p.PrintLine('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
    p.PrintLine()

    p.PrintLine('uint64_t epoch_num = coalesced_data_num*%d/%d;' % (stencil.burst_width/type_width[stencil.input.type], unroll_factor))
    p.PrintLine()

    p.PrintLine('#pragma HLS dataflow', 0)
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('load(input_stream_%d_%d, var_input_%d_%d, coalesced_data_num);' % ((c, i)*2))
            p.PrintLine('unpack_%s(' % GetSupoType(stencil.input.type))
            p.DoIndent()
            for unroll_index in reversed(range(unroll_factor)):
                p.PrintLine('%s,' % GetTensorAt(stencil.input.name, stencil.input.offset+unroll_index, c))
            p.PrintLine('input_stream_%d_%d, coalesced_data_num);' % (c, i))
            p.UnIndent()
    p.PrintLine()

    output_stream = ', '.join(', '.join('output_stream_%d_%d' % (c, x) for x in range(dram_bank)) for c in range(output_chan))
    input_stream = ', '.join(', '.join('input_stream_%d_%d' % (c, x) for x in range(dram_bank)) for c in range(input_chan))
    tile_num_dim = ', '.join('tile_num_dim_%d' % d for d in range(stencil.dim-1))
    input_size_dim = ', '.join('input_size_dim_%d' % d for d in range(stencil.dim))
    PrintForwarding(p, stencil, stencil.input.name)
    p.PrintLine()
    for stage in stencil.GetStagesChronologically():
        inputs = tuple(reversed(range(unroll_factor))) if stage.IsOutput() else [start for start, end in stencil.GetReuseBuffers()[stage.name][1:] if start==end]
        for unroll_index in range(unroll_factor):
            params = ['%s_offset_%s_chan_%d' % (stage.name, inputs[unroll_index], c) for c in range(stage.output.chan)]
            params += ['from_%s_to_%s_param_%d_chan_%d_pe_%d' % (input_name, stage.name, i, c, unroll_index) for input_name, input_window in stage.window.items() for i in range(len(input_window)) for c in range(stencil.buffers[input_name].chan)]
            params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
            params.append('epoch_num')
            p.PrintFunc('compute_%s<%d>' % (stage.name, unroll_index), params, ';')

        if not stage.IsOutput():
            p.PrintLine()
            PrintForwarding(p, stencil, stage.name)
        p.PrintLine()

    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('pack_%s(output_stream_%d_%d,' % (GetSupoType(stencil.output.type), c, i))
            p.DoIndent()
            for unroll_index in reversed(range(unroll_factor)):
                p.PrintLine('%s,' % GetTensorAt(stencil.output.name, unroll_index, c))
            p.PrintLine('coalesced_data_num);')
            p.UnIndent()
            p.PrintLine('store(var_output_%d_%d, output_stream_%d_%d, coalesced_data_num);' % ((c, i)*2))

    p.UnScope()
    p.PrintLine()
    p.PrintLine('}//extern "C"')

def PrintHeader(p):
    for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int', 'hls_stream']:
        p.PrintLine('#include<%s.h>' % header)
    p.PrintLine()

def PrintLoad(printer):
    printer.PrintLine('void load(hls::stream<ap_uint<BURST_WIDTH> >& to, ap_uint<BURST_WIDTH>* from, uint64_t data_num)')
    printer.DoScope()
    printer.PrintLine('load_epoch:', 0)
    printer.PrintLine('for(uint64_t i = 0; i < data_num; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('to<<from[i];')
    printer.UnScope()
    printer.UnScope()

def PrintUnpack(printer, burst_width, data_type, unroll_factor):
    coalesced_size = burst_width//type_width[data_type]
    ii = 1
    if coalesced_size > unroll_factor:
        ii = coalesced_size/unroll_factor
    GetCoalescedIdx = lambda i: ('%'+str(len(str(coalesced_size)))+'d') % i
    GetDstName = lambda i: ('to_chain_%0'+str(len(str(unroll_factor-1)))+'d') % i
    printer.PrintLine('void unpack_%s(' % GetSupoType(data_type))
    printer.DoIndent()
    for unroll_index in range(unroll_factor):
        printer.PrintLine(('hls::stream<%s>& %s,') % (data_type, GetDstName(unroll_index)))
    printer.PrintLine('hls::stream<ap_uint<%d> >& from, uint64_t data_num)' % burst_width)
    printer.UnIndent()
    printer.DoScope()
    printer.PrintLine('unpack_%s_epoch:' % GetSupoType(data_type), 0)
    printer.PrintLine('for(uint64_t i = 0; i < data_num; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=%d' % ii, 0)
    printer.PrintLine('ap_uint<%d> tmp;' % burst_width)
    printer.PrintLine('from>>tmp;')
    if IsFloat(data_type):
        printer.PrintLine('uint%d_t raw_bits;' % type_width[data_type])
    if coalesced_size >= unroll_factor:
        for i in range(coalesced_size):
            if IsFloat(data_type):
                printer.PrintLine('raw_bits = tmp(%s*%d-1, %s*%d); %s<<*((%s*)(&raw_bits));' % (GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], GetDstName(i%unroll_factor), data_type))
            else:
                printer.PrintLine('%s<<tmp(%s*%d-1, %s*%d);' % (GetDstName(i%unroll_factor), GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type]))
    else:
        printer.PrintLine('switch(i&%d)' % (unroll_factor//coalesced_size-1))
        printer.DoScope()
        for batch in range(unroll_factor//coalesced_size):
            printer.PrintLine('case %d:' % batch)
            printer.DoScope()
            for i in range(coalesced_size):
                if IsFloat(data_type):
                    printer.PrintLine('raw_bits = tmp(%s*%d-1, %s*%d);%s<<*((%s*)(&raw_bits));' % (GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], GetDstName(i+batch*coalesced_size), data_type))
                else:
                    printer.PrintLine('%s<<tmp(%s*%d-1, %s*%d);' % (GetDstName(i+batch*coalesced_size), GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type]))
            printer.PrintLine('break;')
            printer.UnScope()
        printer.UnScope()
    printer.UnScope()
    printer.UnScope()

def PrintPack(printer, burst_width, data_type, unroll_factor):
    coalesced_size = burst_width//type_width[data_type]
    ii = 1
    if coalesced_size > unroll_factor:
        ii = coalesced_size/unroll_factor
    GetCoalescedIdx = lambda i: ('%'+str(len(str(coalesced_size)))+'d') % i
    GetDstName = lambda i: ('from_chain_%0'+str(len(str(unroll_factor-1)))+'d') % i
    printer.PrintLine('void pack_%s(hls::stream<ap_uint<%d> >& to,' % (GetSupoType(data_type), burst_width))
    printer.DoIndent()
    for unroll_index in range(unroll_factor):
        printer.PrintLine(('hls::stream<%s>& %s,') % (data_type, GetDstName(unroll_index)))
    printer.PrintLine('uint64_t data_num)')
    printer.UnIndent()
    printer.DoScope()
    printer.PrintLine('pack_%s_epoch:' % GetSupoType(data_type), 0)
    printer.PrintLine('for(uint64_t i = 0; i < data_num; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=%d' % ii, 0)
    printer.PrintLine('ap_uint<%d> tmp;' % burst_width)
    if IsFloat(data_type):
        printer.PrintLine('%s raw_bits;' % data_type)
    if coalesced_size >= unroll_factor:
        for i in range(coalesced_size):
            if IsFloat(data_type):
                printer.PrintLine('%s>>raw_bits; tmp(%s*%d-1, %s*%d) = *((uint%d_t*)(&raw_bits));' % (GetDstName(i%unroll_factor), GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], type_width[data_type]))
            else:
                printer.PrintLine('tmp(%s*%d-1, %s*%d) = %s.read();' % (GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], GetDstName(i%unroll_factor)))
    else:
        printer.PrintLine('switch(i&%d)' % (unroll_factor//coalesced_size-1))
        printer.DoScope()
        for batch in range(unroll_factor//coalesced_size):
            printer.PrintLine('case %d:' % batch)
            printer.DoScope()
            for i in range(coalesced_size):
                if IsFloat(data_type):
                    printer.PrintLine('%s>>raw_bits; tmp(%s*%d-1, %s*%d) = *((uint%d_t*)(&raw_bits));' % (GetDstName(i+batch*coalesced_size), GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], type_width[data_type]))
                else:
                    printer.PrintLine('tmp(%s*%d-1, %s*%d) = %s.read();' % (GetCoalescedIdx(i+1), type_width[data_type], GetCoalescedIdx(i), type_width[data_type], GetDstName(i+batch*coalesced_size)))
            printer.PrintLine('break;')
            printer.UnScope()
        printer.UnScope()
    printer.PrintLine('to<<tmp;')
    printer.UnScope()
    printer.UnScope()

def PrintStore(printer):
    printer.PrintLine('void store(ap_uint<BURST_WIDTH>* to, hls::stream<ap_uint<BURST_WIDTH> >& from, uint64_t data_num)')
    printer.DoScope()
    printer.PrintLine('store_epoch:', 0)
    printer.PrintLine('for(uint64_t i = 0; i < data_num; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('from>>to[i];')
    printer.UnScope()
    printer.UnScope()

def PrintForwarder(printer, forwarder):
    printer.PrintFunc('template<typename T, uint32_t fifo_depth> void forward_%d' % forwarder, ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]+['hls::stream<T>& src']+['uint32_t data_num'])
    printer.DoScope()
    printer.PrintLine('forward_%d_epoch:' % forwarder, 0)
    printer.PrintLine('for(uint32_t i = 0; i < data_num+fifo_depth; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('if(i<fifo_depth)')
    printer.DoScope()
    printer.PrintLine('T tmp = 0;')
    for dst in range(forwarder):
        printer.PrintLine('dst_%d<<tmp;' % dst)
    printer.UnScope()
    printer.PrintLine('else if(i<data_num)')
    printer.DoScope()
    printer.PrintLine('T tmp(src.read());')
    for dst in range(forwarder):
        printer.PrintLine('dst_%d<<tmp;' % dst)
    printer.UnScope()
    printer.PrintLine('else')
    printer.DoScope()
    printer.PrintLine('src.read();')
    printer.UnScope()
    printer.UnScope()
    printer.UnScope()

def PrintForwarding(printer, stencil, src_name):
    next_fifo = stencil.GetNextFIFO()
    logger.debug(next_fifo)
    unroll_factor = stencil.unroll_factor
    dsts = stencil.GetAllPoints()[src_name]
    reuse_buffer = stencil.GetReuseBuffers()[src_name]
    forwardings = {}
    for dst_name, dst_point_dicts in dsts.items():
        for offset, points in dst_point_dicts.items():
            forwardings[offset] = []
            for c in range(stencil.buffers[src_name].chan):
                params = []
                for unroll_index, point_index in points.items():
                    params.append('from_%s_to_%s_param_%d_chan_%d_pe_%d' % (src_name, dst_name, point_index, c, unroll_index))
                if offset in next_fifo[src_name]:
                    params.append('%s_offset_%d_chan_%d' % (src_name, next_fifo[src_name][offset], c))
                params.append('%s_offset_%d_chan_%d' % (src_name, offset, c))
                params.append('epoch_num')
                forwardings[offset] += [['forward_%d<%s, %s>' % (len(params)-2, stencil.buffers[src_name].type, stencil.GetReuseBufferLength(src_name, offset)), params, ';']]
    for offset, arg_lists in sorted(forwardings.items()):
        for args in arg_lists:
            printer.PrintFunc(*args)

def PrintCode(stencil, output_file):
    logger.info('generate kernel code as %s' % output_file.name)
    printer = Printer(output_file)

    PrintHeader(printer)

    printer.PrintLine()

    PrintDefine(printer, 'BURST_WIDTH', stencil.burst_width)
    printer.PrintLine()

    PrintGuard(printer, 'UNROLL_FACTOR', stencil.unroll_factor)
    for i in range(len(stencil.tile_size)-1):
        PrintGuard(printer, 'TILE_SIZE_DIM_%d' % i, stencil.tile_size[i])
    PrintGuard(printer, 'BURST_WIDTH', stencil.burst_width)
    printer.PrintLine()

    PrintLoad(printer)
    printer.PrintLine()
    for data_type in {stencil.input.type}:
        PrintUnpack(printer, stencil.burst_width, data_type, stencil.unroll_factor)
        printer.PrintLine()
    for data_type in {stencil.output.type}:
        PrintPack(printer, stencil.burst_width, data_type, stencil.unroll_factor)
        printer.PrintLine()
    PrintStore(printer)
    printer.PrintLine()

    for forwarder in stencil.GetForwarders():
        PrintForwarder(printer, forwarder)
        printer.PrintLine()

    for stage in stencil.stages.values():
        PrintComputeStage(printer, stencil, stage)
        printer.PrintLine()

    PrintInterface(printer, stencil)

