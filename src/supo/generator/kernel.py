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
    stencil_window = GetOverallStencilWindow(stage.PreserveBorderFrom() if stage.PreserveBorderFrom() else stencil.input, stage.output)
    overall_idx = GetStencilWindowOffset(stencil_window)
    iteration = 1
    parent_buffer = stage.PreserveBorderFrom()
    while parent_buffer is not None and parent_buffer.parent is not None:
        parent_buffer = parent_buffer.parent.PreserveBorderFrom()
        iteration += 1
    delay = (GetStencilDistance(stencil_window, stencil.tile_size) - Serialize(overall_idx, stencil.tile_size))*iteration

    printer.PrintLine('template<uint32_t pe_id>')
    printer.PrintLine('void compute_%s(%s,' % (stage.name, ', '.join('hls::stream<%s>& %s_chan_%d' % (stage.output.type, stage.name, c) for c in range(stencil.buffers[stage.name].chan))))
    printer.DoIndent()
    for param in [(stencil.buffers[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, GetIndicesId(indices))) for input_name, input_window in stage.window.items() for indices in input_window for c in range(stencil.buffers[input_name].chan)]:
        printer.PrintLine('hls::stream<%s>& %s,' % param)
    if stage.output.PreserveBorderTo():
        for param in ['border_from_%s_dim_%d' % (stage.name, d) for d in range(stencil.dim-1)]:
            for c in range(stage.output.chan):
                printer.PrintLine('hls::stream<%s>& %s_left_chan_%d,' % (stage.output.type, param, c))
                printer.PrintLine('hls::stream<%s>& %s_right_chan_%d,' % (stage.output.type, param, c))
    if stage.PreserveBorderFrom():
        for d in range(stencil.dim-1):
            printer.PrintLine('uint32_t input_bound_dim_%d,' % d)
    for d in range(stencil.dim):
        printer.PrintLine('uint32_t input_size_dim_%d,' % d)
    printer.PrintLine('uint64_t epoch_num)')
    printer.UnIndent()
    printer.DoScope()

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
        printer.PrintLine('bool margin_conditions[%d];' % stencil.dim)
        #printer.PrintLine('#pragma HLS array_partition variable=margin_conditions complete', 0)
        printer.PrintLine('#pragma HLS resource variable=margin_conditions latency=1 core=RAM_2P_LUTRAM', 0)
        for d in range(stencil.dim):
            printer.DoScope()
            printer.PrintLine('#pragma HLS latency min=1', 0)
            printer.PrintLine('margin_conditions[%d] = %s;' % (d, MarginCondition(d)))
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
        printer.PrintLine('if(%s)' % (' || '.join('margin_conditions[%d]' % d for d in range(stencil.dim))))
        printer.DoScope()
        preserve_border_from = stage.PreserveBorderFrom()
        printer.PrintLine('%s_chan_%d<<load_%s_chan_%d_at_%s;' % (stage.name, c, preserve_border_from.name, c, GetIndicesId(stage.idx)))
        #printer.PrintLine('printf("bypass: epoch%%d pe%%d %s %s val=%%d\\n", epoch, pe_id, %s, %s load_%s_chan_%d_at_%s);' % (' '.join('%c=%%d' % coords_in_tile[d] for d in range(stencil.dim)), ' '.join('%c=%%d' % coords_in_orig[d] for d in range(stencil.dim)), ', '.join(coords_in_tile[:stencil.dim]), ', '.join(coords_in_orig[:stencil.dim]), preserve_border_from.name, c, GetIndicesId(stage.idx)))
        printer.UnScope()
        printer.PrintLine('else')
        printer.DoScope()

    LoadPrinter = lambda node: 'param_%s%s[unroll_index]%s' % (node.name, '' if stencil.extra_params[node.name].dup is None else '[%d]' % node.chan, ''.join(['[%d]'%x for x in node.idx])) if node.name in stencil.extra_params else 'load_%s_chan_%d_at_%s' % (node.name, node.chan, GetIndicesId(node.idx))
    StorePrinter = lambda node: '%s store_%s_chan_%d' % (stage.output.type, node.name, node.chan)

    for expr in stage.expr:
        expr.PrintCode(printer, stencil.buffers, LoadPrinter, StorePrinter, add_latency=True)

    for c in range(stage.output.chan):
        printer.PrintLine('%s_chan_%d<<store_%s_chan_%d;' % ((stage.name, c)*2))
        #printer.PrintLine('printf("calc: epoch%%d pe%%d %%d inputs=%s val=%%d\\n", epoch, pe_id, epoch*%d+pe_id-%d, %s, store_%s_chan_%d);' % (' '.join(['%d']*len(params)), stencil.unroll_factor, delay, ', '.join('load_%s'%p[1] for p in params), stage.name, c))

    if stage.output.PreserveBorderTo():
        printer.PrintLine()
        for d in range(stencil.dim-1):
            if stencil_dim[d] < 2:
                continue
            printer.PrintLine('if(%s >= %d-1 && %s < input_size_dim_0-%d+1)' % (IndexOrig(d), stencil_dim[d], IndexOrig(d), stencil_dim[d]))
            printer.DoScope()
            printer.PrintLine('switch(%s)' % IndexTile(d))
            printer.DoScope()

            for i in range(output_idx[d], stencil_dim[d]-1):
                printer.PrintLine('case %d:' % i)
            printer.DoScope()
            printer.PrintLine('// duplicate output to border buffer')
            for c in range(stage.output.chan):
                printer.PrintLine('border_from_%s_dim_%d_right_chan_%d<<store_%s_chan_%d;' % (stage.name, d, c, stage.name, c))
            printer.PrintLine('break;')
            printer.UnScope()

            for i in range(stencil.tile_size[d]-stencil_dim[d]+1, stencil.tile_size[d]-stencil_dim[d]+output_idx[d]+1):
                printer.PrintLine('case %d:' % i)
            printer.DoScope()
            printer.PrintLine('// duplicate output to border buffer')
            for c in range(stage.output.chan):
                printer.PrintLine('border_from_%s_dim_%d_left_chan_%d<<store_%s_chan_%d;' % (stage.name, d, c, stage.name, c))
            printer.PrintLine('break;')
            printer.UnScope()

            printer.UnScope()
            printer.UnScope()
    if stage.PreserveBorderFrom():
        printer.UnScope()
        printer.PrintLine()
        PrintIncrementCoordinates(printer, stencil, stage)

    printer.UnScope()
    printer.UnScope()
    printer.PrintLine()

def PrintIncrementCoordinates(printer, stencil, stage):
    overall_stencil_window = GetOverallStencilWindow(*([stage.PreserveBorderFrom(), stage.output] if stencil.preserve_border else [stencil.input, stencil.output]))
    overall_stencil_dim = GetStencilDim(overall_stencil_window)

    PrintIfTile = lambda d: printer.PrintLine('if(%c>=TILE_SIZE_DIM_%d)' % (coords_in_tile[d], d))
    PrintIfTileLastDim = lambda d: printer.PrintLine('if(%c >= input_size_dim_%d)' % (coords_in_tile[d], d))
    PrintIfTensor = lambda d: printer.PrintLine('if(%c >= input_size_dim_%d)' % (coords_in_orig[d], d))
    PrintIncrementTile = lambda d: printer.PrintLine('++%c;' % (coords_in_tile[d]))
    PrintDecrementTile = lambda d: printer.PrintLine('%c -= TILE_SIZE_DIM_%d;' % (coords_in_tile[d], d))
    PrintIncrementOrig = lambda d: printer.PrintLine('%c_base += TILE_SIZE_DIM_%d - %s + 1;' % (coords_in_orig[d], d, overall_stencil_dim[d]))
    PrintDecrementOrig = lambda d: printer.PrintLine('%c = 0;' % coords_in_orig[d])
    PrintDecrementTileLastDim = lambda d: printer.PrintLine('%c -= input_size_dim_%d;' % (coords_in_tile[d], d))

    printer.PrintLine('if(%s)' % ' && '.join('%c_base<input_bound_dim_%d' % (coords_in_orig[d], d) for d in range(stencil.dim-1)))
    printer.DoScope()
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
            p.PrintLine('ap_uint<BURST_WIDTH>* var_output_chan_%d_bank_%d,' % (c, i))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('ap_uint<BURST_WIDTH>* var_input_chan_%d_bank_%d,' % (c, i))
    if extra_params:
        for param in extra_params.values():
            p.PrintLine('%s* var_%s,' % (param.type, param.name))
    p.PrintLine('uint64_t coalesced_data_num,')
    p.PrintLine('uint64_t tile_data_num,')
    for i in range(stencil.dim-1):
        p.PrintLine('uint32_t input_bound_dim_%d,' % i)
    for d in range(stencil.dim-1):
        p.PrintLine('uint32_t input_size_dim_%d,' % d)
    p.PrintLine('uint32_t input_size_dim_%d)' % (stencil.dim-1))
    p.UnIndent()
    p.DoScope()

    bank = 0
    for i in range(dram_bank):
        for c in range(output_chan):
            p.PrintLine('#pragma HLS interface m_axi port=var_output_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%d latency=120' % (c, i, c, bank), 0)
        bank += 1
    if not dram_separate:
        bank = 0
    for i in range(dram_bank):
        for c in range(input_chan):
            p.PrintLine('#pragma HLS interface m_axi port=var_input_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%d latency=120' % (c, i, c, bank), 0)
        bank += 1
    if extra_params:
        for idx, param in enumerate(extra_params.values()):
            p.PrintLine('#pragma HLS interface m_axi port=var_%s offset=slave depth=%d bundle=gmem%d latency=120' % (param.name, reduce(operator.mul, param.size), idx), 0)
    p.PrintLine()
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS interface s_axilite port=var_output_chan_%d_bank_%d bundle=control' % (c, i), 0)
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS interface s_axilite port=var_input_chan_%d_bank_%d bundle=control' % (c, i), 0)
    if extra_params:
        for param in extra_params.values():
            p.PrintLine('#pragma HLS interface s_axilite port=var_%s bundle=control' % param.name, 0)
    p.PrintLine('#pragma HLS interface s_axilite port=coalesced_data_num bundle=control', 0)
    p.PrintLine('#pragma HLS interface s_axilite port=tile_data_num bundle=control', 0)
    for d in range(stencil.dim-1):
        p.PrintLine('#pragma HLS interface s_axilite port=input_bound_dim_%d bundle=control' % d, 0)
    for d in range(stencil.dim):
        p.PrintLine('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % d, 0)
    p.PrintLine('#pragma HLS interface s_axilite port=return bundle=control', 0)
    p.PrintLine()

    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> >  input_stream_chan_%d_bank_%d( "input_stream_chan_%d_bank_%d");' % ((c, i)*2))
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('hls::stream<ap_uint<BURST_WIDTH> > output_stream_chan_%d_bank_%d("output_stream_chan_%d_bank_%d");' % ((c, i)*2))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS stream variable=input_stream_chan_%d_bank_%d depth=32' % (c, i), 0)
    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('#pragma HLS stream variable=output_stream_chan_%d_bank_%d depth=32' % (c, i), 0)
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

    # border buffers
    msg = 'border buffers'
    logger.debug('generate %s' % msg)
    p.PrintLine('// %s' % msg)
    for stage in stencil.GetStagesChronologically():
        if stage.output.PreserveBorderTo():
            for unroll_index in range(unroll_factor):
                for d in range(stencil.dim-1):
                    for c in range(stage.output.chan):
                        param = (stage.output.type, 'border_from_%s_dim_%d_left_chan_%d_pe_%d' % (stage.name, d, c, unroll_index))
                        p.PrintLine('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
                        param = (stage.output.type, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % (stage.name, d, c, unroll_index))
                        p.PrintLine('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
    p.PrintLine()


    p.PrintLine('uint64_t epoch_num = coalesced_data_num*%d/%d;' % (stencil.burst_width*stencil.dram_bank/type_width[stencil.input.type], unroll_factor))
    p.PrintLine()

    p.PrintLine('#pragma HLS dataflow', 0)
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('load(input_stream_chan_%d_bank_%d, var_input_chan_%d_bank_%d, coalesced_data_num);' % ((c, i)*2))
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('unpack_%s(' % GetSupoType(stencil.input.type))
            p.DoIndent()
            for unroll_index in reversed(range(dram_bank-1-i, unroll_factor, dram_bank)):
                p.PrintLine('%s,' % GetTensorAt(stencil.input.name, stencil.input.offset+unroll_index, c))
            p.PrintLine('input_stream_chan_%d_bank_%d, coalesced_data_num);' % (c, i))
            p.UnIndent()
    p.PrintLine()

    output_stream = ', '.join(', '.join('output_stream_chan_%d_bank_%d' % (c, x) for x in range(dram_bank)) for c in range(output_chan))
    input_stream = ', '.join(', '.join('input_stream_chan_%d_bank_%d' % (c, x) for x in range(dram_bank)) for c in range(input_chan))
    tile_num_dim = ', '.join('tile_num_dim_%d' % d for d in range(stencil.dim-1))
    input_size_dim = ', '.join('input_size_dim_%d' % d for d in range(stencil.dim))
    PrintForwarding(p, stencil, stencil.input.name)
    p.PrintLine()
    for stage in stencil.GetStagesChronologically():
        inputs = tuple(reversed(range(unroll_factor))) if stage.IsOutput() else [start for start, end in stencil.GetReuseBuffers()[stage.name][1:] if start==end]
        for unroll_index in range(unroll_factor):
            params = ['%s_offset_%s_chan_%d' % (stage.name, inputs[unroll_index], c) for c in range(stage.output.chan)]
            params += ['from_%s_to_%s_param_%d_chan_%d_pe_%d' % (input_name, stage.name, i, c, unroll_index) for input_name, input_window in stage.window.items() for i in range(len(input_window)) for c in range(stencil.buffers[input_name].chan)]
            if stage.output.PreserveBorderTo():
                for d in range(stencil.dim-1):
                    for c in range(stage.output.chan):
                        param = (stage.name, d, c, unroll_index)
                        params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
            if stage.PreserveBorderFrom():
                params += ['input_bound_dim_%d' % d for d in range(stencil.dim-1)]
            params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
            params.append('epoch_num')
            p.PrintFunc('compute_%s<%d>' % (stage.name, unroll_index), params, ';')

        if not stage.IsOutput():
            p.PrintLine()
            PrintForwarding(p, stencil, stage.name)
        p.PrintLine()

    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('pack_%s(output_stream_chan_%d_bank_%d,' % (GetSupoType(stencil.output.type), c, i))
            p.DoIndent()
            for unroll_index in reversed(range(dram_bank-1-i, unroll_factor, dram_bank)):
                p.PrintLine('%s,' % GetTensorAt(stencil.output.name, unroll_index, c))
            p.PrintLine('coalesced_data_num);')
            p.UnIndent()
    for c in range(input_chan):
        for i in range(dram_bank):
            p.PrintLine('store(var_output_chan_%d_bank_%d, output_stream_chan_%d_bank_%d, coalesced_data_num);' % ((c, i)*2))

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
    GetDstName = lambda i: ('to_%0'+str(len(str(unroll_factor-1)))+'d') % i
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
    GetDstName = lambda i: ('from_%0'+str(len(str(unroll_factor-1)))+'d') % i
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

def PrintForwarderWithBorder(printer, stencil, forwarder_with_border):
    src_name = forwarder_with_border[0]
    forwarder = forwarder_with_border[1]
    stage = stencil.stages[src_name]
    stencil_window = GetOverallStencilWindow(stage.PreserveBorderFrom(), stage.output)

    params = ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]+['hls::stream<T>& src']
    for param in ['border_dim_%d' % d for d in range(stencil.dim-1)]:
        params += ['hls::stream<T>& %s_left' % param, 'hls::stream<T>& %s_right' % param]
    params += ['uint32_t input_bound_dim_%d' % d for d in range(stencil.dim-1)]
    params += ['uint32_t input_size_dim_%d' % d for d in range(stencil.dim)]
    params += ['uint32_t data_num']
    printer.PrintFunc('template<typename T, int32_t i_init> void forward_%s_%d' % forwarder_with_border, params)
    printer.DoScope()
    printer.PrintLine(' int32_t i = i_init;')
    for i in range(1, len(stencil.tile_size)):
        printer.PrintLine('uint16_t %c = 0;' % coords_in_tile[i])
    for i in range(len(stencil.tile_size)-1):
        printer.PrintLine('uint16_t %c_base = 0;' % coords_in_orig[i])
    printer.PrintLine()
    printer.PrintLine('forward_%s_%d_epoch:' % forwarder_with_border, 0)
    printer.PrintLine('for(uint32_t epoch = 0; epoch < data_num; ++epoch)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)

    for i in range(len(stencil.tile_size)-1):
        printer.PrintLine('uint16_t  %c = %c_base+%c;' % (coords_in_orig[i], coords_in_orig[i], coords_in_tile[i]))
    printer.PrintLine('uint16_t& %c = %c;' % (coords_in_orig[len(stencil.tile_size)-1], coords_in_tile[len(stencil.tile_size)-1]))

    IndexTile = lambda d: '%c' % (coords_in_tile[d])
    IndexOrig = lambda d: '%c' % (coords_in_orig[d])
    output_idx = GetStencilWindowOffset(stencil_window)
    stencil_dim = GetStencilDim(stencil_window)
    MarginCondition = lambda d: ('%s<%d || ' % (IndexOrig(d), output_idx[d]) if output_idx[d]>0 else '') + '%s>input_size_dim_%d-%d+%d' % (IndexOrig(d), d, stencil_dim[d], output_idx[d])
    printer.PrintLine('bool margin_conditions[%d];' % stencil.dim)
    #printer.PrintLine('#pragma HLS array_partition variable=margin_conditions complete', 0)
    printer.PrintLine('#pragma HLS resource variable=margin_conditions latency=1 core=RAM_2P_LUTRAM', 0)
    for d in range(stencil.dim):
        printer.DoScope()
        printer.PrintLine('#pragma HLS latency min=1', 0)
        printer.PrintLine('margin_conditions[%d] = %s;' % (d, MarginCondition(d)))
        printer.UnScope()
    printer.PrintLine()

    printer.PrintLine('T tmp(src.read());')

    printer.PrintLine('if(!(%s))' % ' || '.join('margin_conditions[%d]' % d for d in range(stencil.dim)))
    printer.DoScope()
    for d in range(stencil.dim-1):
        printer.PrintLine('switch(%s)' % IndexTile(d))
        printer.DoScope()

        for i in range(output_idx[d]):
            printer.PrintLine('case %d:' % i)
        printer.DoScope()
        for c in range(stage.output.chan):
            printer.PrintLine('tmp = border_dim_%d_left.read();' % d)
        printer.PrintLine('break;')
        printer.UnScope()

        for i in range(stencil.tile_size[d]-stencil_dim[d]+output_idx[d]+1, stencil.tile_size[d]):
            printer.PrintLine('case %d:' % i)
        printer.DoScope()
        for c in range(stage.output.chan):
            printer.PrintLine('tmp = border_dim_%d_right.read();' % d)
        printer.PrintLine('break;')
        printer.UnScope()

        printer.UnScope()
    printer.UnScope()

    for dst in range(forwarder):
        printer.PrintLine('dst_%d<<tmp;' % dst)
    printer.PrintLine()

    PrintIncrementCoordinates(printer, stencil, stage)
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
                func_name = 'forward'
                temp_param = stencil.GetReuseBufferLength(src_name, offset)
                forward_num = len(params)-1
                if offset<stencil.unroll_factor and stencil.buffers[src_name].PreserveBorderTo() and not stencil.buffers[src_name].IsInput():
                    stage = stencil.stages[src_name]
                    stencil_window = GetOverallStencilWindow(stage.PreserveBorderFrom() if stage.PreserveBorderFrom() else stencil.input, stage.output)
                    overall_idx = GetStencilWindowOffset(stencil_window)
                    stencil_dim = GetStencilDim(stencil_window)
                    iteration = 1
                    parent_buffer = stage.PreserveBorderFrom()
                    while parent_buffer is not None and parent_buffer.parent is not None:
                        parent_buffer = parent_buffer.parent.PreserveBorderFrom()
                        iteration += 1
                    delay = (GetStencilDistance(stencil_window, stencil.tile_size) - Serialize(overall_idx, stencil.tile_size))*iteration

                    func_name += '_'+src_name
                    temp_param = '%d-%d' % (unroll_index, delay)
                    for d in range(stencil.dim-1):
                        param = (src_name, d, c, (unroll_index+(stencil.tile_size[d]-stencil_dim[d]+1))%stencil.unroll_factor)
                        params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param]
                        param = (src_name, d, c, (unroll_index-(stencil.tile_size[d]-stencil_dim[d]+1))%stencil.unroll_factor)
                        params += ['border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
                    for d in range(stencil.dim-1):
                        params.append('input_bound_dim_%d' % d)
                    for d in range(stencil.dim):
                        params.append('input_size_dim_%d' % d)

                params.append('epoch_num')
                func_name += '_%d<%s, %s>' % (forward_num, stencil.buffers[src_name].type, temp_param)
                forwardings[offset] += [[func_name, params, ';']]
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
        PrintUnpack(printer, stencil.burst_width, data_type, stencil.unroll_factor//stencil.dram_bank)
        printer.PrintLine()
    for data_type in {stencil.output.type}:
        PrintPack(printer, stencil.burst_width, data_type, stencil.unroll_factor//stencil.dram_bank)
        printer.PrintLine()
    PrintStore(printer)
    printer.PrintLine()

    for forwarder in stencil.GetForwarders():
        PrintForwarder(printer, forwarder)
        printer.PrintLine()

    for forwarder in stencil.GetForwardersWithBorder():
        PrintForwarderWithBorder(printer, stencil, forwarder)
        printer.PrintLine()

    for stage in stencil.stages.values():
        PrintComputeStage(printer, stencil, stage)
        printer.PrintLine()

    PrintInterface(printer, stencil)

