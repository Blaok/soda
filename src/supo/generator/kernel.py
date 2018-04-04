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

from supo.generator.dataflow import *
from supo.generator.utils import *
from supo.grammar import ExtraParam

logger = logging.getLogger('__main__').getChild(__name__)

def PrintComputeInput(printer, stencil):
    unroll_factor = stencil.unroll_factor
    all_points = stencil.GetAllPoints()
    next_fifo = stencil.GetNextFIFO()

    for tensor in [stencil.input]:
        for pe_id in range(unroll_factor):
            printer.PrintLine('void compute_%s_pe_%d(' % (tensor.name, pe_id))
            printer.DoIndent()

            # outputs
            intermediate_offsets = []
            offset = unroll_factor-1-pe_id
            while offset is not None:
                intermediate_offsets.append(offset)
                for output_stage in tensor.children:
                    points = all_points[tensor.name][output_stage.name][offset]
                    for unroll_index, point in points.items():
                        for c in range(stencil.buffers[tensor.name].chan):
                            printer.PrintLine('/* output */ hls::stream<%s>& from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (tensor.type, tensor.name, output_stage.name, point, c, unroll_index))
                offset = next_fifo[tensor.name].get(offset, None)

            # inputs
            for c in range(stencil.buffers[tensor.name].chan):
                printer.PrintLine('/*  input */ hls::stream<%s>& %s_offset_%s_chan_%d,' % (tensor.type, tensor.name, unroll_factor-1-pe_id, c))

            params = []
            if tensor.PreserveBorderTo():
                for d in range(stencil.dim-1):
                    for c in range(tensor.chan):
                        param = (tensor.name, d, c, pe_id)
                        params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
                params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
            for param in params:
                printer.PrintLine('/*  param */ uint32_t %s,' % param)
            printer.PrintLine('/*  param */ uint32_t epoch_num)')
            printer.UnIndent()
            printer.DoScope()

            PrintBufferStatements(printer, intermediate_offsets, unroll_factor, tensor, pe_id)

            printer.PrintLine('compute_%s_pe_%d_epoch:' % (tensor.name, pe_id), 0)
            printer.PrintLine('for(uint32_t epoch = 0; epoch < epoch_num+%d; ++epoch)' %
                ((intermediate_offsets[-1]-intermediate_offsets[0])//unroll_factor))
            printer.DoScope()
            printer.PrintLine('#pragma HLS pipeline II=1', 0)
            PrintBuffers(printer, stencil, tensor, pe_id)
            printer.UnScope()

            printer.UnScope()
            printer.PrintLine()

def PrintBufferStatements(printer, intermediate_offsets, unroll_factor, tensor, pe_id):
            pragmas = []
            for begin, end in zip(intermediate_offsets[:-1], intermediate_offsets[1:]):
                for c in range(tensor.chan):
                    param = '%s_offset_%s_chan_%d' % (tensor.name, end, c)
                    printer.PrintLine('hls::stream<%s> %s("%s");' % (tensor.type, param, param))
                    pragmas.append('#pragma HLS stream variable=%s depth=%d' % (param, (end-begin)/unroll_factor))
            for pragma in pragmas:
                printer.PrintLine(pragma, 0)

def PrintBuffers(printer, stencil, tensor, pe_id):
    unroll_factor = stencil.unroll_factor
    all_points = stencil.GetAllPoints()
    next_fifo = stencil.GetNextFIFO()

    offset = unroll_factor-1-pe_id
    depth = 0
    last_depth = depth
    reads = {}
    writes = {}
    bounds = set()
    while offset is not None:
        if depth != 0:
            read_lower = 'epoch < %d' % depth
            read_upper = 'epoch < epoch_num+%d' % depth
        else:
            read_lower = 'false'
            read_upper = 'epoch < epoch_num'
        if last_depth != 0:
            write_lower = 'epoch < %d' % last_depth
            write_upper = 'epoch < epoch_num+%d' % last_depth
        else:
            write_lower = 'false'
            write_upper = 'epoch < epoch_num'

        if depth not in bounds:
            printer.PrintLine('bool lower_bound_%d = %s;' % (depth, read_lower))
            printer.PrintLine('bool upper_bound_%d = %s;' % (depth, read_upper))
            bounds.add(depth)
        if last_depth not in bounds:
            printer.PrintLine('bool lower_bound_%d = %s;' % (last_depth, read_lower))
            printer.PrintLine('bool upper_bound_%d = %s;' % (last_depth, read_upper))
            bounds.add(last_depth)

        read_depth = depth
        write_depth = last_depth
        reads[read_depth] = '%s_offset_%d_chan_%%d' % (tensor.name, offset)
        for output_stage in tensor.children:
            points = all_points[tensor.name][output_stage.name][offset]
            for unroll_index, point in points.items():
                writes.setdefault(write_depth, []).append((
                    'from_%s_to_%s_param_%d_chan_%%d_pe_%d' %
                    (tensor.name, output_stage.name, point, unroll_index),
                    reads[read_depth]))
        next_offset = next_fifo[tensor.name].get(offset, None)
        if next_offset is not None:
            writes.setdefault(write_depth, []).append((
                '%s_offset_%d_chan_%%d' % (tensor.name, next_offset),
                reads[read_depth]))
            last_depth = depth
            depth += (next_offset-offset)//unroll_factor
        offset = next_offset

    depths = sorted(set(reads.keys())|set(writes.keys()))
    first = 'else '
    printer.PrintLine('if(lower_bound_%d) {}' % depths[0])
    for lower_depth, upper_depth, lower_bound, upper_bound in zip(
            depths+depths[:-1], depths[1:]+depths,
            ['lower']*len(depths)+['upper']*(len(depths)-1),
            ['lower']*(len(depths)-1)+['upper']*len(depths)):
        printer.PrintLine(
            '%sif(%s_bound_%d)' % (first,
                upper_bound, upper_depth))
        printer.DoScope()
        #printer.PrintLine('#pragma HLS latency min=1 max=1', 0)
        read_set = set()
        def PrintIntervalReads():
            read = reads[depth]
            read_set.add(read)
            for c in range(stencil.buffers[tensor.name].chan):
                printer.PrintLine('%s tmp_%s = %s.read();' % (tensor.type, read%c, read%c))
        def PrintIntervalWrites():
            for c in range(stencil.buffers[tensor.name].chan):
                for write, read in writes.get(depth, []):
                    if read in read_set:
                        printer.PrintLine('%s << tmp_%s;' % (write%c, read%c))
                    else:
                        printer.PrintLine('%s << 0;' % write%c)
        for depth in depths:
            if lower_bound == 'lower' and upper_bound == 'lower':
                if lower_depth >= depth:
                    PrintIntervalReads()
            elif lower_bound == 'lower' and upper_bound == 'upper':
                if lower_depth >= depth and upper_depth <= depth:
                    PrintIntervalReads()
            elif lower_bound == 'upper' and upper_bound == 'upper':
                if upper_depth <= depth:
                    PrintIntervalReads()
        for depth in depths:
            if lower_bound == 'lower' and upper_bound == 'lower':
                if lower_depth >= depth:
                    PrintIntervalWrites()
            elif lower_bound == 'lower' and upper_bound == 'upper':
                if lower_depth >= depth and upper_depth <= depth:
                    PrintIntervalWrites()
            elif lower_bound == 'upper' and upper_bound == 'upper':
                if upper_depth <= depth:
                    PrintIntervalWrites()
        if not first:
            first = 'else '
        printer.UnScope()

def PrintComputeStage(printer, stencil, stage):
    unroll_factor = stencil.unroll_factor
    all_points = stencil.GetAllPoints()
    next_fifo = stencil.GetNextFIFO()
    reuse_buffers = stencil.GetReuseBuffers()
    stencil_window = GetOverallStencilWindow(stage.PreserveBorderFrom() if stage.PreserveBorderFrom() else stencil.input, stage.output)
    overall_idx = GetStencilWindowOffset(stencil_window)
    iteration = 1
    parent_tensor = stage.PreserveBorderFrom()
    tensor = stage.output
    while parent_tensor is not None and parent_tensor.parent is not None:
        parent_tensor = parent_tensor.parent.PreserveBorderFrom()
        iteration += 1
    delay = (GetStencilDistance(stencil_window, stencil.tile_size) - Serialize(overall_idx, stencil.tile_size))*iteration

    for pe_id in range(1 if stencil.cluster == 'none' else unroll_factor):
        if stencil.cluster == 'none':
            printer.PrintLine('template<uint32_t pe_id>')
            func_name = stage.name
        else:
            func_name = stage.name + '_pe_%d' % pe_id
        printer.PrintLine('void compute_%s(' % func_name)
        printer.DoIndent()

        # outputs
        intermediate_offsets = []
        if stencil.cluster == 'none':
            for c in range(stencil.buffers[stage.name].chan):
                printer.PrintLine('/* output */ hls::stream<%s>& %s_chan_%d,' % (stage.output.type, stage.name, c))
        elif stage.IsOutput():
            for c in range(stencil.buffers[stage.name].chan):
                printer.PrintLine('/* output */ hls::stream<%s>& %s_offset_%d_chan_%d,' % (stage.output.type, stage.name, unroll_factor-1-pe_id, c))
        else:
            offset = unroll_factor-1-pe_id
            while offset is not None:
                intermediate_offsets.append(offset)
                for output_stage in tensor.children:
                    points = all_points[tensor.name][output_stage.name][offset]
                    for unroll_index, point in points.items():
                        for c in range(stencil.buffers[tensor.name].chan):
                            printer.PrintLine('/* output */ hls::stream<%s>& from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (tensor.type, tensor.name, output_stage.name, point, c, unroll_index))
                offset = next_fifo[tensor.name].get(offset, None)

        # forwarded
        if stage.output.PreserveBorderTo():
            for d in range(stencil.dim-1):
                param = 'border_from_%s_dim_%d' % (stage.name, d)
                for c in range(stage.output.chan):
                    printer.PrintLine(
                        '/* output */ hls::stream<%s>& %s_left_chan_%d,' %
                            (stage.output.type, param, c))
                    printer.PrintLine(
                        '/* output */ hls::stream<%s>& %s_right_chan_%d,' %
                            (stage.output.type, param, c))

        # inputs
        for param in [(stencil.buffers[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, GetIndicesId(indices))) for input_name, input_window in stage.window.items() for indices in input_window for c in range(stencil.buffers[input_name].chan)]:
            printer.PrintLine('/*  input */ hls::stream<%s>& %s,' % param)

        # params
        if stage.PreserveBorderFrom():
            for d in range(stencil.dim-1):
                printer.PrintLine('/*  param */ uint32_t input_bound_dim_%d,' % d)
            for d in range(stencil.dim):
                printer.PrintLine('/*  param */ uint32_t input_size_dim_%d,' % d)
        printer.PrintLine('/*  param */ uint32_t epoch_num)')
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

        bound = ''
        if stencil.cluster != 'none' and not stage.IsOutput():
            bound = '+%d' % ((intermediate_offsets[-1]-intermediate_offsets[0])//unroll_factor)
            PrintBufferStatements(printer, intermediate_offsets, unroll_factor, tensor, pe_id)
            printer.PrintLine()
            for c in range(tensor.chan):
                param = '%s_offset_%d_chan_%d' % (tensor.name, unroll_factor-1-pe_id, c)
                printer.PrintLine('hls::stream<%s> %s("%s");' % (tensor.type, param, param))
                printer.PrintLine('#pragma HLS stream variable=%s depth=1' % param, 0)
            printer.PrintLine()

        printer.PrintLine('uint32_t epoch = 0;')
        printer.PrintLine('compute_%s_epoch:' % func_name, 0)
        printer.PrintLine('while(epoch < epoch_num)')
        printer.DoScope()
        printer.PrintLine('#pragma HLS pipeline II=1', 0)

        # empty test
        params = []
        for input_name, input_window in stage.window.items():
            for indices in input_window:
                for c in range(stencil.buffers[input_name].chan):
                    params.append('%s_chan_%d_at_%s' %
                        (input_name, c, GetIndicesId(indices)))
        printer.PrintLine('if(not (%s))' % ' or '.join(
            '%s.empty()' % param for param in params))
        printer.DoScope()

        if stencil.cluster != 'none' and not stage.IsOutput():
            printer.PrintLine('if(epoch < epoch_num)')
            printer.DoScope()

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
            if stencil.cluster == 'none':
                printer.PrintLine('%s_chan_%d<<store_%s_chan_%d;' % ((stage.name, c)*2))
            else:
                printer.PrintLine('%s_offset_%d_chan_%d<<store_%s_chan_%d;' % ((stage.name, unroll_factor-1-pe_id, c, stage.name, c)))
            #printer.PrintLine('printf("calc: epoch%%d pe%%d %%d inputs=%s val=%%d\\n", epoch, pe_id, epoch*%d+pe_id-%d, %s, store_%s_chan_%d);' % (' '.join(['%d']*len(params)), stencil.unroll_factor, delay, ', '.join('load_%s'%p[1] for p in params), stage.name, c))

        if stage.output.PreserveBorderTo():
            printer.PrintLine()
            for d in range(stencil.dim-1):
                if stencil_dim[d] < 2:
                    continue
                printer.PrintLine('if(%s >= %d-1 && %s < input_size_dim_%d-%d+1)' % (IndexOrig(d), stencil_dim[d], IndexOrig(d), d, stencil_dim[d]))
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

        if stencil.cluster == 'fine' and not stage.IsOutput():
            printer.UnScope()
            printer.PrintLine()
            PrintBuffers(printer, stencil, stage.output, pe_id)
        printer.PrintLine('++epoch;')
        printer.UnScope()
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
    PrintDecrementOrig = lambda d: printer.PrintLine('%c_base = 0;' % coords_in_orig[d])
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

    super_source = create_dataflow_graph(stencil)

    logger.info('generate reuse buffers')
    reuse_buffers = stencil.GetReuseBuffers()
    all_points = stencil.GetAllPoints()
    next_fifo = stencil.GetNextFIFO()
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
            p.PrintLine('#pragma HLS interface m_axi port=var_output_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%do latency=120' % (c, i, c, bank), 0)
        bank += 1
    if not dram_separate:
        bank = 0
    for i in range(dram_bank):
        for c in range(input_chan):
            p.PrintLine('#pragma HLS interface m_axi port=var_input_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%di latency=120' % (c, i, c, bank), 0)
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

    p.PrintLine('uint64_t epoch_num = coalesced_data_num*%d/%d;' % (
        stencil.burst_width*stencil.dram_bank/type_width[stencil.input.type],
        unroll_factor))
    p.PrintLine()

    if stencil.replication_factor > 1:
        _generate_code(p, stencil)

        p.UnScope()
        p.PrintLine()
        p.PrintLine('}//extern "C"')

        return

    GetTensorAt = lambda n, o, c: ('%s_offset_%d_chan_%d') % (n, o, c)

    # reuse buffers
    if stencil.cluster == 'none':
        for name, reuse_buffer in reuse_buffers.items():
            pragmas = []
            msg = 'reuse buffers for %s' % name
            logger.debug('generate %s' % msg)
            p.PrintLine('// %s' % msg)
            for start, end in reuse_buffer[1:]:
                for c in range(stencil.buffers[name].chan):
                    p.PrintLine('hls::stream<%s> %s("%s");' % ((stencil.buffers[name].type,)+(GetTensorAt(name, end, c),)*2))
                    buffer_length = stencil.GetReuseBufferLength(name, end)
                    tensor_name = GetTensorAt(name, end, c)
                    if buffer_length > 1:
                        pragmas.append((tensor_name, buffer_length))
            for pragma in pragmas:
                p.PrintLine('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
            p.PrintLine()
    else:
        p.PrintLine('// %s' % stencil.input.name)
        for unroll_index in range(unroll_factor):
            for c in range(stencil.input.chan):
                p.PrintLine('hls::stream<%s> %s("%s");' % ((stencil.buffers[stencil.input.name].type,)+(GetTensorAt(stencil.input.name, unroll_index, c),)*2))
        p.PrintLine()

    p.PrintLine('// %s' % stencil.output.name)
    for unroll_index in range(unroll_factor):
        for c in range(stencil.output.chan):
            p.PrintLine('hls::stream<%s> %s("%s");' % ((stencil.buffers[stencil.output.name].type,)+(GetTensorAt(stencil.output.name, unroll_index, c),)*2))
    p.PrintLine()

    # params
    msg = 'params'
    logger.debug('generate %s' % msg)
    p.PrintLine('// %s' % msg)
    pragmas = []
    for stage in stencil.GetStagesChronologically():
        for pe_id in range(unroll_factor):
            for input_name, input_window in stage.window.items():
                for i in range(len(input_window)):
                    offset = next(offset for offset, points in
                        all_points[input_name][stage.name].items()
                        if pe_id in points and points[pe_id] == i)
                    fwd_node = super_source.fwd_nodes[(input_name, offset)]
                    cpt_node = super_source.cpt_nodes[(stage.name, pe_id)]
                    extra_depth = super_source.get_extra_depth(
                        (fwd_node, cpt_node))
                    for c in range(stencil.buffers[input_name].chan):
                        var_type = stencil.buffers[input_name].type
                        var_name = 'from_%s_to_%s_param_%d_chan_%d_pe_%d' % (
                            input_name, stage.name, i, c, pe_id)
                        p.PrintLine('hls::stream<%s> %s("%s");' % (
                            var_type, var_name, var_name))
                        if extra_depth > 0:
                            pragmas.append((var_name, extra_depth+1))
                        else:
                            pragmas.append((var_name, 2))
    for pragma in pragmas:
        p.PrintLine('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
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
    next_fifo = stencil.GetNextFIFO()
    if stencil.cluster == 'none':
        PrintForwardCall(p, stencil, stencil.input.name)
        p.PrintLine()

        for stage in stencil.GetStagesChronologically():
            inputs = tuple(reversed(range(unroll_factor))) if stage.IsOutput() else [start for start, end in stencil.GetReuseBuffers()[stage.name][1:] if start==end]
            for unroll_index in range(unroll_factor):
                params = []
                for c in range(stage.output.chan):
                    params.append('/* output */ %s_offset_%s_chan_%d' %
                        (stage.name, inputs[unroll_index], c))
                if stage.output.PreserveBorderTo():
                    for d in range(stencil.dim-1):
                        for c in range(stage.output.chan):
                            param = (stage.name, d, c, unroll_index)
                            params += [
            '/* output */ border_from_%s_dim_%d_left_chan_%d_pe_%d'  % param,
            '/* output */ border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
                for input_name, input_window in stage.window.items():
                    for i in range(len(input_window)):
                        for c in range(stencil.buffers[input_name].chan):
                            params += [
                        '/*  input */ from_%s_to_%s_param_%d_chan_%d_pe_%d'
                        % (input_name, stage.name, i, c, unroll_index)]
                if stage.PreserveBorderFrom():
                    for d in range(stencil.dim-1):
                        params.append('/*  param */ input_bound_dim_%d' % d)
                    for d in range(stencil.dim):
                        params.append('/*  param */ input_size_dim_%d' % d)
                params.append('/*  param */ epoch_num')
                p.PrintFunc('compute_%s<%d>' % (stage.name, unroll_index),
                    params, ';', 0)

            if not stage.IsOutput():
                p.PrintLine()
                PrintForwardCall(p, stencil, stage.name)
            p.PrintLine()
    elif stencil.cluster == 'fine':
        for buffer in [stencil.input]+[stage.output for stage in stencil.GetStagesChronologically()]:
            inputs = tuple(reversed(range(unroll_factor))) if buffer.IsOutput() else [start for start, end in stencil.GetReuseBuffers()[buffer.name][1:] if start==end]
            for pe_id in range(unroll_factor):
                p.PrintLine('compute_%s_pe_%d(' % (buffer.name, pe_id))
                p.DoIndent()

                # outputs
                offset = unroll_factor-1-pe_id
                if buffer.IsOutput():
                    for c in range(stencil.buffers[buffer.name].chan):
                        p.PrintLine('/* output */ %s_offset_%s_chan_%d,' % (buffer.name, offset, c))
                else:
                    while offset is not None:
                        for output_stage in buffer.children:
                            points = all_points[buffer.name][output_stage.name][offset]
                            for unroll_index, point in points.items():
                                for c in range(stencil.buffers[buffer.name].chan):
                                    p.PrintLine('/* output */ from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (buffer.name, output_stage.name, point, c, unroll_index))
                        offset = next_fifo[buffer.name].get(offset, None)

                # inputs
                if buffer.IsInput():
                    for c in range(stencil.buffers[stencil.input.name].chan):
                        p.PrintLine('/*  input */ %s_offset_%s_chan_%d,' % (stencil.input.name, unroll_factor-1-pe_id, c))
                else:
                    for param in ['from_%s_to_%s_param_%d_chan_%d_pe_%d' % (input_name, buffer.name, i, c, pe_id) for input_name, input_window in buffer.parent.window.items() for i in range(len(input_window)) for c in range(stencil.buffers[input_name].chan)]:
                        p.PrintLine('/*  input */ %s,' % param)

                params = []
                if buffer.PreserveBorderTo():
                    for d in range(stencil.dim-1):
                        for c in range(buffer.chan):
                            param = (buffer.name, d, c, pe_id)
                            params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
                if buffer.PreserveBorderFrom():
                    params += ['input_bound_dim_%d' % d for d in range(stencil.dim-1)]
                    params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
                for param in params:
                    p.PrintLine('/*  param */ %s,' % param)
                p.PrintLine('/*  param */ epoch_num);')
                p.UnIndent()
            p.PrintLine()

    for c in range(output_chan):
        for i in range(dram_bank):
            p.PrintLine('pack_%s(output_stream_chan_%d_bank_%d,' % (GetSupoType(stencil.output.type), c, i))
            p.DoIndent()
            for unroll_index in reversed(range(dram_bank-1-i, unroll_factor, dram_bank)):
                p.PrintLine('%s,' % GetTensorAt(stencil.output.name, unroll_index, c))
            p.PrintLine('coalesced_data_num);')
            p.UnIndent()
    for c in range(output_chan):
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

def PrintForwardFunc(printer, forwarder):
    printer.PrintFunc('template<typename T, uint32_t fifo_depth> void forward_%d' % forwarder, ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]+['hls::stream<T>& src']+['uint32_t data_num'])
    printer.DoScope()
    printer.PrintLine('forward_%d_epoch:' % forwarder, 0)
    printer.PrintLine('for(uint32_t i = 0; i < data_num+fifo_depth; ++i)')
    printer.DoScope()
    printer.PrintLine('#pragma HLS pipeline II=1', 0)
    printer.PrintLine('T tmp;')
    printer.PrintLine('if(i<fifo_depth)')
    printer.DoScope()
    printer.PrintLine('tmp = 0;')
    printer.UnScope()
    printer.PrintLine('else')
    printer.DoScope()
    printer.PrintLine('tmp = src.read();')
    printer.UnScope()
    printer.PrintLine('if(i<data_num)')
    printer.DoScope()
    for dst in range(forwarder):
        printer.PrintLine('dst_%d<<tmp;' % dst)
    printer.UnScope()
    printer.UnScope()
    printer.UnScope()

def PrintForwardFuncWithBorder(printer, stencil, forwarder_with_border):
    src_name = forwarder_with_border[0]
    forwarder = forwarder_with_border[1]
    stage = stencil.stages[src_name]
    stencil_window = GetOverallStencilWindow(stage.PreserveBorderFrom(), stage.output)

    params = ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]
    params += ['hls::stream<T>& src']
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

def PrintForwardCall(printer, stencil, src_name):
    forwardings = stencil.GetForwardings(src_name)
    for offset, args in sorted(forwardings.items()):
        forward_num = len(args[1])
        temp_param = forwardings[offset][4]
        func_name = '%s_%d<%s, %s>' % (args[0], forward_num,
            stencil.buffers[src_name].type, temp_param)
        for c in range(stencil.buffers[src_name].chan):
            printer.PrintFunc(
                func_name,
                [s%c for s in args[1]]+
                [s%c for s in args[2]]+
                args[3], ';', 0)

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

    if stencil.cluster == 'none':
        for forwarder in stencil.GetForwarders():
            PrintForwardFunc(printer, forwarder)
            printer.PrintLine()

        for forwarder in stencil.GetForwardersWithBorder():
            PrintForwardFuncWithBorder(printer, stencil, forwarder)
            printer.PrintLine()
        for stage in stencil.stages.values():
            PrintComputeStage(printer, stencil, stage)
            printer.PrintLine()
    elif stencil.cluster == 'fine':
        PrintComputeInput(printer, stencil)
        for stage in stencil.GetStagesChronologically():
            PrintComputeStage(printer, stencil, stage)

    if stencil.replication_factor > 1:
        dst_lists = set()
        super_source = create_dataflow_graph(stencil)
        def add_dst_lists(tensor):
            rf = stencil.replication_factor
            dst_ids = [start%rf for start, end
                in stencil.get_replicated_reuse_buffers()[tensor.name][1:]
                if start == end]
            dst_ids = tuple(dst_id if dst_id in dst_ids else None
                for dst_id in range(stencil.replication_factor))
            dst_lists.add(dst_ids)
        for node in super_source.tpo_node_generator():
            if isinstance(node, SuperSourceNode):
                add_dst_lists(stencil.input)
            elif isinstance(node, ComputeNode):
                if not node.stage.IsOutput():
                    add_dst_lists(node.stage.output)
        for dst_list in dst_lists:
            _print_reconnect_func(printer, dst_list)
    printer.PrintLine()

    PrintInterface(printer, stencil)

def _generate_code(printer, stencil):
    super_source = create_dataflow_graph(stencil)

    if stencil.replication_factor > 1:
        pragmas = []
        hls_streams = set()
        for chan in range(stencil.input.chan):
            for replica_id in range(stencil.replication_factor):
                var_type = stencil.input.type
                var_name = ('compute_%s_chan_%d_replica_%d' %
                    (stencil.input.name, chan, replica_id))
                printer.PrintLine('hls::stream<%s> %s("%s");' %
                    (var_type, var_name, var_name))
                pragmas.append((var_name, 2))
                hls_streams.add(var_name)
        for src_node, dst_node in super_source.bfs_edge_generator():
            logger.debug('%s -> %s' % (repr(src_node), repr(dst_node)))
            if isinstance(dst_node, ForwardNode):
                for chan in range(dst_node.tensor.chan):
                    if isinstance(src_node, ComputeNode):
                        for replica_id in range(stencil.replication_factor):
                            var_type = dst_node.tensor.type
                            var_name = ('compute_%s_chan_%d_replica_%d' %
                                (dst_node.tensor.name, chan, replica_id))
                            if var_name in hls_streams:
                                continue
                            printer.PrintLine('hls::stream<%s> %s("%s");' %
                                    (var_type, var_name, var_name))
                            pragmas.append((var_name, 2))
                            hls_streams.add(var_name)
                    for replica_id in range(stencil.replication_factor):
                        var_type = dst_node.tensor.type
                        var_name = 'forward_%s_offset_%d_chan_%d_replica_%d' % (
                            dst_node.tensor.name, dst_node.offset,
                            chan, replica_id)
                        if var_name in hls_streams:
                            continue
                        printer.PrintLine('hls::stream<%s> %s("%s");' %
                            (var_type, var_name, var_name))
                        pragmas.append((var_name, max(1, dst_node.depth)))
                        hls_streams.add(var_name)
            elif (isinstance(src_node, ForwardNode) and
                  isinstance(dst_node, ComputeNode)):
                for replica_id in range(stencil.replication_factor):
                    for chan in range(src_node.tensor.chan):
                        var_type = src_node.tensor.type
                        var_name = ('from_%s_offset_%d_to_compute_%s_chan_%d_'
                            'replica_%d' % (src_node.tensor.name,
                                src_node.offset, dst_node.stage.name,
                                chan, replica_id))
                        if var_name in hls_streams:
                            continue
                        printer.PrintLine('hls::stream<%s> %s("%s");' %
                            (var_type, var_name, var_name))
                        pragmas.append((var_name, max(2,
                            super_source.get_extra_depth((src_node, dst_node))
                            +1)))
                        hls_streams.add(var_name)
            elif (isinstance(src_node, ComputeNode) and
                  isinstance(dst_node, SuperSinkNode)):
                for replica_id in range(stencil.replication_factor):
                    for chan in range(src_node.stage.output.chan):
                        var_type = src_node.stage.output.type
                        var_name = ('compute_%s_chan_%d_replica_%d' %
                            (src_node.stage.name, chan, replica_id))
                        if var_name in hls_streams:
                            continue
                        printer.PrintLine('hls::stream<%s> %s("%s");' %
                            (var_type, var_name, var_name))
                        pragmas.append((var_name, 2))
                        hls_streams.add(var_name)
            else:
                raise SemanticError('unknown dataflow edge: %s -> %s' %
                        (repr(src_node), repr(dst_node)))

        for pragma in pragmas:
            printer.PrintLine(
                '#pragma HLS stream variable=%s depth=%d' % pragma, 0)

        printer.PrintLine()
        printer.PrintLine('#pragma HLS dataflow', 0)

        replicated_all_points = stencil.get_replicated_all_points()
        replicated_reuse_buffers = stencil.get_replicated_reuse_buffers()
        replicated_next_fifo = stencil.get_replicated_next_fifo()

        def print_reconnect(tensor):
            rf = stencil.replication_factor
            offsets = [start for start, end
                in stencil.get_replicated_reuse_buffers()[tensor.name][1:]
                if start == end]
            offset_modulos = [offset % rf for offset in offsets]
            dst_list = [None]*rf
            for dst_id in range(rf):
                if dst_id in offset_modulos:
                    dst_list[dst_id] = dst_id
            for chan in range(tensor.chan):
                params = []
                for replica_id in range(rf):
                    for offset in sorted(offsets,
                            key = lambda offset: offset%rf):
                        params.append(
                            '/* output */ forward_%s_offset_%d_'
                            'chan_%d_replica_%d' %
                            (tensor.name, offset, chan, replica_id))
                for replica_id in range(rf):
                    params.append(
                        '/*  input */ compute_%s_chan_%d_replica_%d' %
                        (tensor.name, chan, replica_id))
                params.append('/*  param */ epoch_num')
                func_name = ('reconnect_%s' %
                    '_'.join(map(str, filter(None.__ne__, dst_list))))
                printer.PrintFunc(
                    '%s<%s>' % (func_name, tensor.type), params, ';', align=0)

        for node in super_source.tpo_node_generator():
            logger.debug('%s' % repr(node))
            if isinstance(node, SuperSourceNode):
                _print_load_call(printer, stencil)
                for chan in range(stencil.input.chan):
                    for bank in range(stencil.dram_bank):
                        printer.PrintLine('unpack_%s(' %
                            GetSupoType(stencil.input.type))
                        printer.DoIndent()
                        for replica_id in range(
                                stencil.dram_bank-1-bank,
                                stencil.replication_factor,
                                stencil.dram_bank):
                            printer.PrintLine('compute_%s_chan_%d_replica_%d,' %
                                (stencil.input.name, chan, replica_id))
                        printer.PrintLine('input_stream_chan_%d_bank_%d, '
                            'coalesced_data_num);' % (chan, bank))
                        printer.UnIndent()
                print_reconnect(stencil.input)
            elif isinstance(node, ForwardNode):
                for replica_id in range(stencil.replication_factor):
                    for chan in range(node.tensor.chan):
                        params = []
                        output_num = 0
                        for dst_name, points in replicated_all_points[
                                node.tensor.name].items():
                            if node.offset in points:
                                params.append('/* output */ from_%s_offset_%d_'
                                    'to_compute_%s_chan_%d_replica_%d' % (
                                        node.tensor.name, node.offset,
                                        dst_name, chan, replica_id))
                                output_num += 1
                        next_offset = (replicated_next_fifo[node.tensor.name]
                            .get(node.offset))
                        if next_offset is not None:
                            params.append('/* output */ forward_%s_offset_%d_'
                                'chan_%d_replica_%d' % (node.tensor.name,
                                    next_offset, chan, replica_id))
                            output_num += 1
                        params.append('/*  input */ forward_%s_offset_%d_'
                            'chan_%d_replica_%d' % (node.tensor.name,
                                node.offset, chan, replica_id))
                        params.append('/*  param */ epoch_num')
                        printer.PrintFunc('forward_%d<%s, %d>' % (output_num,
                            node.tensor.type, node.depth), params, ';', align=0)
            elif isinstance(node, ComputeNode):
                for replica_id in range(stencil.replication_factor):
                    params = []
                    for chan in range(node.stage.output.chan):
                        params.append('/* output */ compute_%s_chan_%d_'
                            'replica_%d' % (node.stage.output.name, chan,
                                replica_id))
                    for input_name in node.stage.inputs:
                        all_points = replicated_all_points[input_name]
                        for offset in all_points[node.stage.name]:
                            for chan in range(stencil.tensors[input_name].chan):
                                params.append('/*  input */ from_%s_offset_%d_'
                                    'to_compute_%s_chan_%d_replica_%d' % (
                                        input_name, offset,
                                        node.stage.name, chan, replica_id))
                    params.append('/*  param */ epoch_num')
                    printer.PrintFunc('compute_%s<%d>' % (node.stage.name,
                        replica_id), params, ';', align=0)
                if not node.stage.IsOutput():
                    print_reconnect(node.stage.output)
            elif isinstance(node, SuperSinkNode):
                for chan in range(stencil.output.chan):
                    for bank in range(stencil.dram_bank):
                        printer.PrintLine('pack_%s(output_stream_chan_%d_'
                            'bank_%d,' % (GetSupoType(stencil.output.type),
                                chan, bank))
                        printer.DoIndent()
                        for replica_id in range(
                                stencil.dram_bank-1-bank,
                                stencil.replication_factor,
                                stencil.dram_bank):
                            printer.PrintLine('compute_%s_chan_%d_replica_%d,'
                                % (stencil.output.name, chan, replica_id))
                        printer.PrintLine('coalesced_data_num);')
                        printer.UnIndent()
                _print_store_call(printer, stencil)
            else:
                raise SemanticError('unknown dataflow node: %s' % repr(node))

def _print_load_call(printer, stencil):
    for c in range(stencil.input.chan):
        for i in range(stencil.dram_bank):
            printer.PrintLine('load(input_stream_chan_%d_bank_%d, '
                'var_input_chan_%d_bank_%d, coalesced_data_num);' %
                ((c, i)*2))

def _print_store_call(printer, stencil):
    for c in range(stencil.output.chan):
        for i in range(stencil.dram_bank):
            printer.PrintLine('store(var_output_chan_%d_bank_%d, '
                'output_stream_chan_%d_bank_%d, coalesced_data_num);' %
                ((c, i)*2))

def _print_reconnect_func(printer, dst_list):
    rf = len(dst_list)
    params = []
    for replica_id in range(rf):
        for idx, dst in enumerate(dst_list):
            if dst is not None:
                params.append('/* output */ hls::stream<T>& '
                    'dst_%d_replica_%d' % (idx, replica_id))
    for replica_id in range(rf):
        params.append('/*  input */ hls::stream<T>& src_replica_%d' %
            replica_id)
    params.append('/*  param */ uint32_t epoch_num')
    func_name = 'reconnect_%s' % '_'.join(map(str, filter(None.__ne__, dst_list)))
    printer.PrintFunc('template<typename T> void %s' % func_name, params, align=0)
    printer.DoScope()
    for replica_id in range(rf):
        printer.PrintLine('T buf_%d = 0;' % replica_id)

    printer.PrintLine('uint32_t epoch = 0;')
    printer.PrintLine('%s:' % func_name, 0)
    printer.PrintLine('while(epoch < epoch_num)')
    printer.DoScope()
    printer.PrintLine('if(not (%s))' % ' or '.join(
        'src_replica_%d.empty()' % replica_id for replica_id
        in range(rf)))
    printer.DoScope()
    for replica_id in range(rf):
        printer.PrintLine('T val_%d = src_replica_%d.read();' %
            (replica_id, replica_id))

    for replica_id in range(rf):
        for dst_id in dst_list:
            if dst_id is not None:
                if replica_id-dst_id >= 0:
                    printer.PrintLine('dst_%d_replica_%d << val_%d;' %
                        (dst_id, replica_id, replica_id-dst_id))
                else:
                    printer.PrintLine('dst_%d_replica_%d << buf_%d;' %
                        (dst_id, replica_id, rf+replica_id-dst_id))
    for replica_id in range(rf):
        printer.PrintLine('buf_%d = val_%d;' % (replica_id, replica_id))
    printer.PrintLine('++epoch;')
    printer.UnScope()
    printer.UnScope()
    printer.UnScope()

