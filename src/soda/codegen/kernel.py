from collections import OrderedDict
from collections import namedtuple
from functools import reduce
import copy
import logging
import operator

from haoda import ir
from soda import core
from soda import grammar
from soda import util

_logger = logging.getLogger('__main__').getChild(__name__)

def print_compute_input(printer, stencil):
  unroll_factor = stencil.unroll_factor
  all_points = stencil.get_all_points()
  next_fifo = stencil.get_next_fifo()

  for tensor in [stencil.input]:
    for pe_id in range(unroll_factor):
      printer.println('void compute_%s_pe_%d(' % (tensor.name, pe_id))
      printer.do_indent()

      # outputs
      local_offsets = []
      offset = unroll_factor-1-pe_id
      while offset is not None:
        local_offsets.append(offset)
        for output_stage in tensor.children:
          points = all_points[tensor.name][output_stage.name][offset]
          for unroll_index, point in points.items():
            for c in range(stencil.tensors[tensor.name].chan):
              printer.println('/* output */ hls::stream<%s>& from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (tensor.type, tensor.name, output_stage.name, point, c, unroll_index))
        offset = next_fifo[tensor.name].get(offset, None)

      # inputs
      for c in range(stencil.tensors[tensor.name].chan):
        printer.println('/*  input */ hls::stream<%s>& %s_offset_%s_chan_%d,' % (tensor.type, tensor.name, unroll_factor-1-pe_id, c))

      params = []
      if tensor.preserve_border_to():
        for d in range(stencil.dim-1):
          for c in range(tensor.chan):
            param = (tensor.name, d, c, pe_id)
            params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
        params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
      for param in params:
        printer.println('/*  param */ uint32_t %s,' % param)
      printer.println('/*  param */ uint32_t epoch_num)')
      printer.un_indent()
      printer.do_scope()

      print_tensor_statements(printer, local_offsets, unroll_factor, tensor, pe_id)

      printer.println('compute_%s_pe_%d_epoch:' % (tensor.name, pe_id), 0)
      printer.println('for(uint32_t epoch = 0; epoch < epoch_num+%d; ++epoch)' %
        ((local_offsets[-1]-local_offsets[0])//unroll_factor))
      printer.do_scope()
      printer.println('#pragma HLS pipeline II=1', 0)
      print_tensors(printer, stencil, tensor, pe_id)
      printer.un_scope()

      printer.un_scope()
      printer.println()

def print_tensor_statements(printer, local_offsets, unroll_factor, tensor, pe_id):
      pragmas = []
      for begin, end in zip(local_offsets[:-1], local_offsets[1:]):
        for c in range(tensor.chan):
          param = '%s_offset_%s_chan_%d' % (tensor.name, end, c)
          printer.println('hls::stream<%s> %s("%s");' % (tensor.type, param, param))
          pragmas.append('#pragma HLS stream variable=%s depth=%d' % (param, (end-begin)/unroll_factor))
      for pragma in pragmas:
        printer.println(pragma, 0)

def print_tensors(printer, stencil, tensor, pe_id):
  unroll_factor = stencil.unroll_factor
  all_points = stencil.get_all_points()
  next_fifo = stencil.get_next_fifo()

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
      printer.println('bool lower_bound_%d = %s;' % (depth, read_lower))
      printer.println('bool upper_bound_%d = %s;' % (depth, read_upper))
      bounds.add(depth)
    if last_depth not in bounds:
      printer.println('bool lower_bound_%d = %s;' % (last_depth, read_lower))
      printer.println('bool upper_bound_%d = %s;' % (last_depth, read_upper))
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
  printer.println('if(lower_bound_%d) {}' % depths[0])
  for lower_depth, upper_depth, lower_bound, upper_bound in zip(
      depths+depths[:-1], depths[1:]+depths,
      ['lower']*len(depths)+['upper']*(len(depths)-1),
      ['lower']*(len(depths)-1)+['upper']*len(depths)):
    printer.println(
      '%sif(%s_bound_%d)' % (first,
        upper_bound, upper_depth))
    printer.do_scope()
    #printer.println('#pragma HLS latency min=1 max=1', 0)
    read_set = set()
    def print_interval_reads():
      read = reads[depth]
      read_set.add(read)
      for c in range(stencil.tensors[tensor.name].chan):
        printer.println('%s tmp_%s = %s.read();' % (tensor.type, read%c, read%c))
    def print_interval_writes():
      for c in range(stencil.tensors[tensor.name].chan):
        for write, read in writes.get(depth, []):
          if read in read_set:
            printer.println('%s << tmp_%s;' % (write%c, read%c))
          else:
            printer.println('%s << 0;' % write%c)
    for depth in depths:
      if lower_bound == 'lower' and upper_bound == 'lower':
        if lower_depth >= depth:
          print_interval_reads()
      elif lower_bound == 'lower' and upper_bound == 'upper':
        if lower_depth >= depth and upper_depth <= depth:
          print_interval_reads()
      elif lower_bound == 'upper' and upper_bound == 'upper':
        if upper_depth <= depth:
          print_interval_reads()
    for depth in depths:
      if lower_bound == 'lower' and upper_bound == 'lower':
        if lower_depth >= depth:
          print_interval_writes()
      elif lower_bound == 'lower' and upper_bound == 'upper':
        if lower_depth >= depth and upper_depth <= depth:
          print_interval_writes()
      elif lower_bound == 'upper' and upper_bound == 'upper':
        if upper_depth <= depth:
          print_interval_writes()
    if not first:
      first = 'else '
    printer.un_scope()

def print_compute_stage(printer, stencil, stage):
  unroll_factor = stencil.unroll_factor
  all_points = stencil.get_all_points()
  next_fifo = stencil.get_next_fifo()
  reuse_buffers = stencil.get_reuse_buffers()
  stencil_window = util.get_overall_stencil_window(stage.preserve_border_from() if stage.preserve_border_from() else stencil.input, stage.output)
  overall_idx = util.get_stencil_window_offset(stencil_window)
  iteration = 1
  parent_tensor = stage.preserve_border_from()
  tensor = stage.output
  while parent_tensor is not None and parent_tensor.parent is not None:
    parent_tensor = parent_tensor.parent.preserve_border_from()
    iteration += 1
  delay = (util.get_stencil_distance(stencil_window, stencil.tile_size) - util.serialize(overall_idx, stencil.tile_size))*iteration

  for pe_id in range(1 if stencil.cluster == 'none' else unroll_factor):
    if stencil.cluster == 'none':
      printer.println('template<uint32_t pe_id>')
      func_name = stage.name
    else:
      func_name = stage.name + '_pe_%d' % pe_id
    printer.println('void compute_%s(' % func_name)
    printer.do_indent()

    # outputs
    local_offsets = []
    if stencil.cluster == 'none':
      for c in range(stencil.tensors[stage.name].chan):
        printer.println('/* output */ hls::stream<%s>& %s_chan_%d,' % (stage.output.type, stage.name, c))
    elif stage.is_output():
      for c in range(stencil.tensors[stage.name].chan):
        printer.println('/* output */ hls::stream<%s>& %s_offset_%d_chan_%d,' % (stage.output.type, stage.name, unroll_factor-1-pe_id, c))
    else:
      offset = unroll_factor-1-pe_id
      while offset is not None:
        local_offsets.append(offset)
        for output_stage in tensor.children:
          points = all_points[tensor.name][output_stage.name][offset]
          for unroll_index, point in points.items():
            for c in range(stencil.tensors[tensor.name].chan):
              printer.println('/* output */ hls::stream<%s>& from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (tensor.type, tensor.name, output_stage.name, point, c, unroll_index))
        offset = next_fifo[tensor.name].get(offset, None)

    # forwarded
    if stage.output.preserve_border_to():
      for d in range(stencil.dim-1):
        param = 'border_from_%s_dim_%d' % (stage.name, d)
        for c in range(stage.output.chan):
          printer.println(
            '/* output */ hls::stream<%s>& %s_left_chan_%d,' %
              (stage.output.type, param, c))
          printer.println(
            '/* output */ hls::stream<%s>& %s_right_chan_%d,' %
              (stage.output.type, param, c))

    # inputs
    for param in [(stencil.tensors[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, util.get_indices_id(indices))) for input_name, input_window in stage.window.items() for indices in input_window for c in range(stencil.tensors[input_name].chan)]:
      printer.println('/*  input */ hls::stream<%s>& %s,' % param)

    # params
    if stage.preserve_border_from():
      for d in range(stencil.dim-1):
        printer.println('/*  param */ uint32_t input_bound_dim_%d,' % d)
      for d in range(stencil.dim):
        printer.println('/*  param */ uint32_t input_size_dim_%d,' % d)
    printer.println('/*  param */ uint32_t epoch_num)')
    printer.un_indent()
    printer.do_scope()

    if stage.preserve_border_from():
      msg = 'aux parameters for %s' % stage.name
      _logger.debug('generate '+msg)
      printer.println('// '+msg)
      printer.println('int32_t i = pe_id-%d;' % delay)
      for i in range(1, len(stencil.tile_size)):
        printer.println('uint16_t %c = 0;' % COORDS_IN_TILE[i])
      for i in range(len(stencil.tile_size)-1):
        printer.println('uint16_t %c_base = 0;' % COORDS_IN_ORIG[i])
      printer.println()

    bound = ''
    if stencil.cluster != 'none' and not stage.is_output():
      bound = '+%d' % ((local_offsets[-1]-local_offsets[0])//unroll_factor)
      print_tensor_statements(printer, local_offsets, unroll_factor, tensor, pe_id)
      printer.println()
      for c in range(tensor.chan):
        param = '%s_offset_%d_chan_%d' % (tensor.name, unroll_factor-1-pe_id, c)
        printer.println('hls::stream<%s> %s("%s");' % (tensor.type, param, param))
        printer.println('#pragma HLS stream variable=%s depth=1' % param, 0)
      printer.println()

    printer.println('uint32_t epoch = 0;')
    printer.println('compute_%s_epoch:' % func_name, 0)
    printer.println('while(epoch < epoch_num)')
    printer.do_scope()
    printer.println('#pragma HLS pipeline II=1', 0)

    # empty test
    params = []
    for input_name, input_window in stage.window.items():
      for indices in input_window:
        for c in range(stencil.tensors[input_name].chan):
          params.append('%s_chan_%d_at_%s' %
            (input_name, c, util.get_indices_id(indices)))
    printer.println('if(not (%s))' % ' or '.join(
      '%s.empty()' % param for param in params))
    printer.do_scope()

    if stencil.cluster != 'none' and not stage.is_output():
      printer.println('if(epoch < epoch_num)')
      printer.do_scope()

    if stage.preserve_border_from():
      for i in range(len(stencil.tile_size)-1):
        printer.println('uint16_t  %c = %c_base+%c;' % (COORDS_IN_ORIG[i], COORDS_IN_ORIG[i], COORDS_IN_TILE[i]))
      printer.println('uint16_t& %c = %c;' % (COORDS_IN_ORIG[len(stencil.tile_size)-1], COORDS_IN_TILE[len(stencil.tile_size)-1]))

      IndexTile = lambda d: '%c' % (COORDS_IN_TILE[d])
      IndexOrig = lambda d: '%c' % (COORDS_IN_ORIG[d])
      output_idx = util.get_stencil_window_offset(stencil_window)
      stencil_dim = get_stencil_dim(stencil_window)
      MarginCondition = lambda d: ('%s<%d || ' % (IndexOrig(d), output_idx[d]) if output_idx[d]>0 else '') + '%s>input_size_dim_%d-%d+%d' % (IndexOrig(d), d, stencil_dim[d], output_idx[d])
      printer.println('bool margin_conditions[%d];' % stencil.dim)
      #printer.println('#pragma HLS array_partition variable=margin_conditions complete', 0)
      printer.println('#pragma HLS resource variable=margin_conditions latency=1 core=RAM_2P_LUTRAM', 0)
      for d in range(stencil.dim):
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        printer.println('margin_conditions[%d] = %s;' % (d, MarginCondition(d)))
        printer.un_scope()
      printer.println()

    for input_name, input_window in stage.window.items():
      params = []
      for indices in input_window:
        for c in range(stencil.tensors[input_name].chan):
          params.append((stencil.tensors[input_name].type, '%s_chan_%d_at_%s' % (input_name, c, util.get_indices_id(indices))))

      for param in params:
        printer.println('%s load_%s = %s.read();' % (param[0], param[1], param[1]))
      printer.println()

    if stage.preserve_border_from():
      printer.println('if(%s)' % (' || '.join('margin_conditions[%d]' % d for d in range(stencil.dim))))
      printer.do_scope()
      preserve_border_from = stage.preserve_border_from()
      printer.println('%s_chan_%d<<load_%s_chan_%d_at_%s;' % (stage.name, c, preserve_border_from.name, c, util.get_indices_id(stage.idx)))
      #printer.println('printf("bypass: epoch%%d pe%%d %s %s val=%%d\\n", epoch, pe_id, %s, %s load_%s_chan_%d_at_%s);' % (' '.join('%c=%%d' % COORDS_IN_TILE[d] for d in range(stencil.dim)), ' '.join('%c=%%d' % COORDS_IN_ORIG[d] for d in range(stencil.dim)), ', '.join(COORDS_IN_TILE[:stencil.dim]), ', '.join(COORDS_IN_ORIG[:stencil.dim]), preserve_border_from.name, c, util.get_indices_id(stage.idx)))
      printer.un_scope()
      printer.println('else')
      printer.do_scope()

    LoadPrinter = lambda node: 'param_%s%s[unroll_index]%s' % (node.name, '' if stencil.extra_params[node.name].dup is None else '[%d]' % node.chan, ''.join(['[%d]'%x for x in node.idx])) if node.name in stencil.extra_params else 'load_%s_chan_%d_at_%s' % (node.name, node.chan, util.get_indices_id(node.idx))
    StorePrinter = lambda node: '%s store_%s_chan_%d' % (stage.output.type, node.name, node.chan)

    for expr in stage.expr:
      expr.print_code(printer, stencil.tensors, LoadPrinter, StorePrinter, add_latency=True)

    for c in range(stage.output.chan):
      if stencil.cluster == 'none':
        printer.println('%s_chan_%d<<store_%s_chan_%d;' % ((stage.name, c)*2))
      else:
        printer.println('%s_offset_%d_chan_%d<<store_%s_chan_%d;' % ((stage.name, unroll_factor-1-pe_id, c, stage.name, c)))
      #printer.println('printf("calc: epoch%%d pe%%d %%d inputs=%s val=%%d\\n", epoch, pe_id, epoch*%d+pe_id-%d, %s, store_%s_chan_%d);' % (' '.join(['%d']*len(params)), stencil.unroll_factor, delay, ', '.join('load_%s'%p[1] for p in params), stage.name, c))

    if stage.output.preserve_border_to():
      printer.println()
      for d in range(stencil.dim-1):
        if stencil_dim[d] < 2:
          continue
        printer.println('if(%s >= %d-1 && %s < input_size_dim_%d-%d+1)' % (IndexOrig(d), stencil_dim[d], IndexOrig(d), d, stencil_dim[d]))
        printer.do_scope()
        printer.println('switch(%s)' % IndexTile(d))
        printer.do_scope()

        for i in range(output_idx[d], stencil_dim[d]-1):
          printer.println('case %d:' % i)
        printer.do_scope()
        printer.println('// duplicate output to border buffer')
        for c in range(stage.output.chan):
          printer.println('border_from_%s_dim_%d_right_chan_%d<<store_%s_chan_%d;' % (stage.name, d, c, stage.name, c))
        printer.println('break;')
        printer.un_scope()

        for i in range(stencil.tile_size[d]-stencil_dim[d]+1, stencil.tile_size[d]-stencil_dim[d]+output_idx[d]+1):
          printer.println('case %d:' % i)
        printer.do_scope()
        printer.println('// duplicate output to border buffer')
        for c in range(stage.output.chan):
          printer.println('border_from_%s_dim_%d_left_chan_%d<<store_%s_chan_%d;' % (stage.name, d, c, stage.name, c))
        printer.println('break;')
        printer.un_scope()

        printer.un_scope()
        printer.un_scope()
    if stage.preserve_border_from():
      printer.un_scope()
      printer.println()
      print_increment_coordinates(printer, stencil, stage)

    if stencil.cluster == 'fine' and not stage.is_output():
      printer.un_scope()
      printer.println()
      print_tensors(printer, stencil, stage.output, pe_id)
    printer.println('++epoch;')
    printer.un_scope()
    printer.un_scope()
    printer.un_scope()
    printer.println()

def print_increment_coordinates(printer, stencil, stage):
  overall_stencil_window = util.get_overall_stencil_window(*([stage.preserve_border_from(), stage.output] if stencil.preserve_border else [stencil.input, stencil.output]))
  overall_stencil_dim = get_stencil_dim(overall_stencil_window)

  PrintIfTile = lambda d: printer.println('if(%c>=TILE_SIZE_DIM_%d)' % (COORDS_IN_TILE[d], d))
  PrintIfTileLastDim = lambda d: printer.println('if(%c >= input_size_dim_%d)' % (COORDS_IN_TILE[d], d))
  PrintIfTensor = lambda d: printer.println('if(%c >= input_size_dim_%d)' % (COORDS_IN_ORIG[d], d))
  PrintIncrementTile = lambda d: printer.println('++%c;' % (COORDS_IN_TILE[d]))
  PrintDecrementTile = lambda d: printer.println('%c -= TILE_SIZE_DIM_%d;' % (COORDS_IN_TILE[d], d))
  PrintIncrementOrig = lambda d: printer.println('%c_base += TILE_SIZE_DIM_%d - %s + 1;' % (COORDS_IN_ORIG[d], d, overall_stencil_dim[d]))
  PrintDecrementOrig = lambda d: printer.println('%c_base = 0;' % COORDS_IN_ORIG[d])
  PrintDecrementTileLastDim = lambda d: printer.println('%c -= input_size_dim_%d;' % (COORDS_IN_TILE[d], d))

  printer.println('if(%s)' % ' && '.join('%c_base<input_bound_dim_%d' % (COORDS_IN_ORIG[d], d) for d in range(stencil.dim-1)))
  printer.do_scope()
  printer.println('i+=%d;' % stencil.unroll_factor)
  if len(stencil.tile_size)>1:
    PrintIfTile(0)
    printer.do_scope()
    printer.println('#pragma HLS latency min=1', 0)
    PrintDecrementTile(0)
    PrintIncrementTile(1)
    if len(stencil.tile_size)>2:
      PrintIfTile(1)
      printer.do_scope()
      printer.println('#pragma HLS latency min=1', 0)
      PrintDecrementTile(1)
      PrintIncrementTile(2)
      if len(stencil.tile_size)>3:
        PrintIfTile(2)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementTile(2)
        PrintIncrementTile(3)

        PrintIfTileLastDim(3)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementTileLastDim(3)
        PrintIncrementOrig(0)
        PrintIfTensor(0)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementOrig(0)
        PrintIncrementOrig(1)
        PrintIfTensor(1)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementOrig(1)
        PrintIncrementOrig(2)
        printer.un_scope()
        printer.un_scope()
        printer.un_scope()

        printer.un_scope()
      else:
        PrintIfTileLastDim(2)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementTileLastDim(2)
        PrintIncrementOrig(0)

        PrintIfTensor(0)
        printer.do_scope()
        printer.println('#pragma HLS latency min=1', 0)
        PrintDecrementOrig(0)
        PrintIncrementOrig(1)
        printer.un_scope()

        printer.un_scope()
      printer.un_scope()
    else:
      PrintIfTileLastDim(1)
      printer.do_scope()
      printer.println('#pragma HLS latency min=1', 0)
      PrintDecrementTileLastDim(1)
      PrintIncrementOrig(0)
      printer.un_scope()
    printer.un_scope()
  else:
    PrintIfTileLastDim(0)
    printer.do_scope()
    printer.println('#pragma HLS latency min=1', 0)
    PrintDecrementTileLastDim(0)
    printer.un_scope()
  printer.un_scope()

def print_interface(p, stencil):
  tile_size = stencil.tile_size
  unroll_factor = stencil.unroll_factor
  app_name = stencil.app_name
  extra_params = stencil.extra_params
  dram_bank = stencil.dram_bank
  dram_separate = stencil.dram_separate
  input_chan = stencil.input.chan
  output_chan = stencil.output.chan
  super_source = stencil.dataflow_super_source

  _logger.info('generate reuse buffers')
  reuse_buffers = stencil.get_reuse_buffers()
  all_points = stencil.get_all_points()
  next_fifo = stencil.get_next_fifo()
  overall_stencil_window = util.get_overall_stencil_window(stencil.input, stencil.output)

  p.println('extern "C"')
  p.println('{')
  p.println()
  p.println('void %s_kernel(' % app_name)
  p.do_indent()
  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('ap_uint<BURST_WIDTH>* var_output_chan_%d_bank_%d,' % (c, i))
  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('ap_uint<BURST_WIDTH>* var_input_chan_%d_bank_%d,' % (c, i))
  if extra_params:
    for param in extra_params.values():
      p.println('%s* var_%s,' % (param.type, param.name))
  p.println('uint64_t coalesced_data_num,')
  p.println('uint64_t tile_data_num,')
  for i in range(stencil.dim-1):
    p.println('uint32_t input_bound_dim_%d,' % i)
  for d in range(stencil.dim-1):
    p.println('uint32_t input_size_dim_%d,' % d)
  p.println('uint32_t input_size_dim_%d)' % (stencil.dim-1))
  p.un_indent()
  p.do_scope()

  bank = 0
  for i in range(dram_bank):
    for c in range(output_chan):
      p.println('#pragma HLS interface m_axi port=var_output_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%do latency=120' % (c, i, c, bank), 0)
    bank += 1
  if not dram_separate:
    bank = 0
  for i in range(dram_bank):
    for c in range(input_chan):
      p.println('#pragma HLS interface m_axi port=var_input_chan_%d_bank_%d offset=slave depth=65536 bundle=chan%dbank%di latency=120' % (c, i, c, bank), 0)
    bank += 1
  if extra_params:
    for idx, param in enumerate(extra_params.values()):
      p.println('#pragma HLS interface m_axi port=var_%s offset=slave depth=%d bundle=gmem%d latency=120' % (param.name, reduce(operator.mul, param.size), idx), 0)
  p.println()
  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('#pragma HLS interface s_axilite port=var_output_chan_%d_bank_%d bundle=control' % (c, i), 0)
  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('#pragma HLS interface s_axilite port=var_input_chan_%d_bank_%d bundle=control' % (c, i), 0)
  if extra_params:
    for param in extra_params.values():
      p.println('#pragma HLS interface s_axilite port=var_%s bundle=control' % param.name, 0)
  p.println('#pragma HLS interface s_axilite port=coalesced_data_num bundle=control', 0)
  p.println('#pragma HLS interface s_axilite port=tile_data_num bundle=control', 0)
  for d in range(stencil.dim-1):
    p.println('#pragma HLS interface s_axilite port=input_bound_dim_%d bundle=control' % d, 0)
  for d in range(stencil.dim):
    p.println('#pragma HLS interface s_axilite port=input_size_dim_%d bundle=control' % d, 0)
  p.println('#pragma HLS interface s_axilite port=return bundle=control', 0)
  p.println()

  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('hls::stream<ap_uint<BURST_WIDTH> >  input_stream_chan_%d_bank_%d( "input_stream_chan_%d_bank_%d");' % ((c, i)*2))
  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('hls::stream<ap_uint<BURST_WIDTH> > output_stream_chan_%d_bank_%d("output_stream_chan_%d_bank_%d");' % ((c, i)*2))
  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('#pragma HLS stream variable=input_stream_chan_%d_bank_%d depth=32' % (c, i), 0)
  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('#pragma HLS stream variable=output_stream_chan_%d_bank_%d depth=32' % (c, i), 0)
  p.println()

  if extra_params:
    for param in extra_params.values():
      if param.dup:
        dup = ('[%d]' % param.dup)
      else:
        dup = ''
      p.println('%s %s%s[UNROLL_FACTOR][%s];' % (param.type, param.name, dup, ']['.join(map(str, param.size))))
    p.println()

    for param in extra_params.values():
      p.println('#pragma HLS array_partition variable=%s complete dim=1' % param.name, 0)
      dim_offset = 1
      if param.dup:
        p.println('#pragma HLS array_partition variable=%s complete dim=2' % param.name, 0)
        dim_offset = 2
      for partitioning in param.partitioning:
        p.println('#pragma HLS array_partition variable=%s %s dim=%d%s' % (
          param.name,
          partitioning.partition_type,
          dim_offset+1 if partitioning.dim is None else partitioning.dim+dim_offset,
          '' if partitioning.factor is None else ' factor=%d' % partitioning.factor,
        ), 0)
    p.println()

    for param in extra_params.values():
      if len(param.size) > 1:
        for dim, size in enumerate(param.size):
          p.println('uint32_t %s_index_dim_%d = 0;' % (param.name, dim))
      p.println('%s_init:' % param.name, 0)
      p.println('for(int %s_index = 0; %s_index < %d; ++%s_index)' % (param.name, param.name, reduce(operator.mul, param.size), param.name))
      p.do_scope()
      p.println('#pragma HLS pipeline II=1', 0)
      p.println('%s& %s_tmp = var_%s[%s_index];' % (param.type, param.name, param.name, param.name))
      p.println('%s_unrolled:' % param.name, 0)
      p.println('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
      p.do_scope()
      p.println('#pragma HLS unroll',0)
      if param.dup is None:
        p.println('%s[unroll_index]%s = %s_tmp;' % ((param.name, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name)))
      else:
        for i in range(param.dup):
          p.println('%s[%d][unroll_index]%s = %s_tmp;' % (param.name, i, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name))
      p.un_scope()
      if len(param.size) > 1:
        for dim in range(len(param.size)):
          p.println('++%s_index_dim_%d;' % (param.name, dim))
          if dim<len(param.size)-1:
            p.println('if(%s_index_dim_%d==%d)' % (param.name, dim, param.size[len(param.size)-1-dim]))
            p.do_scope()
            p.println('%s_index_dim_%d = 0;' % (param.name, dim))
      for size in param.size[:-1]:
        p.un_scope()
      p.un_scope()
    p.println()

  extra_params_str = ''.join([param.name+', ' for param in extra_params.values()])

  p.println('uint64_t epoch_num = coalesced_data_num*%d/%d;' % (
    stencil.burst_width*stencil.dram_bank/util.TYPE_WIDTH[stencil.input.type],
    unroll_factor))
  p.println()

  if stencil.replication_factor > 1:
    _generate_code(p, stencil)

    p.un_scope()
    p.println()
    p.println('}//extern "C"')

    return

  GetTensorAt = lambda n, o, c: ('%s_offset_%d_chan_%d') % (n, o, c)

  # reuse buffers
  if stencil.cluster == 'none':
    for name, reuse_buffer in reuse_buffers.items():
      pragmas = []
      msg = 'reuse buffers for %s' % name
      _logger.debug('generate %s' % msg)
      p.println('// %s' % msg)
      for start, end in reuse_buffer[1:]:
        for c in range(stencil.tensors[name].chan):
          p.println('hls::stream<%s> %s("%s");' % ((stencil.tensors[name].type,)+(GetTensorAt(name, end, c),)*2))
          buffer_length = stencil.get_reuse_buffer_length(name, end)
          tensor_name = GetTensorAt(name, end, c)
          if buffer_length > 1:
            pragmas.append((tensor_name, buffer_length))
      for pragma in pragmas:
        p.println('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
      p.println()
  else:
    p.println('// %s' % stencil.input.name)
    for unroll_index in range(unroll_factor):
      for c in range(stencil.input.chan):
        p.println('hls::stream<%s> %s("%s");' % ((stencil.tensors[stencil.input.name].type,)+(GetTensorAt(stencil.input.name, unroll_index, c),)*2))
    p.println()

  p.println('// %s' % stencil.output.name)
  for unroll_index in range(unroll_factor):
    for c in range(stencil.output.chan):
      p.println('hls::stream<%s> %s("%s");' % ((stencil.tensors[stencil.output.name].type,)+(GetTensorAt(stencil.output.name, unroll_index, c),)*2))
  p.println()

  # params
  msg = 'params'
  _logger.debug('generate %s' % msg)
  p.println('// %s' % msg)
  pragmas = []
  for stage in stencil.get_stages_chronologically():
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
          for c in range(stencil.tensors[input_name].chan):
            var_type = stencil.tensors[input_name].type
            var_name = 'from_%s_to_%s_param_%d_chan_%d_pe_%d' % (
              input_name, stage.name, i, c, pe_id)
            p.println('hls::stream<%s> %s("%s");' % (
              var_type, var_name, var_name))
            if extra_depth > 0:
              pragmas.append((var_name, extra_depth+1))
            else:
              pragmas.append((var_name, 2))
  for pragma in pragmas:
    p.println('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
  p.println()

  # border buffers
  msg = 'border buffers'
  _logger.debug('generate %s' % msg)
  p.println('// %s' % msg)
  for stage in stencil.get_stages_chronologically():
    if stage.output.preserve_border_to():
      for unroll_index in range(unroll_factor):
        for d in range(stencil.dim-1):
          for c in range(stage.output.chan):
            param = (stage.output.type, 'border_from_%s_dim_%d_left_chan_%d_pe_%d' % (stage.name, d, c, unroll_index))
            p.println('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
            param = (stage.output.type, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % (stage.name, d, c, unroll_index))
            p.println('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
  p.println()

  p.println('#pragma HLS dataflow', 0)
  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('load(input_stream_chan_%d_bank_%d, var_input_chan_%d_bank_%d, coalesced_data_num);' % ((c, i)*2))
  for c in range(input_chan):
    for i in range(dram_bank):
      p.println('unpack_%s(' % util.get_soda_type(stencil.input.type))
      p.do_indent()
      for unroll_index in reversed(range(dram_bank-1-i, unroll_factor, dram_bank)):
        p.println('%s,' % GetTensorAt(stencil.input.name, stencil.input.offset+unroll_index, c))
      p.println('input_stream_chan_%d_bank_%d, coalesced_data_num);' % (c, i))
      p.un_indent()
  p.println()

  output_stream = ', '.join(', '.join('output_stream_chan_%d_bank_%d' % (c, x) for x in range(dram_bank)) for c in range(output_chan))
  input_stream = ', '.join(', '.join('input_stream_chan_%d_bank_%d' % (c, x) for x in range(dram_bank)) for c in range(input_chan))
  tile_num_dim = ', '.join('tile_num_dim_%d' % d for d in range(stencil.dim-1))
  input_size_dim = ', '.join('input_size_dim_%d' % d for d in range(stencil.dim))
  next_fifo = stencil.get_next_fifo()
  if stencil.cluster == 'none':
    print_forward_call(p, stencil, stencil.input.name)
    p.println()

    for stage in stencil.get_stages_chronologically():
      inputs = tuple(reversed(range(unroll_factor))) if stage.is_output() else [start for start, end in stencil.get_reuse_buffers()[stage.name][1:] if start==end]
      for unroll_index in range(unroll_factor):
        params = []
        for c in range(stage.output.chan):
          params.append('/* output */ %s_offset_%s_chan_%d' %
            (stage.name, inputs[unroll_index], c))
        if stage.output.preserve_border_to():
          for d in range(stencil.dim-1):
            for c in range(stage.output.chan):
              param = (stage.name, d, c, unroll_index)
              params += [
      '/* output */ border_from_%s_dim_%d_left_chan_%d_pe_%d'  % param,
      '/* output */ border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
        for input_name, input_window in stage.window.items():
          for i in range(len(input_window)):
            for c in range(stencil.tensors[input_name].chan):
              params += [
            '/*  input */ from_%s_to_%s_param_%d_chan_%d_pe_%d'
            % (input_name, stage.name, i, c, unroll_index)]
        if stage.preserve_border_from():
          for d in range(stencil.dim-1):
            params.append('/*  param */ input_bound_dim_%d' % d)
          for d in range(stencil.dim):
            params.append('/*  param */ input_size_dim_%d' % d)
        params.append('/*  param */ epoch_num')
        p.print_func('compute_%s<%d>' % (stage.name, unroll_index),
          params, ';', 0)

      if not stage.is_output():
        p.println()
        print_forward_call(p, stencil, stage.name)
      p.println()
  elif stencil.cluster == 'fine':
    for tensor in [stencil.input]+[stage.output for stage in stencil.get_stages_chronologically()]:
      inputs = tuple(reversed(range(unroll_factor))) if tensor.is_output() else [start for start, end in stencil.get_reuse_buffers()[tensor.name][1:] if start==end]
      for pe_id in range(unroll_factor):
        p.println('compute_%s_pe_%d(' % (tensor.name, pe_id))
        p.do_indent()

        # outputs
        offset = unroll_factor-1-pe_id
        if tensor.is_output():
          for c in range(stencil.tensors[tensor.name].chan):
            p.println('/* output */ %s_offset_%s_chan_%d,' % (tensor.name, offset, c))
        else:
          while offset is not None:
            for output_stage in tensor.children:
              points = all_points[tensor.name][output_stage.name][offset]
              for unroll_index, point in points.items():
                for c in range(stencil.tensors[tensor.name].chan):
                  p.println('/* output */ from_%s_to_%s_param_%d_chan_%d_pe_%d,' % (tensor.name, output_stage.name, point, c, unroll_index))
            offset = next_fifo[tensor.name].get(offset, None)

        # inputs
        if tensor.is_input():
          for c in range(stencil.tensors[stencil.input.name].chan):
            p.println('/*  input */ %s_offset_%s_chan_%d,' % (stencil.input.name, unroll_factor-1-pe_id, c))
        else:
          for param in ['from_%s_to_%s_param_%d_chan_%d_pe_%d' % (input_name, tensor.name, i, c, pe_id) for input_name, input_window in tensor.parent.window.items() for i in range(len(input_window)) for c in range(stencil.tensors[input_name].chan)]:
            p.println('/*  input */ %s,' % param)

        params = []
        if tensor.preserve_border_to():
          for d in range(stencil.dim-1):
            for c in range(tensor.chan):
              param = (tensor.name, d, c, pe_id)
              params += ['border_from_%s_dim_%d_left_chan_%d_pe_%d' % param, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % param]
        if tensor.preserve_border_from():
          params += ['input_bound_dim_%d' % d for d in range(stencil.dim-1)]
          params += ['input_size_dim_%d' % d for d in range(stencil.dim)]
        for param in params:
          p.println('/*  param */ %s,' % param)
        p.println('/*  param */ epoch_num);')
        p.un_indent()
      p.println()

  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('pack_%s(output_stream_chan_%d_bank_%d,' % (util.get_soda_type(stencil.output.type), c, i))
      p.do_indent()
      for unroll_index in reversed(range(dram_bank-1-i, unroll_factor, dram_bank)):
        p.println('%s,' % GetTensorAt(stencil.output.name, unroll_index, c))
      p.println('coalesced_data_num);')
      p.un_indent()
  for c in range(output_chan):
    for i in range(dram_bank):
      p.println('store(var_output_chan_%d_bank_%d, output_stream_chan_%d_bank_%d, coalesced_data_num);' % ((c, i)*2))

  p.un_scope()
  p.println()
  p.println('}//extern "C"')

def print_header(p):
  for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio', 'string', 'ap_int', 'hls_stream']:
    p.println('#include<%s.h>' % header)
  p.println()

def print_load(printer):
  printer.println('void load(hls::stream<ap_uint<BURST_WIDTH> >& to, ap_uint<BURST_WIDTH>* from, uint64_t data_num)')
  printer.do_scope()
  printer.println('load_epoch:', 0)
  printer.println('for(uint64_t i = 0; i < data_num; ++i)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=1', 0)
  printer.println('to<<from[i];')
  printer.un_scope()
  printer.un_scope()

def print_unpack(printer, burst_width, data_type, unroll_factor):
  coalesced_size = burst_width//util.TYPE_WIDTH[data_type]
  ii = 1
  if coalesced_size > unroll_factor:
    ii = coalesced_size/unroll_factor
  GetCoalescedIdx = lambda i: ('%'+str(len(str(coalesced_size)))+'d') % i
  GetDstName = lambda i: ('to_%0'+str(len(str(unroll_factor-1)))+'d') % i
  printer.println('void unpack_%s(' % util.get_soda_type(data_type))
  printer.do_indent()
  for unroll_index in range(unroll_factor):
    printer.println(('hls::stream<%s>& %s,') % (data_type, GetDstName(unroll_index)))
  printer.println('hls::stream<ap_uint<%d> >& from, uint64_t data_num)' % burst_width)
  printer.un_indent()
  printer.do_scope()
  printer.println('unpack_%s_epoch:' % util.get_soda_type(data_type), 0)
  printer.println('for(uint64_t i = 0; i < data_num; ++i)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=%d' % ii, 0)
  printer.println('ap_uint<%d> tmp;' % burst_width)
  printer.println('from>>tmp;')
  if util.is_float(data_type):
    printer.println('uint%d_t raw_bits;' % util.TYPE_WIDTH[data_type])
  if coalesced_size >= unroll_factor:
    for i in range(coalesced_size):
      if util.is_float(data_type):
        printer.println('raw_bits = tmp(%s*%d-1, %s*%d); %s<<*((%s*)(&raw_bits));' % (GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], GetDstName(i%unroll_factor), data_type))
      else:
        printer.println('%s<<tmp(%s*%d-1, %s*%d);' % (GetDstName(i%unroll_factor), GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type]))
  else:
    printer.println('switch(i&%d)' % (unroll_factor//coalesced_size-1))
    printer.do_scope()
    for batch in range(unroll_factor//coalesced_size):
      printer.println('case %d:' % batch)
      printer.do_scope()
      for i in range(coalesced_size):
        if util.is_float(data_type):
          printer.println('raw_bits = tmp(%s*%d-1, %s*%d);%s<<*((%s*)(&raw_bits));' % (GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], GetDstName(i+batch*coalesced_size), data_type))
        else:
          printer.println('%s<<tmp(%s*%d-1, %s*%d);' % (GetDstName(i+batch*coalesced_size), GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type]))
      printer.println('break;')
      printer.un_scope()
    printer.un_scope()
  printer.un_scope()
  printer.un_scope()

def print_pack(printer, burst_width, data_type, unroll_factor):
  coalesced_size = burst_width//util.TYPE_WIDTH[data_type]
  ii = 1
  if coalesced_size > unroll_factor:
    ii = coalesced_size/unroll_factor
  GetCoalescedIdx = lambda i: ('%'+str(len(str(coalesced_size)))+'d') % i
  GetDstName = lambda i: ('from_%0'+str(len(str(unroll_factor-1)))+'d') % i
  printer.println('void pack_%s(hls::stream<ap_uint<%d> >& to,' % (util.get_soda_type(data_type), burst_width))
  printer.do_indent()
  for unroll_index in range(unroll_factor):
    printer.println(('hls::stream<%s>& %s,') % (data_type, GetDstName(unroll_index)))
  printer.println('uint64_t data_num)')
  printer.un_indent()
  printer.do_scope()
  printer.println('pack_%s_epoch:' % util.get_soda_type(data_type), 0)
  printer.println('for(uint64_t i = 0; i < data_num; ++i)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=%d' % ii, 0)
  printer.println('ap_uint<%d> tmp;' % burst_width)
  if util.is_float(data_type):
    printer.println('%s raw_bits;' % data_type)
  if coalesced_size >= unroll_factor:
    for i in range(coalesced_size):
      if util.is_float(data_type):
        printer.println('%s>>raw_bits; tmp(%s*%d-1, %s*%d) = *((uint%d_t*)(&raw_bits));' % (GetDstName(i%unroll_factor), GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], util.TYPE_WIDTH[data_type]))
      else:
        printer.println('tmp(%s*%d-1, %s*%d) = %s.read();' % (GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], GetDstName(i%unroll_factor)))
  else:
    printer.println('switch(i&%d)' % (unroll_factor//coalesced_size-1))
    printer.do_scope()
    for batch in range(unroll_factor//coalesced_size):
      printer.println('case %d:' % batch)
      printer.do_scope()
      for i in range(coalesced_size):
        if util.is_float(data_type):
          printer.println('%s>>raw_bits; tmp(%s*%d-1, %s*%d) = *((uint%d_t*)(&raw_bits));' % (GetDstName(i+batch*coalesced_size), GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], util.TYPE_WIDTH[data_type]))
        else:
          printer.println('tmp(%s*%d-1, %s*%d) = %s.read();' % (GetCoalescedIdx(i+1), util.TYPE_WIDTH[data_type], GetCoalescedIdx(i), util.TYPE_WIDTH[data_type], GetDstName(i+batch*coalesced_size)))
      printer.println('break;')
      printer.un_scope()
    printer.un_scope()
  printer.println('to<<tmp;')
  printer.un_scope()
  printer.un_scope()

def print_store(printer):
  printer.println('void store(ap_uint<BURST_WIDTH>* to, hls::stream<ap_uint<BURST_WIDTH> >& from, uint64_t data_num)')
  printer.do_scope()
  printer.println('store_epoch:', 0)
  printer.println('for(uint64_t i = 0; i < data_num; ++i)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=1', 0)
  printer.println('from>>to[i];')
  printer.un_scope()
  printer.un_scope()

def print_forward_func(printer, forwarder):
  printer.print_func('template<typename T, uint32_t fifo_depth> void forward_%d' % forwarder, ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]+['hls::stream<T>& src']+['uint32_t data_num'])
  printer.do_scope()
  printer.println('forward_%d_epoch:' % forwarder, 0)
  printer.println('for(uint32_t i = 0; i < data_num+fifo_depth; ++i)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=1', 0)
  printer.println('T tmp;')
  printer.println('if(i<fifo_depth)')
  printer.do_scope()
  printer.println('tmp = 0;')
  printer.un_scope()
  printer.println('else')
  printer.do_scope()
  printer.println('tmp = src.read();')
  printer.un_scope()
  printer.println('if(i<data_num)')
  printer.do_scope()
  for dst in range(forwarder):
    printer.println('dst_%d<<tmp;' % dst)
  printer.un_scope()
  printer.un_scope()
  printer.un_scope()

def print_forward_func_with_border(printer, stencil, forwarder_with_border):
  src_name = forwarder_with_border[0]
  forwarder = forwarder_with_border[1]
  stage = stencil.stages[src_name]
  stencil_window = util.get_overall_stencil_window(stage.preserve_border_from(), stage.output)

  params = ['hls::stream<T>& dst_%d'%i for i in range(forwarder)]
  params += ['hls::stream<T>& src']
  for param in ['border_dim_%d' % d for d in range(stencil.dim-1)]:
    params += ['hls::stream<T>& %s_left' % param, 'hls::stream<T>& %s_right' % param]
  params += ['uint32_t input_bound_dim_%d' % d for d in range(stencil.dim-1)]
  params += ['uint32_t input_size_dim_%d' % d for d in range(stencil.dim)]
  params += ['uint32_t data_num']
  printer.print_func('template<typename T, int32_t i_init> void forward_%s_%d' % forwarder_with_border, params)
  printer.do_scope()
  printer.println(' int32_t i = i_init;')
  for i in range(1, len(stencil.tile_size)):
    printer.println('uint16_t %c = 0;' % COORDS_IN_TILE[i])
  for i in range(len(stencil.tile_size)-1):
    printer.println('uint16_t %c_base = 0;' % COORDS_IN_ORIG[i])
  printer.println()
  printer.println('forward_%s_%d_epoch:' % forwarder_with_border, 0)
  printer.println('for(uint32_t epoch = 0; epoch < data_num; ++epoch)')
  printer.do_scope()
  printer.println('#pragma HLS pipeline II=1', 0)

  for i in range(len(stencil.tile_size)-1):
    printer.println('uint16_t  %c = %c_base+%c;' % (COORDS_IN_ORIG[i], COORDS_IN_ORIG[i], COORDS_IN_TILE[i]))
  printer.println('uint16_t& %c = %c;' % (COORDS_IN_ORIG[len(stencil.tile_size)-1], COORDS_IN_TILE[len(stencil.tile_size)-1]))

  IndexTile = lambda d: '%c' % (COORDS_IN_TILE[d])
  IndexOrig = lambda d: '%c' % (COORDS_IN_ORIG[d])
  output_idx = util.get_stencil_window_offset(stencil_window)
  stencil_dim = get_stencil_dim(stencil_window)
  MarginCondition = lambda d: ('%s<%d || ' % (IndexOrig(d), output_idx[d]) if output_idx[d]>0 else '') + '%s>input_size_dim_%d-%d+%d' % (IndexOrig(d), d, stencil_dim[d], output_idx[d])
  printer.println('bool margin_conditions[%d];' % stencil.dim)
  #printer.println('#pragma HLS array_partition variable=margin_conditions complete', 0)
  printer.println('#pragma HLS resource variable=margin_conditions latency=1 core=RAM_2P_LUTRAM', 0)
  for d in range(stencil.dim):
    printer.do_scope()
    printer.println('#pragma HLS latency min=1', 0)
    printer.println('margin_conditions[%d] = %s;' % (d, MarginCondition(d)))
    printer.un_scope()
  printer.println()

  printer.println('T tmp(src.read());')

  printer.println('if(!(%s))' % ' || '.join('margin_conditions[%d]' % d for d in range(stencil.dim)))
  printer.do_scope()
  for d in range(stencil.dim-1):
    printer.println('switch(%s)' % IndexTile(d))
    printer.do_scope()

    for i in range(output_idx[d]):
      printer.println('case %d:' % i)
    printer.do_scope()
    for c in range(stage.output.chan):
      printer.println('tmp = border_dim_%d_left.read();' % d)
    printer.println('break;')
    printer.un_scope()

    for i in range(stencil.tile_size[d]-stencil_dim[d]+output_idx[d]+1, stencil.tile_size[d]):
      printer.println('case %d:' % i)
    printer.do_scope()
    for c in range(stage.output.chan):
      printer.println('tmp = border_dim_%d_right.read();' % d)
    printer.println('break;')
    printer.un_scope()

    printer.un_scope()
  printer.un_scope()

  for dst in range(forwarder):
    printer.println('dst_%d<<tmp;' % dst)
  printer.println()

  print_increment_coordinates(printer, stencil, stage)
  printer.un_scope()
  printer.un_scope()

def print_forward_call(printer, stencil, src_name):
  forwardings = stencil.get_forwardings(src_name)
  for offset, args in sorted(forwardings.items()):
    forward_num = len(args[1])
    temp_param = forwardings[offset][4]
    func_name = '%s_%d<%s, %s>' % (args[0], forward_num,
      stencil.tensors[src_name].type, temp_param)
    for c in range(stencil.tensors[src_name].chan):
      printer.print_func(
        func_name,
        [s%c for s in args[1]]+
        [s%c for s in args[2]]+
        args[3], ';', 0)

def print_code(stencil, output_file):
  _logger.info('generate kernel code as %s' % output_file.name)
  printer = util.Printer(output_file)

  print_header(printer)

  printer.println()

  core.print_define(printer, 'BURST_WIDTH', stencil.burst_width)
  printer.println()

  core.print_guard(printer, 'UNROLL_FACTOR', stencil.unroll_factor)
  for i in range(len(stencil.tile_size)-1):
    core.print_guard(printer, 'TILE_SIZE_DIM_%d' % i, stencil.tile_size[i])
  core.print_guard(printer, 'BURST_WIDTH', stencil.burst_width)
  printer.println()

  print_load(printer)
  printer.println()
  for data_type in {util.get_c_type(stmt.soda_type)
                    for stmt in stencil.input_stmts}:
    print_unpack(printer, stencil.burst_width, data_type, stencil.unroll_factor//stencil.dram_bank)
    printer.println()
  for data_type in {util.get_c_type(stmt.soda_type)
                    for stmt in stencil.output_stmts}:
    print_pack(printer, stencil.burst_width, data_type, stencil.unroll_factor//stencil.dram_bank)
    printer.println()
  print_store(printer)
  printer.println()

  print_data_struct(printer)
  print_read_data(printer)
  print_write_data(printer)

  node_signatures = OrderedDict()
  for node in stencil.dataflow_super_source.tpo_node_gen():
    node_signatures.setdefault(ir.NodeSignature(node), []).append(node)

  for idx, node_signature in enumerate(node_signatures):
    printer.println()
    _print_node_definition(printer, node_signature, idx)


#  if stencil.cluster == 'none':
#    for forwarder in stencil.get_forwarders():
#      print_forward_func(printer, forwarder)
#      printer.println()
#
#    for forwarder in stencil.get_forwarders_with_border():
#      print_forward_func_with_border(printer, stencil, forwarder)
#      printer.println()
#    for stage in stencil.stages.values():
#      print_compute_stage(printer, stencil, stage)
#      printer.println()
#  elif stencil.cluster == 'fine':
#    print_compute_input(printer, stencil)
#    for stage in stencil.get_stages_chronologically():
#      print_compute_stage(printer, stencil, stage)
#
#  if stencil.replication_factor > 1:
#    dst_lists = set()
#    super_source = stencil.dataflow_super_source
#    def add_dst_lists(tensor):
#      rf = stencil.replication_factor
#      dst_ids = [start%rf for start, end
#        in stencil.get_replicated_reuse_buffers()[tensor.name][1:]
#        if start == end]
#      dst_ids = tuple(dst_id if dst_id in dst_ids else None
#        for dst_id in range(stencil.replication_factor))
#      dst_lists.add(dst_ids)
#    for node in super_source.tpo_node_generator():
#      if isinstance(node, SuperSourceNode):
#        add_dst_lists(stencil.input)
#      elif isinstance(node, ComputeNode):
#        if not node.stage.is_output():
#          add_dst_lists(node.stage.output)
#    for dst_list in dst_lists:
#      _print_reconnect_func(printer, dst_list)
#  printer.println()

  print_interface(printer, stencil)

def _print_node_definition(printer, node_signature, node_id):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  fifo_st_prefix = 'fifo_st_'
  fifo_ref_prefix = 'fifo_ref_'
  read_fifo_func = 'ReadFIFO'
  func_name = 'Node%s' % node_id
  func_lower_name = 'node_%s' % node_id

  def get_delays(obj, delays):
    if isinstance(obj, ir.DelayedRef):
      delays.append(obj)
    return obj
  delays = []
  for let in node_signature.lets:
    let.visit(get_delays, delays)
  for expr in node_signature.exprs:
    expr.visit(get_delays, delays)
  _logger.debug('delays: %s', delays)

  fifo_loads = tuple(
    '/* input*/ hls::stream<{_.c_type}>* {_.ld_name}'.format(**locals())
    for _ in node_signature.loads)
  fifo_stores = tuple(
    '/*output*/ hls::stream<%s>* %s%s' % (expr.c_type, fifo_st_prefix, idx)
    for idx, expr in enumerate(node_signature.exprs))
  printer.print_func('void {func_name}'.format(**locals()),
                     fifo_stores+fifo_loads, align=0)
  do_scope(func_lower_name)
  for delay in delays:
    println(delay.c_buf_decl)
    println(delay.c_ptr_decl)
  println('{func_lower_name}_epoch:'.format(**locals()), indent=0)
  println('for (bool enable = true; enable;)')
  do_scope('for {func_lower_name}_epoch'.format(**locals()))
  for fifo_in in node_signature.loads:
    println('{fifo_in.c_type} {fifo_in.ref_name};'.format(**locals()))
  println('if (%s)' % (' && '.join('!{_.ld_name}->empty()'.format(**locals())
                                   for _ in node_signature.loads)))
  do_scope('if not empty')
  for fifo_in in node_signature.loads:
    println('bool&& {fifo_in.ref_name}_enable = '
      'ReadData(&{fifo_in.ref_name}, {fifo_in.ld_name});'.format(**locals()))
  println('enable = %s;' % (
    ' && '.join('{_.ref_name}_enable'.format(**locals())
                for _ in node_signature.loads)))

  for delay in delays:
    println(delay.c_buf_load)
    println(delay.c_buf_store)
    println(delay.c_ptr_incr)

  for let in node_signature.lets:
    println(let.c_expr)
  for idx, expr in enumerate(node_signature.exprs):
    println('WriteData(%s%s, %s, true);' % (fifo_st_prefix, idx, expr.c_expr))
  un_scope()
  un_scope()
  for idx, expr in enumerate(node_signature.exprs):
    println('WriteData(%s%s, %s(0), false);' %
            (fifo_st_prefix, idx, expr.c_type))
  un_scope()
  _logger.debug('printing: %s', node_signature)

def print_data_struct(printer):
  println = printer.println
  println('template<typename T> struct Data')
  printer.do_scope()
  println('T data;')
  println('bool ctrl;')
  printer.un_scope()

def print_read_data(printer):
  println = printer.println
  println('template<typename T> inline bool ReadData'
          '(T* data, hls::stream<Data<T> >* from)')
  printer.do_scope()
  println('#pragma HLS inline', indent=0)
  println('Data<T>&& tmp = from->read();')
  println('#pragma HLS data_pack variable=tmp', indent=0)
  println('*data = tmp.data;')
  println('return tmp.ctrl;')
  printer.un_scope()

def print_write_data(printer):
  println = printer.println
  println('template<typename T> inline void WriteData'
          '(hls::stream<Data<T> >* to, const T& data, bool ctrl)')
  printer.do_scope()
  println('#pragma HLS inline', indent=0)
  println('Data<T> tmp;')
  println('#pragma HLS data_pack variable=tmp', indent=0)
  println('tmp.data = data;')
  println('tmp.ctrl = ctrl;')
  println('to->write(tmp);')
  printer.un_scope()

def _generate_code(printer, stencil):
  super_source = stencil.dataflow_super_source

  if stencil.replication_factor > 1:
    pragmas = []
    hls_streams = set()
    for chan in range(stencil.input.chan):
      for replica_id in range(stencil.replication_factor):
        var_type = stencil.input.type
        var_name = ('compute_%s_chan_%d_replica_%d' %
          (stencil.input.name, chan, replica_id))
        printer.println('hls::stream<%s> %s("%s");' %
          (var_type, var_name, var_name))
        pragmas.append((var_name, 2))
        hls_streams.add(var_name)
    for src_node, dst_node in super_source.bfs_edge_generator():
      _logger.debug('%s -> %s' % (repr(src_node), repr(dst_node)))
      if isinstance(dst_node, ForwardNode):
        for chan in range(dst_node.tensor.chan):
          if isinstance(src_node, ComputeNode):
            for replica_id in range(stencil.replication_factor):
              var_type = dst_node.tensor.type
              var_name = ('compute_%s_chan_%d_replica_%d' %
                (dst_node.tensor.name, chan, replica_id))
              if var_name in hls_streams:
                continue
              printer.println('hls::stream<%s> %s("%s");' %
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
            printer.println('hls::stream<%s> %s("%s");' %
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
            printer.println('hls::stream<%s> %s("%s");' %
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
            printer.println('hls::stream<%s> %s("%s");' %
              (var_type, var_name, var_name))
            pragmas.append((var_name, 2))
            hls_streams.add(var_name)
      else:
        raise SemanticError('unknown dataflow edge: %s -> %s' %
            (repr(src_node), repr(dst_node)))

    for pragma in pragmas:
      printer.println(
        '#pragma HLS stream variable=%s depth=%d' % pragma, 0)

    printer.println()
    printer.println('#pragma HLS dataflow', 0)

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
        printer.print_func(
          '%s<%s>' % (func_name, tensor.type), params, ';', align=0)

    for node in super_source.tpo_node_generator():
      _logger.debug('%s' % repr(node))
      if isinstance(node, SuperSourceNode):
        _print_load_call(printer, stencil)
        for chan in range(stencil.input.chan):
          for bank in range(stencil.dram_bank):
            printer.println('unpack_%s(' %
              util.get_soda_type(stencil.input.type))
            printer.do_indent()
            for replica_id in range(
                stencil.dram_bank-1-bank,
                stencil.replication_factor,
                stencil.dram_bank):
              printer.println('compute_%s_chan_%d_replica_%d,' %
                (stencil.input.name, chan, replica_id))
            printer.println('input_stream_chan_%d_bank_%d, '
              'coalesced_data_num);' % (chan, bank))
            printer.un_indent()
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
            printer.print_func('forward_%d<%s, %d>' % (output_num,
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
          printer.print_func('compute_%s<%d>' % (node.stage.name,
            replica_id), params, ';', align=0)
        if not node.stage.is_output():
          print_reconnect(node.stage.output)
      elif isinstance(node, SuperSinkNode):
        for chan in range(stencil.output.chan):
          for bank in range(stencil.dram_bank):
            printer.println('pack_%s(output_stream_chan_%d_'
              'bank_%d,' % (util.get_soda_type(stencil.output.type),
                chan, bank))
            printer.do_indent()
            for replica_id in range(
                stencil.dram_bank-1-bank,
                stencil.replication_factor,
                stencil.dram_bank):
              printer.println('compute_%s_chan_%d_replica_%d,'
                % (stencil.output.name, chan, replica_id))
            printer.println('coalesced_data_num);')
            printer.un_indent()
        _print_store_call(printer, stencil)
      else:
        raise SemanticError('unknown dataflow node: %s' % repr(node))

def _print_load_call(printer, stencil):
  for c in range(stencil.input.chan):
    for i in range(stencil.dram_bank):
      printer.println('load(input_stream_chan_%d_bank_%d, '
        'var_input_chan_%d_bank_%d, coalesced_data_num);' %
        ((c, i)*2))

def _print_store_call(printer, stencil):
  for c in range(stencil.output.chan):
    for i in range(stencil.dram_bank):
      printer.println('store(var_output_chan_%d_bank_%d, '
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
  printer.print_func('template<typename T> void %s' % func_name, params, align=0)
  printer.do_scope()
  for replica_id in range(rf):
    printer.println('T buf_%d = 0;' % replica_id)

  printer.println('uint32_t epoch = 0;')
  printer.println('%s:' % func_name, 0)
  printer.println('while(epoch < epoch_num)')
  printer.do_scope()
  printer.println('if(not (%s))' % ' or '.join(
    'src_replica_%d.empty()' % replica_id for replica_id
    in range(rf)))
  printer.do_scope()
  for replica_id in range(rf):
    printer.println('T val_%d = src_replica_%d.read();' %
      (replica_id, replica_id))

  for replica_id in range(rf):
    for dst_id in dst_list:
      if dst_id is not None:
        if replica_id-dst_id >= 0:
          printer.println('dst_%d_replica_%d << val_%d;' %
            (dst_id, replica_id, replica_id-dst_id))
        else:
          printer.println('dst_%d_replica_%d << buf_%d;' %
            (dst_id, replica_id, rf+replica_id-dst_id))
  for replica_id in range(rf):
    printer.println('buf_%d = val_%d;' % (replica_id, replica_id))
  printer.println('++epoch;')
  printer.un_scope()
  printer.un_scope()
  printer.un_scope()

