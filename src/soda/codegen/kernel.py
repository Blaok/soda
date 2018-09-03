from collections import OrderedDict
from functools import reduce
import copy
import logging
import math
import operator

from haoda import ir
from soda import core
from soda import grammar
from soda import util

_logger = logging.getLogger('__main__').getChild(__name__)

'''
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
'''

def _print_interface(printer, stencil):
  println = printer.println
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  tile_size = stencil.tile_size
  unroll_factor = stencil.unroll_factor
  app_name = stencil.app_name
  super_source = stencil.dataflow_super_source
  burst_width = stencil.burst_width

  _logger.info('generate reuse buffers')
  reuse_buffers = stencil.reuse_buffers
  all_points = stencil.all_points
  next_fifo = stencil.next_fifo
  overall_stencil_window = core.get_overall_stencil_window(
      stencil.tensors[stencil.input_names[0]],
      stencil.tensors[stencil.output_names[0]])

  outputs = [(stmt.name, bank) for stmt in stencil.output_stmts
                               for bank in stmt.dram]
  inputs = [(stmt.name, bank) for stmt in stencil.input_stmts
                              for bank in stmt.dram]

  get_port_name = util.get_port_name
  get_port_buf_name = util.get_port_buf_name
  get_fifo_name = util.get_fifo_name
  get_tensor_name_at = util.get_tensor_name_at

  println('extern "C"')
  println('{')
  println()
  println('void %s_kernel(' % app_name)
  do_indent()
  for name, bank in outputs + inputs:
    println('ap_uint<BURST_WIDTH>* {},'.format(get_port_name(name, bank)))
  for param in stencil.param_names:
    println('%s* var_%s,' % (param.type, param.name))
  println('uint64_t coalesced_data_num,')
  println('uint64_t tile_data_num,')
  for i in range(stencil.dim-1):
    println('uint32_t input_bound_dim_%d,' % i)
  for d in range(stencil.dim-1):
    println('uint32_t input_size_dim_%d,' % d)
  println('uint32_t input_size_dim_%d)' % (stencil.dim-1))
  un_indent()
  do_scope()

  for name, bank in outputs:
    println('#pragma HLS interface m_axi port={} offset=slave depth=65536 '
            'bundle={} latency=120'.format(get_port_name(name, bank),
                                           get_port_name(name, bank)), 0)
  for name, bank in inputs:
    println('#pragma HLS interface m_axi port={} offset=slave depth=65536 '
            'bundle={} latency=120'.format(get_port_name(name, bank),
                                           get_port_name(name, bank)), 0)

  for idx, param in enumerate(stencil.param_stmts):
    println('#pragma HLS interface m_axi port=var_{} offset=slave depth={} '
            'bundle=gmem{} latency=120'.format(
                param.name, reduce(operator.mul, param.size), idx), 0)

  println()
  for name, bank in outputs + inputs:
    println('#pragma HLS interface s_axilite port={} '
            'bundle=control'.format(get_port_name(name, bank)), 0)

  for param in stencil.param_stmts:
    println('#pragma HLS interface s_axilite port=var_{} '
            'bundle=control'.format(param.name), 0)
  println('#pragma HLS interface s_axilite port=coalesced_data_num '
          'bundle=control', 0)
  println('#pragma HLS interface s_axilite port=tile_data_num '
          'bundle=control', 0)
  for d in range(stencil.dim-1):
    println('#pragma HLS interface s_axilite port=input_bound_dim_%d '
            'bundle=control' % d, 0)
  for d in range(stencil.dim):
    println('#pragma HLS interface s_axilite port=input_size_dim_%d '
            'bundle=control' % d, 0)
  println('#pragma HLS interface s_axilite port=return bundle=control', 0)
  println()

  # port buf declarations
  for name, bank in inputs + outputs:
    println('hls::stream<Data<ap_uint<BURST_WIDTH>>> {0}("{0}");'.format(
        get_port_buf_name(name, bank)))
  # port buf depths
    println('#pragma HLS stream variable={} depth=32'.format(
        get_port_buf_name(name, bank)), 0)
    println('#pragma HLS data_pack variable={}'.format(
        get_port_buf_name(name, bank)), indent=0)
  println()

  # internal fifos
  for node in stencil.dataflow_super_source.tpo_node_gen():
    for fifo in node.fifos:
      println('hls::stream<Data<{0}>> {1}("{1}");'.format(fifo.c_type, fifo.c_expr))
      println('#pragma HLS stream variable={} depth={}'.format(
          fifo.c_expr, max(fifo.depth, 2)), 0)
      println('#pragma HLS data_pack variable={}'.format(fifo.c_expr),
              indent=0)

  '''
  if extra_params:
    for param in extra_params.values():
      if param.dup:
        dup = ('[%d]' % param.dup)
      else:
        dup = ''
      println('%s %s%s[UNROLL_FACTOR][%s];' % (param.type, param.name, dup, ']['.join(map(str, param.size))))
    println()

    for param in extra_params.values():
      println('#pragma HLS array_partition variable=%s complete dim=1' % param.name, 0)
      dim_offset = 1
      if param.dup:
        println('#pragma HLS array_partition variable=%s complete dim=2' % param.name, 0)
        dim_offset = 2
      for partitioning in param.partitioning:
        println('#pragma HLS array_partition variable=%s %s dim=%d%s' % (
          param.name,
          partitioning.partition_type,
          dim_offset+1 if partitioning.dim is None else partitioning.dim+dim_offset,
          '' if partitioning.factor is None else ' factor=%d' % partitioning.factor,
        ), 0)
    println()

    for param in extra_params.values():
      if len(param.size) > 1:
        for dim, size in enumerate(param.size):
          println('uint32_t %s_index_dim_%d = 0;' % (param.name, dim))
      println('%s_init:' % param.name, 0)
      println('for(int %s_index = 0; %s_index < %d; ++%s_index)' % (param.name, param.name, reduce(operator.mul, param.size), param.name))
      p.do_scope()
      println('#pragma HLS pipeline II=1', 0)
      println('%s& %s_tmp = var_%s[%s_index];' % (param.type, param.name, param.name, param.name))
      println('%s_unrolled:' % param.name, 0)
      println('for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)')
      p.do_scope()
      println('#pragma HLS unroll',0)
      if param.dup is None:
        println('%s[unroll_index]%s = %s_tmp;' % ((param.name, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name)))
      else:
        for i in range(param.dup):
          println('%s[%d][unroll_index]%s = %s_tmp;' % (param.name, i, ''.join(['[%s_index_dim_%d]' % (param.name, x) for x in range(len(param.size)-1, -1, -1)]) if len(param.size)>1 else '[%s_index]' % param.name, param.name))
      p.un_scope()
      if len(param.size) > 1:
        for dim in range(len(param.size)):
          println('++%s_index_dim_%d;' % (param.name, dim))
          if dim<len(param.size)-1:
            println('if(%s_index_dim_%d==%d)' % (param.name, dim, param.size[len(param.size)-1-dim]))
            p.do_scope()
            println('%s_index_dim_%d = 0;' % (param.name, dim))
      for size in param.size[:-1]:
        p.un_scope()
      p.un_scope()
    println()

  extra_params_str = ''.join([param.name+', ' for param in extra_params.values()])
  '''

  # TODO: fix this number
  println('uint64_t epoch_num = coalesced_data_num*{}/{};'.format(
      stencil.burst_width//sum(util.get_width_in_bits(_)
                               for _ in stencil.input_types),
      unroll_factor))
  println()

  # TODO: replication not supported
  '''
  if stencil.replication_factor > 1:
    _generate_code(p, stencil)

    p.un_scope()
    println()
    println('}  //extern "C"')

    return

  # reuse buffers
  if stencil.cluster == 'none':
    for name, reuse_buffer in reuse_buffers.items():
      pragmas = []
      msg = 'reuse buffers for %s' % name
      _logger.debug('generate %s', msg)
      println('// %s' % msg)
      for start, end in reuse_buffer[1:]:
        println('hls::stream<{0}> {1}("{1}");'.format(
            stencil.tensors[name].c_type, get_tensor_name_at(name, end)))
        buffer_length = stencil.reuse_buffer_lengths[name][end]
        tensor_name = get_tensor_name_at(name, end)
        if buffer_length > 1:
          pragmas.append((tensor_name, buffer_length))
      for pragma in pragmas:
        println('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
      println()
  else:
    println('// %s' % stencil.input.name)
    for unroll_index in range(unroll_factor):
      for c in range(stencil.input.chan):
        println('hls::stream<{0}> {1}("{1}");'.format(
            stencil.tensors[stencil.input.name].type,
            get_tensor_name_at(stencil.input.name, unroll_index, c)))
    println()

  for output_name in stencil.output_names:
    println('// %s' % output_name)
    for unroll_index in range(unroll_factor):
      println('hls::stream<{0}> {1}("{1}");'.format(
          stencil.tensors[output_name].c_type,
          get_tensor_name_at(output_name, unroll_index)))
  println()

  # params
  msg = 'params'
  _logger.debug('generate %s' % msg)
  println('// %s' % msg)
  pragmas = []
  for tensor in stencil.chronological_tensors:
    for pe_id in range(unroll_factor):
      for input_name, input_window in tensor.ld_indices.items():
        for param_id in range(len(input_window)):
          offset = next(offset for offset, points in
            all_points[input_name][tensor.name].items()
            if pe_id in points and points[pe_id] == param_id)
          fwd_node = super_source.fwd_nodes[(input_name, offset)]
          cpt_node = super_source.cpt_nodes[(tensor.name, pe_id)]
          extra_depth = super_source.get_extra_depth(
            (fwd_node, cpt_node))
          var_type = stencil.tensors[input_name].c_type
          var_name = 'from_%s_to_%s_param_%d_pe_%d' % (
            input_name, tensor.name, param_id, pe_id)
          println('hls::stream<{0}> {1}("{1}");'.format(var_type, var_name))
          if extra_depth > 0:
            pragmas.append((var_name, extra_depth+1))
          else:
            pragmas.append((var_name, 2))
  for pragma in pragmas:
    println('#pragma HLS stream variable=%s depth=%d' % pragma, 0)
  println()
  '''

  '''
  # border buffers
  msg = 'border buffers'
  _logger.debug('generate %s' % msg)
  println('// %s' % msg)
  for tensor in stencil.chronological_tensors:
    if tensor.output.preserve_border_to():
      for unroll_index in range(unroll_factor):
        for d in range(stencil.dim-1):
          for c in range(tensor.output.chan):
            param = (tensor.output.type, 'border_from_%s_dim_%d_left_chan_%d_pe_%d' % (tensor.name, d, c, unroll_index))
            println('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
            param = (tensor.output.type, 'border_from_%s_dim_%d_right_chan_%d_pe_%d' % (tensor.name, d, c, unroll_index))
            println('hls::stream<%s> %s("%s");' % (param[0], param[1], param[1]))
  println()
  '''

  println('#pragma HLS dataflow', 0)
  for name, bank in inputs:
    println('BurstRead(&{}, {}, coalesced_data_num);'.format(
        get_port_buf_name(name, bank), get_port_name(name, bank)))

  for node in super_source.tpo_node_gen():
    module_trait_id = stencil.module_table[node][1]
    _print_module_func_call(printer, node, module_trait_id)

  for name, bank in outputs:
    println('BurstWrite({}, &{});'.format(
        get_port_name(name, bank), get_port_buf_name(name, bank)))

  un_scope()
  println()
  println('}//extern "C"')

def print_header(printer):
  println = printer.println
  for header in ['float', 'math', 'stdbool', 'stddef', 'stdint', 'stdio',
                 'string', 'ap_int', 'hls_stream']:
    println('#include<%s.h>' % header)
  println()

def _print_burst_read(printer):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  println('void BurstRead(hls::stream<Data<ap_uint<BURST_WIDTH>>>* to, ap_uint<BURST_WIDTH>* from, uint64_t data_num)')
  do_scope()
  println('load_epoch:', 0)
  println('for (uint64_t epoch = 0; epoch < data_num;)')
  do_scope()
  println('#pragma HLS pipeline II=1', 0)
  println('const uint64_t next_epoch = epoch + 1;')
  println('WriteData(to, from[epoch], next_epoch < data_num);')
  println('epoch = next_epoch;')
  un_scope()
  un_scope()

def _print_burst_write(printer):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  println('void BurstWrite(ap_uint<BURST_WIDTH>* to, hls::stream<Data<ap_uint<BURST_WIDTH>>>* from)')
  do_scope()
  println('uint64_t epoch = 0;')
  println('store_epoch:', 0)
  println('for (bool enable = true; enable; ++epoch)')
  do_scope()
  println('#pragma HLS pipeline II=1', 0)
  println('ap_uint<BURST_WIDTH> buf;')
  println('enable = ReadData(&buf, from);')
  println('to[epoch] = buf;')
  un_scope()
  un_scope()

'''
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
'''

def print_code(stencil, output_file):
  _logger.info('generate kernel code as %s' % output_file.name)
  printer = util.Printer(output_file)

  print_header(printer)

  printer.println()

  util.print_define(printer, 'BURST_WIDTH', stencil.burst_width)
  printer.println()

  util.print_guard(printer, 'UNROLL_FACTOR', stencil.unroll_factor)
  for i in range(len(stencil.tile_size)-1):
    util.print_guard(printer, 'TILE_SIZE_DIM_%d' % i, stencil.tile_size[i])
  util.print_guard(printer, 'BURST_WIDTH', stencil.burst_width)
  printer.println()

  _print_data_struct(printer)
  _print_reinterpret(printer)
  _print_read_data(printer)
  _print_write_data(printer)

  _print_burst_read(printer)
  _print_burst_write(printer)

  for module_trait_id, module_trait in enumerate(stencil.module_traits):
    _print_module_definition(printer, module_trait, module_trait_id,
                           burst_width=stencil.burst_width)


  '''
  if stencil.cluster == 'none':
    for forwarder in stencil.get_forwarders():
      print_forward_func(printer, forwarder)
      printer.println()

    for forwarder in stencil.get_forwarders_with_border():
      print_forward_func_with_border(printer, stencil, forwarder)
      printer.println()
    for stage in stencil.stages.values():
      print_compute_stage(printer, stencil, stage)
      printer.println()
  elif stencil.cluster == 'fine':
    print_compute_input(printer, stencil)
    for stage in stencil.get_stages_chronologically():
      print_compute_stage(printer, stencil, stage)

  if stencil.replication_factor > 1:
    dst_lists = set()
    super_source = stencil.dataflow_super_source
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
        if not node.stage.is_output():
          add_dst_lists(node.stage.output)
    for dst_list in dst_lists:
      _print_reconnect_func(printer, dst_list)
  printer.println()
  '''

  _print_interface(printer, stencil)

def _print_module_func_call(printer, node, module_trait_id, **kwargs):
  println = printer.println
  print_func = printer.print_func
  func_name = util.get_func_name(module_trait_id)
  func_lower_name = util.get_module_name(module_trait_id)

  def get_load_set(obj, loads):
    if isinstance(obj, ir.FIFO):
      loads[obj] = None
    return obj
  loads = OrderedDict()
  node.visit_loads(get_load_set, loads)
  loads = tuple(loads)

  # find dram reads
  reads_in_lets = tuple(_.expr for _ in node.lets)
  reads_in_exprs = tuple(node.exprs.values())
  dram_reads = OrderedDict()
  for dram_ref in core.get_dram_refs(reads_in_lets + reads_in_exprs):
    for bank in dram_ref.dram:
      dram_reads[util.get_port_buf_name(dram_ref.var, bank)] = None
  dram_reads = tuple('/* input*/ &' + _ for _ in dram_reads)

  # find dram writes
  writes_in_lets = tuple(_.name for _ in node.lets
                         if not isinstance(_.name, str))
  dram_writes = OrderedDict()
  for dram_ref in core.get_dram_refs(writes_in_lets):
    for bank in dram_ref.dram:
      dram_writes[util.get_port_buf_name(dram_ref.var, bank)] = None
  dram_writes = tuple('/*output*/ &' + _ for _ in dram_writes)

  output_fifos = tuple('/*output*/ &' + _.c_expr for _ in node.exprs)
  input_fifos = tuple('/* input*/ &' + _.c_expr for _ in loads)
  params = dram_writes + output_fifos + input_fifos + dram_reads

  print_func(func_name, params, suffix=';', align=0)

def _print_module_definition(printer, module_trait, module_trait_id, **kwargs):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  fifo_st_prefix = 'fifo_st_'
  fifo_ref_prefix = 'fifo_ref_'
  read_fifo_func = 'ReadFIFO'
  func_name = util.get_func_name(module_trait_id)
  func_lower_name = util.get_module_name(module_trait_id)
  ii = 1

  def get_delays(obj, delays):
    if isinstance(obj, ir.DelayedRef):
      delays.append(obj)
    return obj
  delays = []
  for let in module_trait.lets:
    let.visit(get_delays, delays)
  for expr in module_trait.exprs:
    expr.visit(get_delays, delays)
  _logger.debug('delays: %s', delays)

  fifo_loads = tuple('/* input*/ hls::stream<Data<{}>>* {}'.format(
      _.c_type, _.ld_name) for _ in module_trait.loads)
  fifo_stores = tuple('/*output*/ hls::stream<Data<{}>>* {}{}'.format(
      expr.c_type, fifo_st_prefix, idx)
    for idx, expr in enumerate(module_trait.exprs))

  # look for DRAM access
  reads_in_lets = tuple(_.expr for _ in module_trait.lets)
  writes_in_lets = tuple(_.name for _ in module_trait.lets
                         if not isinstance(_.name, str))
  reads_in_exprs = module_trait.exprs
  dram_reads = core.get_dram_refs(reads_in_lets + reads_in_exprs)
  dram_writes = core.get_dram_refs(writes_in_lets)
  drams = ()
  dram_read_map = OrderedDict()
  dram_write_map = OrderedDict()
  if dram_reads:  # this is an unpacking module
    assert not dram_writes, 'cannot read and write DRAM in the same module'
    for dram_read in dram_reads:
      dram_read_map.setdefault(dram_read.dram, []).append(dram_read)
    burst_width = kwargs.pop('burst_width')
    for dram in dram_read_map:
      batch_size = len(dram_read_map[dram])   # number of elements per cycle
      dram_read_map[dram] = OrderedDict((_.offset, _)
                                        for _ in dram_read_map[dram])
      dram_reads = dram_read_map[dram]
      assert tuple(sorted(dram_reads.keys())) == tuple(range(batch_size)), \
             'unexpected DRAM accesses pattern %s' % dram_reads
      batch_width = sum(util.get_width_in_bits(_.soda_type)
                        for _ in dram_reads.values())
      del dram_reads
      if burst_width > batch_width:
        assert burst_width % batch_width == 0, 'cannot process such a burst'
        # a single burst consumed in multiple cycles
        coalescing_factor = burst_width // batch_width
        ii = coalescing_factor
      else:
        assert batch_width % burst_width == 0, 'cannot process such a burst'
        # multiple bursts consumed in a single cycle
        reassemble_factor = batch_width // burst_width
    dram_reads = tuple(next(iter(_.values())) for _ in dram_read_map.values())
    fifo_loads += tuple(
        '/* input*/ hls::stream<Data<ap_uint<{burst_width}>>>* '
        '{dram.dram_fifo_name}'.format(
            burst_width=burst_width, dram=_) for _ in dram_reads)
  elif dram_writes:   # this is a packing module
    for dram_write in dram_writes:
      dram_write_map.setdefault(dram_write.dram, []).append(dram_write)
    burst_width = kwargs.pop('burst_width')
    for dram in dram_write_map:
      batch_size = len(dram_write_map[dram])  # number of elements per cycle
      dram_write_map[dram] = OrderedDict((_.offset, _)
                                         for _ in dram_write_map[dram])
      dram_writes = dram_write_map[dram]
      assert tuple(sorted(dram_writes.keys())) == tuple(range(batch_size)), \
             'unexpected DRAM accesses pattern %s' % dram_writes
      batch_width = sum(util.get_width_in_bits(_.soda_type)
                        for _ in dram_writes.values())
      del dram_writes
      if burst_width > batch_width:
        assert burst_width % batch_width == 0, 'cannot process such a burst'
        # a single burst consumed in multiple cycles
        coalescing_factor = burst_width // batch_width
        ii = coalescing_factor
      else:
        assert batch_width % burst_width == 0, 'cannot process such a burst'
        # multiple bursts consumed in a single cycle
        reassemble_factor = batch_width // burst_width
    dram_writes = tuple(next(iter(_.values()))
                        for _ in dram_write_map.values())
    fifo_stores += tuple(
        '/*output*/ hls::stream<Data<ap_uint<{burst_width}>>>* '
        '{dram.dram_fifo_name}'.format(
            burst_width=burst_width, dram=_) for _ in dram_writes)

  # print function
  printer.print_func('void {func_name}'.format(**locals()),
                     fifo_stores+fifo_loads, align=0)
  do_scope(func_name)

  # print inter-iteration declarations
  for delay in delays:
    println(delay.c_buf_decl)
    println(delay.c_ptr_decl)

  # print loop
  println('{func_lower_name}_epoch:'.format(**locals()), indent=0)
  println('for (bool enable = true; enable;)')
  do_scope('for {func_lower_name}_epoch'.format(**locals()))
  println('#pragma HLS pipeline II=%d' % ii, 0)
  for delay in delays:
    println('#pragma HLS dependence variable=%s inter false' %
            delay.buf_name, 0)

  # print emptyness tests
  println('if (%s)' % (' && '.join(
      '!{fifo}->empty()'.format(fifo=_)
      for _ in tuple(_.ld_name for _ in module_trait.loads) +
               tuple(_.dram_fifo_name for _ in dram_reads))))
  do_scope('if not empty')

  # print intra-iteration declarations
  for fifo_in in module_trait.loads:
    println('{fifo_in.c_type} {fifo_in.ref_name};'.format(**locals()))
  for dram in (next(iter(_.values())) for _ in dram_read_map.values()):
    println('ap_uint<{burst_width}> {dram.dram_buf_name};'.format(
        burst_width=burst_width, dram=dram))
  for dram in (next(iter(_.values())) for _ in dram_write_map.values()):
    println('ap_uint<{burst_width}> {dram.dram_buf_name};'.format(
        burst_width=burst_width, dram=dram))

  # print enable conditions
  if not dram_write_map:
    for fifo_in in module_trait.loads:
      println('const bool {fifo_in.ref_name}_enable = '
        'ReadData(&{fifo_in.ref_name}, {fifo_in.ld_name});'.format(**locals()))
  for dram in dram_reads:
    println('const bool {dram.dram_buf_name}_enable = '
            'ReadData(&{dram.dram_buf_name}, {dram.dram_fifo_name});'.format(
                dram=dram))
  if not dram_write_map:
    println('const bool enabled = %s;' % (
      ' && '.join(tuple('{_.ref_name}_enable'.format(_=_)
                        for _ in module_trait.loads) +
                  tuple('{_.dram_buf_name}_enable'.format(_=_)
                        for _ in dram_reads))))
    println('enable = enabled;')

  # print delays (if any)
  for delay in delays:
    println('const {} {};'.format(delay.c_type, delay.c_buf_load))

  # print lets
  def mutate_dram_ref_for_writes(obj, kwargs):
    if isinstance(obj, ir.DRAMRef):
      coalescing_idx = kwargs.pop('coalescing_idx')
      unroll_factor = kwargs.pop('unroll_factor')
      type_width = util.get_width_in_bits(obj.soda_type)
      lsb = (coalescing_idx * unroll_factor + obj.offset) * type_width
      msb = lsb + type_width - 1
      return grammar.Var(name='{dram.dram_buf_name}({msb}, {lsb})'.format(
          dram=obj, msb=msb, lsb=lsb), idx=())
    return obj

  # mutate dram ref for writes
  if dram_write_map:
    for coalescing_idx in range(coalescing_factor):
      for fifo_in in module_trait.loads:
        if coalescing_idx == coalescing_factor - 1:
          prefix = 'const bool {fifo_in.ref_name}_enable = '.format(
              fifo_in=fifo_in)
        else:
          prefix = ''
        println('{prefix}ReadData(&{fifo_in.ref_name},'
                ' {fifo_in.ld_name});'.format(fifo_in=fifo_in, prefix=prefix))
      if coalescing_idx == coalescing_factor - 1:
        println('const bool enabled = %s;' % (
          ' && '.join(tuple('{_.ref_name}_enable'.format(_=_)
                            for _ in module_trait.loads) +
                      tuple('{_.dram_buf_name}_enable'.format(_=_)
                            for _ in dram_reads))))
        println('enable = enabled;')
      for idx, let in enumerate(module_trait.lets):
        let = let.visit(mutate_dram_ref_for_writes,
                        {'coalescing_idx': coalescing_idx,
                         'unroll_factor': len(module_trait.lets)})
        println('{} = Reinterpret<ap_uint<{width}>>({});'.format(
            let.name, let.expr.c_expr,
            width=util.get_width_in_bits(let.expr.soda_type)))
    for dram in map(lambda _: next(iter(_.values())), dram_write_map.values()):
      println('WriteData({}, {}, enabled);'.format(
          dram.dram_fifo_name, dram.dram_buf_name))
  else:
    for let in module_trait.lets:
      println(let.c_expr)

  def mutate_dram_ref_for_reads(obj, kwargs):
    if isinstance(obj, ir.DRAMRef):
      coalescing_idx = kwargs.pop('coalescing_idx')
      unroll_factor = kwargs.pop('unroll_factor')
      type_width = util.get_width_in_bits(obj.soda_type)
      lsb = (coalescing_idx * unroll_factor + obj.offset) * type_width
      msb = lsb + type_width - 1
      return grammar.Var(
          name='Reinterpret<{c_type}>(static_cast<ap_uint<{width}>>('
               '{dram.dram_buf_name}({msb}, {lsb})))'.format(
                   c_type=obj.c_type, dram=obj, msb=msb, lsb=lsb,
                   width=msb-lsb+1), idx=())
    return obj

  # mutate dram ref for reads
  if dram_read_map:
    for coalescing_idx in range(coalescing_factor):
      for idx, expr in enumerate(module_trait.exprs):
        println('WriteData({}{}, {}, {});'.format(
            fifo_st_prefix, idx,
            expr.visit(mutate_dram_ref_for_reads,
                       {'coalescing_idx': coalescing_idx,
                        'unroll_factor': len(module_trait.exprs)}).c_expr,
            'true' if coalescing_idx < coalescing_factor - 1 else 'enabled'))
  else:
    for idx, expr in enumerate(module_trait.exprs):
      println('WriteData({}{}, {}({}), enabled);'.format(
              fifo_st_prefix, idx, expr.c_type, expr.c_expr))

  for delay in delays:
    println(delay.c_buf_store)
    println('{} = {};'.format(delay.ptr, delay.c_next_ptr_expr))

  un_scope()
  un_scope()
  un_scope()
  _logger.debug('printing: %s', module_trait)

def _print_data_struct(printer):
  println = printer.println
  println('template<typename T> struct Data')
  printer.do_scope()
  println('T data;')
  println('bool ctrl;')
  printer.un_scope(suffix=';')

def _print_reinterpret(printer):
  println = printer.println
  println('template<typename To, typename From>')
  println('inline To Reinterpret(const From& val)')
  printer.do_scope()
  println('return reinterpret_cast<const To&>(val);')
  printer.un_scope()

def _print_read_data(printer):
  println = printer.println
  println('template<typename T> inline bool ReadData'
          '(T* data, hls::stream<Data<T>>* from)')
  printer.do_scope()
  println('#pragma HLS inline', indent=0)
  println('const Data<T>& tmp = from->read();')
  println('*data = tmp.data;')
  println('return tmp.ctrl;')
  printer.un_scope()

def _print_write_data(printer):
  println = printer.println
  println('template<typename T> inline void WriteData'
          '(hls::stream<Data<T>>* to, const T& data, bool ctrl)')
  printer.do_scope()
  println('#pragma HLS inline', indent=0)
  println('Data<T> tmp;')
  println('tmp.data = data;')
  println('tmp.ctrl = ctrl;')
  println('to->write(tmp);')
  printer.un_scope()

'''
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
'''
