import functools
import logging
import operator
from typing import List, TextIO

import soda.tensor
from haoda import ir, util
from soda import core, grammar
from soda import util as soda_util

logger = logging.getLogger().getChild(__name__)

STENCIL_DIM_FMT = util.MetaFmt('kStencilDim%s')
TYPE_FMT = util.MetaFmt('Type_%s')
WIDTH_FMT = util.MetaFmt('kWidth_%s')


def print_header(printer):
  # C headers
  for header in ('assert', 'float', 'math', 'stdbool', 'stddef', 'stdint',
                 'stdio', 'stdlib', 'string'):
    printer.println(f'#include <c{header}>')
  printer.println()

  # C++ headers
  for header in ('algorithm', 'iomanip', 'iostream', 'memory', 'random',
                 'regex', 'string', 'unordered_map', 'vector'):
    printer.println(f'#include <{header}>')
  printer.println()

  # Other system headers
  for header in ('frt',):
    printer.println(f'#include <{header}.h>')
  printer.println()

  # using declarations
  for name in ('clog', 'endl', 'regex', 'regex_match', 'string', 'unique_ptr',
               'unordered_map', 'vector'):
    printer.println(f'using std::{name};')
  printer.println()


def print_func(printer: util.CppPrinter, stencil: soda.core.Stencil):
  stmts = stencil.input_stmts + stencil.output_stmts

  # factories for meta variables
  data_fmt = util.MetaFmt('var_%s_ptr')
  extent_fmt = util.MetaFmt('var_%s_extent')
  stride_fmt = util.MetaFmt('var_%s_stride')
  min_fmt = util.MetaFmt('var_%s_min')

  # print function signature
  params: List[str] = []
  for stmt in stmts + stencil.param_stmts:
    prefix = 'const ' if isinstance(stmt, grammar.InputStmt) else ''
    params.extend((f'{prefix}{TYPE_FMT[stmt.name]}* {data_fmt[stmt.name]}',
                   f'const int32_t {extent_fmt[stmt.name]}[{stencil.dim}]',
                   f'const int32_t {stride_fmt[stmt.name]}[{stencil.dim}]',
                   f'const int32_t {min_fmt[stmt.name]}[{stencil.dim}]'))

  tile_size_fmt = util.MetaFmt('tile_size_%s')
  params.extend((
      'const char* bitstream',
      f'const int burst_width = {stencil.burst_width}',
      *(f'const int {tile_size_fmt[d]} = {stencil.tile_size[d]}'
        for d in range(stencil.dim - 1)),
      f'const int unroll_factor = {stencil.unroll_factor}',
  ))
  printer.print_func(name=f'int {stencil.app_name}', params=params, align=0)
  printer.do_scope()

  printer.printlns(
      '// load bitstream',
      'auto instance = fpga::Instance(bitstream);',
      'auto args_info = instance.GetArgsInfo();'
      '',
  )

  bank_count_fmt = util.MetaFmt('bank_count_%s')
  regex_fmt = util.MetaFmt('regex_%s')
  elem_count_per_cycle_fmt = util.MetaFmt('elem_count_per_cycle_%s')
  tile_count_fmt = util.MetaFmt('tile_count_dim_%d')
  printer.printlns(
      '// find out how many banks are used for each tensor',
      *(f'int {bank_count_fmt[x.name]} = 0;' for x in stmts),
      *(f'const regex {regex_fmt[x.name]}'
        f'(R"(^bank_\\d+_{x.name}$)");' for x in stmts),
  )
  with printer.for_('const auto& arg', 'args_info'):
    printer.printlns(f'if (regex_match(arg.name, {regex_fmt[x.name]})) '
                     f'++{bank_count_fmt[x.name]};' for x in stmts)
  printer.printlns(
      '',
      ('auto round_up = [](int64_t a, int64_t b) -> int64_t '
       '{ return ((a - 1) / b + 1) * b; };'),
      '',
      '// some run-time constants',
      *(f'const int {elem_count_per_cycle_fmt[x.name]} = '
        f'burst_width / {WIDTH_FMT[x.name]} * {bank_count_fmt[x.name]};'
        for x in stmts),
  )
  for d in range(stencil.dim - 1):
    printer.println(f'int32_t {tile_count_fmt[d]} = '
                    f'({extent_fmt[stencil.input_names[0]]}[{d}] - '
                    f'{STENCIL_DIM_FMT[d]} + 1 - 1) / ({tile_size_fmt[d]} - '
                    f'{STENCIL_DIM_FMT[d]} + 1) + 1;')
  printer.printlns(
      ('int64_t tile_count = %s;' %
       ' * '.join(f'{tile_count_fmt[d]}' for d in range(stencil.dim - 1))),
      '',
  )

  printer.printlns(
      '// align each linearized tile to multiples of burst_width',
      ('int64_t elem_count_per_tile = %s * '
       f'{extent_fmt[stencil.input_names[0]]}[{stencil.dim - 1}];' %
       ' * '.join(f'{tile_size_fmt[d]}' for d in range(stencil.dim - 1))),
      ('int64_t cycle_count_per_tile = (elem_count_per_tile - 1) / '
       f'{elem_count_per_cycle_fmt[stencil.input_names[0]]} + 1;'),
      ('int64_t elem_count_aligned_per_tile_i = cycle_count_per_tile * '
       f'{elem_count_per_cycle_fmt[stencil.input_stmts[0].name]};'),
      ('int64_t elem_count_aligned_per_tile_o = cycle_count_per_tile * '
       f'{elem_count_per_cycle_fmt[stencil.output_stmts[0].name]};'),
      '',
  )

  printer.println('// calculate size of each buffer')
  buf_size_fmt = util.MetaFmt('buf_size_%s')
  for stmt in stencil.input_stmts:
    printer.println(
        f'int64_t {buf_size_fmt[stmt.name]} = '
        f'(tile_count * elem_count_aligned_per_tile_i + '
        f'round_up(kStencilDistance, {elem_count_per_cycle_fmt[stmt.name]}))'
        f' / {bank_count_fmt[stmt.name]} * sizeof({TYPE_FMT[stmt.name]});')
  for stmt in stencil.output_stmts:
    printer.println(
        f'int64_t {buf_size_fmt[stmt.name]} = '
        f'(tile_count * elem_count_aligned_per_tile_o + '
        f'round_up(kStencilDistance, {elem_count_per_cycle_fmt[stmt.name]}))'
        f' / {bank_count_fmt[stmt.name]} * sizeof({TYPE_FMT[stmt.name]});')
  printer.println()

  printer.println('// allocate memory for each buffer')
  buf_fmt = util.MetaFmt('buf_%s')
  for stmt in stmts:
    printer.printlns(
        (f'vector<unique_ptr<{TYPE_FMT[stmt.name]}, decltype(&free)>> '
         f'{buf_fmt[stmt.name]};'),
        f'{buf_fmt[stmt.name]}.reserve({bank_count_fmt[stmt.name]});',
    )
    with printer.for_('int bank = 0', f'bank < {bank_count_fmt[stmt.name]}',
                      '++bank'):
      printer.println(
          f'{buf_fmt[stmt.name]}.emplace_back('
          f'static_cast<{TYPE_FMT[stmt.name]}*>(aligned_alloc('
          f'4096, round_up({buf_size_fmt[stmt.name]}, 4096))), &free);')
  printer.println()

  printer.println('// tiling')
  for dim in range(stencil.dim - 2, -1, -1):
    printer.println(f'for(int32_t tile_index_dim_{dim} = 0; '
                    f'tile_index_dim_{dim} < {tile_count_fmt[dim]}; '
                    f'++tile_index_dim_{dim})')
    printer.do_scope()
    printer.println(f'int32_t actual_tile_size_dim_{dim} = '
                    f'(tile_index_dim_{dim}=={tile_count_fmt[dim]}-1) ? '
                    f'{extent_fmt[stencil.input_names[0]]}[{dim}] - '
                    f'({tile_size_fmt[dim]} - {STENCIL_DIM_FMT[dim]} + 1) * '
                    f'tile_index_dim_{dim} : {tile_size_fmt[dim]};')

  printer.println('#pragma omp parallel for', 0)
  var = soda_util.COORDS_IN_TILE[stencil.dim - 1]
  printer.println(
      f'for(int32_t {var} = 0; '
      f'{var} < {extent_fmt[stencil.input_names[0]]}[{stencil.dim - 1}]; '
      f'++{var})')
  printer.do_scope()
  for dim in range(stencil.dim - 2, -1, -1):
    printer.println('for(int32_t {0} = 0; {0} < actual_tile_size_dim_{1}; '
                    '++{0})'.format(soda_util.COORDS_IN_TILE[dim], dim))
    printer.do_scope()

  printer.printlns(
      ('// (%s) is coordinates in tiled image' %
       ', '.join(soda_util.COORDS_TILED)),
      ('// (%s) is coordinates in original image' %
       ', '.join(soda_util.COORDS_IN_ORIG)),
      '// (%s) is coordinates in a tile' % ', '.join(soda_util.COORDS_IN_TILE),
  )
  offset_in_tile = ' + '.join(
      '%s%s' % (soda_util.COORDS_IN_TILE[x], ''.join(f' * {tile_size_fmt[d]}'
                                                     for d in range(x)))
      for x in range(stencil.dim))
  stmt = stencil.input_stmts[0]
  printer.printlns(
      (f'int32_t burst_index = ({offset_in_tile}) / '
       f'{elem_count_per_cycle_fmt[stmt.name]};'),
      (f'int32_t burst_residue = ({offset_in_tile}) % '
       f'{elem_count_per_cycle_fmt[stmt.name]};'),
  )
  for dim in range(stencil.dim - 1):
    printer.println(
        f'int32_t {soda_util.COORDS_IN_ORIG[dim]} = tile_index_dim_{dim} * '
        f'({tile_size_fmt[dim]} - {STENCIL_DIM_FMT[dim]}) + '
        f'{soda_util.COORDS_IN_TILE[dim]};')
  printer.printlns(
      ('int32_t %s = %s;' % (soda_util.COORDS_IN_ORIG[stencil.dim - 1],
                             soda_util.COORDS_IN_TILE[stencil.dim - 1])),
      (f'int64_t tiled_offset = (%s) * elem_count_aligned_per_tile_i + '
       f'burst_index * {elem_count_per_cycle_fmt[stmt.name]} + burst_residue;' %
       ' + '.join('%stile_index_dim_%d' % (''.join(f'{tile_count_fmt[d]} * '
                                                   for d in range(x)), x)
                  for x in range(stencil.dim - 1))),
      ('int64_t original_offset = %s;' %
       ' + '.join(f'%s * {stride_fmt[stencil.input_names[0]]}[%d]' %
                  (soda_util.COORDS_IN_ORIG[x], x)
                  for x in range(stencil.dim))),
  )
  printer.printlns(f'{buf_fmt[x]}'
                   f'[tiled_offset % {bank_count_fmt[x]}].get()'
                   f'[tiled_offset / {bank_count_fmt[x]}] = '
                   f'{data_fmt[x]}[std::max(int64_t(0), original_offset - '
                   f'{stencil.tensors[x].produce_offset})];'
                   for x in stencil.input_names)
  for dim in range(stencil.dim * 2 - 1):
    printer.un_scope()
  printer.println()

  for d in range(stencil.dim - 1):
    printer.println(
        f'clog << "INFO: tile_count[{d}] = " << {tile_count_fmt[d]} '
        f'<< ", tile_size[{d}] = " << {tile_size_fmt[d]} << endl;')
  for name in stencil.input_names + stencil.output_names:
    for item in 'extent', 'stride', 'min':
      fmt = locals()[item + '_fmt']
      printer.println(
          'clog << "INFO: %s" << endl;' %
          ', '.join(f'{name}.{item}[{d}] = " << {fmt[name]}[{d}] << "'
                    for d in range(stencil.dim)))
  printer.println()

  stmt = stencil.input_stmts[0]
  printer.printlns(
      ('int64_t tile_data_count = '
       f'((int64_t({extent_fmt[stmt.name]}[{stencil.dim - 1}])%s - 1) / '
       f'{elem_count_per_cycle_fmt[stmt.name]} + 1) * '
       f'{elem_count_per_cycle_fmt[stmt.name]} / '
       'unroll_factor;' %
       (''.join(f' * {tile_size_fmt[d]}' for d in range(stencil.dim - 1)))),
      ('int64_t cycle_count = '
       f'((int64_t({extent_fmt[stmt.name]}[{stencil.dim - 1}])%s * %s + '
       f'kStencilDistance - 1) / {elem_count_per_cycle_fmt[stmt.name]} + 1);' %
       (''.join(f' * {tile_size_fmt[d]}' for d in range(stencil.dim - 1)),
        ' * '.join(tile_count_fmt[d] for d in range(stencil.dim - 1)))),
      ('clog << "INFO: tile_data_count = " << tile_data_count '
       '<< ", cycle_count = " << cycle_count << endl;'),
      '',
  )

  printer.println('int arg_idx = 0;')
  iter_fmt = util.MetaFmt('iter_%s')
  for stmt in stmts:
    printer.println(
        f'auto {iter_fmt[stmt.name]} = {buf_fmt[stmt.name]}.begin();')
  with printer.for_('const auto& arg', 'args_info'):
    with printer.if_('arg.name == "coalesced_data_num"'):
      printer.printlns(
          'instance.SetArg(arg_idx, cycle_count);',
          '++arg_idx;',
      )
      for stmt in stmts:
        direction = 'Write' if isinstance(stmt, grammar.InputStmt) else 'Read'
        with printer.elif_(f'regex_match(arg.name, {regex_fmt[stmt.name]})'):
          printer.printlns(
              (f'auto buf = fpga::{direction}Only('
               f'{iter_fmt[stmt.name]}->get(), '
               f'{buf_size_fmt[stmt.name]} / sizeof({TYPE_FMT[stmt.name]}));'),
              'instance.AllocBuf(arg_idx, buf);',
              'instance.SetArg(arg_idx, buf);',
              f'++{iter_fmt[stmt.name]};',
              '++arg_idx;',
          )

  printer.printlns(
      '',
      'instance.WriteToDevice();',
      'instance.Exec();',
      'instance.ReadFromDevice();',
      'instance.Finish();',
      '',
      ('clog << "Load throughput: " << std::setprecision(3) '
       '<< instance.LoadThroughputGbps() << " GB/s" << endl;'),
      ('clog << "Compute latency: " << std::setprecision(3) '
       '<< instance.ComputeTimeSeconds() << " s" << endl;'),
      ('clog << "Store throughput: " << std::setprecision(3) '
       '<< instance.StoreThroughputGbps() <<" GB/s" << endl;'),
      '',
  )

  for dim in range(stencil.dim - 2, -1, -1):
    printer.println(
        f'for(int32_t tile_index_dim_{dim} = 0; tile_index_dim_{dim} < '
        f'{tile_count_fmt[dim]}; ++tile_index_dim_{dim})')
    printer.do_scope()
    printer.println(f'int32_t actual_tile_size_dim_{dim} = '
                    f'(tile_index_dim_{dim} == {tile_count_fmt[dim]}-1) ? '
                    f'{extent_fmt[stencil.input_names[0]]}[{dim}] - '
                    f'({tile_size_fmt[dim]} - {STENCIL_DIM_FMT[dim]} + 1)'
                    f' * tile_index_dim_{dim} : {tile_size_fmt[dim]};')

  overall_stencil_window = core.get_overall_stencil_window(
      stencil.tensors[stencil.input_names[0]],
      stencil.tensors[stencil.output_names[0]])
  overall_stencil_offset = core.get_stencil_window_offset(
      overall_stencil_window)
  overall_stencil_dim = core.get_stencil_dim(overall_stencil_window)
  printer.println('#pragma omp parallel for', 0)
  printer.println('for(int32_t {var} = {}; {var} < '
                  f'{extent_fmt[stencil.output_names[0]]}[{stencil.dim - 1}]'
                  ' - {}; ++{var})'.format(
                      max(0, overall_stencil_offset[stencil.dim - 1]),
                      max(0, (overall_stencil_dim[stencil.dim - 1] - 1 -
                              overall_stencil_offset[stencil.dim - 1])),
                      var=soda_util.COORDS_IN_TILE[stencil.dim - 1]))
  printer.do_scope()
  for dim in range(stencil.dim - 2, -1, -1):
    printer.println(
        'for(int32_t {var} = {}; {var} < actual_tile_size_dim_{} - {}; '
        '++{var})'.format(
            max(0, overall_stencil_offset[dim]),
            dim,
            max(0, overall_stencil_dim[dim] - 1 - overall_stencil_offset[dim]),
            var=soda_util.COORDS_IN_TILE[dim]))
    printer.do_scope()

  printer.printlns(
      ('// (%s) is coordinates in tiled image' %
       ', '.join(soda_util.COORDS_TILED)),
      ('// (%s) is coordinates in original image' %
       ', '.join(soda_util.COORDS_IN_ORIG)),
      '// (%s) is coordinates in a tile' % ', '.join(soda_util.COORDS_IN_TILE),
  )
  offset_in_tile = ' + '.join(
      '%s%s' % (soda_util.COORDS_IN_TILE[x], ''.join(f' * {tile_size_fmt[d]}'
                                                     for d in range(x)))
      for x in range(stencil.dim))
  for dim in range(stencil.dim - 1):
    printer.println(
        f'int32_t {soda_util.COORDS_IN_ORIG[dim]} = tile_index_dim_{dim} '
        f'* ({tile_size_fmt[dim]}-{STENCIL_DIM_FMT[dim]} + 1) + '
        f'{soda_util.COORDS_IN_TILE[dim]};')
  printer.printlns(
      ('int32_t %s = %s;' % (soda_util.COORDS_IN_ORIG[stencil.dim - 1],
                             soda_util.COORDS_IN_TILE[stencil.dim - 1])),
      ('int64_t original_offset = %s;' %
       ' + '.join(f'%s * {stride_fmt[stencil.output_names[0]]}[%d]' %
                  (soda_util.COORDS_IN_ORIG[x], x)
                  for x in range(stencil.dim))),
  )
  for stmt in stencil.output_stmts:
    overall_stencil_window = core.get_overall_stencil_window(
        map(stencil.tensors.get, stencil.input_names),
        stencil.tensors[stmt.name])
    overall_stencil_distance = core.get_stencil_distance(
        overall_stencil_window, stencil.tile_size)
    stencil_offset = overall_stencil_distance - soda_util.serialize(
        core.get_stencil_window_offset(overall_stencil_window),
        stencil.tile_size)
    printer.printlns(
        (f'int32_t burst_index_{stmt.name} = '
         f'({offset_in_tile} + {stencil_offset}) / '
         f'{elem_count_per_cycle_fmt[stmt.name]};'),
        (f'int32_t burst_residue_{stmt.name} = '
         f'({offset_in_tile} + {stencil_offset}) % '
         f'{elem_count_per_cycle_fmt[stmt.name]};'),
        (f'int64_t tiled_offset_{stmt.name} = '
         f'(%s) * elem_count_aligned_per_tile_o + burst_index_{stmt.name} * '
         f'{elem_count_per_cycle_fmt[stmt.name]} + burst_residue_{stmt.name};' %
         ('+'.join('%stile_index_dim_%d' % (''.join(f'{tile_count_fmt[d]} * '
                                                    for d in range(x)), x)
                   for x in range(stencil.dim - 1)))),
        (f'{data_fmt[stmt.name]}[original_offset] = {buf_fmt[stmt.name]}'
         f'[tiled_offset_{stmt.name} % {bank_count_fmt[stmt.name]}].get()'
         f'[tiled_offset_{stmt.name} / {bank_count_fmt[stmt.name]}];'),
    )
  for dim in range(stencil.dim * 2 - 1):
    printer.un_scope()

  printer.println('return 0;')
  printer.un_scope()
  printer.println()


def print_test(printer, stencil):
  stencil.dim = stencil.dim

  printer.printlns(
      '#ifdef SODA_TEST_MAIN',
      '',
      '#include <ap_int.h>',
      '',
      'int main(int argc, char *argv[])',
  )
  printer.do_scope()

  with printer.if_(f'argc < 2 || argc > {stencil.dim + 2}'):
    printer.printlns(
        (r'clog << "Usage: \n  " << argv[0] << " <xclbin> %s" << endl;' %
         ' '.join(map('[input dimension {}]'.format, range(stencil.dim)))),
        'return 2;',
    )

  printer.println('const char* xclbin = argv[1];')
  default_sizes = []
  for d in range(stencil.dim):
    if d != stencil.dim - 1:
      default_sizes.append(str(stencil.tile_size[d]))
    else:
      default_sizes.append(f'soda::app::{STENCIL_DIM_FMT[d]} + 1')
  printer.println(f'int32_t dims[{stencil.dim}] = {{%s}};' %
                  ', '.join(default_sizes))

  for d in range(stencil.dim):
    with printer.if_(f'argc > {d + 2}'):
      printer.println(f'dims[{d}] = std::max(dims[{d}], atoi(argv[{d + 2}]));')
  printer.println()

  data_fmt = util.MetaFmt('data_%s')
  extent_fmt = util.MetaFmt('extent_%s')
  stride_fmt = util.MetaFmt('stride_%s')
  min_fmt = util.MetaFmt('min_%s')
  for tensor in stencil.tensors.values():
    # data
    printer.println(f'vector<{tensor.c_type}> {data_fmt[tensor.name]}(%s);' %
                    (' * '.join('dims[%d]' % x for x in range(stencil.dim))))

    # extent
    printer.println(
        f'int32_t {extent_fmt[tensor.name]}[{stencil.dim}] = {{%s}};' %
        ', '.join(map('dims[{}]'.format, range(stencil.dim))))

    # stride
    stride = []
    for d in range(stencil.dim):
      if d == 0:
        stride.append('1')
      else:
        stride.append(' * '.join(map('dims[{}]'.format, range(d))))
    printer.println(
        f'int32_t {stride_fmt[tensor.name]}[{stencil.dim}] = {{%s}};' %
        ', '.join(stride))

    # min
    printer.println(f'int32_t {min_fmt[tensor.name]}[{stencil.dim}] = {{%s}};' %
                    ', '.join('0' * stencil.dim))

  for param in stencil.param_stmts:
    printer.println(
        f'{param.c_type} {data_fmt[param.name]}%s;' %
        functools.reduce(operator.add, ['[%s]' % x for x in param.size]))
  printer.println()

  if any(x.is_float for x in stencil.input_types):
    printer.printlns(
        'std::default_random_engine generator;',
        'std::uniform_real_distribution<double> distribution(0.0, 1.0);',
        '',
    )

  for name in stencil.input_names:
    printer.println('// initialization can be parallelized with -fopenmp')
    printer.println('#pragma omp parallel for', 0)
    for d in range(0, stencil.dim):
      dim = stencil.dim - d - 1
      printer.println(
          'for(int32_t {var} = 0; {var} < dims[{}]; ++{var})'.format(
              dim, var=soda_util.COORDS_IN_ORIG[dim]))
      printer.do_scope()
    init_val = '+'.join(soda_util.COORDS_IN_ORIG[0:stencil.dim])
    if stencil.symbol_table[name].is_float:
      init_val = 'distribution(generator)'
    printer.println(f'{data_fmt[name]}[%s] = %s;' %
                    (' + '.join(f'%s * {stride_fmt[name]}[%d]' %
                                (soda_util.COORDS_IN_ORIG[d], d)
                                for d in range(stencil.dim)), init_val))
    for d in range(0, stencil.dim):
      printer.un_scope()
    printer.println()

  for param in stencil.param_stmts:
    printer.println('#pragma omp parallel for', 0)
    for dim, size in enumerate(param.size):
      printer.println('for(int32_t {var} = 0; {var}<{}; ++{var})'.format(
          size, var=soda_util.COORDS_IN_ORIG[dim]))
      printer.do_scope()
    printer.println(f'{data_fmt[param.name]}%s = %s;' %
                    (param.name,
                     sum(('[%s]' % (soda_util.COORDS_IN_ORIG[d])
                          for d in range(len(param.size))), ''), '+'.join(
                              soda_util.COORDS_IN_ORIG[0:len(param.size)])))
    for d in param.size:
      printer.un_scope()
    printer.println()

  params = []
  for name in stencil.input_names + stencil.output_names + stencil.param_names:
    params.append(f'{data_fmt[name]}.data()')
    params.extend((extent_fmt[name], stride_fmt[name], min_fmt[name]))
  params.append('xclbin')
  printer.print_func(name=f'soda::app::{stencil.app_name}',
                     params=params,
                     align=0,
                     suffix=';')
  printer.println()

  printer.printlns('int error_count = 0;', '')

  for tensor in stencil.chronological_tensors:
    if tensor.is_input():
      continue
    logger.debug(f'emit code to produce {tensor.name}')
    printer.println('// produce %s, can be parallelized with -fopenmp' %
                    tensor.name)
    printer.println('#pragma omp parallel for', 0)
    stencil_window = core.get_overall_stencil_window(
        tuple(map(stencil.tensors.get, stencil.input_names))
        if tensor.is_output else tensor.parents.values(), tensor)
    stencil_dim = core.get_stencil_dim(stencil_window)
    output_idx = core.get_stencil_window_offset(stencil_window)
    for d in range(0, stencil.dim):
      dim = stencil.dim - d - 1
      printer.println(
          'for(int32_t {var} = {}; {var} < dims[{}] - {}; ++{var})'.format(
              max(0, output_idx[dim]),
              dim,
              max(0, stencil_dim[dim] - output_idx[dim] - 1),
              var=soda_util.COORDS_IN_ORIG[dim]))
      printer.do_scope()

    def mutate_load_for_host(obj, args):
      if isinstance(obj, ir.Ref):
        if obj.name in stencil.param_names:
          return ir.make_var(
              f'{data_fmt[obj.name]}%s' % ''.join('[%d]' % _ for _ in obj.idx),
              haoda_type=obj.haoda_type,
          )
        return ir.make_var(
            f'{data_fmt[obj.name]}[%s]' %
            (' + '.join(f'(%s%+d)*{stride_fmt[obj.name]}[%d]' %
                        (soda_util.COORDS_IN_ORIG[d],
                         obj.idx[d] - tensor.st_ref.idx[d], d)
                        for d in range(stencil.dim))),
            haoda_type=obj.haoda_type,
        )
      return obj

    def mutate_store_for_host(obj, args):
      if isinstance(obj, ir.Ref):
        if obj.name in stencil.output_names:
          return ir.make_var(
              '%s result_%s' % (obj.c_type, obj.name),
              haoda_type=obj.haoda_type,
          )
        return ir.make_var(
            f'{data_fmt[obj.name]}[%s]' %
            '+'.join(f'%s*{stride_fmt[obj.name]}[%d]' %
                     (soda_util.COORDS_IN_ORIG[d], d)
                     for d in range(stencil.dim)),
            haoda_type=obj.haoda_type,
        )
      return obj

    for let in tensor.lets:
      printer.printlns(
          '// let {} {} = {}'.format(let.haoda_type, let.name, let.expr),
          ('const {} {} = {};'.format(
              let.c_type, let.name,
              let.expr.visit(mutate_load_for_host).c_expr)),
      )
    printer.printlns(
        '// {} = {}'.format(tensor.st_ref, tensor.expr),
        ('{} = {};'.format(tensor.st_ref.visit(mutate_store_for_host),
                           tensor.expr.visit(mutate_load_for_host).c_expr)),
    )
    if tensor.is_output():
      run_result = f'{data_fmt[tensor.name]}[%s]' % (' + '.join(
          f'%s * {stride_fmt[tensor.name]}[%d]' %
          (soda_util.COORDS_IN_ORIG[d], d) for d in range(stencil.dim)))
      printer.printlns(
          '%s val_fpga = %s;' % (tensor.c_type, run_result),
          '%s val_cpu = result_%s;' % (tensor.c_type, tensor.name),
      )
      if tensor.haoda_type.is_float:
        printer.println('double threshold = 0.00001;')
        with printer.if_('nullptr != getenv("THRESHOLD")'):
          printer.println('threshold = atof(getenv("THRESHOLD"));')
        printer.println('threshold *= threshold;')
        with printer.if_(
            'double(val_fpga-val_cpu) * double(val_fpga-val_cpu) > '
            'threshold && '
            'double(val_fpga-val_cpu) * double(val_fpga-val_cpu) / '
            '(double(val_cpu) * double(val_cpu)) > threshold'):
          params = (', '.join(['%d'] * stencil.dim),
                    ', '.join(soda_util.COORDS_IN_ORIG[:stencil.dim]))
          printer.printlns(
              ('fprintf(stderr, "%%lf != %%lf @(%s)\\n", double'
               '(val_fpga), double(val_cpu), %s);' % params),
              '++error_count;',
          )
      else:
        with printer.if_('val_fpga != val_cpu'):
          params = (', '.join(['%d'] * stencil.dim),
                    ', '.join(soda_util.COORDS_IN_ORIG[:stencil.dim]))
          with printer.if_('error_count < 10'):
            printer.println('fprintf(stderr, "%%ld != %%ld @(%s)\\n", int64_t'
                            '(val_fpga), int64_t(val_cpu), %s);' % params)
          printer.println('++error_count;')
    for d in range(0, stencil.dim):
      printer.un_scope()
    printer.println()

  printer.printlns(
      r'clog << "INFO: " << (error_count ? "FAIL" : "PASS") << "!\n";',
      '',
  )

  printer.println('return error_count ? 1 : 0;')
  printer.un_scope()
  printer.println('#endif  // SODA_TEST_MAIN')


def print_code(stencil: core.Stencil, host_file: TextIO) -> None:
  logger.info('generate host source code as %s' % host_file.name)
  printer = util.CppPrinter(host_file)
  print_define = lambda key, value: util.print_define(printer, key, value)

  print_header(printer)

  all_stmts = stencil.input_stmts + stencil.output_stmts + stencil.param_stmts

  printer.printlns(
      'namespace soda {',
      'namespace app {',
      '// app-specific constants',
      *(f'constexpr int {STENCIL_DIM_FMT[i]} = {d};'
        for i, d in enumerate(core.get_stencil_dim(stencil.stencil_window))),
      f'constexpr int kStencilDistance = {stencil.stencil_distance};',
      *(f'constexpr int {WIDTH_FMT[x.name]} = {x.width_in_bits};'
        for x in all_stmts),
      '',
      '// type alias',
      *(f'using {TYPE_FMT[x.name]} = {x.c_type};' for x in all_stmts),
      '',
  )
  print_func(printer, stencil)
  printer.printlns(
      '} // namespace app',
      '} // namespace soda',
      '',
  )

  print_test(printer, stencil)
