import collections
import itertools
import logging
from typing import IO, TYPE_CHECKING, Any, Dict, List, Tuple, Union

from haoda import ir, util
from haoda.ir import visitor

from soda import dataflow
from soda.codegen.xilinx import opencl

if TYPE_CHECKING:
  from soda.core import Stencil

_logger = logging.getLogger().getChild(__name__)

SUPPORTED_INTERFACES = 'm_axi', 'axis', 'tapa::mmap', 'tapa::stream'

DATA_TYPE_FMT = dict(
    m_axi='Data<{0.c_type}>',
    axis='ap_axiu<{0.width_in_bits}, 0, 0, 0>',
)


def _check_interface(interface: str) -> None:
  if interface not in SUPPORTED_INTERFACES:
    raise NotImplementedError(f'interface "{interface}" is not implemented')


def _print_interface(
    printer: util.CppPrinter,
    stencil: 'Stencil',
    inputs: List[Tuple[str, ir.Type, int]],
    outputs: List[Tuple[str, ir.Type, int]],
    super_source: dataflow.SuperSourceNode,
    interface: str,
) -> None:
  """Prints the top-level module for the given arguments.

  Prints the top-level interfaces and sub-module instances with proper interface
  pragmas, hls::stream declarations and references, and module function calls.
  Currently only streaming applications are supported.

  Args:
    printer: CppPrinter to which the code is emitted.
    stencil: core.Stencil to print.
    inputs: List of (name, ir.Type, bank) tuples, specifying the input
        interfaces.
    outputs: List of (name, ir.Type, bank) tuples, specifying the output
        interfaces.
    super_source: SuperSourceNode of a DAG of HAODA nodes.
  """
  pack = 'aggregate bit' if opencl.hls == 'vitis_hls' else 'data_pack'

  get_bundle_name = util.get_bundle_name
  get_port_name = util.get_port_name
  get_port_buf_name = util.get_port_buf_name

  if interface in {'m_axi', 'axis'}:
    printer.println('extern "C" {\n')

  params: List[str] = []
  if interface == 'm_axi':
    params.extend(f'ap_uint<{stencil.burst_width}>* {get_port_name(name, bank)}'
                  for name, _, bank in outputs + inputs)
    params.append('uint64_t coalesced_data_num')
  elif interface == 'axis':
    params.extend(f'hls::stream<ap_axiu<{stencil.burst_width}, 0, 0, 0>>&'
                  f' {get_port_name(name, bank)}'
                  for name, haoda_type, bank in outputs + inputs)
  elif interface == 'tapa::mmap':
    for name, haoda_type, bank in outputs + inputs:
      vec_t = _vec_t(
          haoda_type.c_type,
          stencil.burst_width // haoda_type.width_in_bits,
      )
      params.append(f'tapa::mmap<{vec_t}> {get_port_name(name, bank)}')
    params.append('uint64_t coalesced_data_num')
  elif interface == 'tapa::stream':
    for tuples, io in ((outputs, 'o'), (inputs, 'i')):
      for name, haoda_type, bank in tuples:
        vec_t = _vec_t(
            haoda_type.c_type,
            stencil.burst_width // haoda_type.width_in_bits,
        )
        params.append(f'tapa::{io}stream<{vec_t}>& {get_port_name(name, bank)}')

  printer.print_func(f'void {stencil.kernel_name}', params, align=0)

  # print function body
  printer.do_scope()

  if interface == 'm_axi':
    printer.printlns(
        *(f'#pragma HLS interface m_axi port = {get_port_name(name, bank)} '
          f'offset = slave bundle = {get_bundle_name(name, bank)}'
          for name, _, bank in outputs + inputs),
        *(f'#pragma HLS interface s_axilite port = {get_port_name(name, bank)} '
          'bundle = control' for name, _, bank in outputs + inputs),
        '#pragma HLS interface s_axilite port = coalesced_data_num '
        'bundle = control',
        '#pragma HLS interface s_axilite port = return bundle = control',
        '',
    )
  elif interface == 'axis':
    printer.printlns(
        *(f'#pragma HLS interface axis port = {get_port_name(name, bank)}'
          for name, _, bank in outputs + inputs),
        '#pragma HLS interface ap_ctrl_none port = return',
    )

  # port buf declarations
  if interface == 'm_axi':
    printer.printlns(
        itertools.chain.from_iterable((
            f'hls::stream<Data<ap_uint<{stencil.burst_width}>>>'
            f' {get_port_buf_name(name, bank)}'
            f'("{get_port_buf_name(name, bank)}");',
            '#pragma HLS stream'
            f' variable = {get_port_buf_name(name, bank)} depth = 32',
            f'#pragma HLS {pack} variable = {get_port_buf_name(name, bank)}',
        ) for name, haoda_type, bank in inputs + outputs))
  elif interface == 'tapa::mmap':
    for name, haoda_type, bank in inputs + outputs:
      vec_t = _vec_t(
          haoda_type.c_type,
          stencil.burst_width // haoda_type.width_in_bits,
      )
      buf_name = get_port_buf_name(name, bank)
      printer.println(f'tapa::stream<{vec_t}, 2> {buf_name}("{buf_name}");')
  printer.println()

  # internal fifos
  if interface in {'m_axi', 'axis'}:
    printer.printlns(
        itertools.chain.from_iterable((
            f'hls::stream<{DATA_TYPE_FMT[interface].format(fifo)}>'
            f' {fifo.c_expr}("{fifo.c_expr}");',
            '#pragma HLS stream'
            f' variable = {fifo.c_expr} depth = {fifo.depth + 3}',
            f'#pragma HLS {pack} variable = {fifo.c_expr}' if interface ==
            'm_axi' else '',
        ) for node in super_source.tpo_valid_node_gen() for fifo in node.fifos))
  elif interface.startswith('tapa::'):
    printer.printlns(f'tapa::stream<{fifo.c_type}, {fifo.depth + 3}> '
                     f'{fifo.c_expr}("{fifo.c_expr}");'
                     for node in super_source.tpo_valid_node_gen()
                     for fifo in node.fifos)
  printer.println()

  # start of dataflow region
  if interface in {'m_axi', 'axis'}:
    printer.println('#pragma HLS dataflow', 0)
  elif interface.startswith('tapa::'):
    printer.println('tapa::task()')
    printer.do_indent()

  # load modules
  if interface == 'm_axi':
    printer.printlns(f'BurstRead({get_port_buf_name(name, bank)},'
                     f' {get_port_name(name, bank)}, coalesced_data_num);'
                     for name, _, bank in inputs)
  elif interface == 'tapa::mmap':
    for name, haoda_type, bank in inputs:
      sanitized_vec_t = _sanitized_vec_t(
          haoda_type, stencil.burst_width // haoda_type.width_in_bits)
      printer.printlns(
          f'.invoke(BurstRead{sanitized_vec_t},',
          f'  {get_port_buf_name(name, bank)},',
          f'  {get_port_name(name, bank)},',
          '  coalesced_data_num)',
      )

  # SODA modules
  for node in super_source.tpo_valid_node_gen():
    module_trait_id = super_source.module_table[node][1]
    _print_module_func_call(printer, node, module_trait_id, interface)

  # store modules
  if interface == 'm_axi':
    printer.printlns(f'BurstWrite({get_port_name(name, bank)},'
                     f' {get_port_buf_name(name, bank)}, coalesced_data_num);'
                     for name, _, bank in outputs)
  elif interface == 'tapa::mmap':
    for name, haoda_type, bank in outputs:
      sanitized_vec_t = _sanitized_vec_t(
          haoda_type, stencil.burst_width // haoda_type.width_in_bits)
      printer.printlns(
          f'.invoke(BurstWrite{sanitized_vec_t},',
          f'  {get_port_name(name, bank)},',
          f'  {get_port_buf_name(name, bank)},',
          '  coalesced_data_num)',
      )

  # end of dataflow region
  if interface.startswith('tapa::'):
    printer.un_indent()
    printer.println(';')

  printer.un_scope()
  if interface in {'m_axi', 'axis'}:
    printer.printlns('', '}  // extern "C"')


def print_header(
    printer: util.CppPrinter,
    interface: str = SUPPORTED_INTERFACES[0],
) -> None:
  third_party_headers = ['ap_int']
  if interface == 'm_axi':
    third_party_headers.append('hls_stream')
  elif interface == 'axis':
    third_party_headers += 'ap_axi_sdata', 'hls_stream'
  elif interface.startswith('tapa::'):
    third_party_headers.append('tapa')

  printer.printlns(
      *map('#include <c{}>'.format, [
          'float',
          'math',
          'stdbool',
          'stddef',
          'stdint',
          'stdio',
          'string',
      ]),
      '',
      *(map('#include <{}>'.format, [
          'algorithm',
          'tuple',
      ])),
      '',
      *(map('#include <{}.h>'.format, third_party_headers)),
      '',
  )


def _print_burst_read_m_axi(printer):
  printer.println('''template <typename T>
void BurstRead(hls::stream<Data<T>>& to, T* from, uint64_t data_num) {
load:
  for (uint64_t i = 0; i < data_num;) {
#pragma HLS pipeline II = 1
    const uint64_t next_i = i + 1;
    WriteData(to, from[i], next_i < data_num);
    i = next_i;
  }
}
''')


def _print_burst_write_m_axi(printer):
  printer.println('''template <typename T>
void BurstWrite(T* to, hls::stream<Data<T>>& from, uint64_t data_num) {
store:
  for (uint64_t i = 0; i < data_num; ++i) {
#pragma HLS pipeline II = 1
    T buf;
    ReadData(buf, from);
    to[i] = buf;
  }
}
''')


def _print_burst_read_tapa(
    printer: util.CppPrinter,
    haoda_type: ir.Type,
    length: int,
) -> None:
  printer.printlns(
      f'void BurstRead{_sanitized_vec_t(haoda_type, length)}(',
      f'  tapa::ostream<{_vec_t(haoda_type.c_type, length)}>& dest,',
      f'  tapa::async_mmap<{_vec_t(haoda_type.c_type, length)}>& src,',
      '  uint64_t n)',
  )
  printer.do_scope()
  printer.println('load:', 0)
  with printer.for_('uint64_t i_req = 0, i_resp = 0', 'i_resp < n', ''):
    printer.println('#pragma HLS pipeline II = 1', 0)
    with printer.if_(' && '.join((
        'i_req < n',
        'src.read_addr.try_write(i_req)',
    ))):
      printer.println('++i_req;')
    with printer.if_('!dest.full() && !src.read_data.empty()'):
      printer.printlns(
          'dest.try_write(src.read_data.read(nullptr));',
          '++i_resp;',
      )
  printer.un_scope()
  printer.println()


def _print_burst_write_tapa(
    printer: util.CppPrinter,
    haoda_type: ir.Type,
    length: int,
) -> None:
  printer.printlns(
      f'void BurstWrite{_sanitized_vec_t(haoda_type, length)}(',
      f'  tapa::async_mmap<{_vec_t(haoda_type.c_type, length)}>& dest,',
      f'  tapa::istream<{_vec_t(haoda_type.c_type, length)}>& src,',
      '  uint64_t n)',
  )
  printer.do_scope()
  printer.println('store:', 0)
  with printer.for_('uint64_t i_req = 0, i_resp = 0', 'i_resp < n', ''):
    printer.println('#pragma HLS pipeline II = 1', 0)
    with printer.if_(' && '.join((
        'i_req < n',
        '!src.empty()',
        '!dest.write_addr.full()',
        '!dest.write_data.full()',
    ))):
      printer.printlns(
          'dest.write_addr.try_write(i_req);',
          'dest.write_data.try_write(src.read(nullptr));',
          '++i_req;',
      )
    with printer.if_('!dest.write_resp.empty()'):
      printer.printlns(
          'ap_uint<9> resp = dest.write_resp.read(nullptr);',
          'i_resp += ++resp;',
      )
  printer.un_scope()
  printer.println()


def _vec_t(elem_type: str, length: int) -> str:
  return f'tapa::vec_t<{elem_type}, {length}>'


def _sanitized_vec_t(elem_type: str, length: int) -> str:
  return f'_{elem_type}x{length}'


def print_code(
    stencil: 'Stencil',
    output_file: IO[str],
    interface: str = SUPPORTED_INTERFACES[0],
):
  _check_interface(interface)

  _logger.info('generate kernel code as %s' % output_file.name)
  printer = util.CppPrinter(output_file)

  printer.printlns(stencil.meta_lines)

  print_header(printer, interface)

  # emit tuple struct definitions
  printer.printlns(x.c_type_def for x in stencil.tuple_types)

  if interface in {'m_axi', 'axis'}:
    printer.printlns(
        '#ifdef __SYNTHESIS__',
        '#warning this file should be used for simulation only',
        '#warning synthesis result may be sub-optimal',
        '#endif  // __SYNTHESIS__',
        '',
    )

  if interface in {'m_axi', 'axis'}:
    _print_reinterpret(printer)

  if interface == 'm_axi':
    _print_data_struct(printer)
    _print_read_data_m_axi(printer)
    _print_write_data_m_axi(printer)
    _print_burst_read_m_axi(printer)
    _print_burst_write_m_axi(printer)
  elif interface == 'axis':
    _print_read_data_axis(printer)
    _print_write_data_axis(printer)
  elif interface == 'tapa::mmap':
    for input_iface_haoda_type in {
        stmt.haoda_type: None for stmt in stencil.input_stmts
    }:
      _print_burst_read_tapa(
          printer, input_iface_haoda_type,
          stencil.burst_width // input_iface_haoda_type.width_in_bits)
    for output_iface_haoda_type in {
        stmt.haoda_type: None for stmt in stencil.output_stmts
    }:
      _print_burst_write_tapa(
          printer, output_iface_haoda_type,
          stencil.burst_width // output_iface_haoda_type.width_in_bits)

  for module_trait_id, module_trait in enumerate(stencil.module_traits):
    print_module_definition(
        printer,
        module_trait,
        module_trait_id,
        stencil.burst_width,
        interface,
    )

  outputs = []
  inputs = []
  for stmt in stencil.output_stmts:
    for bank in stmt.dram:
      outputs.append((stmt.name, stmt.haoda_type, bank))
  for stmt in stencil.input_stmts:
    for bank in stmt.dram:
      inputs.append((stmt.name, stmt.haoda_type, bank))
  for stmt in stencil.param_stmts:
    inputs.append(('var_%s' % stmt.name, stmt.type, 0))
  _print_interface(printer, stencil, inputs, outputs,
                   stencil.dataflow_super_source, interface)


def _print_module_func_call(printer: util.CppPrinter, node: ir.Module,
                            module_trait_id: int, interface: str) -> None:
  func_name = util.get_func_name(module_trait_id)

  if interface == 'm_axi':
    get_port_name = util.get_port_buf_name
  elif interface == 'axis':
    get_port_name = util.get_port_name
  elif interface == 'tapa::mmap':
    get_port_name = util.get_port_buf_name
  elif interface == 'tapa::stream':
    get_port_name = util.get_port_name

  dram_reads = tuple('  /* input*/ ' + get_port_name(dram_ref.var, bank)
                     for dram_ref, bank in node.dram_reads)
  dram_writes = tuple('  /*output*/ ' + get_port_name(dram_ref.var, bank)
                      for dram_ref, bank in node.dram_writes)
  output_fifos = tuple('  /*output*/ ' + _ for _ in node.output_fifos)
  input_fifos = tuple('  /* input*/ ' + _ for _ in node.input_fifos)
  params = dram_writes + output_fifos + input_fifos + dram_reads

  if interface in {'m_axi', 'axis'}:
    printer.print_func(func_name, params, suffix=';', align=0)
  elif interface.startswith('tapa::'):
    printer.printlns(
        f'.invoke<tapa::detach>({func_name},',
        *(x + ',' for x in params[:-1]),
        params[-1] + ')',
    )


def _mutate_dram_ref_for_writes(obj: ir.Node, kwargs: Dict[str, Any]) -> None:
  if isinstance(obj, ir.DRAMRef):
    coalescing_idx = kwargs.pop('coalescing_idx')
    unroll_factor = kwargs.pop('unroll_factor')
    interface = kwargs.pop('interface')
    num_bank_map = kwargs.pop('num_bank_map')
    type_width = obj.haoda_type.width_in_bits
    elem_idx = coalescing_idx * unroll_factor + obj.offset
    num_banks = num_bank_map[obj.var]
    bank = obj.dram[elem_idx % num_banks]
    if interface in {'m_axi', 'axis'}:
      lsb = (elem_idx // num_banks) * type_width
      msb = lsb + type_width - 1
      return ir.Var(name=f'{obj.dram_buf_name(bank)}({msb}, {lsb})', idx=())
    if interface.startswith('tapa::'):
      return ir.Var(
          name=f'{obj.dram_buf_name(bank)}.set({elem_idx // num_banks}',
          idx=(),
      )
  return obj


def _mutate_dram_ref_for_reads(obj: ir.Node, kwargs: Dict[str, Any]) -> None:
  if isinstance(obj, ir.DRAMRef):
    coalescing_idx = kwargs['coalescing_idx']
    unroll_factor = kwargs['unroll_factor']
    interface = kwargs['interface']
    num_bank_map = kwargs['num_bank_map']
    expr = kwargs['expr']
    type_width = obj.haoda_type.width_in_bits
    elem_idx = coalescing_idx * unroll_factor + obj.offset
    num_banks = num_bank_map[obj.var]
    bank = _get_dram_attr(expr, 'dram')[elem_idx % num_banks]
    if interface in {'m_axi', 'axis'}:
      lsb = (elem_idx // num_banks) * type_width
      msb = lsb + type_width - 1
      return ir.Var(
          name=f'Reinterpret<{obj.c_type}>('
          f'static_cast<ap_uint<{msb - lsb + 1}>>('
          f'{obj.dram_buf_name(bank)}({msb}, {lsb})))',
          idx=(),
      )
    if interface.startswith('tapa::'):
      return ir.Var(
          name=f'{obj.dram_buf_name(bank)}[{elem_idx // num_banks}]',
          idx=(),
      )
  return obj


def _process_accesses(
    module_trait: ir.ModuleTrait,
    burst_width: int,
    interface: str,
):
  # input/output channels
  if interface in {'m_axi', 'axis'}:
    fifo_loads = [
        f'/* input*/ hls::stream<{DATA_TYPE_FMT[interface].format(x)}>&'
        f' {x.ld_name}' for x in module_trait.loads
    ]
    fifo_stores = [
        f'/*output*/ hls::stream<{DATA_TYPE_FMT[interface].format(expr)}>&'
        f' {ir.FIFORef.ST_PREFIX}{idx}'
        for idx, expr in enumerate(module_trait.exprs)
    ]
  elif interface.startswith('tapa::'):
    fifo_loads = [
        f'tapa::istream<{x.c_type}>& {x.ld_name}' for x in module_trait.loads
    ]
    fifo_stores = [
        f'tapa::ostream<{expr.c_type}>& {ir.FIFORef.ST_PREFIX}{idx}'
        for idx, expr in enumerate(module_trait.exprs)
    ]

  # format strings for input/output channels for packing/unpacking modules
  if interface == 'm_axi':
    fifo_load_fmt = ("f'/* input*/ hls::stream<Data<ap_uint<{burst_width}>>>&"
                     " {x.dram_fifo_name(bank)}'")
    fifo_store_fmt = ("f'/*output*/ hls::stream<Data<ap_uint<{burst_width}>>>&"
                      " {x.dram_fifo_name(bank)}'")
  elif interface == 'axis':
    fifo_load_fmt = (
        "f'/* input*/ hls::stream<ap_axiu<{burst_width}, 0, 0, 0>>&"
        " {x.dram_fifo_name(bank)}'")
    fifo_store_fmt = (
        "f'/*output*/ hls::stream<ap_axiu<{burst_width}, 0, 0, 0>>&"
        " {x.dram_fifo_name(bank)}'")
  elif interface.startswith('tapa::'):
    fifo_load_fmt = (
        "f'tapa::istream<{_vec_t(x.c_type,"
        " burst_width // x.width_in_bits)}>& {x.dram_fifo_name(bank)}'")
    fifo_store_fmt = (
        "f'tapa::ostream<{_vec_t(x.c_type,"
        " burst_width // x.width_in_bits)}>& {x.dram_fifo_name(bank)}'")

  # dict mapping variable name to
  #   dict mapping bank tuple to
  #     dict mapping offset to ir.DRAMRef
  dram_read_map: Dict[str, Dict[Tuple[int, ...], Dict[int, ir.DRAMRef]]]
  dram_read_map = collections.defaultdict(dict)
  dram_write_map: Dict[str, Dict[Tuple[int, ...], Dict[int, ir.DRAMRef]]]
  dram_write_map = collections.defaultdict(dict)

  num_bank_map: Dict[str, int] = {}
  all_dram_reads: List[ir.DRAMRef] = []
  dram_reads: List[ir.DRAMRef] = []
  coalescing_factor = 0
  ii = 1

  exprs = [_.expr for _ in module_trait.lets]
  exprs.extend(module_trait.exprs)
  dram_read_refs: Tuple[ir.DRAMRef, ...] = visitor.get_dram_refs(exprs)
  dram_write_refs: Tuple[ir.DRAMRef, ...] = visitor.get_dram_refs(
      _.name for _ in module_trait.lets if not isinstance(_.name, str))

  # temporary dict mapping variable name to
  #   dict mapping bank tuple to
  #     list of ir.DRAMRef
  dram_map: Dict[str, Dict[Tuple[int, ...], List[ir.DRAMRef]]]
  dram_map = collections.defaultdict(lambda: collections.defaultdict(list))

  if dram_read_refs:  # this is an unpacking module
    assert not dram_write_refs, 'cannot read and write DRAM in the same module'
    for dram_ref in dram_read_refs:
      dram_map[dram_ref.var][dram_ref.dram].append(dram_ref)
    _logger.debug('dram read map: %s', dram_map)
    for var in dram_map:
      for dram in dram_map[var]:
        # number of elements per cycle
        batch_size = len(dram_map[var][dram])
        dram_read_map[var][dram] = {_.offset: _ for _ in dram_map[var][dram]}
        offset_dict = dram_read_map[var][dram]
        num_banks = len(dram)
        if var in num_bank_map:
          assert num_bank_map[var] == num_banks, 'inconsistent num banks'
        else:
          num_bank_map[var] = num_banks
        _logger.debug('dram reads: %s', offset_dict)
        assert tuple(sorted(offset_dict.keys())) == tuple(range(batch_size)), \
               'unexpected DRAM accesses pattern %s' % offset_dict
        batch_width = sum(
            _.haoda_type.width_in_bits for _ in offset_dict.values())
        if burst_width * num_banks >= batch_width:
          assert burst_width * num_banks % batch_width == 0, \
              'cannot process such a burst'
          # a single burst consumed in multiple cycles
          coalescing_factor = burst_width * num_banks // batch_width
          ii = coalescing_factor
        else:
          assert batch_width * num_banks % burst_width == 0, \
              'cannot process such a burst'
          # multiple bursts consumed in a single cycle
          # reassemble_factor = batch_width // (burst_width * num_banks)
          _logger.error('DRAM channels: %s', dram)
          _logger.error('burst width  : %d', burst_width)
          _logger.error('batch width  : %d', batch_width)
          raise util.InternalError(
              'cannot process such a burst yet: burst width too small')
      dram_reads = [next(iter(_.values())) for _ in dram_read_map[var].values()]
      all_dram_reads += dram_reads
      fifo_loads.extend(
          eval(
              fifo_load_fmt,
              dict(burst_width=burst_width, _vec_t=_vec_t),
              locals(),
          ) for x in dram_reads for bank in x.dram)
  elif dram_write_refs:  # this is a packing module
    for dram_ref in dram_write_refs:
      dram_map[dram_ref.var][dram_ref.dram].append(dram_ref)
    _logger.debug('dram write map: %s', dram_map)
    for var in dram_map:
      for dram in dram_map[var]:
        # number of elements per cycle
        batch_size = len(dram_map[var][dram])
        dram_write_map[var][dram] = {_.offset: _ for _ in dram_map[var][dram]}
        offset_dict = dram_write_map[var][dram]
        num_banks = len(dram)
        if var in num_bank_map:
          assert num_bank_map[var] == num_banks, 'inconsistent num banks'
        else:
          num_bank_map[var] = num_banks
        _logger.debug('dram writes: %s', offset_dict)
        assert tuple(sorted(offset_dict.keys())) == tuple(range(batch_size)), \
               'unexpected DRAM accesses pattern %s' % offset_dict
        batch_width = sum(
            _.haoda_type.width_in_bits for _ in offset_dict.values())
        if burst_width * num_banks >= batch_width:
          assert burst_width * num_banks % batch_width == 0, \
              'cannot process such a burst'
          # a single burst consumed in multiple cycles
          coalescing_factor = burst_width * num_banks // batch_width
          ii = coalescing_factor
        else:
          assert batch_width * num_banks % burst_width == 0, \
              'cannot process such a burst'
          # multiple bursts consumed in a single cycle
          # reassemble_factor = batch_width // (burst_width * num_banks)
          raise util.InternalError('cannot process such a burst yet')
      dram_writes = [
          next(iter(_.values())) for _ in dram_write_map[var].values()
      ]
      fifo_stores.extend(
          eval(
              fifo_store_fmt,
              dict(burst_width=burst_width, _vec_t=_vec_t),
              locals(),
          ) for x in dram_writes for bank in x.dram)

  return (
      fifo_loads,
      fifo_stores,
      dram_read_map,
      dram_write_map,
      num_bank_map,
      all_dram_reads,
      coalescing_factor,
      ii,
  )


def print_module_definition(
    printer: util.CppPrinter,
    module_trait: ir.ModuleTrait,
    module_trait_id: int,
    burst_width: int,
    interface: str = SUPPORTED_INTERFACES[0],
) -> None:
  pack = 'aggregate bit' if opencl.hls == 'vitis_hls' else 'data_pack'

  func_name = util.get_func_name(module_trait_id)
  func_lower_name = util.get_module_name(module_trait_id)

  delays = {x: x for x in visitor.get_instances_of(module_trait, ir.DelayedRef)}
  _logger.debug('delays: %s', delays)

  (
      fifo_loads,
      fifo_stores,
      dram_read_map,
      dram_write_map,
      num_bank_map,
      dram_reads,
      coalescing_factor,
      ii,
  ) = _process_accesses(
      module_trait,
      burst_width,
      interface,
  )

  dram_rw_map = {**dram_read_map, **dram_write_map}

  # print function
  printer.print_func(f'void {func_name}', fifo_stores + fifo_loads, align=0)
  printer.do_scope(func_name)

  if interface == 'm_axi':
    printer.printlns(
        *(f'#pragma HLS {pack} variable = {dram_ref.dram_fifo_name(bank)}'
          for dram_ref, bank in module_trait.dram_writes +
          module_trait.dram_reads),
        *(f'#pragma HLS {pack} variable = {arg}'
          for arg in module_trait.output_fifos + module_trait.input_fifos),
    )

  # print inter-iteration declarations
  printer.printlns({x.c_ptr_decl: None for x in delays})
  printer.printlns(x.c_buf_decl for x in delays)
  printer.printlns(
      '#ifdef __VITIS_HLS__',
      *(f'#pragma HLS bind_storage variable = {delay.buf_name} '
        'type = ram_s2p latency = 3' for delay in delays),
      '#endif  // __VITIS_HLS__',
  )

  # print loop
  printer.println(f'{func_lower_name}:', indent=0)
  if interface in {'m_axi', 'axis'}:
    printer.println('for (bool enable = true; enable;)')
  else:
    printer.println('for (;;)')
  printer.do_scope(f'for {func_lower_name}')
  printer.printlns(
      f'#pragma HLS pipeline II = {ii}',
      *('#pragma HLS dependence inter true '
        f'variable = {delay.buf_name} distance = {delay.delay}'
        for delay in delays),
  )

  # print emptyness tests
  printer.println('if (%s)' % (' && '.join(
      f'!{fifo}.empty()' for fifo in [_.ld_name for _ in module_trait.loads] +
      [_.dram_fifo_name(bank) for _ in dram_reads for bank in _.dram])))
  printer.do_scope('if not empty')

  # print intra-iteration declarations
  printer.printlns(
      f'{fifo_in.c_type} {fifo_in.ref_name};' for fifo_in in module_trait.loads)
  if interface in {'m_axi', 'axis'}:
    printer.printlns(
        f'ap_uint<{burst_width}> {dram.dram_buf_name(bank)};'
        for var, accesses in dram_rw_map.items()
        for dram in (next(iter(_.values())) for _ in accesses.values())
        for bank in dram.dram)
  elif interface.startswith('tapa::'):
    printer.printlns(
        f'{_vec_t(dram.c_type, burst_width//dram.width_in_bits)} '
        f'{dram.dram_buf_name(bank)};' for var, accesses in dram_rw_map.items()
        for dram in (next(iter(_.values())) for _ in accesses.values())
        for bank in dram.dram)

  if interface in {'m_axi', 'axis'}:
    # print enable conditions
    if not dram_write_map:
      printer.printlns(f'const bool {fifo_in.ref_name}_enable = '
                       f'ReadData({fifo_in.ref_name}, {fifo_in.ld_name});'
                       for fifo_in in module_trait.loads)
    printer.printlns(
        f'const bool {x.dram_buf_name(bank)}_enable = '
        f'ReadData({x.dram_buf_name(bank)}, {x.dram_fifo_name(bank)});'
        for x in dram_reads
        for bank in x.dram)
    if not dram_write_map:
      printer.println(
          'const bool enabled = %s;' %
          ' && '.join([f'{y.ref_name}_enable' for y in module_trait.loads] + [
              f'{x.dram_buf_name(bank)}_enable' for x in dram_reads
              for bank in x.dram
          ]))
      printer.println('enable = enabled;')
  elif interface.startswith('tapa::'):
    # print FIFO reads
    if not dram_write_map:
      printer.printlns(f'{fifo_in.ref_name} = {fifo_in.ld_name}.read(nullptr);'
                       for fifo_in in module_trait.loads)
    printer.printlns(f'{dram.dram_buf_name(bank)} = '
                     f'{dram.dram_fifo_name(bank)}.read(nullptr);'
                     for dram in dram_reads
                     for bank in dram.dram)

  # print delays (if any)
  printer.printlns(f'const {x.c_type} {x.c_buf_load};' for x in delays)

  # mutate dram ref for writes
  if dram_write_map:
    for coalescing_idx in range(coalescing_factor):
      if interface in {'m_axi', 'axis'}:
        for fifo_in in module_trait.loads:
          if coalescing_idx == coalescing_factor - 1:
            prefix = f'const bool {fifo_in.ref_name}_enable = '
          else:
            prefix = ''
          printer.println(f'{prefix}ReadData({fifo_in.ref_name},'
                          f' {fifo_in.ld_name});')
        if coalescing_idx == coalescing_factor - 1:
          printer.printlns(
              'const bool enabled = %s;' %
              ' && '.join([f'{x.ref_name}_enable' for x in module_trait.loads] +
                          [
                              f'{x.dram_buf_name(bank)}_enable'
                              for x in dram_reads
                              for bank in x.dram
                          ]),
              'enable = enabled;',
          )
      elif interface.startswith('tapa::'):
        printer.printlns(
            f'{fifo_in.ref_name} = {fifo_in.ld_name}.read(nullptr);'
            for fifo_in in module_trait.loads)
      for idx, let in enumerate(module_trait.lets):
        let = let.visit(
            _mutate_dram_ref_for_writes,
            dict(
                coalescing_idx=coalescing_idx,
                unroll_factor=len(dram_write_map[let.name.var][let.name.dram]),
                num_bank_map=num_bank_map,
                interface=interface,
            ))
        if interface in {'m_axi', 'axis'}:
          printer.println(
              f'{let.name} = '
              f'Reinterpret<ap_uint<{let.expr.haoda_type.width_in_bits}>>'
              f'({let.expr.c_expr});')
        elif interface.startswith('tapa::'):
          printer.println(f'{let.name}, {let.expr.c_expr});')
    if interface in {'m_axi', 'axis'}:
      printer.printlns(
          f'WriteData({dram.dram_fifo_name(bank)}, '
          f'{dram.dram_buf_name(bank)}, enabled);' for var in dram_write_map
          for dram in (
              next(iter(_.values())) for _ in dram_write_map[var].values())
          for bank in dram.dram)
    elif interface.startswith('tapa::'):
      printer.printlns(
          f'{dram.dram_fifo_name(bank)}.write({dram.dram_buf_name(bank)});'
          for var in dram_write_map for dram in (
              next(iter(_.values())) for _ in dram_write_map[var].values())
          for bank in dram.dram)
  else:
    printer.printlns(let.c_expr for let in module_trait.lets)

  # mutate dram ref for reads
  if dram_read_map:
    for coalescing_idx in range(coalescing_factor):
      for idx, expr in enumerate(module_trait.exprs):
        var = _get_dram_attr(expr, 'var')
        dram = _get_dram_attr(expr, 'dram')
        c_expr = expr.visit(
            _mutate_dram_ref_for_reads,
            dict(
                coalescing_idx=coalescing_idx,
                unroll_factor=len(dram_read_map[var][dram]),
                num_bank_map=num_bank_map,
                interface=interface,
                expr=expr,
            )).c_expr
        if interface in {'m_axi', 'axis'}:
          if coalescing_idx < coalescing_factor - 1:
            enabled = 'true'
          else:
            enabled = 'enabled'
          printer.println(
              f'WriteData({ir.FIFORef.ST_PREFIX}{idx}, {c_expr}, {enabled});')
        elif interface.startswith('tapa::'):
          printer.println(f'{ir.FIFORef.ST_PREFIX}{idx}.write({c_expr});')
  else:
    if interface in {'m_axi', 'axis'}:
      printer.printlns(f'WriteData({ir.FIFORef.ST_PREFIX}{idx}, '
                       f'{expr.c_type}({expr.c_expr}), enabled);'
                       for idx, expr in enumerate(module_trait.exprs))
    elif interface.startswith('tapa::'):
      printer.printlns(f'{ir.FIFORef.ST_PREFIX}{idx}.write({expr.c_expr});'
                       for idx, expr in enumerate(module_trait.exprs))

  printer.printlns(x.c_buf_store for x in delays)
  printer.printlns({f'{x.c_next_ptr_expr};': None for x in delays})

  printer.un_scope()
  printer.un_scope()
  printer.un_scope()
  printer.println()
  _logger.debug('printing: %s', module_trait)


def _get_dram_attr(node: Union[ir.Pack, ir.DRAMRef], name: str):
  if isinstance(node, ir.Pack):
    attr = getattr(node.exprs[0], name)
    for expr in node.exprs:
      assert isinstance(expr, ir.DRAMRef)
      assert attr == getattr(expr, name)
    return attr
  if isinstance(node, ir.DRAMRef):
    return getattr(node, name)
  raise TypeError


def _print_data_struct(printer):
  printer.println('''template<typename T>
struct Data {
  T data;
  bool ctrl;
};
''')


def _print_reinterpret(printer):
  printer.println('''template<typename To, typename From>
To Reinterpret(From val) {
#pragma HLS inline
  return reinterpret_cast<To&>(val);
}
''')


def _print_read_data_m_axi(printer):
  printer.println('''template<typename T>
bool ReadData(T& data, hls::stream<Data<T>>& from) {
#pragma HLS inline
  const auto tmp = from.read();
  data = tmp.data;
  return tmp.ctrl;
}
''')


def _print_write_data_m_axi(printer):
  printer.println('''template<typename T>
void WriteData(hls::stream<Data<T>>& to, const T& data, bool ctrl) {
#pragma HLS inline
  Data<T> tmp;
  tmp.data = data;
  tmp.ctrl = ctrl;
  to.write(tmp);
}
''')


def _print_read_data_axis(printer):
  printer.println('''template<typename T, int D>
bool ReadData(T& data, hls::stream<ap_axiu<D, 0, 0, 0>>& from) {
#pragma HLS inline
  const auto tmp = from.read();
  data = Reinterpret<T>(tmp.data);
  return !tmp.last;
}
''')


def _print_write_data_axis(printer):
  printer.println('''template<typename T, int D>
void WriteData(hls::stream<ap_axiu<D, 0, 0, 0>>& to, const T& data, bool ctrl) {
#pragma HLS inline
  ap_axiu<D, 0, 0, 0> tmp = {data, 0, 0, !ctrl};
  tmp.keep.b_not();
  tmp.strb.b_not();
  to.write(tmp);
}
''')


def print_connectivity(stencil: 'Stencil', output_file: IO[str]) -> None:
  printer = util.Printer(output_file)
  printer.println('[connectivity]')
  for stmt in stencil.input_stmts + stencil.output_stmts:
    for bank in stmt.dram:
      port_name = util.get_port_name(stmt.name, bank)
      printer.println(f'sp={stencil.kernel_name}_1.{port_name}:DDR[{bank}]')
