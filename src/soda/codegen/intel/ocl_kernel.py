import collections
import logging
from typing import Dict, List, TextIO

from haoda import ir, util
from soda import core

_logger = logging.getLogger().getChild(__name__)


def print_code(stencil: core.Stencil, output_file: TextIO) -> None:
  """Prints the top-level code with the given arguments.

  Prints the OpenCL kernels with proper pragmas and channel declarations and
  references.

  Args:
    stencil: Stencil object to print.
    output_file: TextIO to write to.
  """
  _logger.info('generate kernel code as %s' % output_file.name)
  printer = util.CppPrinter(output_file)
  println = printer.println

  println('#include <ihc_apint.h>', indent=0)
  println()

  println('#pragma OPENCL EXTENSION cl_intel_channels : enable', indent=0)
  println()

  # internal fifos
  super_source = stencil.dataflow_super_source
  for node in super_source.tpo_node_gen():
    for fifo in node.fifos:
      println('channel {fifo.cl_type} {fifo.cl_expr} '
              '__attribute__((depth({depth})));'.format(fifo=fifo,
                                                        depth=fifo.depth + 2))

  println()

  instance_count = collections.defaultdict(int)  # type: Dict[int, int]
  for node in super_source.tpo_node_gen():
    module_trait, module_trait_id = super_source.module_table[node]
    print_kernel('{}_module_{}_instance_{}'.format(
        stencil.app_name, module_trait_id, instance_count[module_trait_id]),
                 printer,
                 node,
                 module_trait,
                 module_trait_id,
                 burst_width=stencil.burst_width)
    instance_count[module_trait_id] += 1
    println()


def print_kernel(name: str,
                 printer: util.Printer,
                 node: ir.Module,
                 module_trait: ir.ModuleTrait,
                 module_trait_id: int,
                 burst_width: int = 256) -> None:
  if node.dram_reads and node.dram_writes:
    raise ValueError('cannot read and write DRAM in the same module')
  println = printer.println

  # print I/O info
  for port, arg in zip(module_trait.loads, node.input_fifos):
    println('//  input <{0.cl_type}> <- {1}'.format(port, arg))
  for expr, arg in zip(module_trait.exprs, node.output_fifos):
    println('// output <{}> -> {}'.format(expr.cl_type, arg))

  # print kernel function
  printer.println('__kernel')
  kernel_attrs = ['reqd_work_group_size(1, 1, 1)', 'max_global_work_dim(0)']

  def print_kernel_attrs():
    for attr in kernel_attrs:
      println('__attribute(({}))'.format(attr))

  if node.dram_reads or node.dram_writes:
    params = [
        '__global {} {}* restrict {}'.format(
            '__attribute((buffer_location("HBM{}")))'.format(bank),
            dram_ref.haoda_type.get_cl_vec_type(burst_width),
            util.get_port_name(dram_ref.var, bank))
        for dram_ref, bank in node.dram_reads or node.dram_writes
    ]
    params.append('ulong coalesced_data_num')
    kernel_attrs.append('uses_global_work_offset(0)')
    print_kernel_attrs()
    printer.print_func(name='void {}'.format(name), params=params, align=0)
  else:
    kernel_attrs.append('autorun')
    print_kernel_attrs()
    println('void {}()'.format(name))
  printer.do_scope(name)

  # prepare for any DelayedRef
  def get_delays(obj: ir.Node, delays: List[ir.DelayedRef]) -> ir.Node:
    if isinstance(obj, ir.DelayedRef):
      delays.append(obj)
    return obj

  delays = []  # type: List[ir.DelayedRef]
  for let in module_trait.lets:
    let.visit(get_delays, delays)
  for expr in module_trait.exprs:
    expr.visit(get_delays, delays)
  _logger.debug('delays: %s', delays)

  # inter-iteration declarations
  for delay in delays:
    println(delay.cl_buf_decl)
    println(delay.cl_ptr_decl)

  def mutate_dram_ref(obj: ir.Node, kwargs: Dict[str, int]) -> ir.Node:
    if isinstance(obj, ir.DRAMRef):
      dram_throughput = len(obj.dram) * burst_width // obj.width_in_bits
      fifo_throughput = len(node.input_fifos) or len(node.output_fifos)
      if dram_throughput != fifo_throughput:
        raise NotImplementedError(
            "input throughput {} != output throughput {}".format(
                dram_throughput, fifo_throughput))

      coalescing_idx = kwargs['coalescing_idx']
      unroll_factor = kwargs['unroll_factor']
      elem_idx = coalescing_idx * unroll_factor + obj.offset
      return ir.Var(
          name='{buf}[{idx}]'.format(
              buf=obj.dram_buf_name(obj.dram[elem_idx % len(obj.dram)]),
              idx=elem_idx // len(obj.dram),
          ),
          idx=(),
      )
    if isinstance(obj, ir.Let) and isinstance(obj.name, ir.DRAMRef):
      return ir.Var(name='{} = {};'.format(
          obj.name.visit(mutate_dram_ref, kwargs), obj.expr.cl_expr),
                    idx=())
    return obj

  if node.dram_reads or node.dram_writes:
    loop_args = 'ulong i = 0', 'i < coalesced_data_num', '++i'
  else:
    loop_args = '', '', ''

  if delays:
    println('#pragma ivdep', indent=0)
  with printer.for_(*loop_args):
    # print DelayedRef (if any)
    for delay in delays:
      println('const {} {};'.format(delay.cl_type, delay.c_buf_load))

    # print load from DRAM (if any)
    for dram_ref, bank in node.dram_reads:
      println('{gentype} {buf}[{n}];'.format(
          gentype=dram_ref.cl_type,
          n=burst_width // dram_ref.width_in_bits,
          buf=dram_ref.dram_buf_name(bank),
      ))
    for dram_ref, bank in node.dram_reads:
      println('vstore{n}({vec_ptr}[i], 0, {buf});'.format(
          vec_ptr=util.get_port_name(dram_ref.var, bank),
          n=burst_width // dram_ref.width_in_bits,
          buf=dram_ref.dram_buf_name(bank),
      ))

    # read from FIFOs
    for port, arg in zip(module_trait.loads, node.input_fifos):
      println('{0.cl_type} {0.ref_name} = read_channel_intel({1});'.format(
          port, arg))

    # declare buffer for DRAM writes
    for dram_ref, bank in node.dram_writes:
      n = burst_width // dram_ref.width_in_bits
      buf = dram_ref.dram_buf_name(bank)
      println('{gentype} {buf}[{n}];'.format(
          gentype=dram_ref.cl_type,
          n=n,
          buf=buf,
      ))

    mutate_kwargs = {
        'coalescing_idx': 0,
        'unroll_factor': len(node.input_fifos) or len(node.output_fifos),
    }

    # print Let (if any)
    for let in module_trait.lets:
      println(let.visit(mutate_dram_ref, mutate_kwargs).cl_expr)

    for expr, arg in zip(module_trait.exprs, node.output_fifos):
      println('write_channel_intel({}, {});'.format(
          arg,
          expr.visit(mutate_dram_ref, mutate_kwargs).cl_expr))

    # update DelayedRef (if any)
    for delay in delays:
      println(delay.c_buf_store)
      println('{} = {};'.format(delay.ptr, delay.cl_next_ptr_expr))

    # print store to DRAM (if any)
    for dram_ref, bank in node.dram_writes:
      println('{vec_ptr}[i] = vload{n}(0, {buf});'.format(
          vec_ptr=util.get_port_name(dram_ref.var, bank),
          n=burst_width // dram_ref.width_in_bits,
          buf=dram_ref.dram_buf_name(bank),
      ))

  printer.un_scope()  # end of kernel function
  _logger.debug('printing: %s', module_trait)
