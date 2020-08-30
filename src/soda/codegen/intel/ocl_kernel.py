import collections
import logging
from typing import Dict, List, TextIO, Union

from haoda import ir, util
from soda import core

_logger = logging.getLogger().getChild(__name__)


def print_code(
    stencil: core.Stencil,
    output_file: TextIO,
    code_reuse: bool = False,
) -> None:
  """Prints the top-level code with the given arguments.

  Prints the OpenCL kernels with proper pragmas and channel declarations and
  references.

  Args:
    stencil: Stencil object to print.
    output_file: TextIO to write to.
    code_reuse: Whether the kernel should reuse module definition code.
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
  for node in super_source.tpo_valid_node_gen():
    for fifo in node.fifos:
      println(f'channel {fifo.cl_type} {fifo.cl_expr};')

  println()

  if code_reuse:
    for module_trait_id, module_trait in enumerate(stencil.module_traits):
      print_kernel(
          f'{stencil.app_name}_module_{module_trait_id}',
          printer,
          module_trait,  # prints the kernel as a low-level function
          module_trait,
          module_trait_id,
          burst_width=stencil.burst_width,
      )
      println()

  # in generated bitstream, kernels are sorted in alphabetical order
  # SODA relies on the correct ordering for memory channels of each tensor
  # so here we make sure the kernel names in alphabetical order
  width = len(str(sum(1 for _ in super_source.tpo_valid_node_gen()) - 1))
  instance_idx: Dict[int, int] = collections.defaultdict(int)
  overall_idx = 0
  for node in super_source.tpo_valid_node_gen():
    module_trait, module_trait_id = super_source.module_table[node]
    instance_name = (f'{stencil.app_name}_{overall_idx:0{width}}_'
                     f'module_{module_trait_id}_'
                     f'instance_{instance_idx[module_trait_id]}')
    if code_reuse:
      print_kernel_instantiation(
          instance_name,
          f'{stencil.app_name}_module_{module_trait_id}',
          printer,
          node,
          module_trait,
          burst_width=stencil.burst_width,
      )
    else:
      print_kernel(
          instance_name,
          printer,
          node,
          module_trait,
          module_trait_id,
          burst_width=stencil.burst_width,
      )
    instance_idx[module_trait_id] += 1
    overall_idx += 1
    println()


def print_kernel_instantiation(
    instance_name: str,
    module_name: str,
    printer: util.CppPrinter,
    node: ir.Module,
    module_trait: ir.ModuleTrait,
    burst_width: int = 256,
) -> None:
  print_func(
      f'void {instance_name}',
      printer,
      node,
      module_trait,
      burst_width,
      mode='kernel',
  )
  printer.do_scope(instance_name)
  print_func(
      module_name,
      printer,
      node,
      module_trait,
      burst_width,
      mode='macro',
  )
  printer.un_scope()


def print_func(
    name: str,
    printer: util.CppPrinter,
    node: ir.Module,
    module_trait: ir.ModuleTrait,
    burst_width: int,
    mode: str,
) -> None:
  """Print function signature for various use cases.

  `mode` must be one of 'kernel_no_code_reuse', 'kernel', or 'macro'.

  For no-code-reuse kernels, only 'kernel_no_code_reuse' is used.
  For code-reuse kernels, this function is called for three different purposes.
  For kernel instantiations, 'kernel' should be used. For macros, 'macro' should
  be used.

  In 'kernel_no_code_reuse' mode, FIFO info is printed in comments, function
  parameters do not contain the FIFOs, and global pointers contain location
  attributes.

  In 'kernel' mode, FIFO info is not printed in comments, function parameters do
  not contain the FIFOs, and global pointers do contain location attributes.

  In 'macro' mode, FIFO info is not printed in comments, function parameters
  do contain the FIFOs, and only variable names are included in the parameters.
  """
  if node.dram_reads and node.dram_writes:
    raise ValueError('cannot read and write DRAM in the same module')

  params: List[str] = []
  if mode == 'macro':
    # print params in func
    params.extend(f'/*  input channel {port.cl_type} */ {arg}'
                  for port, arg in zip(module_trait.loads, node.input_fifos))
    params.extend(f'/* output channel {expr.cl_type} */ {arg}'
                  for expr, arg in zip(module_trait.exprs, node.output_fifos))
  elif mode == 'kernel_no_code_reuse':
    # print I/O info as comments
    for port, arg in zip(module_trait.loads, node.input_fifos):
      printer.println('//  input <{0.cl_type}> <- {1}'.format(port, arg))
    for expr, arg in zip(module_trait.exprs, node.output_fifos):
      printer.println('// output <{}> -> {}'.format(expr.cl_type, arg))
  kernel_attrs = ['reqd_work_group_size(1, 1, 1)', 'max_global_work_dim(0)']
  if node.dram_reads or node.dram_writes:
    if mode == 'macro':
      params.extend('/* __global {}* restrict */ {}'.format(
          dram_ref.haoda_type.get_cl_vec_type(burst_width),
          util.get_port_name(dram_ref.var, bank),
      ) for dram_ref, bank in node.dram_reads or node.dram_writes)
      params.append('/* ulong */ coalesced_data_num')
    else:
      params.extend('__global {}{}* restrict {}'.format(
          f'__attribute((buffer_location("HBM{bank}"))) ',
          dram_ref.haoda_type.get_cl_vec_type(burst_width),
          util.get_port_name(dram_ref.var, bank),
      ) for dram_ref, bank in node.dram_reads or node.dram_writes)
      params.append('ulong coalesced_data_num')

    kernel_attrs.append('uses_global_work_offset(0)')
  else:
    kernel_attrs.append('autorun')

  if mode.startswith('kernel'):
    printer.println('__kernel')
    for attr in kernel_attrs:
      printer.println(f'__attribute(({attr}))')
  printer.print_func(
      name=name,
      params=params,
      align=0,
      suffix=';' if mode == 'macro' else '',
  )


def print_kernel(name: str,
                 printer: util.CppPrinter,
                 node: Union[ir.Module, ir.ModuleTrait],
                 module_trait: ir.ModuleTrait,
                 module_trait_id: int,
                 burst_width: int = 256) -> None:
  """Print kernel as a low-level function or a top-level kernel.

  If `node` is `ir.Module`, the printed kernel is a top-level kernel instance.
  If `node` is `ir.ModuleTrait`, it should be the same object as `module_trait`,
  and the printed kernel is a low-level function, and
  `print_kernel_instantiation` should be called to print the kernel instances.

  Args:
      name (str): Name of the function.
      printer (util.Printer): Code is printed using this printer.
      node (Union[ir.Module, ir.ModuleTrait]):
      module_trait (ir.ModuleTrait): [description]
      module_trait_id (int): [description]
      burst_width (int, optional): [description]. Defaults to 256.

  Raises:
      ValueError: [description]
      NotImplementedError: [description]

  Returns:
      [type]: [description]
  """
  code_reuse = isinstance(node, ir.ModuleTrait)
  if code_reuse:
    if node is not module_trait:
      raise ValueError(
          '`node` shoud either be an `ir.Module` or is `module_trait`')
    printer.eol = ' \\\n'

  if node.dram_reads and node.dram_writes:
    raise ValueError('cannot read and write DRAM in the same module')
  println = printer.println

  print_func(
      f'#define {name}' if code_reuse else f'void {name}',
      printer,
      node,
      module_trait,
      burst_width,
      mode='macro' if code_reuse else 'kernel_no_code_reuse',
  )
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
      if node.dram_reads:
        output_count = len({x[0].var for x in node.dram_reads})
        fifo_throughput = len(node.output_fifos) // output_count
      else:
        input_count = len({x[0].var for x in node.dram_writes})
        fifo_throughput = len(node.input_fifos) // input_count
      if dram_throughput != fifo_throughput:
        raise NotImplementedError(f'memory throughput {dram_throughput} != '
                                  f'processing throughput {fifo_throughput}')

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
    println('_Pragma("ivdep")' if code_reuse else '#pragma ivdep', indent=0)
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

  if code_reuse:
    printer.eol = '\n'
  printer.un_scope()  # end of kernel function
  _logger.debug('printing: %s', module_trait)
