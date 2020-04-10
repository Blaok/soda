import collections
import concurrent.futures
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from typing import BinaryIO, Dict, Iterable, List, Optional, TextIO, Tuple

from haoda import ir, util
from haoda.backend import xilinx as backend
from haoda.report.xilinx import hls as hls_report
from soda import core
from soda.codegen.xilinx import hls_kernel

_logger = logging.getLogger().getChild(__name__)

def print_code(stencil: core.Stencil, xo_file: BinaryIO,
               platform: Optional[str] = None,
               jobs: Optional[int] = os.cpu_count(),
               rpt_file: Optional[str] = None) -> None:
  """Generate hardware object file for the given Stencil.

  Working `vivado` and `vivado_hls` is required in the PATH.

  Args:
    stencil: Stencil object to generate from.
    xo_file: file object to write to.
    platform: path to the SDAccel platform directory.
    jobs: maximum number of jobs running in parallel.
    rpt_file: path of the generated report; None disables report generation.
  """

  m_axi_names = []
  m_axi_bundles = []
  inputs = []
  outputs = []
  for stmt in stencil.output_stmts + stencil.input_stmts:
    for bank in stmt.dram:
      haoda_type = ir.Type('uint%d' % stencil.burst_width)
      bundle_name = util.get_bundle_name(stmt.name, bank)
      m_axi_names.append(bundle_name)
      m_axi_bundles.append((bundle_name, haoda_type))

  for stmt in stencil.output_stmts:
    for bank in stmt.dram:
      haoda_type = ir.Type('uint%d' % stencil.burst_width)
      bundle_name = util.get_bundle_name(stmt.name, bank)
      outputs.append((util.get_port_name(stmt.name, bank), bundle_name,
                      haoda_type, util.get_port_buf_name(stmt.name, bank)))
  for stmt in stencil.input_stmts:
    for bank in stmt.dram:
      haoda_type = ir.Type('uint%d' % stencil.burst_width)
      bundle_name = util.get_bundle_name(stmt.name, bank)
      inputs.append((util.get_port_name(stmt.name, bank), bundle_name,
                     haoda_type, util.get_port_buf_name(stmt.name, bank)))

  top_name = stencil.app_name + '_kernel'

  if 'XDEVICE' in os.environ:
    xdevice = os.environ['XDEVICE'].replace(':', '_').replace('.', '_')
    if platform is None or not os.path.exists(platform):
      platform = os.path.join('/opt/xilinx/platforms', xdevice)
    if platform is None or not os.path.exists(platform):
      if 'XILINX_SDX' in os.environ:
        platform = os.path.join(os.environ['XILINX_SDX'], 'platforms', xdevice)
  if platform is None or not os.path.exists(platform):
    raise ValueError('Cannot determine platform from environment.')
  device_info = backend.get_device_info(platform)

  with tempfile.TemporaryDirectory(prefix='sodac-xrtl-') as tmpdir:
    kernel_xml = os.path.join(tmpdir, 'kernel.xml')
    with open(kernel_xml, 'w') as kernel_xml_obj:
      print_kernel_xml(top_name, inputs, outputs, kernel_xml_obj)

    kernel_file = os.path.join(tmpdir, 'kernel.cpp')
    with open(kernel_file, 'w') as kernel_fileobj:
      hls_kernel.print_code(stencil, kernel_fileobj)

    args = []
    for module_trait_id, module_trait in enumerate(stencil.module_traits):
      sio = io.StringIO()
      hls_kernel.print_module_definition(util.CppPrinter(sio),
                                          module_trait,
                                          module_trait_id,
                                          burst_width=stencil.burst_width)
      args.append((len(sio.getvalue()), synthesis_module, tmpdir, [kernel_file],
                   util.get_func_name(module_trait_id), device_info))
    args.sort(key=lambda x: x[0], reverse=True)

    super_source = stencil.dataflow_super_source
    job_server = util.release_job_slot()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
      threads = [executor.submit(*x[1:]) for x in args]
      for future in concurrent.futures.as_completed(threads):
        returncode, stdout, stderr = future.result()
        log_func = _logger.error if returncode != 0 else _logger.debug
        if stdout:
          log_func(stdout.decode())
        if stderr:
          log_func(stderr.decode())
        if returncode != 0:
          util.pause_for_debugging()
          sys.exit(returncode)
    util.acquire_job_slot(job_server)

    # generate HLS report
    depths: Dict[int, int] = {}
    hls_resources = hls_report.HlsResources()
    _logger.info(hls_resources)
    for module_id, nodes in enumerate(super_source.module_trait_table.values()):
      module_name = util.get_func_name(module_id)
      report_file = os.path.join(tmpdir, 'report', module_name + '_csynth.xml')
      hls_resource = hls_report.resources(report_file)
      use_count = len(nodes)
      try:
        perf = hls_report.performance(report_file)
        _logger.info('%s, usage: %5d times, II: %3d, Depth: %3d', hls_resource,
                    use_count, perf.ii, perf.depth)
        depths[module_id] = perf.depth
      except hls_report.BadReport as e:
        _logger.warn('%s in %s report (%s)', e, module_name, report_file)
        _logger.info('%s, usage: %5d times', hls_resource, use_count)
        raise e
      hls_resources += hls_resource * use_count
    _logger.info('total usage:')
    _logger.info(hls_resources)
    if rpt_file:
      rpt_json = collections.OrderedDict([('name', top_name)] +
                                         list(hls_resources))
      with open(rpt_file, mode='w') as rpt_fileobj:
        json.dump(rpt_json, rpt_fileobj, indent=2)

    # update the module pipeline depths
    stencil.dataflow_super_source.update_module_depths(depths)

    hdl_dir = os.path.join(tmpdir, 'hdl')
    with open(os.path.join(hdl_dir, 'Dataflow.v'), mode='w') as dataflow_v:
      print_top_module(backend.VerilogPrinter(dataflow_v),
                       stencil.dataflow_super_source, inputs, outputs)

    util.pause_for_debugging()

    xo_filename = os.path.join(tmpdir, stencil.app_name + '.xo')
    with backend.PackageXo(
        xo_filename,
        top_name,
        kernel_xml,
        hdl_dir,
        iface_names=(x[0] for x in inputs + outputs)) as proc:
      stdout, stderr = proc.communicate()
    log_func = _logger.error if proc.returncode != 0 else _logger.debug
    log_func(stdout.decode())
    log_func(stderr.decode())
    with open(xo_filename, mode='rb') as xo_fileobj:
      shutil.copyfileobj(xo_fileobj, xo_file)

def synthesis_module(tmpdir, kernel_files, module_name, device_info):
  """Synthesis a module in kernel files.

  Returns:
    (returncode, stdout, stderr) results of the subprocess.
  """
  job_server = util.acquire_job_slot()
  with tempfile.TemporaryFile(mode='w+b') as tarfileobj:
    with backend.RunHls(
        tarfileobj, kernel_files, module_name, device_info['clock_period'],
        device_info['part_num'], reset_low=False) as proc:
      stdout, stderr = proc.communicate()
    if proc.returncode == 0:
      tarfileobj.seek(0)
      with tarfile.open(mode='r', fileobj=tarfileobj) as tar:
        tar.extractall(tmpdir, (
            f for f in tar.getmembers()
            if f.name.startswith('hdl') or f.name.startswith('report')))
  util.release_job_slot(job_server)
  return proc.returncode, stdout, stderr

FIFO_PORT_SUFFIXES = dict(
    data_in='_din',
    not_full='_full_n',
    write_enable='_write',
    data_out='_dout',
    not_empty='_empty_n',
    read_enable='_read',
    not_block='_blk_n')

AXIS_PORT_SUFFIXES = collections.OrderedDict(
    data='_TDATA',  # producer -> consumer
    keep='_TKEEP',  # producer -> consumer
    strb='_TSTRB',  # producer -> consumer
    last='_TLAST',  # producer -> consumer
    valid='_TVALID',  # producer -> consumer
    ready='_TREADY',  # producer <- consumer
)

def print_kernel_xml(name: str, axis_inputs: Iterable[Tuple[str, str,
                                                                ir.Type, str]],
                     axis_outputs: Iterable[Tuple[str, str, ir.Type,
                                                  str]], kernel_xml: TextIO):
  """Generate kernel.xml file.

  Args:
    name: Name of the kernel.
    axis_inputs: Sequence of (port_name, _, haoda_type, _) of input axis ports
    axis_outputs: Sequence of (port_name, _, haoda_type, _) of output axis ports
    kernel_xml: File object to write to.
  """
  args: List[backend.Arg] = []
  for cat, axis_ports in ((backend.Cat.ISTREAM, axis_inputs),
                          (backend.Cat.OSTREAM, axis_outputs)):
    for port_name, _, haoda_type, _ in axis_ports:
      ctype = f'stream<ap_axiu<{haoda_type.width_in_bits}, 0, 0, 0>>&'
      width = 8 + haoda_type.width_in_bits // 8 * 2
      args.append(backend.Arg(cat=cat, name=port_name, port=port_name,
                             ctype=ctype, width=width))
  backend.print_kernel_xml(name=name, args=args, kernel_xml=kernel_xml)

def print_top_module(printer, super_source, inputs, outputs):
  println = printer.println
  println('`timescale 1 ns / 1 ps')

  input_args = 'ap_clk', 'ap_rst'
  output_args = ()
  args = list(input_args + output_args)
  for port_name, _, _, _ in inputs + outputs:
    args.extend(port_name + suffix for suffix in AXIS_PORT_SUFFIXES.values())
  printer.module('Dataflow', args)
  println()

  for arg in input_args:
    println('input  %s;' % arg)
  for arg in output_args:
    println('output %s;' % arg)
  for port_name, _, haoda_type, _ in outputs:
    width = haoda_type.width_in_bits
    kwargs = dict(port_name=port_name,
                  **FIFO_PORT_SUFFIXES,
                  **AXIS_PORT_SUFFIXES)
    println('output [{}:0] {port_name}{data};'.format(width - 1, **kwargs))
    println('output [{}:0] {port_name}{keep};'.format(width // 8 - 1, **kwargs))
    println('output [{}:0] {port_name}{strb};'.format(width // 8 - 1, **kwargs))
    println('output {port_name}{last};'.format(**kwargs))
    println('output {port_name}{valid};'.format(**kwargs))
    println('input {port_name}{ready};'.format(**kwargs))
    println('wire {port_name}{data_in};'.format(**kwargs))
    println('wire {port_name}{not_full};'.format(**kwargs))
    println('wire {port_name}{write_enable};'.format(**kwargs))
  for port_name, _, haoda_type, _ in inputs:
    width = haoda_type.width_in_bits
    kwargs = dict(port_name=port_name,
                  **FIFO_PORT_SUFFIXES,
                  **AXIS_PORT_SUFFIXES)
    println('input [{}:0] {port_name}{data};'.format(width - 1, **kwargs))
    println('input [{}:0] {port_name}{keep};'.format(width // 8 - 1, **kwargs))
    println('input [{}:0] {port_name}{strb};'.format(width // 8 - 1, **kwargs))
    println('input {port_name}{last};'.format(**kwargs))
    println('input {port_name}{valid};'.format(**kwargs))
    println('output {port_name}{ready};'.format(**kwargs))
    println('wire {port_name}{data_out};'.format(**kwargs))
    println('wire {port_name}{not_empty};'.format(**kwargs))
    println('wire {port_name}{read_enable};'.format(**kwargs))
  println()

  fifos = set()
  for port_name, _, haoda_type, _ in inputs + outputs:
    width = haoda_type.width_in_bits
    kwargs = dict(port_name=port_name, **AXIS_PORT_SUFFIXES)
    println('wire [{}:0] {port_name}{data};'.format(width - 1, **kwargs))
    println('wire [{}:0] {port_name}{keep};'.format(width // 8 - 1, **kwargs))
    println('wire [{}:0] {port_name}{strb};'.format(width // 8 - 1, **kwargs))
    println('wire {port_name}{last};'.format(**kwargs))
    println('wire {port_name}{valid};'.format(**kwargs))
    println('wire {port_name}{ready};'.format(**kwargs))
    fifos.add((width, 2))
  println()

  ap_rst_reg_level = 16
  println('wire ap_rst_reg_0 = ap_rst;')
  for i in range(ap_rst_reg_level):
    println('(* shreg_extract = "no"%s *) reg ap_rst_reg_%d;' %
            (', max_fanout = %d' %
             4 ** (i + 1) if i + 1 < ap_rst_reg_level else '', i + 1))
  if ap_rst_reg_level > 0:
    with printer.initial():
      for i in range(ap_rst_reg_level):
        println("#0 ap_rst_reg_%d = 1'b1;" % (i + 1))
    with printer.always('posedge ap_clk'):
      for i in range(ap_rst_reg_level):
        println('ap_rst_reg_%d <= ap_rst_reg_%d;' % (i + 1, i))
  println('wire ap_rst_reg = ap_rst_reg_%d;' % ap_rst_reg_level)

  for port_name, _, haoda_type, _ in outputs:
    width = haoda_type.width_in_bits
    kwargs = dict(port_name=port_name,
                  width=width // 8,
                  ones='1' * (width // 8),
                  **AXIS_PORT_SUFFIXES)
    println("assign {port_name}{keep} = {width}'b{ones};".format(**kwargs))
    println("assign {port_name}{strb} = {width}'b{ones};".format(**kwargs))
    println("assign {port_name}{last} = 1'b0;".format(**kwargs))

  for port_name, _, haoda_type, _ in inputs:
    width = haoda_type.width_in_bits
    kwargs = dict(name=port_name, **FIFO_PORT_SUFFIXES, **AXIS_PORT_SUFFIXES)
    args = collections.OrderedDict((
        ('clk', 'ap_clk'),
        ('reset', 'ap_rst_reg'),
        ('if_read_ce', "1'b1"),
        ('if_write_ce', "1'b1"),
        ('if{data_in}'.format(**kwargs),
          '{name}{data}'.format(**kwargs)),
        ('if{not_full}'.format(**kwargs),
          '{name}{ready}'.format(**kwargs)),
        ('if{write_enable}'.format(**kwargs),
          '{name}{valid}'.format(**kwargs)),
        ('if{data_out}'.format(**kwargs),
          '{name}{data_out}'.format(**kwargs)),
        ('if{not_empty}'.format(**kwargs),
          '{name}{not_empty}'.format(**kwargs)),
        ('if{read_enable}'.format(**kwargs),
          '{name}{read_enable}'.format(**kwargs))
    ))
    printer.module_instance('fifo_w{width}_d{depth}_A'.format(
        width=width, depth=2), port_name + '_fifo', args)
    println()


  for port_name, _, haoda_type, _ in outputs:
    width = haoda_type.width_in_bits
    kwargs = dict(name=port_name, **FIFO_PORT_SUFFIXES, **AXIS_PORT_SUFFIXES)
    args = collections.OrderedDict((
        ('clk', 'ap_clk'),
        ('reset', 'ap_rst_reg'),
        ('if_read_ce', "1'b1"),
        ('if_write_ce', "1'b1"),
        ('if{data_in}'.format(**kwargs),
          '{name}{data_in}'.format(**kwargs)),
        ('if{not_full}'.format(**kwargs),
          '{name}{not_full}'.format(**kwargs)),
        ('if{write_enable}'.format(**kwargs),
          '{name}{write_enable}'.format(**kwargs)),
        ('if{data_out}'.format(**kwargs),
          '{name}{data}'.format(**kwargs)),
        ('if{not_empty}'.format(**kwargs),
          '{name}{valid}'.format(**kwargs)),
        ('if{read_enable}'.format(**kwargs),
          '{name}{ready}'.format(**kwargs))
    ))
    printer.module_instance('fifo_w{width}_d{depth}_A'.format(
        width=width, depth=2), port_name + '_fifo', args)
    println()

  for module in super_source.tpo_valid_node_gen():
    for fifo in module.fifos:
      kwargs = {
          'name' : fifo.c_expr,
          'msb' : fifo.width_in_bits - 1,
          **FIFO_PORT_SUFFIXES
      }
      println('wire [{msb}:0] {name}{data_in};'.format(**kwargs))
      println('wire {name}{not_full};'.format(**kwargs))
      println('wire {name}{write_enable};'.format(**kwargs))
      println('wire [{msb}:0] {name}{data_out};'.format(**kwargs))
      println('wire {name}{not_empty};'.format(**kwargs))
      println('wire {name}{read_enable};'.format(**kwargs))
      println()

      args = collections.OrderedDict((
          ('clk', 'ap_clk'),
          ('reset', 'ap_rst_reg'),
          ('if_read_ce', "1'b1"),
          ('if_write_ce', "1'b1"),
          ('if{data_in}'.format(**kwargs),
           '{name}{data_in}'.format(**kwargs)),
          ('if{not_full}'.format(**kwargs),
           '{name}{not_full}'.format(**kwargs)),
          ('if{write_enable}'.format(**kwargs),
           '{name}{write_enable}'.format(**kwargs)),
          ('if{data_out}'.format(**kwargs),
           '{name}{data_out}'.format(**kwargs)),
          ('if{not_empty}'.format(**kwargs),
           '{name}{not_empty}'.format(**kwargs)),
          ('if{read_enable}'.format(**kwargs),
           '{name}{read_enable}'.format(**kwargs))
      ))
      printer.module_instance('fifo_w{width}_d{depth}_A'.format(
          width=fifo.width_in_bits, depth=fifo.depth+2), fifo.c_expr, args)
      println()

  for module in super_source.tpo_valid_node_gen():
    module_trait, module_trait_id = super_source.module_table[module]
    args = collections.OrderedDict(
        (('ap_clk', 'ap_clk'), ('ap_rst', 'ap_rst_reg'), ('ap_start', "1'b1")))
    for dram_ref, bank in module.dram_writes:
      kwargs = dict(port=dram_ref.dram_fifo_name(bank),
                    fifo=util.get_port_name(dram_ref.var, bank),
                    **FIFO_PORT_SUFFIXES,
                    **AXIS_PORT_SUFFIXES)
      args['{port}_V{data_in}'.format(**kwargs)] = \
                       '{fifo}{data_in}'.format(**kwargs)
      args['{port}_V{not_full}'.format(**kwargs)] = \
                       '{fifo}{not_full}'.format(**kwargs)
      args['{port}_V{write_enable}'.format(**kwargs)] = \
                       '{fifo}{write_enable}'.format(**kwargs)
    for port, fifo in zip(module_trait.output_fifos, module.output_fifos):
      kwargs = dict(port=port, fifo=fifo, **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_in}'.format(**kwargs)] = \
                       '{fifo}{data_in}'.format(**kwargs)
      args['{port}_V{not_full}'.format(**kwargs)] = \
                       '{fifo}{not_full}'.format(**kwargs)
      args['{port}_V{write_enable}'.format(**kwargs)] = \
                       '{fifo}{write_enable}'.format(**kwargs)
    for port, fifo in zip(module_trait.input_fifos, module.input_fifos):
      kwargs = dict(port=port,
                    fifo=fifo,
                    **FIFO_PORT_SUFFIXES)
      args['{port}_V{data_out}'.format(**kwargs)] = \
                       "{{1'b1, {fifo}{data_out}}}".format(**kwargs)
      args['{port}_V{not_empty}'.format(**kwargs)] = \
                       '{fifo}{not_empty}'.format(**kwargs)
      args['{port}_V{read_enable}'.format(**kwargs)] = \
                       '{fifo}{read_enable}'.format(**kwargs)
    for dram_ref, bank in module.dram_reads:
      kwargs = dict(port=dram_ref.dram_fifo_name(bank),
                    fifo=util.get_port_name(dram_ref.var, bank),
                    **FIFO_PORT_SUFFIXES,
                    **AXIS_PORT_SUFFIXES)
      args['{port}_V{data_out}'.format(**kwargs)] = \
                       "{{1'b1, {fifo}{data_out}}}".format(**kwargs)
      args['{port}_V{not_empty}'.format(**kwargs)] = \
                       '{fifo}{not_empty}'.format(**kwargs)
      args['{port}_V{read_enable}'.format(**kwargs)] = \
                       '{fifo}{read_enable}'.format(**kwargs)
    printer.module_instance(util.get_func_name(module_trait_id), module.name,
                            args)
    println()
  printer.endmodule()

  for module in super_source.tpo_valid_node_gen():
    for fifo in module.fifos:
      fifos.add((fifo.width_in_bits, fifo.depth + 2))
  for fifo in fifos:
    printer.fifo_module(*fifo)
