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
from typing import (IO, Dict, Iterable, List, Optional, Sequence, Set, TextIO,
                    Tuple)

from haoda import util
from haoda.backend import xilinx as backend
from haoda.report.xilinx import hls as hls_report
from soda import core, dataflow
from soda.codegen.xilinx import hls_kernel

_logger = logging.getLogger().getChild(__name__)


def print_code(
    stencil: core.Stencil,
    xo_file: IO[bytes],
    device_info: Dict[str, str],
    jobs: Optional[int] = os.cpu_count(),
    rpt_file: Optional[str] = None,
    interface: str = 'm_axi',
) -> None:
  """Generate hardware object file for the given Stencil.

  Working `vivado` and `vivado_hls` is required in the PATH.

  Args:
    stencil: Stencil object to generate from.
    xo_file: file object to write to.
    device_info: dict of 'part_num' and 'clock_period'.
    jobs: maximum number of jobs running in parallel.
    rpt_file: path of the generated report; None disables report generation.
    interface: interface type, supported values are 'm_axi' and 'axis'.
  """

  iface_names = []  # for axis
  m_axi_names = []  # for m_axi
  inputs = []
  outputs = []

  for stmt in stencil.output_stmts:
    for bank in stmt.dram:
      port_name = util.get_port_name(stmt.name, bank)
      bundle_name = util.get_bundle_name(stmt.name, bank)
      iface_names.append(port_name)
      m_axi_names.append(bundle_name)
      outputs.append((port_name, bundle_name, stencil.burst_width,
                      util.get_port_buf_name(stmt.name, bank)))
  for stmt in stencil.input_stmts:
    for bank in stmt.dram:
      port_name = util.get_port_name(stmt.name, bank)
      bundle_name = util.get_bundle_name(stmt.name, bank)
      iface_names.append(port_name)
      m_axi_names.append(bundle_name)
      inputs.append((port_name, bundle_name, stencil.burst_width,
                     util.get_port_buf_name(stmt.name, bank)))

  top_name = stencil.kernel_name
  with tempfile.TemporaryDirectory(prefix='sodac-xrtl-') as tmpdir:
    kernel_xml = os.path.join(tmpdir, 'kernel.xml')
    with open(kernel_xml, 'w') as kernel_xml_obj:
      print_kernel_xml(top_name, inputs, outputs, kernel_xml_obj, interface)

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

    if interface == 'm_axi':
      sio = io.StringIO()
      print_dataflow_hls_interface(util.CppPrinter(sio), top_name, inputs,
                                   outputs)
      dataflow_kernel = os.path.join(tmpdir, 'dataflow_kernel.cpp')
      with open(dataflow_kernel, 'w') as dataflow_kernel_obj:
        dataflow_kernel_obj.write(sio.getvalue())
      args.append((len(sio.getvalue()), synthesis_module, tmpdir,
                   [dataflow_kernel], top_name, device_info))
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
    if interface == 'm_axi':
      hls_resources = hls_report.resources(
          os.path.join(tmpdir, 'report', top_name + '_csynth.xml'))
      hls_resources -= hls_report.resources(
          os.path.join(tmpdir, 'report', 'Dataflow_csynth.xml'))

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
    module_name = 'Dataflow'
    if interface == 'axis':
      module_name = top_name
    with open(os.path.join(hdl_dir, f'{module_name}.v'), mode='w') as fileobj:
      print_top_module(
          backend.VerilogPrinter(fileobj),
          stencil.dataflow_super_source,
          inputs,
          outputs,
          module_name,
          interface,
      )

    util.pause_for_debugging()

    xo_filename = os.path.join(tmpdir, stencil.app_name + '.xo')
    kwargs = {}
    if interface == 'm_axi':
      kwargs['m_axi_names'] = m_axi_names
    elif interface == 'axis':
      kwargs['iface_names'] = iface_names
    with backend.PackageXo(
        xo_filename,
        top_name,
        kernel_xml,
        hdl_dir,
        **kwargs,
    ) as proc:
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
    with backend.RunHls(tarfileobj,
                        kernel_files,
                        module_name,
                        device_info['clock_period'],
                        device_info['part_num'],
                        reset_low=False) as proc:
      stdout, stderr = proc.communicate()
    if proc.returncode == 0:
      tarfileobj.seek(0)
      with tarfile.open(mode='r', fileobj=tarfileobj) as tar:
        tar.extractall(
            tmpdir,
            (f for f in tar.getmembers()
             if f.name.startswith('hdl') or f.name.startswith('report')))
  util.release_job_slot(job_server)
  return proc.returncode, stdout, stderr


FIFO_PORT_SUFFIXES = dict(data_in='_din',
                          not_full='_full_n',
                          write_enable='_write',
                          data_out='_dout',
                          not_empty='_empty_n',
                          read_enable='_read',
                          not_block='_blk_n')

AXIS_PORT_SUFFIXES = collections.OrderedDict(
    data='_TDATA',  # producer -> consumer
    valid='_TVALID',  # producer -> consumer
    ready='_TREADY',  # producer <- consumer
)


def print_kernel_xml(
    name: str,
    inputs: Iterable[Tuple[str, str, int, str]],
    outputs: Iterable[Tuple[str, str, int, str]],
    kernel_xml: TextIO,
    interface: str = 'm_axi',
) -> None:
  """Generate kernel.xml file.

  Args:
    name: Name of the kernel.
    inputs: Iterable of (port_name, bundle_name, width, _) of input ports.
    outputs: Iterable of (port_name, bundle_name, width, _) of output ports.
    kernel_xml: File object to write to.
    interface: Interface type, supported values are 'm_axi' and 'axis'.
  """
  args: List[backend.Arg] = []
  if interface == 'm_axi':
    for ports in outputs, inputs:
      for port_name, bundle_name, width, _ in ports:
        args.append(
            backend.Arg(
                cat=backend.Cat.MMAP,
                name=port_name,
                port=bundle_name,
                ctype=f'ap_uint<{width}>*',
                width=width,
            ))
    args.append(
        backend.Arg(
            cat=backend.Cat.SCALAR,
            name='coalesced_data_num',
            port='',
            ctype='uint64_t',
            width=64,
        ))
  elif interface == 'axis':
    for cat, ports in ((backend.Cat.ISTREAM, inputs), (backend.Cat.OSTREAM,
                                                       outputs)):
      for port_name, _, width, _ in ports:
        ctype = f'stream<ap_axiu<{width}, 0, 0, 0>>&'
        args.append(
            backend.Arg(
                cat=cat,
                name=port_name,
                port='',
                ctype=ctype,
                width=width,
            ))
  else:
    raise util.InternalError(f'unexpected interface `{interface}`')

  backend.print_kernel_xml(name=name, args=args, kernel_xml=kernel_xml)


def print_top_module(
    printer: backend.VerilogPrinter,
    super_source: dataflow.SuperSourceNode,
    inputs: Sequence[Tuple[str, str, int, str]],
    outputs: Sequence[Tuple[str, str, int, str]],
    module_name: str = 'Dataflow',
    interface: str = 'm_axi',
) -> None:
  """Generate kernel.xml file.

  Args:
    printer: printer to print to
    super_source: SuperSourceNode carrying the IR tree.
    inputs: sequence of (port_name, bundle_name, width, _) of input ports
    outputs: sequence of (port_name, bundle_name, width, _) of output ports
    module_name: name of the module
    interface: interface type, supported values are 'm_axi' and 'axis'
  """
  printer.printlns('`timescale 1 ns / 1 ps', '`default_nettype none')

  ports = *inputs, *outputs

  # unpack suffixes
  data_in = FIFO_PORT_SUFFIXES['data_in']
  not_full = FIFO_PORT_SUFFIXES['not_full']
  write_enable = FIFO_PORT_SUFFIXES['write_enable']
  data_out = FIFO_PORT_SUFFIXES['data_out']
  not_empty = FIFO_PORT_SUFFIXES['not_empty']
  read_enable = FIFO_PORT_SUFFIXES['read_enable']
  not_block = FIFO_PORT_SUFFIXES['not_block']
  data = AXIS_PORT_SUFFIXES['data']
  valid = AXIS_PORT_SUFFIXES['valid']
  ready = AXIS_PORT_SUFFIXES['ready']

  # prepare arguments
  input_args = ['ap_clk']
  output_args: List[str] = []
  if interface == 'm_axi':
    input_args += 'ap_rst', 'ap_start', 'ap_continue'
    output_args += 'ap_done', 'ap_idle', 'ap_ready'
  elif interface == 'axis':
    input_args.append('ap_rst_n')

  args = list(input_args + output_args)
  if interface == 'm_axi':
    for port_name, _, _, _ in outputs:
      args.append(f'{port_name}_V_V{data_in}')
      args.append(f'{port_name}_V_V{not_full}')
      args.append(f'{port_name}_V_V{write_enable}')
    for port_name, _, _, _ in inputs:
      args.append(f'{port_name}_V_V{data_out}')
      args.append(f'{port_name}_V_V{not_empty}')
      args.append(f'{port_name}_V_V{read_enable}')
  elif interface == 'axis':
    for port_name, _, _, _ in ports:
      args.extend(port_name + suffix for suffix in AXIS_PORT_SUFFIXES.values())

  # print module interface
  printer.module(module_name, args)
  printer.println()

  # print signals for modules
  printer.printlns(
      *(f'input  wire {arg};' for arg in input_args),
      *(f'output wire {arg};' for arg in output_args),
  )
  for port_name, _, width, _ in outputs:
    if interface == 'm_axi':
      printer.printlns(
          f'output wire [{width - 1}:0] {port_name}_V_V{data_in};',
          f'input  wire {port_name}_V_V{not_full};',
          f'output wire {port_name}_V_V{write_enable};',
      )
    elif interface == 'axis':
      printer.printlns(
          f'output wire [{width - 1}:0] {port_name}{data};',
          f'output wire {port_name}{valid};',
          f'input  wire {port_name}{ready};',
          f'wire [{width - 1}:0] {port_name}_V_V{data_in};',
          f'wire {port_name}_V_V{not_full};',
          f'wire {port_name}_V_V{write_enable};',
      )
  for port_name, _, width, _ in inputs:
    if interface == 'm_axi':
      printer.printlns(
          f'input  wire [{width - 1}:0] {port_name}_V_V{data_out};',
          f'input  wire {port_name}_V_V{not_empty};',
          f'output wire {port_name}_V_V{read_enable};',
      )
    elif interface == 'axis':
      printer.printlns(
          f'input  wire [{width - 1}:0] {port_name}{data};',
          f'input  wire {port_name}{valid};',
          f'output wire {port_name}{ready};',
          f'wire [{width - 1}:0] {port_name}_V_V{data_out};',
          f'wire {port_name}_V_V{not_empty};',
          f'wire {port_name}_V_V{read_enable};',
      )
  printer.println()

  # not used
  printer.printlns(
      "reg ap_done = 1'b0;",
      "reg ap_idle = 1'b1;",
      "reg ap_ready = 1'b0;",
  )
  if interface == 'axis':
    printer.println("wire ap_start = 1'b1;")

  # print signals for FIFOs
  if interface == 'm_axi':
    for port_name, _, width, _ in outputs:
      printer.printlns(
          f'reg [{width - 1}:0] {port_name}{data_in};',
          f'wire {port_name}_V_V{write_enable};',
      )
    for port_name, _, _, _ in inputs:
      printer.println(f'wire {port_name}_V_V{read_enable};')
  printer.println()

  # register reset signal
  ap_rst_reg_level = 8
  rst = 'ap_rst'
  if interface == 'axis':
    rst = '~ap_rst_n'
  printer.printlns(
      f'wire ap_rst_reg_0 = {rst};',
      *(f'(* shreg_extract = "no", max_fanout = {8 ** i} *) reg ap_rst_reg_{i};'
        for i in range(1, ap_rst_reg_level)),
      f'(* shreg_extract = "no" *) reg ap_rst_reg_{ap_rst_reg_level};',
      f'wire ap_rst_reg = ap_rst_reg_{ap_rst_reg_level};',
  )
  if ap_rst_reg_level > 0:
    with printer.always('posedge ap_clk'):
      printer.printlns(f'ap_rst_reg_{i + 1} <= ap_rst_reg_{i};'
                       for i in range(ap_rst_reg_level))

  with printer.always('posedge ap_clk'):
    with printer.if_('ap_rst_reg'):
      printer.printlns(
          "ap_done <= 1'b0;",
          "ap_idle <= 1'b1;",
          "ap_ready <= 1'b0;",
      )
      printer.else_()
      printer.println('ap_idle <= ~ap_start;')
  printer.println()

  if interface == 'm_axi':
    # used by cosim for deadlock detection
    printer.printlns(
        f'reg {port_name}_V_V{not_block};' for port_name, _, _, _ in ports)
    with printer.always('*'):
      printer.printlns(
          *(f'{port_name}_V_V{not_block} = {port_name}_V_V{not_full};'
            for port_name, _, _, _ in outputs),
          *(f'{port_name}_V_V{not_block} = {port_name}_V_V{not_empty};'
            for port_name, _, _, _ in inputs),
      )
  printer.println()

  fifos: Set[Tuple[int, int]] = set()  # used for printing FIFO modules

  if interface == 'axis':
    for port_name, _, width, _ in inputs:
      printer.module_instance(
          'fifo_w{width}_d{depth}_A'.format(width=width, depth=2),
          port_name + '_fifo',
          args={
              'clk': 'ap_clk',
              'reset': 'ap_rst_reg',
              'if_read_ce': "1'b1",
              'if_write_ce': "1'b1",
              f'if{data_in}': f'{port_name}{data}',
              f'if{not_full}': f'{port_name}{ready}',
              f'if{write_enable}': f'{port_name}{valid}',
              f'if{data_out}': f'{port_name}_V_V{data_out}',
              f'if{not_empty}': f'{port_name}_V_V{not_empty}',
              f'if{read_enable}': f'{port_name}_V_V{read_enable}',
          },
      )
      fifos.add((width, 2))
      printer.println()

    for port_name, _, width, _ in outputs:
      printer.module_instance(
          f'fifo_w{width}_d2_A',
          port_name + '_fifo',
          args={
              'clk': 'ap_clk',
              'reset': 'ap_rst_reg',
              'if_read_ce': "1'b1",
              'if_write_ce': "1'b1",
              f'if{data_in}': f'{port_name}_V_V{data_in}',
              f'if{not_full}': f'{port_name}_V_V{not_full}',
              f'if{write_enable}': f'{port_name}_V_V{write_enable}',
              f'if{data_out}': f'{port_name}{data}',
              f'if{not_empty}': f'{port_name}{valid}',
              f'if{read_enable}': f'{port_name}{ready}',
          },
      )
      fifos.add((width, 2))
      printer.println()

  # print FIFO instances
  for module in super_source.tpo_valid_node_gen():
    for fifo in module.fifos:
      name = fifo.c_expr
      msb = fifo.width_in_bits - 1
      # fifo.depth is the "extra" capacity of a FIFO; the base depth is 3, 1 for
      # registering the input, 1 for keeping II=1 when FIFO is (almost) full, 1
      # for keeping II=1 when FIFO is relaxed from back pressure (necessary
      # because the optimal FIFO depths may require back pressure)
      depth = fifo.depth + 3
      printer.printlns(
          f'wire [{msb}:0] {name}{data_in};',
          f'wire {name}{not_full};',
          f'wire {name}{write_enable};',
          f'wire [{msb}:0] {name}{data_out};',
          f'wire {name}{not_empty};',
          f'wire {name}{read_enable};',
          '',
      )
      printer.module_instance(
          f'fifo_w{fifo.width_in_bits}_d{depth}_A',
          name,
          args={
              'clk': 'ap_clk',
              'reset': 'ap_rst_reg',
              'if_read_ce': "1'b1",
              'if_write_ce': "1'b1",
              f'if{data_in}': f'{name}{data_in}',
              f'if{not_full}': f'{name}{not_full}',
              f'if{write_enable}': f'{name}{write_enable}',
              f'if{data_out}': f'{name}{data_out}',
              f'if{not_empty}': f'{name}{not_empty}',
              f'if{read_enable}': f'{name}{read_enable}',
          },
      )
      fifos.add((fifo.width_in_bits, depth))
      printer.println()

  # print module instances
  for module in super_source.tpo_valid_node_gen():
    module_trait, module_trait_id = super_source.module_table[module]
    arg_dict = {
        'ap_clk': 'ap_clk',
        'ap_rst': 'ap_rst_reg',
        'ap_start': "1'b1",
    }
    for dram_ref, bank in module.dram_writes:
      port = dram_ref.dram_fifo_name(bank)
      fifo = util.get_port_name(dram_ref.var, bank)
      arg_dict.update({
          f'{port}_V{data_in}': f'{fifo}_V_V{data_in}',
          f'{port}_V{not_full}': f'{fifo}_V_V{not_full}',
          f'{port}_V{write_enable}': f'{fifo}_V_V{write_enable}',
      })
    for port, fifo in zip(module_trait.output_fifos, module.output_fifos):
      arg_dict.update({
          f'{port}_V{data_in}': f'{fifo}{data_in}',
          f'{port}_V{not_full}': f'{fifo}{not_full}',
          f'{port}_V{write_enable}': f'{fifo}{write_enable}',
      })
    for port, fifo in zip(module_trait.input_fifos, module.input_fifos):
      arg_dict.update({
          f'{port}_V{data_out}': f"{{1'b1, {fifo}{data_out}}}",
          f'{port}_V{not_empty}': f'{fifo}{not_empty}',
          f'{port}_V{read_enable}': f'{fifo}{read_enable}',
      })
    for dram_ref, bank in module.dram_reads:
      port = dram_ref.dram_fifo_name(bank)
      fifo = util.get_port_name(dram_ref.var, bank)
      arg_dict.update({
          f'{port}_V{data_out}': f"{{1'b1, {fifo}_V_V{data_out}}}",
          f'{port}_V{not_empty}': f'{fifo}_V_V{not_empty}',
          f'{port}_V{read_enable}': f'{fifo}_V_V{read_enable}',
      })
    printer.module_instance(util.get_func_name(module_trait_id), module.name,
                            arg_dict)
    printer.println()
  printer.endmodule()
  printer.println('`default_nettype wire')

  # print FIFO modules
  for fifo in fifos:
    printer.fifo_module(*fifo)


def print_dataflow_hls_interface(
    printer: backend.util.CppPrinter,
    top_name: str,
    inputs: Sequence[Tuple[str, str, int, str]],
    outputs: Sequence[Tuple[str, str, int, str]],
) -> None:
  ports = *outputs, *inputs

  printer.printlns(
      '#include <cstddef>',
      '#include <cstdint>',
      '#include <ap_int.h>',
      '#include <hls_stream.h>',
  )

  printer.println('template<typename T>')
  printer.print_func('void BurstRead',
                     ['hls::stream<T>& to', 'T* from', 'uint64_t data_num'],
                     align=0)
  printer.do_scope()
  printer.println('burst_read:', 0)
  with printer.for_('uint64_t epoch = 0', 'epoch < data_num', '++epoch'):
    printer.println('#pragma HLS pipeline II=1', 0)
    printer.println('to.write(from[epoch]);')
  printer.un_scope()

  printer.println('template<typename T>')
  printer.print_func('void BurstWrite',
                     ['T* to', 'hls::stream<T>& from', 'uint64_t data_num'],
                     align=0)
  printer.do_scope()
  printer.println('burst_write:', 0)
  with printer.for_('uint64_t epoch = 0', 'epoch < data_num', '++epoch'):
    printer.println('#pragma HLS pipeline II=1', 0)
    printer.println('to[epoch] = from.read();')
  printer.un_scope()

  params = [
      f'hls::stream<ap_uint<{width}>>& {name}' for name, _, width, _ in ports
  ]
  printer.print_func('void Dataflow', params, align=0)
  printer.do_scope()
  printer.printlns(
      *(f'volatile ap_uint<{width}> {name}_read = {name}.read();'
        for name, _, width, _ in inputs),
      *(f'{name}.write(ap_uint<{width}>());' for name, _, width, _ in outputs),
  )
  printer.un_scope()

  params = [f'ap_uint<{width}>* {name}' for name, _, width, _ in ports]
  params.append('uint64_t coalesced_data_num')
  printer.print_func('void %s' % top_name, params, align=0)
  printer.do_scope()
  printer.println('#pragma HLS dataflow', 0)
  for port_name, bundle_name, _, _ in ports:
    printer.println(
        f'#pragma HLS interface m_axi port={port_name} '
        f'offset=slave bundle={bundle_name}', 0)
  for port_name, _, _, _ in ports:
    printer.println(
        f'#pragma HLS interface s_axilite port={port_name} '
        'bundle=control', 0)
  printer.println(
      '#pragma HLS interface s_axilite port=coalesced_data_num '
      'bundle=control', 0)
  printer.println(
      '#pragma HLS interface s_axilite port=return '
      'bundle=control', 0)
  printer.println()
  for _, _, width, name in ports:
    printer.println(f'hls::stream<ap_uint<{width}>> {name}("{name}");')
    printer.println(f'#pragma HLS stream variable={name} depth=32', 0)
  for port_name, _, _, buf_name in inputs:
    printer.print_func(
        'BurstRead',
        [buf_name, port_name, 'coalesced_data_num'],
        suffix=';',
        align=0,
    )
  printer.print_func(
      'Dataflow',
      [name for _, _, _, name in ports],
      suffix=';',
      align=0,
  )
  for port_name, _, _, buf_name in outputs:
    printer.print_func(
        'BurstWrite',
        [port_name, buf_name, 'coalesced_data_num'],
        suffix=';',
        align=0,
    )
  printer.un_scope()
