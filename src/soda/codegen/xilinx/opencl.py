import argparse
import os
import shutil
import sys
import tempfile

from haoda.backend import xilinx as backend
from soda import core
from soda.codegen.xilinx import header
from soda.codegen.xilinx import hls_kernel as kernel
from soda.codegen.xilinx import host, rtl_kernel

def add_arguments(parser):
  parser.add_argument(
      '--xocl', type=str, dest='output_dir', metavar='dir', nargs='?', const='',
      help='directory to generate kernel and host code; default names are'
      'used; default to the current working directory; may be overridden by '
      '--xocl-header, --xocl-host, or --xocl-kernel')
  parser.add_argument(
      '--xocl-header', type=str, dest='header_file', metavar='file',
      help='host C++ header code; overrides --xocl')
  parser.add_argument(
      '--xocl-host', type=str, dest='host_file', metavar='file',
      help='host C++ source code for the Xilinx OpenCL flow; overrides --xocl')
  parser.add_argument(
      '--xocl-kernel', type=str, dest='kernel_file', metavar='file',
      help='Vivado HLS C++ kernel code for the Xilinx OpenCL flow; overrides '
      '--xocl')
  parser.add_argument(
      '--xocl-platform', type=str, dest='xocl_platform', metavar='dir',
      help='SDAccel platform directory of the Xilinx OpenCL flow')
  parser.add_argument(
      '--xocl-part-num', type=str, dest='xocl_part_num', metavar='string',
      help='part number used for Vivado HLS')
  parser.add_argument(
      '--xocl-clock-period', type=str, dest='xocl_clock_period', metavar='ns',
      help='target clock period in nanoseconds, used for Vivado HLS')
  parser.add_argument('--xocl-hw-xo', type=str, dest='xo_file', metavar='file',
                      help='hardware object file for the Xilinx OpenCL flow')
  parser.add_argument(
      '--xocl-hw-xo-rpt', type=str, dest='xo_rpt', metavar='file',
      help='Vivado HLS report for the Xilinx OpenCL hardware object')
  parser.add_argument(
    '--xocl-interface', type=str, dest='interface', metavar='(m_axi|axis)',
    default='m_axi', help='interface type of the Xilinx OpenCL code')

def print_code(
    stencil: core.Stencil,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
  if args.kernel_file is not None:
    with tempfile.TemporaryFile(mode='w+') as tmp:
      kernel.print_code(stencil, tmp, interface=args.interface)
      tmp.seek(0)
      if args.kernel_file == '-':
        shutil.copyfileobj(tmp, sys.stdout)
      else:
        with open(args.kernel_file, 'w') as kernel_file:
          shutil.copyfileobj(tmp, kernel_file)

  if args.host_file is not None:
    with tempfile.TemporaryFile(mode='w+') as tmp:
      host.print_code(stencil, tmp)
      tmp.seek(0)
      if args.host_file == '-':
        shutil.copyfileobj(tmp, sys.stdout)
      else:
        with open(args.host_file, 'w') as host_file:
          shutil.copyfileobj(tmp, host_file)

  if args.header_file is not None:
    with tempfile.TemporaryFile(mode='w+') as tmp:
      header.print_code(stencil, tmp)
      tmp.seek(0)
      if args.header_file == '-':
        shutil.copyfileobj(tmp, sys.stdout)
      else:
        with open(args.header_file, 'w') as header_file:
          shutil.copyfileobj(tmp, header_file)

  if args.xo_file is not None:
    with tempfile.TemporaryFile(mode='w+b') as tmp_obj:
      rtl_kernel.print_code(
          stencil,
          tmp_obj,
          device_info=backend.parse_device_info(
              parser,
              args,
              platform_name='xocl_platform',
              part_num_name='xocl_part_num',
              clock_period_name='xocl_clock_period',
          ),
          rpt_file=args.xo_rpt,
          interface=args.interface,
      )
      tmp_obj.seek(0)
      if args.xo_file == '-':
        shutil.copyfileobj(tmp_obj, sys.stdout)   # type: ignore
      else:
        with open(args.xo_file, 'wb') as xo_file:
          shutil.copyfileobj(tmp_obj, xo_file)

  if args.output_dir is not None and (args.kernel_file is None or
                                      args.host_file is None or
                                      args.header_file is None):
    if args.kernel_file is None:
      dram_in = args.dram_in if args.dram_in else '_'
      dram_out = args.dram_out if args.dram_out else '_'
      kernel_file_name = os.path.join(
          args.output_dir, '%s_kernel-tile%s-unroll%d-ddr%s.cpp' % (
              stencil.app_name,
              'x'.join('%d'%x for x in stencil.tile_size[:-1]),
              stencil.unroll_factor, dram_in + '-' + dram_out))
    else:
      kernel_file_name = args.kernel_file
    with tempfile.TemporaryFile(mode='w+') as tmp:
      kernel.print_code(stencil, tmp, interface=args.interface)
      tmp.seek(0)
      with open(kernel_file_name, 'w') as kernel_file:
        shutil.copyfileobj(tmp, kernel_file)
    if args.host_file is None:
      host_file_name = os.path.join(args.output_dir, stencil.app_name + '.cpp')
    else:
      host_file_name = args.host_file
    with tempfile.TemporaryFile(mode='w+') as tmp:
      host.print_code(stencil, tmp)
      tmp.seek(0)
      with open(host_file_name, 'w') as host_file:
        shutil.copyfileobj(tmp, kernel_file)
    if args.header_file is None:
      header_file_name = os.path.join(args.output_dir, stencil.app_name + '.h')
    else:
      header_file_name = args.header_file
    with tempfile.TemporaryFile(mode='w+') as tmp:
      header.print_code(stencil, tmp)
      tmp.seek(0)
      with open(header_file_name, 'w') as header_file:
        shutil.copyfileobj(tmp, header_file)
