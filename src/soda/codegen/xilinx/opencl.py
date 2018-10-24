import sys
import os

from soda.codegen import header
from soda.codegen import host
from soda.codegen import kernel

def add_arguments(parser):
  parser.add_argument(
      '--xocl', type=str, dest='output_dir', metavar='dir',
      help='directory to generate kernel, source, and header; default names '
      'used; default to the current working directory; may be overridden by '
      '--xocl-kernel, --xocl-host, or --header')
  parser.add_argument(
      '--xocl-kernel', type=str, dest='kernel_file', metavar='file',
      help='Vivado HLS C++ kernel code for the Xilinx OpenCL flow; overrides '
      '--xocl')
  parser.add_argument(
      '--xocl-host', type=str, dest='host_file', metavar='file',
      help='host C++ source code for the Xilinx OpenCL flow; overrides --xocl')
  parser.add_argument(
      '--header', type=str, dest='header_file', metavar='file',
      help='host C++ header code; overrides --xocl')

def print_code(stencil, args):
  if args.kernel_file is not None:
    if args.kernel_file == '-':
      kernel.print_code(stencil, sys.stdout)
    else:
      with open(args.kernel_file, 'w') as kernel_file:
        kernel.print_code(stencil, kernel_file)

  if args.host_file is not None:
    if args.host_file == '-':
      host.print_code(stencil, sys.stdout)
    else:
      with open(args.host_file, 'w') as host_file:
        host.print_code(stencil, host_file)

  if args.header_file is not None:
    if args.header_file == '-':
      header.print_code(stencil, sys.stdout)
    else:
      with open(args.header_file, 'w') as header_file:
        header.print_code(stencil, header_file)

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
    with open(kernel_file_name, 'w') as kernel_file:
      kernel.print_code(stencil, kernel_file)
    if args.host_file is None:
      host_file_name = os.path.join(args.output_dir, stencil.app_name + '.cpp')
    else:
      host_file_name = args.host_file
    with open(host_file_name, 'w') as host_file:
      host.print_code(stencil, host_file)
    if args.header_file is None:
      header_file_name = os.path.join(args.output_dir, stencil.app_name + '.h')
    else:
      header_file_name = args.header_file
    with open(header_file_name, 'w') as header_file:
      header.print_code(stencil, header_file)
