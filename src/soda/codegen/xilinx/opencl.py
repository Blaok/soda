import argparse
import logging
import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

from absl import flags
from haoda import util
from haoda.backend import xilinx as backend

from soda.codegen.xilinx import header
from soda.codegen.xilinx import hls_kernel as kernel
from soda.codegen.xilinx import host, rtl_kernel

if TYPE_CHECKING:
  from soda.core import Stencil

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'xocl-platform',
    None,
    'Vitis shell platform name or directory for the Xilinx OpenCL flow',
)
flags.DEFINE_string(
    'xocl-part-num',
    None,
    'part number used for Xilinx HLS',
)
flags.DEFINE_string(
    'xocl-clock-period',
    None,
    'target clock period in nanoseconds, used for Xilinx HLS',
)

util.define_alias_flags(__name__)

_logger = logging.getLogger().getChild(__name__)

hls = 'vitis_hls'


class ArgParseActionForHls(argparse.Action):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    global hls  # pylint: disable=global-statement
    hls = values


def add_arguments(parser):
  parser.add_argument(
      '--xocl',
      type=str,
      dest='output_dir',
      metavar='dir',
      nargs='?',
      const='',
      help=('(deprecated) directory to generate kernel and host code; '
            'default names are used; '
            'default to the current working directory; '
            'may be overridden by --xocl-header, --xocl-host, or '
            '--xocl-kernel'),
  )
  parser.add_argument(
      '--xocl-header',
      type=str,
      dest='header_file',
      metavar='file',
      help='(deprecated) host C++ header code; overrides --xocl',
  )
  parser.add_argument(
      '--xocl-host',
      type=str,
      dest='host_file',
      metavar='file',
      help=('(deprecated) host C++ source code for the Xilinx OpenCL flow; '
            'overrides --xocl'),
  )
  parser.add_argument(
      '--xocl-kernel',
      type=str,
      dest='kernel_file',
      metavar='file',
      help=('Xilinx HLS C++ kernel code for the Xilinx OpenCL flow; '
            'overrides --xocl'),
  )
  parser.add_argument(
      '--xocl-hw-xo',
      type=str,
      dest='xo_file',
      metavar='file',
      help='hardware object file for the Xilinx OpenCL flow',
  )
  parser.add_argument(
      '--xocl-hw-xo-work-dir',
      type=str,
      dest='xo_work_dir',
      metavar='file',
      help='work directory for the Xilinx OpenCL hardware object',
  )
  parser.add_argument(
      '--xocl-connectivity',
      type=argparse.FileType('w'),
      dest='connectivity',
      metavar='file',
      help='connectivity.ini file for the Xilinx OpenCL flow',
  )
  parser.add_argument(
      '--xocl-constraint',
      type=str,
      dest='constraint',
      metavar='file',
      help='constraint.tcl file for the Xilinx OpenCL flow',
  )
  parser.add_argument(
      '--xocl-interface',
      type=str,
      dest='interface',
      choices=('m_axi', 'axis', 'tapa::mmap', 'tapa::stream'),
      default='m_axi',
      help='interface type of the Xilinx OpenCL code',
  )
  parser.add_argument(
      '--xocl-hls',
      type=str,
      choices=('vivado_hls', 'vitis_hls'),
      default='vivado_hls',
      action=ArgParseActionForHls,  # modifies global var hls directly
      help='use Vivado HLS or Vitis HLS style for Xilinx HLS',
  )


def print_code(
    stencil: 'Stencil',
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
  _logger.info('using %s', hls)

  if args.connectivity is not None:
    kernel.print_connectivity(stencil, args.connectivity)
    args.connectivity.close()

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
    if args.constraint is not None and args.connectivity is None:
      parser.error('constraint must be generated together with connectivity')
    with tempfile.TemporaryFile(mode='w+b') as tmp_obj:
      connectivity_file = None
      if args.connectivity is not None:
        connectivity_file = args.connectivity.name
      rtl_kernel.print_code(
          stencil,
          tmp_obj,
          device_info=backend.parse_device_info_from_flags(
              platform_name='xocl-platform',
              part_num_name='xocl-part-num',
              clock_period_name='xocl-clock-period',
          ),
          work_dir=args.xo_work_dir,
          connectivity_file=connectivity_file,
          constraint_file=args.constraint,
          interface=args.interface,
      )
      tmp_obj.seek(0)
      if args.xo_file == '-':
        shutil.copyfileobj(tmp_obj, sys.stdout)  # type: ignore
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
          args.output_dir, '%s_kernel-tile%s-unroll%d-ddr%s.cpp' %
          (stencil.app_name, 'x'.join('%d' % x for x in stencil.tile_size[:-1]),
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
