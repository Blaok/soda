import shutil
import sys
import tempfile

from soda import core
from soda.codegen.intel import ocl_kernel as kernel


def add_arguments(parser):
  parser.add_argument(
      '--iocl-kernel',
      type=str,
      dest='iocl_kernel',
      metavar='file',
      help='Intel OpenCL kernel code for the Intel FPGA OpenCL flow')


def print_code(stencil: core.Stencil, args):
  if args.iocl_kernel is not None:
    with tempfile.TemporaryFile(mode='w+') as tmp:
      kernel.print_code(stencil, tmp)
      tmp.seek(0)
      if args.iocl_kernel == '-':
        shutil.copyfileobj(tmp, sys.stdout)
      else:
        with open(args.iocl_kernel, 'w') as iocl_kernel:
          shutil.copyfileobj(tmp, iocl_kernel)
