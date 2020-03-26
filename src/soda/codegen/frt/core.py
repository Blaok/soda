import argparse
import shutil
import sys
import tempfile

from soda import core
from soda.codegen.frt import host


def add_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--frt-host',
                      type=str,
                      dest='frt_host',
                      metavar='file',
                      help='FPGA runtime host code')


def print_code(stencil: core.Stencil, args: argparse.Namespace) -> None:
  if args.frt_host is not None:
    with tempfile.TemporaryFile(mode='w+') as tmp:
      host.print_code(stencil, tmp)
      tmp.seek(0)
      if args.frt_host == '-':
        shutil.copyfileobj(tmp, sys.stdout)
      else:
        with open(args.frt_host, 'w') as frt_host:
          shutil.copyfileobj(tmp, frt_host)
