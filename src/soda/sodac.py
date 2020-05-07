#!/usr/bin/python3
import argparse
import logging
import os
import sys

import textx

from haoda import util
from soda import core, grammar
from soda.codegen.frt import core as frt
from soda.codegen.intel import opencl as iocl
from soda.codegen.xilinx import opencl as xocl
from soda.model import xilinx as model
from soda.optimization import args as opt_args

logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')
logger = logging.getLogger().getChild(os.path.basename(__file__))

def main():
  parser = argparse.ArgumentParser(
    prog='sodac',
    description='Stencil with Optimized Dataflow Architecture '
                '(SODA) compiler')
  parser.add_argument('--verbose', '-v',
                      action='count',
                      dest='verbose',
                      help='increase verbosity')
  parser.add_argument('--quiet', '-q',
                      action='count',
                      dest='quiet',
                      help='decrease verbosity')
  parser.add_argument('--recursion-limit',
                      type=int,
                      dest='recursion_limit',
                      help='override Python recursion limit')
  parser.add_argument('--burst-width',
                      type=int,
                      dest='burst_width',
                      help='override burst width')
  parser.add_argument('--unroll-factor',
                      type=int,
                      metavar='UNROLL_FACTOR',
                      dest='unroll_factor',
                      help='override unroll factor')
  parser.add_argument('--replication-factor',
                      type=int,
                      metavar='REPLICATION_FACTOR',
                      dest='replication_factor',
                      help='override replication factor')
  parser.add_argument('--tile-size',
                      type=int,
                      nargs='+',
                      metavar='TILE_SIZE',
                      dest='tile_size',
                      help='override tile size; '
                           '0 means no overriding on that dimension')
  parser.add_argument('--dram-in',
                      type=str,
                      dest='dram_in',
                      help='override DRAM configuration for input')
  parser.add_argument('--dram-out',
                      type=str,
                      dest='dram_out',
                      help='override DRAM configuration for output')
  parser.add_argument('--iterate',
            type=int,
            metavar='#ITERATION',
            dest='iterate',
            help='override iterate directive; '
            'repeat execution multiple times iteratively')
  parser.add_argument('--border',
            type=str,
            metavar='(ignore|preserve)',
            dest='border',
            help='override border handling strategy')
  parser.add_argument('--cluster',
            type=str,
            metavar='(none|fine|coarse|full)',
            dest='cluster',
            help='module clustering level, `none` generates '
               'standalone compute / forward modules, `fine` '
               'fuses forwarders into compute modules, `coarse` '
               'fuses each stage together, `full` fuses '
               'everything together')
  parser.add_argument(type=str,
            dest='soda_src',
            metavar='file',
            help='soda source code')

  xocl.add_arguments(parser.add_argument_group('Xilinx OpenCL backend'))
  iocl.add_arguments(parser.add_argument_group('Intel OpenCL backend'))
  frt.add_arguments(parser.add_argument_group('FPGA runtime backend'))
  opt_args.add_arguments(parser.add_argument_group('SODA optimizations'))

  parser.add_argument('--model-file',
            type=str,
            dest='model_file',
            metavar='file',
            help='resource model specified as json file')
  parser.add_argument('--estimation-file',
            type=str,
            dest='estimation_file',
            metavar='file',
            help='report resource and performance estimation as '
               'json file')

  args = parser.parse_args()
  verbose = 0 if args.verbose is None else args.verbose
  quiet = 0 if args.quiet is None else args.quiet
  logging_level = (quiet-verbose)*10+logging.getLogger().getEffectiveLevel()
  if logging_level > logging.CRITICAL:
    logging_level = logging.CRITICAL
  if logging_level < logging.DEBUG:
    logging_level = logging.DEBUG
  logging.getLogger().setLevel(logging_level)
  logger.info('set log level to %s', logging.getLevelName(logging_level))
  # TODO: check tile size

  if args.recursion_limit is not None:
    sys_recursion_limit = sys.getrecursionlimit()
    if sys_recursion_limit > args.recursion_limit:
      logger.warning(
        'Python system recursion limit (%d) > specified value (%d); '
        'the latter will be ignored', sys_recursion_limit, args.recursion_limit)
    else:
      sys.setrecursionlimit(args.recursion_limit)
      logger.warning('Python recursion limit is set to %d',
                     sys.getrecursionlimit())

  soda_mm = textx.metamodel_from_str(grammar.GRAMMAR, classes=grammar.CLASSES)
  logger.info('build metamodel')
  try:
    if args.soda_src == '-':
      soda_file_name = sys.stdin.name
      soda_model = soda_mm.model_from_str(sys.stdin.read())
    else:
      with open(args.soda_src, 'r') as soda_file:
        soda_model = soda_mm.model_from_str(soda_file.read())
        soda_file_name = soda_file.name
    logger.info('%s parsed as soda file', soda_file_name)
    logger.debug('soda program parsed:\n  %s',
                 str(soda_model).replace('\n', '\n  '))

    tile_size = []
    for dim in range(soda_model.dim-1):
      if (args.tile_size is not None and
          dim < len(args.tile_size) and
          args.tile_size[dim] > 0):
        tile_size.append(args.tile_size[dim])
      else:
        tile_size.append(soda_model.tile_size[dim])
    tile_size.append(0)

    if args.replication_factor is None:
      if args.unroll_factor is not None:
        unroll_factor = args.unroll_factor
      else:
        unroll_factor = soda_model.unroll_factor
      replication_factor = 1
    else:
      unroll_factor = args.replication_factor
      replication_factor = args.replication_factor

    stencil = core.Stencil(
      burst_width=args.burst_width if args.burst_width is not None
                     else soda_model.burst_width,
      border=args.border if args.border is not None
                 else soda_model.border,
      iterate=args.iterate if args.iterate is not None
                 else soda_model.iterate,
      cluster=args.cluster if args.cluster is not None
                 else soda_model.cluster,
      dram_in=args.dram_in,
      dram_out=args.dram_out,
      app_name=soda_model.app_name,
      input_stmts=soda_model.input_stmts,
      param_stmts=soda_model.param_stmts,
      local_stmts=soda_model.local_stmts,
      output_stmts=soda_model.output_stmts,
      dim=soda_model.dim,
      tile_size=tile_size,
      unroll_factor=unroll_factor,
      replication_factor=replication_factor,
      optimizations=opt_args.get_kwargs(args),
    )

    logger.debug('stencil obtained: %s', stencil)

    xocl.print_code(stencil, args, parser)
    iocl.print_code(stencil, args)
    frt.print_code(stencil, args)

    if args.estimation_file is not None:
      if args.model_file is None:
        if args.soda_src.endswith('.soda'):
          model_file = args.soda_src[:-len('.soda')]+'_model.json'
        else:
          logger.fatal('cannot find resource model file')
          sys.exit(1)
      else:
        model_file = args.model_file

      def print_estimation():
        def print_estimation():
          model.print_estimation(stencil,
                       model_file,
                       estimation_file)
        if args.estimation_file == '-':
          estimation_file = sys.stdout
          print_estimation()
        else:
          with open(args.estimation_file, 'w') as estimation_file:
            print_estimation()
      if model_file == '-':
        model_file = sys.stdin
        print_estimation()
      else:
        with open(model_file) as model_file:
          print_estimation()


  except textx.exceptions.TextXSyntaxError as e:
    logger.error(e)
    sys.exit(1)
  except util.SemanticError as e:
    logger.error(e)
    sys.exit(1)
  except util.SemanticWarn as w:
    logger.warning(w)

if __name__ == '__main__':
  main()
