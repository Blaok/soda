import argparse
from typing import Dict


def add_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--inline',
                      type=str,
                      metavar='(yes|no)',
                      dest='inline',
                      nargs='?',
                      const='yes',
                      default='no')
  parser.add_argument(
      '--computation-reuse',
      type=str,
      metavar=
      '(yes|no|greedy|optimal|glore|built-in|built-in:greedy|built-in:optimal)',
      dest='computation_reuse',
      nargs='?',
      const='yes',
      default='no',
      help='enable computation reuse or not')


def get_kwargs(args: argparse.Namespace) -> Dict[str, str]:
  optimizations = {}
  if args.computation_reuse != 'no':
    optimizations['computation-reuse'] = args.computation_reuse
  if args.inline != 'no':
    optimizations['inline'] = args.inline
  return optimizations
