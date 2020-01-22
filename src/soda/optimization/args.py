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
      '--temporal-cse',
      type=str,
      metavar=
      '(yes|no|greedy|optimal|glore|built-in|built-in:greedy|built-in:optimal)',
      dest='temporal_cse',
      nargs='?',
      const='yes',
      default='no',
      help='enable temporal common subexpression elimination or not')


def get_kwargs(args: argparse.Namespace) -> Dict[str, str]:
  optimizations = {}
  if args.temporal_cse != 'no':
    optimizations['tcse'] = args.temporal_cse
  if args.inline != 'no':
    optimizations['inline'] = args.inline
  return optimizations
