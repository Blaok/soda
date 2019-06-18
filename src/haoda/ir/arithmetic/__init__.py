from typing import (
    overload,
    Callable,
    Iterable,
)

import collections

from haoda import ir
from haoda.ir.arithmetic import base


# pylint: disable=function-redefined
@overload
def simplify(expr: ir.Node, logger: Callable[..., None] = None) -> ir.Node:
  ...


# pylint: disable=function-redefined
@overload
def simplify(expr: Iterable[ir.Node], logger: Callable[..., None] = None) \
    -> Iterable[ir.Node]:
  ...


def simplify(expr, logger=None):
  """Simplifies expressions.

  Args:
    expr: A haoda.ir.Node or a sequence of haoda.ir.Node.

  Returns:
    Simplified haoda.ir.Node or sequence.
  """

  if expr is None:
    if logger is not None:
      logger.debug('None expr, no simplification.')
    return expr

  passes = base.flatten
  if logger is not None:
    passes = base.compose(passes, lambda node: base.print_tree(node, logger))

  if isinstance(expr, collections.Iterable):
    return type(expr)(map(passes, expr))

  return passes(expr)
