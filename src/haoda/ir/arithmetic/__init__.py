import logging

from haoda.ir.arithmetic import base

_logger = logging.getLogger().getChild(__name__)

def simplify(expr, lets=()):
  """Simplifies expr and corresponding lets.

  Args:
    expr: grammar.Expr
    lets: Sequence of grammar.Let

  Returns:
    (expr, lets): simplified expr and lets.
  """

  if expr is None:
    _logger.debug('None expr, no simplification.')
    return expr, lets

  passes = base.compose(
      base.flatten,
      base.print_tree)

  return passes(expr), tuple(map(passes, lets))
