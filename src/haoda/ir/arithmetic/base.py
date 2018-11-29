import functools
import logging

from haoda import ir
from haoda import util

_logger = logging.getLogger().getChild(__name__)

def compose(*funcs):
  """Composes functions. The first function in funcs are invoked the first.
  """
  # Somehow pylint gives false positive for f and g.
  # pylint: disable=undefined-variable
  return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs, lambda x: x)

def flatten(node):
  """Flattens an node if possible.

  Flattens an node if it is:
    + a singleton Binary; or
    + a compound Operand; or
    + a Unary with an identity operator.

  An Operand is a compound Operand if and only if its attr is a ir.Node.

  A Unary has identity operator if and only if all its operators are '+' or '-',
  and the number of '-' is even.

  Args:
    node: ir.Node to flatten.

  Returns:
    node: flattened ir.Node.

  Raises:
    util.InternalError: if Operand is undefined.
  """

  def visitor(node, args=None):
    if issubclass(type(node), ir.BinaryOp):
      if len(node.operand) == 1:
        return flatten(node.operand[0])
    if isinstance(node, ir.Operand):
      for attr in node.ATTRS:
        val = getattr(node, attr)
        if val is not None:
          if issubclass(type(val), ir.Node):
            return flatten(val)
          break
      else:
        raise util.InternalError('undefined Operand')
    if isinstance(node, ir.Unary):
      minus_count = node.operator.count('-')
      if minus_count % 2 == 0:
        plus_count = node.operator.count('+')
        if plus_count + minus_count == len(node.operator):
          return flatten(node.operand)
    return node

  if not issubclass(type(node), ir.Node):
    return node

  return node.visit(visitor)

def print_tree(node, printer=_logger.debug):
  """Prints the node type as a tree.

  Args:
    node: ir.Node to print.
    args: Singleton list of the current tree height.

  Returns:
    node: Input ir.Node as-is.
  """

  def pre_recursion(node, args):
    args[0] += 1

  def post_recursion(node, args):
    args[0] -= 1

  def visitor(node, args):
    printer('%s+-%s: %s' % (' ' * args[0], type(node).__name__, node))

  if not issubclass(type(node), ir.Node):
    return node

  printer('root')
  return node.visit(visitor, args=[1], pre_recursion=pre_recursion,
                    post_recursion=post_recursion)
