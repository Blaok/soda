from typing import (
    Dict,
    List,
    Tuple,
)
import functools
import logging
from collections import OrderedDict

from haoda import ir
from haoda import util

_logger = logging.getLogger().getChild(__name__)


def compose(*funcs):
  """Composes functions. The first function in funcs are invoked the first.
  """
  # Somehow pylint gives false positive for f and g.
  # pylint: disable=undefined-variable
  return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs, lambda x: x)


def flatten(node: ir.Node) -> ir.Node:
  """Flattens an node if possible.

  Flattens an node if it is:
    + a singleton BinaryOp; or
    + a compound BinaryOp with reduction operators; or
    + a compound Operand; or
    + a Unary with an identity operator; or
    + a Call of a reduction functions.

  An Operand is a compound Operand if and only if its attr is a ir.Node.

  A Unary has identity operator if and only if all its operators are '+' or '-',
  and the number of '-' is even; or all of its operators are '!' and the number
  of '!' is even.

  Reduction functions are defined in ir.REDUCTION_FUNCS.

  Args:
    node: ir.Node to flatten.

  Returns:
    node: flattened ir.Node.

  Raises:
    util.InternalError: if Operand is undefined.
  """

  def visitor(node, args=None):
    if isinstance(node, ir.BinaryOp):

      # Flatten singleton BinaryOp
      if len(node.operand) == 1:
        return flatten(node.operand[0])

      # Flatten BinaryOp with reduction operators
      new_operator, new_operand = [], []
      for child_operator, child_operand in zip((None, *node.operator),
                                               node.operand):
        if child_operator is not None:
          new_operator.append(child_operator)
        # The first operator can always be flattened if two operations has the
        # same type.
        if child_operator in (None, '||', '&&', *'|&+*') and \
            type(child_operand) is type(node):
          new_operator.extend(child_operand.operator)
          new_operand.extend(child_operand.operand)
        else:
          new_operand.append(child_operand)
      # At least 1 operand is flattened.
      if len(new_operand) > len(node.operand):
        return flatten(type(node)(operator=new_operator, operand=new_operand))

    # Flatten compound Operand
    if isinstance(node, ir.Operand):
      for attr in node.ATTRS:
        val = getattr(node, attr)
        if val is not None:
          if isinstance(val, ir.Node):
            return flatten(val)
          break
      else:
        raise util.InternalError('undefined Operand')

    # Flatten identity unary operators
    if isinstance(node, ir.Unary):
      minus_count = node.operator.count('-')
      if minus_count % 2 == 0:
        plus_count = node.operator.count('+')
        if plus_count + minus_count == len(node.operator):
          return flatten(node.operand)
      not_count = node.operator.count('!')
      if not_count % 2 == 0 and not_count == len(node.operator):
        return flatten(node.operand)

    # Flatten reduction functions
    if isinstance(node, ir.Call):
      operator = getattr(node, 'name')
      if operator in ir.REDUCTION_FUNCS:
        operands = []
        for operand in getattr(node, 'arg'):
          if (isinstance(operand, ir.Call) and
              getattr(operand, 'name') == operator):
            operands.extend(getattr(operand, 'arg'))
          else:
            operands.append(operand)
        if len(operands) > len(getattr(node, 'arg')):
          return flatten(ir.Call(name=operator, arg=operands))

    return node

  if not isinstance(node, ir.Node):
    return node

  return node.visit(visitor)


def reverse_distribute(node: ir.Node) -> ir.Node:
  """Apply distributive property in reverse, if possible.

  Currently only left- and right-distribution of multiplication over addition is
  supported.

  Args:
    node: ir.Node to process.

  Returns:
    Processed node.
  """

  def visitor(node: ir.Node, left_distribute: bool) -> ir.Node:
    """Apply left- or right-distributive property in reverse, if possible

    Args:
      node: ir.Node to process.
      left_distribute: Whether to apply *left*-distributive property.

    Returns:
      Processed node.
    """
    if isinstance(node, ir.AddSub):
      items = OrderedDict()  # type: Dict[ir.Node, List[Tuple[str, ir.Node]]]
      for operator, operand in zip(('+',) + getattr(node, 'operator'),
                                   getattr(node, 'operand')):
        if (operator == '+' and isinstance(operand, ir.MulDiv) and
            getattr(operand, 'operator') == ('*',)):
          if left_distribute:
            coeff, item = getattr(operand, 'operand')
          else:
            item, coeff = getattr(operand, 'operand')
          items.setdefault(coeff, []).append((operator, item))
      new_operators = []
      new_operands = []
      for coeff, item in items.items():
        operator, operand = zip(*item)
        assert operator[0] == '+'
        new_operators.append(operator[0])
        if len(operand) > 1:
          new_item = ir.AddSub(operator=operator[1:], operand=operand)
        else:
          new_item = operand[0]
        if left_distribute:
          children = coeff, new_item
        else:
          children = new_item, coeff
        new_operands.append(ir.MulDiv(operator=('*',), operand=children))
      if len(new_operands) > 1:
        assert new_operators[0] == '+'
        new_node = ir.AddSub(operator=tuple(new_operators[1:]),
                             operand=tuple(new_operands))
        if new_node != node:
          return new_node
      elif new_operands and new_operands[0] != node:
        return new_operands[0]
    return node

  return node.visit(visitor, True).visit(visitor, False)


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

  if not isinstance(node, ir.Node):
    return node

  printer('root')
  return node.visit(visitor,
                    args=[1],
                    pre_recursion=pre_recursion,
                    post_recursion=post_recursion)


def propagate_type(node, symbol_table):

  def visitor(node, symbol_table):
    if node.haoda_type is None:
      if isinstance(node, (ir.Ref, ir.Var)):
        node.haoda_type = symbol_table[node.name]
    return node

  return node.visit(visitor, symbol_table)
