import logging
import operator

from haoda import ir
from soda import core
from soda import grammar

_logger = logging.getLogger().getChild(__name__)

def shift(obj, offset, excluded=(), op=operator.add):
  """Shift soda.grammar.Ref with the given offset.

  All soda.grammar.Ref, excluding the given names, will be shifted with the
  given offset using the given operator. The operator will be applied pointwise
  on the original index and the given offset.

  Args:
    obj: A haoda.ir.Node or a soda.core.Tensor object.
    offset: Second operand given to the operator.
    excluded: Sequence of names to be excluded from the mutation. Default to ().
    op: Shifting operator. Should be either operator.add or operator.sub.
  """
  if op not in (operator.add, operator.sub):
    _logger.warn('shifting with neither + nor -, which most likely is an error')
  def visitor(obj, args):
    if isinstance(obj, grammar.Ref):
      if obj.name not in excluded:
        new_idx = tuple(op(a, b) for a, b in zip(obj.idx, offset))
        _logger.debug('reference %s(%s) shifted to %s(%s)',
                      obj.name, ', '.join(map(str, obj.idx)),
                      obj.name, ', '.join(map(str, new_idx)))
        obj.idx = new_idx
  if isinstance(obj, ir.Node):
    obj.visit(visitor)
  elif isinstance(obj, core.Tensor):
    obj.mutate(visitor)
  else:
    raise TypeError('argument is not an IR node or a Tensor')
  return obj
