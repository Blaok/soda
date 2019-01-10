import logging
import operator

from haoda import ir
from soda import core

_logger = logging.getLogger().getChild(__name__)

def shift(obj, offset, excluded=(), op=operator.sub, verbose=False):
  """Shift soda.ir.Ref with the given offset.

  All soda.ir.Ref, excluding the given names, will be shifted with the
  given offset using the given operator. The operator will be applied pointwise
  on the original index and the given offset.

  Args:
    obj: A haoda.ir.Node or a soda.core.Tensor object.
    offset: Second operand given to the operator.
    excluded: Sequence of names to be excluded from the mutation. Default to ().
    op: Shifting operator. Should be either add or sub. Default to sub.
    verbose: Whether to log shiftings. Default to False.
  Returns:
    Mutated obj. If obj is an IR node, it will be a different object than the
    input. If obj is a tensor, it will be the same object but with fields
    mutated.
  """
  if op not in (operator.add, operator.sub):
    _logger.warn('shifting with neither + nor -, which most likely is an error')
  def visitor(obj, args):
    if isinstance(obj, ir.Ref):
      if obj.name not in excluded:
        new_idx = tuple(op(a, b) for a, b in zip(obj.idx, offset))
        if verbose:
          _logger.debug('reference %s(%s) shifted to %s(%s)',
                       obj.name, ', '.join(map(str, obj.idx)),
                       obj.name, ', '.join(map(str, new_idx)))
        obj.idx = new_idx
  if isinstance(obj, ir.Node):
    return obj.visit(visitor)
  if isinstance(obj, core.Tensor):
    obj.mutate(visitor)
  else:
    raise TypeError('argument is not an IR node or a tensor')
  return obj
