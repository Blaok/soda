import collections

from haoda import ir
from soda import core
from soda import grammar

def get_load_tuple(obj):
  """Get all load references as a tuple.

  Args:
    obj: A haoda.ir.Node object or a soda.core.Tensor object.

  Returns:
    A tuple of all the load references.

  Raises:
    TypeError: If obj is not an IR node or a Tensor.
  """
  def visitor(obj, loads):
    if isinstance(obj, grammar.Ref):
      loads.append(obj)
    return obj
  loads = []
  if issubclass(type(obj), ir.Node):
    obj.visit(visitor, loads)
  elif isinstance(obj, core.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a Tensor')
  return tuple(loads)

def get_load_set(obj):
  """Get all unique load references as a tuple.

  Args:
    obj: A haoda.ir.Node object.

  Returns:
    A tuple of all unique loads.

  Raises:
    TypeError: If obj is not an IR node.
  """
  def visitor(obj, loads):
    if isinstance(obj, grammar.Ref):
      loads[obj] = None
    return obj
  loads = collections.OrderedDict()
  if issubclass(type(obj), ir.Node):
    obj.visit(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a Tensor')
  return tuple(loads)

def get_load_dict(obj):
  """Get all load references as a dict mapping names to lists of loads.

  Args:
    obj: A soda.core.Tensor object.

  Returns:
    A dict mapping accessed tensor names to the corresponding lists of loads.

  Raises:
    TypeError: If obj is not a Tensor.
  """
  def visitor(obj, loads):
    if isinstance(obj, grammar.Ref):
      loads.setdefault(obj.name, []).append(obj)
    return obj
  loads = collections.OrderedDict()
  if isinstance(obj, core.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not a Tensor')
  return loads
