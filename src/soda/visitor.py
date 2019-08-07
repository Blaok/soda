from typing import (
  Dict,
  Iterable,
  List,
  Mapping,
  Optional,
  Tuple,
  Union,
)

import collections

from haoda import ir
from soda import tensor

def get_load_tuple(obj: Union[ir.Node, tensor.Tensor]) -> Tuple[ir.Ref, ...]:
  """Get all load references as a tuple.

  Args:
    obj: A ir.Node object or a tensor.Tensor object.

  Returns:
    A tuple of all the load references.

  Raises:
    TypeError: If obj is not an IR node or a tensor.Tensor.
  """
  def visitor(obj: ir.Node, loads: List[ir.Ref]) -> ir.Node:
    if isinstance(obj, ir.Ref):
      loads.append(obj)
    return obj
  loads = []  # type: List[ir.Ref]
  if isinstance(obj, ir.Node):
    obj.visit(visitor, loads)
  elif isinstance(obj, tensor.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a tensor.Tensor')
  return tuple(loads)

def get_load_set(obj: Union[ir.Node, tensor.Tensor]) -> Tuple[ir.Ref, ...]:
  """Get all unique load references as a tuple.

  Args:
    obj: A haoda.ir.Node object or a tensor.Tensor object.

  Returns:
    A tuple of all unique loads.

  Raises:
    TypeError: If obj is not an IR node or a tensor.Tensor.
  """
  def visitor(obj: ir.Node, loads: Dict[ir.Ref, None]) -> ir.Node:
    if isinstance(obj, ir.Ref):
      loads[obj] = None
    return obj
  loads = collections.OrderedDict()   # type: Dict[ir.Ref, None]
  if isinstance(obj, ir.Node):
    obj.visit(visitor, loads)
  elif isinstance(obj, tensor.Tensor):
    obj.visit_loads(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a tensor.Tensor')
  return tuple(loads)

def get_load_dict(obj: Union[ir.Node, tensor.Tensor]) -> Dict[ir.Ref,
                                                            List[ir.Ref]]:
  """Get all load references as a dict mapping names to lists of loads.

  Args:
    obj: A haoda.ir.Node object or a tensor.Tensor object.

  Returns:
    A dict mapping accessed ir.Ref names to the corresponding lists of loads.

  Raises:
    TypeError: If obj is not an IR node or a tensor.Tensor.
  """
  def visitor(obj: ir.Node, loads: Dict[ir.Ref, List[ir.Ref]]) -> ir.Node:
    if isinstance(obj, ir.Ref):
      loads.setdefault(obj.name, []).append(obj)  # type: ignore
    return obj
  loads = collections.OrderedDict()   # type: Dict[ir.Ref, List[ir.Ref]]
  if isinstance(obj, tensor.Tensor):
    obj.visit_loads(visitor, loads)
  elif isinstance(obj, ir.Node):
    obj.visit(visitor, loads)
  else:
    raise TypeError('argument is not an IR node or a tensor.Tensor')
  return loads

def get_normalize_index(
    obj: Union[ir.Node, Iterable[ir.Node]],
    references: Optional[Mapping[str, Tuple[int, ...]]] = None
    ) -> Tuple[int, ...]:
  """Get the normalize index that will make the least access index 0.

  Args:
    obj: A node or an iterable of nodes.
  Returns:
    Normalize index as a tuple.
  Raises:
    TypeError: If argument is not an ir.Node or an iterable of ir.Nodes.
  """
  if not isinstance(obj, (collections.Iterable, ir.Node)):
    raise TypeError('argument is not an ir.Node or an iterable of ir.Nodes')
  if isinstance(obj, ir.Node):
    obj = (obj,)
  try:
    def get_idx(load: ir.Ref) -> Tuple[int, ...]:
      if references is None:
        return load.idx
      ref = references.get(getattr(load, 'name'))
      if ref is None:
        return load.idx
      return tuple(x - y for x, y in zip(load.idx, ref))
    return get_idx(min(sum(map(get_load_tuple, obj), ()),
                       key=lambda load: tuple(reversed(get_idx(load)))))
  except ValueError as e:
    if str(e) == 'min() arg is an empty sequence':
      return ()
    raise e
