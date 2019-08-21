import collections

from haoda import ir

def get_read_fifo_set(module):
  """Get all read FIFOs as a tuple. Each FIFO only appears once.

  Args:
    module: A haoda.ir.Module object.

  Returns:
    A tuple of all FIFOs that are read in the module.

  Raises:
    TypeError: If argument is not a module.
  """
  def visitor(obj, args):
    if isinstance(obj, ir.FIFO):
      args[obj] = None
    return obj
  fifo_loads = collections.OrderedDict()
  if isinstance(module, ir.Module):
    module.visit_loads(visitor, fifo_loads)
  else:
    raise TypeError('argument is not a module')
  return tuple(fifo_loads)


def get_instances_of(node_or_iterable, class_or_tuple):
  """Get all ir.Node references of specific classes as a tuple.

  Args:
    node_or_iterable: A haoda.ir.Node object or an Iterable of
        haoda.ir.Node objects.
    class_or_tuple: Wanted class or tuple of wanted classes.

  Returns:
    A tuple of all wanted references.

  Raises:
    TypeError: If obj is not an IR node or a sequence.
  """
  def visitor(node, instances):
    if isinstance(node, class_or_tuple):
      instances.append(node)
    return node
  if isinstance(node_or_iterable, collections.Iterable):
    return sum(
        (get_instances_of(node, class_or_tuple) for node in node_or_iterable),
        ())
  instances = []
  if isinstance(node_or_iterable, ir.Node):
    node_or_iterable.visit(visitor, instances)
  else:
    raise TypeError('argument is not an IR node or a sequence')
  return tuple(instances)

def get_vars(node_or_iterable):
  return get_instances_of(node_or_iterable, ir.Var)

def get_dram_refs(node_or_iterable):
  return get_instances_of(node_or_iterable, ir.DRAMRef)
