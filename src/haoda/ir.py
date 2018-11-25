from collections import Iterable, OrderedDict, deque
import copy
import logging
import math

from cached_property import cached_property

from soda import grammar
from soda import util

_logger = logging.getLogger('__main__').getChild(__name__)

class FIFO(grammar.Node):
  """A reference to another node in a soda.grammar.Expr.

  This is used to represent a read/write from/to a Module in an output's Expr.
  It replaces Ref in soda.grammar, which is used to represent an element
  reference to a tensor.

  Attributes:
    read_module: Module reading from this FIFO.
    read_lat: int, at what cycle of a pipelined loop it is being read.
    write_module: Module writing to this FIFO.
    write_lat: int, at what cycle of a pipelined loop it is being written.
    depth: int, FIFO depth.
  """
  IMMUTABLE_ATTRS = 'read_module', 'write_module'
  SCALAR_ATTRS = 'read_module', 'read_lat', 'write_module', 'write_lat', 'depth'

  def __init__(self, write_module, read_module,
               depth=None, write_lat=None, read_lat=None):
    super().__init__(write_module=write_module, read_module=read_module,
                     depth=depth, write_lat=write_lat, read_lat=read_lat)

  def __repr__(self):
    return 'fifo[%d]: %s%s => %s%s' % (self.depth, repr(self.write_module),
      '' if self.write_lat is None else ' ~%s'%self.write_lat,
      repr(self.read_module),
      '' if self.read_lat is None else ' ~%s'%self.read_lat)

  def __hash__(self):
    return hash(tuple(getattr(self, _) for _ in self.IMMUTABLE_ATTRS))

  def __eq__(self, other):
    return all(getattr(self, _) == getattr(other, _)
               for _ in type(self).IMMUTABLE_ATTRS)
  @property
  def edge(self):
    return self.write_module, self.read_module

  @property
  def soda_type(self):
    return self.write_module.exprs[self].soda_type

  @property
  def c_expr(self):
    return 'from_{}_to_{}'.format(self.write_module.name, self.read_module.name)

class Module(object):
  """A node in the dataflow graph.

  This is the base class for a dataflow module. It defines the parent (input)
  nodes, children (output) nodes, output expressions, input schedules, and
  output schedules. It also has a name to help identify itself.

  Attributes:
    parents: Set of parent (input) Module.
    children: Set of child (output) Module.
    lets: List of soda.grammar.Let expressions.
    exprs: Dict of {FIFO: soda.grammar.Expr}, stores an output's expression.
  """
  def __init__(self):
    """Initializes attributes into empty list or dict.
    """
    self.parents = []
    self.children = []
    self.lets = []
    self.exprs = OrderedDict()

  @property
  def name(self):
    return 'module_%u' % hash(self)

  @property
  def fifos(self):
    return tuple(self.exprs.keys())

  @property
  def fifo_dict(self):
    return {(self, fifo.read_module): fifo for fifo in self.exprs}

  def fifo(self, dst_node):
    return self.fifo_dict[(self, dst_node)]

  def get_latency(self, dst_node):
    return self.fifo(dst_node).write_lat or 0

  def visit_loads(self, callback, args=None):
    obj = copy.copy(self)
    obj.lets = tuple(_.visit(callback, args) for _ in self.lets)
    obj.exprs = OrderedDict()
    for fifo in self.exprs:
      obj.exprs[fifo] = self.exprs[fifo].visit(callback, args)
    return obj

  @property
  def dram_reads(self):
    return self._interfaces['dram_reads']

  @property
  def dram_writes(self):
    return self._interfaces['dram_writes']

  @property
  def input_fifos(self):
    return self._interfaces['input_fifos']

  @property
  def output_fifos(self):
    return self._interfaces['output_fifos']

  @cached_property
  def _interfaces(self):
    def get_load_set(obj, loads):
      if isinstance(obj, FIFO):
        loads[obj] = None
      return obj
    loads = OrderedDict()
    self.visit_loads(get_load_set, loads)
    loads = tuple(loads)

    # find dram reads
    reads_in_lets = tuple(_.expr for _ in self.lets)
    reads_in_exprs = tuple(self.exprs.values())
    dram_reads = OrderedDict()
    for dram_ref in get_dram_refs(reads_in_lets + reads_in_exprs):
      for bank in dram_ref.dram:
        dram_reads[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_reads = tuple(dram_reads.values())

    # find dram writes
    writes_in_lets = tuple(_.name for _ in self.lets
                           if not isinstance(_.name, str))
    dram_writes = OrderedDict()
    for dram_ref in get_dram_refs(writes_in_lets):
      for bank in dram_ref.dram:
        dram_writes[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_writes = tuple(dram_writes.values())

    output_fifos = tuple(_.c_expr for _ in self.exprs)
    input_fifos = tuple(_.c_expr for _ in loads)

    return {
        'dram_writes' : dram_writes,
        'output_fifos' : output_fifos,
        'input_fifos' : input_fifos,
        'dram_reads' : dram_reads
    }

  def __str__(self):
    return '%s @ 0x%x: %s' % (type(self).__name__, id(self),
      self.__dict__)

  def __repr__(self):
    return '%s @ 0x%x' % (type(self).__name__, id(self))

  def add_child(self, child):
    """Add a child (low level).

    This method only handles children and parents field; lets and exprs are
    not updated.

    Arguments:
      child: Module, child being added
    """
    if child not in self.children:
      self.children.append(child)
    if self not in child.parents:
      child.parents.append(self)

  def bfs_node_gen(self):
    """BFS over descendant nodes.

    This method is a BFS traversal generator over all descendant nodes.
    """
    node_queue = deque([self])
    seen_nodes = {self}
    while node_queue:
      node = node_queue.popleft()
      yield node
      for child in node.children:
        if child not in seen_nodes:
          node_queue.append(child)
          seen_nodes.add(child)

  def dfs_node_gen(self):
    """DFS over descendant nodes.

    This method is a DFS traversal generator over all descendant nodes.
    """
    node_stack = [self]
    seen_nodes = {self}
    while node_stack:
      node = node_stack.pop()
      yield node
      for child in node.children:
        if child not in seen_nodes:
          node_stack.append(child)
          seen_nodes.add(child)

  def tpo_node_gen(self):
    """Traverse descendant nodes in topological order.

    This method is a generator that traverses all descendant nodes in
    topological order.
    """
    nodes = OrderedDict()
    for node in self.bfs_node_gen():
      nodes[node] = len(node.parents)
    while nodes:
      for node in nodes:
        if nodes[node] == 0:
          yield node
          for child in node.children:
            nodes[child] -= 1
          del nodes[node]
          break
      else:
        return

  def bfs_edge_gen(self):
    """BFS over descendant edges.

    This method is a BFS traversal generator over all descendant edges.
    """
    node_queue = deque([self])
    seen_nodes = {self}
    while node_queue:
      node = node_queue.popleft()
      for child in node.children:
        yield node, child
        if child not in seen_nodes:
          node_queue.append(child)
          seen_nodes.add(child)

  def dfs_edge_gen(self):
    """DFS over descendant edges.

    This method is a DFS traversal generator over all descendant edges.
    """
    node_stack = [self]
    seen_nodes = {self}
    while node_stack:
      node = node_stack.pop()
      for child in node.children:
        yield node, child
        if child not in seen_nodes:
          node_stack.append(child)
          seen_nodes.add(child)

  def get_descendants(self):
    """Get all descendant nodes.

    This method returns all descendant nodes as a set.

    Returns:
      Set of descendant Module.
    """
    return {self}.union(*map(Module.get_descendants, self.children))

  def get_connections(self):
    """Get all descendant edges.

    This method returns all descendant edges as a set.

    Returns:
      Set of descendant (src Module, dst Module) tuple.
    """
    return ({(self, child) for child in self.children}
        .union(*map(Module.get_connections, self.children)))


class DelayedRef(grammar.Node):
  """A delayed FIFO reference.

  Attributes:
    delay: int
    ref: FIFO
  """
  SCALAR_ATTRS = ('delay', 'ref')
  @property
  def soda_type(self):
    return self.ref.soda_type

  def __str__(self):
    return '%s delayed %d' % (self.ref, self.delay)

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.delay, self.ref))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('delay', 'ref'))

  @property
  def buf_name(self):
    return '{ref.c_expr}_delayed_{delay}_buf'.format(**self.__dict__)

  @property
  def ptr(self):
    return '{ref.c_expr}_delayed_{delay}_ptr'.format(**self.__dict__)

  @property
  def ptr_type(self):
    return 'uint%d' % int(math.log2(self.delay)+1)

  @property
  def c_expr(self):
    return '{ref.c_expr}_delayed_{delay}'.format(**self.__dict__)

  @property
  def c_ptr_type(self):
    return util.get_c_type(self.ptr_type)

  @property
  def c_ptr_decl(self):
    return '{} {} = 0;'.format(self.c_ptr_type, self.ptr)

  @property
  def c_buf_ref(self):
    return '{}[{}]'.format(self.buf_name, self.ptr)

  @property
  def c_buf_decl(self):
    return '{} {}[{}];'.format(self.c_type, self.buf_name, self.delay)

  @property
  def c_buf_load(self):
    return '{} = {};'.format(self.c_expr, self.c_buf_ref)

  @property
  def c_buf_store(self):
    return '{} = {};'.format(self.c_buf_ref, self.ref.ref_name)

  @property
  def c_next_ptr_expr(self):
    return '{ptr} < {depth} ? {c_ptr_type}({ptr}+1) : {c_ptr_type}(0)'.format(
        ptr=self.ptr, c_ptr_type=self.c_ptr_type, depth=self.delay-1)

class FIFORef(grammar.Node):
  """A FIFO reference.

  Attributes:
    fifo: FIFO it is linked to
    lat: int, at what cycle of a pipelined loop it is being referenced.
    ref_id: int, reference id in the current scope
  Properties:
    c_type: str
    c_expr: str
    soda_type: str
    ld_name: str
    st_name: str
    ref_name: str
  """
  SCALAR_ATTRS = ('fifo', 'lat', 'ref_id')
  LD_PREFIX = 'fifo_ld_'
  ST_PREFIX = 'fifo_st_'
  REF_PREFIX = 'fifo_ref_'
  def __str__(self):
    return '<%s fifo_ref_%d%s>' % (self.soda_type, self.ref_id,
                                   '@%s'%self.lat if self.lat else '')

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.lat, self.ref_id))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('lat', 'ref_id'))

  @property
  def soda_type(self):
    return self.fifo.soda_type

  @property
  def ld_name(self):
    return '{LD_PREFIX}{ref_id}'.format(**self.__dict__, **type(self).__dict__)

  @property
  def ref_name(self):
    return '{REF_PREFIX}{ref_id}'.format(**self.__dict__, **type(self).__dict__)

  @property
  def c_expr(self):
    return self.ref_name

class DRAMRef(grammar.Node):
  """A DRAM reference.

  Attributes:
    soda_type: str
    dram: [int], DRAM id it is accessing
    var: str, variable name it is accessing
    offset: int
  """
  SCALAR_ATTRS = 'soda_type', 'dram', 'var', 'offset'
  def __str__(self):
    return 'dram<bank {} {}@{}>'.format(util.lst2str(self.dram),
                                        self.var, self.offset)

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash((self.dram, self.offset))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in ('dram', 'offset'))
  @property
  def c_expr(self):
    return str(self)

  def dram_buf_name(self, bank):
    assert bank in self.dram, 'unexpected bank {}'.format(bank)
    return 'dram_{}_bank_{}_buf'.format(self.var, bank)

  def dram_fifo_name(self, bank):
    assert bank in self.dram, 'unexpected bank {}'.format(bank)
    return 'dram_{}_bank_{}_fifo'.format(self.var, bank)

class ModuleTrait(grammar.Node):
  """A immutable, hashable trait of a dataflow module.

  Attributes:
    lets: tuple of lets
    exprs: tuple of exprs
    template_types: tuple of template types (TODO)
    template_ints: tuple of template ints (TODO)

  Properties:
    loads: tuple of FIFORefs
  """
  LINEAR_ATTRS = ('lets', 'exprs', 'template_types', 'template_ints')

  def __init__(self, node):
    def mutate(obj, loads):
      if isinstance(obj, FIFO):
        if loads:
          if obj not in loads:
            load_id = next(reversed(loads.values())).ref_id+1
          else:
            return loads[obj]
        else:
          load_id = 0
        fifo_ref = FIFORef(fifo=obj, lat=obj.read_lat, ref_id=load_id)
        loads[obj] = fifo_ref
        return fifo_ref
      return obj
    loads = OrderedDict()
    node = node.visit_loads(mutate, loads)
    self.loads = tuple(loads.values())
    super().__init__(lets=tuple(node.lets), exprs=tuple(node.exprs.values()),
                     template_types=tuple(), template_ints=tuple())
    _logger.debug('Signature: %s', self)

  def __repr__(self):
    return '%s(loads: %s, lets: %s, exprs: %s)' % (
        type(self).__name__,
        util.idx2str(self.loads),
        util.idx2str(self.lets),
        util.idx2str(self.exprs))

  @property
  def dram_reads(self):
    return self._interfaces['dram_reads']

  @property
  def dram_writes(self):
    return self._interfaces['dram_writes']

  @property
  def input_fifos(self):
    return self._interfaces['input_fifos']

  @property
  def output_fifos(self):
    return self._interfaces['output_fifos']

  @cached_property
  def _interfaces(self):
    # find dram reads
    reads_in_lets = tuple(_.expr for _ in self.lets)
    reads_in_exprs = tuple(self.exprs)
    dram_reads = OrderedDict()
    for dram_ref in get_dram_refs(reads_in_lets + reads_in_exprs):
      for bank in dram_ref.dram:
        dram_reads[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_reads = tuple(dram_reads.values())

    # find dram writes
    writes_in_lets = tuple(_.name for _ in self.lets
                           if not isinstance(_.name, str))
    dram_writes = OrderedDict()
    for dram_ref in get_dram_refs(writes_in_lets):
      for bank in dram_ref.dram:
        dram_writes[(dram_ref.var, bank)] = (dram_ref, bank)
    dram_writes = tuple(dram_writes.values())

    output_fifos = tuple('{}{}'.format(FIFORef.ST_PREFIX, idx)
                         for idx, expr in enumerate(self.exprs))
    input_fifos = tuple(_.ld_name for _ in self.loads)

    return {
        'dram_writes' : dram_writes,
        'output_fifos' : output_fifos,
        'input_fifos' : input_fifos,
        'dram_reads' : dram_reads
    }

def get_dram_refs(obj):
  if isinstance(obj, Iterable):
    return sum(map(get_dram_refs, obj), [])
  dram_refs = []
  def get_dram_refs_callback(obj, args):
    if isinstance(obj, DRAMRef):
      args.append(obj)
    return obj
  obj.visit(get_dram_refs_callback, dram_refs)
  return dram_refs
