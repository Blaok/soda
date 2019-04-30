from typing import (
    overload,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
import collections
import itertools
import logging
import operator
import random

import cached_property

from haoda import ir
from haoda import util
from haoda.ir import arithmetic
from haoda.ir.arithmetic import base
from soda import grammar
from soda import mutator
from soda import visitor as soda_visitor

RelativeAttr = int
AbsoluteAttr = int
Attr = Union[RelativeAttr, Tuple[RelativeAttr, Optional[AbsoluteAttr]]]

OrderedDict = collections.OrderedDict
class OrderedCounter(collections.Counter,   # type: ignore
                     collections.OrderedDict):
  pass

_logger = logging.getLogger().getChild(__name__)

REDUCTION_OPS = {
  '+': ir.AddSub,
  '*': ir.MulDiv
}

def extract_attr(node: ir.Node) -> Tuple[Tuple[int, ...], ir.Node]:
  """Extract attributes from a node.

  Extract relative and absolute attributes from a node. The relative attribute
  would be the load index and the absolute attribute is the normalized node.

  Args:
    node: The ir.node to be extracted.

  Returns:
    Tuple of rattr and aattr.
  """
  load = soda_visitor.get_load_set(node)[0]
  return load.idx, mutator.shift(node, load.idx)

def assemble_attr(rattr: Tuple[int, ...], aattr: ir.Node) -> ir.Node:
  """Assemble a node from attributes.

  The absolute attribute must be a normalized ir.Node. The relative attribute
  will be used as the shifting offset to obtain the original ir.Node.

  Args:
    rattr: The relative attribute.
    aattr: The absolute attribute.

  Returns:
    ir.Node assembled from the attributes.
  """
  return mutator.shift(aattr, rattr, op=operator.add)

class Linearizer:
  """Apply and restore linearization.

  This class stores the necessory information needed to apply and restore
  linearization. Instances of this class is callable.

  Attributes:
    num_dim: Integer, number of dimensions.
    maxs: List of integers, maximum index in each dimension.
    mins: List of integers, minimum index in each dimension.
    weights: List of integers, weight of each dimension.

  Properties:
    dims: Tuple of integers, all dimension indices.
    sizes: Tuple of integers, size of each linearized dimension.
  """
  def __init__(self, rattrs: Sequence[Sequence[int]]):
    """Initialize the Linearizer with the given relative attribute tuples.

    Args:
      rattrs: Sequence of relative attributes. Each attribute is a sequence of
        integers.
    """
    self.num_dim = len(rattrs[0])
    self.maxs = [0] * self.num_dim
    self.mins = [0] * self.num_dim
    self.weights = [1] * self.num_dim
    for d in self.dims:
      self.maxs[d] = max(rattr[d] for rattr in rattrs)
      self.mins[d] = min(rattr[d] for rattr in rattrs)
    for d in range(1, self.num_dim):
      self.weights[d] = self.weights[d - 1] * self.sizes[d - 1]

  @property
  def dims(self) -> Tuple[int, ...]:
    return tuple(range(self.num_dim))

  @property
  def sizes(self) -> Tuple[int, ...]:
    # make sure different rows do not overlap
    return tuple((self.maxs[d] - self.mins[d] + 1) * 2 - 1 for d in self.dims)

  def apply(self, rattr: Sequence[int]) -> int:
    return sum((rval - min_val) * weight
               for rval, weight, min_val in zip(rattr, self.weights, self.mins))

  def restore(self, rattr: int) -> Tuple[int, ...]:
    restored_attr = []  # type: List[int]
    for d in reversed(self.dims):
      rval = rattr // self.weights[d]
      rattr -= rval * self.weights[d]
      restored_attr.append(self.mins[d] + rval)
    return tuple(reversed(restored_attr))

  # pylint: disable=function-redefined
  @overload
  def __call__(self, rattr: Sequence[int]) -> int:
    ...

  # pylint: disable=function-redefined
  @overload
  def __call__(self, rattr: int) -> Tuple[int, ...]:
    ...

  def __call__(self, rattr):
    if isinstance(rattr, int):
      return self.restore(rattr)
    if isinstance(rattr, Sequence) and isinstance(rattr[0], int):
      return self.apply(rattr)
    raise TypeError('rattr needs to be an int or a Sequence of int')

def range_from_middle(n: int) -> Iterator[int]:
  """A range function that yields number from the middle to the sides.

  Args:
    n: Integer, the upper bound of the range.
  Yields:
    Integers, starting from the n / 2 towards 0 and n - 1.
  """
  middle = n // 2
  if n % 2 == 0:
    for shift in range(0, middle):
      yield middle - shift - 1
      yield middle + shift
  else:
    yield middle
    for shift in range(1, middle + 1):
      yield middle - shift
      yield middle + shift

def shuffle_range(n: int) -> Iterator[int]:
  lst = list(range(n))
  random.shuffle(lst)
  return iter(lst)

def set_optimizations(optimizations: Sequence[str]) -> None:
  Expression.set_optimizations(optimizations)

_temporal_cse_counter = 0
def temporal_cse(stencil: 'soda.core.Stencil'   # type: ignore
                 ) -> 'soda.core.Stencil':  # type: ignore
  """Eliminate temporal common subexpressions.

  Eliminate temporal common subexpressions. The stencil object will be modified.

  Args:
    stencil: soda.core.Stencil object to work on.

  Returns:
    Modified stencil object.
  """
  _logger.debug('invoke stencil temporal common subexpression elimination')
  propagate_type = lambda node: base.propagate_type(node, stencil.symbol_table)

  # pylint: disable=unsubscriptable-object
  def visitor(node: ir.Node, args: Tuple[Dict[ir.BinaryOp, str],
                                         Set[ir.BinaryOp]]) -> ir.Node:
    """Visitor for temporal common subexpression elimination.

    Args:
      args: Tuple of (cses, used); cses is a dict mapping expressions to names
          of the new variables; used is a dict mapping used common
          subexpressions to those that have common subexpressions recursively
          eliminated.
    Returns:
      Optimized ir.Node with temporal common subexpressions eliminated.
    """
    try:
      cses, used = args
      expression = Expression(node)
      if expression.best_schedule is not None:
        if not expression.best_schedule.common_subexpressions:
          _logger.debug('no temporal_cse found')
          return node
        _logger.debug('best schedule: (cost: %d)',
                      expression.best_schedule.cost)
        expression.best_schedule.print_tree()
        for expr in expression.best_schedule.common_subexpressions:
          expr = propagate_type(expr)
          # pylint: disable=global-statement
          global _temporal_cse_counter
          cses[expr] = 'tcse_var_%d' % _temporal_cse_counter
          stencil.symbol_table[cses[expr]] = expr.haoda_type
          _logger.debug('common subexpression: %s <= %s',
                        cses[expr], ir.unparenthesize(expr))
          _temporal_cse_counter += 1
        return expression.best_schedule.get_ir_node_with_cse(cses, used)
    except Expression.CannotHandle:
      pass
    return node

  new_local_stmts = []
  cses = {}   # type: Dict[ir.BinaryOp, str]
  used = {}   # type: Dict[ir.BinaryOp, ir.BinaryOp]
  transform = lambda node: propagate_type(node).visit(visitor, (cses, used))
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    cses = {}
    used = {}
    stmt.expr = transform(stmt.expr)
    stmt.let = tuple(map(transform, stmt.let))
    # used points to old exprs so here it has to propagate type again
    cses = {propagate_type(used[k]): v for k, v in cses.items() if k in used}
    for expr, var in cses.items():
      new_local_stmts.append(grammar.LocalStmt(
          ref=ir.Ref(name=var, lat=None, idx=(0,) * stencil.dim),
          haoda_type=expr.haoda_type, expr=expr, let=stmt.let))
      _logger.debug('temporal cse stmt: %s', new_local_stmts[-1])
  stencil.local_stmts.extend(new_local_stmts)

  # invalidate cached_property
  del stencil.__dict__['symbol_table']
  del stencil.__dict__['local_names']
  del stencil.__dict__['local_types']

  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    _logger.debug('simplify %s', stmt.name)
    stmt.expr = arithmetic.simplify(stmt.expr)
    stmt.let = arithmetic.simplify(stmt.let)
  return stencil

class ScheduleBase:
  """Base class of Schedule and Schedules.

  Attributes:
    rattrs: Tuple of relative attributes.
    aattrs: Tuple of absolute attributes or None.
  """
  def __init__(self, rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[AbsoluteAttr, ...]]) -> None:
    self.rattrs = rattrs
    self.aattrs = aattrs

  def __getitem__(self, key: int) -> Tuple[RelativeAttr,
                                           Optional[AbsoluteAttr]]:
    return self.rattrs[key], None if self.aattrs is None else self.aattrs[key]

  def __len__(self) -> int:
    return len(self.rattrs)

  def __iter__(self) -> Iterator[Tuple[RelativeAttr, Optional[AbsoluteAttr]]]:
    yield from zip(self.rattrs, self.aattrs or itertools.repeat(None))

class CommSchedule(ScheduleBase):
  """A schedule of an expression.

  A schedule represents a general schedule of n operands, described as a binary
  tree.

  Attributes:
    left: Left child of the schedule. It can be an integer if it is a leaf node,
        representing the index to the attributes; otherwise it is a
        CommSchedule.
    right: Right child of the schedule.
    distance: Distance between the left child and the right child.
  Properties:
    norm_attrs: Generator of all normalized attributes as Iterator[Attr].
    uniq_expr_dict: Unique expressions of this schedule as
        Dict[Tuple[Attr, ...], CommSchedule].
    uniq_expr_set: Unique expressions of this schedule as
        Set[Tuple[Attr, ...]].
    cost: Number of operations required for this schedule.
    common_subexpressions: Tuple of ir.BinaryOp, normalized.
    op_type: Class of self.operator, only works if self.operator is not None.
  """
  def __init__(self, left: Union['CommSchedule', int, None],
               right: Union['CommSchedule', int, None],
               distance: RelativeAttr,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[AbsoluteAttr, ...]] = None) -> None:
    self.left, self.right, self.distance = left, right, distance
    super().__init__(rattrs, aattrs)
    self._len = 1   # number of operations
    for child in left, right:
      if isinstance(child, CommSchedule):
        self._len += len(child)

  def __len__(self) -> int:
    """Number of operations."""
    return self._len

  def __lt__(self, rhs: 'CommSchedule') -> bool:
    return self.cost < rhs.cost

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, CommSchedule):
      return NotImplemented
    return self.norm_attr_set == other.norm_attr_set

  def __hash__(self) -> int:
    return hash(self.norm_attr_set)

  def __str__(self) -> str:
    return self.to_str_with_offset(0)

  def to_str_with_offset(self, offset: int = 0) -> str:
    """Return the string representation assuming an offset.
    """
    if isinstance(self.left, CommSchedule):
      left = self.left.to_str_with_offset(offset)
    else:
      left = str(self.left)
    offset += self.distance
    if isinstance(self.right, CommSchedule):
      right = self.right.to_str_with_offset(offset)
    else:
      right = str(self.right)
    return '(%s==%s=>%s)' % (left, self.distance, right)

  def print_tree(self, printer=_logger.debug) -> None:
    base.print_tree(self.ir_node, printer)

  def bind_expression(self, expression: Optional['Expression']) \
      -> 'CommSchedule':
    """Bind an Expression to the schedule.
    """
    if expression is None:
      del self.aattrs_as_ir_nodes   # type: ignore
      del self.linearizer   # type: ignore
      del self.aattr_table  # type: ignore
      del self.operator   # type: ignore
    else:
      self.aattrs_as_ir_nodes = expression.aattrs_as_ir_nodes
      self.linearizer = expression.linearizer
      self.aattr_table = expression.aattr_table
      self.operator = expression.operator
    for child in self.left, self.right:
      if isinstance(child, CommSchedule):
        child.bind_expression(expression)
    return self

  @property
  def children(self) -> Iterator['CommSchedule']:
    if not hasattr(self, 'yielded_children'):
      self.yielded_children = []  # type: List[CommSchedule]
    yield from self.yielded_children
    yield from self.children_gen

  @cached_property.cached_property
  def children_gen(self) -> Iterator['CommSchedule']:
    self.yielded_children.append(self)
    yield self
    for child in self.left, self.right:
      if isinstance(child, CommSchedule):
        for schedule in child.children:
          self.yielded_children.append(schedule)
          yield schedule

  @cached_property.cached_property
  def cost(self) -> int:
    return len(set(self.children))

  def get_attrs_with_offset(self, offset: int = 0) -> Iterator[Attr]:
    """Generate all attributes with the given offset.

    Args:
      offset: The offset of the smallest relative attribute.

    Yields:
      Attributes in this schedule, NOT necessarily sorted by their relative
      attributes.
    """
    if isinstance(self.left, CommSchedule):   # Left child is not a leaf.
      yield from self.left.get_attrs_with_offset(offset)
    else:                                     # Left child is a leaf.
      if self.aattrs is None:  # Null absolute attributes.
        yield offset
      else:                   # Non-null absolute attributes.
        yield offset, self.left

    offset += self.distance

    if isinstance(self.right, CommSchedule):  # Right child is not a leaf.
      yield from self.right.get_attrs_with_offset(offset)
    else:                                     # Right child is a leaf.
      if self.aattrs is None:  # Null absolute attributes.
        yield offset
      else:                   # Non-null absolute attributes.
        yield offset, self.right

  @property
  def norm_attrs(self) -> Iterator[Attr]:
    return self.get_attrs_with_offset()

  @cached_property.cached_property
  def norm_attr_set(self) -> FrozenSet[Attr]:
    return frozenset(self.norm_attrs)

  @cached_property.cached_property
  def uniq_expr_set(self) -> Set[FrozenSet[Attr]]:
    """Unique expressions of this schedule.

    Returns:
      A dict mapping norm_attr_sets to a list of schedules whose normalized
      attributes equals the keys.
    """
    exprs = set()
    exprs.add(self.norm_attr_set)
    for child in self.left, self.right:
      if isinstance(child, CommSchedule):
        exprs |= child.uniq_expr_set
    return exprs

  @property
  def uniq_expr_dict(self) -> Dict[FrozenSet[Attr], List['CommSchedule']]:
    """Unique expressions of this schedule.

    Returns:
      A dict mapping norm_attr_sets to a list of schedules whose normalized
      attributes equals the keys.
    """
    exprs = OrderedDict()   # type: Dict[FrozenSet[Attr], List[CommSchedule]]
    exprs[self.norm_attr_set] = [self]
    for child in self.left, self.right:
      if isinstance(child, CommSchedule):
        for attrs, schedules in child.uniq_expr_dict.items():
          exprs.setdefault(attrs, []).extend(schedules)
    return exprs

  @property
  def op_type(self) -> Type[ir.BinaryOp]:
    return REDUCTION_OPS[self.operator]

  def get_ir_node_with_offset(self, offset: int = 0) -> ir.BinaryOp:
    """Get the IR node with the given offset.

    Args:
      offset: The offset of the smallest relative attribute.

    Returns:
      An ir.BinaryOp as the root of the IR.
    """
    if isinstance(self.left, CommSchedule):   # Left child is not a leaf.
      left_child = self.left.get_ir_node_with_offset(offset)  # type: ir.Node
    else:                                     # Left child is a leaf.
      left_child = assemble_attr(self.linearizer(offset),
                                 self.aattr_table[self.left])

    offset += self.distance

    if isinstance(self.right, CommSchedule):  # Right child is not a leaf.
      right_child = self.right.get_ir_node_with_offset(offset)  # type: ir.Node
    else:                                     # Right child is a leaf.
      right_child = assemble_attr(self.linearizer(offset),
                                  self.aattr_table[self.right])

    return self.op_type(operator=(self.operator,),
                        operand=(left_child, right_child))

  @cached_property.cached_property
  def ir_node(self) -> ir.BinaryOp:
    return self.get_ir_node_with_offset(self.rattrs[0])

  def get_ir_node_with_cse(
      self, cses: Dict[ir.Node, str],
      used: Optional[Dict[ir.Node, ir.Node]] = None) -> ir.Node:
    return mutator.replace_expressions(self.ir_node, cses, used)

  @cached_property.cached_property
  def common_subexpressions(self) -> Tuple[ir.BinaryOp, ...]:
    exprs = tuple(aattr for aattr in self.aattrs_as_ir_nodes
                  if isinstance(aattr, ir.BinaryOp))
    absolute_cses = tuple(
        aattr for aattr, count in OrderedCounter(exprs).items() if count > 1)
    for child in self.children:
      base.print_tree(mutator.normalize(child.ir_node))
    relative_cses = tuple(mutator.normalize(schedules[0].ir_node)
                          for schedules in self.uniq_expr_dict.values()
                          if len(schedules) > 1)
    for relative_cse in relative_cses:
      _logger.debug('relative cse: %s', relative_cse)
    return absolute_cses + relative_cses

class CommSchedules(ScheduleBase):
  """Schedules of an Expression.

  Class Attributes:
    range_func: The range function to use for range(n).
    skip: Whether to skip iterations if cost has exceeded the current minimum.
    lazy: Whether to evaluate the Cartesian product lazily.

  Attributes:
    operands: String of binary mask of the operands.
    cache: A mapping from operands to CommSchedules, or None.
    stat: A list of [cache_hit, cache_miss, loop 1 trip count, loop 2 trip
        count, loop 3 trip count], or None.
    max_cost: The cut-off cost, or None. If not None, any schedule must has
        less cost than this value to be included in the schedules.
  """
  range_func = range_from_middle
  skip = True
  lazy = True
  @staticmethod
  def set_optimizations(optimizations: Sequence[str]) -> None:
    if 'reorder-exploration' in optimizations:
      CommSchedules.range_func = range_from_middle
    if 'no-reorder-exploration' in optimizations:
      CommSchedules.range_func = lambda n: iter(range(n))
    if 'skip-with-partial-cost' in optimizations:
      CommSchedules.skip = True
    if 'no-skip-with-partial-cost' in optimizations:
      CommSchedules.skip = False
    if 'lazy-evaluation' in optimizations:
      CommSchedules.lazy = True
    if 'no-lazy-evaluation' in optimizations:
      CommSchedules.lazy = False

  def __init__(self, rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[AbsoluteAttr, ...]] = None,
               operands: Optional[str] = None,
               # pylint: disable=unsubscriptable-object
               cache: Optional[Dict[Tuple[int, ...], 'CommSchedules']] = None,
               stat: Optional[List[int]] = None,
               max_cost: Optional[int] = None,
               timeout: Optional[int] = None) -> None:
    super().__init__(rattrs, aattrs)
    if operands is None:
      self.operands = '1' * len(self.rattrs)
    else:
      self.operands = operands
    self.cache = cache
    if cache is not None:
      cache[self.key(self.operands)] = self
    if stat is None:
      self.stat = [0, 0, 0, 0, 0]
    else:
      self.stat = stat
    if max_cost is None:
      self.max_cost = collections.Counter(self.operands)['1']
    else:
      self.max_cost = max_cost
    self.timeout = 300
    if timeout is not None:
      self.timeout = timeout

  def __iter__(self) -> Iterator[CommSchedule]:   # type: ignore
    if hasattr(self, 'schedules'):
      return iter(getattr(self, 'schedules'))
    return self.generator

  def key(self, operands) -> Tuple[int, ...]:
    offset = self.rattrs[next(idx for idx, bit in enumerate(operands)
                              if bit == '1')]
    key = [self.rattrs[idx] - offset for idx, bit in enumerate(operands)
           if bit == '1']
    if self.aattrs is not None:
      key.extend(self.aattrs[idx] for idx, bit in enumerate(operands)
                 if bit == '1')
    return tuple(key)

  @property
  def generator(self) -> Iterator[CommSchedule]:
    """Generates possible schedules via dynamic programming.

    This generator will lazily materialize sub-problems for the Cartesian
    product.

    Yields:
      One CommSchedule at a time. If self.skip is on, the cost of generated
      CommSchedule with be monotonically decreasing.
    """
    n = collections.Counter(self.operands)['1']
    num_operands = len(self.rattrs)
    indices = [i for i in range(num_operands) if self.operands[i] == '1']
    schedules = []
    skipped = False
    if n == 1:
      schedule = self.aattrs[indices[0]] if self.aattrs is not None else None
      schedules.append(schedule)
      self.schedules = schedules
      self.max_cost = 0
      yield schedule  # type: ignore
      return
    #if self.aattrs is None or len(set(
    #    self.aattrs[i] for i in range(num_operands)
    #    if self.operands[i] == '1')) == 1:
    #  selector = lambda indices, m: [tuple(indices[1:m])]
    #  _logger.debug('using associative selector for %s', self.operands)
    #else:
    #  selector = lambda indices, m: itertools.combinations(indices[1:], m)
    #  _logger.debug('using commutative selector for %s', self.operands)
    selector = lambda indices, m: itertools.combinations(indices[1:], m)
    for m in CommSchedules.range_func(n - 1):
      selections = selector(indices, m)
      for selection in selections:
        self.stat[2] += 1
        left_indices = (indices[0],) + selection
        right_indices = [i for i in indices if i not in left_indices]
        left_operands = ''.join('1' if i in left_indices else '0'
                                for i in range(num_operands))
        right_operands = ''.join('1' if i in right_indices else '0'
                                 for i in range(num_operands))
        for left in self.get_schedules(left_operands):
          self.stat[3] += 1
          left_cost = 1
          if isinstance(left, CommSchedule):
            left_cost += left.cost
          if self.skip and left_cost > self.max_cost:
            skipped = True
            continue
          for right in self.get_schedules(right_operands):
            self.stat[4] += 1
            right_cost = 1
            if isinstance(right, CommSchedule):
              right_cost += right.cost
            if self.skip and right_cost > self.max_cost:
              skipped = True
              continue
            distance = self.rattrs[right_indices[0]]
            distance -= self.rattrs[left_indices[0]]
            schedule = CommSchedule(left, right, distance,  # type: ignore
                                    self.rattrs, self.aattrs)
            cost = schedule.cost  # type: ignore
            if cost < self.max_cost:
              self.max_cost = cost
            schedules.append(schedule)
            yield schedule  # type: ignore
    self.schedules = schedules

  schedule_cache = {}   # type: Dict[CommSchedule, CommSchedule]
  def make_schedule(self, left, right, distance):
    new_schedule = CommSchedule(left, right, distance, self.rattrs, self.aattrs)
    schedule = CommSchedules.schedule_cache.get(new_schedule)
    if schedule is not None:
      return schedule
    CommSchedules.schedule_cache[new_schedule] = new_schedule
    return new_schedule

  @property
  def best(self) -> CommSchedule:
    best = None
    try:
      with util.timeout(self.timeout):
        for schedule in self:
          if best is None or schedule.cost < best.cost:
            best = schedule
            _logger.debug('schedule: %s', best)
            _logger.info('cost: %d', best.cost)
    except TimeoutError:
      pass
    self.print_stats()
    if best is None:
      raise util.InternalError('cannot find best schedule')
    return best

  @property
  def cache_hit(self) -> int:
    return self.stat[0]

  @property
  def cache_miss(self) -> int:
    return self.stat[1]

  @property
  def cache_hit_rate(self) -> float:
    try:
      return self.cache_hit / (self.cache_hit + self.cache_miss)
    except ZeroDivisionError:
      return float('nan')

  def get_schedules(self, operands: str) -> Iterator[CommSchedule]:
    """Get schedules with the given operands.

    If self.cache is not None and the same arguments were given in previous runs
    of this object, the result will be fetched from the cache.

    Args:
      operands: Bit-mask of the operands.
    Returns:
      CommSchedules with the given operands.
    """
    if self.cache is not None:
      schedules = self.cache.get(self.key(operands))
      if schedules is not None:
        self.stat[0] += 1
        if hasattr(schedules, 'schedules'):
          return iter(schedules.schedules)  # type: ignore
        return schedules.generator
    self.stat[1] += 1
    return CommSchedules(
        self.rattrs, self.aattrs, operands=operands, cache=self.cache,
        stat=self.stat, max_cost=min(
            self.max_cost, collections.Counter(operands)['1'])).generator

  def print_stats(self,
                  logger: Callable[..., None] = _logger.info) -> None:
    logger('loops: | L1: %d | L2: %d | L3: %d |', *self.stat[2:])
    logger('cache: | hit: %d | miss: %d | hit rate: %2.3f %% |',
           self.cache_hit, self.cache_miss, self.cache_hit_rate * 100)

class GreedySchedules(ScheduleBase):
  """Schedules of an Expression, found greedily.
  """
  def __init__(self, rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[
                   Union[AbsoluteAttr, CommSchedule, None], ...]] = None,
               cache: Optional[Dict] = None) -> None:
    super().__init__(rattrs, aattrs)  # type: ignore

  def linear_schedule(self, indices: Iterable[int]) -> CommSchedule:
    """Schedule the attributes linearily.

    Args:
      indices: Iterable of ints, indicating what attributes should be used.

    Returns:
      CommSchedule of the attributes, scheduled as a linear tree.
    """
    indices = tuple(indices)
    distance = self.rattrs[indices[-1]] - self.rattrs[indices[0]]
    other_args = distance, self.rattrs, self.aattrs
    if len(indices) == 2:
      if self.aattrs is None:
        return CommSchedule(None, None, *other_args)
      return CommSchedule(self.aattrs[indices[0]], self.aattrs[indices[-1]],
                          *other_args)
    if self.aattrs is None:
      return CommSchedule(self.linear_schedule(indices[:-1]), None, *other_args)
    return CommSchedule(self.linear_schedule(indices[:-1]),
                        self.aattrs[indices[-1]], *other_args)

  @property
  def generator(self) -> Iterator[CommSchedule]:
    attr_map = {attr: idx for idx, attr in enumerate(self)}
    reuses = {}   # type: Dict[CommSchedule, List[Tuple[int, int]]]
    for left, right in itertools.combinations(self, 2):
      left_rattr, left_aattr = left
      right_rattr, right_aattr = right
      operation = CommSchedule(
        left_aattr, right_aattr, right_rattr - left_rattr,
        self.rattrs, self.aattrs)
      if operation in reuses:
        continue
      #_logger.debug('look for operation: %s', operation)
      # look for reuse of this operation over all operands
      used = set()  # type: Set[int]
      reuses[operation] = []
      for idx_l, (rattr_l, aattr_l) in enumerate(self):
        if aattr_l != left_aattr or idx_l in used:
          continue
        rattr_r, aattr_r = rattr_l + right_rattr - left_rattr, right_aattr
        idx_r = attr_map.get((rattr_r, aattr_r))
        if idx_r is None or idx_r in used:
          continue

        reuses[operation].append((idx_l, idx_r))
        used |= {idx_l, idx_r}

    # filter out operations that cannot be reused
    # they may not all be useful because they overlap
    reuses = {k: v for k, v in reuses.items() if len(v) > 1}
    if not reuses:
      yield self.linear_schedule(range(len(self.rattrs)))
      return

    new_attrs = OrderedDict(enumerate(self))
    used = set()
    # only activate reused operations with the maximum reuse
    # this avoids some of the sub-optimalities
    max_reuse = max(map(len, reuses.values()))
    _logger.debug('max reuse: %d', max_reuse)

    for operation, reused_indices in sorted(
        reuses.items(), key=lambda item: (-len(item[1]), item[0].distance)):
      # filter out indices that have been used by previous reuse operations
      reused_indices = [(idx_l, idx_r) for idx_l, idx_r in reused_indices
                        if idx_l not in used and idx_r not in used]
      if len(reused_indices) < max_reuse:
        continue
      for idx_l, idx_r in reused_indices:
        _logger.debug('reusing %s for %s + %s', operation,
                      '%s:%s' % self[idx_l], '%s:%s' % self[idx_r])
        new_attrs[idx_l] = new_attrs[idx_l][0], operation   # type: ignore
        del new_attrs[idx_r]
        used |= {idx_l, idx_r}

    _logger.debug('new attrs: %s (%d)',
                  util.lst2str('%s:%s' % attr for attr in new_attrs.values()),
                  len(new_attrs))
    new_rattrs, new_aattrs = zip(*new_attrs.values())
    yield from GreedySchedules(new_rattrs, new_aattrs).generator

  @cached_property.cached_property
  def best(self) -> CommSchedule:
    return next(self.generator)

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    return

Schedule = CommSchedule
Schedules = GreedySchedules

class Expression:
  """An expression suitable for temporal CSE.

  Attributes:
    operator: String of the operator.
    operands: Tuple of all operands.
    rattrs: Tuple of relative attributes as integer offsets.
    aattrs: Tuple of absolute attributes as integer tags, or None.
    aattrs_as_ir_nodes: Tuple of absolute attributes as IR nodes.
    linearizer: A linearizer mapping relative attributes to tuples.
    aattr_table: A dict mapping absolute attributes to IR nodes.
  """
  @staticmethod
  def set_optimizations(optimizations: Sequence[str] = (
      'reorder-exploration', 'skip-with-partial-cost',
      'lazy-evaluation')) -> None:
    CommSchedules.set_optimizations(optimizations)

  class CannotHandle(Exception):
    def __init__(self, msg, details: str = '') -> None:
      details = details or (': ' + str(details))
      super().__init__('cannot handle ' + str(msg) + ' yet' + str(details))

  def __init__(self, polynomial: ir.Node) -> None:
    """Figure out whether a ir.Node is suitable for temporal CSE.

    Construct a TCSE Expression of the input polynomial. If it cannot be handled
    but is a valid ir.Node instance, it raises Expression.CannotHandle so that
    the recursive visitor can continue to find polynomials.

    Args:
      polynomial: ir.BinaryOp to work with.

    Raises:
      Expression.CannotHandle: If the input cannot be handled but is not error.
      TypeError: If the input is not an instance of ir.Node.
    """
    if isinstance(polynomial, ir.BinaryOp):
      if any(op != polynomial.operator[0]   # type: ignore
             for op in polynomial.operator):  # type: ignore
        raise Expression.CannotHandle('mixed operators',
                                      polynomial.operator)  # type: ignore
      self.operator = polynomial.operator[0]  # type: ignore
      if self.operator not in ('+', '*'):
        raise Expression.CannotHandle('%s operator' % self.operator)
      for operand in polynomial.operand:  # type: ignore
        if len(soda_visitor.get_load_set(operand)) > 1:
          raise Expression.CannotHandle('multi-index operands', operand)
      self.operands = tuple(sorted(
          polynomial.operand,   # type: ignore
          key=lambda x: tuple(reversed(soda_visitor.get_load_set(x)[0].idx))))
      rattrs, aattrs = zip(*map(extract_attr, self.operands))
      self.aattrs_as_ir_nodes = aattrs

      self.linearizer = Linearizer(rattrs)

      # linearize the relative attribtues and memorize the mapping
      self.rattrs = tuple(map(self.linearizer, rattrs))

      # linearize the absolute attributes
      if len(set(aattrs)) == 1:
        self.aattrs = None  # type: Optional[Tuple[int, ...]]
        self.aattr_table = {None: aattrs[0]} \
            # type: Dict[Optional[int], ir.Node]
      else:
        tag = 0
        operand_table = {}      # type: Dict[ir.Node, int]
        self.aattr_table = {}
        for aattr in aattrs:
          if aattr not in operand_table:
            operand_table[aattr] = tag
            self.aattr_table[tag] = aattr
            tag += 1
        self.aattrs = tuple(operand_table[aattr] for aattr in aattrs)

      _logger.debug(
          'polynomial: %s%s', self.operator, util.idx2str(self.operands))
      _logger.debug('rattrs: %s', util.idx2str(self.rattrs))
      _logger.debug('aattrs: %s', util.idx2str(self.aattrs_as_ir_nodes))
    elif isinstance(polynomial, ir.Node):
      raise Expression.CannotHandle(type(polynomial).__name__)
    else:
      raise TypeError('expect an instance of ir.BinaryOp')

  @cached_property.cached_property
  def schedules(self) -> Schedules:
    return Schedules(self.rattrs, self.aattrs, cache={})

  @cached_property.cached_property
  def best_schedule(self) -> Schedule:
    return self.schedules.best.bind_expression(self)
