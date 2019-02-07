from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, \
    Sequence, Set, Tuple, Type, Union
import collections
import ctypes
import enum
import itertools
import logging
import os
import subprocess

import cached_property

from haoda import ir
from haoda import util
from haoda.ir import arithmetic
from haoda.ir.arithmetic import base
from soda import grammar
from soda import mutator
from soda import visitor as soda_visitor

RelativeAttr = Union[int, Tuple[int, ...]]
Operation = Tuple[Union[RelativeAttr, Tuple[RelativeAttr, Any]], ...]

OrderedDict = collections.OrderedDict
class OrderedCounter(collections.Counter, collections.OrderedDict):
  pass

_logger = logging.getLogger().getChild(__name__)


REDUCTION_OPS = {
  '+': ir.AddSub,
  '*': ir.MulDiv
}

_temporal_cse_counter = 0
def temporal_cse(stencil: 'soda.core.Stencil') -> 'soda.core.Stencil':
  """Eliminate temporal common subexpressions.

  Eliminate temporal common subexpressions. The stencil object will be modified.

  Args:
    stencil: soda.core.Stencil object to work on.

  Returns:
    Modified stencil object.
  """
  _logger.debug('invoke stencil temporal common subexpression elimination')

  # pylint: disable=unsubscriptable-object
  def visitor(node: ir.Node, args: Tuple[Mapping[ir.BinaryOp, str],
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
          # pylint: disable=global-statement
          global _temporal_cse_counter
          cses[expr] = 'tcse_var_%d' % _temporal_cse_counter
          _logger.debug('common subexpression: %s <= %s',
                        cses[expr], ir.unparenthesize(expr))
          _temporal_cse_counter += 1
        return expression.best_schedule.get_ast_with_cse(cses, used)
    except Expression.CannotHandle:
      pass
    return node

  new_local_stmts = []
  transform = lambda node: base.propagate_type(
      node, stencil.symbol_table).visit(visitor, (cses, used))
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    cses, used = {}, {}
    stmt.expr = transform(stmt.expr)
    stmt.let = tuple(map(transform, stmt.let))
    cses = {used[k]: v for k, v in cses.items() if k in used}
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

  This base class provides the functionality of making an Operation from the
  relative attributes and absolute attributes.

  Attributes:
    rattr: Tuple of relative attributes.
    aattr: Tuple of absolute attributes or None.
  """
  def __init__(self, rattr: Tuple[RelativeAttr, ...],
               aattr: Optional[tuple]) -> None:
    self.rattr = rattr
    self.aattr = aattr

  def make_operation(self, idx: slice) -> Operation:
    """Make operation from the relative and absolute attributes.

    Args:
      idx: A slice representing which attributes to include in the operation.
    Returns:
      A tuple of (rattr, aattr).
    """
    offset = self.rattr[idx][0]
    if self.aattr is None:
      if isinstance(offset, int):
        normalize = lambda val: val - offset
      else:
        normalize = lambda val: tuple(x - y for x, y in zip(val, offset))
      return tuple(map(normalize, self.rattr[idx]))

    def normalize_int(val: Tuple[int, int]) -> Tuple[int, int]:
      rval, aval = val
      return rval - offset, aval
    def normalize_tuple(val: Tuple[Tuple[int, ...], Tuple[int, ...]]) \
        -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
      rval, aval = val
      return tuple(x - y for x, y in zip(rval, offset)), aval
    normalize = normalize_int if isinstance(offset, int) else normalize_tuple
    return tuple(map(normalize, zip(self.rattr[idx], self.aattr[idx])))

class Schedule(ScheduleBase):
  """A schedule of an expression.

  A schedule represents a general schedule of n operands, described by the
  relative attributes and absolute attrbiutes.

  Attributes:
    brepr: B-repr of the schedule.
    operations: List of slices representing the operations.
    operation_set: Set of (Operation, brepr) tuple.
    operator: String of the operator or None.
  Properties:
    cost: Number of operations required of this schedule.
    common_subexpressions: Tuple of ir.BinaryOp, normalized.
    op_type: Class of self.operator, only works if self.operator is not None.
  """
  def __init__(self, brepr: str, operations: List[slice],
               operation_set: Set[Tuple[Operation, str]],
               rattr: Tuple[RelativeAttr, ...],
               aattr: Optional[tuple] = None) -> None:
    self.brepr = brepr
    self.operations = operations
    self.operation_set = operation_set
    super().__init__(rattr, aattr)

  def __len__(self) -> int:
    """Number of operations."""
    return len(self.brepr) // 2

  def __lt__(self, rhs) -> bool:
    return self.cost < rhs.cost

  @cached_property.cached_property
  def cost(self) -> int:
    return len(self.operation_set) + 1  # because itself is also an operation

  @cached_property.cached_property
  def common_subexpressions(self) -> Tuple[ir.BinaryOp]:
    operations = tuple(mutator.normalize(self.get_ast(operation))
                       for operation in self.operations)
    return tuple(mutator.normalize(
        schedule for schedule, count in OrderedCounter(operations).items()
        if count > 1))

  @property
  def op_type(self) -> Type[ir.BinaryOp]:
    return REDUCTION_OPS[self.operator]

  @cached_property.cached_property
  # pylint: disable=unsubscriptable-object
  def brepr_index_table(self) -> Mapping[int, int]:
    mapping = {}
    node_count = 0
    for idx, bit in enumerate(self.brepr):
      if bit != '0':
        node_count += 1
        mapping[node_count] = idx + 1
    return mapping

  def bind_operator(self, operator: Optional[str]) -> 'Schedule':
    if operator is None:
      del self.operator
    else:
      self.operator = operator
    return self

  def bind_aattr(self, aattr: Optional[tuple]) -> 'Schedule':
    self.aattr = aattr
    return self

  def print_tree(self, printer=_logger.debug) -> None:
    printer('B-repr: %s', self.brepr)
    base.print_tree(self.get_ast(), printer)

  def get_brepr_slice(self, operation: slice) -> slice:
    stop = self.brepr_index_table[operation.stop]
    return slice(stop - (operation.stop - operation.start) * 2 + 1, stop)

  def get_brepr(self, operation: slice) -> str:
    return self.brepr[self.get_brepr_slice(operation)]

  def get_ast(self, operation: Optional[slice] = None) -> ir.BinaryOp:
    """Get the AST of an operation.

    Args:
      operation: A slice representing the operation, or None.
    Returns:
      An ir.BinaryOp as the root of the AST.
    """
    class Child(enum.Enum):
      LEFT = 0
      RIGHT = 1
    if operation is None:
      operation = slice(0, len(self.rattr))
    # pylint: disable=not-callable
    root = self.op_type(operator=(self.operator,), operand=(None, None))
    stack = [(root, Child.RIGHT), (root, Child.LEFT)]
    operands = (ir.Ref(name=aval, idx=rval, lat=None)
                for rval, aval in zip(self.rattr[operation],
                                      self.aattr[operation]))
    def add_child(stack: List[Tuple[ir.BinaryOp, Child]],
                  child: ir.BinaryOp) -> None:
      """Pop the task stack and add child to the task.

      Args:
        stack: List of (ir.BinaryOp, Child), meaning which child to add to which
            IR node.
        child: Child IR node that going to be added to the stack top node.
      Returns:
        None
      """
      op, side = stack.pop(-1)
      if side == Child.LEFT:
        op.operand = (child, None)
      else:
        op.operand = (op.operand[0], child)

    for bit in self.get_brepr(operation)[1:]:
      if bit == '0':
        # pylint: disable=not-callable
        child = self.op_type(operator=(self.operator,), operand=(None, None))
        add_child(stack, child)
        stack.extend(((child, Child.RIGHT), (child, Child.LEFT)))
      else:
        child = next(operands)
        add_child(stack, child)
    assert not stack, "task leftover: {}".format(
        util.lst2str(node for node, _ in stack))
    return root

  def get_ast_with_cse(self, cses: Dict[ir.Node, str],
                       used: Optional[Set[ir.Node]] = None) -> ir.BinaryOp:
    return mutator.replace_expressions(self.get_ast(), cses, used)

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

class Schedules(ScheduleBase):
  """Schedules of an Expression.

  Class Attributes:
    range_func: The range function to use for range(n).
    skip: Whether to skip iterations if cost has exceeded the current minimum.
    lazy: Whether to evaluate the Cartesian product lazily.

  Attributes:
    num_ops: Number of operations to consider, may not equal len(rattr) - 1.
    offset: Number of operations to skip.
    cache: A mapping from num_ops to a mapping from offset to Schedules, or
        None.
    stat: A list of [cache_hit, cache_miss, loop 1 trip count, loop 2 trip
        count, loop 3 trip count], or None.
    max_cost: The cut-off cost, or None. Any schedule with max_cost or
  """
  range_func = range_from_middle
  skip = True
  lazy = True
  libtcse = None
  @staticmethod
  def set_optimizations(optimizations: Sequence[str]) -> None:
    if 'reorder-exploration' in optimizations:
      Schedules.range_func = range_from_middle
    if 'no-reorder-exploration' in optimizations:
      Schedules.range_func = range
    if 'skip-with-partial-cost' in optimizations:
      Schedules.skip = True
    if 'no-skip-with-partial-cost' in optimizations:
      Schedules.skip = False
    if 'lazy-cartesian-product' in optimizations:
      Schedules.lazy = True
    if 'no-lazy-cartesian-product' in optimizations:
      Schedules.lazy = False
    if 'c-temporal-cse' in optimizations:
      tcse_so = os.path.join(os.path.dirname(__file__), "libtemporal-cse.so")
      try:
        with subprocess.Popen(
            ['make'], cwd=os.path.dirname(tcse_so),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) as make_proc:
          for msg in make_proc.stdout.read().decode().splitlines():
            _logger.info(msg)
          for msg in make_proc.stderr.read().decode().splitlines():
            _logger.info(msg)
        Schedules.libtcse = ctypes.CDLL(tcse_so)
      except OSError as e:
        _logger.warning(e)
        Schedules.libtcse = None
    if 'no-c-temporal-cse' in optimizations:
      Schedules.libtcse = None

  def __init__(self, rattr: Tuple[Union[int, Tuple[int, ...]]],
               aattr: Optional[tuple] = None,
               num_ops: int = None, offset: int = 0,
               # pylint: disable=unsubscriptable-object
               cache: Optional[Mapping[int, Mapping[int, 'Schedules']]] = None,
               stat: Optional[List[int]] = None,
               max_cost: int = None) -> None:
    super().__init__(rattr, aattr)
    self.num_ops = len(self.rattr) - 1 if num_ops is None else num_ops
    self.offset = offset
    self.cache = cache
    if cache is not None:
      cache.setdefault(self.num_ops, {})[self.offset] = self
    self.stat = stat
    if stat is None:
      self.stat = [0, 0, 0, 0, 0]
    self.max_cost = self.num_ops if max_cost is None else max_cost

  def __iter__(self) -> Iterator[Schedule]:
    return iter(self.schedules)

  @cached_property.cached_property
  def schedules(self) -> Tuple[Schedule, ...]:
    return tuple(self.generator)

  @property
  def generator(self) -> Iterator[Schedule]:
    if Schedules.lazy:
      return self.lazy_generator
    return self.materializing_generator

  @property
  def materializing_generator(self) -> Iterator[Schedule]:
    """Generates possible schedules via dynamic programming.

    This generator will materialize both sub-problems for the Cartesian product.

    Yields:
      One Schedule at a time. If self.skip is on, the cost of generated Schedule
      with be monotonically decreasing.
    """
    n, k = self.num_ops, self.offset
    if n == 0:
      yield Schedule('1', [], set(), self.aattr, self.rattr)
      return
    for m in Schedules.range_func(n):
      self.stat[2] += 1
      for prefix, suffix in itertools.product(
          self.get_schedules(m, k),
          self.get_schedules(n - m - 1, k + m + 1)):
        self.stat[3] += 1
        self.stat[4] += 1
        operations = []
        operation_set = set()
        # Only slices with len > 1 are operations.
        if m > 0:
          operations.append(slice(k, k + m + 1))
          operation_set.add((self.make_operation(operations[-1]),
                             prefix.brepr))
          if Schedules.skip and len(operation_set) >= self.max_cost:
            continue
        if n > m + 1:
          operations.append(slice(k + m + 1, k + n + 1))
          operation_set.add((self.make_operation(operations[-1]),
                             suffix.brepr))
          if Schedules.skip and len(operation_set) >= self.max_cost:
            continue
        operations.extend(prefix.operations)
        operation_set |= prefix.operation_set
        if Schedules.skip and len(operation_set) >= self.max_cost:
          continue
        operations.extend(suffix.operations)
        operation_set |= suffix.operation_set
        if Schedules.skip and len(operation_set) >= self.max_cost:
          continue
        self.max_cost = len(operation_set)
        yield Schedule('0{}{}'.format(prefix.brepr, suffix.brepr),
                        operations, operation_set,
                        self.rattr, self.aattr)

  @property
  def lazy_generator(self) -> Iterator[Schedule]:
    """Generates possible schedules via dynamic programming.

    This generator will lazily materialize sub-problems for the Cartesian
    product.

    Yields:
      One Schedule at a time. If self.skip is on, the cost of generated Schedule
      with be monotonically decreasing.
    """
    n, k = self.num_ops, self.offset
    if n == 0:
      yield Schedule('1', [], set(), self.aattr, self.rattr)
      return
    for m in Schedules.range_func(n):
      self.stat[2] += 1
      for prefix in self.get_schedules(m, k):
        self.stat[3] += 1
        prefix_operations = []
        prefix_operation_set = set()
        # Only slices with len > 1 are operations.
        if m > 0:
          prefix_operations.append(slice(k, k + m + 1))
          prefix_operation_set.add((
              self.make_operation(prefix_operations[-1]),
              prefix.brepr))
          if Schedules.skip and len(prefix_operation_set) >= self.max_cost:
            continue
        prefix_operations.extend(prefix.operations)
        prefix_operation_set |= prefix.operation_set
        if Schedules.skip and len(prefix_operation_set) >= self.max_cost:
          continue
        for suffix in self.get_schedules(n - m - 1, k + m + 1):
          self.stat[4] += 1
          operations = list(prefix_operations)
          operation_set = set(prefix_operation_set)
          # Only slices with len > 1 are operations.
          if n > m + 1:
            operations.append(slice(k + m + 1, k + n + 1))
            operation_set.add((
                self.make_operation(operations[-1]),
                suffix.brepr))
            if Schedules.skip and len(operation_set) >= self.max_cost:
              continue
          operations.extend(suffix.operations)
          operation_set |= suffix.operation_set
          if Schedules.skip and len(operation_set) >= self.max_cost:
            continue
          self.max_cost = len(operation_set)
          yield Schedule('0{}{}'.format(prefix.brepr, suffix.brepr),
                          operations, operation_set,
                          self.rattr, self.aattr)

  @cached_property.cached_property
  def best(self) -> Schedule:
    if Schedules.libtcse is not None:
      n = len(self.rattr)

      # prepare C arguments
      c_rattrs = (ctypes.c_int64 * n)()
      if self.aattr is None:
        c_aattrs = ctypes.POINTER(ctypes.c_int64)()
      else:
        c_aattrs = (ctypes.c_int64 * n)()
      c_n = ctypes.c_uint64(n)
      c_cost = ctypes.c_uint64()
      c_brepr = (ctypes.c_char * (2 * n))()
      c_operations = (ctypes.c_uint64 * (2 * (n - 2)))()
      c_stat = (ctypes.c_uint64 * len(self.stat))()

      # prepare relative attributes
      if isinstance(self.rattr[0], int):
        for i in range(n):
          c_rattrs[i] = self.rattr[i]
      else:
        # relative attributes are tuples; linearize them
        def linearize(rattr: Sequence[int], weights: Sequence[int]) -> int:
          return sum(rattr * weight for rattr, weight in zip(rattr, weights))

        num_dim = len(self.rattr[0])
        maxs = [None] * num_dim
        mins = [None] * num_dim
        weights = [1] * num_dim
        for d in range(num_dim - 1):
          maxs[d] = max(rattr[d] for rattr in self.rattr)
          mins[d] = min(rattr[d] for rattr in self.rattr)
        for d in range(1, num_dim):
          weights[d] = weights[d - 1] * (maxs[d - 1] - mins[d - 1] + 1)
        for i in range(n):
          c_rattrs[i] = linearize(self.rattr[i], weights)

      # prepare absolute attributes
      if self.aattr is not None:
        tag = 0
        aattr_table = {}
        for aattr in self.aattr:
          if aattr not in aattr_table:
            aattr_table[aattr] = tag
            tag += 1
        for i in range(n):
          c_aattrs[i] = aattr_table[self.aattr[i]]

      Schedules.libtcse.TemporalCse(
          c_rattrs, c_aattrs, c_n, ctypes.byref(c_cost),
          c_brepr, c_operations, c_stat)

      for i in range(len(self.stat)):
        self.stat[i] = c_stat[i]
      operations = tuple(slice(c_operations[i * 2], c_operations[i * 2 + 1])
                         for i in range(n - 2))
      c_best = Schedule(c_brepr.value.decode(), operations, None,
                        self.rattr, self.aattr)
      c_best.cost = c_cost.value
      return c_best

    best = None
    if self.skip:
      for best in self:
        pass
      self.print_stats()
      return best
    for schedule in self:
      if best is None or schedule.cost < best.cost:
        best = schedule
        _logger.debug('current cost: %d', best.cost)
    self.print_stats()
    return best

  @property
  def cache_hit(self) -> int:
    return self.stat[0]

  @property
  def cache_miss(self) -> int:
    return self.stat[1]

  @property
  def cache_hit_rate(self) -> float:
    return self.cache_hit / (self.cache_hit + self.cache_miss)

  def get_schedules(self, num_ops, offset) -> Tuple[Schedule]:
    """Get schedules with the given operation number and offset.

    If self.cache is not None and the same arguments were given in previous runs
    of this object, the result will be fetched from the cache.

    Args:
      num_ops: Number of operations to consider, may not equal len(rattr) - 1.
      offset: Number of operations to skip.
    Returns:
      Schedules with the given operation number and offset.
    """
    if self.cache is not None:
      if num_ops in self.cache:
        if offset in self.cache[num_ops]:
          schedules = self.cache[num_ops][offset]
          self.stat[0] += 1
          return schedules.schedules
    self.stat[1] += 1
    return Schedules(self.rattr, self.aattr, num_ops=num_ops, offset=offset,
                     cache=self.cache, stat=self.stat,
                     max_cost=self.max_cost + 1).schedules

  def print_stats(self,
                  logger: Callable[[str, Any], None] = _logger.info) -> None:
    logger('loops: | L1: %d | L2: %d | L3: %d |', *self.stat[2:])
    logger('cache: | hit: %d | miss: %d | hit rate: %2.3f %% |',
           self.cache_hit, self.cache_miss, self.cache_hit_rate * 100)

class Expression:
  """An expression suitable for temporal CSE.

  Attributes:
    operator: String of the operator.
    operands: Tuple of all operands.
    rattr: Tuple of relative attributes.
    aattr: Tuple of absolute attributes or None.
  """
  nullify_uniform_aattr = True
  @staticmethod
  def set_optimizations(optimizations: Sequence[str] = (
      'reorder-exploration', 'skip-with-partial-cost',
      'lazy-cartesian-product', 'nullify-uniform-aattr')) -> None:
    if 'nullify-uniform-aattr' in optimizations:
      uniform_uniform_aattr = True
    if 'no-nullify-uniform-aattr' in optimizations:
      uniform_uniform_aattr = False
    Schedules.set_optimizations(optimizations)

  class CannotHandle(Exception):
    def __init__(self, msg, details: str = '') -> None:
      details = details or (': ' + str(details))
      super().__init__('cannot handle ' + str(msg) + ' yet' + str(details))

  def __init__(self, polynomial: ir.BinaryOp) -> None:
    """Figure out whether a ir.BinaryOp is suitable for temporal CSE.

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
      if any(op != polynomial.operator[0] for op in polynomial.operator):
        raise Expression.CannotHandle('mixed operators', polynomial.operator)
      self.operator = polynomial.operator[0]
      if self.operator not in ('+', '*'):
        raise Expression.CannotHandle('%s operator' % self.operator)
      for operand in polynomial.operand:
        if len(soda_visitor.get_load_set(operand)) > 1:
          raise Expression.CannotHandle('multi-index operands', operand)
      self.operands = tuple(sorted(
          polynomial.operand,
          key=lambda x: tuple(reversed(soda_visitor.get_load_set(x)[0].idx))))
      self.rattr = tuple(soda_visitor.get_load_set(operand)[0].idx
                         for operand in self.operands)
      self.aattr = tuple(soda_visitor.get_load_set(operand)[0].name
                         for operand in self.operands)
      _logger.debug(
          'polynomial: %s%s', self.operator, util.idx2str(self.operands))
      _logger.debug('rattr: %s', util.idx2str(self.rattr))
      _logger.debug('aattr: %s', util.idx2str(self.aattr))
    elif isinstance(polynomial, ir.Node):
      raise Expression.CannotHandle(type(polynomial).__name__)
    else:
      raise TypeError('expect an instance of ir.BinaryOp')

  @cached_property.cached_property
  def schedules(self) -> Schedules:
    if Expression.nullify_uniform_aattr:
      if self.aattr is not None:
        if len(set(self.aattr)) == 1:
          return Schedules(self.rattr, None, cache={})
    return Schedules(self.rattr, self.aattr, cache={})

  @cached_property.cached_property
  def best_schedule(self) -> Schedule:
    best_schedule = self.schedules.best
    return best_schedule.bind_operator(self.operator).bind_aattr(self.aattr)


def set_optimizations(optimizations: Sequence[str]) -> None:
  Schedules.set_optimizations(optimizations)
