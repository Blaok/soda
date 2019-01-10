from typing import Dict, Iterable, Iterator, Optional, Set, Tuple, Union
import collections
import itertools
import logging

import cached_property

from haoda import ir
from haoda import util
from haoda.ir import arithmetic
from haoda.ir.arithmetic import base
from soda import grammar
from soda import mutator
from soda import visitor as soda_visitor

_OrderedDict = collections.OrderedDict
_logger = logging.getLogger().getChild(__name__)

_temporal_cse_counter = 0
def temporal_cse(stencil):
  """Eliminate temporal common subexpressions.

  Eliminate temporal common subexpressions. The stencil object will be modified.

  Args:
    stencil: soda.core.Stencil object to work on.
  """
  _logger.debug('invoke stencil temporal common subexpression elimination')

  def visitor(node: ir.Node, args: Tuple[dict, set]) -> ir.Node:
    """Visitor for temporal common subexpression elimination.

    Args:
      args: Tuple of (cses, used). cses is a dict mapping expressions to names
          of the new variables. used is a set that will hold the expressions of
          all used variables.
    Returns:
      Optimized ir.Node with temporal common subexpressions eliminated.
    """
    try:
      cses, used = args
      schedules = Schedules(node)
      if not schedules.best.common_subexpressions:
        _logger.debug('no temporal_cse found')
        return node
      _logger.debug('best schedule: %s', schedules.best)
      for expr in schedules.best.common_subexpressions:
        # pylint: disable=global-statement
        global _temporal_cse_counter
        cses[expr] = 'temporal_cse%d' % _temporal_cse_counter
        _temporal_cse_counter += 1
      return schedules.best.expr_with_cse(cses, used)
    except Schedules.CannotHandle:
      return node

  new_local_stmts = []
  transform = lambda node: base.propagate_type(
      node, stencil.symbol_table).visit(visitor, (cses, used))
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    cses, used = {}, set()
    stmt.expr = transform(stmt.expr)
    stmt.let = tuple(map(transform, stmt.let))
    cses = {k: v for k, v in cses.items() if k in used}
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
    stmt.expr = arithmetic.simplify(stmt.expr)
    stmt.let = arithmetic.simplify(stmt.let)

class Schedule(str):
  """Class representing a schedule of an expression.

  A schedule can represent a general schedule of n operands, or a specific
  schedule of a concrete expression, described by the operator and operands.

  Attributes:
    operator: String of the operator or None.
    operands: Iterable of all operands or None.
  Properties:
    bound: Whether this schedule is bound to a specific expression.
    sub_schedule_generator: A generator that yields all sub-schedules.
    sub_schedules: A tuple of all sub-schedules.
    cost: Number of "add" operations required of this schedule.
    expr: AST of the expression represented, available only if bound is True.
  """
  def __init__(self, obj):
    super().__init__()
    self.operator, self.operands = None, None

  def __getitem__(self, key: Union[slice, int]) -> 'Schedule':
    """Get a slice from current schedule.

    If self.bound is True, the result will be bound with proper operator and
    operands. Otherwise, it is the same as str.__getitem__.

    Args:
      key: A slice or int representing the items to get.
    Returns:
      The resulting Schedule.
    """
    schedule = Schedule(super().__getitem__(key))
    if self.bound:
      if isinstance(key, slice):
        if key.step is not None:
          raise ValueError('cannot get item with step != 1')
        start = key.start and self.count('1', 0, key.start)
        stop = key.stop and ((start or 0) + schedule.count('1'))
        _logger.debug('[start:stop]: [%s:%s]', start, stop)
        schedule = schedule.bind(self.operator, self.operands[start:stop])
      else:
        schedule = schedule.bind(self.operator, self.operands[key])
    return schedule

  def __lt__(self, rhs) -> bool:
    return self.cost < rhs.cost

  def __eq__(self, other) -> bool:
    if self.bound:
      if isinstance(other, Schedule) and other.bound:
        return str(self), self.operator, self.normalized_operands == \
            str(other), other.operator, other.normalized_operands
      return False
    if isinstance(other, Schedule) and other.bound:
      return False
    return str(self) == str(other)

  def __hash__(self):
    return hash((str(self), self.operator, self.normalized_operands))

  def bind(self, operator: str, operands: Iterable[ir.Node],
           verbose: bool = False) -> 'Schedule':
    """Bind the schedule to an expression.

    If (operator, operands) doesn't equal (self.operator, self.operands), a new
    Schedule object will be constructed and returned with the given arguments.
    Otherwise, self is returned. In either case, self is not modified. Two None
    values will unbind the schedule.

    Args:
      operator: String of the operator or None.
      operands: Iterable of all operands or None.
      verbose: Whether to log bindings. Default to False.
    Raises:
      ValueError: If one and only one of operator and operands is None.
      RuntimeError: If the number of operands doesn't match the schedule.
    Returns:
      A schedule bound with the given operator and operands.
    """
    if operator is None and operands is not None or \
        operator is not None and operands is None:
      raise ValueError('operator and operands must both or neither be None')
    if (operator, operands) != (self.operator, self.operands):
      schedule = Schedule(self)
      schedule.operator, schedule.operands = operator, operands
      if operands is not None:
        if verbose:
          _logger.debug('bind %s to %s', util.idx2str(operands), self)
        if len(operands) != (len(self) + 1) // 2:
          raise RuntimeError('operands must have the same length as schedule')
      return schedule
    return self

  def unbind(self) -> 'Schedule':
    return self.bind(None, None)

  @property
  def bound(self) -> bool:
    return self.operator is not None and self.operands is not None

  @property
  def sub_schedule_generator(self) -> Iterator['Schedule']:
    num_1 = 0
    for i, c in enumerate(self):
      if c == '0':
        schedule = DisjointSubScheduleIterator(self, i).next()
        operands = self.operands[num_1:num_1 + (len(schedule) + 1) // 2]
        yield schedule.bind(self.operator, operands)
      else:
        num_1 += 1

  @property
  def sub_schedules(self) -> Tuple['Schedule']:
    return tuple(self.sub_schedule_generator)

  @property
  def normalized_operands(self) -> Optional[Tuple[ir.Node]]:
    if self.operands is None:
      return None
    return mutator.normalize(self.operands)

  @cached_property.cached_property
  def cost(self) -> int:
    return len(set(self.sub_schedules))

  @property
  def common_subexpressions(self) -> Tuple[ir.BinaryOp]:
    if self.bound:
      return tuple(mutator.normalize(
          schedule.expr for schedule, count in
          collections.Counter(self.sub_schedules).items() if count > 1))
    raise ValueError("unbound schedule doesn't have common expressions")

  @property
  def expr(self) -> ir.BinaryOp:
    return self.expr_with_cse({})

  def expr_with_cse(self, cses: Dict[ir.Node, str],
                    used: Set[ir.Node] = None) -> ir.BinaryOp:
    """Generate AST.

    Recursively construct the AST of the schedule represented by self. Self must
    be bound.

    Args:
      cses: Dict mapping common subexpressions to the new names.
      used: Set of used common subexpressions.
    Returns:
      The ir.BinaryOp as the AST.
    Raises:
      ValueError: If self is not bound.
    """
    if not self.bound:
      raise ValueError("unbound schedule doesn't have an expression")
    iterator = DisjointSubScheduleIterator(self, 1)
    if iterator.curr == '0':
      left_child = Schedule(next(iterator)).bind(
          self.operator,
          self.operands[:(iterator.pos + 1) // 2]).expr_with_cse(cses, used)
    else:
      iterator.skip()
      left_child = self.operands[(iterator.pos - 1) // 2]
    if iterator.curr == '0':
      prev_pos = iterator.pos
      right_child = Schedule(next(iterator)).bind(
          self.operator,
          self.operands[prev_pos // 2:]).expr_with_cse(cses, used)
    else:
      right_child = self.operands[iterator.pos // 2]
    children = [left_child, right_child]
    for i, child in enumerate(children):
      normalized_child = mutator.normalize(child)
      if normalized_child in cses:
        if used is not None:
          used.add(normalized_child)
        norm_idx = soda_visitor.get_normalize_index(child)
        new_var = cses[normalized_child]
        _logger.debug('replace %s with %s%s', child, new_var, norm_idx)
        children[i] = ir.Ref(name=new_var, idx=norm_idx, lat=None)
    return {'+': ir.AddSub, '*': ir.MulDiv}[self.operator](
        operator=self.operator, operand=children)

class DisjointSubScheduleIterator:
  """Iterator that yields disjoint sub-schedules.

  An iterator that yields disjoint sub-schedules of a schedule with an optional
  starting offset.

  Attributes:
    _schedule: Schedule to iterate over.
    pos: Current iterator offset.
  """
  def __init__(self, schedule: Schedule, start: int = 0):
    self._schedule, self.pos = schedule.unbind(), start

  def __next__(self) -> Schedule:
    """Find the next valid sub-schedule from the current offset.

    Returns:
      Next valid schedule.
    """
    start = self.pos
    if start >= len(self._schedule):
      raise StopIteration
    num_0, num_1 = 0, 0
    for i, c in enumerate(self._schedule[start:]):
      if c == '0':
        num_0 += 1
      else:
        num_1 += 1
      if num_1 - num_0 == 1:
        stop = start + i + 1
        break
    self.pos += stop - start
    return self._schedule[start:stop]

  def next(self) -> Schedule:
    return next(self)

  def skip(self, step: int = 1) -> None:
    self.pos += step

  @property
  def curr(self) -> Schedule:
    return self._schedule[self.pos]

class Schedules:
  """Generator for all schedules of an expression.

  Use dynamic programming to generate all schedules of an expression.

  Attributes:
    operator: String of the operator.
    operands: Tuple of all operands.
  """
  _schedule_cache = {0: (Schedule('1'),)}
  class CannotHandle(Exception):
    def __init__(self, msg, details: str = ''):
      details = details or (': ' + str(details))
      super().__init__('cannot handle ' + str(msg) + ' yet' + str(details))

  def __init__(self, polynomial: ir.BinaryOp):
    """Constructs the Schedules.

    Construct all possible schedules of the input polynomial. If it cannot be
    handled but is a valid ir.Node instance, it raises Schedules.CannotHandle
    so that the recursive visitor can continue to find polynomials.

    Args:
      polynomial: ir.BinaryOp to work with.
    Raises:
      Schedules.CannotHandle: If the input cannot be handled but is not error.
      TypeError: If the input is not an instance of ir.Node.
    """
    if isinstance(polynomial, ir.BinaryOp):
      if any(op != polynomial.operator[0] for op in polynomial.operator):
        raise Schedules.CannotHandle('mixed operators', polynomial.operator)
      self.operator = polynomial.operator[0]
      if self.operator not in ('+', '*'):
        raise Schedules.CannotHandle('%s operator' % self.operator)
      for operand in polynomial.operand:
        if len(soda_visitor.get_load_set(operand)) > 1:
          raise Schedules.CannotHandle('multi-index operands', operand)
      self.operands = tuple(sorted(
          polynomial.operand,
          key=lambda x: tuple(reversed(soda_visitor.get_load_set(x)[0].idx))))
      _logger.debug(
          'polynomial: %s%s', self.operator, util.idx2str(self.operands))
    elif isinstance(polynomial, ir.Node):
      raise Schedules.CannotHandle(type(polynomial).__name__)
    else:
      raise TypeError('expect an instance of ir.BinaryOp')

  def __iter__(self) -> Iterator[Schedule]:
    return iter(map(lambda x: x.bind(self.operator, self.operands),
                    self._get_b_reprs(len(self.operands) - 1)))

  @cached_property.cached_property
  def best(self) -> Schedule:
    return min(self)

  def _get_b_reprs(self, n: int) -> Tuple[Schedule]:
    if n not in Schedules._schedule_cache:
      Schedules._schedule_cache[n] = tuple(self._generate_b_reprs(n))
    return Schedules._schedule_cache[n]

  def _generate_b_reprs(self, n: int) -> Iterator[Schedule]:
    for m in range(n):
      for prefix, suffix in itertools.product(self._get_b_reprs(m),
                                              self._get_b_reprs(n - 1 - m)):
        yield Schedule('0' + prefix + suffix)

# The following is developped but deprecated because of its inefficiency.
class KernelIntersectionMatrix(_OrderedDict):
  """A kernel intersection matrix.

  An instance of this class is a dict mapping offsets to dicts mapping shifted
  operands to original operands.

  Attributes:
    operands: A tuple of all original operands.
  """
  def __init__(self, polynomial):
    """Create the kernel intersection matrix for a polynomial.

    Args:
      polynomial: soda.ir.BinaryOp to work on.
    """
    super().__init__()
    def _visitor(polynomial, args=None):
      if isinstance(polynomial, ir.BinaryOp):
        for operand in polynomial.operand:
          if len(soda_visitor.get_load_set(operand)) > 1:
            _logger.warn('cannot handle multi-index operands yet: %s', operand)
            return
        for op in polynomial.operator:
          if op != '+':
            _logger.warn('cannot handle %s operator yet', op)
            return
        _logger.debug(polynomial)
        indices = sorted(
            {load.idx for load in soda_visitor.get_load_set(polynomial)},
            key=base.compose(reversed, tuple))
        offsets = tuple(tuple(a - b for a, b in zip(offset, indices[0]))
                        for offset in indices)
        _logger.debug('shifting offsets: %s', offsets)
        _logger.debug(indices)
        for offset in offsets:
          shifted = mutator.shift(polynomial, offset)
          ops = _OrderedDict()
          for op, op0 in zip(shifted.operand, polynomial.operand):
            for load, load0 in zip(soda_visitor.get_load_set(op),
                                   soda_visitor.get_load_set(op0)):
              if tuple(reversed(load.idx)) >= tuple(reversed(indices[0])):
                ops[load] = load0
                operands[load0] = None
          self[offset] = ops

    operands = _OrderedDict()
    polynomial.visit(_visitor)
    self.operands = tuple(operands)

  def pretty_print(self, printer=_logger.debug, covering=None):
    """Pretty print the kernel intersection matrix.

    Pretty print the kernel intersection matrix. If covering is not None, only
    covered items are printed.

    Args:
      printer: A function printing 1 line at a time. Default to _logger.debug.
      covering: A Covering mapping operands to (offset, shifted operand) tuples.
    """
    title = 'KIM'
    strs = {}
    lens = {col: len(str(col)) for col in self.cols}
    for row in self.rows:
      lens[None] = max(lens.get(None, len(title)), len(str(row)))
      strs[row] = {}
      for col in self.cols:
        if col in self[row]:
          strs[row][col] = str(self[row][col])
          lens[col] = max(lens[col], len(strs[row][col]))
    def print_divider():
      def generate_divider():
        for col in (None,) + self.cols:
          yield '-' * lens[col]
      printer('+-%s-+', '-+-'.join(generate_divider()))
    def generate_header():
      yield ('{!s:%d}' % lens[None]).format(title)
      for col in self.cols:
        yield ('{!s:%d}' % lens[col]).format(col)
    print_divider()
    printer('| %s |', ' | '.join(generate_header()))
    print_divider()
    for row in self.rows:
      def generate_item():
        yield ('{!s:%d}' % lens[None]).format(row)
        for col in self.cols:
          max_len = lens[col]
          if col in self[row]:
            if covering is not None:
              if covering[self[row][col]][0] == row:
                yield ('{!s:%d}' % max_len).format(self[row][col])
              else:
                yield ' ' * max_len
            else:
              yield ('{!s:%d}' % max_len).format(self[row][col])
          else:
            yield ' ' * max_len
      printer('| %s |', ' | '.join(generate_item()))
    print_divider()

  @cached_property.cached_property
  def rows(self):
    """Produces the row headers of the kernel intersection matrix.

    Returns:
      A tuple of row headers.
    """
    return tuple(self)

  @cached_property.cached_property
  def cols(self):
    """Produces the column headers of the kernel intersection matrix.

    If all headers have attribute idx, they will be sorted according to idx.
    Otherwise, they will be returned in the seen order.

    Returns:
      A tuple of column headers.
    """
    operands = tuple(_OrderedDict.fromkeys(
        op for ops in self.values() for op in ops))
    try:
      return tuple(sorted(operands, key=lambda x: tuple(reversed(x.idx))))
    except AttributeError:
      return operands

  @cached_property.cached_property
  def operand_appearances(self):
    """All operand appearances.

    Returns:
      A dict mapping operands to a list of its appearances. Each appearance is
      a (row, col) tuple.
    """
    operand_appearances = _OrderedDict((op, []) for op in self.operands)
    for row in self.rows:
      for col in self.cols:
        if col in self[row]:
          operand_appearances[self[row][col]].append((row, col))
    return operand_appearances

  @property
  def coverings(self):
    """Generates valid coverings.

    Yields:
      Each valid covering.
    """
    for appearance in itertools.product(*self.operand_appearances.values()):
      yield Covering(self, zip(self.operands, appearance))

class Covering(_OrderedDict):
  """A covering of a kernel intersection matrix.

  An instance of this class is a dict mapping operands to
  (offset, shifted operand) tuples.

  Attributes:
    matrix: The corresponding KernelIntersectionMatrix.
  """
  def __init__(self, matrix: KernelIntersectionMatrix, *args, **kwargs):
    self.matrix = matrix
    super().__init__(*args, **kwargs)

  def pretty_print(self, printer=_logger.debug):
    self.matrix.pretty_print(printer=printer, covering=self)

  @cached_property.cached_property
  def cost(self) -> int:
    """Return the cost of this covering.
    """
    for row in self.matrix.rows:
      for col in self.matrix.cols:
        try:
          if self[self.matrix[row][col]][0] is row:
            _logger.debug('row: %s, col: %s', row, col)
        except KeyError:
          pass
    return 0

  @cached_property.cached_property
  def has_rectangles(self) -> bool:
    """Return whether this covering has any rectange that is at least 2x2.
    """
    row_count = _OrderedDict.fromkeys(self.matrix.rows, 0)
    col_count = _OrderedDict.fromkeys(self.matrix.cols, 0)
    for row, col in self.values():
      row_count[row] += 1
      col_count[col] += 1
    row_count = _OrderedDict(((row, count) for row, count in row_count.items()
                              if count > 1))
    col_count = _OrderedDict(((col, count) for col, count in col_count.items()
                              if count > 1))
    if len(row_count) > 1 and len(col_count) > 1:
      _logger.debug(util.lst2str(row_count.items()))
      _logger.debug(util.lst2str(col_count.items()))
      return True
    return False
