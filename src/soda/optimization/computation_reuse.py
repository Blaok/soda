import collections
import heapq
import itertools
import json
import logging
import operator
import os
import random
import shutil
import subprocess
from ctypes import util as ctypes_util
from typing import (Any, Callable, Dict, FrozenSet, Iterator, List,
                    MutableMapping, Optional, Sequence, Set, Tuple, Union,
                    overload)

import cached_property
import pulp
from haoda import ir, util
from haoda.ir import arithmetic
from haoda.ir.arithmetic import base

import soda.visitor
from soda import grammar, mutator

RelativeAttr = int
AbsoluteAttr = int
Attr = Union[RelativeAttr, Tuple[RelativeAttr, Optional[AbsoluteAttr]]]
Index = Tuple[int, ...]

OrderedDict = collections.OrderedDict


class OrderedCounter(  # type: ignore
    collections.Counter, collections.OrderedDict):
  pass


_logger = logging.getLogger().getChild(__name__)


def extract_attr(node: ir.Node) -> Tuple[Tuple[int, ...], ir.Node]:
  """Extract attributes from a node.

  Extract relative and absolute attributes from a node. The relative attribute
  would be the load index and the absolute attribute is the normalized node.

  Args:
    node: The ir.node to be extracted.

  Returns:
    Tuple of rattr and aattr.
  """
  load = soda.visitor.get_load_set(node)[0]
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
    maxs: List of integers, maximum index in each dimension.
    mins: List of integers, minimum index in each dimension.
    sizes: Tuple of integers, size of each linearized dimension.

  Properties:
    num_dim: Integer, number of dimensions.
    weights: List of integers, weight of each dimension.
    dims: Tuple of integers, all dimension indices.
  """

  def __init__(self,
               rattrs: Sequence[Sequence[int]],
               tile_size: Sequence[int] = ()):
    """Initialize the Linearizer with the given relative attribute tuples.

    Args:
      rattrs: Sequence of relative attributes. Each attribute is a sequence of
        integers.
    """
    num_dim = len(rattrs[0])
    self.maxs = [0] * num_dim
    self.mins = [0] * num_dim
    for d in self.dims:
      self.maxs[d] = max(rattr[d] for rattr in rattrs)
      self.mins[d] = min(rattr[d] for rattr in rattrs)
    if tile_size:
      self.sizes = tuple(tile_size)[:-1] + (
          (self.maxs[-1] - self.mins[-1] + 1) * 2 - 1,)
    else:
      self.sizes = tuple(
          (self.maxs[d] - self.mins[d] + 1) * 2 - 1 for d in self.dims)

  @property
  def num_dim(self) -> int:
    return len(self.maxs)

  @property
  def weights(self) -> List[int]:
    weights = [1] * self.num_dim
    for d in self.dims[1:]:
      weights[d] = weights[d - 1] * self.sizes[d - 1]
    return weights

  @property
  def dims(self) -> Tuple[int, ...]:
    return tuple(range(self.num_dim))

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


def computation_reuse(
    stencil: 'soda.core.Stencil'  # type: ignore
) -> 'soda.core.Stencil':  # type: ignore
  """Reuse computation by eliminating common subexpressions.

  The stencil object will be modified.

  Args:
    stencil: soda.core.Stencil object to work on.

  Returns:
    Modified stencil object.
  """
  method = stencil.optimizations.get('computation-reuse')
  if method is None or method == 'no':
    return stencil
  _logger.debug('invoke stencil computation reuse')

  # pylint: disable=unsubscriptable-object
  def visitor(node: ir.Node, cses: MutableMapping[ir.BinaryOp,
                                                  ir.Ref]) -> ir.Node:
    """Visitor for computation reuse.

    Args:
      csel: A dict mapping expressions to the new ir.Refs.

    Returns:
      Optimized ir.Node with computation reused.
    """
    try:
      expression = Expression(node, stencil)
      if expression.best_schedule is not None:
        _logger.debug('best schedule: (cost: %s)',
                      expression.best_schedule.cost)
        return expression.best_schedule.get_ir_node_with_cr(stencil, cses)
    except Expression.CannotHandle:
      pass
    return node

  new_local_stmts = []
  cses = OrderedDict()  # type: Dict[ir.BinaryOp, ir.Ref]
  seen = set()  # type: Set[ir.Expr]
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    stmt.propagate_type(stencil.symbol_table)
    stmt.expr = stmt.expr.visit(visitor, cses)
    stmt.let = tuple(let.visit(visitor, cses) for let in stmt.let)
    for expr, ref in cses.items():
      if expr in seen:
        continue
      seen.add(expr)
      expr = stencil.propagate_type(expr, stmt)
      new_local_stmts.append(
          grammar.LocalStmt(ref=ref,
                            haoda_type=expr.haoda_type,
                            expr=expr,
                            let=stmt.let,
                            stencil=stencil))
      _logger.debug('computation reuse stmt: %s', new_local_stmts[-1])
  stencil.local_stmts.extend(new_local_stmts)

  # invalidate cached_property
  stencil.__dict__.pop('symbol_table', None)
  stencil.__dict__.pop('local_names', None)
  stencil.__dict__.pop('local_types', None)

  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    _logger.debug('simplify:   %s', stmt)
    stmt.expr = arithmetic.simplify(stmt.expr)
    stmt.let = arithmetic.simplify(stmt.let)
    _logger.debug('simplified: %s', stmt)
  _logger.info('stencil after CR: \n  %s', str(stencil).replace('\n', '\n  '))
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

  def __getitem__(self,
                  key: int) -> Tuple[RelativeAttr, Optional[AbsoluteAttr]]:
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
    cost: Tuple of number of operations and total reuse distance required for
        this schedule.
    dependers: A dict mapping variable ids to its dependers set, which is an
        ordered dict mapping dependers to None.
    dependees: A dict mapping variable ids to its dependees dict, which maps
        dependees to a tuple of the first and last accessed offset.
  """

  def __init__(self,
               left: Union['CommSchedule', int, None],
               right: Union['CommSchedule', int, None],
               distance: RelativeAttr,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[AbsoluteAttr, ...]] = None) -> None:
    self.left, self.right, self.distance = left, right, distance
    super().__init__(rattrs, aattrs)
    self._len = 1  # number of operations
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
      del self.aattrs_as_ir_nodes  # type: ignore
      del self.linearizer  # type: ignore
      del self.aattr_table  # type: ignore
      del self.operator  # type: ignore
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
  def cost(self) -> Tuple[int, int]:
    return self.num_ops, self.total_distance

  @cached_property.cached_property
  def num_ops(self) -> int:
    return len(set(self.children))

  def to_json(self):
    j = {'distance': self.distance}
    for child in 'left', 'right':
      j[child] = getattr(self, child)
      if isinstance(j[child], CommSchedule):
        j[child] = j[child].to_json()
    return j

  def _calc_dependency(self) -> None:
    """Calculate the dependency between reused variables.

    This function creates two dependency tables, i.e. dependers and dependees.
    """

    def get_attrs(schedule: CommSchedule,
                  reuses: MutableMapping[Schedule, int],
                  offset: Optional[int] = None) -> Iterator[Tuple[int, int]]:
      """Get all accesses of variables in a schedule.

      Args:
        schedule: The CommSchedule to work on.
        reuses: Dict mapping a reused CommSchedule to the variable id.
        offset: The global offset of the input schedule.

      Yields:
        Tuple of the offset and the variable id.
      """
      reused_vid = reuses.get(schedule)
      if reused_vid is not None and offset is not None:
        yield offset, reused_vid
      else:
        if offset is None:
          offset = 0
        if isinstance(schedule.left, CommSchedule):
          yield from get_attrs(schedule.left, reuses, offset)
        else:
          yield offset, 0
        offset += schedule.distance
        if isinstance(schedule.right, CommSchedule):
          yield from get_attrs(schedule.right, reuses, offset)
        else:
          yield offset, 0

    cr_vars = OrderedDict([(self, 1)])
    cr_vars_table = {1: self}
    # var_0 is input, var_1 is output
    for child, count in OrderedCounter(self.children).items():
      if count > 1:
        cr_vars[child] = len(cr_vars) + 1
        cr_vars_table[len(cr_vars)] = child
    for schedule, vid in cr_vars.items():
      _logger.debug('var_%d: %s', vid, schedule)

    vars_to_process = collections.deque([self])
    vars_processed = {0}
    dependers = OrderedDict()  # type: Dict[int, Dict[int, None]]
    dependees = OrderedDict()  # type: Dict[int, Dict[int, Tuple[int, int]]]
    while vars_to_process:
      schedule = vars_to_process.popleft()
      dst_vid = cr_vars[schedule]
      vars_processed.add(dst_vid)
      for offset, src_vid in get_attrs(schedule, cr_vars):
        _logger.debug('var_%d accesses var_%d @ %d', dst_vid, src_vid, offset)
        dependers.setdefault(src_vid, OrderedDict()).setdefault(dst_vid, None)
        dependees.setdefault(dst_vid,
                             OrderedDict()).setdefault(src_vid,
                                                       (offset, offset))
        min_offset, max_offset = dependees[dst_vid][src_vid]
        dependees[dst_vid][src_vid] = (min(offset,
                                           min_offset), max(offset, max_offset))
        if (src_vid not in vars_processed and
            cr_vars_table[src_vid] not in vars_to_process):
          vars_to_process.append(cr_vars_table[src_vid])

    def inline() -> Iterator[Tuple[int, int]]:
      """Inline a variable if it is only accessed once by another variable.

      Yields:
        Tuple of variable ids of inlined source and destination variables.
      """
      if _logger.isEnabledFor(logging.DEBUG):
        _logger.debug('looking for inline')
        _logger.debug('  dependers:')
        for src_vid, dst_vids in dependers.items():
          _logger.debug('    var_%d is accessed by %s', src_vid,
                        ', '.join(map('var_{}'.format, dst_vids)))
        _logger.debug('  dependees:')
        for dst_vid in dependees:
          for src_vid, (min_offset, max_offset) in dependees[dst_vid].items():
            _logger.debug('    var_%d accesses var_%d @ [%d, %d]', dst_vid,
                          src_vid, min_offset, max_offset)

      for src_vid, dst_vids in dependers.items():
        if len(dst_vids) == 1:
          dst_vid = tuple(dst_vids)[0]
          min_offset, max_offset = dependees[dst_vid][src_vid]
          if min_offset == max_offset:
            _logger.debug(
                'var_%d is only accessed by var_%d @ %d, '
                'it should be inlined', src_vid, dst_vid, min_offset)
            yield src_vid, dst_vid
            yield from inline()
            break
      else:
        _logger.debug('  cannot find inline oppotunities')

    for src_vid, dst_vid in inline():
      _logger.debug('inlining var_%d', src_vid)
      offset = dependees[dst_vid][src_vid][0]
      for src_src_vid, (min_offset, max_offset) in dependees[src_vid].items():
        new_min_offset = min_offset + offset
        new_max_offset = max_offset + offset
        _logger.debug('var_%d accesses var_%d @ %d', dst_vid, src_vid, offset)
        _logger.debug('var_%d accesses var_%d @ [%d, %d]', src_vid, src_src_vid,
                      min_offset, max_offset)
        _logger.debug('therefore var_%d accesses var_%d @ [%d, %d] via var_%d',
                      dst_vid, src_src_vid, new_min_offset, new_max_offset,
                      src_vid)
        if src_src_vid in dependees[dst_vid]:
          _logger.debug('var_%d used to access var_%d @ [%d, %d]', dst_vid,
                        src_src_vid, *dependees[dst_vid][src_src_vid])
        old_min_offset, old_max_offset = dependees[dst_vid].get(
            src_src_vid, (new_min_offset, new_max_offset))
        dependees[dst_vid][src_src_vid] = (min(old_min_offset, new_min_offset),
                                           max(old_max_offset, new_max_offset))
        _logger.debug('after inlining, var_%d accesses var_%d @ [%d, %d]',
                      dst_vid, src_src_vid, *dependees[dst_vid][src_src_vid])
      src_src_vids = list(dependees[src_vid])
      for src_src_vid in src_src_vids:
        dependers[src_src_vid][dst_vid] = None
        del dependers[src_src_vid][src_vid]
      del dependers[src_vid]
      del dependees[dst_vid][src_vid]
      del dependees[src_vid]
      del cr_vars_table[src_vid]
    self._dependers, self._dependees = dependers, dependees
    self._cr_vars_table = cr_vars_table

  @property
  def dependers(self) -> Dict[int, Dict[int, None]]:
    """The dependers table.

    Returns:
      A dict mapping variable ids to its dependers set, which is an ordered dict
      mapping dependers to None.
    """
    if not hasattr(self, '_dependers'):
      self._calc_dependency()
    return self._dependers

  @property
  def dependees(self) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """The dependees table.

    Returns:
      A dict mapping variable ids to its dependees dict, which maps dependees to
      a tuple of the first and last accessed offset.
    """
    if not hasattr(self, '_dependees'):
      self._calc_dependency()
    return self._dependees

  @property
  def cr_vars_table(self) -> Dict[int, 'CommSchedule']:
    if not hasattr(self, '_cr_vars_table'):
      self._calc_dependency()
    return self._cr_vars_table

  @cached_property.cached_property
  def total_distance(self) -> int:
    """Calculate the total reuse distance.

    The total reuse distance is a measure of hardware resource cost in terms of
    required reuse buffer size. It is calculated by summing up the reuse
    distance of all reused variables. The reused distance is determined by the
    offset when a variable is first produced and last consumed. The latter is
    static and the former is calculated via solving an ILP.

    Returns:
      The total reuse distance.
    """

    # solve ILP for optimal offsets
    lp_problem = pulp.LpProblem("optimal_offsets", pulp.LpMinimize)
    lp_vars = {0: 0, 1: 0}
    lp_helper_vars = {}
    objectives = []
    for src_vid in self.dependers:
      lp_var = pulp.LpVariable('produced_offset_%d' % src_vid, cat='Integer')
      lp_helper_var = pulp.LpVariable('consumed_offset_%d' % src_vid,
                                      cat='Integer')
      lp_vars.setdefault(src_vid, lp_var)
      lp_helper_vars[src_vid] = lp_helper_var
      objectives.append(lp_helper_var - lp_vars[src_vid])
    lp_problem += sum(objectives)
    for src_vid, dst_vids in self.dependers.items():
      for dst_vid in dst_vids:
        min_offset, max_offset = self.dependees[dst_vid][src_vid]
        lp_problem += lp_vars[src_vid] <= min_offset + lp_vars[dst_vid]
        lp_problem += lp_helper_vars[src_vid] >= max_offset + lp_vars[dst_vid]
    lp_status = pulp.LpStatus[lp_problem.solve()]
    _logger.debug('ILP status: %s', lp_status)
    _logger.debug('var_0 should be produced @ 0 and kept until %d',
                  int(pulp.value(lp_helper_vars[0])))
    self._produce_offsets = {}  # type: Dict[int, int]
    self._consume_offsets = {}  # type: Dict[int, int]
    for vid, lp_var in lp_vars.items():
      if vid == 1:  # output
        continue
      produce_offset = 0 if vid == 0 else int(pulp.value(lp_var))
      consume_offset = int(pulp.value(lp_helper_vars[vid]))
      _logger.debug('var_%d should be produced @ %d and kept until %d', vid,
                    produce_offset, consume_offset)
      self._produce_offsets[vid] = produce_offset
      self._consume_offsets[vid] = consume_offset
    total_distance = int(pulp.value(lp_problem.objective))
    max_rattr = max(
        attr if isinstance(attr, int) else attr[0] for attr in self.norm_attrs)
    assert max_rattr <= total_distance, '%s < %s' % (total_distance, max_rattr)
    return total_distance

  @property
  def produce_offsets(self) -> Dict[int, int]:
    """Maps variable ids to offsets at which they are first produced"""
    if not hasattr(self, '_produce_offsets'):
      _ = self.total_distance
    return self._produce_offsets

  @property
  def consume_offsets(self) -> Dict[int, int]:
    """Maps variable ids to offsets at which they are last consumed"""
    if not hasattr(self, '_consume_offsets'):
      _ = self.total_distance
    return self._consume_offsets

  def get_attrs_with_offset(self, offset: int = 0) -> Iterator[Attr]:
    """Generate all attributes with the given offset.

    Args:
      offset: The offset of the smallest relative attribute.

    Yields:
      Attributes in this schedule, NOT necessarily sorted by their relative
      attributes.
    """
    if isinstance(self.left, CommSchedule):  # Left child is not a leaf.
      yield from self.left.get_attrs_with_offset(offset)
    else:  # Left child is a leaf.
      if self.aattrs is None:  # Null absolute attributes.
        yield offset
      else:  # Non-null absolute attributes.
        yield offset, self.left

    offset += self.distance

    if isinstance(self.right, CommSchedule):  # Right child is not a leaf.
      yield from self.right.get_attrs_with_offset(offset)
    else:  # Right child is a leaf.
      if self.aattrs is None:  # Null absolute attributes.
        yield offset
      else:  # Non-null absolute attributes.
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
    exprs = OrderedDict()  # type: Dict[FrozenSet[Attr], List[CommSchedule]]
    exprs[self.norm_attr_set] = [self]
    for child in self.left, self.right:
      if isinstance(child, CommSchedule):
        for attrs, schedules in child.uniq_expr_dict.items():
          exprs.setdefault(attrs, []).extend(schedules)
    return exprs

  def get_ir_node_with_offset(self, offset: int = 0) -> ir.Node:
    """Get the IR node with the given offset.

    Args:
      offset: The offset of the smallest relative attribute.

    Returns:
      An ir.BinaryOp as the root of the IR.
    """
    if isinstance(self.left, CommSchedule):  # Left child is not a leaf.
      left_child = self.left.get_ir_node_with_offset(offset)  # type: ir.Node
    else:  # Left child is a leaf.
      left_child = assemble_attr(self.linearizer(offset),
                                 self.aattr_table[self.left])

    offset += self.distance

    if isinstance(self.right, CommSchedule):  # Right child is not a leaf.
      right_child = self.right.get_ir_node_with_offset(offset)  # type: ir.Node
    else:  # Right child is a leaf.
      right_child = assemble_attr(self.linearizer(offset),
                                  self.aattr_table[self.right])

    return ir.from_reduction(self.operator, (left_child, right_child))

  @cached_property.cached_property
  def ir_node(self) -> ir.Node:
    return self.get_ir_node_with_offset(self.rattrs[0])

  @cached_property.cached_property
  def _rcr_write_idx_table(self) -> Dict[ir.BinaryOp, Tuple[int, ...]]:
    """Returns the CSE write index table.

    Returns:
      A dict mapping normalized common subexpressions to the index at which the
      new variable should write to.
    """
    table = {}
    for vid in self.dependers:
      if vid == 0:  # 0 is the input
        continue
      expr = mutator.normalize(self.cr_vars_table[vid].ir_node)
      # okay not to add references because expr does not contain cr vars
      # TODO: support multi-stage input
      table[mutator.normalize(expr)] = util.add_inv(
          soda.visitor.get_normalize_index(expr))
    return table

  def get_ir_node_with_rcr(self,
                           stencil,
                           rcrs: MutableMapping[ir.BinaryOp, ir.Ref],
                           write_idx_table=None) -> ir.Node:
    """Returns an ir.Node with relative computation reuse.

    Args:
      stencil: soda.core.Stencil object, whose symbol table will be updated.
      rcrs: Dict mapping common subexpressions to the ir.Ref representing the
        write access to the new variable. This dict is used as both an input and
        an output. If the keys contain other common subexpressions, they should
        be replaced with the new variables.
      cse_write_idx_table: The CSE write index table, used to detect common
        subexpressions.
    """
    if write_idx_table is None:
      write_idx_table = self._rcr_write_idx_table
    operands = []  # type: List[ir.Node]
    _logger.debug('get ir node with cse for %s', self)
    if _logger.isEnabledFor(logging.DEBUG):
      for rattr, aattr in self:
        _logger.debug('  rattr: %s aattr: %s', rattr, aattr)
    for rattr, aattr in (  # type: ignore
        (self.rattrs[0], self.left), (self.rattrs[0] + self.distance,
                                      self.right)):
      # replace the aattr with proper normalized node
      if isinstance(aattr, CommSchedule):
        # node_without_cse is used for key-ing write_idx_table
        # okay not to add references because expr does not contain cr vars
        # TODO: support multi-stage input
        node_without_cse = mutator.shift(
            aattr.ir_node, soda.visitor.get_normalize_index(aattr.ir_node))
        node_with_cse = aattr.get_ir_node_with_rcr(stencil, rcrs,
                                                   write_idx_table)
        # node_with_cse_norm is used for key-ing rcrs
        node_with_cse_norm = mutator.normalize(
            node_with_cse, {ref.name: ref.idx for ref in rcrs.values()})
        idx = write_idx_table.get(node_without_cse)
        if idx is not None:
          # this aattr is a common subexpression, need to be replaced
          # cses is key-ed by expressions with recursive CSE applied
          if node_with_cse_norm not in rcrs:
            # this is the first time we see this aattr, allocate a new name
            node = ir.Ref(name=stencil.new_cr_var(), idx=idx, lat=None)
            _logger.debug('  replace %s with %s', node_without_cse, node)
            stencil.symbol_table[node.name] = node_without_cse.haoda_type
            rcrs[node_with_cse_norm] = node
          else:
            # we have seen this aattr, replace with the known
            node = rcrs[node_with_cse_norm]
        else:
          # this aattr is not a common subexpression
          node = mutator.shift(node_with_cse, self.linearizer(rattr))
      else:
        node = self.aattr_table[aattr]
      operands.append(assemble_attr(self.linearizer(rattr), node))

    return arithmetic.simplify(ir.from_reduction(self.operator,
                                                 tuple(operands)))

  def get_ir_node_with_cr(self, stencil, crs: Dict[ir.BinaryOp,
                                                   ir.Ref]) -> ir.Node:
    rcrs = crs.copy()
    ir_node_with_rcr = self.get_ir_node_with_rcr(stencil, rcrs)
    norm_refs = {getattr(ref, 'name'): ref.idx for ref in rcrs.values()}

    # try to apply absolute CSE
    binary_aattrs = collections.defaultdict(
        list)  # type: Dict[ir.BinaryOp, List[Tuple[int, ...]]]

    def add_to_count(node: ir.Node, norm_idx: Index = ()) -> None:
      reduction = ir.to_reduction(node)
      if reduction is not None:
        for op in reduction[1]:
          if isinstance(op, ir.BinaryOp):
            idx = soda.visitor.get_normalize_index(op, references=norm_refs)
            if norm_idx:
              idx = tuple(x - y for x, y in zip(idx, norm_idx))
            binary_aattrs[mutator.normalize(op,
                                            references=norm_refs)].append(idx)

    _logger.debug('counting complex aattrs')
    norm_idx = soda.visitor.get_normalize_index(self.ir_node,
                                                references=norm_refs)
    add_to_count(ir_node_with_rcr, norm_idx)  # not normalized; need norm_idx
    for tcs in rcrs:
      add_to_count(tcs)  # all normalized
    _logger.debug('complex aattrs count')
    acrs = {}
    for op, indices in binary_aattrs.items():
      count = len(indices)
      _logger.debug(' %s x%d', op, count)
      if count > 1:
        new_name = stencil.new_cr_var()
        # select the least index to minimize buffer size required
        min_idx = min(indices, key=lambda x: tuple(reversed(x)))
        acrs[op] = ir.Ref(name=new_name, idx=util.add_inv(min_idx), lat=None)
        stencil.symbol_table[new_name] = getattr(op, 'haoda_type')

    do_acr = lambda op: mutator.replace_expressions(
        op, acrs, references=norm_refs)
    rcrs = OrderedDict((do_acr(k), v) for k, v in rcrs.items())
    crs.update(rcrs)
    crs.update(acrs)
    if _logger.isEnabledFor(logging.INFO):
      for tcs, ref in rcrs.items():
        _logger.info('rcr: %s => %s', ir.unparenthesize(tcs), ref)
      for tcs, ref in acrs.items():
        _logger.info('acr: %s => %s', ir.unparenthesize(tcs), ref)

    reduction = ir.to_reduction(ir_node_with_rcr)
    assert reduction is not None
    return arithmetic.simplify(
        ir.from_reduction(reduction[0], tuple(map(do_acr, reduction[1]))))


def make_schedule_from_json(j: Dict[str, Any], offset: int,
                            null_aattr: bool) -> CommSchedule:
  left, right, distance = j['left'], j['right'], j['distance']
  attrs = []  # type: List[Tuple[RelativeAttr, AbsoluteAttr]]
  if isinstance(left, dict):
    left = make_schedule_from_json(left, offset, null_aattr)
    attrs.extend(left)
  elif isinstance(left, int):
    if null_aattr:
      left = None
    attrs.append((offset, left))

  offset += distance

  if isinstance(right, dict):
    right = make_schedule_from_json(right, offset, null_aattr)
    attrs.extend(right)
  elif isinstance(right, int):
    if null_aattr:
      right = None
    attrs.append((offset, right))

  attrs.sort(key=lambda attr: attr[0])
  rattrs, aattrs = zip(*attrs)
  return CommSchedule(left, right, distance, rattrs,
                      None if null_aattr else aattrs)


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

  def __init__(
      self,
      rattrs: Tuple[RelativeAttr, ...],
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

  def __iter__(self) -> Iterator[CommSchedule]:  # type: ignore
    if hasattr(self, 'schedules'):
      return iter(getattr(self, 'schedules'))
    return self.generator

  def key(self, operands) -> Tuple[int, ...]:
    offset = self.rattrs[next(
        idx for idx, bit in enumerate(operands) if bit == '1')]
    key = [
        self.rattrs[idx] - offset
        for idx, bit in enumerate(operands)
        if bit == '1'
    ]
    if self.aattrs is not None:
      key.extend(
          self.aattrs[idx] for idx, bit in enumerate(operands) if bit == '1')
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
        left_operands = ''.join(
            '1' if i in left_indices else '0' for i in range(num_operands))
        right_operands = ''.join(
            '1' if i in right_indices else '0' for i in range(num_operands))
        for left in self.get_schedules(left_operands):
          self.stat[3] += 1
          left_cost = 1
          if isinstance(left, CommSchedule):
            left_cost += left.num_ops
          if self.skip and left_cost > self.max_cost:
            skipped = True
            continue
          for right in self.get_schedules(right_operands):
            self.stat[4] += 1
            right_cost = 1
            if isinstance(right, CommSchedule):
              right_cost += right.num_ops
            if self.skip and right_cost > self.max_cost:
              skipped = True
              continue
            distance = self.rattrs[right_indices[0]]
            distance -= self.rattrs[left_indices[0]]
            rattrs = tuple(self.rattrs[i]
                           for i, op in enumerate(self.operands)
                           if op != '0')
            aattrs = None
            if self.aattrs is not None:
              aattrs = tuple(self.aattrs[i]
                             for i, op in enumerate(self.operands)
                             if op != '0')
            schedule = CommSchedule(  # type: ignore
                left, right, distance, rattrs, aattrs)
            num_ops = schedule.num_ops  # type: ignore
            if num_ops < self.max_cost:
              self.max_cost = num_ops
            schedules.append(schedule)
            yield schedule  # type: ignore
    self.schedules = schedules

  schedule_cache = {}  # type: Dict[CommSchedule, CommSchedule]

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
            _logger.info('cost: %s', best.cost)
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
    return CommSchedules(self.rattrs,
                         self.aattrs,
                         operands=operands,
                         cache=self.cache,
                         stat=self.stat,
                         max_cost=min(
                             self.max_cost,
                             collections.Counter(operands)['1'])).generator

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    logger('loops: | L1: %d | L2: %d | L3: %d |', *self.stat[2:])
    logger('cache: | hit: %d | miss: %d | hit rate: %2.3f %% |', self.cache_hit,
           self.cache_miss, self.cache_hit_rate * 100)


class GreedySchedules(ScheduleBase):
  """Schedules of an Expression, found greedily.
  """
  timeout = 1
  num_pruned = 5

  def __init__(self,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[Union[AbsoluteAttr, CommSchedule, None],
                                      ...]] = None,
               linearizer: Optional[Linearizer] = None,
               cache: Optional[Dict] = None) -> None:
    self.linearizer = linearizer
    super().__init__(rattrs, aattrs)  # type: ignore

  def __lt__(self, other: 'GreedySchedules') -> bool:
    return self.comparison_key.cost < other.comparison_key.cost

  @cached_property.cached_property
  def comparison_key(self) -> CommSchedule:
    return linear_schedule(tuple(self))

  @property
  def generator(self) -> Iterator[CommSchedule]:
    attr_map = {attr: idx for idx, attr in enumerate(self)}
    reuses = OrderedDict()  # type: Dict[CommSchedule, List[Tuple[int, int]]]
    has_conflict = collections.defaultdict(
        lambda: False)  # type: Dict[CommSchedule, bool]
    for left, right in itertools.combinations(self, 2):
      left_rattr, left_aattr = left
      right_rattr, right_aattr = right
      distance = right_rattr - left_rattr
      new_aattr = left_aattr, right_aattr
      operation = CommSchedule(left_aattr, right_aattr, distance,
                               (left_rattr, right_rattr),
                               new_aattr)  # type: ignore
      if operation in reuses:
        continue
      #_logger.debug('look for operation: %s', operation)
      # look for reuse of this operation over all operands
      reuses[operation] = []
      # conflict groups
      # group_lists maps group id to a list of (idx_l, idx_r) tuple, in order
      # group_table maps attr idx to group id
      group_lists = []  # type: List[List[Tuple[int, int]]]
      group_table = {}  # type: Dict[int, int]
      for idx_l, (rattr_l, aattr_l) in enumerate(self):
        if aattr_l != left_aattr:
          continue
        rattr_r, aattr_r = rattr_l + right_rattr - left_rattr, right_aattr
        idx_r = attr_map.get((rattr_r, aattr_r))
        if idx_r is None:
          continue
        # (idx_l, idx_r) is compatible with `operation`
        group_id = group_table.get(idx_l)
        if group_id is None:
          group_id = group_table.get(idx_r)
        if group_id is None:
          # new conflict group
          group_id = len(group_lists)
          group_lists.append([])
        group_lists[group_id].append((idx_l, idx_r))
        group_table[idx_l] = group_id
        group_table[idx_r] = group_id

      for group_list in group_lists:
        if len(group_list) > 1:
          _logger.debug('conflict group of %s: %s', operation, group_list)
          has_conflict[operation] = True

      for group_list in group_lists:
        if len(group_list) % 2 != 0:
          reuses[operation].extend(group_list[::2])

      min_idx_l = min((x[0] for x in reuses[operation]), default=0)
      max_idx_l = max((x[0] for x in reuses[operation]), default=-1)
      _logger.debug('min_idx_l: %s | max_idx_l: %s', min_idx_l, max_idx_l)

      for group_list in group_lists:
        if len(group_list) % 2 == 0:
          span_0 = (self.rattrs[max(group_list[-2][0], max_idx_l)] -
                    self.rattrs[min(group_list[0][0], min_idx_l)])
          span_1 = (self.rattrs[max(group_list[-1][0], max_idx_l)] -
                    self.rattrs[min(group_list[1][0], min_idx_l)])
          _logger.debug('span 0: %d, span 1: %d', span_0, span_1)
          reuses[operation].extend(group_list[1 if span_1 < span_0 else 0::2])
      if len(reuses[operation]) > 1:
        _logger.debug('reuses[%s]: %s', operation, reuses[operation])
      reuses[operation].sort()  # ?

    # filter out operations that cannot be reused
    # they may not all be useful because they overlap
    reuses = {k: v for k, v in reuses.items() if len(v) > 1}
    if not reuses:
      yield linear_schedule(tuple(self))
      return

    def aligns(dis: int, dim: int) -> bool:
      """Returns whether dis aligns with dim.

      A distance aligns with a dimension if the indices of two points with that
      distance differ and only differ in that dimension.
      """
      assert self.linearizer is not None
      zipped = zip(self.linearizer(dis), self.linearizer.mins,
                   self.linearizer.dims)
      return all(idx != min_idx if d == dim else idx == min_idx
                 for idx, min_idx, d in zipped)

    if self.linearizer is not None and len(reuses) > len(self):
      _logger.debug('linearizer: mins: %s maxs: %s', self.linearizer.mins,
                    self.linearizer.maxs)
      for dim in reversed(self.linearizer.dims):
        if any(aligns(op.distance, dim) for op in reuses):
          # only consider reuse with a single dimension, if possible
          reuses = {
              k: [(idx_l, idx_r)
                  for idx_l, idx_r in v
                  if aligns(self.rattrs[idx_r] - self.rattrs[idx_l], dim)
                 ] for k, v in reuses.items() if aligns(k.distance, dim)
          }
          break

    candidates = []
    for op in reuses:
      # find all compatible reuses that include op
      _logger.debug('find all compatible reuses that include %s', op)
      new_attrs = OrderedDict(enumerate(self))
      used = set()  # type: Set[int]

      def do_reuse_for(schedule: CommSchedule) -> None:
        reused_indices = [(idx_l, idx_r)
                          for idx_l, idx_r in reuses[schedule]
                          if idx_l not in used and idx_r not in used]
        if len(reused_indices) > 1:
          for idx_l, idx_r in reused_indices:
            _logger.debug('reusing %s for %s + %s', schedule,
                          '%s:%s' % self[idx_l], '%s:%s' % self[idx_r])
            # pylint: disable=cell-var-from-loop
            new_attrs[idx_l] = new_attrs[idx_l][0], schedule  # type: ignore
            del new_attrs[idx_r]
            used.update({idx_l, idx_r})

      do_reuse_for(op)
      for operation in sorted(reuses,
                              key=lambda s: (-len(reuses[s]), s.distance)):
        do_reuse_for(operation)

      new_rattrs, new_aattrs = zip(*new_attrs.values())
      candidates.append((has_conflict[op],
                         GreedySchedules(new_rattrs, new_aattrs,
                                         self.linearizer)))

    nsmallest = heapq.nsmallest(GreedySchedules.num_pruned, candidates)

    if _logger.isEnabledFor(logging.DEBUG):
      for conflict, schedule in nsmallest:
        _logger.debug('candidate: %s cost: %s', schedule.comparison_key,
                      (conflict, *schedule.comparison_key.cost))

    for _, schedule in nsmallest:
      yield from schedule.generator

  @cached_property.cached_property
  def best(self) -> CommSchedule:
    generator = self.generator
    best = next(generator)
    try:
      with util.timeout(GreedySchedules.timeout):
        for schedule in generator:
          _logger.debug('schedule: %s', schedule)
          if schedule.cost < best.cost:
            best = schedule
            _logger.info('schedule: %s num ops: %d total distance: %d', best,
                         *best.cost)
    except TimeoutError:
      _logger.warning('timeout after %s sec', self.timeout)
    self.print_stats()
    _logger.info('schedule: %s num ops: %d total distance: %d', best,
                 *best.cost)
    return best

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    return


class BeamSchedules(ScheduleBase):
  """Schedules of an Expression, found using beam search.
  """
  beam_width = 5

  def __init__(self,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[Union[AbsoluteAttr, CommSchedule, None],
                                      ...]] = None,
               linearizer: Optional[Linearizer] = None,
               cache: Optional[Dict] = None) -> None:
    self.linearizer = linearizer
    super().__init__(rattrs, aattrs)  # type: ignore

  def __lt__(self, other: 'BeamSchedules') -> bool:
    return self.linear_schedule.cost < other.linear_schedule.cost

  @cached_property.cached_property
  def linear_schedule(self) -> CommSchedule:
    return linear_schedule(tuple(self))

  @property
  def candidates(self) -> List[Tuple[bool, 'BeamSchedules']]:
    """Return new candidates with computation reuse.

    Returns:
        List[Tuple[bool, BeamSchedules]]: A list of (has_conflict, candidates).
    """

    attr_map = {attr: idx for idx, attr in enumerate(self)}
    reuses = OrderedDict()  # type: Dict[CommSchedule, List[Tuple[int, int]]]
    has_conflict = collections.defaultdict(
        lambda: False)  # type: Dict[CommSchedule, bool]
    for left, right in itertools.combinations(self, 2):
      left_rattr, left_aattr = left
      right_rattr, right_aattr = right
      distance = right_rattr - left_rattr
      new_aattr = left_aattr, right_aattr
      operation = CommSchedule(left_aattr, right_aattr, distance,
                               (left_rattr, right_rattr),
                               new_aattr)  # type: ignore
      if operation in reuses:
        continue
      #_logger.debug('look for operation: %s', operation)
      # look for reuse of this operation over all operands
      reuses[operation] = []
      # conflict groups
      # group_lists maps group id to a list of (idx_l, idx_r) tuple, in order
      # group_table maps attr idx to group id
      group_lists = []  # type: List[List[Tuple[int, int]]]
      group_table = {}  # type: Dict[int, int]
      for idx_l, (rattr_l, aattr_l) in enumerate(self):
        if aattr_l != left_aattr:
          continue
        rattr_r, aattr_r = rattr_l + right_rattr - left_rattr, right_aattr
        idx_r = attr_map.get((rattr_r, aattr_r))
        if idx_r is None:
          continue
        # (idx_l, idx_r) is compatible with `operation`
        group_id = group_table.get(idx_l)
        if group_id is None:
          group_id = group_table.get(idx_r)
        if group_id is None:
          # new conflict group
          group_id = len(group_lists)
          group_lists.append([])
        group_lists[group_id].append((idx_l, idx_r))
        group_table[idx_l] = group_id
        group_table[idx_r] = group_id

      for group_list in group_lists:
        if len(group_list) > 1:
          _logger.debug('conflict group of %s: %s', operation, group_list)
          has_conflict[operation] = True

      for group_list in group_lists:
        if len(group_list) % 2 != 0:
          reuses[operation].extend(group_list[::2])

      min_idx_l = min((x[0] for x in reuses[operation]), default=0)
      max_idx_l = max((x[0] for x in reuses[operation]), default=-1)
      _logger.debug('min_idx_l: %s | max_idx_l: %s', min_idx_l, max_idx_l)

      for group_list in group_lists:
        if len(group_list) % 2 == 0:
          span_0 = (self.rattrs[max(group_list[-2][0], max_idx_l)] -
                    self.rattrs[min(group_list[0][0], min_idx_l)])
          span_1 = (self.rattrs[max(group_list[-1][0], max_idx_l)] -
                    self.rattrs[min(group_list[1][0], min_idx_l)])
          _logger.debug('span 0: %d, span 1: %d', span_0, span_1)
          reuses[operation].extend(group_list[1 if span_1 < span_0 else 0::2])
      if len(reuses[operation]) > 1:
        _logger.debug('reuses[%s]: %s', operation, reuses[operation])
      reuses[operation].sort()  # ?

    # filter out operations that cannot be reused
    # they may not all be useful because they overlap
    reuses = {k: v for k, v in reuses.items() if len(v) > 1}
    if not reuses:
      return []

    def aligns(dis: int, dim: int) -> bool:
      """Returns whether dis aligns with dim.

      A distance aligns with a dimension if the indices of two points with that
      distance differ and only differ in that dimension.
      """
      assert self.linearizer is not None
      zipped = zip(self.linearizer(dis), self.linearizer.mins,
                   self.linearizer.dims)
      return all(idx != min_idx if d == dim else idx == min_idx
                 for idx, min_idx, d in zipped)

    if self.linearizer is not None and len(reuses) > len(self):
      _logger.debug('linearizer: mins: %s maxs: %s', self.linearizer.mins,
                    self.linearizer.maxs)
      for dim in reversed(self.linearizer.dims):
        if any(aligns(op.distance, dim) for op in reuses):
          # only consider reuse with a single dimension, if possible
          reuses = {
              k: [(idx_l, idx_r)
                  for idx_l, idx_r in v
                  if aligns(self.rattrs[idx_r] - self.rattrs[idx_l], dim)
                 ] for k, v in reuses.items() if aligns(k.distance, dim)
          }
          break

    candidates = []
    for op in reuses:
      # find all compatible reuses that include op
      _logger.debug('find all compatible reuses that include %s', op)
      new_attrs = OrderedDict(enumerate(self))
      used = set()  # type: Set[int]

      def do_reuse_for(schedule: CommSchedule) -> None:
        reused_indices = [(idx_l, idx_r)
                          for idx_l, idx_r in reuses[schedule]
                          if idx_l not in used and idx_r not in used]
        if len(reused_indices) > 1:
          for idx_l, idx_r in reused_indices:
            _logger.debug('reusing %s for %s + %s', schedule,
                          '%s:%s' % self[idx_l], '%s:%s' % self[idx_r])
            # pylint: disable=cell-var-from-loop
            new_attrs[idx_l] = new_attrs[idx_l][0], schedule  # type: ignore
            del new_attrs[idx_r]
            used.update({idx_l, idx_r})

      do_reuse_for(op)
      for operation in sorted(reuses,
                              key=lambda s: (-len(reuses[s]), s.distance)):
        do_reuse_for(operation)

      new_rattrs, new_aattrs = zip(*new_attrs.values())
      candidates.append((has_conflict[op],
                         BeamSchedules(new_rattrs, new_aattrs,
                                       self.linearizer)))

    return candidates

  @cached_property.cached_property
  def best(self) -> CommSchedule:
    candidates = [(False, self)]
    best = self.linear_schedule
    while candidates:
      candidates = heapq.nsmallest(
          BeamSchedules.beam_width,
          sum((x.candidates for _, x in candidates), []))
      for _, candidate in candidates:
        schedule = candidate.linear_schedule
        _logger.debug('schedule: %s', schedule)
        if schedule.cost < best.cost:
          best = schedule
          _logger.info('schedule: %s num ops: %d total distance: %d', best,
                       *best.cost)
    _logger.info('schedule: %s num ops: %d total distance: %d', best,
                 *best.cost)
    return best

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    return


def linear_schedule(attrs) -> CommSchedule:
  """Schedule the attributes linearily.

  Args:
    indices: Iterable of ints, indicating what attributes should be used.

  Returns:
    CommSchedule of the attributes, scheduled as a linear tree.
  """
  rattrs, aattrs = zip(*attrs)
  # rattrs doesn't have to start with 0 but has to be sorted
  if list(rattrs) != sorted(rattrs):
    raise util.InputError('rattrs not sorted: %s' % str(rattrs))
  distance = rattrs[1] - rattrs[0]
  other_args = distance, rattrs, aattrs
  if len(attrs) == 2:
    return CommSchedule(aattrs[0], aattrs[1], *other_args)
  return CommSchedule(aattrs[0], linear_schedule(attrs[1:]), *other_args)


class GloreSchedules(ScheduleBase):
  """Schedules of an Expression, found using GLORE paper's heuristics.
  """

  def __init__(self,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[Union[AbsoluteAttr, CommSchedule, None],
                                      ...]] = None,
               linearizer: Optional[Linearizer] = None,
               cache: Optional[Dict] = None) -> None:
    if linearizer is None:
      raise util.InputError('linearizer must not be None for GloreSchedules')
    self.linearizer = linearizer
    super().__init__(rattrs, aattrs)  # type: ignore

  @property
  def generator(self) -> Iterator[CommSchedule]:
    num_dim = self.linearizer.num_dim
    for direction in ((1,) + (0,) * (num_dim - 1), (1,) * num_dim):
      _logger.debug('explore direction %s', direction)

      # maps line_id to (attrs, reuse distance list)
      # line_id is the unique identifier of a line along the specific direction
      reuse_groups = collections.defaultdict(lambda: ([], [
      ]))  # type: Dict[Index, Tuple[List[Tuple[Index, Any]], List[int]]]

      def log_reuse_group(logger=_logger.debug,
                          reuse_groups=reuse_groups,
                          direction=direction) -> None:
        for line_id, (group, reuse_distance_list) in reuse_groups.items():
          logger('logging reuse groups')
          logger('  line #%s along direction <%s>:', line_id, direction)
          rattr_indices, aattrs = zip(*group)
          logger('    rattrs: %s', util.lst2str(rattr_indices))
          logger('    aattrs: %s', util.lst2str(aattrs))
          logger('    reuse distances: %s', reuse_distance_list)
        logger('')

      # step 1: grouping
      _logger.debug('attrs: %s', util.lst2str(self))
      for rattr, aattr in self:
        rattr_idx = self.linearizer(rattr)
        if collections.Counter(direction)[1] > 1:  # diagonal
          line_id = tuple(x - rattr_idx[0] for x in rattr_idx[1:])
        else:
          line_id = rattr_idx[1:]
        reuse_groups[line_id][0].append((rattr_idx, aattr))
      _logger.debug('step 1')
      log_reuse_group()

      # step 2: sorting
      for line_id, (group, reuse_distance_list) in reuse_groups.items():
        group.sort(key=lambda attr: tuple(reversed(attr[0])), reverse=True)
        for rattr_idx, aattr in group:
          reuse_distance_list.append(group[0][0][0] - rattr_idx[0])
      _logger.debug('step 2')
      log_reuse_group()

      # step 3: find inner-group redundancy
      InnerGroupReuseKeyType = Tuple[int, Index, Index, Tuple[Any, ...]]
      InnerGroupReuseValueType = List[Tuple[List[Tuple[Index, Any]], Index,
                                            Optional[CommSchedule],
                                            List[Tuple[int, Any]]]]

      # dict mappping (stride, reused_distances, not_reused_distances, aattrs)
      # to list of (attrs, line_id, reused_schedule, new_attrs)
      inner_group_reuses = collections.defaultdict(
          list)  # type: Dict[InnerGroupReuseKeyType, InnerGroupReuseValueType]

      def log_inner_group_reuses(logger=_logger.debug,
                                 inner_group_reuses=inner_group_reuses) -> None:
        for (stride, reused, not_reused,
             aattrs), groups in inner_group_reuses.items():
          logger('logging inner group reuses')
          logger(
              '  stride: %d; reused distances: %s; not reused: %s; aattrs: %s',
              stride, reused, not_reused, util.lst2str(aattrs))
          for attrs, line_id, reused_schedule, new_attrs in groups:
            logger('    attrs: %s; line: %s; schedule: %s; new attrs: %s',
                   attrs, line_id, reused_schedule, new_attrs)
        logger('')

      for line_id, (group, reuse_distance_list) in reuse_groups.items():
        if len(group) > 3:
          reuse_dict = {}
          for stride in range(reuse_distance_list[1], reuse_distance_list[-1]):
            distance_map = {
                dist: attr for attr, dist in zip(group, reuse_distance_list)
            }
            distances = list(reuse_distance_list)
            reused_lst = []
            not_reused_lst = []
            new_attrs = []
            while distances:
              dist = distances.pop(0)
              if dist + stride in distances and (
                  distance_map[dist][1],
                  distance_map[dist + stride][1]) == (distance_map[0][1],
                                                      distance_map[stride][1]):
                distances.remove(dist + stride)
                reused_lst.append(dist)
                left_attr = (self.linearizer(distance_map[stride][0]),
                             distance_map[stride][1])
                right_attr = (self.linearizer(distance_map[0][0]),
                              distance_map[0][1])
                reused_schedule = linear_schedule((left_attr, right_attr))
                new_attrs.append(
                    (self.linearizer(distance_map[dist + stride][0]),
                     reused_schedule))
              else:
                not_reused_lst.append(dist)
                new_attrs.append((self.linearizer(distance_map[dist][0]),
                                  distance_map[dist][1]))
            reuse_dict[stride] = tuple(reused_lst), tuple(
                not_reused_lst), reused_schedule, new_attrs
          # best stride is the one with most reused items and least stride
          if reuse_dict:
            _logger.debug('reuse dict: %s', reuse_dict)
            stride, (reused, not_reused, reused_schedule,
                     new_attrs) = max(reuse_dict.items(),
                                      key=lambda item:
                                      (len(item[1][0]), -item[0]))
            new_attrs.sort(key=lambda attr: attr[0])
            _, aattrs = zip(*new_attrs)
            inner_group_reuses[(stride, reused, not_reused, aattrs)].append(
                (group, line_id, reused_schedule, new_attrs))
            continue
        _logger.debug('no reuse found')
        _, aattrs = zip(*reversed(group))
        inner_group_reuses[(0, (), tuple(reuse_distance_list), aattrs)].append(
            (group, line_id, None, [(self.linearizer(rattr_idx), aattr)
                                    for rattr_idx, aattr in reversed(group)]))
      _logger.debug('step 3')
      log_inner_group_reuses()

      # step 4: find inter-group redundancy
      all_attrs = []  # type: List[Tuple[int, Optional[CommSchedule]]]
      for (stride, reused, not_reused, _), groups in inner_group_reuses.items():
        if len(groups) > 1 and len(reused) + len(not_reused) > 1:  # with reuse
          groups.sort(key=lambda item: item[1])
          group, line_id, optional_reused_schedule, new_attrs = groups[0]
          _logger.debug('groups[0]: %s', groups[0])
          reused_expr = linear_schedule(new_attrs)
          for group, line_id, optional_reused_schedule, new_attrs in groups:
            _logger.debug('reusing %s @ %d', reused_expr, new_attrs[0][0])
            all_attrs.append((new_attrs[0][0], reused_expr))
          for group, line_id, optional_reused_schedule, new_attrs in groups:
            _logger.debug('reuse distance list: %s', reuse_groups[line_id][1])
            _logger.debug(
                'direction id: %s, stride: %s, reused: %s, not_reused: %s',
                line_id, stride, reused, not_reused)
            break
        else:  # without reuse
          for group, line_id, optional_reused_schedule, new_attrs in groups:
            all_attrs.extend(new_attrs)
      all_attrs.sort(key=lambda attr: attr[0])
      _logger.debug('attrs: %s', util.lst2str(map(util.idx2str, all_attrs)))
      schedule = linear_schedule(all_attrs)
      _logger.debug('schedule: %s (%d)', schedule, schedule.num_ops)
      yield schedule

  @cached_property.cached_property
  def best(self) -> CommSchedule:
    return min(self.generator, key=lambda schedule: schedule.num_ops)

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    return


class ExternalSchedules(ScheduleBase):
  """Schedules of an Expression, found externally.
  """

  def __init__(self,
               rattrs: Tuple[RelativeAttr, ...],
               aattrs: Optional[Tuple[Union[AbsoluteAttr, CommSchedule, None],
                                      ...]] = None,
               linearizer: Optional[Linearizer] = None,
               cache: Optional[Dict] = None) -> None:
    self.linearizer = linearizer
    super().__init__(rattrs, aattrs)  # type: ignore
    self.cmd = ['soda-cr']

  def from_json(self, j: Dict[str, Any]) -> CommSchedule:
    return make_schedule_from_json(j, j['rattrs'][0], self.aattrs is None)

  @cached_property.cached_property
  def best(self) -> CommSchedule:
    attrs = {
        'rattrs': self.rattrs,
        'aattrs': self.aattrs or [1] * len(self.rattrs)
    }  # type: Dict[str, Any]
    if self.linearizer is not None and len(self.rattrs) >= 32:
      attrs['linearizer'] = {
          'maxs': self.linearizer.maxs,
          'mins': self.linearizer.mins,
          'sizes': self.linearizer.sizes,
      }
    if len(self.rattrs) < 32:
      attrs['num_pruned'] = 64
    elif len(self.rattrs) < 64:
      attrs['num_pruned'] = 4
    elif len(self.rattrs) < 128:
      attrs['num_pruned'] = 3
    elif len(self.rattrs) < 256:
      attrs['num_pruned'] = 2
    else:
      attrs['num_pruned'] = 1
    jemalloc = ctypes_util.find_library('jemalloc')
    if jemalloc is not None:
      os.environ['LD_PRELOAD'] = jemalloc
    result = json.loads(
        subprocess.run(self.cmd,
                       input=json.dumps(attrs),
                       stdout=subprocess.PIPE,
                       universal_newlines=True,
                       check=True).stdout)
    return self.from_json(result)

  def print_stats(self, logger: Callable[..., None] = _logger.info) -> None:
    pass


Schedule = CommSchedule
Schedules = Union[CommSchedules, GreedySchedules, GloreSchedules, BeamSchedules,
                  ExternalSchedules]


class Expression:
  """An expression suitable for computation reuse.

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

  def __init__(self, polynomial: ir.Node, stencil) -> None:
    """Figure out whether a ir.Node is suitable for computation reuse.

    Construct a CR Expression of the input polynomial. If it cannot be handled
    but is a valid ir.Node instance, it raises Expression.CannotHandle so that
    the recursive visitor can continue to find polynomials.

    Args:
      polynomial: ir.BinaryOp to work with.
      method: One of 'yes', 'greedy', 'optimal'.

    Raises:
      Expression.CannotHandle: If the input cannot be handled but is not error.
      TypeError: If the input is not an instance of ir.Node.
    """
    self.method = stencil.optimizations.get('computation-reuse') or 'greedy'
    reduction = ir.to_reduction(polynomial)
    if reduction is not None:
      self.operator = reduction[0]
      for operand in reduction[1]:
        if len(soda.visitor.get_load_set(operand)) > 1:
          raise Expression.CannotHandle('multi-index operands', str(operand))
        if not soda.visitor.get_load_set(operand):
          raise Expression.CannotHandle('const operand', str(operand))
      self.operands = tuple(
          sorted(reduction[1],
                 key=lambda x: tuple(
                     reversed(soda.visitor.get_load_set(x)[0].idx))))
      rattrs, aattrs = zip(*map(extract_attr, self.operands))
      self.aattrs_as_ir_nodes = aattrs

      self.linearizer = Linearizer(rattrs, stencil.tile_size)

      # linearize the relative attribtues and memorize the mapping
      self.rattrs = tuple(map(self.linearizer, rattrs))

      # linearize the absolute attributes
      if len(set(aattrs)) == 1:
        self.aattrs = None  # type: Optional[Tuple[int, ...]]
        self.aattr_table = {None: aattrs[0]} \
            # type: Dict[Optional[int], ir.Node]

      else:
        tag = 0
        operand_table = {}  # type: Dict[ir.Node, int]
        self.aattr_table = {}
        for aattr in aattrs:
          if aattr not in operand_table:
            operand_table[aattr] = tag
            self.aattr_table[tag] = aattr
            tag += 1
        self.aattrs = tuple(operand_table[aattr] for aattr in aattrs)

      _logger.debug('polynomial: %s%s', self.operator,
                    util.idx2str(self.operands))
      _logger.debug('rattrs: %s', util.idx2str(self.rattrs))
      _logger.debug('aattrs: %s', util.idx2str(self.aattrs_as_ir_nodes))
    elif isinstance(polynomial, ir.Node):
      raise Expression.CannotHandle(type(polynomial).__name__)
    else:
      raise TypeError('expect an instance of ir.BinaryOp')

  @cached_property.cached_property
  def schedules(self) -> Schedules:
    external_cr = shutil.which('soda-cr')
    args = self.rattrs, self.aattrs, self.linearizer
    if self.method == 'glore':
      return GloreSchedules(*args)
    if external_cr is None or self.method.startswith('built-in'):
      if self.method.endswith('optimal'):
        return CommSchedules(self.rattrs, self.aattrs, cache={})
      if self.method.endswith('greedy'):
        return GreedySchedules(*args)
      return BeamSchedules(*args)
    external_schedules = ExternalSchedules(*args)
    if self.method == 'optimal':
      external_schedules.cmd.append('--brute-force')
    elif self.method == 'greedy':
      external_schedules.cmd.append('--greedy')
    elif self.method == 'beam':
      external_schedules.cmd.append('--beam')
    return external_schedules

  @cached_property.cached_property
  def best_schedule(self) -> Schedule:
    return self.schedules.best.bind_expression(self)
