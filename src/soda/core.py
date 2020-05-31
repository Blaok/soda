import collections
import itertools
import logging
import operator
from typing import Dict, List, Tuple, Union

import cached_property
import pulp
import toposort
from haoda import ir, util
from haoda.ir import arithmetic

import soda.tensor
import soda.util
from soda import dataflow, visitor
from soda.optimization import computation_reuse as cr
from soda.optimization import inline

_logger = logging.getLogger().getChild(__name__)


class Stencil():
  """
  Attributes:
    iterate: int, number of iteration to implement.
    border: Reserved.
    preserve_border: Reserved.
    cluster: Reserved.
    burst_width: int, width of bits for DRAM burst access.
    app_name: str, application's name.
    tile_size: List of int.
    unroll_factor: int.
    replication_factor: int.
    optimizations: Set of enabled optimizations.
    dim: int.
    param_stmts: List of ParamStmt.
    input_stmts: List of InputStmt.
    local_stmts: List of LocalStmt.
    output_stmts: List of OutputStmt.

  Cached properties:
    tensors: Dict from str of name to Tensor.
    input_names: Tuple of str, names of input tensors.
    param_names: Tuple of str, names of param tensors.
    local_names: Tuple of str, names of local tensors.
    output_names: Tuple of str, names of output tensors.
  """

  def __init__(self, **kwargs):
    self.iterate = kwargs.pop('iterate')
    if self.iterate < 1:
      raise util.SemanticError('cannot iterate %d times' % self.iterate)
    self.border = kwargs.pop('border')
    self.preserve_border = self.border == 'preserve'
    self.cluster = kwargs.pop('cluster')
    # platform determined
    self.burst_width = kwargs.pop('burst_width')
    # application determined
    self.app_name = kwargs.pop('app_name')
    # parameters that can be explored
    self.tile_size = tuple(kwargs.pop('tile_size'))
    self.unroll_factor = kwargs.pop('unroll_factor')
    self.replication_factor = kwargs.pop('replication_factor')
    # stage-independent
    self.dim = kwargs.pop('dim')
    self.param_stmts = kwargs.pop('param_stmts')
    # stage-specific
    self.input_stmts = kwargs.pop('input_stmts')
    self.local_stmts = kwargs.pop('local_stmts')
    self.output_stmts = kwargs.pop('output_stmts')
    self.optimizations = {}
    if 'optimizations' in kwargs:
      self.optimizations = kwargs.pop('optimizations')

    if 'dram_in' in kwargs:
      dram_in = kwargs.pop('dram_in')
      if dram_in is not None:
        if ':' in dram_in:
          input_stmt_map = {_.name: _ for _ in self.input_stmts}
          for dram_map in dram_in.split('^'):
            var_name, bank_list = dram_map.split(':')
            if var_name not in input_stmt_map:
              raise util.SemanticError('no input named `{}`'.format(var_name))
            input_stmt_map[var_name].dram = tuple(map(int,
                                                      bank_list.split('.')))
        else:
          for input_stmt in self.input_stmts:
            input_stmt.dram = tuple(map(int, dram_in.split('.')))

    if 'dram_out' in kwargs:
      dram_out = kwargs.pop('dram_out')
      if dram_out is not None:
        if ':' in dram_out:
          output_stmt_map = {_.name: _ for _ in self.output_stmts}
          for dram_map in dram_out.split(','):
            var_name, bank_list = dram_map.split(':')
            if var_name not in output_stmt_map:
              raise util.SemanticError('no output named `{}`'.format(var_name))
            output_stmt_map[var_name].dram = tuple(
                map(int, bank_list.split('.')))
        else:
          for output_stmt in self.output_stmts:
            output_stmt.dram = tuple(map(int, dram_out.split('.')))

    if self.iterate > 1:
      if len(self.input_stmts) != len(self.output_stmts):
        raise util.SemanticError(
            'number of input tensors must be the same as output if iterate > 1 '
            'times, currently there are %d input(s) but %d output(s)' %
            (len(self.input_stmts), len(self.output_stmts)))
      if self.input_types != self.output_types:
        raise util.SemanticError(
            'input must have the same type(s) as output if iterate > 1 '
            'times, current input has type %s but output has type %s' %
            (util.lst2str(self.input_types), util.lst2str(self.output_types)))
      _logger.debug(
          'pipeline %d iterations of [%s] -> [%s]' %
          (self.iterate, ', '.join('%s: %s' % (stmt.haoda_type, stmt.name)
                                   for stmt in self.input_stmts), ', '.join(
                                       '%s: %s' % (stmt.haoda_type, stmt.name)
                                       for stmt in self.output_stmts)))

    for stmt in itertools.chain(self.local_stmts, self.output_stmts):
      _logger.debug('simplify %s', stmt.name)
      # LocalStmt and OutputStmt must remember the stencil object
      # for type propagation
      stmt.stencil = self
      stmt.expr = arithmetic.simplify(stmt.expr)
      stmt.let = arithmetic.simplify(stmt.let)

    self._cr_counter = 0
    cr.computation_reuse(self)
    if 'inline' in self.optimizations:
      inline.inline(self)
    inline.rebalance(self)

    for stmt in itertools.chain(self.local_stmts, self.output_stmts):
      stmt.propagate_type()

    # soda frontend successfully parsed
    _logger.debug('producer tensors: [%s]',
                  ', '.join(tensor.name for tensor in self.producer_tensors))
    _logger.debug('consumer tensors: [%s]',
                  ', '.join(tensor.name for tensor in self.consumer_tensors))

    # TODO: build Ref table and Var table
    # generate reuse buffers and get haoda nodes
    # pylint: disable=pointless-statement
    self.dataflow_super_source
    _logger.debug('dataflow: %s', self.dataflow_super_source)

    _logger.debug('module table: %s', dict(self.module_table))
    _logger.debug('module traits: %s', self.module_traits)

  def __str__(self) -> str:
    stmts = (self.input_stmts + self.param_stmts + self.local_stmts +
             self.output_stmts)
    return '''kernel: {0.app_name}
burst width: {0.burst_width}
iterate: {0.iterate}
unroll factor: {0.unroll_factor}
{stmts}
border: {0.border}
cluster: {0.cluster}'''.format(self, stmts='\n'.join(map(str, stmts)))

  @property
  def kernel_name(self) -> str:
    return f'{self.app_name}_kernel'

  @cached_property.cached_property
  def dataflow_super_source(self):
    return dataflow.create_dataflow_graph(self)

  @property
  def module_table(self):
    return self.dataflow_super_source.module_table

  @property
  def module_traits(self):
    return self.dataflow_super_source.module_traits

  def new_cr_var(self) -> str:
    while True:
      var = 'cr_var_%d' % (self._cr_counter)
      self._cr_counter += 1
      if var not in {
          stmt.name
          for stmt in self.input_stmts + self.local_stmts + self.output_stmts
      }:
        return var

  @cached_property.cached_property
  def stmt_table(
      self
  ) -> Dict[str, Union[soda.grammar.InputStmt, soda.grammar.LocalStmt,
                       soda.grammar.OutputStmt, soda.grammar.ParamStmt]]:
    return {
        stmt.name: stmt for stmt in self.input_stmts + self.local_stmts +
        self.output_stmts + self.param_stmts
    }

  @cached_property.cached_property
  def input_types(self):
    return tuple(tensor.haoda_type for tensor in self.input_stmts)

  @cached_property.cached_property
  def param_types(self):
    return tuple(tensor.haoda_type for tensor in self.param_stmts)

  @cached_property.cached_property
  def local_types(self):
    return tuple(tensor.haoda_type for tensor in self.local_stmts)

  @cached_property.cached_property
  def output_types(self):
    return tuple(tensor.haoda_type for tensor in self.output_stmts)

  @cached_property.cached_property
  def input_names(self):
    return tuple(stmt.name for stmt in self.input_stmts)

  @cached_property.cached_property
  def param_names(self):
    return tuple(stmt.name for stmt in self.param_stmts)

  @cached_property.cached_property
  def local_names(self):
    return tuple(stmt.name for stmt in self.local_stmts)

  @cached_property.cached_property
  def output_names(self):
    return tuple(stmt.name for stmt in self.output_stmts)

  @cached_property.cached_property
  def symbol_table(self) -> Dict[str, str]:
    """Constructs a dict mapping tensors' names to haoda types.

    Constructs a symbol table mapping names of the tensors to their haoda types.
    This table does not contain the local variables (ir.Let) in the LocalStmt
    and OutputStmt. For those, use stmt.symbol_table.

    Returns:
      symbol_table: A dict mapping tensors' names (str) to haoda types (str).

    Raises:
      util.InputError if names conflict.
    """
    symbol_table: Dict[str, str] = {}
    for name, haoda_type in zip(
        itertools.chain(self.input_names, self.local_names, self.output_names),
        itertools.chain(self.input_types, self.local_types, self.output_types)):
      if name in symbol_table:
        raise util.InputError('conflicting stmt name: %s' % name)
      symbol_table[name] = haoda_type
    return symbol_table

  @property
  def propagate_type(self):
    """Returns a callable that propagates type, optionally for a specific stmt.

    Returns:
      propagate_type: A callable that propagates type. It takes an optional
          argument of stmt, used for propagating types including the local
          variables (ir.Let).
    """

    def propagate_type(node, stmt=None):
      symbol_table = self.symbol_table
      if stmt is not None:
        symbol_table = stmt.symbol_table
      return arithmetic.base.propagate_type(node, symbol_table)

    return propagate_type

  @cached_property.cached_property
  def norm_refs(self):
    norm_refs = {}

    def get_norm_idx(stmt) -> Tuple[int, ...]:
      norm_idx = norm_refs.get(stmt.name)
      if norm_idx is None:
        loads = visitor.get_load_tuple(stmt.expr)
        for let in stmt.let:
          loads += visitor.get_load_tuple(let)

        def all_indices():
          for load in loads:
            if load.name in self.input_names:
              yield load.idx
            else:
              yield tuple(x + y for x, y in zip(
                  load.idx, get_norm_idx(self.stmt_table[load.name])))

        norm_idx = tuple(x - y for x, y in zip(
            min(all_indices(), key=lambda idx: tuple(reversed(tuple(idx)))),
            stmt.ref.idx))
        norm_refs[stmt.name] = norm_idx
        _logger.debug('%s has norm idx %s', stmt.name, norm_idx)
      return norm_idx

    for stmt in self.local_stmts + self.output_stmts:
      get_norm_idx(stmt)
    _logger.debug('norm refs %s', norm_refs)
    return norm_refs

  @cached_property.cached_property
  def tensors(self):
    """Constructs high-level DAG and creates the tensors.

    Returns:
      An collections.OrderedDict mapping a tensor's name to the tensor.
    """
    # TODO: check for name conflicts
    tensor_map = collections.OrderedDict()
    for stmt in self.input_stmts:
      tensor = soda.tensor.Tensor(stmt, self.tile_size)
      tensor_map[stmt.name] = tensor

    def name_in_iter(name, iteration):
      if name in self.input_names:
        if iteration > 0:
          return name + '_iter%d' % iteration
        return name
      if name in self.output_names:
        if iteration < self.iterate - 1:
          return (self.input_names[self.output_names.index(name)] + '_iter%d' %
                  (iteration + 1))
        return name
      if name in self.local_names:
        if iteration > 0:
          return name + '_iter%d' % iteration
        return name
      if name in self.param_names:
        return name
      raise util.InternalError('unknown name: %s' % name)

    for iteration in range(self.iterate):
      _logger.debug('iterate %s', iteration)
      _logger.debug('map: %s', self.symbol_table)

      def mutate_name_callback(obj, mutated):
        if isinstance(obj, ir.Ref):
          obj.haoda_type = self.symbol_table[obj.name]
          # pylint: disable=cell-var-from-loop
          obj.name = name_in_iter(obj.name, iteration)
        return obj

      tensors = []
      for stmt in itertools.chain(self.local_stmts, self.output_stmts):
        tensor = soda.tensor.Tensor(stmt.visit(mutate_name_callback),
                                    self.tile_size)
        tensor_map[tensor.name] = tensor
        tensors.append(tensor)

      for tensor in tensors:
        _logger.debug('%s', tensor)

      for tensor in tensors:
        tensor.propagate_type()
        loads = visitor.get_load_dict(tensor)
        for parent_name, ld_refs in loads.items():
          ld_refs = sorted(
              ld_refs,
              key=lambda ref: soda.util.serialize(ref.idx, self.tile_size))
          parent_tensor = tensor_map[parent_name]
          parent_tensor.children[tensor.name] = tensor
          tensor.parents[parent_name] = parent_tensor
          tensor.ld_refs[parent_name] = ld_refs

    # solve ILP for optimal reuse buffer
    lp_problem = pulp.LpProblem("optimal_reuse_buffer", pulp.LpMinimize)
    lp_vars = {self.input_names[0]: 0}  # set 1 and only 1 reference point
    lp_helper_vars = {}  # type: Dict[str, pulp.LpVariable]
    objectives = []
    constraints = []
    for tensor in tensor_map.values():
      lp_var = pulp.LpVariable('produced_offset_' + tensor.name, cat='Integer')
      lp_helper_var = pulp.LpVariable('consumed_offset_' + tensor.name,
                                      cat='Integer')
      lp_vars.setdefault(tensor.name, lp_var)
      lp_helper_vars[tensor.name] = lp_helper_var
      # tensor need to be kept for this long
      objectives.append(lp_helper_var - lp_vars[tensor.name])
      # tensor cannot be consumed until it is produced
      constraints.append(lp_helper_var >= lp_vars[tensor.name])
    lp_problem += sum(objectives)
    lp_problem.extend(constraints)
    for st_tensor in tensor_map.values():
      for ld_tensor_name, offsets in st_tensor.ld_offsets.items():
        oldest_access = min(offsets)
        newest_access = max(offsets)
        _logger.debug('%s @ %s accesses %s @ [%s, %s]', st_tensor.name,
                      st_tensor.st_offset, ld_tensor_name, oldest_access,
                      newest_access)
        # newest ld_tensor access must have been produced
        # when st_tensor is produced
        lp_problem += lp_vars[ld_tensor_name] <= lp_vars[st_tensor.name] + (
            st_tensor.st_offset - newest_access)
        # oldest ld_tensor access must have been not consumed
        # when st_tensor is produced
        lp_problem += lp_helper_vars[ld_tensor_name] >= lp_vars[
            st_tensor.name] + (st_tensor.st_offset - oldest_access)

    lp_status = lp_problem.solve()
    lp_status_str = pulp.LpStatus[lp_status]
    total_distance = int(pulp.value(lp_problem.objective))
    _logger.debug('ILP status: %s %s', lp_status_str, total_distance)
    _logger.info('total reuse distance: %d', total_distance)

    if lp_status != pulp.LpStatusOptimal:
      _logger.error('ILP error: %s\n%s', lp_status_str, lp_problem)
      raise util.InternalError('unexpected ILP status: %s' % lp_status_str)

    # some inputs may need to be delayed relative to others
    base = min(int(pulp.value(lp_vars[x])) for x in self.input_names)

    # set produce offsets
    for tensor in tensor_map.values():
      produce_offset = int(pulp.value(lp_vars[tensor.name])) - base
      consume_offset = int(pulp.value(lp_helper_vars[tensor.name])) - base
      tensor.produce_offset = produce_offset
      tensor.consume_offset = consume_offset
      tensor.max_access = 0  # pixels before current produce
      _logger.debug('%s should be produced @ %d and kept until %d', tensor.name,
                    produce_offset, consume_offset)

    # calculate overall acceses
    for ld_tensor in tensor_map.values():
      for st_tensor in ld_tensor.children.values():
        oldest_access = st_tensor.st_offset - min(
            st_tensor.ld_offsets[ld_tensor.name]
        ) + st_tensor.produce_offset - ld_tensor.produce_offset
        newest_access = st_tensor.st_offset - max(
            st_tensor.ld_offsets[ld_tensor.name]
        ) + st_tensor.produce_offset - ld_tensor.produce_offset
        _logger.debug(
            '  producing %s @ %s accesses [%s, %s] pixels before %s '
            'produced @ %s', st_tensor.name, st_tensor.produce_offset,
            newest_access, oldest_access, ld_tensor.name,
            ld_tensor.produce_offset)
        ld_tensor.max_access = max(ld_tensor.max_access, oldest_access)

    for tensor in tensor_map.values():
      _logger.debug('%s should be kept for %s pixels', tensor.name,
                    tensor.max_access)

    # high-level DAG construction finished
    for tensor in tensor_map.values():
      if tensor.name in self.input_names:
        _logger.debug('<input tensor>: %s', tensor)
      elif tensor.name in self.output_names:
        _logger.debug('<output tensor>: %s', tensor)
      else:
        _logger.debug('<local tensor>: %s', tensor)
    return tensor_map

  @cached_property.cached_property
  def chronological_tensors(self) -> List[soda.tensor.Tensor]:
    """Computes the offsets of tensors.

    Returns:
      A list of Tensor, in chronological order.
    """
    return list(
        map(
            self.tensors.get,
            toposort.toposort_flatten({
                tensor.name: set(tensor.parents)
                for tensor in self.tensors.values()
            })))

  @cached_property.cached_property
  def input_partition(self):
    pixel_width_i = sum(self.pixel_width_i)
    if (self.burst_width / pixel_width_i * self.dram_bank / 2 >
        self.unroll_factor / 2):
      return int(self.burst_width / pixel_width_i * self.dram_bank / 2)
    return int(self.unroll_factor / 2)

  @cached_property.cached_property
  def output_partition(self):
    pixel_width_o = sum(self.pixel_width_o)
    if (self.burst_width / pixel_width_o * self.dram_bank / 2 >
        self.unroll_factor / 2):
      return int(self.burst_width / pixel_width_o * self.dram_bank / 2)
    return int(self.unroll_factor / 2)

  @cached_property.cached_property
  def pixel_width_i(self):
    return [x.width_in_bits for x in self.input_stmts]

  @cached_property.cached_property
  def pixel_width_o(self):
    return [x.width_in_bits for x in self.output_stmts]

  @cached_property.cached_property
  def producer_tensors(self):
    return tuple(filter(soda.tensor.Tensor.is_producer, self.tensors.values()))

  @cached_property.cached_property
  def consumer_tensors(self):
    return tuple(filter(soda.tensor.Tensor.is_consumer, self.tensors.values()))

  @cached_property.cached_property
  def reuse_buffers(self):
    """Constructs the reuse buffers.

    Returns:
      A dict mapping a tensor's name to its reuse buffers.
    """
    unroll_factor = self.unroll_factor
    self._reuse_buffer_lengths = {}
    reuse_buffers = {}
    for tensor in self.producer_tensors:
      reuse_buffer = _get_reuse_buffer(self.tile_size, tensor, unroll_factor)
      reuse_buffer_length = {}
      reuse_buffers[tensor.name] = reuse_buffer
      self._reuse_buffer_lengths[tensor.name] = reuse_buffer_length
      first = [True] * unroll_factor
      for start, end in reuse_buffer[1:]:
        if first[start % unroll_factor]:
          first[start % unroll_factor] = False
          if start >= unroll_factor:
            reuse_buffer_length[end] = end // unroll_factor
            continue
        reuse_buffer_length[end] = (end - start) // unroll_factor
    return reuse_buffers

  @cached_property.cached_property
  def all_points(self):
    all_points = {}
    for tensor in self.producer_tensors:
      all_points[tensor.name] = _get_points(self.tile_size, tensor,
                                            self.unroll_factor)
    return all_points

  @cached_property.cached_property
  def next_fifo(self):
    """Constructs the next fifo offset mapping.

    Returns:
      A dict mapping a tensor's name and offset to the next offset.
    """
    next_fifo = {}
    for name, reuse_buffer in self.reuse_buffers.items():
      next_fifo[name] = {}
      for start, end in reuse_buffer[1:]:
        if start < end:
          next_fifo[name][start] = end
    _logger.debug('next_fifo: %s' % next_fifo)
    return next_fifo

  @cached_property.cached_property
  def reuse_buffer_lengths(self):
    """Constructs the reuse buffer lengths.

    Returns:
      A dict mapping a tensor's name to its reuse buffers' lengths.
    """
    # pylint: disable=pointless-statement
    self.reuse_buffers
    return self._reuse_buffer_lengths

  def get_replicated_next_fifo(self):
    if not hasattr(self, 'replicated_next_fifo'):
      self.replicated_next_fifo = {}
      for name, reuse_buffer in self.get_replicated_reuse_buffers().items():
        self.replicated_next_fifo[name] = {}
        for start, end in reuse_buffer[1:]:
          if start < end:
            self.replicated_next_fifo[name][start] = end
      _logger.debug('replicated_next_fifo: %s' % self.replicated_next_fifo)
    return self.replicated_next_fifo

  def get_replicated_reuse_buffer_length(self, name, offset):
    if not hasattr(self, 'replicated_reuse_buffer_lengths'):
      self.get_replicated_reuse_buffers()
    return self.replicated_reuse_buffer_lengths[name][offset]

  def get_replicated_reuse_buffers(self):
    if not hasattr(self, 'replicated_reuse_buffers'):
      replication_factor = self.replication_factor
      self.replicated_reuse_buffer_lengths = {}
      self.replicated_reuse_buffers = {}
      for tensor in self.producer_tensors:
        reuse_buffer = _get_replicated_reuse_buffer(self.tile_size, tensor,
                                                    replication_factor)
        self.replicated_reuse_buffers[tensor.name] = reuse_buffer
        self.replicated_reuse_buffer_lengths[tensor.name] = {}
        first = [True] * self.replication_factor
        for start, end in reuse_buffer[1:]:
          if first[start % replication_factor]:
            first[start % replication_factor] = False
            if start >= replication_factor:
              self.replicated_reuse_buffer_lengths[
                  tensor.name][end] = end // replication_factor
              continue
          self.replicated_reuse_buffer_lengths[
              tensor.name][end] = (end - start) // replication_factor
      _logger.debug('replicated_reuse_buffers: %s' %
                    self.replicated_reuse_buffers)
      _logger.debug('replicated_reuse_buffer_lengths: %s' %
                    self.replicated_reuse_buffer_lengths)
    return self.replicated_reuse_buffers

  def get_replicated_all_points(self):
    if not hasattr(self, 'replicated_all_points'):
      self.replicated_all_points = {}
      for tensor in self.producer_tensors:
        self.replicated_all_points[tensor.name] = _get_replicated_points(
            self.tile_size, tensor)
      _logger.debug('replicated_all_points: %s' % self.replicated_all_points)
    return self.replicated_all_points

  def _calculate_stencil_window(self) -> None:
    stencil_window = get_overall_stencil_window(
        map(self.tensors.get, self.input_names),
        self.tensors[self.output_names[0]])
    stencil_distance = get_stencil_distance(stencil_window, self.tile_size)
    stencil_offset = stencil_distance - soda.util.serialize(
        get_stencil_window_offset(stencil_window), self.tile_size)

    self._stencil_window = stencil_window
    self._stencil_distance = max(stencil_distance, stencil_offset)

  @property
  def stencil_distance(self) -> int:
    if not hasattr(self, '_stencil_distance'):
      self._calculate_stencil_window()
    return getattr(self, '_stencil_distance')

  @property
  def stencil_window(self) -> int:
    if not hasattr(self, '_stencil_window'):
      self._calculate_stencil_window()
    return getattr(self, '_stencil_window')


def _get_reuse_chains(tile_size, tensor, unroll_factor):
  """Generates reuse chains for a Tensor.

  Generates reuse chains for a Tensor under the given tile size and unroll
  factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the reuse chains belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A list of tuples where each tuple represents a reuse chain and each
    element of the tuple represents the offset from the lastest input.
  """

  _logger.debug('get reuse chains of tensor %s', tensor.name)

  def unroll_offsets(child):
    unrolled_offsets = set()
    for unroll_idx in range(unroll_factor):
      for offset in child.ld_offsets[tensor.name]:
        unrolled_offsets.add(unroll_idx + child.st_offset - offset +
                             child.produce_offset - tensor.produce_offset)
    return unrolled_offsets

  A_dag = set()
  for child in tensor.children.values():
    A_dag |= unroll_offsets(child)
  _logger.debug('Aâ€  of tensor %s: %s', tensor.name, A_dag)

  chains = []
  for chain_idx in reversed(range(unroll_factor)):
    chains.append(
        tuple(
            sorted(offset for offset in A_dag
                   if offset % unroll_factor == chain_idx)))
  _logger.debug('reuse chains: %s', chains)

  for idx, chain in enumerate(chains):
    _logger.debug('reuse chain %d of tensor %s: %s', idx, tensor.name, chain)
  return chains


def _get_points(tile_size, tensor, unroll_factor):
  """Generates offset-to-point mapping for a Tensor.

  Generates a mapping which can be used to determine the accessed point index
  from the offset for a Tensor, under the given tile size and unroll factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the mapping belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A dict of name str to a dict of offset to a dict of unroll index to
    point index.
  """

  all_points = {}  # {name: {offset: {unroll_idx: point_idx}}}
  for child in tensor.children.values():
    all_points[child.name] = {}
    offsets = child.ld_offsets[tensor.name]
    for unroll_idx in range(unroll_factor):
      for idx, offset in enumerate(offsets):
        all_points[child.name].setdefault(
            unroll_idx + child.st_offset - offset + child.produce_offset -
            tensor.produce_offset, {})[unroll_factor - 1 - unroll_idx] = idx
  for child in tensor.children.values():
    for offset, points in all_points[child.name].items():
      for unroll_idx, point in points.items():
        _logger.debug(
            '%s <- %s @ offset=%d <=> %s @ unroll_idx=%d', child.name,
            tensor.name, offset,
            util.idx2str(
                list(child.ld_indices[tensor.name].values())[point].idx),
            unroll_idx)
  return all_points


def _get_reuse_buffer(tile_size, tensor, unroll_factor):
  """Generates reuse buffer for a Tensor.

  Generates a list representing the reuse buffer for a Tensor, under the given
  tile size and unroll factor.

  Args:
    tile_size: An iterable representing the tile size in each dimension.
    tensor: A Tensor to which the mapping belongs.
    unroll_factor: An int representing the unroll factor.

  Returns:
    A list whose first element is an int representing the length of the
    reuse buffer (capacity of data element), followed by unroll_factor
    number of (start, end) tuples, where start and end are the offsets from
    the lastest input of each piece of the reuse buffer.
  """

  reuse_buffer = [None]  # [length, (start, end), (start, end), ...]
  offsets = []
  for chain_id, chain in enumerate(
      _get_reuse_chains(tile_size, tensor, unroll_factor)):
    reuse_buffer.append((unroll_factor - 1 - chain_id, chain[0]))
    _logger.debug('chain id: %s, chain[0]: %s', chain_id, chain[0])
    offsets.append(chain[0])
    for j in range(len(chain) - 1):
      reuse_buffer.append((chain[j], chain[j + 1]))
      offsets.append(chain[j + 1])
  reuse_buffer[0] = max(offsets) + 1
  _logger.debug('reuse chains of tensor %s: %s' % (tensor.name, reuse_buffer))
  return reuse_buffer


def _get_replicated_reuse_chains(tile_size, tensor, replication_factor):
  _logger.debug('\033[1mget replicated reuse chains of tensor %s\033[0m' %
                tensor.name)
  A_dag = set()
  for stage in tensor.children:
    offsets = soda.util.serialize_iter(stage.window[tensor.name], tile_size)
    A_dag |= {
        max(offsets) - offset + stage.delay[tensor.name] for offset in offsets
    }
  _logger.debug('Aâ€  of tensor %s: %s' % (tensor.name, A_dag))
  chains = sum(
      reversed([
          tuple(
              [tuple(sorted(x
                            for x in A_dag
                            if x % replication_factor == i))])
          for i in range(replication_factor)
      ]), ())
  for idx, chain in enumerate(chains):
    _logger.debug('reuse chain %d of tensor %s: %s' % (idx, tensor.name, chain))
  return chains


def _get_replicated_points(tile_size, tensor):
  all_points = {}  # {name:{offset:point_index}}
  for stage in tensor.children:
    all_points[stage.name] = {}
    offsets = soda.util.serialize_iter(stage.window[tensor.name], tile_size)
    max_offset = max(offsets)
    for idx, offset in enumerate(offsets):
      all_points[stage.name][max_offset - offset +
                             stage.delay[tensor.name]] = idx
  for stage in tensor.children:
    for offset, points in all_points[stage.name].items():
      _logger.debug('%s <- %s @ offset=%d <=> (%s)' %
                    (stage.name, tensor.name, offset, ', '.join(
                        map(str, stage.window[tensor.name][points]))))
  return all_points


def _get_replicated_reuse_buffer(tile_size, tensor, replication_factor):
  reuse_buffer = [None]  # [length, (start, end), (start, end), ...]
  offsets = []
  for chain in _get_replicated_reuse_chains(tile_size, tensor,
                                            replication_factor):
    if chain:
      reuse_buffer.append((chain[0], chain[0]))
      offsets.append(chain[0])
      for j in range(len(chain) - 1):
        reuse_buffer.append((chain[j], chain[j + 1]))
        offsets.append(chain[j + 1])
  reuse_buffer[0] = max(offsets) + 1
  _logger.debug('reuse chains of tensor %s: %s' % (tensor.name, reuse_buffer))
  return reuse_buffer


def get_indices_id(indices):
  return '_'.join(str(idx).replace('-', 'm') for idx in indices)


def get_stencil_distance(stencil_window, tile_size):
  return (
      max(soda.util.serialize_iter(stencil_window, tile_size)) +
      soda.util.serialize(get_stencil_window_offset(stencil_window), tile_size))


def get_stencil_dim(points):
  dimension = len(next(iter(points)))
  return [
      max_index - min_index + 1 for max_index, min_index in
      zip([max([point[dim] for point in points]) for dim in range(dimension)],
          [min([point[dim] for point in points]) for dim in range(dimension)])
  ]

_overall_stencil_window_cache = {} \
    # type: Dict[Tuple[int, int], Tuple[Tuple[int, ...], ...]]


def get_overall_stencil_window(input_tensor, output_tensor):
  if isinstance(input_tensor, collections.Iterable):
    all_points = tuple(
        sorted(
            set.union(*(set(get_overall_stencil_window(_, output_tensor))
                        for _ in input_tensor))))
    _logger.debug('overall stencil window of %s (%s) <- {%s} is %s (%d points)',
                  output_tensor.name,
                  ', '.join(['0'] * len(output_tensor.st_idx)),
                  ', '.join(_.name for _ in input_tensor), all_points,
                  len(all_points))
    return all_points
  # normalize store index to 0
  idx = (id(input_tensor), id(output_tensor))
  if idx in _overall_stencil_window_cache:
    return _overall_stencil_window_cache[idx]
  _logger.debug('get overall stencil window of %s <- %s', output_tensor.name,
                input_tensor.name)
  all_points = set()
  for name, points in output_tensor.ld_indices.items():
    _logger.debug('%s@%s <- %s', output_tensor.name,
                  util.idx2str(output_tensor.st_idx),
                  util.idx2str(points.values()))
    if name != input_tensor.name:
      recursive_points = get_overall_stencil_window(input_tensor,
                                                    output_tensor.parents[name])
      _logger.debug('recursive points: %s', util.idx2str(recursive_points))
      all_points |= set.union(*[{
          tuple(map(lambda a, b, c: a + b - c, _, point, output_tensor.st_idx))
          for _ in recursive_points
      }
                                for point in points])
    else:
      all_points |= {
          tuple(map(operator.sub, point, output_tensor.st_idx))
          for point in points
      }
  all_points = tuple(sorted(all_points))
  _logger.debug('overall stencil window of %s (%s) <- %s is %s (%d points)',
                output_tensor.name,
                ', '.join(['0'] * len(output_tensor.st_idx)), input_tensor.name,
                all_points, len(all_points))
  _overall_stencil_window_cache[idx] = all_points
  return all_points


def get_stencil_window_offset(stencil_window):
  # only works if window is normalized to store at 0
  return tuple(-min(p[d]
                    for p in stencil_window)
               for d in range(len(next(iter(stencil_window)))))
