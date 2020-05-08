# coding: utf-8
import collections
import itertools
import logging
from typing import Dict, Iterator, List, Tuple

import cached_property
import pulp

from haoda import ir, util

_logger = logging.getLogger().getChild(__name__)


class SuperSourceNode(ir.Module):
  """A node representing the super source in the dataflow graph.

  A super source doesn't have parent nodes.

  Attributes:
      fwd_nodes (Dict[Tuple[str, int], ForwardNode]): Dict mapping tuples of
        (tensor name, offset) to nodes.
      cpt_nodes (Dict[Tuple[str, int], ComputeNode]): Dict mapping tuples of
        (stage_name, pe_id) to nodes.
      super_sink (SuperSinkNode): The super sink node of this DAG.
  """

  def __init__(
      self,
      fwd_nodes: Dict[Tuple[str, int], 'ForwardNode'],
      cpt_nodes: Dict[Tuple[str, int], 'ComputeNode'],
      super_sink: 'SuperSinkNode',
  ):
    super().__init__()
    self.fwd_nodes = fwd_nodes
    self.cpt_nodes = cpt_nodes
    self.super_sink = super_sink

  @property
  def graphviz(self) -> str:
    output = 'digraph {\n'
    for src, dst in self.dfs_edge_gen():
      output += f'  "{repr(src)}" -> "{repr(dst)}"\n'
    output += '}\n'
    return output

  def verify_mode_depths(self) -> None:
    latency_table = {}
    lp_problem = pulp.LpProblem('verify_fifo_depths', pulp.LpMinimize)
    for node in self.tpo_valid_node_gen():
      if self in node.parents:
        latency_table[node] = 0
      else:
        latency_table[node] = pulp.LpVariable(
            name=f'latency_{node.name}',
            lowBound=0,
            cat='Integer',
        )
        lp_problem.extend(
            parent.get_latency(node) +
            latency_table[parent] <= latency_table[node]
            for parent in node.parents)
        lp_problem.extend(
            parent.get_latency(node) + latency_table[parent] +
            parent.fifo(node).depth >= latency_table[node]
            for parent in node.parents)

    lp_status = lp_problem.solve()
    if lp_status == pulp.LpStatusOptimal:
      _logger.debug('II=1 check: PASS')
    elif lp_status == pulp.LpStatusInfeasible:
      _logger.warn('II=1 check: FAIL')
    else:
      lp_status_str = pulp.LpStatus[lp_status]
      _logger.error('ILP error: %s\n%s', lp_status_str, lp_problem)
      raise util.InternalError('unexpected ILP status: %s' % lp_status_str)

    for node in self.tpo_valid_node_gen():
      if self in node.parents:
        min_capacity = 0
      else:
        min_capacity = min(
            parent.get_latency(node) + int(pulp.value(latency_table[parent])) +
            parent.fifo(node).depth for parent in node.parents)

      debug_enabled = _logger.isEnabledFor(logging.DEBUG)
      check_failed = int(pulp.value(latency_table[node])) > min_capacity
      if debug_enabled or check_failed:
        (_logger.debug if debug_enabled else _logger.warn)(
            'II=1 check %s: %s: latency %d %s min capacity %d',
            '✖' if check_failed else '✔',
            repr(node),
            int(pulp.value(latency_table[node])),
            '>' if check_failed else '<=',
            min_capacity,
        )

  def update_module_depths(
      self,
      depths: Dict[int, int],
  ) -> None:
    """Update module pipeline depths and FIFO depths.

    The FIFO depths are determined by solving an ILP problem:

    + Optimization objective: minimize the sum (weighted by FIFO width) of all
    FIFO depths.
    + Constraints: the whole DAG can be fully pipelined without artificial
    stalls.

    For every non-overlapping path between a pair of nodes,
    the latency of each token is the maximum minimum latency among all paths.
    To enable full pipelining,
    this latency must not exceed the maximum latency of any path.
    The minimum latency of each path is the sum of the FIFO write latency in
    each module and the number of edges (FIFOs),
    since the minimum latency of a FIFO is 1.
    The maximum latency of each path is the sum of the FIFO write latency in
    each module and the total depth of FIFOs.

    Args:
        depths (Dict[int, int]): Dict mapping module ids to pipeline depths.
    """
    # update module pipeline depths
    for src_node, dst_node in self.bfs_valid_edge_gen():
      module_id = self.module_table[src_node][1]
      depth = depths.get(module_id)
      if depth is not None:
        fifo = src_node.fifo(dst_node)
        if fifo.write_lat != depth:
          _logger.debug('%s write latency changed %s -> %d', fifo,
                        fifo.write_lat, depth)
          fifo.write_lat = depth

    # set up ILP problem, variables, and objective
    lp_problem = pulp.LpProblem('optimal_fifo_depths', pulp.LpMinimize)
    lp_vars = {}
    for src_node, dst_node in self.bfs_valid_edge_gen():
      lp_vars[(src_node, dst_node)] = pulp.LpVariable(
          name=f'depth_{src_node.fifo(dst_node).c_expr}',
          lowBound=0,
          cat='Integer',
      )
    lp_problem += sum(
        x.fifo(y).haoda_type.width_in_bits * v for (x, y), v in lp_vars.items())

    # add ILP constraints
    latency_table = {
        x: pulp.LpVariable(name=f'latency_{x.name}', lowBound=0, cat='Integer')
        for x in self.tpo_valid_node_gen()
    }
    for node in self.tpo_valid_node_gen():
      if self in node.parents:
        latency_table[node] = 0
      else:
        lp_problem.extend(
            parent.get_latency(node) +
            latency_table[parent] <= latency_table[node]
            for parent in node.parents)
        lp_problem.extend(
            parent.get_latency(node) + latency_table[parent] +
            lp_vars[(parent, node)] >= latency_table[node]
            for parent in node.parents)

    # solve ILP
    lp_status = lp_problem.solve()
    if lp_status != pulp.LpStatusOptimal:
      lp_status_str = pulp.LpStatus[lp_status]
      _logger.error('ILP error: %s\n%s', lp_status_str, lp_problem)
      raise util.InternalError('unexpected ILP status: %s' % lp_status_str)

    # update FIFO depths
    for (src_node, dst_node), lp_var in lp_vars.items():
      depth = int(pulp.value(lp_var))
      fifo = src_node.fifo(dst_node)
      if fifo.depth != depth:
        _logger.debug('%s * depth %d -> %d', fifo, fifo.depth, depth)
        fifo.depth = depth

    self.verify_mode_depths()

  @property
  def name(self):
    return 'super_source'

  def __repr__(self) -> str:
    return '\033[35msuper source\033[0m'

  @cached_property.cached_property
  def module_table(self) -> Dict[ir.Node, Tuple[ir.ModuleTrait, int]]:
    """Module table maps an IR node to (module_trait, module_id).

    Returns:
      A dict mapping an IR node to (module_trait, module_id) tuple.
    """
    self._module_traits: Dict[ir.ModuleTrait,
                              List[ir.Node]] = collections.OrderedDict()
    module_table: Dict[ir.Node, Tuple[ir.ModuleTrait,
                                      int]] = collections.OrderedDict()

    for node in self.tpo_valid_node_gen():
      self._module_traits.setdefault(ir.ModuleTrait(node), []).append(node)
    for idx, module_trait in enumerate(self._module_traits):
      for node in self._module_traits[module_trait]:
        module_table[node] = module_trait, idx
    return module_table

  @cached_property.cached_property
  def module_traits(self) -> Tuple[ir.ModuleTrait, ...]:
    return tuple(self.module_trait_table)

  @property
  def module_trait_table(self) -> Dict[ir.ModuleTrait, List[ir.Node]]:
    # pylint: disable=pointless-statement
    self.module_table
    return self._module_traits

  def tpo_valid_node_gen(self) -> Iterator[ir.Module]:
    """Traverse valid descendant nodes in tpological order.

    Load and store nodes are ordered in the same way as they are specified in
    soda files.

    Yields:
        Iterator[ir.Module]: Nodes that are not super source or super sink.
    """
    yield from self.load_nodes
    yield from filter(
        lambda x: not isinstance(x, MemoryNode) and is_valid_node(x),
        super().tpo_node_gen(),
    )
    yield from self.store_nodes

  def bfs_valid_edge_gen(self) -> Iterator[ir.Module]:
    return filter(is_valid_edge, self.bfs_edge_gen())

  @property
  def load_nodes(self) -> Tuple['LoadNode', ...]:
    return self.children

  @property
  def store_nodes(self) -> Tuple['StoreNode', ...]:
    return self.super_sink.parents


class SuperSinkNode(ir.Module):
  """A node representing the super sink in the dataflow graph.

  A super sink doesn't have child nodes.
  """

  @property
  def name(self):
    return 'super_sink'

  def __repr__(self) -> str:
    return '\033[34msuper sink\033[0m'


class ForwardNode(ir.Module):
  """A node representing a forward module in the dataflow graph.

  Attributes:
    tensor: Tensor corresponding to this node.
    offset: Int representing the offset of this tensor.
  """

  def __init__(self, **kwargs):
    super().__init__()
    self.tensor = kwargs.pop('tensor')
    self.offset = kwargs.pop('offset')

  def __repr__(self):
    return '\033[32mforward %s @%d\033[0m' % (self.tensor.name, self.offset)

  @property
  def name(self):
    return '{}_offset_{}'.format(self.tensor.name, self.offset)


class ComputeNode(ir.Module):
  """A node representing a compute module in the dataflow graph.

  Attributes:
    tensor: Tensor corresponding to this node.
    pe_id: Int representing the PE id.
    fifo_map: {str: {idx: Node}}
  """

  def __init__(self, **kwargs):
    super().__init__()
    self.tensor = kwargs.pop('tensor')
    self.pe_id = kwargs.pop('pe_id')
    self.fifo_map = collections.defaultdict(dict)

  def __repr__(self):
    return '\033[31mcompute %s #%d\033[0m' % (self.tensor.name, self.pe_id)

  @property
  def name(self):
    return '{}_pe_{}'.format(self.tensor.name, self.pe_id)


class MemoryNode(ir.Module):

  def __init__(self, var: str, bank: int):
    super().__init__()
    self.var = var
    self.bank = bank

  @property
  def name(self) -> str:
    return f'{self.var}_bank_{self.bank}'

  def __str__(self) -> str:
    return f'dram<bank {self.bank} {self.var}>'


class LoadNode(MemoryNode):

  def __repr__(self) -> str:
    return f'\033[33mload {self.var} bank {self.bank}\033[0m'


class StoreNode(MemoryNode):

  def __repr__(self) -> str:
    return f'\033[36mstore {self.var} bank {self.bank}\033[0m'


def is_valid_node(node: ir.Module) -> bool:
  return not isinstance(node, (SuperSourceNode, SuperSinkNode))


def is_valid_edge(edge: Tuple[ir.Module, ir.Module]) -> bool:
  return all(map(is_valid_node, edge))


# pylint: disable=too-many-branches,too-many-statements
def create_dataflow_graph(stencil):
  chronological_tensors = stencil.chronological_tensors
  super_source = SuperSourceNode(
      fwd_nodes={},
      cpt_nodes={},
      super_sink=SuperSinkNode(),
  )

  load_nodes = {
      stmt.name:
      tuple(LoadNode(var=stmt.name, bank=bank) for bank in stmt.dram)
      for stmt in stencil.input_stmts
  }
  store_nodes = {
      stmt.name:
      tuple(StoreNode(var=stmt.name, bank=bank) for bank in stmt.dram)
      for stmt in stencil.output_stmts
  }

  for mem_node in itertools.chain(*load_nodes.values()):
    super_source.add_child(mem_node)
  for mem_node in itertools.chain(*store_nodes.values()):
    mem_node.add_child(super_source.super_sink)

  def color_id(node):
    if isinstance(node, LoadNode):
      return f'\033[33mload {node.var}[bank{node.bank}]\033[0m'
    if isinstance(node, StoreNode):
      return f'\033[36mstore {node.var}[bank{node.bank}]\033[0m'
    if isinstance(node, ForwardNode):
      return f'\033[32mforward {node.tensor.name} @{node.offset}\033[0m'
    if isinstance(node, ComputeNode):
      return f'\033[31mcompute {node.tensor.name} #{node.pe_id}\033[0m'
    return 'unknown node'

  def color_attr(node):
    result = []
    for k, v in node.__dict__.items():
      if (node.__class__, k) in ((SuperSourceNode, 'parents'), (SuperSinkNode,
                                                                'children')):
        continue
      if k in ('parents', 'children'):
        result.append('%s: [%s]' % (k, ', '.join(map(color_id, v))))
      else:
        result.append('%s: %s' % (k, repr(v)))
    return '{%s}' % ', '.join(result)

  def color_print(node):
    return '%s: %s' % (color_id(node), color_attr(node))

  print_node = color_id

  if stencil.replication_factor > 1:
    replicated_next_fifo = stencil.get_replicated_next_fifo()
    replicated_all_points = stencil.get_replicated_all_points()
    replicated_reuse_buffers = stencil.get_replicated_reuse_buffers()

    def add_fwd_nodes(src_name):
      dsts = replicated_all_points[src_name]
      reuse_buffer = replicated_reuse_buffers[src_name][1:]
      nodes_to_add = []
      for dst_point_dicts in dsts.values():
        for offset in dst_point_dicts:
          if (src_name, offset) in super_source.fwd_nodes:
            continue
          fwd_node = ForwardNode(
              tensor=stencil.tensors[src_name],
              offset=offset,
              depth=stencil.get_replicated_reuse_buffer_length(
                  src_name, offset))
          _logger.debug('create %s', print_node(fwd_node))
          init_offsets = [start for start, end in reuse_buffer if start == end]
          if offset in init_offsets:
            if src_name in [stencil.input.name]:
              load_node_count = len(load_nodes[src_name])
              load_nodes[src_name][load_node_count - 1 -
                                   offset % load_node_count].add_child(fwd_node)
            else:
              (super_source.cpt_nodes[(src_name, 0)].add_child(fwd_node))
          super_source.fwd_nodes[(src_name, offset)] = fwd_node
          if offset in replicated_next_fifo[src_name]:
            nodes_to_add.append(
                (fwd_node, (src_name, replicated_next_fifo[src_name][offset])))
      for src_node, key in nodes_to_add:
        src_node.add_child(super_source.fwd_nodes[key])

    add_fwd_nodes(stencil.input.name)

    for stage in stencil.get_stages_chronologically():
      cpt_node = ComputeNode(stage=stage, pe_id=0)
      _logger.debug('create %s', print_node(cpt_node))
      super_source.cpt_nodes[(stage.name, 0)] = cpt_node
      for input_name, input_window in stage.window.items():
        for i in range(len(input_window)):
          offset = next(offset for offset, points in (
              replicated_all_points[input_name][stage.name].items())
                        if points == i)
          fwd_node = super_source.fwd_nodes[(input_name, offset)]
          _logger.debug('  access %s', print_node(fwd_node))
          fwd_node.add_child(cpt_node)
      if stage.is_output():
        super_source.cpt_nodes[stage.name,
                               0].add_child(store_nodes[stage.name][0])
      else:
        add_fwd_nodes(stage.name)

  else:
    next_fifo = stencil.next_fifo
    all_points = stencil.all_points
    reuse_buffers = stencil.reuse_buffers

    def add_fwd_nodes(src_name):
      dsts = all_points[src_name]
      reuse_buffer = reuse_buffers[src_name][1:]
      nodes_to_add = []
      for dst_point_dicts in dsts.values():
        for offset in dst_point_dicts:
          if (src_name, offset) in super_source.fwd_nodes:
            continue
          fwd_node = ForwardNode(tensor=stencil.tensors[src_name],
                                 offset=offset)
          #depth=stencil.get_reuse_buffer_length(src_name, offset))
          _logger.debug('create %s', print_node(fwd_node))
          # init_offsets is the start of each reuse chain
          init_offsets = [
              next(end
                   for start, end in reuse_buffer
                   if start == unroll_idx)
              for unroll_idx in reversed(range(stencil.unroll_factor))
          ]
          _logger.debug('reuse buffer: %s', reuse_buffer)
          _logger.debug('init offsets: %s', init_offsets)
          if offset in init_offsets:
            if src_name in stencil.input_names:
              # fwd from external input
              load_node_count = len(load_nodes[src_name])
              load_nodes[src_name][load_node_count - 1 -
                                   offset % load_node_count].add_child(fwd_node)
            else:
              # fwd from output of last stage
              # tensor name and offset are used to find the cpt node
              cpt_offset = next(
                  unroll_idx for unroll_idx in range(stencil.unroll_factor)
                  if init_offsets[unroll_idx] == offset)
              cpt_node = super_source.cpt_nodes[(src_name, cpt_offset)]
              cpt_node.add_child(fwd_node)
          super_source.fwd_nodes[(src_name, offset)] = fwd_node
          if offset in next_fifo[src_name]:
            nodes_to_add.append(
                (fwd_node, (src_name, next_fifo[src_name][offset])))
      for src_node, key in nodes_to_add:
        # fwd from another fwd node
        src_node.add_child(super_source.fwd_nodes[key])

    for input_name in stencil.input_names:
      add_fwd_nodes(input_name)

    for tensor in chronological_tensors:
      if tensor.is_input():
        continue
      for unroll_index in range(stencil.unroll_factor):
        pe_id = stencil.unroll_factor - 1 - unroll_index
        cpt_node = ComputeNode(tensor=tensor, pe_id=pe_id)
        _logger.debug('create %s', print_node(cpt_node))
        super_source.cpt_nodes[(tensor.name, pe_id)] = cpt_node
        for input_name, input_window in tensor.ld_indices.items():
          for i in range(len(input_window)):
            offset = next(offset for offset, points in all_points[input_name][
                tensor.name].items() if pe_id in points and points[pe_id] == i)
            fwd_node = super_source.fwd_nodes[(input_name, offset)]
            _logger.debug('  access %s', print_node(fwd_node))
            fwd_node.add_child(cpt_node)
      if tensor.is_output():
        for pe_id in range(stencil.unroll_factor):
          super_source.cpt_nodes[tensor.name, pe_id].add_child(
              store_nodes[tensor.name][pe_id % len(store_nodes[tensor.name])])
      else:
        add_fwd_nodes(tensor.name)

  # pylint: disable=too-many-nested-blocks
  for src_node in super_source.tpo_valid_node_gen():
    for dst_node in filter(is_valid_node, src_node.children):
      # 5 possible edge types:
      # 1. load => fwd
      # 2. fwd => fwd
      # 3. fwd => cpt
      # 4. cpt => fwd
      # 5. cpt => store
      if isinstance(src_node, LoadNode):
        write_lat = 0
      elif isinstance(src_node, ForwardNode):
        write_lat = 2
      elif isinstance(src_node, ComputeNode):
        write_lat = src_node.tensor.st_ref.lat
      else:
        raise util.InternalError('unexpected source node: %s' % repr(src_node))

      fifo = ir.FIFO(src_node, dst_node, depth=0, write_lat=write_lat)
      lets: List[ir.Let] = []
      if isinstance(src_node, LoadNode):
        expr = ir.DRAMRef(
            haoda_type=dst_node.tensor.haoda_type,
            dram=(src_node.bank,),
            var=dst_node.tensor.name,
            offset=(stencil.unroll_factor - 1 - dst_node.offset) //
            len(stencil.stmt_table[dst_node.tensor.name].dram),
        )
      elif isinstance(src_node, ForwardNode):
        if isinstance(dst_node, ComputeNode):
          dst = src_node.tensor.children[dst_node.tensor.name]
          src_name = src_node.tensor.name
          unroll_idx = dst_node.pe_id
          point = all_points[src_name][dst.name][src_node.offset][unroll_idx]
          idx = list(dst.ld_indices[src_name].values())[point].idx
          _logger.debug('%s%s referenced by <%s> @ unroll_idx=%d is %s',
                        src_name, util.idx2str(idx), dst.name, unroll_idx,
                        print_node(src_node))
          dst_node.fifo_map[src_name][idx] = fifo
        delay = stencil.reuse_buffer_lengths[src_node.tensor.name]\
                                            [src_node.offset]
        offset = src_node.offset - delay
        for parent in src_node.parents:  # fwd node has only 1 parent
          for fifo_r in parent.fifos:
            if fifo_r.edge == (parent, src_node):
              break
        if delay > 0:
          # TODO: build an index somewhere
          for let in src_node.lets:
            # pylint: disable=undefined-loop-variable
            if isinstance(let.expr, ir.DelayedRef) and let.expr.ref == fifo_r:
              var_name = let.name
              var_type = let.haoda_type
              break
          else:
            var_name = 'let_%d' % len(src_node.lets)
            # pylint: disable=undefined-loop-variable
            var_type = fifo_r.haoda_type
            lets.append(
                ir.Let(haoda_type=var_type,
                       name=var_name,
                       expr=ir.DelayedRef(delay=delay, ref=fifo_r)))
          expr = ir.Var(name=var_name, idx=[])
          expr.haoda_type = var_type
        else:
          expr = fifo_r  # pylint: disable=undefined-loop-variable
      elif isinstance(src_node, ComputeNode):

        def replace_refs_callback(obj, args):
          if isinstance(obj, ir.Ref):
            _logger.debug(
                'replace %s with %s',
                obj,
                # pylint: disable=cell-var-from-loop,undefined-loop-variable
                src_node.fifo_map[obj.name][obj.idx])
            # pylint: disable=cell-var-from-loop,undefined-loop-variable
            return src_node.fifo_map[obj.name][obj.idx]
          return obj

        _logger.debug('lets: %s', src_node.tensor.lets)
        lets = [_.visit(replace_refs_callback) for _ in src_node.tensor.lets]
        _logger.debug('replaced lets: %s', lets)
        _logger.debug('expr: %s', src_node.tensor.expr)
        expr = src_node.tensor.expr.visit(replace_refs_callback)
        _logger.debug('replaced expr: %s', expr)
        if isinstance(dst_node, StoreNode):
          dram_ref = ir.DRAMRef(
              haoda_type=src_node.tensor.haoda_type,
              dram=(dst_node.bank,),
              var=src_node.tensor.name,
              offset=(src_node.pe_id) //
              len(stencil.stmt_table[src_node.tensor.name].dram),
          )
          dst_node.lets.append(ir.Let(haoda_type=None, name=dram_ref,
                                      expr=fifo))
      else:
        raise util.InternalError('unexpected node of type %s' % type(src_node))

      src_node.exprs[fifo] = expr
      src_node.lets.extend(_ for _ in lets if _ not in src_node.lets)
      _logger.debug('fifo [%d]: %s%s => %s', fifo.depth, color_id(src_node),
                    '' if fifo.write_lat is None else ' ~%d' % fifo.write_lat,
                    color_id(dst_node))

  super_source.update_module_depths({})

  return super_source
