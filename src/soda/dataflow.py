from collections import OrderedDict
from collections import deque
import copy
import logging

from haoda import ir
from soda import core
from soda import grammar

_logger = logging.getLogger('__main__').getChild(__name__)

class Delay(grammar._Node):
  def __str__(self):
    return '%s delayed %d' % (self.ref, self.delay)

class SuperSourceNode(ir.Node):
  """A node representing the super source in the dataflow graph.

  A super source doesn't have parent nodes.

  Attributes:
    fwd_nodes: {(tensor_name, offset): node}
    cpt_nodes: {(stage_name, pe_id): node}
    _paths: {node: [(src, ... dst), ... ]}
    _extra_depths: {(src_node, dst_node): extra_depth)}
  """

  def find_paths(self, node):
    if not hasattr(self, '_paths'):
      self._paths = {self: [(self,)]}
      for src_node, dst_node in self.dfs_edge_generator():
        self._paths.setdefault(dst_node, []).extend(
          path+(dst_node,) for path in self._paths[src_node])
    return self._paths[node]

  # TODO: make this general and move it to haoda.ir
  def get_extra_depth(self, edge):
    if not hasattr(self, '_extra_depths'):
      self._extra_depths = OrderedDict()
      node_heights = OrderedDict()
      for node in self.tpo_node_gen():
        node_heights[node] = max(
          (node_heights[parent] + parent.get_latency(node) for parent in node.parents), default=0)
        for parent in node.parents:
          extra_depth = node_heights[node] - (
            node_heights[parent] + parent.get_latency(node))
          if extra_depth > 0:
            self._extra_depths[(parent, node)] = extra_depth
            _logger.debug('\033[31moops\033[0m, need to add %d to %s',
                          extra_depth, (parent, node))
    return self._extra_depths.get(edge, 0)

class SuperSinkNode(ir.Node):
  """A node representing the super sink in the dataflow graph.

  A super sink doesn't have child nodes.
  """
  pass

class ForwardNode(ir.Node):
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

class ComputeNode(ir.Node):
  """A node representing a compute module in the dataflow graph.

  Attributes:
    tensor: Tensor corresponding to this node.
    pe_id: Int representing the PE id.
  """
  def __init__(self, **kwargs):
    super().__init__()
    self.tensor = kwargs.pop('tensor')
    self.pe_id = kwargs.pop('pe_id')

  def __repr__(self):
    return '\033[31mcompute %s #%d\033[0m' % (self.tensor.name, self.pe_id)

def create_dataflow_graph(stencil):
  super_source = SuperSourceNode()
  super_sink = SuperSinkNode()

  super_source.fwd_nodes = OrderedDict()  # {(tensor_name, offset): node}
  super_source.cpt_nodes = OrderedDict()  # {(stage_name, pe_id): node}

  def color_id(node):
    if node.__class__ is (ir.Node):
      return repr(node)
    elif node.__class__ is SuperSourceNode:
      return '\033[33msuper source\033[0m'
    elif node.__class__ is SuperSinkNode:
      return '\033[36msuper sink\033[0m'
    elif node.__class__ is ForwardNode:
      return ('\033[32mforward %s @%d\033[0m' %
          (node.tensor.name, node.offset))
    elif node.__class__ is ComputeNode:
      return ('\033[31mcompute %s #%d\033[0m' %
          (node.tensor.name, node.pe_id))
    return 'unknown node'

  def color_attr(node):
    result = []
    for k, v in node.__dict__.items():
      if (node.__class__, k) in ((SuperSourceNode, 'parents'),
                                 (SuperSinkNode, 'children')):
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
          init_offsets = [start
            for start, end in reuse_buffer if start == end]
          if offset in init_offsets:
            if src_name in [stencil.input.name]:
              super_source.add_child(fwd_node)
            else:
              (super_source.cpt_nodes[(src_name, 0)]
                .add_child(fwd_node))
          super_source.fwd_nodes[(src_name, offset)] = fwd_node
          if offset in replicated_next_fifo[src_name]:
            nodes_to_add.append(
              (fwd_node, (src_name,
                replicated_next_fifo[src_name][offset])))
      for src_node, key in nodes_to_add:
        src_node.add_child(super_source.fwd_nodes[key])

    add_fwd_nodes(stencil.input.name)

    for stage in stencil.get_stages_chronologically():
      cpt_node = ComputeNode(stage=stage, pe_id=0)
      _logger.debug('create %s', print_node(cpt_node))
      super_source.cpt_nodes[(stage.name, 0)] = cpt_node
      for input_name, input_window in stage.window.items():
        for i in range(len(input_window)):
          offset = next(offset for offset, points in
            (replicated_all_points[input_name][stage.name]
              .items())
            if points == i)
          fwd_node = super_source.fwd_nodes[(input_name, offset)]
          _logger.debug('  access %s', print_node(fwd_node))
          fwd_node.add_child(cpt_node)
      if stage.is_output():
        super_source.cpt_nodes[stage.name, 0].add_child(super_sink)
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
          fwd_node = ForwardNode(
            tensor=stencil.tensors[src_name], offset=offset)
          #depth=stencil.get_reuse_buffer_length(src_name, offset))
          _logger.debug('create %s', print_node(fwd_node))
          # init_offsets is the start of each reuse chain
          init_offsets = [start for start, end in reuse_buffer if start == end]
          if offset in init_offsets:
            if src_name in stencil.input_names:
              # fwd from external input
              super_source.add_child(fwd_node)
            else:
              # fwd from output of last stage
              # tensor name and offset are used to find the cpt node
              cpt_offset = next(unroll_idx
                for unroll_idx in range(stencil.unroll_factor)
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

    for tensor in stencil.chronological_tensors:
      if tensor.is_input():
        continue
      for unroll_index in range(stencil.unroll_factor):
        pe_id = stencil.unroll_factor-1-unroll_index
        cpt_node = ComputeNode(tensor=tensor, pe_id=pe_id)
        _logger.debug('create %s', print_node(cpt_node))
        super_source.cpt_nodes[(tensor.name, pe_id)] = cpt_node
        for input_name, input_window in tensor.ld_indices.items():
          for i in range(len(input_window)):
            offset = next(offset for offset, points in
              all_points[input_name][tensor.name].items()
              if pe_id in points and points[pe_id] == i)
            fwd_node = super_source.fwd_nodes[(input_name, offset)]
            _logger.debug('  access %s', print_node(fwd_node))
            fwd_node.add_child(cpt_node)
      if tensor.is_output():
        for pe_id in range(stencil.unroll_factor):
          super_source.cpt_nodes[tensor.name, pe_id].add_child(super_sink)
      else:
        add_fwd_nodes(tensor.name)

  for src_node in super_source.tpo_node_gen():
    for dst_node in src_node.children:
      # 5 possible edge types:
      # 1. src => fwd
      # 2. fwd => fwd
      # 3. fwd => cpt
      # 4. cpt => fwd
      # 5. cpt => sink
      if isinstance(src_node, SuperSourceNode):
        write_lat = 0
      elif isinstance(src_node, ForwardNode):
        write_lat = 2
      elif isinstance(src_node, ComputeNode):
        write_lat = src_node.tensor.st_ref.lat
      else:
        raise core.InternalError('unexpected source node: %s', repr(src_node))

      if isinstance(dst_node, ForwardNode):
        depth = stencil.reuse_buffer_lengths[dst_node.tensor.name]\
                                            [dst_node.offset]
      elif isinstance(dst_node, ComputeNode):
        depth = 0
      elif isinstance(dst_node, SuperSinkNode):
        depth = 0
      else:
        raise core.InternalError('unexpected destination node: %s',
                                 repr(dst_node))

      fifo = ir.FIFO(src_node, dst_node, depth, write_lat)
      if isinstance(src_node, SuperSourceNode):
        lets = []
        expr = None
      elif isinstance(src_node, ForwardNode):
        lets = []
        delay = stencil.reuse_buffer_lengths[src_node.tensor.name]\
                                            [src_node.offset]
        offset = src_node.offset - delay
        for parent in src_node.parents: # fwd node has only 1 parent
          for fifo_r in parent.fifos:
            if fifo_r.edge == (parent, src_node):
              break
        expr = Delay(delay=delay, ref=fifo_r)
      elif isinstance(src_node, ComputeNode):
        def replace_refs_callback(obj, args):
          obj = copy.copy(obj)
          if hasattr(obj, 'ref') and obj.ref:
            #if isinstance(obj, grammar.Ref):
            # to confirm -- is this right?
            # TODO: build an index somewhere
            offset = stencil.unroll_factor - 1 - src_node.pe_id + \
                     core.serialize(obj.ref.idx, stencil.tile_size)
            for parent in src_node.parents:
              if parent.tensor.name == obj.ref.name and parent.offset == offset:
                for fifo_r in parent.fifos:
                  if fifo_r.edge == (parent, src_node):
                    break
            _logger.debug('replace %s with %s', obj.ref, fifo_r)
            obj.ref = fifo_r
          return obj
        lets = [_.visit(replace_refs_callback) for _ in src_node.tensor.lets]
        expr = src_node.tensor.expr.visit(replace_refs_callback)

      src_node.exprs[fifo] = expr
      src_node.lets |= set(lets)
      _logger.debug('fifo [%d]: %s%s => %s', fifo.depth, color_id(src_node),
                    '' if fifo.write_lat is None else ' ~%d' % fifo.write_lat,
                    color_id(dst_node))

  for src_node in super_source.tpo_node_gen():
    for dst_node in src_node.children:
      src_node.fifo(dst_node).depth += super_source.get_extra_depth((src_node, dst_node))

  for src_node, dst_node in super_source.bfs_edge_gen():
    fifo = src_node.fifo(dst_node)
    _logger.debug('fifo [%d]: %s%s => %s', fifo.depth, color_id(src_node),
                  '' if fifo.write_lat is None else ' ~%d' % fifo.write_lat,
                  color_id(dst_node))
  for node in super_source.bfs_node_gen():
    if node.__class__ is ir.Node:
      _logger.error('private object ir.Node(%s) shall not be found here', node)
    else:
      for fifo, expr in node.exprs.items():
        _logger.debug('%s.exprs: %s', color_id(node), expr)
  return super_source
