from collections import deque
import logging

_logger = logging.getLogger('__main__').getChild(__name__)

class _Node(object):
    """A node in the dataflow graph, used to calculate the FIFO depth.

    After the reuse chains are generated, the topology of the dataflow graph
    is then determined. However, the cycle-accurate FIFO depth is yet unknown.
    To calculate the FIFO depth, the all end-to-end paths must be checked to
    match the latency. This class is used to represent a module node in the
    dataflow graph and calculate the latency.

    Attributes:
        parents: List of parent nodes.
        children: List of child nodes.
    """
    def __init__(self):
        self.parents = []
        self.children = []

    def __str__(self):
        return '%s @ %x: %s' % (self.__class__.__name__, id(self),
            self.__dict__)

    def __repr__(self):
        return '%s @ %x' % (self.__class__.__name__, id(self))

    def add_child(self, child):
        self.children.append(child)
        child.parents.append(self)

    def bfs_node_generator(self):
        node_queue = deque([self])
        seen_nodes = {self}
        while node_queue:
            node = node_queue.popleft()
            yield node
            for child in node.children:
                if child not in seen_nodes:
                    node_queue.append(child)
                    seen_nodes.add(child)

    def dfs_node_generator(self):
        node_stack = [self]
        seen_nodes = {self}
        while node_stack:
            node = node_stack.pop()
            yield node
            for child in node.children:
                if child not in seen_nodes:
                    node_stack.append(child)
                    seen_nodes.add(child)

    def tpo_node_generator(self):
        nodes = {}
        for node in self.bfs_node_generator():
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

    def bfs_edge_generator(self):
        node_queue = deque([self])
        seen_nodes = {self}
        while node_queue:
            node = node_queue.popleft()
            for child in node.children:
                yield node, child
                if child not in seen_nodes:
                    node_queue.append(child)
                    seen_nodes.add(child)

    def dfs_edge_generator(self):
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
        return {self}.union(*map(_Node.get_descendants, self.children))

    def get_connections(self):
        return ({(self, child) for child in self.children}
                .union(*map(_Node.get_connections, self.children)))

class SuperSourceNode(_Node):
    """A node representing the super source in the dataflow graph.

    A super source doesn't have parent nodes.

    Attributes:
        fwd_nodes: {(tensor_name, offset): node}
        cpt_nodes: {(stage_name, pe_id): node}
        _paths: {node: [(src, ... dst), ... ]}
        _extra_depths: {(src_node, dst_node): extra_depth)}
    """
    def get_latency(self):
        return 0

    def find_paths(self, node):
        if not hasattr(self, 'paths'):
            self._paths = {self: [(self,)]}
            for src_node, dst_node in self.dfs_edge_generator():
                self._paths.setdefault(dst_node, []).extend(
                    path+(dst_node,) for path in self._paths[src_node])
        return self._paths[node]

    def get_extra_depth(self, edge):
        if not hasattr(self, '_extra_depths'):
            self._extra_depths = {}
            node_heights = {}
            for node in self.tpo_node_generator():
                node_heights[node] = max(
                    (node_heights[p] + p.get_latency() for p in node.parents),
                    default=0)
                for parent in node.parents:
                    extra_depth = node_heights[node] - (
                        node_heights[parent] + parent.get_latency())
                    if extra_depth > 0:
                        self._extra_depths[(parent, node)] = extra_depth
                        _logger.debug('\033[31moops\033[0m, need to add %d to '
                            '%s' % (extra_depth, (parent, node)))
        return self._extra_depths.get(edge, 0)

class SuperSinkNode(_Node):
    """A node representing the super sink in the dataflow graph.

    A super sink doesn't have child nodes.
    """
    pass

class ForwardNode(_Node):
    """A node representing a forward module in the dataflow graph.

    Attributes:
        tensor: Tensor corresponding to this node.
        offset: Int representing the offset of this tensor.
        depth: Int representing the FIFO depth.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.tensor = kwargs.pop('tensor')
        self.offset = kwargs.pop('offset')
        self.depth = kwargs.pop('depth')

    def __repr__(self):
        return '\033[32mforward %s @%d\033[0m' % (self.tensor.name, self.offset)

    def get_latency(self):
        return 2

class ComputeNode(_Node):
    """A node representing a compute module in the dataflow graph.

    Attributes:
        stage: Stage corresponding to this node.
        pe_id: Int representing the PE id.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.stage = kwargs.pop('stage')
        self.pe_id = kwargs.pop('pe_id')

    def __repr__(self):
        return '\033[31mcompute %s #%d\033[0m' % (self.stage.name, self.pe_id)

    def get_latency(self):
        return self.stage.expr[0].depth or 0

def create_dataflow_graph(stencil):
    super_source = SuperSourceNode()
    super_sink = SuperSinkNode()

    super_source.fwd_nodes = {}  # {(tensor_name, offset): node}
    super_source.cpt_nodes = {}  # {(stage_name, pe_id): node}

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
                    _logger.debug('create %s' % repr(fwd_node))
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
            _logger.debug('create %s' % repr(cpt_node))
            super_source.cpt_nodes[(stage.name, 0)] = cpt_node
            for input_name, input_window in stage.window.items():
                for i in range(len(input_window)):
                    offset = next(offset for offset, points in
                        (replicated_all_points[input_name][stage.name]
                            .items())
                        if points == i)
                    fwd_node = super_source.fwd_nodes[(input_name, offset)]
                    _logger.debug('  access %s' % repr(fwd_node))
                    fwd_node.add_child(cpt_node)
            if stage.is_output():
                super_source.cpt_nodes[stage.name, 0].add_child(super_sink)
            else:
                add_fwd_nodes(stage.name)

    else:
        next_fifo = stencil.get_next_fifo()
        all_points = stencil.get_all_points()
        reuse_buffers = stencil.get_reuse_buffers()

        def add_fwd_nodes(src_name):
            dsts = all_points[src_name]
            reuse_buffer = reuse_buffers[src_name][1:]
            nodes_to_add = []
            for dst_point_dicts in dsts.values():
                for offset in dst_point_dicts:
                    if (src_name, offset) in super_source.fwd_nodes:
                        continue
                    fwd_node = ForwardNode(
                        tensor=stencil.tensors[src_name],
                        offset=offset,
                        depth=stencil.get_reuse_buffer_length(src_name, offset))
                    _logger.debug('create %s' % repr(fwd_node))
                    init_offsets = [start
                        for start, end in reuse_buffer if start == end]
                    if offset in init_offsets:
                        if src_name in [stencil.input.name]:
                            super_source.add_child(fwd_node)
                        else:
                            (super_source.cpt_nodes[(src_name, next(i
                                for i in range(stencil.unroll_factor)
                                if init_offsets[i] == offset))]
                                .add_child(fwd_node))
                    super_source.fwd_nodes[(src_name, offset)] = fwd_node
                    if offset in next_fifo[src_name]:
                        nodes_to_add.append(
                            (fwd_node, (src_name, next_fifo[src_name][offset])))
            for src_node, key in nodes_to_add:
                src_node.add_child(super_source.fwd_nodes[key])

        add_fwd_nodes(stencil.input.name)

        for stage in stencil.get_stages_chronologically():
            for unroll_index in range(stencil.unroll_factor):
                pe_id = stencil.unroll_factor-1-unroll_index
                cpt_node = ComputeNode(stage=stage, pe_id=pe_id)
                _logger.debug('create %s' % repr(cpt_node))
                super_source.cpt_nodes[(stage.name, pe_id)] = cpt_node
                for input_name, input_window in stage.window.items():
                    for i in range(len(input_window)):
                        offset = next(offset for offset, points in
                            all_points[input_name][stage.name].items()
                            if pe_id in points and points[pe_id] == i)
                        fwd_node = super_source.fwd_nodes[(input_name, offset)]
                        _logger.debug('  access %s' % repr(fwd_node))
                        fwd_node.add_child(cpt_node)
            if stage.is_output():
                for pe_id in range(stencil.unroll_factor):
                    (super_source.cpt_nodes[stage.name, pe_id]
                        .add_child(super_sink))
            else:
                add_fwd_nodes(stage.name)

    def color_id(node):
        if node.__class__ is (_Node):
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
                    (node.stage.name, node.pe_id))
        return 'unknown node'

    def color_attr(node):
        result = []
        for k, v in node.__dict__.items():
            if (node.__class__, k) in ((SuperSourceNode, 'parents'),
                                       (SuperSinkNode, 'children')):
                continue
            if k in ('parents', 'children'):
                result.append('%s: [%s]' %
                    (k, ', '.join(map(color_id, v))))
            else:
                result.append('%s: %s' % (k, repr(v)))
        return '{%s}' % ', '.join(result)

    def color_print(node):
        return '%s: %s' % (color_id(node), color_attr(node))

    for src_node, dst_node in super_source.bfs_edge_generator():
        _logger.debug('%s -> %s' %
            (color_id(src_node), color_id(dst_node)))
    for node in super_source.bfs_node_generator():
        if node.__class__ is _Node:
            _logger.error('private object _Node(%s) '
                          'shall not be found here' % node)
        else:
            _logger.debug(color_print(node))
    return super_source
