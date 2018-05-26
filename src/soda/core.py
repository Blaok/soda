from collections import deque
from functools import reduce
import copy
import logging
import operator
import sys

from soda import dataflow
from soda import grammar

# constants
COORDS_TILED = 'xyzw'
COORDS_IN_TILE = 'ijkl'
COORDS_IN_ORIG = 'pqrs'
TYPE_WIDTH = {
    'uint8_t':8,
    'uint16_t':16,
    'uint32_t':32,
    'uint64_t':64,
    'int8_t':8,
    'int16_t':16,
    'int32_t':32,
    'int64_t':64,
    'float':32,
    'double':64
}
MAX_DRAM_BANK = 4

_logger = logging.getLogger('__main__').getChild(__name__)

class InternalError(Exception):
    pass

class SemanticError(Exception):
    pass

class SemanticWarn(Exception):
    pass

# Tensor.name: str
# Tensor.type: str
# Tensor.chan: int
# Tensor.idx: (int, ...)
# Tensor.parent: Stage
# Tensor.children: {Stage, ...}
# Tensor.offset: int
# Tensor.border: ('preserve', {Stage, ...})
class Tensor(object):
    def __init__(self, node):
        self.name = node.name
        self.type = node.type
        self.chan = node.chan
        if isinstance(node, grammar.Output):
            self.idx = next(iter(node.expr)).idx
            for e in node.expr:
                if e.idx != self.idx:
                    raise InternalError('Normalization went wrong')
        elif isinstance(node, grammar.Local):
            self.idx = next(iter(node.expr)).idx
            for e in node.expr:
                if e.idx != self.idx:
                    raise InternalError('Normalization went wrong')
        else:
            self.idx = None
        self.parent = None
        self.children = set()
        self.offset = 0
        self.border = None

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s = %s' % (k, v) for k, v in self.__dict__.items()))

    def preserve_border_to(self):
        if self.border is not None and self.border[0] == 'preserve':
            return self.border[1]
        return None

    def preserve_border_from(self):
        return self.parent is not None and self.parent.preserve_border_from()

    def is_output(self):
        return len(self.children) == 0

    def is_input(self):
        return self.parent is None or self.parent.is_input()

# Stage.window: {str: [(int, ...), ...], ...}
# Stage.offset: {str: [int, ...], ...}
# Stage.delay: {str: int, ...}
# Stage.expr: [StageExpr, ...]
# Stage.inputs: {str: Tensor, ...}
# Stage.output: Tensor
# Stage.border: ('preserve', Tensor)
class Stage(object):
    def __init__(self, **kwargs):
        self.window = kwargs.pop('window')
        self.offset = kwargs.pop('offset')
        self.delay = kwargs.pop('delay', {})
        self.expr = kwargs.pop('expr')
        self.inputs = kwargs.pop('inputs')
        self.output = kwargs.pop('output')
        self.border = None

        # shortcuts
        self.name = self.output.name
        self.idx = self.output.idx

    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s = %s' % (k, v) for k, v in self.__dict__.items()))

    def preserve_border_from(self):
        if self.border is not None and self.border[0] == 'preserve':
            return self.border[1]
        return None

    def preserve_border_to(self):
        return self.output.preserve_border_from()

    def is_output(self):
        return self.output.is_output()

    def is_input(self):
        return len(self.inputs) == 0

class Stencil(object):
    def __init__(self, **kwargs):
        self.iterate = kwargs.pop('iterate')
        if self.iterate < 1:
            raise SemanticError('cannot iterate %d times' % self.iterate)
        self.border = kwargs.pop('border')
        self.preserve_border = self.border == 'preserve'
        self.cluster = kwargs.pop('cluster')
        # platform determined
        self.burst_width = kwargs.pop('burst_width')
        self.dram_bank = kwargs.pop('dram_bank')
        # application determined
        self.app_name = kwargs.pop('app_name')
        # parameters can be explored
        self.tile_size = kwargs.pop('tile_size')
        self.unroll_factor = kwargs.pop('unroll_factor')
        self.replication_factor = kwargs.pop('replication_factor')
        self.dram_separate = kwargs.pop('dram_separate')
        if self.dram_separate:
            if self.dram_bank%2 != 0:
                _logger.fatal('Number of DRAM banks has to be even when '
                              'separated')
                sys.exit(-1)
            else:
                self.dram_bank = int(self.dram_bank/2)
        # stage-independent
        self.dim = kwargs.pop('dim')
        self.extra_params = kwargs.pop('extra_params')
        # stage-specific
        input_node = kwargs.pop('input')
        output_node = kwargs.pop('output')

        if self.iterate > 1:
            if input_node.type != output_node.type:
                raise SemanticError(
                    'input must have the same type as output if iterate > 1 '
                    'times, current input has type %s but output has type %s' %
                    (input_node.type, output_node.type))
            if input_node.chan != output_node.chan:
                raise SemanticError(
                    'input must have the same number of channel as output if '
                    'iterate > 1 times, current input has %d chanels but '
                    'output has %d channels' % (input_node.chan,
                                                output_node.chan))
            _logger.debug('pipeline %d iterations of %s -> %s' %
                          (self.iterate, input_node.name, output_node.name))

        local_nodes = kwargs.pop('locals')
        local_names = set(x.name for x in local_nodes)

        new_local_nodes = []
        preserved_borders = {}  # {to: {from, ...}, ...}
        NameFromIter = lambda n, i: n+'_iter%d' % i if i > 0 else n
        def LocalLoadCallBack(n):
            if n == input_node.name:
                return NameFromIter(n, iteration)
            if n in local_names:
                return NameFromIter(n, iteration) if n in local_names else n
            return n
        def OutputLoadCallBack(n):
            if n == input_node.name or n in local_names:
                return NameFromIter(n, iteration-1)
            return n
        for iteration in range(1, self.iterate):
            new_local = grammar.Local(output_node=output_node)
            new_local.mutate_load(OutputLoadCallBack)
            new_local.mutate_store(
                lambda n: NameFromIter(input_node.name, iteration))
            if self.preserve_border:
                border_from = NameFromIter(input_node.name, iteration-1)
                new_local.preserve_border_from(border_from)
                preserved_borders.setdefault(
                    new_local.name, set()).add(border_from)
            new_local_nodes.append(new_local)
            for local in local_nodes:
                new_local = copy.deepcopy(local)
                new_local.mutate_load(LocalLoadCallBack)
                new_local.mutate_store(lambda n: NameFromIter(n, iteration))
                new_local_nodes.append(new_local)
        if self.preserve_border:
            border_from = NameFromIter(input_node.name, self.iterate-1)
            output_node.preserve_border_from(border_from)
            preserved_borders.setdefault(
                output_node.name, set()).add(border_from)
        output_node.mutate_load(lambda n: NameFromIter(n, self.iterate-1))
        local_nodes += new_local_nodes

        _logger.debug(input_node)
        for local in local_nodes:
            _logger.debug('Local(')
            _logger.debug('  name: %s,' % local.name)
            _logger.debug('  type: %s,' % local.type)
            _logger.debug('  chan: %s,' % local.chan)
            _logger.debug('  border: %s,' % local.border)
            for e in local.expr:
                _logger.debug('  expr  : [')
                _logger.debug(
                    '    %s%s' % (e, '])' if e is local.expr[-1] else ''))
        _logger.debug('Output(')
        _logger.debug('   name: %s,' % output_node.name)
        _logger.debug('   type: %s,' % output_node.type)
        _logger.debug('   chan: %s,' % output_node.chan)
        _logger.debug('   border: %s,' % output_node.border)
        for e in output_node.expr:
            _logger.debug('  expr  : [')
            _logger.debug(
                '    %s%s' % (e, '])' if e is output_node.expr[-1] else ','))

        self.tensors = {i.name: Tensor(i) for i in local_nodes}
        if input_node.name in local_names:
            raise SemanticError('input name conflict with tensor: %s' %
                                input_node.name)
        else:
            self.input = Tensor(input_node)
            self.tensors[self.input.name] = self.input
        if output_node.name in local_names:
            raise SemanticError('output name conflict with tensor: %s' %
                                output_node.name)
        else:
            self.output = Tensor(output_node)
            self.tensors[self.output.name] = self.output

        self.stages = {}
        for local in local_nodes:
            child_tensor = self.tensors[local.name]
            parent_tensors = self._get_parent_tensors_for(local)
            window = self._get_window_for(local)
            this_stage = Stage(window=window,
                offset={n: serialize_iter(w, self.tile_size)
                        for n, w in window.items()},
                delay={},
                expr=self._get_expr_for(local),
                inputs=parent_tensors,
                output=child_tensor,
                border=local.border)
            self.stages[local.name] = this_stage
            child_tensor.parent = this_stage
            for tensor in parent_tensors.values():
                tensor.children.add(this_stage)

        parent_tensors = self._get_parent_tensors_for(output_node)
        window = self._get_window_for(output_node)
        output_stage = Stage(window=window,
            offset={n: serialize_iter(w, self.tile_size)
                    for n, w in window.items()},
            delay={},
            expr=self._get_expr_for(output_node),
            inputs=parent_tensors,
            output=self.output,
            border=output_node.border)
        self.stages[output_node.name] = output_stage
        self.output.parent = output_stage
        for tensor in parent_tensors.values():
            tensor.children.add(output_stage)

        # let tensor/stage remember which stages/tensors to send/recv as borders
        for dst, srcs in preserved_borders.items():
            for src in srcs:
                _logger.debug('border from %s to %s' % (src, dst))
                if self.tensors[src].border is None:
                    self.tensors[src].border = ('preserve', set())
                self.tensors[src].border[1].add(self.stages[dst])
                if self.stages[dst].border is None:
                    self.stages[dst].border = ('preserve', self.tensors[src])
                else:
                    if self.stages[dst].border[1] != self.tensors[src]:
                        raise InternalError("Border isn't self-consistent")

        # now that we have global knowledge of the tensors we can calculate the
        # offsets of tensors
        _logger.info('calculate tensor offsets')
        processing_queue = deque([self.input.name])
        processed_tensors = {self.input.name}
        self.chronological_tensors = [self.input]
        _logger.debug('tensor %s is at offset %d' %
                      (self.input.name, self.input.offset))
        while processing_queue:
            b = self.tensors[processing_queue.popleft()]
            _logger.debug('inspecting tensor %s\'s children' % b.name)
            for s in b.children:
                if ({x.name for x in s.inputs.values()} <= processed_tensors
                    and s.name not in processed_tensors):
                    # good, all inputs are processed
                    # can determine offset of current tensor
                    _logger.debug(
                        'input%s for tensor %s (i.e. %s) %s processed' % (
                            '' if len(s.inputs) == 1 else 's',
                            s.name,
                            ', '.join([x.name for x in s.inputs.values()]),
                            'is' if len(s.inputs) == 1 else 'are'))
                    stage_offset = serialize(s.expr[0].idx, self.tile_size)

                    # synchronization check
                    def sync(stage, offset):
                        if stage is None:
                            return offset
                        stage_offset = serialize(stage.expr[0].idx,
                                                 self.tile_size)
                        loads = {}
                        for e in stage.expr:
                            for l in e.loads:
                                loads.setdefault(l.name, []).append(l.idx)
                        for n in loads:
                            loads[n] = serialize_iter(loads[n],
                                                          self.tile_size)
                        for l in loads.values():
                            l[0], l[-1] = (stage_offset - max(l),
                                           stage_offset - min(l))
                            del l[1:-1]
                        _logger.debug(
                            'loads in tensor %s: %s' % (stage.name, loads))
                        for tensor in stage.inputs.values():
                            stage_distance = stage.offset[tensor.name][-1]
                            _logger.debug(
                                'want to access tensor %s at offset [%d, %d] '
                                'to generate tensor %s at offset %d' % (
                                    tensor.name, offset+loads[tensor.name][0],
                                    offset+loads[tensor.name][-1], stage.name,
                                    offset))
                            tensor_offset = (tensor.offset +
                                             stage_distance -
                                             stage_offset)
                            if offset < tensor_offset:
                                _logger.debug(
                                    'but tensor %s won\'t be available until '
                                    'offset %d' % (stage.name, tensor_offset))
                                offset = tensor_offset
                                _logger.debug(
                                    'need to access tensor %s at offset '
                                    '[%d, %d] to generate tensor %s at offset '
                                    '%d' % (tensor.name,
                                            offset+loads[tensor.name][0],
                                            offset+loads[tensor.name][-1],
                                            stage.name, offset))
                        return offset
                    _logger.debug('intend to generate tensor %s at offset %d'
                            % (s.name, s.output.offset))
                    s.output.offset = sync(s, s.output.offset)
                    _logger.debug('decide to generate tensor %s at offset %d'
                            % (s.name, s.output.offset))

                    # add delay
                    for x in s.inputs.values():
                        delay = s.output.offset - (x.offset +
                                                   s.offset[x.name][-1] -
                                                   stage_offset)
                        if delay > 0:
                            _logger.debug(
                                'tensor %s arrives at tensor %s '
                                'at offset %d < %d; add %d delay' % (
                                    x.name, s.name,
                                    x.offset+s.offset[x.name][-1]-stage_offset,
                                    s.output.offset, delay))
                        else:
                            _logger.debug(
                                'tensor %s arrives at tensor %s '
                                'at offset %d = %d; good' % (
                                    x.name, s.name,
                                    x.offset+s.offset[x.name][-1]-stage_offset,
                                    s.output.offset))
                        s.delay[x.name] = max(delay, 0)
                        _logger.debug('set delay of %s <- %s to %d' %
                            (s.name, x.name, s.delay[x.name]))

                    processing_queue.append(s.name)
                    processed_tensors.add(s.name)
                    self.chronological_tensors.append(s.output)
                else:
                    for bb in s.inputs.values():
                        if bb.name not in processed_tensors:
                            _logger.debug(
                                'tensor %s requires tensor %s as an input' %
                                (s.name, bb.name))
                            _logger.debug(
                                'but tensor %s isn\'t processed yet' % bb.name)
                            _logger.debug(
                                'add %s to scheduling queue' % bb.name)
                            processing_queue.append(bb.name)

        # setup preserving borders, has to be done here because overall window
        # cannot be generated before dependency graph is created
        for node in local_nodes+[output_node]:
            if hasattr(node, 'preserve_border'):
                # preserve border from node.preserve_border to node.name
                windows = self.stages[node.name].window
                windows.setdefault(
                    node.preserve_border,
                    list(set(windows.get(node.preserve_border, set()))|
                         {next(iter(node.expr)).idx}))
                stencil_window = get_overall_stencil_window(
                    self.tensors[node.preserve_border],
                    self.tensors[node.name])
                self.stages[node.name].delay.setdefault(
                    node.preserve_border,
                    get_stencil_distance(stencil_window, self.tile_size)-
                        serialize(get_stencil_window_offset(stencil_window),
                                  self.tile_size))
                _logger.debug(
                    'window for %s@%s is %s' %
                    (node.name,
                     ', '.join(map(str, node.expr[0].idx)), windows))
                self.stages[node.name].inputs.setdefault(
                    node.preserve_border, self.tensors[node.preserve_border])
                self.tensors[node.preserve_border].children.add(
                    self.stages[node.name])

        _logger.debug('tensors: '+str(list(self.tensors.keys())))
        def LoadPrinter(node):
            if node.name in self.extra_params:
                return '%s(%s)' % (node.name,
                                   ', '.join(map(str, node.idx)))
            return '%s[%d](%s)' % (node.name, node.chan,
                                   ', '.join(map(str, node.idx)))
        def StorePrinter(node):
            return '%s[%d](%s)' % (node.name, node.chan,
                                   ', '.join(map(str, node.idx)))

        for s in self.stages.values():
            _logger.debug(
                'stage: %s@(%s) <- [%s]' %
                (s.name, ', '.join(map(str, s.idx)),
                 ', '.join('%s@%s' % (x.name, list(set(s.window[x.name])))
                                      for x in s.inputs.values())))
        for s in self.stages.values():
            for e in s.expr:
                _logger.debug('stage.expr: %s' % e)
        for s in self.stages.values():
            for n, w in s.offset.items():
                _logger.debug('stage.offset: %s@%d <- %s@[%s]' %
                              (s.name, serialize(s.output.idx, self.tile_size),
                               n, ', '.join(map(str, w))))
        for s in self.stages.values():
            for n, d in s.delay.items():
                _logger.debug('stage.delay: %s <- %s delayed %d' %
                              (s.name, n, d))

        # parameters generated from the above parameters
        self.pixel_width_i = TYPE_WIDTH[self.input.type]
        self.pixel_width_o = TYPE_WIDTH[self.output.type]
        burst_width = self.burst_width
        pixel_width_i = self.pixel_width_i
        pixel_width_o = self.pixel_width_o
        unroll_factor = self.unroll_factor
        dram_bank = self.dram_bank
        if (burst_width/pixel_width_i*dram_bank/2 >
                unroll_factor/2):
            self.input_partition = burst_width/pixel_width_i*dram_bank/2
        else:
            self.input_partition = unroll_factor/2
        if burst_width/pixel_width_o*dram_bank/2 > unroll_factor/2:
            self.output_partition = burst_width/pixel_width_o*dram_bank/2
        else:
            self.output_partition = unroll_factor/2

        self.dataflow_super_source = dataflow.create_dataflow_graph(self)

    def get_producer_tensors(self):
        return [tensor for tensor in self.tensors.values()
                if len(tensor.children) > 0]

    def get_consumer_tensors(self):
        return [tensor for tensor in self.tensors.values()
                if tensor.parent is not None]

    def get_stages_chronologically(self):
        return [self.stages[tensor.name]
                for tensor in self.chronological_tensors
                if tensor.name in self.stages]

    # return [Tensor, ...]
    def _get_parent_tensors_for(self, node):
        return {x: self.tensors[x]
                for x in {x.name for x in node.get_loads()
                          if x.name not in self.extra_params}}

    # return {name: [(idx, ...), ...]}
    def _get_window_for(self, node):
        loads = node.get_loads() # [Load, ...]
        load_names = {l.name for l in loads
                      if l.name not in self.extra_params}
        windows = {name: sorted({l.idx for l in loads if l.name == name},
                                key=lambda x: serialize(x, self.tile_size))
                   for name in load_names}
        _logger.debug('window for %s@(%s) is %s' %
            (node.name, ', '.join(map(str, node.expr[0].idx)), windows))
        return windows

    # return [StageExpr, ...]
    def _get_expr_for(self, node):
        if isinstance(node, grammar.Output):
            return node.expr
        if isinstance(node, grammar.Local):
            return node.expr
        raise SemanticError('cannot get expression for %s' % str(type(node)))

    def get_reuse_buffers(self):
        if not hasattr(self, 'reuse_buffers'):
            unroll_factor = self.unroll_factor
            self.reuse_buffer_lengths = {}
            self.reuse_buffers = {}
            for tensor in self.get_producer_tensors():
                reuse_buffer = _get_reuse_buffer(self.tile_size,
                                                 tensor,
                                                 unroll_factor)
                reuse_buffer_length = {}
                self.reuse_buffers[tensor.name] = reuse_buffer
                self.reuse_buffer_lengths[tensor.name] = reuse_buffer_length
                first = [True]*unroll_factor
                for start, end in reuse_buffer[1:]:
                    if first[start%unroll_factor]:
                        first[start%unroll_factor] = False
                        if start >= unroll_factor:
                            reuse_buffer_length[end] = end//unroll_factor
                            continue
                    reuse_buffer_length[end] = (end-start)//unroll_factor
        return self.reuse_buffers

    def get_all_points(self):
        if not hasattr(self, 'all_points'):
            self.all_points = {}
            for tensor in self.get_producer_tensors():
                self.all_points[tensor.name] = _get_points(self.tile_size,
                                                           tensor,
                                                           self.unroll_factor)
        return self.all_points

    def get_next_fifo(self):
        if not hasattr(self, 'next_fifo'):
            self.next_fifo = {}
            for name, reuse_buffer in self.get_reuse_buffers().items():
                self.next_fifo[name] = {}
                for start, end in reuse_buffer[1:]:
                    if start < end:
                        self.next_fifo[name][start] = end
            _logger.debug('next_fifo: %s' % self.next_fifo)
        return self.next_fifo

    def get_forwarders(self):
        if not hasattr(self, 'forwarders'):
            all_points = self.get_all_points()
            self.forwarders = set()
            self.forwarders_with_border = set()
            next_fifo = self.get_next_fifo()
            for src_name, dsts in all_points.items():
                for dst_name, dst_point_dicts in dsts.items():
                    for offset, points in sorted(dst_point_dicts.items()):
                        if (
                            offset < self.unroll_factor and
                            self.tensors[src_name].preserve_border_to() and
                            not self.tensors[src_name].is_input()):
                            self.forwarders_with_border |= {(
                                src_name,
                                len(self.get_forwardings(src_name)[offset][1]))}
                        else:
                            self.forwarders |= {
                                len(self.get_forwardings(src_name)[offset][1])}
            self.forwarders = sorted(self.forwarders)
            self.forwarders_with_border = sorted(self.forwarders_with_border)
        return self.forwarders

    def get_forwarders_with_border(self):
        if not hasattr(self, 'forwarders_with_border'):
            self.get_forwarders()
        return self.forwarders_with_border

    def get_reuse_buffer_length(self, name, offset):
        if not hasattr(self, 'reuse_buffer_lengths'):
            self.get_reuse_buffers()
        return self.reuse_buffer_lengths[name][offset]

    def get_forwardings(self, src_name):
        if hasattr(self, 'forwardings'):
            if src_name in self.forwardings:
                return self.forwardings[src_name]
        else:
            self.forwardings = {}
        next_fifo = self.get_next_fifo()
        unroll_factor = self.unroll_factor
        dsts = self.get_all_points()[src_name]
        reuse_buffer = self.get_reuse_buffers()[src_name]

        # {offset: [func_name, outputs, inputs, params, temp_param]}
        forwardings = {}

        for dst_name, dst_point_dicts in dsts.items():
            for offset, points in dst_point_dicts.items():
                forwardings.setdefault(offset, ['', [], [], [], None])
                func_name = forwardings[offset][0]
                outputs = forwardings[offset][1]
                inputs = forwardings[offset][2]
                params = forwardings[offset][3]
                for unroll_idx, point_index in points.items():
                    outputs.insert(
                        0,
                        '/* output */ '
                        'from_%s_to_%s_param_%d_chan_%%d_pe_%d' %
                        (src_name, dst_name, point_index, unroll_idx))

                if func_name:
                    continue
                if offset in next_fifo[src_name]:
                    outputs.append(
                        '/* output */ %s_offset_%d_chan_%%d' %
                        (src_name, next_fifo[src_name][offset]))
                inputs.append(
                    '/*  input */ %s_offset_%d_chan_%%d' %
                    (src_name, offset))
                func_name = 'forward'
                temp_param = self.get_reuse_buffer_length(src_name, offset)
                forward_num = len(params)-1
                if (
                    offset < self.unroll_factor and
                    self.tensors[src_name].preserve_border_to() and
                    not self.tensors[src_name].is_input()):
                    stage = self.stages[src_name]
                    if stage.preserve_border_from():
                        self_window_input = stage.preserve_border_from()
                    else:
                        self_window_input = self.input
                    self_window = get_overall_stencil_window(
                        self_window_input, stage.output)
                    overall_idx = get_stencil_window_offset(self_window)
                    self_dim = get_stencil_dim(self_window)
                    iteration = 1
                    parent = stage.preserve_border_from()
                    while (
                        parent is not None and
                        parent.parent is not None):
                        parent = parent.parent.preserve_border_from()
                        iteration += 1
                    delay = (
                        get_stencil_distance(self_window, self.tile_size)-
                        serialize(overall_idx, self.tile_size))*iteration

                    func_name += '_'+src_name
                    temp_param = '%d-%d' % (unroll_idx, delay)
                    for d in range(self.dim-1):
                        param_offset = (
                            (self.tile_size[d]-self_dim[d]+1)*
                            reduce(
                                operator.mul,
                                [self.tile_size[dd] for dd in range(d)],
                                1))
                        param = (
                            src_name, d,
                            (unroll_idx+param_offset)%self.unroll_factor)
                        inputs.append(
                            '/*  input */ border_from_%s_dim_%d_'
                            'left_chan_%%d_pe_%d' % param)
                        param = (
                            src_name, d,
                            (unroll_idx-param_offset)%self.unroll_factor)
                        inputs.append(
                            '/*  input */ border_from_%s_dim_%d_'
                            'right_chan_%%d_pe_%d' % param)
                    for d in range(self.dim-1):
                        params.append('/*  param */ input_bound_dim_%d' % d)
                    for d in range(self.dim):
                        params.append('/*  param */ input_size_dim_%d' % d)

                params.append('/*  param */ epoch_num')
                forwardings[offset][0] = func_name
                forwardings[offset][4] = temp_param
        self.forwardings[src_name] = forwardings
        return forwardings

    def get_replicated_next_fifo(self):
        if not hasattr(self, 'replicated_next_fifo'):
            self.replicated_next_fifo = {}
            for name, reuse_buffer in (self.get_replicated_reuse_buffers()
                    .items()):
                self.replicated_next_fifo[name] = {}
                for start, end in reuse_buffer[1:]:
                    if start < end:
                        self.replicated_next_fifo[name][start] = end
            _logger.debug('replicated_next_fifo: %s'
                % self.replicated_next_fifo)
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
            for tensor in self.get_producer_tensors():
                reuse_buffer = _get_replicated_reuse_buffer(self.tile_size,
                    tensor, replication_factor)
                self.replicated_reuse_buffers[tensor.name] = reuse_buffer
                self.replicated_reuse_buffer_lengths[tensor.name] = {}
                first = [True]*self.replication_factor
                for start, end in reuse_buffer[1:]:
                    if first[start%replication_factor]:
                        first[start%replication_factor] = False
                        if start >= replication_factor:
                            self.replicated_reuse_buffer_lengths[tensor.name][
                                end] = end//replication_factor
                            continue
                    self.replicated_reuse_buffer_lengths[tensor.name][end] = (
                        end-start)//replication_factor
            _logger.debug('replicated_reuse_buffers: %s' %
                self.replicated_reuse_buffers)
            _logger.debug('replicated_reuse_buffer_lengths: %s' %
                self.replicated_reuse_buffer_lengths)
        return self.replicated_reuse_buffers

    def get_replicated_all_points(self):
        if not hasattr(self, 'replicated_all_points'):
            self.replicated_all_points = {}
            for tensor in self.get_producer_tensors():
                self.replicated_all_points[tensor.name
                    ] = _get_replicated_points(self.tile_size, tensor)
            _logger.debug('replicated_all_points: %s' %
                self.replicated_all_points)
        return self.replicated_all_points

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

    _logger.debug('get reuse chains of tensor %s' % tensor.name)

    def unroll_offsets(offsets):
        unrolled_offsets = set()
        for unroll_idx in range(unroll_factor):
            for offset in offsets:
                unrolled_offsets.add(max(offsets) + unroll_idx - offset +
                                     stage.delay[tensor.name])
        return unrolled_offsets

    A_dag = set()
    for stage in tensor.children:
        A_dag |= unroll_offsets(serialize_iter(stage.window[tensor.name],
                                               tile_size))
    _logger.debug('A† of tensor %s: %s' % (tensor.name, A_dag))

    chains = []
    for chain_idx in reversed(range(unroll_factor)):
        chains.append(tuple(sorted(
            offset for offset in A_dag if offset % unroll_factor == chain_idx)))
    _logger.debug(chains)

    for idx, chain in enumerate(chains):
        _logger.debug('reuse chain %d of tensor %s: %s' %
                      (idx, tensor.name, chain))
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

    all_points = {} # {name: {offset: {unroll_idx: point_idx}}}
    for stage in tensor.children:
        all_points[stage.name] = {}
        offsets = serialize_iter(stage.window[tensor.name], tile_size)
        for unroll_idx in range(unroll_factor):
            for idx, offset in enumerate(offsets):
                all_points[stage.name].setdefault(
                    max(offsets) - offset + stage.delay[tensor.name] +
                        unroll_idx,
                    {})[unroll_factor-1-unroll_idx] = idx
    for stage in tensor.children:
        for offset, points in all_points[stage.name].items():
            for unroll_idx, point in points.items():
                _logger.debug(
                    '%s <- %s @ offset=%d <=> (%s) @ unroll_idx=%d' %
                    (stage.name, tensor.name, offset,
                     ', '.join(map(str, stage.window[tensor.name][point])),
                     unroll_idx))
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
    for chain in _get_reuse_chains(tile_size, tensor, unroll_factor):
        reuse_buffer.append((chain[0], chain[0]))
        offsets.append(chain[0])
        for j in range(len(chain)-1):
            reuse_buffer.append((chain[j], chain[j+1]))
            offsets.append(chain[j+1])
    reuse_buffer[0] = max(offsets)+1
    _logger.debug('reuse chains of tensor %s: %s' % (tensor.name, reuse_buffer))
    return reuse_buffer

def _get_replicated_reuse_chains(tile_size, tensor, replication_factor):
    _logger.debug('\033[1mget replicated reuse chains of tensor %s\033[0m' %
        tensor.name)
    A_dag = set()
    for stage in tensor.children:
        offsets = serialize_iter(stage.window[tensor.name], tile_size)
        A_dag |= {max(offsets)-offset+stage.delay[tensor.name]
            for offset in offsets}
    _logger.debug('A† of tensor %s: %s' % (tensor.name, A_dag))
    chains = sum(reversed([
        tuple([tuple(sorted(x for x in A_dag if x%replication_factor == i))])
        for i in range(replication_factor)]), ())
    for idx, chain in enumerate(chains):
        _logger.debug('reuse chain %d of tensor %s: %s' %
            (idx, tensor.name, chain))
    return chains

def _get_replicated_points(tile_size, tensor):
    all_points = {} # {name:{offset:point_index}}
    for stage in tensor.children:
        all_points[stage.name] = {}
        offsets = serialize_iter(stage.window[tensor.name], tile_size)
        max_offset = max(offsets)
        for idx, offset in enumerate(offsets):
            all_points[stage.name][
                max_offset-offset+stage.delay[tensor.name]] = idx
    for stage in tensor.children:
        for offset, points in all_points[stage.name].items():
            _logger.debug('%s <- %s @ offset=%d <=> (%s)' % (
                stage.name, tensor.name, offset,
                ', '.join(map(str, stage.window[tensor.name][points]))))
    return all_points

def _get_replicated_reuse_buffer(tile_size, tensor, replication_factor):
    reuse_buffer = [None] # [length, (start, end), (start, end), ...]
    offsets = []
    for chain in _get_replicated_reuse_chains(tile_size, tensor,
            replication_factor):
        if chain:
            reuse_buffer.append((chain[0], chain[0]))
            offsets.append(chain[0])
            for j in range(len(chain)-1):
                reuse_buffer.append((chain[j], chain[j+1]))
                offsets.append(chain[j+1])
    reuse_buffer[0] = max(offsets)+1
    _logger.debug('reuse chains of tensor %s: %s' %
        (tensor.name, reuse_buffer))
    return reuse_buffer

class Printer(object):
    def __init__(self, out):
        self.out = out
        self.indent = 0
        self.assign = 0
        self.comments = []

    def println(self, line='', local_indent=-1):
        if local_indent < 0:
            local_indent = self.indent
        if line:
            self.out.write('%s%s\n' % (' '*local_indent*4, line))
        else:
            self.out.write('\n')

    def do_indent(self):
        self.indent += 1

    def un_indent(self):
        self.indent -= 1

    def do_scope(self, comment=''):
        self.println('{')
        self.do_indent()
        self.comments.append(comment)

    def un_scope(self, comment=''):
        self.un_indent()
        popped_comment = self.comments.pop()
        if comment:
            self.println('} // %s' % comment)
        else:
            if popped_comment:
                self.println('} // %s' % popped_comment)
            else:
                self.println('}')

    def new_var(self):
        self.assign += 1
        return self.last_var()

    def last_var(self, offset=-1):
        return 'assign_%d' % (self.assign+offset)

    def print_func(self, name, params, suffix='', align=80):
        lines = [name+'(']
        for param in params:
            if ((self.indent + min(1, len(lines)-1))*4+
                    len(lines[-1])+len(param+', ')) > align:
                lines.append(param+', ')
            else:
                lines[-1] += param+', '
        if lines[-1][-2:] == ', ':
            lines[-1] = lines[-1][:-2]+')'+suffix
        line = lines.pop(0)
        self.println(line)
        if lines:
            self.do_indent()
            for line in lines:
                self.println(line)
            self.un_indent()

def get_c_type(soda_type):
    if soda_type in {
            'uint8', 'uint16', 'uint32', 'uint64',
            'int8', 'int16', 'int32', 'int64'}:
        return soda_type+'_t'
    return soda_type

def get_soda_type(c_type):
    return c_type[:-2] if c_type[-2:] == '_t' else c_type

def is_float(soda_type):
    return soda_type in {'float', 'double'}

def print_guard(printer, var, val):
    printer.println('#if %s != %d' % (var, val))
    printer.println('#error %s != %d' % (var, val))
    printer.println('#endif//%s != %d' % (var, val))

def print_define(printer, var, val):
    printer.println('#ifndef %s' % var)
    printer.println('#define %s %d' % (var, val))
    printer.println('#endif//%s' % var)

def get_indices_id(indices):
    return '_'.join(str(idx).replace('-', 'm') for idx in indices)

def serialize(vec, tile_size):
    return sum((vec[i]*reduce(operator.mul, tile_size[:i])
                for i in range(1, len(tile_size))),
               vec[0])

def serialize_iter(iterative, tile_size):
    return [serialize(x, tile_size) for x in iterative]

def get_stencil_distance(stencil_window, tile_size):
    return (max(serialize_iter(stencil_window, tile_size))+
            serialize(get_stencil_window_offset(stencil_window), tile_size))

def get_stencil_dim(points):
    dimension = len(next(iter(points)))
    return [max_index-min_index+1 for max_index, min_index in zip(
        [max([point[dim] for point in points]) for dim in range(dimension)],
        [min([point[dim] for point in points]) for dim in range(dimension)])]

_overall_stencil_window_cache = {}
def get_overall_stencil_window(input_tensor, output_tensor):
    # normalize store index to 0
    idx = (id(input_tensor), id(output_tensor))
    if idx in _overall_stencil_window_cache:
        return _overall_stencil_window_cache[idx]
    _logger.debug('get overall stencil window of %s <- %s' %
                  (output_tensor.name, input_tensor.name))
    all_points = set()
    if output_tensor.parent is not None:
        for name, points in output_tensor.parent.window.items():
            if name != input_tensor.name:
                recursive_points = get_overall_stencil_window(
                    input_tensor, output_tensor.parent.inputs[name])
                all_points |= set.union(*[{
                    tuple(map(lambda a, b, c: a + b - c,
                              p, point, output_tensor.idx))
                    for p in recursive_points} for point in points])
            else:
                all_points |= {tuple(map(operator.sub,
                                         point, output_tensor.idx))
                               for point in points}
    _logger.debug(
        'overall stencil window of %s (%s) <- %s is %s (%d points)' %
        (output_tensor.name, ', '.join(['0']*len(output_tensor.idx)),
         input_tensor.name, all_points, len(all_points)))
    _overall_stencil_window_cache[idx] = all_points
    return all_points

def get_stencil_window_offset(stencil_window):
    # only works if window is normalized to store at 0
    return tuple(-min(p[d] for p in stencil_window)
                 for d in range(len(next(iter(stencil_window)))))
