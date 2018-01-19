#!/usr/bin/python3.6
from collections import deque, namedtuple
from fractions import Fraction
from functools import reduce
import json
import logging
import math
import operator
import os
import sys

import supo.grammar

# constants
coords_tiled = 'xyzw'
coords_in_tile = 'ijkl'
coords_in_orig = 'pqrs'
type_width = {'uint8_t':8, 'uint16_t':16, 'uint32_t':32, 'uint64_t':64, 'int8_t':8, 'int16_t':16, 'int32_t':32, 'int64_t':64, 'float':32, 'double':64}
max_dram_bank = 4

logger = logging.getLogger('__main__').getChild(__name__)

class MutableInt():
    def __init__(self, value):
        self.value = value
    def __index__(self):
        return self.value
    def str(self):
        return str(self.value)
    def get(self):
        return self.value
    def set(self, value):
        self.value = value
        return self.value

Buffer = namedtuple('Buffer', ['name', 'type', 'chan', 'idx', 'parent', 'children', 'offset'])
# Buffer.name: str
# Buffer.type: str
# Buffer.chan: int
# Buffer.idx: [(int, ...), ...]
# Buffer.parent: [Stage]
# Buffer.children: [Stage, ...]
# Buffer.offset: MutableInt

Stage = namedtuple('Stage', ['window', 'offset', 'delay', 'expr', 'inputs', 'output'])
# Stage.window: {str: [(int, ...), ...], ...}
# Stage.offset: {str: [int, ...], ...}
# Stage.delay: {str: int, ...}
# Stage.expr: [OutputExpr, ...]
# Stage.inputs: {str: Buffer, ...}
# Stage.output: Buffer

class Stencil(object):
    def __init__(self, **kwargs):
        # platform determined
        self.burst_width = kwargs.pop('burst_width')
        self.dram_bank = kwargs.pop('dram_bank')
        # application determined
        self.app_name = kwargs.pop('app_name')
        # parameters can be explored
        self.tile_size = kwargs.pop('tile_size')
        self.unroll_factor = kwargs.pop('unroll_factor')
        self.dram_separate = kwargs.pop('dram_separate')
        if self.dram_separate:
            if self.dram_bank%2 != 0:
                logging.getLogger(__name__).fatal('Number of DRAM banks has to be even when separated')
                sys.exit(-1)
            else:
                self.dram_bank = int(self.dram_bank/2)
        # stage-independent
        self.dim = kwargs.pop('dim')
        self.extra_params = kwargs.pop('extra_params')
        # stage-specific
        input_node = kwargs.pop('input')
        output_node = kwargs.pop('output')
        intermediates = kwargs.pop('intermediates')
        self.buffers = {i.name: Buffer(i.name, i.type, i.chan, [e.idx for e in i.expr], parent=[], children=[], offset=MutableInt(0)) for i in intermediates}
        if input_node.name in self.buffers:
            raise SemanticError('input name conflict with buffers: %s' % self.input.name)
        else:
            self.input = Buffer(input_node.name, input_node.type, input_node.chan, idx=[], parent=[], children=[], offset=MutableInt(0))
            self.buffers[self.input.name] = self.input
        if output_node.name in self.buffers:
            raise SemanticError('output name conflict with buffers: %s' % self.output.name)
        else:
            self.output = Buffer(output_node.name, output_node.type, output_node.chan, [e.idx for e in output_node.output_expr], parent=[], children=[], offset=MutableInt(0))
            self.buffers[self.output.name] = self.output

        self.stages = {}
        for intermediate in intermediates:
            child_buffer = self.buffers[intermediate.name]
            parent_buffers = self.GetParentBuffersFor(intermediate)
            window = self.GetWindowFor(intermediate)
            this_stage = Stage(window,
                offset={n: [Serialize(x, self.tile_size) for x in w] for n, w in window.items()},
                delay={},
                expr=self.GetExprFor(intermediate),
                inputs=parent_buffers,
                output=child_buffer)
            self.stages[intermediate.name] = this_stage
            child_buffer.parent.append(this_stage)
            for b in parent_buffers.values():
                b.children.append(this_stage)

        parent_buffers = self.GetParentBuffersFor(output_node)
        window = self.GetWindowFor(output_node)
        output_stage = Stage(window,
            offset={n: [Serialize(x, self.tile_size) for x in w] for n, w in window.items()},
            delay={},
            expr=self.GetExprFor(output_node),
            inputs=parent_buffers,
            output=self.output)
        self.stages[output_node.name] = output_stage
        self.output.parent.append(output_stage)
        for b in parent_buffers.values():
            b.children.append(output_stage)

        # now that we have global knowledge of the buffers we can calculate the offsets of buffers
        logger.info('calculate buffer offsets')
        processing_queue = deque([self.input.name])
        processed_buffers = {self.input.name}
        logger.debug('buffer %s is at offset %d' % (self.input.name, self.input.offset.get()))
        while len(processing_queue)>0:
            b = self.buffers[processing_queue.popleft()]
            logger.debug('inspecting buffer %s\'s children' % b.name)
            for s in b.children:
                if {x.name for x in s.inputs.values()} <= processed_buffers and s.output.name not in processed_buffers:
                    # good, all inputs are processed, can determine offset of current buffer
                    logger.debug('input%s for buffer %s (i.e. %s) %s processed' % ('' if len(s.inputs)==1 else 's', s.output.name, ', '.join([x.name for x in s.inputs.values()]), 'is' if len(s.inputs)==1 else 'are'))
                    s.output.offset.set(max([s.output.offset.get()] + [x.offset.get()+s.offset[x.name][-1] for x in s.inputs.values()]))
                    logger.debug('buffer %s is at offset %d' % (s.output.name, s.output.offset.get()))
                    for x in s.inputs.values():
                        delay = s.output.offset.get() - (x.offset.get()+s.offset[x.name][-1])
                        if delay>0:
                            logger.debug('buffer %s arrives at buffer %s at offset %d < %d; add %d delay' % (x.name, s.output.name, x.offset.get()+s.offset[x.name][-1], s.output.offset.get(), delay))
                        else:
                            logger.debug('buffer %s arrives at buffer %s at offset %d = %d; good' % (x.name, s.output.name, x.offset.get()+s.offset[x.name][-1], s.output.offset.get()))
                        s.delay[x.name] = max(delay, 0)
                        logger.debug('set delay of %s <- %s to %d' % (s.output.name, x.name, s.delay[x.name]))
                    processing_queue.append(s.output.name)
                    processed_buffers.add(s.output.name)
                else:
                    for bb in s.inputs.values():
                        if bb.name not in processed_buffers:
                            logger.debug('buffer %s requires buffer %s as an input' % (s.output.name, bb.name))
                            logger.debug('but buffer %s isn\'t processed yet' % bb.name)
                            logger.debug('add %s to scheduling queue' % bb.name)
                            processing_queue.append(bb.name)

        logger.debug('buffers: '+str(list(self.buffers.keys())))
        LoadPrinter = lambda node: '%s%s' % (node.name, node.idx) if node.name in self.extra_params else '%s[%d](%s)' % (node.name, node.chan, ', '.join([str(x) for x in node.idx]))
        StorePrinter = lambda node: '%s[%d](%s)' % (node.name, node.chan, ', '.join([str(x) for x in node.idx]))
        for s in self.stages.values():
            logger.debug('stage: %s <- [%s]' % (s.output.name, ', '.join(['%s@%s' % (x.name, list(set(s.window[x.name]))) for x in s.inputs.values()])))
        for s in self.stages.values():
            for e in s.expr:
                logger.debug('stage.expr: %s' % e.GetCode(LoadPrinter, StorePrinter))
        for s in self.stages.values():
            for n, w in s.offset.items():
                logger.debug('stage.offset: %s <- %s@[%s]' % (s.output.name, n, ', '.join([str(x) for x in w])))
        for s in self.stages.values():
            for n, d in s.delay.items():
                logger.debug('stage.delay: %s <- %s delayed %d' % (s.output.name, n, d))

        # parameters generated from the above parameters
        self.pixel_width_i = type_width[self.input.type]
        self.pixel_width_o = type_width[self.output.type]
        self.input_partition  = self.burst_width/self.pixel_width_i*self.dram_bank/2 if self.burst_width/self.pixel_width_i*self.dram_bank/2 > self.unroll_factor/2 else self.unroll_factor/2
        self.output_partition = self.burst_width/self.pixel_width_o*self.dram_bank/2 if self.burst_width/self.pixel_width_o*self.dram_bank/2 > self.unroll_factor/2 else self.unroll_factor/2

    def GetProducerBuffers(self):
        return [b for b in self.buffers.values() if len(b.children)>0]

    # return [Buffer, ...]
    def GetParentBuffersFor(self, node):
        return {x: self.buffers[x] for x in {x.name for x in node.GetLoads() if x.name not in self.extra_params}}

    # return {name: [(idx, ...), ...]}
    def GetWindowFor(self, node):
        loads = node.GetLoads() # [Load, ...]
        load_names = {l.name for l in loads if l.name not in self.extra_params}
        return {name: sorted({l.idx for l in loads if l.name == name}, key=lambda x: Serialize(x, self.tile_size)) for name in load_names}

    # return [OutputExpr, ...]
    def GetExprFor(self, node):
        if isinstance(node, supo.grammar.Output):
            return node.output_expr
        if isinstance(node, supo.grammar.Intermediate):
            return node.expr
        raise SemanticError('cannot get expression for %s' % str(type(node)))

class Printer(object):
    def __init__(self, out):
        self.out = out
        self.indent = 0
        self.assign = 0
        self.comments = []

    def PrintLine(self, line = '', local_indent = -1):
        if local_indent < 0:
            local_indent = self.indent
        if line:
            self.out.write('%s%s\n' % (' '*local_indent*4, line))
        else:
            self.out.write('\n')

    def DoIndent(self):
        self.indent += 1

    def UnIndent(self):
        self.indent -= 1

    def DoScope(self, comment=''):
        self.PrintLine('{')
        self.DoIndent()
        self.comments.append(comment)

    def UnScope(self, comment=''):
        self.UnIndent()
        popped_comment = self.comments.pop()
        if comment:
            self.PrintLine('} // %s' % comment)
        else:
            if popped_comment:
                self.PrintLine('} // %s' % popped_comment)
            else:
                self.PrintLine('}')

    def NewVar(self):
        self.assign += 1
        return self.LastVar()

    def LastVar(self, offset=-1):
        return 'assign_%d' % (self.assign+offset)

def GetCType(supo_type):
    if supo_type in {'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'}:
        return supo_type+'_t'
    return supo_type

def PrintGuard(printer, var, val):
    printer.PrintLine('#if %s != %d' % (var, val))
    printer.PrintLine('#error %s != %d' % (var, val))
    printer.PrintLine('#endif//%s != %d' % (var, val))

def PrintDefine(printer, var, val):
    printer.PrintLine('#ifndef %s' % var)
    printer.PrintLine('#define %s %d' % (var, val))
    printer.PrintLine('#endif//%s' % var)

def Serialize(vec, tile_size):
    # convert vec to scalar coordinates
    result = vec[0]
    for i in range(1, len(tile_size)):
        result += vec[i]*reduce(operator.mul, tile_size[0:i])
    return result

def GetStencilDistance(A, tile_size):
    A_serialized = [Serialize(x, tile_size) for x in A]
    return max(A_serialized) - min(A_serialized)

def GetStencilDim(A):
    return [max_index-min_index+1 for max_index, min_index in zip([max([point[dim] for point in A]) for dim in range(len(next(iter(A))))], [min([point[dim] for point in A]) for dim in range(len(next(iter(A))))])]

def GetOverallStencilWindow(input_buffer, output_buffer):
    logger.debug('get overall stencil window of %s <- %s' % (output_buffer.name, input_buffer.name))
    all_points = set()
    if len(output_buffer.parent)>0:
        for name, points in output_buffer.parent[0].window.items():
            if name != input_buffer.name:
                recursive_points = GetOverallStencilWindow(input_buffer, output_buffer.parent[0].inputs[name])
                all_points |= set.union(*[{tuple(map(operator.add, p, point)) for p in recursive_points} for point in points])
            all_points |= set(points)
    logger.debug('overall stencil window of %s <- %s is %s' % (output_buffer.name, input_buffer.name, all_points))
    return all_points

