#!/usr/bin/python3
from functools import reduce
from os.path import join, dirname
from textx import metamodel_from_file
from textx.exceptions import TextXSyntaxError
import logging
import operator
import sys
import supo.generator.kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupoProgram(object):
    def __init__(self, **kwargs):
        self.app_name = kwargs.pop('app_name')
        self.burst_width = kwargs.pop('burst_width')
        self.dram_chan = kwargs.pop('dram_chan')
        self.dram_separate = kwargs.pop('dram_separate') == 'yes'
        self.k = kwargs.pop('k')
        self.input = kwargs.pop('input')
        self.tile_size = self.input.tile_size
        self.dim = len(self.tile_size)
        # TODO: get dims
        self.output = kwargs.pop('output')
        # TODO: check dims
        self.A = self.output.GetA()
        print(self.A)
        self.compute_content = self.output.GetCode().split('\n')
        self.extra_params = None

    def __str__(self):
        return \
            ('burst width: %d\n' % self.burst_width) + \
            ('dram channel: %d\n' % self.dram_chan) + \
            ('dram separate: %s\n' % ('yes' if self.dram_separate else 'no')) + \
            ('%s\n' % self.input) + \
            ('%s\n' % self.output) + \
            ''

class Expression(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.operator = kwargs.pop('operator')

    def __str__(self):
        return '%s%s' % \
            (''.join([
                str(operand)+' '+str(operator)+' ' for operand, operator
                    in zip(self.operand, self.operator)]), str(self.operand[-1]))

    def GetCode(self):
        return '%s%s' % \
            (''.join([
                operand.GetCode()+' '+operator+' ' for operand, operator
                    in zip(self.operand, self.operator)]), self.operand[-1].GetCode())

    def GetLoadIndices(self):
        return reduce(operator.add, [op.GetLoadIndices() for op in self.operand])

class Term(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.operator = kwargs.pop('operator')

    def __str__(self):
        return '%s%s' % \
            (''.join([
                str(operand)+' '+str(operator)+' ' for operand, operator
                    in zip(self.operand, self.operator)]), str(self.operand[-1]))

    def GetCode(self):
        return '%s%s' % \
            (''.join([
                operand.GetCode()+' '+operator+' ' for operand, operator
                    in zip(self.operand, self.operator)]), self.operand[-1].GetCode())

    def GetLoadIndices(self):
        return reduce(operator.add, [op.GetLoadIndices() for op in self.operand])

class Factor(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.sign = kwargs.pop('sign')

    def __str__(self):
        return ('-%s' if self.sign=='-' else '%s') % str(self.operand)

    def GetCode(self):
        return ('-%s' if self.sign=='-' else '%s') % self.operand.GetCode()

    def GetLoadIndices(self):
        return self.operand.GetLoadIndices()

class Operand(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.chan = kwargs.pop('chan')
        if not self.chan:
            self.chan = 1
        self.idx = kwargs.pop('idx')
        self.num = kwargs.pop('num')
        self.expr = kwargs.pop('expr')

    def __str__(self):
        if self.name:
            return '%s(%s)' % (self.name, ', '.join([str(x) for x in self.idx]))
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % str(self.expr)

    def GetCode(self):
        if self.name:
            return 'load_%s_at_%s' % (self.name, '_'.join([str(x).replace('-','m') for x in self.idx]))
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % self.expr.GetCode()

    def GetLoadIndices(self):
        if self.expr:
            return self.expr.GetLoadIndices()
        if self.num:
            return []
        if self.name:
            return [self.idx]

class Input(object):
    def __init__(self, **kwargs):
        self.type = kwargs.pop('type')
        self.name = kwargs.pop('name')
        self.chan = kwargs.pop('chan')
        if not self.chan:
            self.chan = 1
        self.tile_size = kwargs.pop('tile_size')+[0]

    def __str__(self):
        return ('input %s: %s(%s)' % (self.type, self.name, ', '.join([str(x) for x in self.tile_size[0:-1]] + [''])))

class Output(object):
    def __init__(self, **kwargs):
        self.type = kwargs.pop('type')
        self.name = kwargs.pop('name')
        self.chan = kwargs.pop('chan')
        if not self.chan:
            self.chan = 1
        self.idx = kwargs.pop('idx')
        self.expr = kwargs.pop('expr')

    def __str__(self):
        return ('output %s: %s(%s) = %s' % (self.type, self.name, ', '.join([str(x) for x in self.idx]), self.expr))

    def GetCode(self):
        return 'output_type result = '+self.expr.GetCode()

    def GetA(self):
        return self.expr.GetLoadIndices()

def main():
    this_folder = dirname(__file__)
    supo_mm = metamodel_from_file(join(this_folder, 'supo.tx'), classes=[SupoProgram, Expression, Term, Factor, Operand, Input, Output])
    logger.info('Built metamodel.')
    try:
        supo_model = supo_mm.model_from_file(join(this_folder, 'test.supo'))
    except TextXSyntaxError as e:
        logger.error(e)
    else:
        stencil = supo.generator.kernel.Stencil(
            burst_width = supo_model.burst_width,
            dram_chan = supo_model.dram_chan,
            app_name = supo_model.app_name,
            input_name = supo_model.input.name,
            input_type = supo_model.input.type,
            input_chan = supo_model.input.chan,
            output_name = supo_model.output.name,
            output_type = supo_model.output.type,
            output_chan = supo_model.output.chan,
            A = supo_model.A,
            dim = supo_model.dim,
            compute_content = supo_model.compute_content,
            extra_params = supo_model.extra_params,
            tile_size = supo_model.tile_size,
            k = supo_model.k,
            dram_separate = supo_model.dram_separate)
        with open('%s_kernel.cpp' % supo_model.app_name, 'w') as kernel_file:
            supo.generator.kernel.PrintCode(stencil, kernel_file)

if __name__ == '__main__':
    main()
