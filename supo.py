#!/usr/bin/python3
from os.path import join, dirname
import logging
from textx import metamodel_from_file
from textx.export import metamodel_export, model_export
from textx.exceptions import TextXSyntaxError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupoProgram(object):
    def __init__(self, **kwargs):
        self.burst_width = kwargs.pop('burst_width')
        self.dram_chan = kwargs.pop('dram_chan')
        self.dram_separate = kwargs.pop('dram_separate') == 'yes'
        self.input = kwargs.pop('input')
        # TODO: get dims
        self.output = kwargs.pop('output')
        # TODO: check dims

    def __str__(self):
        return \
            ('burst width: %d\n' % self.burst_width) + \
            ('dram channel: %d\n' % self.dram_chan) + \
            ('dram separate: %s\n' % ('yes' if self.dram_separate else 'no')) + \
            ('input %s: %s(%s)\n' % (self.input.type, self.input.name, ', '.join([str(x) for x in self.input.tile_size] + ['']))) + \
            ('output %s: %s(%s) = %s\n' % (self.output.type, self.output.name, ', '.join([str(x) for x in self.output.idx]), self.output.expr)) + \
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

class Factor(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.sign = kwargs.pop('sign')

    def __str__(self):
        return ('-%s' if self.sign=='-' else '%s') % str(self.operand)

    def GetCode(self):
        return ('-%s' if self.sign=='-' else '%s') % self.operand.GetCode()

class Operand(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
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

def main():
    this_folder = dirname(__file__)
    supo_mm = metamodel_from_file(join(this_folder, 'supo.tx'), classes=[SupoProgram, Expression, Term, Factor, Operand])
    logger.info('Built metamodel.')
    try:
        supo_model = supo_mm.model_from_file(join(this_folder, 'test.supo'))
    except TextXSyntaxError as e:
        logger.error(e)
    else:
        print(supo_model)
        print(supo_model.output.expr.GetCode())

if __name__ == '__main__':
    main()
