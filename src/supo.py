#!/usr/bin/python3
from functools import reduce
from os.path import join, dirname
from textx import metamodel_from_str
from textx.exceptions import TextXSyntaxError
import argparse
import logging
import operator
import os
import supo.generator.kernel, supo.generator.host, supo.generator.header
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

supo_grammar = '''
SupoProgram:
(
    ('burst' 'width' ':' burst_width=INT)
    ('dram' 'channel' ':' dram_chan=INT)
    ('dram' 'separate' ':' dram_separate=YesOrNo)
    ('unroll' 'factor' ':' k=INT)
    ('kernel' ':' app_name=ID)
    input=Input
    output=Output
    Comment*
)#;
Comment: /\s*#.*$/;
YesOrNo: 'yes'|'no';
Type: 'int8'|'int16'|'int32'|'int64'|'uint8'|'uint16'|'uint32'|'uint64'|'float'|'double';
PlusOrMinus: '+'|'-';
MulOrDiv: '*'|'/'|'%';
Expression: operand=Term (operator=PlusOrMinus operand=Term)*;
Term: operand=Factor (operator=MulOrDiv operand=Factor)*;
Factor: (sign=PlusOrMinus)? operand=Operand;
Operand: name=ID ('[' chan=INT ']')? '(' idx=INT (',' idx=INT)* ')' | num=Number | '(' expr=Expression ')';
Number: Float|Hex|Bin|Oct|Dec;
Dec: /\d+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Hex: /0[Xx][0-9a-fA-F]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Bin: /0[Bb][01]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Oct: /0[0-7]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Float : /((\d+\.|\d*\.\d+)([+-]?[Ee]\d+)?|\d+[+-]?[Ee]\d+)[FfLl]?/;
Input: 'input' type=Type ':' name=ID ('[' chan=INT ']')? '(' tile_size=INT ',' (tile_size=INT ',')* ')';
Output: 'output' type=Type ':' name=ID ('[' chan=INT ']')? '(' idx=INT (',' idx=INT)* ')' '=' expr=Expression;
'''

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
        self.output = kwargs.pop('output')
        # TODO: check dims
        self.A = self.output.GetA()
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
    parser = argparse.ArgumentParser(prog='supoc', description='Stencil with Unrolling and Pipelining Optimization (SUPO) compiler')
    parser.add_argument('--burst-width',
                        type=int,
                        dest='burst_width',
                        help='override burst width')
    parser.add_argument('--unroll-factor',
                        type=int,
                        metavar='UNROLL_FACTOR',
                        dest='k',
                        help='override unroll factor')
    parser.add_argument('--tile-size',
                        type=int,
                        nargs='+',
                        metavar='TILE_SIZE',
                        dest='tile_size',
                        help='override tile size; 0 means no overriding on that dimension')
    parser.add_argument('--dram-channel',
                        type=int,
                        dest='dram_chan',
                        help='override DRAM channel num')
    parser.add_argument('--dram-separate',
                        type=bool,
                        choices=['yes', 'no'],
                        dest='dram_separate',
                        help='override DRAM separation')
    parser.add_argument(type=argparse.FileType('r'),
                        dest='supo_src',
                        metavar='file',
                        help='supo source code')
    parser.add_argument('--output-dir', '-o',
                        type=str,
                        dest='output_dir',
                        metavar='dir',
                        help='directory to generate kernel, source, and header; default names used; default to the current working directory; may be overridden by --kernel-file, --source-file, or --header-file')
    parser.add_argument('--kernel-file',
                        type=argparse.FileType('w'),
                        dest='kernel_file',
                        metavar='file',
                        help='Vivado HLS C++ kernel code; overrides --output-dir')
    parser.add_argument('--source-file',
                        type=argparse.FileType('w'),
                        dest='host_file',
                        metavar='file',
                        help='host C++ source code; overrides --output-dir')
    parser.add_argument('--header-file',
                        type=argparse.FileType('w'),
                        dest='header_file',
                        metavar='file',
                        help='host C++ header code; overrides --output-dir')

    args = parser.parse_args()
    # TODO: check tile size

    supo_mm = metamodel_from_str(supo_grammar, classes=[SupoProgram, Expression, Term, Factor, Operand, Input, Output])
    logger.debug('Build metamodel')
    try:
        supo_model = supo_mm.model_from_str(args.supo_src.read())
    except TextXSyntaxError as e:
        logger.error(e)
        if args.kernel_file is not None:
            os.remove(args.kernel_file.name)
        if args.host_file is not None:
            os.remove(args.host_file.name)
        if args.header_file is not None:
            os.remove(args.header_file.name)
    else:
        stencil = supo.generator.kernel.Stencil(
            burst_width = args.burst_width if args.burst_width is not None else supo_model.burst_width,
            dram_chan = args.dram_chan if args.dram_chan is not None else supo_model.dram_chan,
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
            tile_size = [args.tile_size[i] if args.tile_size is not None and i<len(args.tile_size) and args.tile_size[i] > 0 else supo_model.tile_size[i] for i in range(supo_model.dim-1)],
            k = args.k if args.k is not None else supo_model.k,
            dram_separate = args.dram_separate if args.dram_separate is not None else supo_model.dram_separate)

        logger.info('kernel        : %s' % stencil.app_name)
        logger.info('burst width   : %d' % stencil.burst_width)
        logger.info('dram channel  : %d' % (stencil.dram_chan*2) if stencil.dram_separate else stencil.dram_chan)
        logger.info('dram separate : %s' % 'yes' if stencil.dram_separate else 'no')
        logger.info('unroll factor : %d' % stencil.k)
        logger.info('tile size     : %s' % str(stencil.tile_size))
        logger.info('dimension     : %d' % stencil.dim)
        logger.info('input name    : %s' % stencil.input_name)
        logger.info('input type    : %s' % stencil.input_type)
        logger.info('input channel : %d' % stencil.input_chan)
        logger.info('output name   : %s' % stencil.output_name)
        logger.info('output type   : %s' % stencil.output_type)
        logger.info('output channel: %d' % stencil.output_chan)
        logger.info('computation   : %s' % stencil.compute_content)

        if args.kernel_file is not None:
            supo.generator.kernel.PrintCode(stencil, args.kernel_file)

        if args.host_file is not None:
            supo.generator.host.PrintCode(stencil, args.host_file)

        if args.header_file is not None:
            supo.generator.header.PrintCode(stencil, args.header_file)

        if args.output_dir is not None or args.kernel_file is None and args.host_file is None and args.header_file is None:
            if args.kernel_file is None:
                with open(join(args.output_dir if args.output_dir is not None else '', '%s_kernel-tile%s-unroll%d-%dddr%s.cpp' % (supo_model.app_name, 'x'.join(['%d'%x for x in supo_model.tile_size[0:-1]]), supo_model.k, supo_model.dram_chan, '-separated' if supo_model.dram_separate else '')), 'w') as kernel_file:
                    supo.generator.kernel.PrintCode(stencil, kernel_file)
            if args.host_file is None:
                with open(join(args.output_dir if args.output_dir is not None else '', '%s.cpp' % supo_model.app_name), 'w') as host_file:
                    supo.generator.host.PrintCode(stencil, host_file)
            if args.header_file is None:
                with open(join(args.output_dir if args.output_dir is not None else '', '%s.h' % supo_model.app_name), 'w') as header_file:
                    supo.generator.header.PrintCode(stencil, header_file)

if __name__ == '__main__':
    main()