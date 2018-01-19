#!/usr/bin/python3.6
from collections import deque, namedtuple
import logging
import operator
import os
import sys

import supo.generator.utils

logger = logging.getLogger('__main__').getChild(__name__)

supo_grammar = '''
SupoProgram:
(
    ('burst' 'width' ':' burst_width=INT)
    ('dram' 'bank' ':' dram_bank=INT)
    ('dram' 'separate' ':' dram_separate=YesOrNo)
    ('unroll' 'factor' ':' unroll_factor=INT)
    ('kernel' ':' app_name=ID)
    input=Input
    output=Output
    (extra_params=ExtraParam)*
    (intermediates=Intermediate)*
    Comment*
)#;
Bin: /0[Bb][01]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Comment: /\s*#.*$/;
Dec: /\d+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Expression: operand=Term (operator=PlusOrMinus operand=Term)*;
ExtraParam: 'param' type=Type (',' attrs=ExtraParamAttr)*  ':' name=ID ('[' size=INT ']')+;
ExtraParamAttr: 'dup' dup=Number | partitioning=Partitioning;
Factor: (sign=PlusOrMinus)? operand=Operand;
Float: /((\d+\.|\d*\.\d+)([+-]?[Ee]\d+)?|\d+[+-]?[Ee]\d+)[FfLl]?/;
Hex: /0[Xx][0-9a-fA-F]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Input: 'input' type=Type ':' name=ID ('[' chan=Integer ']')? '(' tile_size=INT ',' (tile_size=INT ',')* ')';
Integer: ('+'|'-')?(Hex|Bin|Oct|Dec);
Intermediate: 'buffer' type=Type ':' (expr=OutputExpr)+;
MulOrDiv: '*'|'/'|'%';
Number: Float|Integer;
Oct: /0[0-7]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Operand: name=ID ('[' chan=Integer ']')? '(' idx=INT (',' idx=INT)* ')' | num=Number | '(' expr=Expression ')';
Output: 'output' type=Type ':' (output_expr=OutputExpr)+;
OutputExpr: name=ID ('[' chan=Integer ']')? '(' idx=INT (',' idx=INT)* ')' '=' expr=Expression;
Partitioning: 'partition' partition_type='complete' ('dim' '=' dim=Number)? | 'partition' partition_type='cyclic' 'factor' '=' factor=Number ('dim' '=' dim=Number)?;
PlusOrMinus: '+'|'-';
Term: operand=Factor (operator=MulOrDiv operand=Factor)*;
Type: 'int8'|'int16'|'int32'|'int64'|'uint8'|'uint16'|'uint32'|'uint64'|'float'|'double';
YesOrNo: 'yes'|'no';
'''

def StringToInteger(s, none_val=None):
    if s is None:
        return none_val
    if s[0:2] == '0x' or s[0:2] == '0X':
        return int(s, 16)
    if s[0:2] == '0b' or s[0:2] == '0B':
        return int(s, 2)
    if s[0] == '0':
        return int(s, 8)
    return int(s)

class SemanticError(Exception):
    pass

class SemanticWarn(Exception):
    pass

Load = namedtuple('Load', ['name', 'chan', 'idx'])

class SupoProgram(object):
    def __init__(self, **kwargs):
        self.burst_width = kwargs.pop('burst_width')
        self.dram_bank = kwargs.pop('dram_bank')
        self.app_name = kwargs.pop('app_name')
        extra_params = kwargs.pop('extra_params')
        self.extra_params = {} if extra_params is None else {p.name: p for p in extra_params}
        self.input = kwargs.pop('input')
        self.output = kwargs.pop('output')
        self.tile_size = self.input.tile_size
        self.dim = len(self.tile_size)
        self.intermediates = kwargs.pop('intermediates')
        self.unroll_factor = kwargs.pop('unroll_factor')
        self.dram_separate = kwargs.pop('dram_separate')=='yes'

    def __str__(self):
        return \
            ('burst width: %d\n' % self.burst_width) + \
            ('dram bank: %d\n' % self.dram_bank) + \
            ('kernel: %d\n' % self.app_name) + \
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

    def GetCode(self, LoadPrinter, StorePrinter):
        return '%s%s' % \
            (''.join([
                operand.GetCode(LoadPrinter, StorePrinter)+' '+operator+' ' for operand, operator
                    in zip(self.operand, self.operator)]), self.operand[-1].GetCode(LoadPrinter, StorePrinter))

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([op.GetLoads() for op in self.operand], [])
        return self.loads

class Term(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.operator = kwargs.pop('operator')

    def __str__(self):
        return '%s%s' % \
            (''.join([
                str(operand)+' '+str(operator)+' ' for operand, operator
                    in zip(self.operand, self.operator)]), str(self.operand[-1]))

    def GetCode(self, LoadPrinter, StorePrinter):
        return '%s%s' % \
            (''.join([
                operand.GetCode(LoadPrinter, StorePrinter)+' '+operator+' ' for operand, operator
                    in zip(self.operand, self.operator)]), self.operand[-1].GetCode(LoadPrinter, StorePrinter))

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([op.GetLoads() for op in self.operand], [])
        return self.loads

class Factor(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.sign = kwargs.pop('sign')

    def __str__(self):
        return ('-%s' if self.sign=='-' else '%s') % str(self.operand)

    def GetCode(self, LoadPrinter, StorePrinter):
        return ('-%s' if self.sign=='-' else '%s') % self.operand.GetCode(LoadPrinter, StorePrinter)

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = self.operand.GetLoads()
        return self.loads

class Operand(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.chan = StringToInteger(kwargs.pop('chan'), 0)
        self.idx = tuple(kwargs.pop('idx'))
        self.num = kwargs.pop('num')
        self.expr = kwargs.pop('expr')

    def __str__(self):
        if self.name:
            return '%s[%d](%s)' % (self.name, self.chan, ', '.join([str(x) for x in self.idx]))
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % str(self.expr)

    def GetCode(self, LoadPrinter, StorePrinter):
        if self.name:
            return LoadPrinter(self)
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % self.expr.GetCode(LoadPrinter, StorePrinter)

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            if self.expr is not None:
                self.loads = self.expr.GetLoads()
            elif self.num is not None:
                self.loads = []
            elif self.name is not None:
                logger.debug('load at %s[%d](%s)' % (self.name, self.chan, ', '.join([str(x) for x in self.idx])))
                self.loads = [Load(self.name, self.chan, self.idx)]
            else:
                raise SemanticError('invalid Operand %s' % str(self))
        return self.loads

class Input(object):
    def __init__(self, **kwargs):
        self.type = supo.generator.utils.GetCType(kwargs.pop('type'))
        self.name = kwargs.pop('name')
        self.chan = StringToInteger(kwargs.pop('chan'), 1)
        if(self.chan<1):
            raise SemanticError('input %s has 0 channels' % self.name)
        self.tile_size = kwargs.pop('tile_size')+[0]

    def __str__(self):
        return ('input %s: %s[%d](%s)' % (self.type, self.chan, self.name, ', '.join([str(x) for x in self.tile_size[0:-1]] + [''])))

class ExtraParam(object):
    def __init__(self, **kwargs):
        self.type = supo.generator.utils.GetCType(kwargs.pop('type'))
        self.name = kwargs.pop('name')
        self.size = kwargs.pop('size')
        attrs = kwargs.pop('attrs')
        self.dup = None
        self.partitioning = []
        for attr in attrs:
            if attr.dup is not None:
                if self.dup is not None:
                    warn_msg = 'parameter duplication factor redefined as %d, previously defined as %d' % (StringToInteger(attr.dup), self.dup)
                    raise SemanticWarn(warn_msg)
                self.dup = StringToInteger(attr.dup)
            if attr.partitioning is not None:
                attr.partitioning.dim = StringToInteger(attr.partitioning.dim)
                attr.partitioning.factor = StringToInteger(attr.partitioning.factor)
                self.partitioning.append(attr.partitioning)

    def __str__(self):
        return ('param %s: %s[%s]%s%s' % (
            self.type,
            self.name,
            ']['.join([str(x) for x in self.size]),
            '' if self.dup is None else (', dup x%d' % self.dup),
            ''.join([', %s partition%s%s' % (
                x.partition_type,
                '' if x.dim is None else ' in dim %d' % x.dim,
                '' if x.factor is None else' with factor=%d' % x.factor
            ) for x in self.partitioning]) if self.partitioning else ''
        ))

class Output(object):
    def __init__(self, **kwargs):
        self.type = supo.generator.utils.GetCType(kwargs.pop('type'))
        self.output_expr = kwargs.pop('output_expr')
        for e in self.output_expr:
            if hasattr(self, 'name'):
                if self.name != e.name:
                    err_msg = 'output had name %s but now renamed to %s' % (self.name, e.name)
                    raise SemanticError(err_msg)
            else:
                self.name = e.name
                logger.debug('output named as %s' % e.name)
            if hasattr(self, 'chan'):
                if e.chan in self.chan:
                    logger.warn('output channel %d redefined' % e.chan)
                self.chan |= {e.chan}
            else:
                self.chan = {e.chan}
        if self.chan != set(range(len(self.chan))):
            err_msg = ('output channel poorly-defined: %s' % str(self.chan))
            raise SemanticError(err_msg)
        self.chan = len(self.chan)

    def __str__(self):
        return ('output %s: %s' % (self.type, self.output_expr))

    def GetCode(self, LoadPrinter, StorePrinter):
        return [x.GetCode(LoadPrinter, StorePrinter) for x in self.output_expr]

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([e.GetLoads() for e in self.output_expr], [])
        return self.loads

class Intermediate(object):
    def __init__(self, **kwargs):
        self.type = supo.generator.utils.GetCType(kwargs.pop('type'))
        self.expr = kwargs.pop('expr')
        for e in self.expr:
            if hasattr(self, 'name'):
                if self.name != e.name:
                    err_msg = 'intermediate had name %s but now renamed to %s' % (self.name, e.name)
                    raise SemanticError(err_msg)
            else:
                self.name = e.name
                logger.debug('intermediate named as %s' % e.name)
            if hasattr(self, 'chan'):
                if e.chan in self.chan:
                    logger.warn('intermediate channel %d redefined' % e.chan)
                self.chan |= {e.chan}
            else:
                self.chan = {e.chan}
        if self.chan != set(range(len(self.chan))):
            err_msg = ('intermediate channel poorly-defined: %s' % str(self.chan))
            raise SemanticError(err_msg)
        self.chan = len(self.chan)

    def __str__(self):
        return ('intermediate %s: %s' % (self.type, self.expr))

    def GetCode(self, LoadPrinter, StorePrinter):
        return [x.GetCode(LoadPrinter, StorePrinter) for x in self.expr]

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([e.GetLoads() for e in self.expr], [])
        return self.loads

class OutputExpr(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.chan = StringToInteger(kwargs.pop('chan'), 0)
        self.idx = tuple(kwargs.pop('idx'))
        self.expr = kwargs.pop('expr')
        logger.debug('store at %s[%d](%s)' % (self.name, self.chan, ', '.join([str(x) for x in self.idx])))

    def __str__(self):
        return ('%s[%d](%s) = %s' % (self.name, self.chan, ', '.join([str(x) for x in self.idx]), self.expr))

    def GetCode(self, LoadPrinter, StorePrinter):
        return '%s = %s;' % (StorePrinter(self), self.expr.GetCode(LoadPrinter, StorePrinter))

    def GetLoads(self):
        if not hasattr(self, 'loads'):
            self.loads = self.expr.GetLoads()
        return self.loads

