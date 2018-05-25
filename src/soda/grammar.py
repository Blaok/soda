from collections import namedtuple
import copy
import logging
import math

from soda import core

logger = logging.getLogger('__main__').getChild(__name__)

SODA_GRAMMAR = '''
SodaProgram:
(
    ('burst' 'width' ':' burst_width=INT)
    ('dram' 'bank' ':' dram_bank=INT)
    ('dram' 'separate' ':' dram_separate=YesOrNo)
    ('unroll' 'factor' ':' unroll_factor=INT)
    ('kernel' ':' app_name=ID)
    ('border' ':' border=BorderStrategies)
    ('iterate' ':' iterate=INT)
    ('cluster' ':' cluster=ClusterStrategies)
    input=Input
    output=Output
    (extra_params=ExtraParam)*
    (locals=Local)*
    Comment*
)#;
Bin: /0[Bb][01]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
BorderStrategies: 'ignore'|'preserve';
ClusterStrategies: 'none'|'fine'|'coarse'|'full';
Comment: /\s*#.*$/;
Dec: /\d+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Expression: operand=Term (operator=PlusOrMinus operand=Term)*;
ExtraParam: 'param' type=Type (',' attrs=ExtraParamAttr)*  ':' name=ID ('[' size=INT ']')+;
ExtraParamAttr: 'dup' dup=Number | partitioning=Partitioning;
Factor: (sign=PlusOrMinus)? operand=Operand;
Float: /(((\d*\.\d+|\d+\.)([+-]?[Ee]\d+)?)|(\d+[+-]?[Ee]\d+))[FfLl]?/;
Func: name=FuncName '(' operand=Expression (',' operand=Expression)* ')';
FuncName: 'cos'|'sin'|'tan'|'acos'|'asin'|'atan'|'atan2'|
    'cosh'|'sinh'|'tanh'|'acosh'|'asinh'|'atanh'|
    'exp'|'frexp'|'ldexp'|'log'|'log10'|'modf'|'exp2'|'expm1'|'ilogb'|'log1p'|'log2'|'logb'|'scalbn'|'scalbln'|
    'pow'|'sqrt'|'cbrt'|'hypot'|
    'erf'|'erfc'|'tgamma'|'lgamma'|
    'ceil'|'floor'|'fmod'|'trunc'|'round'|'lround'|'llround'|'rint'|'lrint'|'llrint'|'nearbyint'|'remainder'|'remquo'|
    'copysign'|'nan'|'nextafter'|'nexttoward'|'fdim'|'fmax'|'fmin'|'fabs'|'abs'|'fma';
Hex: /0[Xx][0-9a-fA-F]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Input: 'input' type=Type ':' name=ID ('[' chan=Integer ']')? '(' tile_size=INT ',' (tile_size=INT ',')* ')';
Integer: ('+'|'-')?(Hex|Bin|Oct|Dec);
Local: 'local' type=Type ':' (expr=StageExpr)+;
MulOrDiv: '*'|'/'|'%';
Number: Float|Integer;
Oct: /0[0-7]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Operand: func=Func | name=ID ('[' chan=Integer ']')? '(' idx=INT (',' idx=INT)* ')' | num=Number | '(' expr=Expression ')';
Output: 'output' type=Type ':' (expr=StageExpr)+;
Partitioning: 'partition' partition_type='complete' ('dim' '=' dim=Number)? | 'partition' partition_type='cyclic' 'factor' '=' factor=Number ('dim' '=' dim=Number)?;
PlusOrMinus: '+'|'-';
StageExpr: name=ID ('[' chan=Integer ']')? '(' idx=INT (',' idx=INT)* ')'
    ('~' depth=Integer)? '=' expr=Expression;
Term: operand=Factor (operator=MulOrDiv operand=Factor)*;
Type: 'int8'|'int16'|'int32'|'int64'|'uint8'|'uint16'|'uint32'|'uint64'|'float'|'double';
YesOrNo: 'yes'|'no';
'''

def string_to_integer(s, none_val=None):
    if s is None:
        return none_val
    if s[0:2] == '0x' or s[0:2] == '0X':
        return int(s, 16)
    if s[0:2] == '0b' or s[0:2] == '0B':
        return int(s, 2)
    if s[0] == '0':
        return int(s, 8)
    return int(s)

Load = namedtuple('Load', ['name', 'chan', 'idx'])

def get_result_type(operand1, operand2, operator):
    for t in ('double', 'float') + sum([('int%d_t'%w, 'uint%d_t'%w) for w in (64, 32, 16, 8)], tuple()):
        if t in (operand1, operand2):
            return t
    raise SemanticError('cannot parse type: %s %s %s' % (operand1, operator, operand2))

class SodaProgram(object):
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
        self.locals = kwargs.pop('locals')
        self.unroll_factor = kwargs.pop('unroll_factor')
        self.dram_separate = kwargs.pop('dram_separate')=='yes'
        self.iterate = kwargs.pop('iterate')
        self.border = kwargs.pop('border')
        self.cluster = kwargs.pop('cluster')

        # normalize
        self.output.normalize(self.extra_params)
        for local in self.locals:
            local.normalize(self.extra_params)

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

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        last_operand = next(iter(self.operand))
        last_var = last_operand.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
        last_type = last_operand.get_type(tensors)
        for this_operand, operator in zip(self.operand[1:], self.operator):
            this_type = get_result_type(last_type, this_operand.get_type(tensors), operator)
            this_var = this_operand.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
            if add_latency:
                printer.println('%s %s[1];' % (this_type, printer.new_var()))
                new_var = printer.last_var()
                printer.println('#pragma HLS resource variable=%s latency=1 core=RAM_2P_LUTRAM' % new_var, 0)
                printer.do_scope()
                printer.println('#pragma HLS latency min=1', 0)
                printer.println('%s[0] = %s %s %s;' % (new_var, last_var, operator, this_var))
                printer.un_scope()
                printer.println('%s %s = %s[0];' % (this_type, printer.new_var(), new_var))
            else:
                printer.println('%s %s = %s %s %s;' % (this_type, printer.new_var(), last_var, operator, this_var))
            last_operand = this_operand
            last_var = printer.last_var()
            last_type = this_type
        return printer.last_var()

    def get_type(self, tensors):
        if not hasattr(self, 'type'):
            last_operand = next(iter(self.operand))
            last_type = last_operand.get_type(tensors)
            for this_operand, operator in zip(self.operand[1:], self.operator):
                this_type = get_result_type(last_type, this_operand.get_type(tensors), operator)
                last_operand = this_operand
                last_type = this_type
            self.type = last_type
        return self.type

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([op.get_loads() for op in self.operand], [])
        return self.loads

    def normalize(self, norm_offset, extra_params):
        for op in self.operand:
            op.normalize(norm_offset, extra_params)
        del self.loads

    def mutate_load(self, cb):
        for op in self.operand:
            op.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

class Term(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.operator = kwargs.pop('operator')

    def __str__(self):
        return '%s%s' % \
            (''.join([
                str(operand)+' '+str(operator)+' ' for operand, operator
                    in zip(self.operand, self.operator)]), str(self.operand[-1]))

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        last_operand = next(iter(self.operand))
        last_var = last_operand.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
        last_type = last_operand.get_type(tensors)
        for this_operand, operator in zip(self.operand[1:], self.operator):
            this_type = get_result_type(last_type, this_operand.get_type(tensors), operator)
            this_var = this_operand.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
            if add_latency:
                printer.println('%s %s[1];' % (this_type, printer.new_var()))
                new_var = printer.last_var()
                printer.println('#pragma HLS resource variable=%s latency=1 core=RAM_2P_LUTRAM' % new_var, 0)
                printer.do_scope()
                printer.println('#pragma HLS latency min=10', 0)
                printer.println('%s[0] = %s %s %s;' % (new_var, last_var, operator, this_var))
                printer.un_scope()
                printer.println('%s %s = %s[0];' % (this_type, printer.new_var(), new_var))
            else:
                printer.println('%s %s = %s %s %s;' % (this_type, printer.new_var(), last_var, operator, this_var))
            last_operand = this_operand
            last_var = printer.last_var()
            last_type = this_type
        return printer.last_var()

    def get_type(self, tensors):
        if not hasattr(self, 'type'):
            last_operand = next(iter(self.operand))
            last_type = last_operand.get_type(tensors)
            for this_operand, operator in zip(self.operand[1:], self.operator):
                this_type = get_result_type(last_type, this_operand.get_type(tensors), operator)
                last_operand = this_operand
                last_type = this_type
            self.type = last_type
        return self.type

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([op.get_loads() for op in self.operand], [])
        return self.loads

    def normalize(self, norm_offset, extra_params):
        for op in self.operand:
            op.normalize(norm_offset, extra_params)
        del self.loads

    def mutate_load(self, cb):
        for op in self.operand:
            op.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

class Factor(object):
    def __init__(self, **kwargs):
        self.operand = kwargs.pop('operand')
        self.sign = kwargs.pop('sign')

    def __str__(self):
        return ('-%s' if self.sign=='-' else '%s') % str(self.operand)

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        this_var = self.operand.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
        printer.println('%s %s = %s%s;' % (self.get_type(tensors), printer.new_var(), '-' if self.sign=='-' else '', this_var))
        return printer.last_var()

    def get_type(self, tensors):
        if not hasattr(self, 'type'):
            self.type = self.operand.get_type(tensors)
        return self.type

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = self.operand.get_loads()
        return self.loads

    def normalize(self, norm_offset, extra_params):
        self.operand.normalize(norm_offset, extra_params)
        del self.loads

    def mutate_load(self, cb):
        self.operand.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

class Func(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.operand = kwargs.pop('operand')

    def __str__(self):
        return '%s(%s)' % (self.name, ', '.join(str(op) for op in self.operand))

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        if add_latency:
            operand_vars = [op.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency) for op in self.operand]
            printer.println('%s %s[1];' % (self.get_type(tensors), printer.new_var()))
            new_var = printer.last_var()
            printer.println('#pragma HLS resource variable=%s latency=1 core=RAM_2P_LUTRAM' % new_var, 0)
            printer.do_scope()
            printer.println('#pragma HLS latency min=1', 0)
            printer.println('%s[0] = %s(%s);' % (new_var, self.name, ', '.join(operand_vars)))
            printer.un_scope()
            printer.println('%s %s = %s[0];' % (self.get_type(tensors), printer.new_var(), new_var))
            return printer.last_var()
        else:
            return '%s(%s)' % (self.name, ', '.join(op.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency) for op in self.operand))

    def get_type(self, tensors):
        if not hasattr(self, 'type'):
            if self.name in ('sqrt',):   # TODO: complete function type mapping
                self.type = next(iter(self.operand)).get_type(tensors)
            else:
                raise SemanticError('cannot get result type of function %s' % self.name)
        return self.type

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([op.get_loads() for op in self.operand], [])
        return self.loads

    def normalize(self, norm_offset, extra_params):
        for op in self.operand:
            op.normalize(norm_offset, extra_params)
        del self.loads

    def mutate_load(self, cb):
        for op in self.operand:
            op.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

class Operand(object):
    def __init__(self, **kwargs):
        self.func = kwargs.pop('func')
        self.name = kwargs.pop('name')
        self.chan = string_to_integer(kwargs.pop('chan'), 0)
        self.idx = tuple(kwargs.pop('idx'))
        self.num = kwargs.pop('num')
        self.expr = kwargs.pop('expr')

    def __str__(self):
        if self.func:
            return str(self.func)
        if self.name:
            return '%s[%d](%s)' % (self.name, self.chan, ', '.join(map(str, self.idx)))
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % str(self.expr)

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        if self.func:
            return self.func.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)
        if self.name:
            return LoadPrinter(self)
        if self.num:
            return str(self.num)
        if self.expr:
            return '(%s)' % self.expr.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)

    def get_type(self, tensors):
        if not hasattr(self, 'type'):
            if self.func:
                self.type = self.func.get_type(tensors)
            elif self.name:
                self.type = tensors[self.name].type
            elif self.num:
                if '.' in self.num or self.num[-1] in 'Ff':
                    if self.num[-1] in 'Ff':
                        self.type = 'float'
                    else:
                        self.type = 'double'
                else:
                    if self.num[0] == '+' or 'u' in self.num or 'U' in self.num:
                        self.type = 'u'
                        width = 2**math.ceil(math.log2(math.log2(string_to_integer(self.num))))
                    else:
                        self.type = ''
                        width = 2**math.ceil(math.log2(math.log2(2*math.fabs(string_to_integer(self.num)))))
                    width = 8 if width < 8 else 64 if width > 64 else width
                    self.type += 'int%d_t' % width
            elif self.expr:
                self.type = self.expr.get_type(tensors)
            else:
                raise SemanticError('invalid Operand %s' % str(self))
        return self.type

    def get_loads(self):
        if not hasattr(self, 'loads'):
            if self.func:
                self.loads = self.func.get_loads()
            elif self.expr:
                self.loads = self.expr.get_loads()
            elif self.num:
                self.loads = []
            elif self.name:
                logger.debug('load at %s[%d](%s)' % (self.name, self.chan, ', '.join(map(str, self.idx))))
                self.loads = [Load(self.name, self.chan, self.idx)]
            else:
                raise SemanticError('invalid Operand %s' % str(self))
        return self.loads

    def normalize(self, norm_offset, extra_params):
        if self.expr:
            self.expr.normalize(norm_offset, extra_params)
            del self.loads
        elif self.num:
            pass
        elif self.name:
            msg = '%s[%d](%s)' % (self.name, self.chan, ', '.join(map(str, self.idx)))
            self.idx = tuple(x-o for x, o in zip(self.idx, norm_offset))
            logger.debug('load at %s normalized to %s[%d](%s)' % (msg, self.name, self.chan, ', '.join(map(str, self.idx))))
            del self.loads

    def mutate_load(self, cb):
        if self.expr:
            self.expr.mutate_load(cb)
        elif self.num:
            pass
        elif self.name:
            self.name = cb(self.name)
        if hasattr(self, 'loads'):
            del self.loads

class Input(object):
    def __init__(self, **kwargs):
        self.type = core.get_c_type(kwargs.pop('type'))
        self.name = kwargs.pop('name')
        self.chan = string_to_integer(kwargs.pop('chan'), 1)
        if(self.chan<1):
            raise SemanticError('input %s has 0 channels' % self.name)
        self.tile_size = kwargs.pop('tile_size')+[0]

    def __str__(self):
        return ('input %s: %s[%d](%s)' % (self.type, self.name, self.chan, ', '.join([str(x) for x in self.tile_size[0:-1]] + [''])))

class ExtraParam(object):
    def __init__(self, **kwargs):
        self.type = core.get_c_type(kwargs.pop('type'))
        self.name = kwargs.pop('name')
        self.size = kwargs.pop('size')
        attrs = kwargs.pop('attrs')
        self.dup = None
        self.partitioning = []
        for attr in attrs:
            if attr.dup is not None:
                if self.dup is not None:
                    warn_msg = 'parameter duplication factor redefined as %d, previously defined as %d' % (string_to_integer(attr.dup), self.dup)
                    raise SemanticWarn(warn_msg)
                self.dup = string_to_integer(attr.dup)
            if attr.partitioning is not None:
                attr.partitioning.dim = string_to_integer(attr.partitioning.dim)
                attr.partitioning.factor = string_to_integer(attr.partitioning.factor)
                self.partitioning.append(attr.partitioning)

    def __str__(self):
        return ('param %s: %s[%s]%s%s' % (
            self.type,
            self.name,
            ']['.join(map(str, self.size)),
            '' if self.dup is None else (', dup x%d' % self.dup),
            ''.join([', %s partition%s%s' % (
                x.partition_type,
                '' if x.dim is None else ' in dim %d' % x.dim,
                '' if x.factor is None else' with factor=%d' % x.factor
            ) for x in self.partitioning]) if self.partitioning else ''
        ))

class Output(object):
    def __init__(self, **kwargs):
        self.type = core.get_c_type(kwargs.pop('type'))
        self.expr = kwargs.pop('expr')
        for e in self.expr:
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
        self.border = None

    def __str__(self):
        return ('output %s: %s' % (self.type, self.expr))

    def preserve_border_from(self, node_name):
        self.preserve_border = node_name
        self.border = ('preserve', node_name)

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        for e in self.expr:
            e.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([e.get_loads() for e in self.expr], [])
        return self.loads

    def normalize(self, extra_params):
        dim = range(len(next(iter(self.expr)).idx))
        norm_offset = tuple(min(min(load.idx[d]-e.idx[d] for load in e.get_loads() if load.name not in extra_params) for e in self.expr) for d in dim)
        for e in self.expr:
            e.normalize(tuple(norm_offset[d]+e.idx[d] for d in dim), extra_params)

    def mutate_load(self, cb):
        for e in self.expr:
            e.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

    def mutate_store(self, cb):
        self.name = cb(self.name)
        for e in self.expr:
            e.name = cb(e.name)
        if hasattr(self, 'loads'):
            del self.loads

class Local(object):
    def __init__(self, **kwargs):
        if 'output_node' in kwargs:
            output_node = kwargs.pop('output_node')
            self.name = output_node.name
            self.type = output_node.type
            self.chan = output_node.chan
            self.border = output_node.border
            self.expr = copy.deepcopy(output_node.expr)
            return
        self.type = core.get_c_type(kwargs.pop('type'))
        self.expr = kwargs.pop('expr')
        for e in self.expr:
            if hasattr(self, 'name'):
                if self.name != e.name:
                    err_msg = 'local had name %s but now renamed to %s' % (self.name, e.name)
                    raise SemanticError(err_msg)
            else:
                self.name = e.name
                logger.debug('local named as %s' % e.name)
            if hasattr(self, 'chan'):
                if e.chan in self.chan:
                    logger.warn('local channel %d redefined' % e.chan)
                self.chan |= {e.chan}
            else:
                self.chan = {e.chan}
        if self.chan != set(range(len(self.chan))):
            err_msg = ('local channel poorly-defined: %s' % str(self.chan))
            raise SemanticError(err_msg)
        self.chan = len(self.chan)
        self.border = None

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join('%s = %s' % (k, v) for k, v in self.__dict__.items() if k[0]!='_'))

    def preserve_border_from(self, node_name):
        self.preserve_border = node_name
        self.border = ('preserve', node_name)

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        for e in self.expr:
            e.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = sum([e.get_loads() for e in self.expr], [])
        return self.loads

    def normalize(self, extra_params):
        dim = range(len(next(iter(self.expr)).idx))
        norm_offset = tuple(min(min(load.idx[d]-e.idx[d] for load in e.get_loads() if load.name not in extra_params) for e in self.expr) for d in dim)
        for e in self.expr:
            e.normalize(tuple(norm_offset[d]+e.idx[d] for d in dim), extra_params)
        if hasattr(self, 'loads'):
            del self.loads

    def mutate_load(self, cb):
        for e in self.expr:
            e.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

    def mutate_store(self, cb):
        self.name = cb(self.name)
        for e in self.expr:
            e.name = cb(e.name)
        if hasattr(self, 'loads'):
            del self.loads

class StageExpr(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.chan = string_to_integer(kwargs.pop('chan'), 0)
        self.idx = tuple(kwargs.pop('idx'))
        self.expr = kwargs.pop('expr')
        self.depth = string_to_integer(kwargs.pop('depth'))
        logger.debug('store at %s[%d](%s)%s' % (self.name, self.chan,
            ', '.join(map(str, self.idx)),
            '' if self.depth is None else ' with depth %d' % self.depth))

    def __str__(self):
        return ('%s[%d](%s) = %s' % (self.name, self.chan, ', '.join(map(str, self.idx)), self.expr))

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join('%s = %s' % (k, v) for k, v in self.__dict__.items() if k[0]!='_'))

    def print_code(self, printer, tensors, LoadPrinter, StorePrinter, add_latency=False):
        printer.println('%s = %s;' % (StorePrinter(self), self.expr.print_code(printer, tensors, LoadPrinter, StorePrinter, add_latency)))

    def get_loads(self):
        if not hasattr(self, 'loads'):
            self.loads = self.expr.get_loads()
        return self.loads

    def normalize(self, norm_offset, extra_params):
        logger.debug('norm offset of %s[%d]: %s' % (self.name, self.chan, norm_offset))
        self.idx = tuple(x-o for x, o in zip(self.idx, norm_offset))
        self.expr.normalize(norm_offset, extra_params)
        del self.loads

    def mutate_load(self, cb):
        self.expr.mutate_load(cb)
        if hasattr(self, 'loads'):
            del self.loads

SODA_GRAMMAR_CLASSES = [
    Expression,
    ExtraParam,
    Factor,
    Func,
    Input,
    Local,
    Operand,
    Output,
    SodaProgram,
    StageExpr,
    Term
]

