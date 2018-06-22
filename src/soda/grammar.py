import logging

from soda import core

logger = logging.getLogger('__main__').getChild(__name__)

SODA_GRAMMAR = r'''
SodaProgram:
(
    ('border' ':' border=BorderStrategies)
    ('burst' 'width' ':' burst_width=INT)
    ('cluster' ':' cluster=ClusterStrategies)
    ('dram' 'bank' ':' dram_bank=INT)
    ('dram' 'separate' ':' dram_separate=YesOrNo)
    ('iterate' ':' iterate=INT)
    ('kernel' ':' app_name=ID)
    ('unroll' 'factor' ':' unroll_factor=INT)
    (input=Input)+
    (param=Param)*
    (local=Local)*
    (output=Output)+
)#;

YesOrNo: 'yes'|'no';

Bin: /0[Bb][01]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Dec: /\d+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Oct: /0[0-7]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Hex: /0[Xx][0-9a-fA-F]+([Uu][Ll][Ll]?|[Ll]?[Ll]?[Uu]?)/;
Int: ('+'|'-')?(Hex|Bin|Oct|Dec);
Float: /(((\d*\.\d+|\d+\.)([+-]?[Ee]\d+)?)|(\d+[+-]?[Ee]\d+))[FfLl]?/;
Num: Float|Int;

BorderStrategies: 'ignore'|'preserve';
ClusterStrategies: 'none'|'fine'|'coarse'|'full';

Comment: /\s*#.*$/;

FuncName: 'cos'|'sin'|'tan'|'acos'|'asin'|'atan'|'atan2'|
    'cosh'|'sinh'|'tanh'|'acosh'|'asinh'|'atanh'|
    'exp'|'frexp'|'ldexp'|'log'|'log10'|'modf'|'exp2'|'expm1'|'ilogb'|'log1p'|'log2'|'logb'|'scalbn'|'scalbln'|
    'pow'|'sqrt'|'cbrt'|'hypot'|
    'erf'|'erfc'|'tgamma'|'lgamma'|
    'ceil'|'floor'|'fmod'|'trunc'|'round'|'lround'|'llround'|'rint'|'lrint'|'llrint'|'nearbyint'|'remainder'|'remquo'|
    'copysign'|'nan'|'nextafter'|'nexttoward'|'fdim'|'fmax'|'fmin'|'fabs'|'abs'|'fma'|
    'min'|'max'|'select';

Input: 'input' type=Type ':' name=ID ('(' tile_size=INT ',' (tile_size=INT ',')* ')')?;
Local: 'local' type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;
Output: 'output' type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;

Let: (type=Type)? name=ID '=' expr=Expr;
Ref: name=ID '(' idx=INT (',' idx=INT)* ')' ('~' lat=Int)?;

Expr: operand=LogicAnd (operator=LogicOrOp operand=LogicAnd)*;
LogicOrOp: '||';

LogicAnd: operand=BinaryOr (operator=LogicAndOp operand=BinaryOr)*;
LogicAndOp: '&&';

BinaryOr: operand=Xor (operator=BinaryOrOp operand=Xor)*;
BinaryOrOp: '|';

Xor: operand=BinaryAnd (operator=XorOp operand=BinaryAnd)*;
XorOp: '^';

BinaryAnd: operand=EqCmp (operator=BinaryAndOp operand=EqCmp)*;
BinaryAndOp: '&';

EqCmp: operand=LtCmp (operator=EqCmpOp operand=LtCmp)*;
EqCmpOp: '=='|'!=';

LtCmp: operand=AddSub (operator=LtCmpOp operand=AddSub)*;
LtCmpOp: '<='|'>='|'<'|'>';

AddSub: operand=MulDiv (operator=AddSubOp operand=MulDiv)*;
AddSubOp: '+'|'-';

MulDiv: operand=Unary (operator=MulDivOp operand=Unary)*;
MulDivOp: '*'|'/'|'%';

Unary: (operator=UnaryOp)* operand=Operand;
UnaryOp: '+'|'-'|'~'|'!';

Operand: Cast | Call | Ref | Num | Var | '(' Expr ')';
Cast: type=Type '(' expr=Expr ')';
Call: name=FuncName '(' arg=Expr (',' arg=Expr)* ')';
Var: name=ID ('[' idx=Int ']')*;

Param: 'param' type=Type (',' attr=ParamAttr)* ':' name=ID ('[' size=INT ']')*;
ParamAttr: 'dup' dup=Int | partitioning=Partitioning;
Partitioning:
    'partition' strategy='complete' ('dim' '=' dim=Int)? |
    'partition' strategy='cyclic' 'factor' '=' factor=Int ('dim' '=' dim=Int)?;

Type: FixedType | FloatType;
FixedType: /u?int[1-9]\d*(_[1-9]\d*)?/;
FloatType: /float[1-9]\d*(_[1-9]\d*)?/ | 'float' | 'double' | 'half';
'''

class _Node(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class Input(_Node):
    def __str__(self):
        result = 'input {}: {}'.format(self.type, self.name)
        if self.tile_size:
            result += '({},)'.format(', '.join(map(str, self.tile_size)))
        return result

class _LocalOrOutput(_Node):
    def __str__(self):
        if self.let:
            let = '\n  {}\n '.format('\n  '.join(map(str, self.let)))
        else:
            let = ''
        return '{} {}:{} {} = {}'.format(
            type(self).__name__.lower(), self.type, let, self.ref, self.expr)

class Local(_LocalOrOutput):
    pass

class Output(_LocalOrOutput):
    pass

class Let(_Node):
    def __str__(self):
        result = '{} = {}'.format(self.name, self.expr)
        if self.type is not None:
            result = '{} {}'.format(self.type, result)
        return result

class Ref(_Node):
    def __str__(self):
        result = '{}({})'.format(self.name, ', '.join(map(str, self.idx)))
        if self.lat is not None:
            result += ' ~{}'.format(self.lat)
        return result

class _BinaryOperand(_Node):
    def __str__(self):
        result = str(self.operand[0])
        for operator, operand in zip(self.operator, self.operand[1:]):
            result += ' {} {}'.format(operator, operand)
        return result

class Expr(_BinaryOperand):
    pass

class LogicAnd(_BinaryOperand):
    pass

class BinaryOr(_BinaryOperand):
    pass

class Xor(_BinaryOperand):
    pass

class BinaryAnd(_BinaryOperand):
    pass

class EqCmp(_BinaryOperand):
    pass

class LtCmp(_BinaryOperand):
    pass

class AddSub(_BinaryOperand):
    pass

class MulDiv(_BinaryOperand):
    pass

class Unary(_Node):
    def __str__(self):
        return ''.join(self.operator)+str(self.operand)

class Cast(_Node):
    def __str__(self):
        return '{}({})'.format(self.type, self.expr)

class Call(_Node):
    def __str__(self):
        return '{}({})'.format(self.name, ', '.join(map(str, self.arg)))

class Var(_Node):
    def __str__(self):
        return self.name+''.join(map('[{}]'.format, self.idx))

class Param(_Node):
    def __str__(self):
        return 'param {}{}: {}{}'.format(
            self.type, ''.join(map(', {}'.format, self.attr)),
            self.name, ''.join(map('[{}]'.format, self.size)))

class ParamAttr(_Node):
    def __str__(self):
        if self.dup is not None:
            return 'dup {}'.format(self.dup)
        result = 'partition {0.strategy}'.format(self.partitioning)
        if self.partitioning.strategy == 'cyclic':
            result += ' factor={}'.format(self.partitioning.factor)
        if self.partitioning.dim is not None:
            result += ' dim={}'.format(self.partitioning.dim)
        return result

def str2int(s, none_val=None):
    if s is None:
        return none_val
    while s[-1] in 'UuLl':
        s = s[:-1]
    if s[0:2] == '0x' or s[0:2] == '0X':
        return int(s, 16)
    if s[0:2] == '0b' or s[0:2] == '0B':
        return int(s, 2)
    if s[0] == '0':
        return int(s, 8)
    return int(s)

def get_result_type(operand1, operand2, operator):
    for t in ('double', 'float') + sum([('int%d_t'%w, 'uint%d_t'%w)
                                        for w in (64, 32, 16, 8)],
                                       tuple()):
        if t in (operand1, operand2):
            return t
    raise core.SemanticError('cannot parse type: %s %s %s' %
        (operand1, operator, operand2))

class SodaProgram(_Node):
    def __str__(self):
        return '\n'.join((
            'border: {}'.format(self.border),
            'burst width: {}'.format(self.burst_width),
            'cluster: {}'.format(self.cluster),
            'dram bank: {}'.format(self.dram_bank),
            'dram separate: {}'.format(self.dram_separate),
            'iterate: {}'.format(self.iterate),
            'kernel: {}'.format(self.app_name),
            'unroll factor: {}'.format(self.unroll_factor),
            '\n'.join(map(str, self.input)),
            '\n'.join(map(str, self.param)),
            '\n'.join(map(str, self.local)),
            '\n'.join(map(str, self.output))))

SODA_GRAMMAR_CLASSES = [
    Input,
    Local,
    Output,
    Let,
    Ref,
    Expr,
    LogicAnd,
    BinaryOr,
    Xor,
    BinaryAnd,
    EqCmp,
    LtCmp,
    AddSub,
    MulDiv,
    Unary,
    Cast,
    Call,
    Var,
    Param,
    ParamAttr,
    SodaProgram,
]
