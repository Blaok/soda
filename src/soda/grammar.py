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
  (input_stmts=InputStmt)+
  (param_stmts=ParamStmt)*
  (local_stmts=LocalStmt)*
  (output_stmts=OutputStmt)+
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

InputStmt: 'input' soda_type=Type ':' name=ID ('(' (tile_size=INT ',')* ')')?;
LocalStmt: 'local' soda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;
OutputStmt: 'output' soda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;

Let: (soda_type=Type)? name=ID '=' expr=Expr;
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

Operand: cast=Cast | call=Call | ref=Ref | num=Num | var=Var | '(' expr=Expr ')';
Cast: soda_type=Type '(' expr=Expr ')';
Call: name=FuncName '(' arg=Expr (',' arg=Expr)* ')';
Var: name=ID ('[' idx=Int ']')*;

ParamStmt: 'param' soda_type=Type (',' attr=ParamAttr)* ':' name=ID ('[' size=INT ']')*;
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

class InputStmt(_Node):
  def __str__(self):
    result = 'input {}: {}'.format(self.soda_type, self.name)
    if self.tile_size:
      result += '({},)'.format(', '.join(map(str, self.tile_size)))
    return result

class _LocalStmtOrOutputStmt(_Node):
  def __str__(self):
    if self.let:
      let = '\n  {}\n '.format('\n  '.join(map(str, self.let)))
    else:
      let = ''
    return '{} {}:{} {} = {}'.format(
      type(self).__name__[:-4].lower(), self.soda_type, let, self.ref, self.expr)

class LocalStmt(_LocalStmtOrOutputStmt):
  pass

class OutputStmt(_LocalStmtOrOutputStmt):
  pass

class Let(_Node):
  def __str__(self):
    result = '{} = {}'.format(self.name, self.expr)
    if self.soda_type is not None:
      result = '{} {}'.format(self.soda_type, result)
    return result

class Ref(_Node):
  def __str__(self):
    result = '{}({})'.format(self.name, ', '.join(map(str, self.idx)))
    if self.lat is not None:
      result += ' ~{}'.format(self.lat)
    return result

class _BinaryOp(_Node):
  def __str__(self):
    result = str(self.operand[0])
    for operator, operand in zip(self.operator, self.operand[1:]):
      result += ' {} {}'.format(operator, operand)
    return result

class Expr(_BinaryOp):
  pass

class LogicAnd(_BinaryOp):
  pass

class BinaryOr(_BinaryOp):
  pass

class Xor(_BinaryOp):
  pass

class BinaryAnd(_BinaryOp):
  pass

class EqCmp(_BinaryOp):
  pass

class LtCmp(_BinaryOp):
  pass

class AddSub(_BinaryOp):
  pass

class MulDiv(_BinaryOp):
  pass

class Unary(_Node):
  def __str__(self):
    return ''.join(self.operator)+str(self.operand)

class Operand(_Node):
  def __str__(self):
    for attr in ('cast', 'call', 'ref', 'num', 'var'):
      if getattr(self, attr) is not None:
        return str(getattr(self, attr))
    else:
      return '(%s)' % str(self.expr)

class Cast(_Node):
  def __str__(self):
    return '{}({})'.format(self.soda_type, self.expr)

class Call(_Node):
  def __str__(self):
    return '{}({})'.format(self.name, ', '.join(map(str, self.arg)))

class Var(_Node):
  def __str__(self):
    return self.name+''.join(map('[{}]'.format, self.idx))

class ParamStmt(_Node):
  def __str__(self):
    return 'param {}{}: {}{}'.format(
      self.soda_type, ''.join(map(', {}'.format, self.attr)),
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
      '\n'.join(map(str, self.input_stmts)),
      '\n'.join(map(str, self.param_stmts)),
      '\n'.join(map(str, self.local_stmts)),
      '\n'.join(map(str, self.output_stmts))))

SODA_GRAMMAR_CLASSES = [
  InputStmt,
  LocalStmt,
  OutputStmt,
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
  Operand,
  Cast,
  Call,
  Var,
  ParamStmt,
  ParamAttr,
  SodaProgram,
]
