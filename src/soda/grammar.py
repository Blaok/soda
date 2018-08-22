import copy
import logging

from soda import util

_logger = logging.getLogger('__main__').getChild(__name__)

SODA_GRAMMAR = r'''
SodaProgram:
(
  ('border' ':' border=BorderStrategies)
  ('burst' 'width' ':' burst_width=INT)
  ('cluster' ':' cluster=ClusterStrategies)
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

InputStmt: 'input' ('dram' dram=INT ('|' dram=INT)*)? soda_type=Type ':' name=ID ('(' (tile_size=INT ',')* ')')?;
LocalStmt: 'local' soda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;
OutputStmt: 'output' ('dram' dram=INT ('|' dram=INT)*)? soda_type=Type ':' (let=Let)* ref=Ref '=' expr=Expr;

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

ParamStmt: 'param' ('dram' dram=INT ('|' dram=INT)*)? soda_type=Type (',' attr=ParamAttr)* ':' name=ID ('[' size=INT ']')*;
ParamAttr: 'dup' dup=Int | partitioning=Partitioning;
Partitioning:
  'partition' strategy='complete' ('dim' '=' dim=Int)? |
  'partition' strategy='cyclic' 'factor' '=' factor=Int ('dim' '=' dim=Int)?;

Type: FixedType | FloatType;
FixedType: /u?int[1-9]\d*(_[1-9]\d*)?/;
FloatType: /float[1-9]\d*(_[1-9]\d*)?/ | 'float' | 'double' | 'half';
'''

class Node(object):
  """A immutable, hashable IR node.
  """
  SCALAR_ATTRS = ()
  LINEAR_ATTRS = ()

  @property
  def ATTRS(self):
    return self.SCALAR_ATTRS + self.LINEAR_ATTRS

  def __init__(self, **kwargs):
    for attr in self.SCALAR_ATTRS:
      setattr(self, attr, kwargs.pop(attr))
    for attr in self.LINEAR_ATTRS:
      setattr(self, attr, tuple(kwargs.pop(attr)))

  def __hash__(self):
    return hash((tuple(getattr(self, _) for _ in self.SCALAR_ATTRS),
                 tuple(tuple(getattr(self, _)) for _ in self.LINEAR_ATTRS)))

  def __eq__(self, other):
    return all(getattr(self, attr) == getattr(other, attr)
               for attr in self.ATTRS)

  @property
  def c_type(self):
    return util.get_c_type(self.soda_type)

  def visit(self, callback, args=None):
    """A general-purpose, flexible, and powerful visitor.

    The args parameter will be passed to the callback callable so that it may
    read or write any information from or to the caller.

    A copy of self will be made and passed to the callback to avoid destructive
    access.

    If a new object is returned by the callback, it will be returned directly
    without recursion.

    If the same object is returned by the callback, if any attribute is
    changed, it will not be recursively visited. If an attribute is unchanged,
    it will be recursively visited.
    """

    self_copy = copy.copy(self)
    obj = callback(self_copy, args)
    self_copy = copy.copy(self)
    scalar_attrs = {attr: getattr(self_copy, attr).visit(callback, args)
                    if issubclass(type(getattr(self_copy, attr)), Node)
                    else getattr(self_copy, attr)
                    for attr in self_copy.SCALAR_ATTRS}
    linear_attrs = {attr: tuple(_.visit(callback, args)
                                if issubclass(type(_), Node) else _
                                for _ in getattr(self_copy, attr))
                    for attr in self_copy.LINEAR_ATTRS}

    for attr in self.SCALAR_ATTRS:
      # old attribute may not exist in mutated object
      if not hasattr(obj, attr):
        continue
      if getattr(obj, attr) is getattr(self, attr):
        if issubclass(type(getattr(obj, attr)), Node):
          setattr(obj, attr, scalar_attrs[attr])
    for attr in self.LINEAR_ATTRS:
      # old attribute may not exist in mutated object
      if not hasattr(obj, attr):
        continue
      setattr(obj, attr, tuple(
          c if a is b and issubclass(type(a), Node) else a
          for a, b, c in zip(getattr(obj, attr), getattr(self, attr),
                             linear_attrs[attr])))
    return obj

class InputStmt(Node):
  """Node for input statement, represents a tiled input tensor.

  Attributes:
    soda_type: Type of this input tensor.
    dram: [int], dram id used to read this input
    name: str, name of this input tensor.
    tile_size: list of tile sizes. The last dimension should be 0.
  """
  SCALAR_ATTRS = 'soda_type', 'name'
  LINEAR_ATTRS = ('tile_size', 'dram',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # pylint: disable=access-member-before-definition
    if not self.dram:
      self.dram = (0,)
    self.tile_size += (0,)

  def __str__(self):
    result = 'input {}: {}'.format(self.soda_type, self.name)
    if self.tile_size[:-1]:
      result += '({},)'.format(', '.join(map(str, self.tile_size[:-1])))
    return result

class LocalStmtOrOutputStmt(Node):
  SCALAR_ATTRS = 'soda_type', 'ref', 'expr'
  LINEAR_ATTRS = ('let',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    var_types = {}
    # pylint: disable=access-member-before-definition
    for let in self.let:
      var_types[let.name] = let.soda_type
    def set_var_type(obj, var_types):
      if isinstance(obj, Var):
        obj.soda_type = var_types[obj.name]
      return obj
    self.let = tuple(_.visit(set_var_type, var_types) for _ in self.let)
    self.expr = self.expr.visit(set_var_type, var_types)

  @property
  def name(self):
    return self.ref.name

  def __str__(self):
    if self.let:
      let = '\n  {}\n '.format('\n  '.join(map(str, self.let)))
    else:
      let = ''
    return '{} {}:{} {} = {}'.format(type(self).__name__[:-4].lower(),
                                     self.soda_type, let, self.ref, self.expr)

class LocalStmt(LocalStmtOrOutputStmt):
  pass

class OutputStmt(LocalStmtOrOutputStmt):
  LINEAR_ATTRS = LocalStmtOrOutputStmt.LINEAR_ATTRS + ('dram',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # pylint: disable=access-member-before-definition
    if not self.dram:
      self.dram = (0,)

class Let(Node):
  SCALAR_ATTRS = 'soda_type', 'name', 'expr'
  def __str__(self):
    result = '{} = {}'.format(self.name, self.expr)
    if self.soda_type is not None:
      result = '{} {}'.format(self.soda_type, result)
    return result

  @property
  def c_expr(self):
    return 'const {} {} = {};'.format(self.c_type, self.name, self.expr.c_expr)

class Ref(Node):
  SCALAR_ATTRS = 'name', 'lat'
  LINEAR_ATTRS = ('idx',)
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.idx = tuple(self.idx)
    if self.lat is not None:
      self.lat = str2int(self.lat)

  def __str__(self):
    result = '{}({})'.format(self.name, ', '.join(map(str, self.idx)))
    if self.lat is not None:
      result += ' ~{}'.format(self.lat)
    return result

class _BinaryOp(Node):
  LINEAR_ATTRS = 'operand', 'operator'
  def __str__(self):
    result = str(self.operand[0])
    for operator, operand in zip(self.operator, self.operand[1:]):
      result += ' {} {}'.format(operator, operand)
    return result

  @property
  def soda_type(self):
  # TODO: derive from all operands
    return self.operand[0].soda_type

  @property
  def c_expr(self):
    result = self.operand[0].c_expr
    for operator, operand in zip(self.operator, self.operand[1:]):
      result += ' {} {}'.format(operator, operand.c_expr)
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

class Unary(Node):
  SCALAR_ATTRS = ('operand',)
  LINEAR_ATTRS = ('operator',)
  def __str__(self):
    return ''.join(self.operator)+str(self.operand)

  @property
  def soda_type(self):
    return self.operand.soda_type

  @property
  def c_expr(self):
    return ''.join(self.operator)+self.operand.c_expr

class Operand(Node):
  SCALAR_ATTRS = 'cast', 'call', 'ref', 'num', 'var', 'expr'
  def __str__(self):
    for attr in ('cast', 'call', 'ref', 'num', 'var'):
      if getattr(self, attr) is not None:
        return str(getattr(self, attr))
    # pylint: disable=useless-else-on-loop
    else:
      return '(%s)' % str(self.expr)

  @property
  def c_expr(self):
    for attr in ('cast', 'call', 'ref', 'num', 'var'):
      attr = getattr(self, attr)
      if attr is not None:
        if hasattr(attr, 'c_expr'):
          return attr.c_expr
        return str(attr)
    # pylint: disable=useless-else-on-loop
    else:
      return '(%s)' % self.expr.c_expr

  @property
  def soda_type(self):
    for attr in self.ATTRS:
      val = getattr(self, attr)
      if val is not None:
        if hasattr(val, 'soda_type'):
          return val.soda_type
        return None
    raise util.InternalError('undefined Operand')

class Cast(Node):
  SCALAR_ATTRS = 'soda_type', 'expr'
  def __str__(self):
    return '{}({})'.format(self.soda_type, self.expr)

  @property
  def c_expr(self):
    return 'static_cast<{} >({})'.format(self.c_type, self.expr.c_expr)

class Call(Node):
  SCALAR_ATTRS = ('name',)
  LINEAR_ATTRS = ('arg',)
  def __str__(self):
    return '{}({})'.format(self.name, ', '.join(map(str, self.arg)))

class Var(Node):
  SCALAR_ATTRS = ('name',)
  LINEAR_ATTRS = ('idx',)
  def __str__(self):
    return self.name+''.join(map('[{}]'.format, self.idx))

  @property
  def c_expr(self):
    return self.name+''.join(map('[{}]'.format, self.idx))

class ParamStmt(Node):
  SCALAR_ATTRS = 'soda_type', 'attr', 'name', 'size'
  def __str__(self):
    return 'param {}{}: {}{}'.format(
      self.soda_type, ''.join(map(', {}'.format, self.attr)),
      self.name, ''.join(map('[{}]'.format, self.size)))

class ParamAttr(Node):
  SCALAR_ATTRS = 'dup', 'partitioning'
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
  raise util.SemanticError('cannot parse type: %s %s %s' %
    (operand1, operator, operand2))

class SodaProgram(Node):
  SCALAR_ATTRS = ('border', 'burst_width', 'cluster', 'iterate', 'app_name',
                  'unroll_factor', 'input_stmts', 'param_stmts', 'local_stmts',
                  'output_stmts')
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    for node in self.input_stmts:
      if hasattr(self, 'tile_size'):
        # pylint: disable=access-member-before-definition
        if self.tile_size != node.tile_size:
          msg = ('tile size %s doesn\'t match previous one %s' %
               # pylint: disable=access-member-before-definition
               (node.tile_size, self.tile_size))
          raise util.SemanticError(msg)
      elif node.tile_size[:-1]:
        self.tile_size = node.tile_size
        self.dim = len(self.tile_size)
    # deal with 1D case
    if not hasattr(self, 'tile_size'):
      # pylint: disable=undefined-loop-variable
      self.tile_size = node.tile_size
      self.dim = len(self.tile_size)

  def __str__(self):
    return '\n'.join(filter(None, (
      'border: {}'.format(self.border),
      'burst width: {}'.format(self.burst_width),
      'cluster: {}'.format(self.cluster),
      'iterate: {}'.format(self.iterate),
      'kernel: {}'.format(self.app_name),
      'unroll factor: {}'.format(self.unroll_factor),
      '\n'.join(map(str, self.input_stmts)),
      '\n'.join(map(str, self.param_stmts)),
      '\n'.join(map(str, self.local_stmts)),
      '\n'.join(map(str, self.output_stmts)))))

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
