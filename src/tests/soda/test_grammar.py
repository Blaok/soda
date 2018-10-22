import unittest

import textx
from textx.exceptions import TextXSyntaxError

from soda import grammar

class TestGrammar(unittest.TestCase):

  def setUp(self):
    self.ref = grammar.Ref(name='foo', idx=(0, 23), lat=None)
    self.expr_ref = grammar.Ref(name='bar', idx=(233, 42), lat=None)
    self.expr_ref.soda_type = 'int8'
    self.expr = grammar.Expr(operand=(self.expr_ref,), operator=())
    self.let_ref = grammar.Ref(name='bar_l', idx=(42, 2333), lat=None)
    self.let_expr = grammar.Expr(operand=(self.let_ref,), operator=())
    self.let = grammar.Let(soda_type='int8', name='foo_l', expr=self.let_expr)
    self.let_ref2 = grammar.Ref(name='bar_l2', idx=(0, 42), lat=None)
    self.let_expr2 = grammar.Expr(operand=(self.let_ref2,), operator=())
    self.let2 = grammar.Let(soda_type='int8', name='foo_l2',
                            expr=self.let_expr2)

  def test_syntax(self):
    soda_mm = textx.metamodel_from_str(
      grammar.SODA_GRAMMAR,
      classes=grammar.SODA_GRAMMAR_CLASSES)
    try:
      soda_program_str = \
r'''
border: ignore
burst width: 512
cluster: none
iterate: 2
kernel: name
unroll factor: 1
input float: bbb
input uint6: a(233, *)
param int8: p0
param int9, dup 3: p1[23]
param int10, partition complete: p2[23]
param int11, partition complete dim=1: p2[23]
param int12, partition cyclic factor=23: p3[233]
param int13, partition cyclic factor=23 dim=2: p4[233][233]
param int14, partition complete, dup 3: p5[23]
local int27:
  int32 l = int32(a(0, 0) ~1 + b(1, 0))
  int32 g = int32(a(0, 0) ~1 + p0 + p1[1][3])
  c(0, 0) ~3 = +-+-l * --+~l
output double:
  l = float18_3(c(0, 1) ~5) + a(1, 0)
  d(0, 0) = sqrt(float15(l <= l / 2))
output double:
  l = float18_3(c(0, 1) ~5) + a(1, 0)
  e(0, 0) = float15(l + l / 2)
'''.strip('\n')
      soda_program = soda_mm.model_from_str(soda_program_str)
      self.maxDiff = None
      self.assertEqual(str(soda_program),
                       soda_program_str.replace('  l = ', '  float18_3 l = '))
      return
    except TextXSyntaxError as e:
      msg = str(e)
    self.fail(msg)

  def test_input(self):
    self.assertEqual(str(grammar.InputStmt(soda_type='int8', name='foo',
                                           tile_size=[], dram=())),
                     'input int8: foo')
    self.assertEqual(str(grammar.InputStmt(soda_type='int8', name='foo',
                                           tile_size=[23], dram=())),
                     'input int8: foo(23, *)')
    self.assertEqual(str(grammar.InputStmt(soda_type='int8', name='foo',
                                           tile_size=[23, 233], dram=())),
                     'input int8: foo(23, 233, *)')

  def test_local(self):
    self.assertEqual(
      str(grammar.LocalStmt(soda_type='int8', let=[],
                ref=self.ref, expr=self.expr, dram=())),
      'local int8: foo(0, 23) = bar(233, 42)')
    self.assertEqual(
      str(grammar.LocalStmt(soda_type='int8', let=[self.let],
                ref=self.ref, expr=self.expr, dram=())),
      'local int8:\n  int8 foo_l = bar_l(42, 2333)\n'
      '  foo(0, 23) = bar(233, 42)')
    self.assertEqual(
      str(grammar.LocalStmt(soda_type='int8', let=[self.let, self.let2],
                ref=self.ref, expr=self.expr, dram=())),
      'local int8:\n  int8 foo_l = bar_l(42, 2333)\n'
      '  int8 foo_l2 = bar_l2(0, 42)\n  foo(0, 23) = bar(233, 42)')

  def test_output(self):
    self.assertEqual(
      str(grammar.OutputStmt(soda_type='int8', let=[],
                 ref=self.ref, expr=self.expr, dram=())),
      'output int8: foo(0, 23) = bar(233, 42)')
    self.assertEqual(
      str(grammar.OutputStmt(soda_type='int8', let=[self.let],
                 ref=self.ref, expr=self.expr, dram=())),
      'output int8:\n  int8 foo_l = bar_l(42, 2333)\n'
      '  foo(0, 23) = bar(233, 42)')
    self.assertEqual(
      str(grammar.OutputStmt(soda_type='int8', let=[self.let, self.let2],
                 ref=self.ref, expr=self.expr, dram=())),
      'output int8:\n  int8 foo_l = bar_l(42, 2333)\n'
      '  int8 foo_l2 = bar_l2(0, 42)\n  foo(0, 23) = bar(233, 42)')

  def test_let(self):
    self.assertEqual(str(grammar.Let(soda_type='int8', name='foo',
                                     expr=self.expr)),
                     'int8 foo = bar(233, 42)')
    self.assertEqual(str(grammar.Let(soda_type=None, name='foo',
                                     expr=self.expr)),
                     'int8 foo = bar(233, 42)')

  def test_ref(self):
    self.assertEqual(str(grammar.Ref(name='foo', idx=[0], lat=None)),
             'foo(0)')
    self.assertEqual(str(grammar.Ref(name='foo', idx=[0], lat=233)),
             'foo(0) ~233')
    self.assertEqual(str(grammar.Ref(name='foo', idx=[0, 23], lat=233)),
             'foo(0, 23) ~233')

  def test_binary_operations(self):
    for operand, operators in [
        (grammar.Expr, ('||',)),
        (grammar.LogicAnd, ('&&',)),
        (grammar.BinaryOr, ('|',)),
        (grammar.Xor, ('^',)),
        (grammar.BinaryAnd, ('&',)),
        (grammar.EqCmp, ('==', '!=')),
        (grammar.LtCmp, ('<=', '>=', '<', '>')),
        (grammar.AddSub, ('+', '-')),
        (grammar.MulDiv, ('*', '/', '%'))]:
      self.assertEqual(
        str(operand(operand=list(map('op{}'.format,
                       range(len(operators)+1))),
              operator=operators)),
        'op0'+''.join(' {} op{}'.format(op, idx+1)
                for idx, op in enumerate(operators)))

  def test_unary(self):
    self.assertEqual(str(grammar.Unary(operator='+-~!'.split(),
                       operand='op')),
             '+-~!op')

  def test_cast(self):
    self.assertEqual(str(grammar.Cast(soda_type='int8', expr='expr')),
             'int8(expr)')

  def test_call(self):
    self.assertEqual(str(grammar.Call(name='pi', arg=[])),
             'pi()')
    self.assertEqual(str(grammar.Call(name='sqrt', arg=['arg'])),
             'sqrt(arg)')
    self.assertEqual(str(grammar.Call(
      name='select', arg=['condition', 'true_val', 'false_val'])),
             'select(condition, true_val, false_val)')

  def test_var(self):
    self.assertEqual(str(grammar.Var(name='foo', idx=[])), 'foo')
    self.assertEqual(str(grammar.Var(name='foo', idx=[0])), 'foo[0]')
    self.assertEqual(str(grammar.Var(name='foo', idx=[0, 1])), 'foo[0][1]')

if __name__ == '__main__':
  unittest.main()
