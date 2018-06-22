import unittest

import textx
from textx.exceptions import TextXSyntaxError

from soda import grammar

class TestGrammar(unittest.TestCase):

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
dram bank: 1
dram separate: no
iterate: 2
kernel: name
unroll factor: 1
input float: bbb
input uint6: a(233,)
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
            self.assertEqual(str(soda_program), soda_program_str)
            return
        except TextXSyntaxError as e:
            msg = str(e)
        self.fail(msg)

    def test_input(self):
        self.assertEqual(
            str(grammar.Input(type='int8', name='foo', tile_size=[])),
            'input int8: foo')
        self.assertEqual(
            str(grammar.Input(type='int8', name='foo', tile_size=[23])),
            'input int8: foo(23,)')
        self.assertEqual(
            str(grammar.Input(type='int8', name='foo', tile_size=[23, 233])),
            'input int8: foo(23, 233,)')

    def test_local(self):
        self.assertEqual(
            str(grammar.Local(type='int8', let=[],
                              ref='foo(0, 23)', expr='bar')),
            'local int8: foo(0, 23) = bar')
        self.assertEqual(
            str(grammar.Local(type='int8', let=['let expr'],
                              ref='foo(0, 23)', expr='bar')),
            'local int8:\n  let expr\n  foo(0, 23) = bar')
        self.assertEqual(
            str(grammar.Local(type='int8', let=['let expr0', 'let expr1'],
                              ref='foo(0, 23)', expr='bar')),
            'local int8:\n  let expr0\n  let expr1\n  foo(0, 23) = bar')

    def test_output(self):
        self.assertEqual(
            str(grammar.Output(type='int8', let=[],
                               ref='foo(0, 23)', expr='bar')),
            'output int8: foo(0, 23) = bar')
        self.assertEqual(
            str(grammar.Output(type='int8', let=['let expr'],
                               ref='foo(0, 23)', expr='bar')),
            'output int8:\n  let expr\n  foo(0, 23) = bar')
        self.assertEqual(
            str(grammar.Output(type='int8', let=['let expr0', 'let expr1'],
                               ref='foo(0, 23)', expr='bar')),
            'output int8:\n  let expr0\n  let expr1\n  foo(0, 23) = bar')

    def test_let(self):
        self.assertEqual(str(grammar.Let(type='int8', name='foo', expr='bar')),
                         'int8 foo = bar')
        self.assertEqual(str(grammar.Let(type=None, name='foo', expr='bar')),
                         'foo = bar')

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
        self.assertEqual(str(grammar.Cast(type='int8', expr='expr')),
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
