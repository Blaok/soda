import copy
import unittest

import textx

from soda import core
from soda import grammar
from soda import util

class TestStencil(unittest.TestCase):

  def setUp(self):
    self.tile_size = [233, 0]
    self.soda_type = 'uint16'
    self.unroll_factor = 1
    self.expr_ref = core.grammar.Ref(name='foo', idx=(233, 42), lat=None)
    self.expr = core.grammar.Expr(operand=(self.expr_ref,), operator=())
    self.input_stmt = core.grammar.InputStmt(
        soda_type=self.soda_type, name='foo_i', tile_size=self.tile_size)
    self.param_stmt = core.grammar.ParamStmt(
        soda_type=self.soda_type, name='foo_p', attr=(), size=())
    self.local_ref = core.grammar.Ref(name='foo_l', idx=(0, 0), lat=None)
    self.local_stmt = core.grammar.LocalStmt(
      soda_type=self.soda_type, let=(), ref=self.local_ref, expr=self.expr)
    self.output_ref = core.grammar.Ref(name='foo_o', idx=(0, 0), lat=None)
    self.output_stmt = core.grammar.OutputStmt(
      soda_type=self.soda_type, let=(), ref=self.output_ref, expr=self.expr)
    self.args = {
      'burst_width': 512,
      'dram_bank': 1,
      'border': 'ignore',
      'iterate': 2,
      'cluster': 'none',
      'app_name': 'foo_bar',
      'input_stmts': [self.input_stmt],
      'param_stmts': [self.param_stmt],
      'local_stmts': [self.local_stmt],
      'output_stmts': [self.output_stmt],
      'dim': len(self.tile_size),
      'tile_size': self.tile_size,
      'unroll_factor': self.unroll_factor,
      'replication_factor': self.unroll_factor,
      'dram_separate': False}
    self.soda_mm = textx.metamodel_from_str(
      grammar.SODA_GRAMMAR, classes=grammar.SODA_GRAMMAR_CLASSES)
    self.blur = self.soda_mm.model_from_str(
r'''
kernel: blur
burst width: 512
dram separate: no
dram bank: 1
unroll factor: 16
input uint16: input(2000,)
local uint16: tmp(0,0)=(input(-1,0)+input(0,0)+input(1,0))/3
output uint16: output(0,0)=(tmp(0,-1)+tmp(0,0)+tmp(0,1))/3
iterate: 2
border: preserve
cluster: none
''')

  def test_number_of_input_stmts_different_with_output(self):
    args = {**self.blur.__dict__, **{'replication_factor': 1}}
    core.Stencil(**args)
    input_stmt = copy.copy(self.input_stmt)
    input_stmt.name = 'bar'
    args['input_stmts'] = [self.input_stmt, input_stmt]
    with self.assertRaises(util.SemanticError) as context:
      core.Stencil(**args)
    self.assertEqual(str(context.exception),
      'number of input tensors must be the same as output if iterate > 1 times,'
      ' currently there are 2 input(s) but 1 output(s)')

  def test_soda_type_of_input_stmts_different_with_output(self):
    args = {**self.blur.__dict__, **{'replication_factor': 1}}
    core.Stencil(**args)
    input_stmt = copy.copy(self.input_stmt)
    input_stmt.soda_type = 'half'
    args['input_stmts'] = [input_stmt]
    with self.assertRaises(util.SemanticError) as context:
      core.Stencil(**args)
    self.assertEqual(str(context.exception),
      'input must have the same type(s) as output if iterate > 1 times, '
      'current input has type [half] but output has type [uint16]')

  def test_high_level_dag_construction(self):
    args = {**self.blur.__dict__, **{'replication_factor': 1}}
    stencil = core.Stencil(**args)
    tensors = ('input', 'tmp', 'input_iter1', 'tmp_iter1', 'output')
    self.assertCountEqual(stencil.tensors, tensors)
    self.assertSequenceEqual([_.name for _ in stencil.chronological_tensors],
                             tensors)

if __name__ == '__main__':
  unittest.main()
