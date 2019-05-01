import logging
import os
import unittest

import textx

from soda import core
from soda import grammar
from soda.optimization import inline

logging.basicConfig(level=logging.FATAL,
                    format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')

_logger = logging.getLogger().getChild(__name__)
if 'DEBUG' in os.environ:
  logging.getLogger().setLevel(logging.DEBUG)

class TestInline(unittest.TestCase):

  def setUp(self):
    self.soda_mm = textx.metamodel_from_str(
      grammar.GRAMMAR, classes=grammar.CLASSES)

  def test_simple_inlining(self):
    program = self.soda_mm.model_from_str(
r'''
kernel: blur
burst width: 512
unroll factor: 16
input float: t0(233, *)
local float: t1(-1, -2) = t0(0, 1)
output float: t2(4, 2) = t1(2, 3)
iterate: 1
border: preserve
cluster: none
''')
    args = {**program.__dict__, **{'replication_factor': 1}}
    stencil = core.Stencil(**args)
    inline.inline(stencil)
    self.assertEqual(len(stencil.local_stmts), 0)
    self.assertEqual(len(stencil.output_stmts), 1)
    self.assertEqual(str(stencil.output_stmts[0]),
                     'output float: t2(4, 2) = t0(3, 6)')

  def test_let_in_local(self):
    program = self.soda_mm.model_from_str(
r'''
kernel: blur
burst width: 512
unroll factor: 16
input float: t0(233, *)
local float: float l = t0(0, 1) t1(-1, -2) = l
output float: t2(4, 2) = t1(2, 3)
iterate: 1
border: preserve
cluster: none
''')
    args = {**program.__dict__, **{'replication_factor': 1}}
    stencil = core.Stencil(**args)
    inline.inline(stencil)
    self.assertEqual(len(stencil.local_stmts), 0)
    self.assertEqual(len(stencil.output_stmts), 1)
    self.assertMultiLineEqual(str(stencil.output_stmts[0]),
r'''output float:
  float l = t0(3, 6)
  t2(4, 2) = l''')

  def test_let_in_output(self):
    program = self.soda_mm.model_from_str(
r'''
kernel: blur
burst width: 512
unroll factor: 16
input float: t0(233, *)
local float: t1(-1, -2) = t0(0, 1)
output float: float l = t1(2, 3) t2(4, 2) = l
iterate: 1
border: preserve
cluster: none
''')
    args = {**program.__dict__, **{'replication_factor': 1}}
    stencil = core.Stencil(**args)
    inline.inline(stencil)
    self.assertEqual(len(stencil.local_stmts), 0)
    self.assertEqual(len(stencil.output_stmts), 1)
    self.assertMultiLineEqual(str(stencil.output_stmts[0]),
r'''output float:
  float l = t0(3, 6)
  t2(4, 2) = l''')

  def test_access_in_different_stmts(self):
    program = self.soda_mm.model_from_str(
r'''
kernel: blur
burst width: 512
unroll factor: 16
input float: t0(233, *)
local float: t1(-1, -2) = t0(0, 1)
local float: t2(0, 0) = t1(0, 0)
output float: t3(4, 2) = t2(0, 0) + t1(0, 0) + t2(0, 1)
iterate: 1
border: preserve
cluster: none
''')
    args = {**program.__dict__, **{'replication_factor': 1}}
    stencil = core.Stencil(**args)
    inline.inline(stencil)
    self.assertEqual(len(stencil.local_stmts), 2)
    self.assertEqual(len(stencil.output_stmts), 1)
    self.assertEqual(str(stencil.output_stmts[0]),
                     'output float: t3(4, 2) = t2(0, 0) + t1(0, 0) + t2(0, 1)')

if __name__ == '__main__':
  unittest.main()
