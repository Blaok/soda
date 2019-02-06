import cProfile
import itertools
import pstats
import os
import unittest

from soda.optimization import temporal_cse

class TestHelpers(unittest.TestCase):
  def test_range_from_middle(self):
    self.assertTupleEqual((1, 0, 2),
                          tuple(temporal_cse.range_from_middle(3)))
    self.assertTupleEqual((1, 2, 0, 3),
                          tuple(temporal_cse.range_from_middle(4)))
    self.assertTupleEqual((2, 1, 3, 0, 4),
                          tuple(temporal_cse.range_from_middle(5)))
    self.assertTupleEqual((2, 3, 1, 4, 0, 5),
                          tuple(temporal_cse.range_from_middle(6)))
    for n in range(100):
      self.assertCountEqual(range(n), temporal_cse.range_from_middle(n))

class TestTemporalCse(unittest.TestCase):
  """Test temporal common sub-expression elimination.

  Attributes:
    caching: Boolean value of whether to enabling caching.
    strict: Boolean value of whether to compare the CSE expressions in addition
        to the cost.
  """
  @property
  def cache(self):
    if self.caching:
      return {}
    return None

  def setUp(self):
    temporal_cse.Schedules.set_optimizations(('reorder-exploration',
                                              'skip-with-partial-cost',
                                              'lazy-cartesian-product',
                                              'no-c-temporal-cse'))
    self.caching = True
    self.strict = True
    if 'PROFILING' in os.environ:
      self.pr = cProfile.Profile()
      self.pr.enable()
      print('\n<<<--- %s ---' % self._testMethodName)

  def tearDown(self):
    if 'PROFILING' in os.environ:
      p = pstats.Stats(self.pr)
      p.strip_dirs()
      p.sort_stats('cumtime')
      p.print_stats()
      print('\n--- %s --->>>' % self._testMethodName)

  def test_simple_temporal_cse(self):
    """Test a simple temporal CSE case.

    Expression: x[0] + 2 * x[1] + x[2] + 2 * x[3]
    Expected result: y[0] = x[0] + 2 * x[1]; y[0] + y[2]
    """
    aattr = (1, 2, 1, 2)
    rattr = (0, 1, 2, 3)
    schedule = temporal_cse.Schedules(rattr, aattr, cache=self.cache).best
    if self.strict:
      self.assertEqual('0011011', schedule.brepr)
      self.assertSetEqual({(((0, 1), (1, 2)), '011')}, schedule.operation_set)
    self.assertEqual(2, schedule.cost)

  def test_more_temporal_cse(self):
    """Test a more complicated temporal CSE case.

    Expression: x[0, 0] + 2 * x[1, 0] + 3 * x[2, 0] + 4 * x[4, 0] +
                x[0, 1] + 2 * x[1, 1] + 3 * x[2, 1] + 4 * x[4, 1] +
                x[0, 2] + 2 * x[1, 2] + 3 * x[2, 2] + 4 * x[4, 2] +
    Expected result:
        y[0, 0] = x[0, 0] + 2 * x[1, 0] + 3 * x[2, 0] + 4 * x[3, 0];
        y[0, 0] + y[0, 1] + y[0, 2]
    """
    m, n = 3, 4
    rattr = tuple(map(tuple, map(reversed,
                                 itertools.product(range(m), range(n)))))
    aattr = tuple(range(1, n + 1)) * m
    schedule = temporal_cse.Schedules(rattr, aattr, cache=self.cache).best
    if self.strict:
      self.assertEqual('00011011000110110011011', schedule.brepr)
      self.assertSetEqual({
          ((((0, 0), 1), ((1, 0), 2)), '011'),
          ((((0, 0), 3), ((1, 0), 4)), '011'),
          ((((0, 0), 1), ((1, 0), 2), ((2, 0), 3), ((3, 0), 4)), '0011011'),
          ((((0, 0), 1), ((1, 0), 2), ((2, 0), 3), ((3, 0), 4), ((0, 1), 1),
            ((1, 1), 2), ((2, 1), 3), ((3, 1), 4)), '000110110011011')},
                          schedule.operation_set)
    self.assertEqual(5, schedule.cost)

  def test_5x5_temporal_cse(self):
    """Test a 5x5 temporal CSE case."""
    m, n = 5, 5
    rattr = tuple(map(tuple, map(reversed,
                                 itertools.product(range(m), range(n)))))
    schedule = temporal_cse.Schedules(rattr, cache=self.cache).best
    self.assertEqual(6, schedule.cost)

class TestTemporalCseWithoutLazyCartesianProduct(TestTemporalCse):
  def setUp(self):
    super().setUp()
    temporal_cse.Schedules.set_optimizations(('no-lazy-cartesian-product',))

class TestTemporalCseWithoutSkipping(TestTemporalCse):
  def setUp(self):
    super().setUp()
    temporal_cse.Schedules.set_optimizations(('no-skip-with-partial-cost',))

  @unittest.skip
  def test_more_temporal_cse(self):
    pass

  @unittest.skip
  def test_5x5_temporal_cse(self):
    pass

class TestTemporalCseWithoutReorderingExploration(TestTemporalCse):
  def setUp(self):
    super().setUp()
    temporal_cse.Schedules.set_optimizations(('no-reorder-exploration',))
    self.strict = False

class TestTemporalCseWithoutCaching(TestTemporalCse):
  def setUp(self):
    super().setUp()
    self.caching = False

  @unittest.skip
  def test_more_temporal_cse(self):
    pass

  @unittest.skip
  def test_5x5_temporal_cse(self):
    pass

class TestTemporalCseWithC(TestTemporalCse):
  def setUp(self):
    super().setUp()
    temporal_cse.Schedules.set_optimizations(('c-temporal-cse',))
    self.strict = False

  def test_9x9_temporal_cse(self):
    """Test a 9x9 temporal CSE case."""
    m, n = 9, 9
    rattr = tuple(map(tuple, map(reversed,
                                 itertools.product(range(m), range(n)))))
    schedule = temporal_cse.Schedules(rattr, cache=self.cache).best
    self.assertEqual(8, schedule.cost)
