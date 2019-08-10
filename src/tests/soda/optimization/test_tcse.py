from typing import (
    Tuple,)

import cProfile
import os
import pstats
import tracemalloc
import unittest

import logging

from soda.optimization import tcse

logging.basicConfig(level=logging.FATAL,
                    format='%(levelname)s:%(name)s:%(lineno)d: %(message)s')

_logger = logging.getLogger().getChild(__name__)
if 'DEBUG' in os.environ:
  logging.getLogger().setLevel(logging.DEBUG)


class TestHelpers(unittest.TestCase):

  def test_range_from_middle(self):
    self.assertTupleEqual((1, 0, 2), tuple(tcse.range_from_middle(3)))
    self.assertTupleEqual((1, 2, 0, 3), tuple(tcse.range_from_middle(4)))
    self.assertTupleEqual((2, 1, 3, 0, 4), tuple(tcse.range_from_middle(5)))
    self.assertTupleEqual((2, 3, 1, 4, 0, 5), tuple(tcse.range_from_middle(6)))
    for n in range(100):
      self.assertCountEqual(range(n), tcse.range_from_middle(n))


class TestLinearizer(unittest.TestCase):

  def test_3x3_linearizer(self):
    rattrs = ((-1, -1), (-1, 0), (-1, 1), (-1, 0), (0, 0), (1, 0), (-1, 1),
              (0, 1), (1, 1))
    linearizer = tcse.Linearizer(rattrs)
    self.assertEqual(linearizer.num_dim, 2)
    self.assertSequenceEqual(linearizer.maxs, (1, 1))
    self.assertSequenceEqual(linearizer.mins, (-1, -1))
    self.assertSequenceEqual(linearizer.weights, (1, 5))
    self.assertSequenceEqual(
        rattrs, [tuple(linearizer(linearizer(rattr))) for rattr in rattrs])


class TestCommSchedule(unittest.TestCase):

  def get_int_attrs(self, idx: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return self.rattrs['int'][idx], self.aattrs[idx]

  def get_tuple_attrs(self, idx: int) \
      -> Tuple[Tuple[Tuple[int, int], ...], Tuple[int, ...]]:
    return self.rattrs['int'][idx], self.aattrs[idx]

  def setUp(self):
    self.rattrs = {
        'int': {
            9: (0, 1, 2, 10, 11, 12, 20, 21, 22),
        },
        'tuple': {
            9: (
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
            ),
        },
    }
    self.aattrs = {9: (1, 1, 1, 1, 2, 1, 1, 1, 1)}

    if 'PROFILING' in os.environ:
      self.pr = cProfile.Profile()
      self.pr.enable()
      tracemalloc.start()
      self.snapshot1 = tracemalloc.take_snapshot()
      print('\n<<<--- %s ---' % self._testMethodName)

  def tearDown(self):
    if 'PROFILING' in os.environ:
      self.snapshot2 = tracemalloc.take_snapshot()
      p = pstats.Stats(self.pr)
      p.strip_dirs()
      p.sort_stats('cumtime')
      p.print_stats()
      top_stats = self.snapshot2.compare_to(self.snapshot1, 'lineno')
      print("[ Top 10 differences ]")
      for stat in top_stats[:10]:
        print(stat)
      print('\n--- %s --->>>' % self._testMethodName)

  def test_norm_attrs(self):
    rattrs, _ = self.get_int_attrs(9)
    # 0 + ((1 + 3) + 2)
    schedule = tcse.CommSchedule(None, None, rattrs[3] - rattrs[1], rattrs)
    schedule = tcse.CommSchedule(schedule, None, rattrs[2] - rattrs[1], rattrs)
    schedule = tcse.CommSchedule(None, schedule, rattrs[1] - rattrs[0], rattrs)
    self.assertSequenceEqual(tuple(sorted(schedule.norm_attrs)),
                             (rattrs[0], rattrs[1], rattrs[2], rattrs[3]))

  def test_uniq_exprs(self):
    rattrs, _ = self.get_int_attrs(9)
    uniq_expr_dict = {}

    # 1 + 3
    schedule = tcse.CommSchedule(None, None, rattrs[3] - rattrs[1], rattrs)
    expr = [0, rattrs[3] - rattrs[1]]
    uniq_expr_dict[frozenset(expr)] = [schedule]

    # (1 + 3) + 2
    schedule = tcse.CommSchedule(schedule, None, rattrs[2] - rattrs[1], rattrs)
    expr.append(rattrs[2] - rattrs[1])
    uniq_expr_dict[frozenset(expr)] = [schedule]

    # 0 + ((1 + 3) + 2)
    schedule = tcse.CommSchedule(None, schedule, rattrs[1] - rattrs[0], rattrs)
    uniq_expr_dict[frozenset(rattrs[:4])] = [schedule]

    self.assertDictEqual(schedule.uniq_expr_dict, uniq_expr_dict)
    self.assertSetEqual(schedule.uniq_expr_set, set(uniq_expr_dict))

    uniq_expr_dict.clear()

    # 0 + 1
    schedule1 = tcse.CommSchedule(None, None, rattrs[1] - rattrs[0], rattrs)
    # 3 + 4
    schedule2 = tcse.CommSchedule(None, None, rattrs[4] - rattrs[3], rattrs)
    expr = [0, rattrs[1] - rattrs[0]]
    self.assertEqual(rattrs[4] - rattrs[3], expr[-1])
    uniq_expr_dict[frozenset(expr)] = [schedule1, schedule2]

    # (0 + 1) + (3 + 4)
    schedule = tcse.CommSchedule(schedule1, schedule2, rattrs[3] - rattrs[0],
                                 rattrs)
    uniq_expr_dict[frozenset(rattrs[0:2] + rattrs[3:5])] = [schedule]
    self.assertDictEqual(schedule.uniq_expr_dict, uniq_expr_dict)
    self.assertSetEqual(schedule.uniq_expr_set, set(uniq_expr_dict))


class TestCommSchedules(unittest.TestCase):
  """Test temporal common sub-expression elimination.

  Attributes:
    caching: Boolean value of whether to enabling caching.
  """

  @property
  def cache(self):
    if self.caching:
      return {}
    return None

  def setUp(self):
    self.Schedules = tcse.CommSchedules
    self.Schedules.set_optimizations(
        ('reorder-exploration', 'skip-with-partial-cost',
         'lazy-cartesian-product', 'no-c-temporal-cse'))
    self.caching = True
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

  def test_simple_tcse(self):
    """Test a simple temporal CSE case.

    Expression: x[0] + 2 * x[1] + x[2] + 2 * x[3]
    Expected result: y[0] = x[0] + 2 * x[1]; y[0] + y[2]
    """
    aattrs = (1, 2, 1, 2)
    rattrs = (0, 1, 2, 3)
    schedule = self.Schedules(rattrs, aattrs, cache=self.cache).best
    self.assertEqual(2, schedule.num_ops)

  def test_3x2_tcse(self):
    """Test a 3x2 temporal CSE case."""
    rattrs = (0, 1, 2, 10, 11, 12)
    schedules = self.Schedules(rattrs, None, cache=self.cache)
    schedule = schedules.best
    schedules.print_stats()
    self.assertEqual(3, schedule.num_ops)
    aattrs = (1, 1, 1, 1, 3, 1)
    schedules = self.Schedules(rattrs, aattrs, cache=self.cache)
    schedule = schedules.best
    schedules.print_stats()
    self.assertEqual(4, schedule.num_ops)

  def test_jacobi2d_tcse(self):
    rattrs = (1, 10, 11, 12, 21)
    schedules = self.Schedules(rattrs, None, cache=self.cache)
    schedule = schedules.best
    schedules.print_stats()
    self.assertEqual(3, schedule.num_ops)
    aattrs = (0, 0, 1, 0, 0)
    schedules = self.Schedules(rattrs, aattrs, cache=self.cache)
    schedule = schedules.best
    schedules.print_stats()
    self.assertEqual(3, schedule.num_ops)


class TestCommSchedulesWithoutLazyCartesianProduct(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.Schedules.set_optimizations(('no-lazy-evaluation',))


class TestCommSchedulesWithoutSkipping(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.Schedules.set_optimizations(('no-skip-with-partial-cost',))


class TestCommSchedulesWithoutReorderingExploration(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.Schedules.set_optimizations(('no-reorder-exploration',))


class TestCommSchedulesWithoutCaching(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.caching = False


class TestGreedySchedules(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.Schedules = tcse.GreedySchedules

  def test_3x3_tcse(self):
    """Test a 3x3 temporal CSE case."""
    rattrs = [(x, y) for y in range(3) for x in range(3)]
    linearizer = tcse.Linearizer(rattrs)
    rattrs = tuple(map(linearizer, rattrs))
    _logger.debug('rattrs: %s', rattrs)

    def test(aattrs, num_ops=None, total_distance=None):
      schedules = self.Schedules(rattrs, aattrs, linearizer)
      _logger.debug('aattrs: %s', aattrs)
      schedule = schedules.best
      schedules.print_stats()
      _logger.debug('schedule: %s', schedule)
      if num_ops is not None:
        self.assertEqual(num_ops, schedule.num_ops)
      if total_distance is not None:
        self.assertGreaterEqual(total_distance, schedule.total_distance)

    test(None, num_ops=4, total_distance=12)
    test((1, 1, 1, 1, 2, 1, 1, 1, 1), num_ops=5, total_distance=13)
    test((1, 1, 2, 3, 3, 1, 4, 4, 1), num_ops=6, total_distance=13)
    test((4, 1, 3, 0, 2, 3, 5, 6, 2), num_ops=8, total_distance=12)
    test((7, 6, 7, 2, 1, 7, 2, 1, 7), num_ops=6, total_distance=12)
    test((2, 3, 6, 4, 3, 3, 4, 4, 3), num_ops=6, total_distance=16)
    test((4, 4, 0, 7, 4, 0, 7, 3, 1), num_ops=6, total_distance=17)
    test((5, 1, 7, 1, 1, 7, 1, 1, 1), num_ops=6, total_distance=17)
    test((1, 6, 5, 5, 4, 1, 1, 6, 5), num_ops=6, total_distance=17)
    test((4, 3, 0, 2, 0, 0, 6, 0, 0), num_ops=7, total_distance=12)
    test((1, 1, 1, 0, 1, 1, 1, 0, 3), num_ops=6, total_distance=18)
    test((1, 2, 1, 2, 3, 2, 1, 2, 1), num_ops=6, total_distance=13)

  def test_5x5_tcse(self):
    """Test a 5x5 temporal CSE case."""
    m, n = 5, 5
    rattrs = [(x, y) for y in range(n) for x in range(m)]
    linearizer = tcse.Linearizer(rattrs)
    schedule = self.Schedules(tuple(map(linearizer, rattrs)),
                              linearizer=linearizer,
                              cache=self.cache).best
    self.assertEqual(6, schedule.num_ops)

  def test_more_tcse(self):
    """Test a more complicated temporal CSE case.

    Expression: x[0, 0] + 2 * x[1, 0] + 3 * x[2, 0] + 4 * x[4, 0] +
                x[0, 1] + 2 * x[1, 1] + 3 * x[2, 1] + 4 * x[4, 1] +
                x[0, 2] + 2 * x[1, 2] + 3 * x[2, 2] + 4 * x[4, 2] +
    Expected result:
        y[0, 0] = x[0, 0] + 2 * x[1, 0] + 3 * x[2, 0] + 4 * x[3, 0];
        y[0, 0] + y[0, 1] + y[0, 2]
    """
    m, n = 3, 4
    rattr = tuple(m * 2 * i + j for i in range(m) for j in range(n))
    aattr = tuple(range(1, n + 1)) * m
    schedule = self.Schedules(rattr, aattr, cache=self.cache).best
    self.assertEqual(5, schedule.num_ops)

  def test_11x11_tcse(self):
    m, n = 11, 11
    rattrs = [0] * m * n
    aattrs = [0] * m * n
    for y in range(n):
      for x in range(m):
        rattrs[y * m + x] = (x, y)
        aattrs[y * m + x] = (x - m // 2)**2 + (y - n // 2)**2
    linearizer = tcse.Linearizer(rattrs)
    rattrs = tuple(map(linearizer, rattrs))
    schedule = self.Schedules(rattrs, aattrs, linearizer=linearizer).best
    self.assertEqual(70, schedule.num_ops)
    self.assertGreaterEqual(245, schedule.total_distance)
    schedule = self.Schedules(rattrs, linearizer=linearizer).best
    self.assertEqual(10, schedule.num_ops)
    self.assertGreaterEqual(220, schedule.total_distance)

  def test_16x16_tcse(self):
    m, n = 16, 16
    rattrs = [0] * m * n
    for y in range(n):
      for x in range(m):
        rattrs[y * m + x] = x, y
    linearizer = tcse.Linearizer(rattrs)
    rattrs = tuple(map(linearizer, rattrs))
    schedule = self.Schedules(rattrs, linearizer=linearizer).best
    self.assertEqual(8, schedule.num_ops)


class TestExternalSchedules(TestCommSchedules):

  def setUp(self):
    super().setUp()
    self.Schedules = tcse.ExternalSchedules

  def test_3x3_tcse(self):
    """Test a 3x3 temporal CSE case."""
    rattrs = (0, 1, 2, 10, 11, 12, 20, 21, 22)
    aattrs = (1, 1, 1, 1, 2, 1, 1, 1, 1)
    schedules = self.Schedules(rattrs, aattrs, cache=self.cache)
    schedule = schedules.best
    schedules.print_stats()
    self.assertEqual(5, schedule.num_ops)
