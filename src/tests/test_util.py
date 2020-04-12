import unittest

from soda import util

class TestUtils(unittest.TestCase):

  def test_deserialize(self):
    idx = (42, 23, 233)
    tile_size = (2333, 233, 0)
    self.assertTupleEqual(idx, tuple(util.deserialize(
      util.serialize(idx, tile_size), tile_size)))

if __name__ == '__main__':
  unittest.main()
