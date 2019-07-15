import unittest

from haoda import util

class TestUtil(unittest.TestCase):

  def test_c_type(self):
    self.assertEqual(util.get_c_type('uint2'), 'ap_uint<2>')
    self.assertEqual(util.get_c_type('int4'), 'ap_int<4>')
    self.assertEqual(util.get_c_type('uint8'), 'uint8_t')
    self.assertEqual(util.get_c_type('int16'), 'int16_t')
    self.assertEqual(util.get_c_type('uint32_16'), 'ap_ufixed<32, 16>')
    self.assertEqual(util.get_c_type('int64_32'), 'ap_fixed<64, 32>')
    self.assertEqual(util.get_c_type('float'), 'float')
    self.assertEqual(util.get_c_type('float32'), 'float')
    self.assertEqual(util.get_c_type('float64'), 'double')
    self.assertEqual(util.get_c_type('double'), 'double')

if __name__ == '__main__':
  unittest.main()
