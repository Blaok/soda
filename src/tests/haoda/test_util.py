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

  def test_type_propagation(self):
    self.assertEqual(util.get_suitable_int_type(15), 'uint4')
    self.assertEqual(util.get_suitable_int_type(16), 'uint5')
    self.assertEqual(util.get_suitable_int_type(15, -1), 'int5')
    self.assertEqual(util.get_suitable_int_type(16, -1), 'int6')
    self.assertEqual(util.get_suitable_int_type(0, -16), 'int5')
    self.assertEqual(util.get_suitable_int_type(0, -17), 'int6')
    self.assertEqual(util.get_suitable_int_type(15, -16), 'int5')
    self.assertEqual(util.get_suitable_int_type(15, -17), 'int6')
    self.assertEqual(util.get_suitable_int_type(16, -16), 'int6')


if __name__ == '__main__':
  unittest.main()
