from functools import reduce
import logging
import operator

# constants
COORDS_TILED = 'xyzw'
COORDS_IN_TILE = 'ijkl'
COORDS_IN_ORIG = 'pqrs'
TYPE_WIDTH = {
  'float': 32,
  'double': 64,
  'half': 16
}
MAX_DRAM_BANK = 4

_logger = logging.getLogger('__main__').getChild(__name__)

class InternalError(Exception):
  pass

class SemanticError(Exception):
  pass

class SemanticWarn(Exception):
  pass

class Printer(object):
  def __init__(self, out):
    self._out = out
    self._indent = 0
    self._assign = 0
    self._comments = []
    self._tab = 2

  def println(self, line='', indent=-1):
    if indent < 0:
      indent = self._indent
    if line:
      self._out.write('%s%s\n' % (' '*indent*self._tab, line))
    else:
      self._out.write('\n')

  def do_indent(self):
    self._indent += 1

  def un_indent(self):
    self._indent -= 1

  def do_scope(self, comment=''):
    self.println('{')
    self.do_indent()
    self._comments.append(comment)

  def un_scope(self, comment='', suffix=''):
    self.un_indent()
    popped_comment = self._comments.pop()
    if comment:
      self.println('}%s // %s' % (suffix, comment))
    else:
      if popped_comment:
        self.println('}%s // %s' % (suffix, popped_comment))
      else:
        self.println('}%s' % suffix)

  def new_var(self):
    self._assign += 1
    return self.last_var()

  def last_var(self, offset=-1):
    return 'assign_%d' % (self._assign+offset)

  def print_func(self, name, params, suffix='', align=80):
    lines = [name+'(']
    for param in params:
      if ((self._indent + min(1, len(lines)-1))*self._tab+
          len(lines[-1])+len(param+', ')) > align:
        lines.append(param+', ')
      else:
        lines[-1] += param+', '
    if lines[-1][-2:] == ', ':
      lines[-1] = lines[-1][:-2]+')'+suffix
    line = lines.pop(0)
    self.println(line)
    if lines:
      self.do_indent()
      for line in lines:
        self.println(line)
      self.un_indent()

def print_define(printer, var, val):
  printer.println('#ifndef %s' % var)
  printer.println('#define %s %d' % (var, val))
  printer.println('#endif//%s' % var)

def print_guard(printer, var, val):
  printer.println('#if %s != %d' % (var, val))
  printer.println('#error %s != %d' % (var, val))
  printer.println('#endif//%s != %d' % (var, val))

def get_c_type(soda_type):
  if soda_type in {
      'uint8', 'uint16', 'uint32', 'uint64',
      'int8', 'int16', 'int32', 'int64'}:
    return soda_type+'_t'
  if soda_type is None:
    return None
  for token in ('int', 'uint'):
    if soda_type.startswith(token):
      return 'ap_{}<{}>'.format(token, soda_type.replace(token, ''))
  return soda_type

def get_soda_type(c_type):
  return c_type[:-2] if c_type[-2:] == '_t' else c_type

def get_width_in_bits(soda_type):
  if isinstance(soda_type, str):
    if soda_type in TYPE_WIDTH:
      return TYPE_WIDTH[soda_type]
    for prefix in 'uint', 'int', 'float':
      if soda_type.startswith(prefix):
        return int(soda_type.lstrip(prefix).split('_')[0])
  else:
    if hasattr(soda_type, 'soda_type'):
      return get_width_in_bits(soda_type.soda_type)
  raise InternalError('unknown soda type: %s' % soda_type)

def get_width_in_bytes(soda_type):
  return (get_width_in_bits(soda_type)-1)//8+1

def is_float(soda_type):
  return soda_type in {'half', 'double'} or soda_type.startswith('float')

def serialize(vec, tile_size):
  return sum((vec[i]*reduce(operator.mul, tile_size[:i])
        for i in range(1, len(tile_size))),
         vec[0])

def serialize_iter(iterative, tile_size):
  return [serialize(x, tile_size) for x in iterative]

def deserialize(offset, tile_size):
  return tuple(deserialize_generator(offset, tile_size))

def deserialize_generator(offset, tile_size):
  for size in tile_size[:-1]:
    yield offset % size
    offset = offset // size
  yield offset

def idx2str(idx):
  return '(%s)' % ', '.join(map(str, idx))

def lst2str(idx):
  return '[%s]' % ', '.join(map(str, idx))

def get_module_name(module_id):
  return 'module_%d' % module_id

def get_func_name(module_id):
  return 'Module%d' % module_id

get_port_name = lambda name, bank: 'bank_{}_{}'.format(bank, name)
get_port_buf_name = lambda name, bank: 'bank_{}_{}_buf'.format(bank, name)
def get_fifo_name(name, offset):
  return 'stream_{}_offset_{}'.format(name, offset)
def get_tensor_name_at(name, offset):
  return 'tensor_{}_offset_{}'.format(name, offset)
