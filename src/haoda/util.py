from typing import (
    Any,
    Iterable,
    Tuple,
    Optional,
    Union,
)
import contextlib
import logging
import os
import signal

# constants
COORDS_TILED = 'xyzw'
COORDS_IN_TILE = 'ijkl'
COORDS_IN_ORIG = 'pqrs'
TYPE_WIDTH = {'float': 32, 'double': 64, 'half': 16}
MAX_DRAM_BANK = 4

_logger = logging.getLogger().getChild(__name__)


class InternalError(Exception):
  pass


class SemanticError(Exception):
  pass


class SemanticWarn(Exception):
  pass


class InputError(Exception):
  pass


class Printer():

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
      self._out.write('%s%s\n' % (' ' * indent * self._tab, line))
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
    return 'assign_%d' % (self._assign + offset)

  def print_func(self, name, params, suffix='', align=80):
    lines = [name + '(']
    for param in params:
      if ((self._indent + min(1,
                              len(lines) - 1)) * self._tab + len(lines[-1]) +
          len(param + ', ')) > align:
        lines.append(param + ', ')
      else:
        lines[-1] += param + ', '
    if lines[-1][-2:] == ', ':
      lines[-1] = lines[-1][:-2] + ')' + suffix
    line = lines.pop(0)
    self.println(line)
    if lines:
      self.do_indent()
      for line in lines:
        self.println(line)
      self.un_indent()

  @contextlib.contextmanager
  def for_(self, *args):
    if len(args) == 3:
      self.println('for ({}; {}; {}) {{'.format(*args))
    elif len(args) == 2:
      self.println('for ({} : {}) {{'.format(*args))
    else:
      raise InternalError('for_ takes 2 or 3 arguments')
    self.do_indent()
    yield
    self.un_indent()
    self.println('}')

  @contextlib.contextmanager
  def do_while(self, cond):
    self.println('do {')
    self.do_indent()
    yield
    self.un_indent()
    self.println('}} while ({});'.format(cond))

  @contextlib.contextmanager
  def if_(self, cond):
    self.println('if ({}) {{'.format(cond))
    self.do_indent()
    yield
    self.un_indent()
    self.println('}')

  @contextlib.contextmanager
  def elif_(self, cond):
    self.un_indent()
    self.println('}} else if ({}) {{'.format(cond))
    self.do_indent()
    yield

  @contextlib.contextmanager
  def else_(self):
    self.un_indent()
    self.println('} else {')
    self.do_indent()
    yield


def print_define(printer, var, val):
  printer.println('#ifndef %s' % var)
  printer.println('#define %s %d' % (var, val))
  printer.println('#endif//%s' % var)


def print_guard(printer, var, val):
  printer.println('#ifdef %s' % var)
  printer.println('#if %s != %d' % (var, val))
  printer.println('#error %s != %d' % (var, val))
  printer.println('#endif//%s != %d' % (var, val))
  printer.println('#endif//%s' % var)


def get_c_type(haoda_type):
  if haoda_type in {
      'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'
  }:
    return haoda_type + '_t'
  if haoda_type is None:
    return None
  if haoda_type == 'float32':
    return 'float'
  if haoda_type == 'float64':
    return 'double'
  for token in ('int', 'uint'):
    if haoda_type.startswith(token):
      bits = haoda_type.replace(token, '').split('_')
      if len(bits) > 1:
        assert len(bits) == 2
        return 'ap_{}<{}, {}>'.format(token.replace('int', 'fixed'), *bits)
      assert len(bits) == 1
      return 'ap_{}<{}>'.format(token, *bits)
  return haoda_type


def get_haoda_type(c_type):
  return c_type[:-2] if c_type[-2:] == '_t' else c_type


def get_width_in_bits(haoda_type):
  if isinstance(haoda_type, str):
    if haoda_type in TYPE_WIDTH:
      return TYPE_WIDTH[haoda_type]
    for prefix in 'uint', 'int', 'float':
      if haoda_type.startswith(prefix):
        return int(haoda_type.lstrip(prefix).split('_')[0])
  else:
    if hasattr(haoda_type, 'haoda_type'):
      return get_width_in_bits(haoda_type.haoda_type)
  raise InternalError('unknown haoda type: %s' % haoda_type)


def get_width_in_bytes(haoda_type):
  return (get_width_in_bits(haoda_type) - 1) // 8 + 1


def same_type(lhs: str, rhs: str) -> bool:
  if is_float(lhs):
    width = TYPE_WIDTH.get(lhs)
    if width is not None:
      lhs = 'float%d' % width
  if is_float(rhs):
    width = TYPE_WIDTH.get(rhs)
    if width is not None:
      rhs = 'float%d' % width
  return lhs == rhs


def common_type(lhs: str, rhs: str) -> str:
  """Return the common type of two operands.

  TODO: Consider fractional.

  Args:
    lhs: Haoda type of operand 1.
    rhs: Haoda type of operand 2.

  Returns:
    The common type of two operands.
  """
  if is_float(lhs) and not is_float(rhs):
    return lhs
  if is_float(rhs) and not is_float(lhs):
    return rhs
  if get_width_in_bits(lhs) < get_width_in_bits(rhs):
    return rhs
  return lhs


def is_float(haoda_type):
  return haoda_type in {'half', 'double'} or haoda_type.startswith('float')


def is_fixed(haoda_type):
  for token in ('int', 'uint'):
    if haoda_type.startswith(token):
      bits = haoda_type.replace(token, '').split('_')
      if len(bits) > 1:
        return True
  return False


def get_suitable_int_type(upper: int, lower: int = 0) -> str:
  """Returns the suitable integer type with the least bits.

  Returns the integer type that can hold all values between max_val and min_val
  (inclusive) and has the least bits.

  Args:
    max_val: Maximum value that needs to be valid.
    min_val: Minimum value that needs to be valid.

  Returns:
    The suitable haoda_type.
  """
  assert upper >= lower
  upper = max(upper, 0)
  lower = min(lower, 0)
  if lower == 0:
    return 'uint%d' % upper.bit_length()
  return 'int%d' % (max(upper.bit_length(), (lower + 1).bit_length()) + 1)


def idx2str(idx):
  return '(%s)' % ', '.join(map(str, idx))


def lst2str(idx):
  return '[%s]' % ', '.join(map(str, idx))


def add_inv(idx: Iterable[Any]) -> Tuple[Any, ...]:
  return tuple(-x for x in idx)


def get_module_name(module_id):
  return 'module_%d' % module_id


def get_func_name(module_id):
  return 'Module%dFunc' % module_id


get_port_name = lambda name, bank: 'bank_{}_{}'.format(bank, name)
get_port_buf_name = lambda name, bank: 'bank_{}_{}_buf'.format(bank, name)


def get_bundle_name(name, bank):
  return '{}_bank_{}'.format(name.replace('<', '_').replace('>', ''), bank)


def pause_for_debugging():
  if _logger.isEnabledFor(logging.DEBUG):
    try:
      _logger.debug('pausing for debugging... send Ctrl-C to resume')
      signal.pause()
    except KeyboardInterrupt:
      pass


@contextlib.contextmanager
def timeout(seconds=1, error_message='Timeout'):

  def handler(signum, frame):
    raise TimeoutError(error_message)

  signal.signal(signal.SIGALRM, handler)
  signal.alarm(seconds)
  yield
  signal.alarm(0)


def get_job_server_fd(job_server_fd: Union[int, Tuple[()], None]
                     ) -> Optional[int]:
  """Get the job server file descriptor from env var if input is a tuple.

  Args:
    job_server_fd: If this is not a tuple, return it directly.

  Returns:
    The job server file descriptor, or None.
  """
  if isinstance(job_server_fd, tuple):
    job_server = os.environ.get("JOB_SERVER_FD")
    if job_server is not None:
      job_server_fd = int(job_server)
    else:
      job_server_fd = None
  return job_server_fd


# pylint: disable=bad-whitespace
def acquire_job_slot(job_server_fd: Union[int, Tuple[()], None] = ()
                    ) -> Optional[int]:
  """Acquire a job slot if input is not None.

  Args:
    job_server_fd: If this is a tuple, obtain the fd from env var; if this is
        none, do nothing.

  Returns:
    The job server file descriptor, or None.
  """
  job_server_fd = get_job_server_fd(job_server_fd)
  if job_server_fd is not None and len(os.read(job_server_fd, 1)) != 1:
    job_server_fd = None
  return job_server_fd


# pylint: disable=bad-whitespace
def release_job_slot(job_server_fd: Union[int, Tuple[()], None] = ()
                    ) -> Optional[int]:
  """Release a job slot if input is not None.

  Args:
    job_server_fd: If this is a tuple, obtain the fd from env var; if this is
        none, do nothing.

  Returns:
    The job server file descriptor, or None.
  """
  job_server_fd = get_job_server_fd(job_server_fd)
  if job_server_fd is not None and os.write(job_server_fd, b'x') != 1:
    job_server_fd = None
  return job_server_fd
