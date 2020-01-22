import collections
import copy
import logging

import cached_property

from haoda import ir
from haoda import util
from soda import grammar
import soda.util

_logger = logging.getLogger().getChild(__name__)

class Tensor():
  """A tensor that corresponse to an input, local, or output.

  This class is used in the high-level DAG for stencil dependency analysis.
  Each tensor either is an input tensor, or has at least 1 parent tensor, which
  will be used to generate this tensor. Meanwhile, each tensor either is an
  output tensor, or has at least 1 child tensor, which will be computed using
  this tensor.

  Attributes:
    haoda_type: str, type of the tensor element.
    parents: Dict from str of name of Tensor to Tensor.
    children: Dict from str of name of Tensor to Tensor.
    st_ref: Ref, name, index, and latency stored.
    offset: int, shift offset in terms of data elements
    lets: Lets of computation.
    expr: Expr of computation.
    ld_refs: Dict from str of name to dict of Ref loaded.

  Property:
    name: str, unique in each SODA program.
    st_offset: int, stencil offset in terms of data elements.
    st_idx, Tuple of int, the index referenced by its parent stage.
    ld_indices: Dict from str of name to dict of accessed indices of the input.
    ld_offsets: Dict from str of name to dict of offsets of the input.
  """
  def __init__(self, stmt, tile_size):
    self.haoda_type = stmt.haoda_type
    self._tile_size = tile_size
    if isinstance(stmt, grammar.LocalStmtOrOutputStmt):
      self.st_ref = copy.copy(stmt.ref)
      self.st_ref.parent = self
      self.lets = stmt.let
      self.expr = stmt.expr
    elif isinstance(stmt, grammar.InputStmt):
      self._name = stmt.name
      self.st_ref = None
      self.lets = []
      self.expr = None
    else:
      raise util.InternalError('cannot initialize a Tensor from %s' %
                               type(stmt))
    _logger.debug('tensor initialized from stmt `%s`', stmt)
    # pylint: disable=protected-access
    _logger.debug('                   at tx position %d', stmt._tx_position)

    # these fields are to be set externally
    self.parents = collections.OrderedDict()
    self.children = collections.OrderedDict()
    self.ld_refs = collections.OrderedDict()

  @property
  def name(self):
    if self.st_ref is not None:
      return self.st_ref.name
    return self._name

  @property
  def st_idx(self):
    if self.st_ref is not None:
      return self.st_ref.idx
    return (0,)*len(self._tile_size)

  @property
  def st_offset(self):
    return soda.util.serialize(self.st_idx, self._tile_size)

  @cached_property.cached_property
  def ld_indices(self):
    return collections.OrderedDict(
        (name, collections.OrderedDict((ref.idx, ref) for ref in refs))
        for name, refs in self.ld_refs.items())

  @cached_property.cached_property
  def ld_offsets(self):
    return collections.OrderedDict(
      (name, collections.OrderedDict(
        (soda.util.serialize(ref.idx, self._tile_size), ref) for ref in refs))
      for name, refs in self.ld_refs.items())

  @property
  def c_type(self):
    return self.haoda_type.c_type

  def propagate_type(self):
    if self.expr is None:
      return

    var_types = {}
    # pylint: disable=access-member-before-definition
    for let in self.lets:
      var_types[let.name] = let.haoda_type

    def visit_haoda_type(obj, args):
      if obj.haoda_type is None:
        if isinstance(obj, ir.Var):
          obj.haoda_type = var_types[obj.name]
      return obj

    self.lets = tuple(_.visit(visit_haoda_type) for _ in self.lets)
    self.expr = self.expr.visit(visit_haoda_type)
    self.st_ref = self.st_ref.visit(visit_haoda_type)

  def mutate(self, callback, args=None):
    self.lets = tuple(_.visit(callback, args) for _ in self.lets)
    self.expr = self.expr.visit(callback, args)
    self.st_ref = self.st_ref.visit(callback, args)

  def visit_loads(self, callback, args=None):
    for let in self.lets:
      let.visit(callback, args)
    self.expr.visit(callback, args)

  def __str__(self):
    return '''Tensor
  {haoda_type}: {name} = {expr}
  store: {st_ref}
  parents: {parents}
  children: {children}'''.format(
      name=self.name, haoda_type=self.haoda_type, expr=self.expr,
      parents=util.idx2str(self.parents), children=util.idx2str(self.children),
      st_ref=str(self.st_ref))

  def is_output(self):
    return len(self.children) == 0

  def is_input(self):
    return len(self.parents) == 0

  def is_producer(self):
    return not self.is_output()

  def is_consumer(self):
    return not self.is_input()
