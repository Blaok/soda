import itertools
import logging

from haoda import util
from soda import mutator
from soda import visitor

_logger = logging.getLogger().getChild(__name__)

def inline(stencil):
  """Inline statements that are only referenced once.
  """
  if not stencil.local_stmts:
    return stencil

  loads = {}
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    for var_name, load_list in itertools.chain(
        *(visitor.get_load_dict(let).items() for let in stmt.let),
        visitor.get_load_dict(stmt.expr).items()):
      if var_name in stencil.input_names:
        continue
      loads.setdefault(var_name, {}).update(zip(load_list,
                                                itertools.repeat(stmt)))
  _logger.debug('loads: %s', {k: util.lst2str(v) for k, v in loads.items()})

  loads = {var_name: load_dict
           for var_name, load_dict in loads.items()
           if len(load_dict) == 1}
  if not loads:
    return stencil

  for var_name, load_dict in loads.items():
    load, load_stmt = next(iter(load_dict.items()))
    idx, store_stmt = 0, stencil.local_stmts[0]
    for idx, store_stmt in enumerate(stencil.local_stmts):
      if store_stmt.name == var_name:
        break
    else:
      raise util.InputError('stmt %s referenced but not defined' % var_name)
    offset = tuple(a - b for a, b in zip(store_stmt.ref.idx, load.idx))
    ref = mutator.shift(store_stmt.ref, offset)
    lets = tuple(mutator.shift(let, offset) for let in store_stmt.let)
    expr = mutator.shift(store_stmt.expr, offset)
    _logger.info('`%s` is referenced only once, replace with `%s`', ref, expr)
    replace_load = lambda obj, args: args[1] if obj == args[0] else obj
    # TODO: resolve let variable name conflicts
    load_stmt.let = lets + tuple(let.visit(replace_load, (ref, expr))
                                 for let in load_stmt.let)
    load_stmt.expr = load_stmt.expr.visit(replace_load, (ref, expr))
    del stencil.local_stmts[idx]

  # invalidate cached_property
  del stencil.__dict__['symbol_table']
  del stencil.__dict__['local_names']
  del stencil.__dict__['local_types']
  return inline(stencil)
