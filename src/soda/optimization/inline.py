import collections
import itertools
import logging

from haoda import util
from haoda.ir import arithmetic
from soda import mutator
from soda import visitor

_logger = logging.getLogger().getChild(__name__)

def inline(stencil):
  """Inline statements that are only referenced once.
  """
  if not stencil.local_stmts:
    return stencil

  refs = {}   # type: Dict[str, Set[Tuple[ir.Ref, ir.LocalOrOutputStmt]]]
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    for var_name, ref_list in visitor.get_load_dict(stmt).items():
      if var_name in stencil.input_names or var_name == stmt.name:
        continue
      refs.setdefault(var_name, set()).update(zip(ref_list,
                                                  itertools.repeat(stmt)))

  refs = {name: next(iter(ref_set))
          for name, ref_set in refs.items() if len(ref_set) == 1}
  if not refs:
    return stencil

  # sort loads to avoid referencing wrong stmt
  local_stmt_table = {stmt.name: idx
                      for idx, stmt in enumerate(stencil.local_stmts)}
  ref_queue = collections.deque(list(refs.items()))
  sorted_refs = []   # type: List[Tuple[ir.Ref, ir.LocalOrOutputStmt]]
  while ref_queue:
    var_name, (ref, load_stmt) = ref_queue.popleft()
    store_stmt = stencil.local_stmts[local_stmt_table[ref.name]]
    accessed_vars = {ref.name for ref in visitor.get_load_set(store_stmt)}
    queued_vars = {var_name for var_name, _ in ref_queue}
    _logger.debug('stmt to be removed: %s', store_stmt)
    _logger.debug('accessed vars: %s', util.lst2str(accessed_vars))
    _logger.debug('queued vars %s', util.lst2str(queued_vars))
    if accessed_vars & queued_vars:
      ref_queue.append((var_name, (ref, load_stmt)))
    else:
      sorted_refs.append((var_name, (ref, load_stmt)))

  for var_name, (ref, load_stmt) in sorted_refs:
    idx, store_stmt = {
        stmt.name: (idx, stmt)
        for idx, stmt in enumerate(stencil.local_stmts)}[var_name]
    offset = tuple(a - b for a, b in zip(store_stmt.ref.idx, ref.idx))
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
  stencil.__dict__.pop('symbol_table', None)
  stencil.__dict__.pop('local_names', None)
  stencil.__dict__.pop('local_types', None)

  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    _logger.debug('simplify  : %s', stmt)
    stmt.expr = arithmetic.simplify(stmt.expr)
    stmt.let = arithmetic.simplify(stmt.let)
    _logger.debug('simplified:  %s', stmt)
  return inline(stencil)
