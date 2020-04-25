import collections
import itertools
import logging

from haoda import ir, util
from haoda.ir import arithmetic

from soda import grammar, mutator, visitor

_logger = logging.getLogger().getChild(__name__)


def inline(stencil):
  """Inline statements that are only referenced once.
  """
  if not stencil.local_stmts:
    return stencil

  refs = {}  # type: Dict[str, Set[Tuple[ir.Ref, ir.LocalOrOutputStmt]]]
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    for var_name, ref_list in visitor.get_load_dict(stmt).items():
      if var_name in stencil.input_names or var_name == stmt.name:
        continue
      refs.setdefault(var_name,
                      set()).update(zip(ref_list, itertools.repeat(stmt)))

  refs = {
      name: next(iter(ref_set))
      for name, ref_set in refs.items()
      if len(ref_set) == 1
  }
  if not refs:
    return stencil

  # sort loads to avoid referencing wrong stmt
  local_stmt_table = {
      stmt.name: idx for idx, stmt in enumerate(stencil.local_stmts)
  }
  ref_queue = collections.deque(list(refs.items()))
  sorted_refs = []  # type: List[Tuple[ir.Ref, ir.LocalOrOutputStmt]]
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
        stmt.name: (idx, stmt) for idx, stmt in enumerate(stencil.local_stmts)
    }[var_name]
    offset = tuple(a - b for a, b in zip(store_stmt.ref.idx, ref.idx))
    ref = mutator.shift(store_stmt.ref, offset)
    lets = tuple(mutator.shift(let, offset) for let in store_stmt.let)
    expr = mutator.shift(store_stmt.expr, offset)
    _logger.info('`%s` is referenced only once, replace with `%s`', ref, expr)
    replace_load = lambda obj, args: args[1] if obj == args[0] else obj
    # TODO: resolve let variable name conflicts
    load_stmt.let = lets + tuple(
        let.visit(replace_load, (ref, expr)) for let in load_stmt.let)
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


def inline2(stencil):
  """Inline statements that are referenced by only one other statement.
  """
  if not stencil.local_stmts:
    return stencil

  refs = collections.OrderedDict(
  )  # type: Dict[str, Dict[ir.LocalOrOutputStmt, List[ir.Ref]]]
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    for var_name, ref_list in visitor.get_load_dict(stmt).items():
      if var_name in stencil.input_names or var_name == stmt.name:
        continue
      refs.setdefault(var_name,
                      collections.OrderedDict()).setdefault(stmt,
                                                            []).extend(ref_list)

  refs = {
      name: next(iter(ref_dict.items()))
      for name, ref_dict in refs.items()
      if len(ref_dict) == 1 and len(
          visitor.get_load_set(
              {stmt.name: stmt.expr
               for stmt in stencil.local_stmts}[name])) == 1
  }
  for name, (stmt, ref_list) in refs.items():
    _logger.info(
        'name: %s stmt: %s ref_list: %s', name, stmt.name,
        util.lst2str(
            visitor.get_load_set(
                {stmt.name: stmt.expr for stmt in stencil.local_stmts}[name])))
  if not refs:
    return stencil

  # sort loads to avoid referencing wrong stmt
  local_stmt_table = {
      stmt.name: idx for idx, stmt in enumerate(stencil.local_stmts)
  }
  ref_queue = collections.deque(list(refs.items()))
  sorted_refs = []  # type: List[Tuple[ir.Ref, ir.LocalOrOutputStmt]]
  while ref_queue:
    var_name, (load_stmt, ref_list) = ref_queue.popleft()
    store_stmt = stencil.local_stmts[local_stmt_table[ref_list[0].name]]
    accessed_vars = {ref.name for ref in visitor.get_load_set(store_stmt)}
    queued_vars = {var_name for var_name, _ in ref_queue}
    _logger.debug('stmt to be removed: %s', store_stmt)
    _logger.debug('accessed vars: %s', util.lst2str(accessed_vars))
    _logger.debug('queued vars %s', util.lst2str(queued_vars))
    if accessed_vars & queued_vars:
      ref_queue.append((var_name, (load_stmt, ref_list)))
    else:
      sorted_refs.append((var_name, (load_stmt, ref_list)))

  for var_name, (load_stmt, ref_list) in sorted_refs:
    idx, store_stmt = {
        stmt.name: (idx, stmt) for idx, stmt in enumerate(stencil.local_stmts)
    }[var_name]
    ref_table = {}
    for ref in ref_list:
      offset = tuple(a - b for a, b in zip(store_stmt.ref.idx, ref.idx))
      ref = mutator.shift(store_stmt.ref, offset)
      lets = tuple(mutator.shift(let, offset) for let in store_stmt.let)
      expr = mutator.shift(store_stmt.expr, offset)
      _logger.info('`%s` is referenced only once by stmt %s, replace with `%s`',
                   ref, load_stmt.name, expr)
      ref_table[ref] = expr
    replace_load = lambda obj, args: args.get(obj, obj)
    # TODO: resolve let variable name conflicts
    load_stmt.let = lets + tuple(
        let.visit(replace_load, ref_table) for let in load_stmt.let)
    load_stmt.expr = load_stmt.expr.visit(replace_load, ref_table)
    del stencil.local_stmts[idx]

  # invalidate cached_property
  stencil.__dict__.pop('symbol_table', None)
  stencil.__dict__.pop('local_names', None)
  stencil.__dict__.pop('local_types', None)

  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    _logger.debug('simplify  : %s', stmt)
    stmt.expr = arithmetic.simplify(
        arithmetic.base.reverse_distribute(stmt.expr))
    stmt.let = arithmetic.simplify(
        tuple(map(arithmetic.base.reverse_distribute, stmt.let)))
    _logger.debug('simplified:  %s', stmt)
  return inline2(stencil)


REBALANCE_THRESHOLDS = {
    ir.Type('float'): 32,
}


def rebalance(stencil):
  """Rebalance the generated code to improve codegen speed.

  This function modifies stencil in-place. Long reduction expressions will be
  divided into groups based on the number of operations. The thresholds are
  defined in REBALANCE_THRESHOLDS.

  Arg:
    stencil: The Stencil object to modify.
  Returns:
    The modified Stencil object.
  """
  for stmt in itertools.chain(stencil.local_stmts, stencil.output_stmts):
    if stmt.haoda_type not in REBALANCE_THRESHOLDS:
      continue
    if isinstance(stmt.expr, ir.AddSub) and set(stmt.expr.operator) == {'+'}:
      reduction = []
      for operator, operand in zip(('+',) + getattr(stmt.expr, 'operator'),
                                   getattr(stmt.expr, 'operand')):
        if isinstance(operand, ir.MulDiv) and operand.operator == ('*',):
          opds = getattr(operand, 'operand')
          if isinstance(opds[0], ir.AddSub):
            reduction.append((opds[1], opds[0]))
          elif isinstance(opds[1], ir.AddSub):
            reduction.append((opds[0], opds[1]))
          else:
            reduction.append((None, operand))
        else:
          reduction.append((None, operand))

      get_num_items = lambda x: 1 if x[0] is None else len(x[1].operand)
      reduction.sort(key=get_num_items, reverse=True)

      num_items = 0
      reductions = [[]]
      for coeff, opds in reduction:
        if num_items + get_num_items(
            (coeff, opds)) > REBALANCE_THRESHOLDS[stmt.haoda_type]:
          reductions.append([])
          num_items = 0
        reductions[-1].append((coeff, opds))
        num_items += get_num_items((coeff, opds))
      if len(reductions) == 1:
        continue
      _logger.info('stmt %s has too many operations, breaking\'em into %d ',
                   stmt.name, len(reductions))
      new_stmts = []
      new_exprs = []
      for reduction in reductions:
        new_operators = []
        new_operands = []
        for coeff, opds in reduction:
          new_operators.append('+')
          if coeff is None:
            new_operands.append(opds)
          else:
            new_operands.append(
                ir.MulDiv(operator=('*',), operand=(opds, coeff)))
        new_exprs.append(
            stencil.propagate_type(
                ir.AddSub(operator=tuple(new_operators[1:]),
                          operand=tuple(new_operands))))
      for new_expr in new_exprs[:-1]:
        new_stmt_name = stencil.new_cr_var()
        new_stmts.append(
            grammar.LocalStmt(ref=ir.Ref(name=new_stmt_name,
                                         lat=None,
                                         idx=(0,) * len(stmt.ref.idx)),
                              haoda_type=new_expr.haoda_type,
                              expr=new_expr,
                              let=stmt.let,
                              stencil=stencil))
        _logger.debug('new stmt: %s', new_stmts[-1])
      stencil.local_stmts.extend(new_stmts)
      stmt.expr = ir.AddSub(
          operator=new_exprs[-1].operator + ('+',) * len(new_stmts),
          operand=new_exprs[-1].operand + tuple(stmt.ref for stmt in new_stmts))

      # invalidate cached_property
      stencil.__dict__.pop('symbol_table', None)
      stencil.__dict__.pop('local_names', None)
      stencil.__dict__.pop('local_types', None)

      _logger.debug('stencil after rebalancing: \n  %s',
                    str(stencil).replace('\n', '\n  '))

      return rebalance(stencil)
  return stencil
