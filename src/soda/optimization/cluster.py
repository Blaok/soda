import collections
import copy
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from haoda import ir, util

from soda import dataflow

_logger = logging.getLogger().getChild(__name__)


class PrettyPrinter:

  def __init__(self, module: ir.Module):
    self._module = module

  def __str__(self) -> str:
    lines = ['', repr(self._module)]
    for parent in self._module.parents:
      lines.append(f'  parent: {repr(parent)}')
    for child in self._module.children:
      lines.append(f'  child:  {repr(child)}')
    for let in self._module.lets:
      lines.append(f'  let:    {let}')
    for k, v in self._module.exprs.items():
      lines.append(f'  value:  {v}')
      lines.append(f'    sink:   {k}')
    return '\n  '.join(lines)


def cluster(
    super_source: dataflow.SuperSourceNode,
    granularity: str = 'none',
    unroll_factor: int = 0,
) -> None:
  """Cluster a dataflow.SuperSourceNode with given granularity."""

  if granularity == 'none':
    return

  nodes_to_merge: Dict[str, List[ir.Module]] = collections.defaultdict(list)
  if granularity == 'full':
    for node in super_source.tpo_node_gen():
      _logger.debug('old node before merging: %s', PrettyPrinter(node))
      if isinstance(node, (dataflow.ForwardNode, dataflow.ComputeNode)):
        nodes_to_merge[''].append(node)

  elif granularity == 'coarse':
    for node in super_source.tpo_node_gen():
      _logger.debug('old node before merging: %s', PrettyPrinter(node))
      if isinstance(node, (dataflow.ForwardNode, dataflow.ComputeNode)):
        nodes_to_merge[node.tensor.name].append(node)

  elif granularity == 'fine':
    if unroll_factor <= 0:
      raise util.InputError(
          'unroll_factor must be specified for fine-grained clustering')
    for node in super_source.tpo_node_gen():
      _logger.debug('old node before merging: %s', PrettyPrinter(node))
      if isinstance(node, dataflow.ComputeNode):
        nodes_to_merge[f'{node.tensor.name}_{node.pe_id}'].append(node)
      elif isinstance(node, dataflow.ForwardNode):
        pe_id = unroll_factor - 1 - node.offset % unroll_factor
        nodes_to_merge[f'{node.tensor.name}_{pe_id}'].append(node)

  else:
    raise util.InputError(f'unknown cluster granularity: {granularity}')

  _logger.debug('=' * 80)
  for name, nodes in nodes_to_merge.items():
    for node in nodes:
      _logger.debug('merging node: %s', PrettyPrinter(node))
    _logger.debug('=' * 80)
    merged_node = merge(super_source, nodes, name)
    _logger.debug('merged node: %s', PrettyPrinter(merged_node))
    _logger.debug('=' * 80)
  for node in super_source.tpo_node_gen():
    _logger.debug('new node after merging: %s', PrettyPrinter(node))
  _logger.debug('=' * 80)
  _logger.debug('done %s clustering', granularity)


def merge(
    super_source: dataflow.SuperSourceNode,
    nodes_to_merge: Iterable[ir.Module],
    name: str = '',
) -> ir.Module:
  """Modify dataflow.SuperSourceNode with dataflow nodes merged.

  Algorithm:
    1. Identify all ir.Module nodes;
    2. Identify nodes that are going to be merged (thus removed);
    3. Identify nodes that are parents of the removed nodes;
    4. Identify nodes that are children of the removed nodes;
    4. Create new node that are merged from all removed nodes;
    5. Update the parents to write to the new node;
    6. Update the children to read from the new node;

  Returns:
    The new ir.Module created as the merging result of all nodes.
  """
  all_nodes = {id(x): x for x in super_source.tpo_node_gen()}
  old_nodes = {id(x): x for x in nodes_to_merge}

  # check that all nodes_to_merge belongs to the super_source.
  for node_id, node in old_nodes.items():
    if node_id not in all_nodes:
      raise util.InternalError(
          f'cannot merge: module {node} does not belong to {super_source}')
  # TODO: check that all nodes_to_merge are connected.

  new_node = ir.Module(name)  # create the new merged module node

  # find and dedupe parents/children of the new node
  parents: Dict[int, ir.Module] = {}
  children: Dict[int, ir.Module] = {}
  for node in old_nodes.values():
    parents.update((id(x), x) for x in node.parents if id(x) not in old_nodes)
    children.update((id(x), x) for x in node.children if id(x) not in old_nodes)
  new_node.parents.extend(parents.values())
  new_node.children.extend(children.values())

  # dict mapping an old ir.FIFO to a new ir.Node due to the topology change
  replacement: Dict[ir.FIFO, ir.Node] = {}

  # dict mapping a new ir.Node back to a list of (ir.FIFO, ir.Node)
  # used to find FIFOs that contain packed types ("packed channel")
  reverse_replacement: Dict[ir.Node, List[Tuple[ir.FIFO, ir.Node]]]
  reverse_replacement = collections.defaultdict(list)

  def add_replacement(
      old_fifo: ir.FIFO,
      producer: ir.Node,
      consumer: ir.Node,
  ) -> None:
    replacement[old_fifo] = consumer
    reverse_replacement[consumer].append((old_fifo, producer))

  # initialize the new module node and figure out how to replace FIFOs
  for node in old_nodes.values():

    def local_replace(obj: ir.Node, local_replacement: dict) -> ir.Node:
      return local_replacement.get(obj, obj)

    # copy existing lets to the new node
    local_replacement = {}
    for let in node.lets:
      new_let = ir.Let(
          name=f'let_{len(new_node.lets)}',
          expr=let.expr,
          haoda_type=let.haoda_type,
      )
      local_replacement[ir.make_var(let.name)] = ir.make_var(
          new_let.name,
          new_let.haoda_type,
      )
      # replace any seen local lets
      new_node.lets.append(new_let.visit(local_replace, local_replacement))

    for fifo, expr in node.exprs.items():
      # replace any local lets
      expr = expr.visit(local_replace, local_replacement)

      if id(fifo.read_module) in old_nodes:
        # child module is merged
        new_let = ir.Let(
            name=f'let_{len(new_node.lets)}',
            expr=expr,
            haoda_type=expr.haoda_type,
        )
        new_node.lets.append(new_let)
        # FIFO is internal to the merged nodes
        # replace with ir.Var referencing ir.Let
        add_replacement(fifo, expr, ir.make_var(new_let.name, expr.haoda_type))
      else:
        # child module is kept
        # FIFO is written by the merged node and read externally
        # replace with new ir.FIFO with updated write_module
        new_node.exprs[fifo] = expr
        add_replacement(
            fifo,
            expr,
            _get_updated_fifo(fifo, write_module=new_node),
        )

  for node in new_node.parents:
    for fifo, expr in node.exprs.items():
      if id(fifo.read_module) in old_nodes:
        # FIFO is written externally and read by the merged node
        # replace with new ir.FIFO with updated read_module
        add_replacement(
            fifo,
            expr,
            _get_updated_fifo(fifo, read_module=new_node),
        )

  # handle packed channel
  producer_replacement: Dict[ir.FIFO, ir.Node] = {}
  consumer_replacement: Dict[ir.FIFO, ir.Node] = {}
  for new_fifo, old_fifos in reverse_replacement.items():
    if len(old_fifos) > 1:
      _logger.debug('packed channel: %s -> %s', repr(new_fifo.write_module),
                    repr(new_fifo.read_module))

      pack = ir.Pack(exprs=[x for _, x in old_fifos])
      for idx, (old_fifo, _) in enumerate(old_fifos):
        # replace old FIFO with ir.Pack for producer
        producer_replacement[old_fifo] = pack

        # replace old FIFO with ir.Unpack for consumer
        unpack = ir.Unpack(expr=new_fifo, idx=idx)
        consumer_replacement[old_fifo] = unpack

        _logger.debug('  channel[%d]: %s', idx, old_fifo)
        _logger.debug('  -> (producer): %s', pack)
        _logger.debug('  -> (consumer): %s', unpack)

  def replace(obj: ir.Node, args=()) -> ir.Node:
    return consumer_replacement.get(obj, replacement.get(obj, obj))

  # replace ir.FIFOs in the children nodes, new node, and parent nodes, in that
  # order, since type inference may depend on old values.
  for node in new_node.children:
    node.parents = [x for x in node.parents if id(x) not in old_nodes]
    node.parents.append(new_node)
    node.lets = [x.visit(replace) for x in node.lets]
    node.exprs = {k: v.visit(replace) for k, v in node.exprs.items()}

  new_node.lets = [x.visit(replace) for x in new_node.lets]
  new_node.exprs = {
      replacement.get(k, k): producer_replacement.get(k, v).visit(replace)
      for k, v in new_node.exprs.items()
  }

  for node in new_node.parents:
    node.children = [x for x in node.children if id(x) not in old_nodes]
    node.children.append(new_node)
    node.exprs = {
        replacement.get(k, k): producer_replacement.get(k, v)
        for k, v in node.exprs.items()
    }

  # clear cache in super_source
  super_source.__dict__.pop('module_table', None)
  super_source.__dict__.pop('module_traits', None)
  super_source.__dict__.pop('_module_traits', None)
  return new_node


def _get_updated_fifo(
    fifo: ir.FIFO,
    *,
    read_module: Optional[ir.Module] = None,
    write_module: Optional[ir.Module] = None,
) -> ir.FIFO:
  """Return a shallow copy of ir.FIFO with read_module/write_module updated."""
  fifo = copy.copy(fifo)
  if read_module is not None:
    fifo.read_module = read_module
  if write_module is not None:
    fifo.write_module = write_module
  return fifo
