import json
import logging
import math
import os
import subprocess

from soda import core
from soda import dataflow
from soda.model import resource

logger = logging.getLogger().getChild(__name__)

record = {}

def accumulate_fifo(estimation_routed, width, depth):
  fifo_size = depth*width
  if fifo_size > 1024:
    inc = 16*1024//width
    modulo_depth = math.ceil(depth/inc)*inc
    query_proc = subprocess.Popen([
      os.environ['XILINX_BRAM_MODEL_PATH'] + '/query',
      str(width), str(modulo_depth)], stdout=subprocess.PIPE, bufsize=0)
    delta = json.loads(query_proc.stdout.read())
    delta = (delta['fifo_w%d_d%d' % (width, modulo_depth)]['BRAM_18K'] +
         delta['fifo_w%d_d%d' % (width, modulo_depth)]['BRAM_36K']*2)

    estimation_routed['BRAM'] += delta
    record.setdefault((width, depth, delta), 0)
    record[(width, depth, delta)] += 1
  else:
    estimation_routed['REG'] += width
    estimation_routed['LUT'] += depth/2

def print_estimation(stencil, model_file, output_file):
  model = json.load(model_file)
  if stencil.app_name not in model['compute']:
    msg = ('stencil app %s not found in model %s' %
      (stencil.app_name, model_file.name))
    raise core.SemanticError(msg)
  if model['vendor'] != 'xilinx':
    msg = 'cannot find a model for Xilinx in model %s' % model_file.name
    raise core.SemanticError(msg)
  estimation = {
    'resource_hls': {
      'BRAM_18K': 0,
      'DSP48E': 0,
      'FF': 0,
      'LUT': 0
    },
    'resource_routed': {
      'BRAM': 0,
      'DSP': 0,
      'LUT': 0,
      'LUTMem': 0,
      'REG': 0
    },
  }
  estimation_hls = estimation['resource_hls']

  # these modules appear and only appear once
  for module in ('Block_proc', 'control_s_axi', 'entry', 'top_level'):
    resource.accum_res(estimation_hls, model[module])

  # axi master
  resource.accum_res(estimation_hls, model['m_axi'],
             stencil.dram_bank*2)

  # load/store
  resource.accum_res(estimation_hls, model['load'],
             stencil.input.chan*stencil.dram_bank)
  resource.accum_res(estimation_hls, model['store'],
             stencil.output.chan*stencil.dram_bank)

  # unpack/pack
  resource.accum_res(estimation_hls,
             model['pack_'+stencil.output.type]['base'],
             stencil.output.chan*stencil.dram_bank)
  resource.accum_res(estimation_hls,
             model['pack_'+stencil.output.type]['directly'],
             stencil.output.chan*stencil.unroll_factor)
  resource.accum_res(estimation_hls,
             model['pack_'+stencil.output.type]['inversely'],
             stencil.output.chan/stencil.unroll_factor)
  resource.accum_res(estimation_hls,
             model['unpack_'+stencil.input.type]['base'],
             stencil.input.chan*stencil.dram_bank)
  resource.accum_res(estimation_hls,
             model['unpack_'+stencil.input.type]['directly'],
             stencil.input.chan*stencil.unroll_factor)
  resource.accum_res(estimation_hls,
             model['unpack_'+stencil.input.type]['inversely'],
             stencil.input.chan/stencil.unroll_factor)

  # modules in the dataflow graph
  super_source = stencil.dataflow_super_source
  if stencil.replication_factor is not None:
    repli = stencil.replication_factor
  else:
    repli = 1
  for node in super_source.bfs_node_generator():
    # forward modules
    if isinstance(node, dataflow.ForwardNode):
      forward_key = 'forward_'+node.tensor.type
      if forward_key not in model:
        msg = 'forward<%s> not found in model %s' % (node.tensor.type,
                               model_file.name)
        raise core.SemanticError(msg)
      resource.accum_res(estimation_hls, model[forward_key]['base'],
                 repli)
      resource.accum_res(estimation_hls,
                 model[forward_key]['overhead_per_dst'],
                 len(node.children)*repli)
      if node.depth > 0:
        resource.accum_res(estimation_hls,
                   model[forward_key]['overhead_of_depth'],
                   repli)
    elif isinstance(node, dataflow.ComputeNode):
      if node.stage.name in (stencil.output.name,):
        stage_name = node.stage.name
      elif stencil.input.name+'_iter' in node.stage.name:
        stage_name = stencil.output.name
      elif node.stage.name.split('_iter')[0] in stencil.tensors:
        stage_name = node.stage.name.split('_iter')[0]
      else:
        msg = 'unknown dataflow compute node: %s' % node.stage.name
        raise core.SemanticError(msg)
      resource.accum_res(estimation_hls,
                 model['compute'][stencil.app_name][stage_name],
                 repli)

  performance = stencil.unroll_factor*model['target_freq']/10**3  # pixel/ns
  dram_bandwidth = model['dram_bandwidth']
  if stencil.dram_separate:
    performance = min(performance,
      dram_bandwidth*stencil.dram_bank/
        (stencil.input.chan*core.TYPE_WIDTH[stencil.input.type]/8),
      dram_bandwidth*stencil.dram_bank/
        (stencil.output.chan*core.TYPE_WIDTH[stencil.output.type]/8))
  else:
    performance = min(performance,
      dram_bandwidth*stencil.dram_bank/(
        stencil.input.chan*core.TYPE_WIDTH[stencil.input.type]/8+
        stencil.output.chan*core.TYPE_WIDTH[stencil.output.type]/8))
  performance *= stencil.iterate

  estimation_routed = estimation['resource_routed']

  estimation_routed['DSP'] = estimation_hls['DSP48E']
  estimation_routed['BRAM'] = estimation_hls['BRAM_18K']
  estimation_routed['REG'] = (estimation_hls['FF'] *
                model['ff_coeff'] +
                model['ff_base'])
  estimation_routed['LUT'] = (estimation_hls['LUT'] *
                model['lut_coeff'] +
                model['lut_base'])
  estimation_routed['LUTMem'] = estimation_hls['LUT']*model['lutmem_coeff']

  # BRAMs
  for src_node, dst_node in super_source.bfs_edge_generator():
    if isinstance(dst_node, dataflow.ForwardNode):
      for c in range(dst_node.tensor.chan*repli):
        accumulate_fifo(estimation_routed,
                core.TYPE_WIDTH[dst_node.tensor.type],
                max(dst_node.depth, 1))
    elif (isinstance(src_node, dataflow.ForwardNode) and
        isinstance(dst_node, dataflow.ComputeNode)):
      extra_depth = super_source.get_extra_depth((src_node, dst_node))
      for c in range(src_node.tensor.chan*repli):
        accumulate_fifo(estimation_routed,
                core.TYPE_WIDTH[src_node.tensor.type],
                max(extra_depth+1, 2))

  estimation_routed['BRAM'] = math.ceil(estimation_routed['BRAM']/2)

  available_hls = model['available_hls']
  available_routed = model['available_routed']
  permissible = (
    all(estimation_hls[resource] < available_hls[resource]
      for resource in estimation_hls) and
    all(estimation_routed[resource] < available_routed[resource]
      for resource in estimation_routed))
  confidence = lambda a, b: math.atan2(abs(a-b)*64, b)/math.pi*2  # .9 -> .9
  permissible_confidence = min(
    *[
      confidence(estimation_hls[resource], available_hls[resource])
      for resource in estimation_hls
    ], *[
      confidence(estimation_routed[resource], available_routed[resource])
      for resource in estimation_routed
    ])
  estimation['permissible'] = permissible
  estimation['permissible_confidence'] = permissible_confidence
  estimation['performance'] = performance
  estimation['performance_unit'] = 'pixel/ns'

  json.dump(estimation, output_file, indent=2, sort_keys=True)
