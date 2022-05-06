import json
import logging
import os
import shutil
import tempfile
from typing import IO, TYPE_CHECKING, Dict, Optional

from haoda import util
from haoda.report.xilinx import hls as hls_report
from tapa import tapac

from soda.codegen.xilinx import hls_kernel

if TYPE_CHECKING:
  from soda.core import Stencil

_logger = logging.getLogger().getChild(__name__)


def print_code(
    stencil: 'Stencil',
    xo_file: IO[bytes],
    device_info: Dict[str, str],
    work_dir: Optional[str] = None,
    connectivity_file: Optional[str] = None,
    constraint_file: Optional[str] = None,
    interface: str = 'm_axi',
) -> None:
  """Generate hardware object file for the given Stencil.

  Working `vivado` and `vivado_hls` is required in the PATH.

  Args:
    stencil: Stencil object to generate from.
    xo_file: file object to write to.
    device_info: dict of 'part_num' and 'clock_period'.
    work_dir: path of the work directory; None creates a temporary one.
    interface: interface type, supported values are 'm_axi' and 'axis'.
  """

  # map HLS interfaces to TAPA interfaces
  interface = {
      'm_axi': 'tapa::mmap',
      'axis': 'tapa::stream',
  }.get(interface, interface)

  # create work directory if necessary
  if work_dir is None:
    tmp_dir = tempfile.TemporaryDirectory(prefix='sodac-xrtl-')
    work_dir = tmp_dir.name
  os.makedirs(work_dir, exist_ok=True)

  # create HLS kernel
  top_name = stencil.kernel_name
  kernel_file = os.path.join(work_dir, f'{top_name}.sodac.xrtl.cpp')
  with open(kernel_file, 'w') as kernel_fileobj:
    hls_kernel.print_code(stencil, kernel_fileobj, interface)
  xo_filename = os.path.join(work_dir, f'{top_name}.sodac.xrtl.xo')

  # run HLS
  argv = [
      kernel_file,
      '--output=' + xo_filename,
      '--work-dir=' + work_dir,
      '--top=' + top_name,
      '--part-num=' + device_info['part_num'],
      '--clock-period=' + device_info['clock_period'],
  ]
  tapac.main(argv + [
      '--run-tapacc',
      '--run-hls',
      '--generate-task-rtl',
  ])

  # read HLS reports
  super_source = stencil.dataflow_super_source
  depths: Dict[int, int] = {}
  total_resource = hls_report.HlsResources()
  total_resource.name = top_name
  for module_id, nodes in enumerate(super_source.module_trait_table.values()):
    module_name = util.get_func_name(module_id)
    report_file = os.path.join(work_dir, 'report', module_name + '_csynth.xml')
    hls_resource = hls_report.resources(report_file)
    use_count = len(nodes)
    total_resource += hls_resource * use_count
    try:
      perf = hls_report.performance(report_file)
      _logger.info('%s, usage: %5d times, II: %3d, Depth: %3d', hls_resource,
                   use_count, perf.ii, perf.depth)
      depths[module_id] = perf.depth
    except hls_report.BadReport as e:
      _logger.warn('%s in %s report (%s)', e, module_name, report_file)
      _logger.info('%s, usage: %5d times', hls_resource, use_count)
      raise e
  _logger.info('%s', total_resource)

  # update the module pipeline depths
  super_source.update_module_depths(depths)
  with open(os.path.join(work_dir, 'program.json')) as program_fp:
    program = json.load(program_fp)
  fifos = program['tasks'][top_name]['fifos']
  for module in super_source.tpo_valid_node_gen():
    for fifo in module.fifos:
      name = fifo.c_expr
      # fifo.depth is the "extra" capacity of a FIFO; the base depth is 3, 1 for
      # registering the input, 1 for keeping II=1 when FIFO is (almost) full, 1
      # for keeping II=1 when FIFO is relaxed from back pressure (necessary
      # because the optimal FIFO depths may require back pressure)
      fifos[name]['depth'] = fifo.depth + 3
  with open(os.path.join(work_dir, 'program.json'), 'w') as program_fp:
    json.dump(program, program_fp, indent=2)

  # create constraint file
  if constraint_file is not None:
    argv.extend((
        f'--connectivity={connectivity_file}',
        f'--constraint={constraint_file}',
        '--enable-synth-util',
        '--run-floorplanning',
    ))

  # pack xo
  tapac.main(argv + [
      '--generate-task-rtl',
      '--generate-top-rtl',
      '--pack-xo',
  ])
  with open(xo_filename, mode='rb') as xo_fp:
    shutil.copyfileobj(xo_fp, xo_file)
