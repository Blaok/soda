from typing import (
  BinaryIO,
  Dict,
  Optional,
  TextIO,
)

import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from haoda import util
from haoda.backend import xilinx as backend

REPORT_UTIL_COMMANDS = r'''
read_verilog [ glob {hdl_dir}/*.v ]
foreach tcl_file [ glob -nocomplain {hdl_dir}/*.tcl ] {{
  source ${{tcl_file}}
}}

synth_design {synth_args}
report_utilization {report_util_args}
'''

class ReportXoUtil(backend.Vivado):
  """Run synthesis and generate resource utilization report.

  This class is a child of subprocess.Popen and will launch Vivado to run
  syntheis and generate resource utilization report. Arguments passed to
  synth_design and report_utilization can be configured via the kwargs params.

  Attributes:
    tmpdir: Temporary working directory for the RTL files and generated report.
    rpt_file: File object of generated resource utilization report.
    rpt_file_name: Name of the generated temporary resource utilization report.
  """
  def __init__(self, xo_file: BinaryIO, rpt_file: TextIO,
               top_name: str = 'Dataflow', part_num: Optional[str] = None,
               synth_kwargs: Optional[Dict[str, str]] = None,
               report_util_kwargs: Optional[Dict[str, str]] = None):
    """Run synthesis and generate resource utilization report.

    Args:
      xo_file: XO file object containing the HDL files.
      rpt_file: File object of generated resource utilization report.
      top_name: Optionally specify a different top name other than "Dataflow".
      part_num: Optionally specify a different part number other than
          automatically determined.
      synth_kwargs: Dict of arguments for the synth_design command.
      report_util_kwargs: Dict of arguments for the synth_design command.

    Raises:
      InputError if input is not a valid XO.
    """
    self.tmpdir = tempfile.TemporaryDirectory(prefix='report-xo-util-')
    self.rpt_file = rpt_file
    self.rpt_file_name = os.path.join(self.tmpdir.name, 'post_synth_util.rpt')
    with zipfile.ZipFile(xo_file) as xo_zip:
      with xo_zip.open('xo.xml') as xo_xml:
        kernel = ET.parse(xo_xml).find('./Kernels/Kernel')
        if kernel is None:
          raise util.InputError('cannot parse XO file')
        ip_dir = kernel.attrib['IP']
        kernel_name = kernel.attrib['Name']
      hdl_dir = os.path.join(self.tmpdir.name, ip_dir, 'src')
      xo_zip.extractall(path=self.tmpdir.name, members=[
        name for name in xo_zip.namelist() if name.startswith(ip_dir + '/src')])
      if part_num is None:
        with open(os.path.join(hdl_dir, kernel_name + '.v')) as hdl_file:
          part_num = RtlHlsInfo(hdl_file)['HLS_INPUT_PART']
    if synth_kwargs is None:
      synth_kwargs = {}
    if report_util_kwargs is None:
      report_util_kwargs = {}
    synth_kwargs.setdefault('top', top_name)
    synth_kwargs.setdefault('part', part_num)
    synth_kwargs.setdefault('directive', 'RuntimeOptimized')
    report_util_kwargs.setdefault('hierarchical', '')
    report_util_kwargs.setdefault('file', self.rpt_file_name)

    synth_args = ' '.join('-{} {}'.format(*kv) for kv in synth_kwargs.items())
    report_util_args = ' '.join('-{} {}'.format(*kv)
                                for kv in report_util_kwargs.items())
    kwargs = {
      'output_dir': os.path.join(self.tmpdir.name, 'output'),
      'hdl_dir': hdl_dir,
      'synth_args': synth_args,
      'report_util_args': report_util_args,
    }
    super().__init__(REPORT_UTIL_COMMANDS.format(**kwargs))

  def __exit__(self, *args):
    super().__exit__(*args)
    try:
      with open(self.rpt_file_name) as src_rpt_file:
        shutil.copyfileobj(src_rpt_file, self.rpt_file)
    except FileNotFoundError:
      raise util.InternalError('failed to generated report file')
    self.tmpdir.cleanup()

RTL_HLS_INFO_REGEX = r'\(\* CORE_GENERATION_INFO=".*,\{(.*)\}" \*\)'

class RtlHlsInfo:
  def __init__(self, rtl_file: TextIO):
    match = re.search(RTL_HLS_INFO_REGEX, rtl_file.read())
    if match is None:
      raise util.InputError('cannot parse RTL file')
    for item in match.group(1).split(','):
      key, value = item.split('=')
      setattr(self, key, value)

  def __getitem__(self, key: str) -> str:
    return getattr(self, key)
