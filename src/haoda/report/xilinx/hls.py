#!/usr/bin/python3
from typing import (
    Iterator,
    Optional,
    TextIO,
    Tuple,
    Union
)

import io
import sys
import xml.etree.ElementTree as ET

class BadReport(Exception):
  pass

class HlsResources:
  """An object representing the HLS resource estimation.

  This class supports add and sub between the same class and mul with a number.
  URAM is not supported yet.

  Attributes:
    RESOURCES: Static, resource attribute names expected.
    name: Optional name of the module.
  """
  RESOURCES = 'FF', 'LUT', 'BRAM_18K', 'DSP48E'
  def __init__(self, obj: Union['HlsResources', ET.Element, TextIO,
                                None] = None) -> None:
    """Initialize an HlsResources from another HlsResources, a report or None.

    If obj is another HlsResources, it will be copied.
    If obj is an xml.etree.ElementTree.Element, it has to be pointing to a
      valid HLS report XML tree.
    If obj is an io.TextIOBase, it will be parsed as a valid HLS report XML.
    If obj is None, all resources will be 0.

    Args:
      obj: Object used for initialization.

    Raises:
      TypeError: If obj is not of a correct type.
    """
    self.name = None  # type: Optional[str]
    if isinstance(obj, HlsResources):
      self.name = obj.name
      for resource in HlsResources.RESOURCES:
        self[resource] = obj[resource]
      return
    if isinstance(obj, ET.Element):
      self.init_from_xml_element(obj)
      return
    if isinstance(obj, io.TextIOBase):
      self.init_from_xml_element(ET.parse(obj).getroot())
      return
    if obj is None:
      for resource in HlsResources.RESOURCES:
        self[resource] = 0
      return
    raise TypeError('obj must be '
                    'an xml.etree.ElementTree.Element object or '
                    'another HlsResources object or '
                    'an io.TextIOBase object or '
                    'None')

  def init_from_xml_element(self, elem: ET.Element,
                            item: str = 'Resources') -> 'HlsResources':
    """Initialize self from an xml.etree.ElementTree.Element.

    Initialize self from an xml.etree.ElementTree.Element with the option of
    reading the used resources or the available resources.

    Args:
      elem: Must be an xml.etree.ElementTree.Element pointing to a valid HLS
      report XML tree.
      item: String specifying what to read. Valid values are 'Resources' and
        'AvailableResources'.

    Returns:
      Updated self.
    """
    if item == 'Resources':
      self.name = elem.findtext('UserAssignments/TopModelName')
    for resource in HlsResources.RESOURCES:
      self[resource] = int(elem.findtext(
          '/'.join(('AreaEstimates', item, resource))))
    return self

  def __getitem__(self, key: str) -> int:
    if key in HlsResources.RESOURCES:
      return getattr(self, key)
    raise ValueError('invalid key: %s' % key)

  def __setitem__(self, key: str, value: int) -> int:
    if key in HlsResources.RESOURCES:
      setattr(self, key, value)
      return value
    raise ValueError('invalid key: %s' % key)

  def __iter__(self) -> Iterator[Tuple[str, int]]:
    return ((r, self[r]) for r in HlsResources.RESOURCES)

  def __str__(self) -> str:
    return 'module: %16s, %s' % (
        self.name or 'n/a', ', '.join('%s: %7d' % item for item in self))

  def __add__(self, other: 'HlsResources') -> 'HlsResources':
    result = HlsResources(self)
    for resource in HlsResources.RESOURCES:
      result[resource] += other[resource]
    return result

  def __sub__(self, other: 'HlsResources') -> 'HlsResources':
    result = HlsResources(self)
    for resource in HlsResources.RESOURCES:
      result[resource] -= other[resource]
    return result

  def __mul__(self, other: float) -> 'HlsResources':
    result = HlsResources(self)
    for resource in HlsResources.RESOURCES:
      result[resource] = int(result[resource] * other)
    return result

  def __eq__(self, other) -> bool:
    if not isinstance(other, HlsResources):
      raise NotImplementedError
    for resource in HlsResources.RESOURCES:
      if self[resource] != other[resource]:
        return False
    return True

  def __hash__(self) -> int:
    return hash(tuple(self[r] for r in HlsResources.RESOURCES))

class HlsPerformance:
  """An object representing the HLS performance estimation.

  Attributes:
    name: Optional name of the module.
    ii: Integer of pipeline II.
    depth: Integer of pipeline depth.
  """
  def __init__(self, obj: Union['HlsPerformance', ET.Element, TextIO,
                                None] = None) -> None:
    """Initialize an HlsPerformance from another HlsPerformance, a report, or
    None.

    If obj is another HlsPerformance, it will be copied.
    If obj is an xml.etree.ElementTree.Element, it has to be pointing to a
      valid HLS report XML tree.
    If obj is an io.TextIOBase, it will be parsed as a valid HLS report XML.
    If obj is None, the ii and depth will be 0.

    Args:
      obj: Object used for initialization.

    Raises:
      TypeError: If obj is not of a correct type.
    """
    self.name, self.ii, self.depth = None, 0, 0
    if isinstance(obj, HlsPerformance):
      self.name, self.ii, self.depth = obj.name, obj.ii, obj.depth
      return
    if isinstance(obj, ET.Element):
      self.init_from_xml_element(obj)
      return
    if isinstance(obj, io.TextIOBase):
      self.init_from_xml_element(ET.parse(obj).getroot())
      return
    if obj is None:
      return
    raise TypeError('obj must be '
                    'an xml.etree.ElementTree.Element object or '
                    'another HlsPerformances object or '
                    'an io.TextIOBase object or '
                    'None')

  def init_from_xml_element(self, elem: ET.Element) -> 'HlsPerformance':
    """Initialize self from an xml.etree.ElementTree.Element.

    Initialize self from an xml.etree.ElementTree.Element.

    Args:
      elem: Must be an xml.etree.ElementTree.Element pointing to a valid HLS
      report XML tree.

    Returns:
      Updated self.
    """
    def raise_if_not_found(item: ET.Element, key: str) -> int:
      val = item.findtext(key)
      if val is None:
        raise BadReport('cannot find %s')
      return int(val)
    for item in elem.findall('PerformanceEstimates/SummaryOfLoopLatency/*'):
      self.ii = raise_if_not_found(item, 'PipelineII')
      self.depth = raise_if_not_found(item, 'PipelineDepth')
    return self

def resources(obj: Union[TextIO, str]) -> HlsResources:
  """Read HLS resource estimation from HLS report XML.

  Args:
    obj: A file-like object or a file path of HLS report XML file, which is used
      to parse the resource usage.

  Returns:
    The HLS resource estimation.

  Raises:
    TypeError: If obj is not a text file-like object nor a path as str.
  """
  if isinstance(obj, io.TextIOBase):
    return HlsResources(obj)
  if isinstance(obj, str):
    with open(obj, 'r') as fd:
      result = resources(fd)
    return result
  raise TypeError('obj must be a text file-like object or a path as str')

def performance(obj: Union[TextIO, str]) -> HlsPerformance:
  """Read HLS performance estimation from HLS report XML.

  Args:
    obj: A file-like object or a file path of HLS report XML file, which is used
      to parse the performance estimation.

  Returns:
    The HLS performance estimation.

  Raises:
    TypeError: If obj is not a text file-like object nor a path as str.
  """
  if isinstance(obj, io.TextIOBase):
    return HlsPerformance(obj)
  if isinstance(obj, str):
    with open(obj, 'r') as fd:
      result = performance(fd)
    return result
  raise TypeError('obj must be a text file-like object or a path as str')

def available_resources(obj: Union[TextIO, str]) -> HlsResources:
  """Read available resources from HLS report XML.

  Args:
    obj: A file-like object or a file path of HLS report XML file, which is used
      to parse the available resource.

  Returns:
    The HLS available resource.

  Raises:
    TypeError: If obj is not a text file-like object nor a path as str.
  """
  if isinstance(obj, io.TextIOBase):
    return HlsResources().init_from_xml_element(
        ET.parse(obj).getroot(), 'AvailableResources')
  if isinstance(obj, str):
    with open(obj, 'r') as fd:
      result = available_resources(fd)
    return result
  raise TypeError('obj must be a text file-like object or a path as str')

def main() -> None:
  if sys.argv[1:]:
    total = available_resources(sys.argv[1])
    total_used = HlsResources()
    for f in sys.argv[1:]:
      used = resources(f)
      total_used += used
      print('     used:', used)
    print('    total:', total_used)
    print('available:', total)

if __name__ == '__main__':
  main()
