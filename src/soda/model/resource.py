#!/usr/bin/python3
import collections
import json
import sys

class XilinxHLSReport():
  def __init__(self, rpt_file):
    def get_resource_dict(resources, usages):
      usage = collections.OrderedDict()
      for resource, used in zip(resources, usages):
        usage[resource] = int(used)
      return usage

    for line in rpt_file:
      if line.startswith('* Project:'):
        self.app = line.split(':')[1].split(' ')[-1][:-8]
      elif line == '== Utilization Estimates\n':
        self.resources_used = collections.OrderedDict()
        self.resources_available = collections.OrderedDict()
        self.resources_instances = collections.OrderedDict()
      elif (hasattr(self, 'resources_used') and
          not self.resources_used and 'Name' in line):
        for resource in line.split('|')[2:-1]:
          resource = resource.strip()
          self.resources_used[resource] = None
          self.resources_available[resource] = None
          self.resources_instances[resource] = None
        self.resources_used[None] = None
        self.resources_available[None] = None
        self.resources_instances[None] = None
      elif (hasattr(self, 'resources_used') and
          None in self.resources_used and '|Total' in line):
        for resource, used in zip(self.resources_used,
                      line.split('|')[2:-1]):
          self.resources_used[resource] = int(used)
        del self.resources_used[None]
      elif (hasattr(self, 'resources_available') and
          None in self.resources_available and '|Available' in line):
        for resource, avail in zip(self.resources_available,
                       line.split('|')[2:-1]):
          self.resources_available[resource] = int(avail)
        del self.resources_available[None]
      elif (hasattr(self, 'resources_instances') and
          None in self.resources_instances and '|Instance' in line):
        for resource, avail in zip(self.resources_instances,
                       line.split('|')[2:-1]):
          self.resources_instances[resource] = int(avail)
        del self.resources_instances[None]
      elif line.startswith('  * Instance:'):
        self.instances = collections.OrderedDict()
        self.instances[None] = None
      elif (hasattr(self, 'instances') and
          None in self.instances and '|compute_' in line):
        pe_id = line.split('|')[2].split('_')[-2]
        if pe_id[-1] == 'u':
          pe_id = int(pe_id[:-1])
        else:
          pe_id = int(pe_id)
        func_name = '_'.join(line.split('|')[2].split('_')[1:-2])
        compute = self.instances.setdefault('compute',
                                            collections.OrderedDict())
        compute = compute.setdefault(func_name, collections.OrderedDict())
        compute[pe_id] = get_resource_dict(self.resources_used,
                           line.split('|')[3:-1])
      elif (hasattr(self, 'instances') and
          None in self.instances and '|forward_' in line):
        data_type = line.split('|')[2].split('_')[2]
        fifo_depth = line.split('|')[2].split('_')[3]
        if fifo_depth[-1] == 'u':
          fifo_depth = int(fifo_depth[:-1])
        else:
          fifo_depth = int(fifo_depth)
        forward = self.instances.setdefault('forward',
                          collections.OrderedDict())
        forward = forward.setdefault(
          int(line.split('|')[2].split('_')[1]), collections.OrderedDict())
        forward = forward.setdefault(data_type, collections.OrderedDict())
        forward = forward.setdefault(fifo_depth, [])
        forward.append(get_resource_dict(self.resources_used,
                         line.split('|')[3:-1]))
      elif (hasattr(self, 'instances') and
          None in self.instances and '|'+self.app+'_kernel' in line):
        if line.split('|')[2].strip().endswith('_m_axi'):
          m_axi = self.instances.setdefault('m_axi', collections.OrderedDict())
          m_axi[line.split('|')[2].split('_')[2]] = get_resource_dict(
            self.resources_used, line.split('|')[3:-1])
        elif line.split('|')[2].strip().endswith('_control_s_axi'):
          self.instances['control_s_axi'] = get_resource_dict(
            self.resources_used, line.split('|')[3:-1])
        elif self.app+'_kernel_entry' in line.split('|')[2]:
          self.instances['entry'] = get_resource_dict(
            self.resources_used, line.split('|')[3:-1])
      elif (hasattr(self, 'instances') and
          None in self.instances and '|Block_proc' in line):
        self.instances['Block_proc'] = get_resource_dict(
          self.resources_used, line.split('|')[3:-1])
      elif (hasattr(self, 'instances') and
          None in self.instances and '|load' in line):
        self.instances.setdefault('load', []).append(
            get_resource_dict(self.resources_used,
                      line.split('|')[3:-1]))
      elif (hasattr(self, 'instances') and
          None in self.instances and '|store' in line):
        self.instances.setdefault('store', []).append(
            get_resource_dict(self.resources_used,
                      line.split('|')[3:-1]))
      elif (hasattr(self, 'instances') and
          None in self.instances and '|pack_' in line):
        pack = self.instances.setdefault('pack', collections.OrderedDict())
        pack = pack.setdefault(0, collections.OrderedDict())
        pack = pack.setdefault('float', [])
        pack.append(get_resource_dict(self.resources_used,
                        line.split('|')[3:-1]))
      elif (hasattr(self, 'instances') and
          None in self.instances and '|unpack_' in line):
        unpack = self.instances.setdefault('unpack', collections.OrderedDict())
        unpack = unpack.setdefault(0, collections.OrderedDict())
        unpack = unpack.setdefault('float', [])
        unpack.append(get_resource_dict(self.resources_used,
                        line.split('|')[3:-1]))
      elif (hasattr(self, 'instances') and
          None in self.instances and '|Total' in line):
        del self.instances[None]
    dram_bank_i = sum(1 for bus in self.instances['m_axi']
              if bus.endswith('i'))
    dram_bank_o = sum(1 for bus in self.instances['m_axi']
              if bus.endswith('o'))
    unroll_factor = len(next(iter(self.instances['compute'])))
    pack = self.instances['pack']
    unpack = self.instances['unpack']
    pack[unroll_factor//dram_bank_o] = pack.pop(0)
    unpack[unroll_factor//dram_bank_i] = unpack.pop(0)

  def __str__(self):
    return json.dumps(self.__dict__, indent=2, sort_keys=True)

  def check(self):
    resources = list(self.resources_instances)
    resources_used = collections.OrderedDict()
    for resource in resources:
      resources_used[resource] = 0
    for module in ('Block_proc', 'entry', 'control_s_axi'):
      if module in self.instances:
        accum_res(resources_used,
                   self.instances[module])
    for module in ('m_axi',):
      accum_res_iter(resources_used,
                       self.instances[module].values())
    for module in ('load', 'store'):
      accum_res_iter(resources_used,
                       self.instances[module])
    for module in ('unpack', 'pack'):
      for degree in self.instances[module].values():
        for data_type in degree.values():
          accum_res_iter(resources_used, data_type)
    for module in ('compute',):
      for var in self.instances[module].values():
        accum_res_iter(resources_used, var.values())
    for module in ('forward',):
      for degree in self.instances[module].values():
        for data_type in degree.values():
          for fifo_depth in data_type.values():
            accum_res_iter(resources_used,
                             fifo_depth)
    for resource in resources:
      total = resources_used[resource]
      ground_truth = self.resources_instances[resource]
      if total != ground_truth:
        msg = '%s shall add up to %d, got %d' % (
          resource, ground_truth, total)
        raise ReportSemanticError(msg)
    return True

class XilinxPostRoutingReport():
  def __init__(self, rpt_file):
    for line in rpt_file:
      line_splited = line.split('|')
      if len(line_splited) > 1 and 'Name' in line_splited[1]:
        self.resources_used = collections.OrderedDict(
          (resource.strip(), None)
          for resource in line_splited[2:-1])
      elif (hasattr(self, 'resources_used') and
          len(line_splited) == len(self.resources_used)+3 and
          'Used Resources' in line):
        iterator = iter(line_splited[2:-1])
        for res in self.resources_used:
          self.resources_used[res] = next(iterator).strip().split()[0]

  def __str__(self):
    return json.dumps(self.__dict__, indent=2, sort_keys=True)

class ReportSemanticError(Exception):
  pass

def accum_res(total, delta, coefficient=1):
  for resource in total:
    total[resource] += delta[resource] * coefficient
def accum_res_iter(total, deltas, coefficient=1):
  for delta in deltas:
    accum_res(total, delta, coefficient)

def main():
  for file_name in sys.argv[1:]:
    with open(file_name) as f:
      if file_name.endswith('kernel_util_routed.rpt'):
        rpt = XilinxPostRoutingReport(f)
        print(rpt)
      elif file_name.endswith('_csynth.rpt'):
        rpt = XilinxHLSReport(f)
        rpt.check()
        print(json.dumps({'resources_hls_used': rpt.resources_used},
                 indent=2))

if __name__ == '__main__':
  main()
