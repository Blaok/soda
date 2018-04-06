#!/usr/bin/env python3
import json
import sys
from collections import OrderedDict

class XilinxHLSReport(object):
    def __init__(self, rpt_file):
        def get_resource_dict(resources, usages):
            usage = OrderedDict()
            for resource, used in zip(resources, usages):
                usage[resource] = int(used)
            return usage

        for line in rpt_file:
            if line.startswith('* Project:'):
                self.app = line.split(':')[1].split(' ')[-1][:-8]
            elif line == '== Utilization Estimates\n':
                self.resources_used = OrderedDict()
                self.resources_available = OrderedDict()
                self.resources_instances = OrderedDict()
            elif (hasattr(self, 'resources_used') and
                    len(self.resources_used) == 0 and 'Name' in line):
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
            elif line.startswith('    * Instance:'):
                self.instances = OrderedDict()
                self.instances[None] = None
            elif (hasattr(self, 'instances') and
                    None in self.instances and '|compute_' in line):
                pe_id = line.split('|')[2].split('_')[-2]
                if pe_id[-1] == 'u':
                    pe_id = int(pe_id[:-1])
                else:
                    pe_id = int(pe_id)
                func_name = '_'.join(line.split('|')[2].split('_')[1:-2])
                compute = self.instances.setdefault('compute', OrderedDict())
                compute = compute.setdefault(func_name, OrderedDict())
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
                                                    OrderedDict())
                forward = forward.setdefault(
                    int(line.split('|')[2].split('_')[1]), OrderedDict())
                forward = forward.setdefault(data_type, OrderedDict())
                forward = forward.setdefault(fifo_depth, [])
                forward.append(get_resource_dict(self.resources_used,
                                                 line.split('|')[3:-1]))
            elif (hasattr(self, 'instances') and
                    None in self.instances and '|'+self.app+'_kernel' in line):
                if line.split('|')[2].strip().endswith('_m_axi'):
                    m_axi = self.instances.setdefault('m_axi', OrderedDict())
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
                pack = self.instances.setdefault('pack', OrderedDict())
                pack = pack.setdefault(0, OrderedDict())
                pack = pack.setdefault('float', [])
                pack.append(get_resource_dict(self.resources_used,
                                              line.split('|')[3:-1]))
            elif (hasattr(self, 'instances') and
                    None in self.instances and '|unpack_' in line):
                unpack = self.instances.setdefault('unpack', OrderedDict())
                unpack = unpack.setdefault(0, OrderedDict())
                unpack = unpack.setdefault('float', [])
                unpack.append(get_resource_dict(self.resources_used,
                                              line.split('|')[3:-1]))
            elif (hasattr(self, 'instances') and
                    None in self.instances and '|Total' in line):
                del self.instances[None]
        dram_bank_i = sum(1 for _ in self.instances['m_axi'] if _.endswith('i'))
        dram_bank_o = sum(1 for _ in self.instances['m_axi'] if _.endswith('o'))
        unroll_factor = len(next(iter(self.instances['compute'])))
        pack = self.instances['pack']
        unpack = self.instances['unpack']
        pack[unroll_factor//dram_bank_o] = pack.pop(0)
        unpack[unroll_factor//dram_bank_i] = unpack.pop(0)

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    def check(self):
        def accumulate_resources(delta):
            for resource in resources:
                resources_used[resource] += delta[resource]
        def accumulate_resources_iterative(deltas):
            for delta in deltas:
                accumulate_resources(delta)
        resources = list(self.resources_instances)
        resources_used = OrderedDict()
        for resource in resources:
            resources_used[resource] = 0
        for module in ('Block_proc', 'entry', 'control_s_axi'):
            if module in self.instances:
                accumulate_resources(self.instances[module])
        for module in ('m_axi',):
            accumulate_resources_iterative(self.instances[module].values())
        for module in ('load', 'store'):
            accumulate_resources_iterative(self.instances[module])
        for module in ('unpack', 'pack'):
            for degree in self.instances[module].values():
                for data_type in degree.values():
                    accumulate_resources_iterative(data_type)
        for module in ('compute',):
            for var in self.instances[module].values():
                accumulate_resources_iterative(var.values())
        for module in ('forward',):
            for degree in self.instances[module].values():
                for data_type in degree.values():
                    for fifo_depth in data_type.values():
                        accumulate_resources_iterative(fifo_depth)
        for resource in resources:
            total = resources_used[resource]
            ground_truth = self.resources_instances[resource]
            if total != ground_truth:
                msg = '%s shall add up to %d, got %d' % (
                    resource, ground_truth, total)
                raise ReportSemanticError(msg)
        return True

class ReportSemanticError(Exception):
    pass

def main():
    for file_name in sys.argv[1:]:
        with open(file_name) as f:
            XilinxHLSReport(f).check()

if __name__ == '__main__':
    main()

