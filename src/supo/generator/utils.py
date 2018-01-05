#!/usr/bin/python3.6
from fractions import Fraction
from functools import reduce
import json
import logging
import math
import operator
import os
import sys

# constants
coords_tiled = 'xyzw'
coords_in_tile = 'ijkl'
coords_in_orig = 'pqrs'
type_width = {'uint8_t':8, 'uint16_t':16, 'uint32_t':32, 'uint64_t':64, 'int8_t':8, 'int16_t':16, 'int32_t':32, 'int64_t':64, 'float':32, 'double':64}
max_dram_chan = 4

logger = logging.getLogger('__main__').getChild(__name__)

class Stencil(object):
    def __init__(self, **kwargs):
        # platform determined
        self.burst_width = kwargs.pop('burst_width')
        self.dram_chan = kwargs.pop('dram_chan')
        # application determined
        self.app_name = kwargs.pop('app_name')
        self.input_name = GetCType(kwargs.pop('input_name'))
        self.input_type = GetCType(kwargs.pop('input_type'))
        self.input_chan = kwargs.pop('input_chan')
        self.output_name = GetCType(kwargs.pop('output_name'))
        self.output_type = GetCType(kwargs.pop('output_type'))
        self.output_chan = kwargs.pop('output_chan')
        self.A = kwargs.pop('A')
        self.dim = kwargs.pop('dim')
        self.extra_params = kwargs.pop('extra_params')
        if not self.extra_params:
            self.extra_params = []
        self.compute_content = kwargs.pop('compute_content')
        # parameters can be explored
        self.tile_size = kwargs.pop('tile_size')
        self.k = kwargs.pop('k')
        self.dram_separate = kwargs.pop('dram_separate')
        if self.dram_separate:
            if self.dram_chan%2 != 0:
                logging.getLogger(__name__).fatal('Number of DRAM channels has to be even when separated')
                sys.exit(-1)
            else:
                self.dram_chan = int(self.dram_chan/2)

class Printer(object):
    def __init__(self, out):
        self.out = out
        self.indent = 0
        self.assign = 0
        self.comments = []

    def PrintLine(self, line = '', local_indent = -1):
        if local_indent < 0:
            local_indent = self.indent
        if line:
            self.out.write('%s%s\n' % (' '*local_indent*4, line))
        else:
            self.out.write('\n')

    def DoIndent(self):
        self.indent += 1

    def UnIndent(self):
        self.indent -= 1

    def DoScope(self, comment=''):
        self.PrintLine('{')
        self.DoIndent()
        self.comments.append(comment)

    def UnScope(self, comment=''):
        self.UnIndent()
        popped_comment = self.comments.pop()
        if comment:
            self.PrintLine('} // %s' % comment)
        else:
            if popped_comment:
                self.PrintLine('} // %s' % popped_comment)
            else:
                self.PrintLine('}')

    def NewVar(self):
        self.assign += 1
        return self.LastVar()

    def LastVar(self, offset=-1):
        return 'assign_%d' % (self.assign+offset)

def GetCType(supo_type):
    if supo_type in {'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'}:
        return supo_type+'_t'
    return supo_type

def GetStencilFromJSON(json_file):
    config = json.loads(json_file.read())
    input_type = config['input_type']
    output_type = config['output_type']
    dram_chan = int(os.environ.get('DRAM_CHAN', config['dram_chan']))
    dram_separate = ('DRAM_SEPARATE' in os.environ) or ('dram_separate' in config and config['dram_separate'])
    app_name = config['app_name']
    tile_size = config.get('St', [0]*config['dim'])
    A = config['A']
    k = int(os.environ.get('UNROLL_FACTOR', config['k']))
    burst_width = config['burst_width']

# generate compute content if it is not present in the config but iter and coeff are
    compute_content = []
    if 'compute_content' in config:
        compute_content = config['compute_content']
    elif ('iter' in config or 'ITER' in os.environ) and 'coeff' in config:
        num_iter = int(os.environ.get('ITER', config['iter']))
        coeff = [Fraction(x) for x in config['coeff']]
        if len(A) != len(coeff):
            sys.stderr.write('A and coeff must have the same length\n')
            sys.exit(1)
        A = {str(a):(a, c) for a, c in zip(A, coeff)}
        A = (GetA(A, num_iter))
        final_var = GetAdderTree(['input_%s*%ff' % ('_'.join([str(x) for x in a[0]]), (1.*a[1].numerator)/a[1].denominator) for a in A.values()], 0)
        compute_content += ['output_type output_point = %s;' % (final_var[0],), 'output[0][output_index_offset] = output_point;' ,'']
        A = [a[0] for a in A.values()]

    extra_params = config.get('extra_params', None)
    i = 0
    while (i < config['dim']-1 and ('TILE_SIZE_DIM_%d' % i) in os.environ):
        tile_size[i] = int(os.environ[('TILE_SIZE_DIM_%d' % i)])
        i += 1

    stencil = Stencil(
        burst_width = burst_width,
        app_name = app_name,
        input_name = 'input',
        input_type = input_type[0],
        input_chan = input_type[1],
        output_name = 'output',
        output_type = output_type[0],
        output_chan = output_type[1],
        A = A,
        dim = config['dim'],
        extra_params = extra_params,
        compute_content = compute_content,
        dram_chan = dram_chan,
        tile_size = tile_size,
        k = k,
        dram_separate = dram_separate)

    return stencil

def PrintGuard(printer, var, val):
    printer.PrintLine('#if %s != %d' % (var, val))
    printer.PrintLine('#error %s != %d' % (var, val))
    printer.PrintLine('#endif//%s != %d' % (var, val))

def PrintDefine(printer, var, val):
    printer.PrintLine('#ifndef %s' % var)
    printer.PrintLine('#define %s %d' % (var, val))
    printer.PrintLine('#endif//%s' % var)

def Serialize(vec, tile_size):
    # convert vec to scalar coordinates
    result = vec[0]
    for i in range(1, len(tile_size)):
        result += vec[i]*reduce(operator.mul, tile_size[0:i])
    return result

def GetStencilDistance(A, tile_size):
    A_serialized = [Serialize(x, tile_size) for x in A]
    return max(A_serialized) - min(A_serialized)

def GetStencilDim(A):
    return [max_index-min_index+1 for max_index, min_index in zip([max([point[dim] for point in A]) for dim in range(len(next(iter(A))))], [min([point[dim] for point in A]) for dim in range(len(next(iter(A))))])]

