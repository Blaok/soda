#!/usr/bin/python3.6
import json
import math
import operator
import os
import sys
from fractions import Fraction
from functools import reduce

# constants
coords_in_tile = 'ijkl'
coords_in_orig = 'pqrs'
type_width = {'uint8_t':8, 'uint16_t':16, 'uint32_t':32, 'uint64_t':64, 'int8_t':8, 'int16_t':16, 'int32_t':32, 'int64_t':64, 'float':32, 'double':64}

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
        self.compute_content = kwargs.pop('compute_content')
        # parameters can be explored
        self.tile_size = kwargs.pop('tile_size')
        self.k = kwargs.pop('k')
        self.dram_separate = kwargs.pop('dram_separate')

class Printer(object):
    def __init__(self, out):
        self.out = out
        self.indent = 0

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

    def DoScope(self):
        self.PrintLine('{')
        self.DoIndent()

    def UnScope(self):
        self.UnIndent()
        self.PrintLine('}')

def GetCType(supo_type):
    if supo_type in {'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64'}:
        return supo_type+'_t'
    return supo_type

