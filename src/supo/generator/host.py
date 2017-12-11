#!/usr/bin/python3.6
import json
import math
import operator
import os
import sys
from fractions import Fraction
from functools import reduce
sys.path.append(os.path.dirname(__file__))
from utils import coords_in_tile, coords_in_orig, type_width, Stencil, Printer, GetStencilFromJSON, PrintDefine, PrintGuard, Serialize, GetStencilDistance, GetStencilDim

def PrintHeader(p):
    for header in ['assert', 'float', 'math', 'stdbool', 'stdint', 'stdio', 'stdlib', 'string', 'fcntl', 'time', 'unistd', 'sys/types', 'sys/stat', 'CL/opencl']:
        p.PrintLine('#include<%s.h>' % header)
    p.PrintLine()

def PrintLoadXCLBIN2(p):
    p.PrintLine('int load_xclbin2_to_memory(const char *filename, char **result, char** device)')
    p.DoScope()
    p.PrintLine('uint64_t size = 0;')
    p.PrintLine('FILE *f = fopen(filename, "rb");')
    p.PrintLine('if(nullptr == f)')
    p.DoScope()
    p.PrintLine('*result = nullptr;')
    p.PrintLine('fprintf(stderr, "ERROR: cannot open %s\\n", filename);')
    p.PrintLine('return -1;')
    p.UnScope()
    p.PrintLine('char magic[8];')
    p.PrintLine('unsigned char cipher[32];')
    p.PrintLine('unsigned char key_block[256];')
    p.PrintLine('uint64_t unique_id;')
    p.PrintLine('fread(magic, sizeof(magic), 1, f);')
    p.PrintLine('if(strcmp(magic, "xclbin2")!=0)')
    p.DoScope()
    p.PrintLine('*result = nullptr;')
    p.PrintLine('fprintf(stderr, "ERROR: %s is not a valid xclbin2 file\\n", filename);')
    p.PrintLine('return -2;')
    p.UnScope()
    p.PrintLine('fread(cipher, sizeof(cipher), 1, f);')
    p.PrintLine('fread(key_block, sizeof(key_block), 1, f);')
    p.PrintLine('fread(&unique_id, sizeof(unique_id), 1, f);')
    p.PrintLine('fread(&size, sizeof(size), 1, f);')
    p.PrintLine('char* p = new char[size+1]();')
    p.PrintLine('*result = p;')
    p.PrintLine('memcpy(p, magic, sizeof(magic));')
    p.PrintLine('p += sizeof(magic);')
    p.PrintLine('memcpy(p, cipher, sizeof(cipher));')
    p.PrintLine('p += sizeof(cipher);')
    p.PrintLine('memcpy(p, key_block, sizeof(key_block));')
    p.PrintLine('p += sizeof(key_block);')
    p.PrintLine('memcpy(p, &unique_id, sizeof(unique_id));')
    p.PrintLine('p += sizeof(unique_id);')
    p.PrintLine('memcpy(p, &size, sizeof(size));')
    p.PrintLine('p += sizeof(size);')
    p.PrintLine('uint64_t size_left = size - sizeof(magic) - sizeof(cipher) - sizeof(key_block) - sizeof(unique_id) - sizeof(size);')
    p.PrintLine('if(size_left != fread(p, sizeof(char), size_left, f))')
    p.DoScope()
    p.PrintLine('delete[] p;')
    p.PrintLine('fprintf(stderr, "ERROR: %s is corrupted\\n", filename);')
    p.PrintLine('return -3;')
    p.UnScope()
    p.PrintLine('*device = p + 5*8;')
    p.PrintLine('printf("%lu %s\\n", size, *device);')
    p.PrintLine('fclose(f);')
    p.PrintLine('return size;')
    p.UnScope()
    p.PrintLine()

def PrintHalideRewriteBuffer(p):
    p.PrintLine('static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,')
    for i in range(0, 4):
        p.PrintLine('                                  int32_t min%d, int32_t extent%d, int32_t stride%d%c' % (i, i, i, ",,,)"[i]))
    p.DoScope()
    for item in ['min', 'extent', 'stride']:
        for i in range(0, 4):
            p.PrintLine('b->%s[%d] = %s%d;' % (item, i, item, i))
    p.PrintLine('return true;')
    p.UnScope()
    p.PrintLine()

def PrintHalideErrorCodes(p):
    i = 0
    for item in ['success', 'generic_error', 'explicit_bounds_too_small', 'bad_elem_size', 'access_out_of_bounds', 'buffer_allocation_too_large', 'buffer_extents_too_large', 'constraints_make_required_region_smaller', 'constraint_violated', 'param_too_small', 'param_too_large', 'out_of_memory', 'buffer_argument_is_null', 'debug_to_file_failed', 'copy_to_host_failed', 'copy_to_device_failed', 'device_malloc_failed', 'device_sync_failed', 'device_free_failed', 'no_device_interface', 'matlab_init_failed', 'matlab_bad_param_type', 'internal_error', 'device_run_failed', 'unaligned_host_ptr', 'bad_fold', 'fold_factor_too_small']:
        p.PrintLine('int halide_error_code_%s = %d;' % (item, i))
        i -= 1
    p.PrintLine()

def PrintHalideErrorReport(p):
    p.PrintLine('FILE* const* error_report = &stderr;')
    p.PrintLine()
    p.PrintLine('int halide_error_bad_elem_size(void *user_context, const char *func_name,')
    p.PrintLine('                               const char *type_name, int elem_size_given, int correct_elem_size) {')
    p.PrintLine('    fprintf(*error_report, "%s has type %s but elem_size of the buffer passed in is %d instead of %d",')
    p.PrintLine('            func_name, type_name, elem_size_given, correct_elem_size);')
    p.PrintLine('    return halide_error_code_bad_elem_size;')
    p.PrintLine('}')
    p.PrintLine('int halide_error_constraint_violated(void *user_context, const char *var, int val,')
    p.PrintLine('                                     const char *constrained_var, int constrained_val) {')
    p.PrintLine('    fprintf(*error_report, "Constraint violated: %s (%d) == %s (%d)",')
    p.PrintLine('            var, val, constrained_var, constrained_val);')
    p.PrintLine('    return halide_error_code_constraint_violated;')
    p.PrintLine('}')
    p.PrintLine('int halide_error_buffer_allocation_too_large(void *user_context, const char *buffer_name, uint64_t allocation_size, uint64_t max_size) {')
    p.PrintLine('    fprintf(*error_report, "Total allocation for buffer %s is %lu, which exceeds the maximum size of %lu",')
    p.PrintLine('            buffer_name, allocation_size, max_size);')
    p.PrintLine('    return halide_error_code_buffer_allocation_too_large;')
    p.PrintLine('}')
    p.PrintLine('int halide_error_buffer_extents_too_large(void *user_context, const char *buffer_name, int64_t actual_size, int64_t max_size) {')
    p.PrintLine('    fprintf(*error_report, "Product of extents for buffer %s is %ld, which exceeds the maximum size of %ld",')
    p.PrintLine('            buffer_name, actual_size, max_size);')
    p.PrintLine('    return halide_error_code_buffer_extents_too_large;')
    p.PrintLine('}')
    p.PrintLine('int halide_error_access_out_of_bounds(void *user_context, const char *func_name, int dimension, int min_touched, int max_touched, int min_valid, int max_valid) {')
    p.PrintLine('    if (min_touched < min_valid) {')
    p.PrintLine('        fprintf(*error_report, "%s is accessed at %d, which is before the min (%d) in dimension %d", func_name, min_touched, min_valid, dimension);')
    p.PrintLine('    } else if (max_touched > max_valid) {')
    p.PrintLine('        fprintf(*error_report, "%s is acccessed at %d, which is beyond the max (%d) in dimension %d", func_name, max_touched, max_valid, dimension);')
    p.PrintLine('    }')
    p.PrintLine('    return halide_error_code_access_out_of_bounds;')
    p.PrintLine('}')
    p.PrintLine()

def PrintWrapped(p, stencil):
    buffers = [[stencil.input_name, stencil.input_type], [stencil.output_name, stencil.output_type]]+stencil.extra_params
    p.PrintLine('static int blur_wrapped(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % ''.join([('buffer_t *var_%s_buffer, ') % x[0] for x in buffers]))
    p.DoScope()
    for b in buffers:
        PrintUnloadBuffer(p, b[0], b[1])

    p.PrintLine('if (var_%s_host_and_dev_are_null)' % stencil.output_name)
    p.DoScope()
    output_str = [", 0, 0, 0"]*4
    for dim in range(0, stencil.dim):
        stride  = '1'
        if dim > 0:
            stride = '*'.join([('%s_size_dim_%d' % (stencil.output_name, x)) for x in range(0, dim)])
        output_str[dim] = (', var_%s_min_%d, %s_size_dim_%d, %s' % (stencil.output_name, dim, stencil.output_name, dim, stride))
    p.PrintLine('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.NewVar(), stencil.output_name, type_width[stencil.output_type]/8, output_str[0], output_str[1], output_str[2], output_str[3]))
    p.PrintLine('(void)%s;' % p.LastVar())
    p.UnScope('if (var_%s_host_and_dev_are_null)' % stencil.output_name)

    p.PrintLine('if (var_%s_host_and_dev_are_null)' % stencil.input_name)
    p.DoScope()
    input_size = ['0']*4
    for dim in range(0, stencil.dim):
        p.PrintLine('int32_t %s = %s_size_dim_%d + %d;' % (p.NewVar(), stencil.output_name, dim, GetStencilDim(stencil.A)[dim]-1))

    input_str = [', 0, 0, 0']*4
    for dim in range(0, stencil.dim):
        stride  = '1'
        if dim > 0:
            stride = '*'.join([p.LastVar(x-stencil.dim) for x in range(0, dim)])
        input_str[dim] = (', var_%s_min_%d, %s, %s' % (stencil.output_name, dim, p.LastVar(dim-stencil.dim), stride))
    p.PrintLine('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.NewVar(), stencil.input_name, type_width[stencil.input_type]/8, input_str[0], input_str[1], input_str[2], input_str[3]))
    p.PrintLine('(void)%s;' % p.LastVar())
    p.UnScope('if (var_%s_host_and_dev_are_null)' % stencil.input_name)

    p.PrintLine('bool %s = %s;' % (p.NewVar(), ' || '.join(['var_%s_host_and_dev_are_null' % x for x in [stencil.output_name, stencil.input_name]])))
    p.PrintLine('bool %s = !(%s);' % (p.NewVar(), p.LastVar(-2)))
    p.PrintLine('if (%s)' % p.LastVar())
    p.DoScope('if (%s)' % p.LastVar())

    PrintCheckElemSize(p, stencil.output_name, stencil.output_type)
    PrintCheckElemSize(p, stencil.input_name, stencil.input_type)
    p.PrintLine()

    p.PrintLine('// allocate buffer for tiled input/output')

    p.UnScope()
    p.PrintLine()

    p.UnScope()

def PrintUnloadBuffer(p, buffer_name, buffer_type):
    p.PrintLine('%s *var_%s = (%s *)(var_%s_buffer->host);' % (buffer_type, buffer_name, buffer_type, buffer_name))
    p.PrintLine('(void)var_%s;' % buffer_name)
    p.PrintLine('const bool var_%s_host_and_dev_are_null = (var_%s_buffer->host == nullptr) && (var_%s_buffer->dev == 0);' % ((buffer_name,)*3))
    p.PrintLine('(void)var_%s_host_and_dev_are_null;' % buffer_name)
    for item in ['min', 'extent', 'stride']:
        for i in range(0, 4):
            if item == 'extent':
                p.PrintLine('int32_t %s_size_dim_%d = var_%s_buffer->%s[%d];' % (buffer_name, i, buffer_name, item, i))
                p.PrintLine('(void)%s_size_dim_%d;' % (buffer_name, i))
            else:
                p.PrintLine('int32_t var_%s_%s_%d = var_%s_buffer->%s[%d];' % (buffer_name, item, i, buffer_name, item, i))
                p.PrintLine('(void)var_%s_%s_%d;' % (buffer_name, item, i))
    p.PrintLine('int32_t var_%s_elem_size = var_%s_buffer->elem_size;' % ((buffer_name,)*2))
    p.PrintLine('(void)var_%s_elem_size;' % buffer_name)

def PrintCheckElemSize(p, buffer_name, buffer_type):
    p.PrintLine('bool %s = var_%s_elem_size == %d;' % (p.NewVar(), buffer_name, type_width[buffer_type]/8))
    p.PrintLine('if (!%s)' % p.LastVar())
    p.DoScope()
    p.PrintLine('int32_t %s = halide_error_bad_elem_size(nullptr, "Buffer %s", "%s", var_%s_elem_size, %d);' % (p.NewVar(), buffer_name, buffer_type, buffer_name, type_width[buffer_type]/8))
    p.PrintLine('return %s;' % p.LastVar())
    p.UnScope()

def PrintCode(stencil, host_file):
    p = Printer(host_file)
    PrintHeader(p)
    p.PrintLine('#include"%s.h"' % stencil.app_name)
    p.PrintLine()

    PrintDefine(p, 'BURST_WIDTH', stencil.burst_width)
    PrintDefine(p, 'PIXEL_WIDTH_I', type_width[stencil.input_type])
    PrintDefine(p, 'PIXEL_WIDTH_O', type_width[stencil.output_type])
    for i, dim in enumerate(GetStencilDim(stencil.A)):
        PrintDefine(p, 'STENCIL_DIM_%d' % i, dim)
    PrintDefine(p, 'STENCIL_DISTANCE', GetStencilDistance(stencil.A, stencil.tile_size))
    PrintDefine(p, 'CHANNEL_NUM_I', stencil.input_chan)
    PrintDefine(p, 'CHANNEL_NUM_O', stencil.output_chan)
    p.PrintLine()

    PrintLoadXCLBIN2(p)
    PrintHalideRewriteBuffer(p)
    PrintHalideErrorCodes(p)
    PrintHalideErrorReport(p)
    PrintWrapped(p, stencil)

def main():
    stencil = GetStencilFromJSON(sys.stdin)
    PrintCode(stencil, sys.stdout)

if __name__ == '__main__':
    main()
