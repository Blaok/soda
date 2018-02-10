#!/usr/bin/python3.6
from collections import deque
from fractions import Fraction
from functools import reduce
import json
import logging
import math
import operator
import os
import sys

from supo.generator.utils import *

logger = logging.getLogger('__main__').getChild(__name__)

def PrintHeader(p):
    for header in ['assert', 'float', 'math', 'stdbool', 'stdint', 'stdio', 'stdlib', 'string', 'fcntl', 'time', 'unistd', 'sys/types', 'sys/stat', 'CL/opencl']:
        p.PrintLine('#include<%s.h>' % header)
    p.PrintLine()

def PrintLoadXCLBIN2(p):
    p.PrintLine('FILE* const* error_report = &stderr;')
    p.PrintLine()
    p.PrintLine('int load_xclbin2_to_memory(const char *filename, char **kernel_binary, char** device)')
    p.DoScope()
    p.PrintLine('uint64_t size = 0;')
    p.PrintLine('FILE *f = fopen(filename, "rb");')
    p.PrintLine('if(nullptr == f)')
    p.DoScope()
    p.PrintLine('*kernel_binary = nullptr;')
    p.PrintLine('fprintf(*error_report, "ERROR: cannot open %s\\n", filename);')
    p.PrintLine('return -1;')
    p.UnScope()
    p.PrintLine('char magic[8];')
    p.PrintLine('unsigned char cipher[32];')
    p.PrintLine('unsigned char key_block[256];')
    p.PrintLine('uint64_t unique_id;')
    p.PrintLine('fread(magic, sizeof(magic), 1, f);')
    p.PrintLine('if(strcmp(magic, "xclbin2")!=0)')
    p.DoScope()
    p.PrintLine('*kernel_binary = nullptr;')
    p.PrintLine('fprintf(*error_report, "ERROR: %s is not a valid xclbin2 file\\n", filename);')
    p.PrintLine('return -2;')
    p.UnScope()
    p.PrintLine('fread(cipher, sizeof(cipher), 1, f);')
    p.PrintLine('fread(key_block, sizeof(key_block), 1, f);')
    p.PrintLine('fread(&unique_id, sizeof(unique_id), 1, f);')
    p.PrintLine('fread(&size, sizeof(size), 1, f);')
    p.PrintLine('char* p = new char[size+1]();')
    p.PrintLine('*kernel_binary = p;')
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
    p.PrintLine('fprintf(*error_report, "ERROR: %s is corrupted\\n", filename);')
    p.PrintLine('return -3;')
    p.UnScope()
    p.PrintLine('*device = p + 5*8;')
    p.PrintLine('fclose(f);')
    p.PrintLine('return size;')
    p.UnScope()
    p.PrintLine()

def PrintHalideRewriteBuffer(p):
    p.PrintLine('static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,')
    for i in range(4):
        p.PrintLine('                                  int32_t min%d, int32_t extent%d, int32_t stride%d%c' % (i, i, i, ",,,)"[i]))
    p.DoScope()
    for item in ['min', 'extent', 'stride']:
        for i in range(4):
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
    p.PrintLine('    if(min_touched < min_valid) {')
    p.PrintLine('        fprintf(*error_report, "%s is accessed at %d, which is before the min (%d) in dimension %d", func_name, min_touched, min_valid, dimension);')
    p.PrintLine('    } else if(max_touched > max_valid) {')
    p.PrintLine('        fprintf(*error_report, "%s is acccessed at %d, which is beyond the max (%d) in dimension %d", func_name, max_touched, max_valid, dimension);')
    p.PrintLine('    }')
    p.PrintLine('    return halide_error_code_access_out_of_bounds;')
    p.PrintLine('}')
    p.PrintLine()

def PrintWrapped(p, stencil):
    buffers = [[stencil.input.name, stencil.input.type], [stencil.output.name, stencil.output.type]]+[[p.name, p.type] for p in stencil.extra_params.values()]
    p.PrintLine('static int %s_wrapped(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % (stencil.app_name, ''.join([('buffer_t *var_%s_buffer, ') % x[0] for x in buffers])))
    p.DoScope()
    for b in buffers:
        PrintUnloadBuffer(p, b[0], b[1])

    p.PrintLine('if(var_%s_host_and_dev_are_null)' % stencil.output.name)
    p.DoScope()
    output_str = [", 0, 0, 0"]*4
    for dim in range(stencil.dim):
        stride  = '1'
        if dim > 0:
            stride = '*'.join([('%s_size_dim_%d' % (stencil.output.name, x)) for x in range(dim)])
        output_str[dim] = (', var_%s_min_%d, %s_size_dim_%d, %s' % (stencil.output.name, dim, stencil.output.name, dim, stride))
    p.PrintLine('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.NewVar(), stencil.output.name, type_width[stencil.output.type]/8, output_str[0], output_str[1], output_str[2], output_str[3]))
    p.PrintLine('(void)%s;' % p.LastVar())
    p.UnScope('if(var_%s_host_and_dev_are_null)' % stencil.output.name)

    p.PrintLine('if(var_%s_host_and_dev_are_null)' % stencil.input.name)
    p.DoScope()
    input_size = ['0']*4
    for dim in range(stencil.dim):
        p.PrintLine('int32_t %s = %s_size_dim_%d + %d;' % (p.NewVar(), stencil.output.name, dim, GetStencilDim(GetOverallStencilWindow(stencil.input, stencil.output))[dim]-1))

    input_str = [', 0, 0, 0']*4
    for dim in range(stencil.dim):
        stride  = '1'
        if dim > 0:
            stride = '*'.join([p.LastVar(x-stencil.dim) for x in range(dim)])
        input_str[dim] = (', var_%s_min_%d, %s, %s' % (stencil.output.name, dim, p.LastVar(dim-stencil.dim), stride))
    p.PrintLine('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.NewVar(), stencil.input.name, type_width[stencil.input.type]/8, input_str[0], input_str[1], input_str[2], input_str[3]))
    p.PrintLine('(void)%s;' % p.LastVar())
    p.UnScope('if(var_%s_host_and_dev_are_null)' % stencil.input.name)

    p.PrintLine('bool %s = %s;' % (p.NewVar(), ' || '.join(['var_%s_host_and_dev_are_null' % x for x in [stencil.output.name, stencil.input.name]])))
    p.PrintLine('bool %s = !(%s);' % (p.NewVar(), p.LastVar(-2)))
    p.PrintLine('if(%s)' % p.LastVar())
    p.DoScope('if(%s)' % p.LastVar())

    PrintCheckElemSize(p, stencil.output.name, stencil.output.type)
    PrintCheckElemSize(p, stencil.input.name, stencil.input.type)
    for param in stencil.extra_params.values():
        PrintCheckElemSize(p, param.name, param.type)
    p.PrintLine()

    p.PrintLine('// allocate buffer for tiled input/output')
    for i in range(stencil.dim-1):
        p.PrintLine('int32_t tile_num_dim_%d = (%s_size_dim_%d-STENCIL_DIM_%d+1+TILE_SIZE_DIM_%d-STENCIL_DIM_%d)/(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1);' % (i, stencil.input.name, i, i, i, i, i, i))
    p.PrintLine()

    p.PrintLine('// change #bank if there is a env var defined')
    p.PrintLine('int dram_bank = %d;' % max_dram_bank)
    p.PrintLine('bool dram_separate = false;')
    p.PrintLine('if(nullptr!=getenv("DRAM_BANK"))')
    p.DoScope()
    p.PrintLine('dram_bank = atoi(getenv("DRAM_BANK"));')
    p.UnScope()
    p.PrintLine('if(nullptr!=getenv("DRAM_SEPARATE"))')
    p.DoScope()
    p.PrintLine('dram_separate = true;')
    p.PrintLine('dram_bank /= 2;')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('// align each linearized tile to multiples of BURST_WIDTH')
    p.PrintLine('int64_t tile_pixel_num = %s*%s_size_dim_%d;' % ('*'.join(['TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)]), stencil.input.name, stencil.dim-1))
    p.PrintLine('int64_t tile_burst_num = (tile_pixel_num-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1;')
    p.PrintLine('int64_t tile_size_linearized_i = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);')
    p.PrintLine('int64_t tile_size_linearized_o = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);')
    p.PrintLine()

    p.PrintLine('// prepare for opencl')
    p.PrintLine('int err;')
    p.PrintLine()
    p.PrintLine('cl_platform_id platforms[16];')
    p.PrintLine('cl_platform_id platform_id;')
    p.PrintLine('cl_uint platform_count;')
    p.PrintLine('cl_device_id device_id;')
    p.PrintLine('cl_context context;')
    p.PrintLine('cl_command_queue commands;')
    p.PrintLine('cl_program program;')
    p.PrintLine('cl_kernel kernel;')
    p.PrintLine()
    p.PrintLine('char cl_platform_vendor[1001];')
    p.PrintLine()

    for c in range(stencil.input.chan):
        for i in range(max_dram_bank):
            p.PrintLine('cl_mem var_%s_%d_bank_%d_cl;' % (stencil.input.name, c, i))
    for c in range(stencil.output.chan):
        for i in range(max_dram_bank):
            p.PrintLine('cl_mem var_%s_%d_bank_%d_cl;' % (stencil.output.name, c, i))
    p.PrintLine()

    for param in stencil.extra_params.values():
        p.PrintLine('cl_mem var_%s_cl;' % param.name)
    p.PrintLine()

    p.PrintLine('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_i/dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_I));' % (stencil.input.name, stencil.input.type, '*'.join(['tile_num_dim_%d'%x for x in range(stencil.dim-1)])))
    p.PrintLine('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_o/dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_O));' % (stencil.output.name, stencil.output.type, '*'.join(['tile_num_dim_%d'%x for x in range(stencil.dim-1)])))
    p.PrintLine()

    p.PrintLine('unsigned char *kernel_binary;')
    p.PrintLine('const char *device_name;')
    p.PrintLine('char target_device_name[64];')
    p.PrintLine('fprintf(*error_report, "INFO: Loading %s\\n", xclbin);')
    p.PrintLine('int kernel_binary_size = load_xclbin2_to_memory(xclbin, (char **) &kernel_binary, (char**)&device_name);')
    p.PrintLine('if(kernel_binary_size < 0)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to load kernel from xclbin: %s\\n", xclbin);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine('for(int i = 0; i<64; ++i)')
    p.DoScope()
    p.PrintLine("target_device_name[i] = (device_name[i]==':'||device_name[i]=='.') ? '_' : device_name[i];")
    p.UnScope()
    p.PrintLine()

    p.PrintLine('err = clGetPlatformIDs(16, platforms, &platform_count);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to find an OpenCL platform\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine('fprintf(*error_report, "INFO: Found %d platforms\\n", platform_count);')
    p.PrintLine()

    p.PrintLine('int platform_found = 0;')
    p.PrintLine('for (unsigned iplat = 0; iplat<platform_count; iplat++)')
    p.DoScope()
    p.PrintLine('err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,nullptr);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine('if(strcmp(cl_platform_vendor, "Xilinx") == 0)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "INFO: Selected platform %d from %s\\n", iplat, cl_platform_vendor);')
    p.PrintLine('platform_id = platforms[iplat];')
    p.PrintLine('platform_found = 1;')
    p.UnScope()
    p.UnScope()
    p.PrintLine('if(!platform_found)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Platform Xilinx not found\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('cl_device_id devices[16];')
    p.PrintLine('cl_uint device_count;')
    p.PrintLine('unsigned int device_found = 0;')
    p.PrintLine('char cl_device_name[1001];')
    p.PrintLine('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
    p.PrintLine('                     16, devices, &device_count);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('for (unsigned int i=0; i<device_count; ++i)')
    p.DoScope()
    p.PrintLine('err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to get device name for device %d\\n", i);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine('printf("INFO: Find device %s\\n", cl_device_name);')
    p.PrintLine('if(strcmp(cl_device_name, target_device_name) == 0 || strcmp(cl_device_name, device_name) == 0)')
    p.DoScope()
    p.PrintLine('device_id = devices[i];')
    p.PrintLine('device_found = 1;')
    p.PrintLine('fprintf(*error_report, "INFO: Selected %s as the target device\\n", device_name);')
    p.UnScope()
    p.UnScope()
    p.PrintLine()

    p.PrintLine('if(!device_found)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Target device %s not found\\n", target_device_name);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
    p.PrintLine('                     1, &device_id, nullptr);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);')
    p.PrintLine('if(!context)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create a compute context\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('commands = clCreateCommandQueue(context, device_id, 0, &err);')
    p.PrintLine('if(!commands)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create a command commands %i\\n",err);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('int status;')
    p.PrintLine('size_t kernel_binary_sizes[1] = {kernel_binary_size};')
    p.PrintLine('program = clCreateProgramWithBinary(context, 1, &device_id, kernel_binary_sizes,')
    p.PrintLine('                                   (const unsigned char **) &kernel_binary, &status, &err);')
    p.PrintLine('if((!program) || (err!=CL_SUCCESS))')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create compute program from binary %d\\n", err);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine('delete[] kernel_binary;')
    p.PrintLine()

    p.PrintLine('err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);')
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('size_t len;')
    p.PrintLine('char buffer[2048];')
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to build program executable\\n");')
    p.PrintLine('clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);')
    p.PrintLine('fprintf(*error_report, "%s\\n", buffer);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('kernel = clCreateKernel(program, "%s_kernel", &err);' % stencil.app_name)
    p.PrintLine('if(!kernel || err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to create compute kernel %d\\n", err);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    for i in range(max_dram_bank):
        p.PrintLine('cl_mem_ext_ptr_t %s_ext_bank_%d, %s_ext_bank_%d;' % (stencil.input.name, i, stencil.output.name, i))
    p.PrintLine('if(dram_separate)')
    p.DoScope()
    p.PrintLine('switch(dram_bank)')
    p.DoScope()
    for i in range(1, max_dram_bank//2+1):
        p.PrintLine('case %d:' % i)
        p.DoIndent()
        for c in range(i):
            p.PrintLine('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.output.name, c%i, c))
        for c in range(i):
            p.PrintLine('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.input.name, c%i, c+i))
        p.PrintLine('break;')
        p.UnIndent()
    p.UnScope()
    p.UnScope()
    p.PrintLine('else')
    p.DoScope()
    for i in range(max_dram_bank):
        p.PrintLine('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.output.name, i, i))
    for i in range(max_dram_bank):
        p.PrintLine('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.input.name, i, i))
    p.UnScope()

    for i in range(max_dram_bank):
        p.PrintLine('%s_ext_bank_%d.obj = 0;' % (stencil.input.name, i))
        p.PrintLine('%s_ext_bank_%d.param = 0;' % (stencil.input.name, i))
    for i in range(max_dram_bank):
        p.PrintLine('%s_ext_bank_%d.obj = 0;' % (stencil.output.name, i))
        p.PrintLine('%s_ext_bank_%d.param = 0;' % (stencil.output.name, i))
    p.PrintLine()

    for c in range(stencil.input.chan):
        for i in range(max_dram_bank):
            p.PrintLine('var_%s_%d_bank_%d_cl = dram_bank > %d ? clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, var_%s_buf_size, &%s_ext_bank_%d, nullptr) : nullptr;' % (stencil.input.name, c, i, i, stencil.input.name, stencil.input.name, i))
    for c in range(stencil.output.chan):
        for i in range(max_dram_bank):
            p.PrintLine('var_%s_%d_bank_%d_cl = dram_bank > %d ? clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, var_%s_buf_size, &%s_ext_bank_%d, nullptr) : nullptr;' % (stencil.output.name, c, i, i, stencil.output.name, stencil.output.name, i))

    for param in stencil.extra_params.values():
        p.PrintLine('var_%s_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, %s*sizeof(%s), nullptr, nullptr);' % (param.name, '*'.join([str(x) for x in param.size]), param.type))

    p.PrintLine('if(%s)' % ' || '.join(['(dram_bank>%d && (%s || %s))' % (x, ' || '.join(['!var_%s_%d_bank_%d_cl' % (stencil.input.name, c, x) for c in range(stencil.input.chan)]), ' || '.join(['!var_%s_%d_bank_%d_cl' % (stencil.output.name, c, x) for c in range(stencil.output.chan)])) for x in range(max_dram_bank)]+['!var_%s_cl' % param.name for param in stencil.extra_params.values()]))
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to allocate device memory\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('cl_event write_events[%d];' % (max_dram_bank*stencil.input.chan+len(stencil.extra_params)))
    for idx, param in enumerate(stencil.extra_params.values()):
        p.PrintLine('%s* var_%s_buf = (%s*)clEnqueueMapBuffer(commands, var_%s_cl, CL_FALSE, CL_MAP_WRITE, 0, %s*sizeof(%s), 0, nullptr, write_events+%d, &err);' % (param.type, param.name, param.type, param.name, '*'.join([str(x) for x in param.size]), param.type, idx))
    for i in range(max_dram_bank):
        for c in range(stencil.input.chan):
            p.PrintLine('%s* var_%s_%d_buf_bank_%d = dram_bank>%d ? (%s*)clEnqueueMapBuffer(commands, var_%s_%d_bank_%d_cl, CL_FALSE, CL_MAP_WRITE, 0, var_%s_buf_size, 0, nullptr, write_events+%d, &err) : nullptr;' % (stencil.input.type, stencil.input.name, c, i, i, stencil.input.type, stencil.input.name, c, i, stencil.input.name, i*stencil.input.chan+c+len(stencil.extra_params)))
    p.PrintLine('clWaitForEvents(dram_bank*%d+%d, write_events);' % (stencil.input.chan, len(stencil.extra_params)))
    p.PrintLine()

    p.PrintLine('// tiling')
    for dim in range(stencil.dim-2, -1, -1):
        p.PrintLine('for(int32_t tile_index_dim_%d = 0; tile_index_dim_%d < tile_num_dim_%d; ++tile_index_dim_%d)' % ((dim,)*4))
        p.DoScope()
        p.PrintLine('int32_t actual_tile_size_dim_%d = (tile_index_dim_%d==tile_num_dim_%d-1) ? %s_size_dim_%d-(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)*tile_index_dim_%d : TILE_SIZE_DIM_%d;' % ((dim,)*3+(stencil.input.name,)+(dim,)*5))

    p.PrintLine('for(int32_t %c = 0; %c < %s_size_dim_%d; ++%c)' % (coords_in_tile[stencil.dim-1], coords_in_tile[stencil.dim-1], stencil.input.name, stencil.dim-1, coords_in_tile[stencil.dim-1]))
    p.DoScope()
    for dim in range(stencil.dim-2, -1, -1):
        p.PrintLine('for(int32_t %c = 0; %c < actual_tile_size_dim_%d; ++%c)' % (coords_in_tile[dim], coords_in_tile[dim], dim, coords_in_tile[dim]))
        p.DoScope()

    p.PrintLine('// (%s) is coordinates in tiled image' % ', '.join(coords_tiled))
    p.PrintLine('// (%s) is coordinates in original image' % ', '.join(coords_in_orig))
    p.PrintLine('// (%s) is coordinates in a tile' % ', '.join(coords_in_tile))
    offset_in_tile = '+'.join(['%c%s' % (coords_in_tile[x], ''.join(['*TILE_SIZE_DIM_%d'%xx for xx in range(x)])) for x in range(stencil.dim)])
    p.PrintLine('int32_t burst_index = (%s)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);' % offset_in_tile)
    p.PrintLine('int32_t burst_residue = (%s)%%(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);' % offset_in_tile)
    for dim in range(stencil.dim-1):
        p.PrintLine('int32_t %c = tile_index_dim_%d*(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)+%c;' % (coords_in_orig[dim], dim, dim, dim, coords_in_tile[dim]))
    p.PrintLine('int32_t %c = %c;' % (coords_in_orig[stencil.dim-1], coords_in_tile[stencil.dim-1]))
    p.PrintLine('int64_t tiled_offset = (%s)*tile_size_linearized_i+burst_index*(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+burst_residue;' % '+'.join(['%stile_index_dim_%d' % (''.join(['tile_num_dim_%d*'%xx for xx in range(x)]), x) for x in range(stencil.dim-1)]))
    p.PrintLine('int64_t original_offset = %s;' % '+'.join(['%c*var_%s_stride_%d' % (coords_in_orig[x], stencil.input.name, x) for x in range(stencil.dim)]))
    p.PrintLine('switch(tiled_offset%dram_bank)')
    p.DoScope()
    for i in range(max_dram_bank):
        p.PrintLine('case %d:' % i)
        p.DoIndent()
        for c in range(stencil.input.chan):
            p.PrintLine('var_%s_%d_buf_bank_%d[tiled_offset/dram_bank] = var_%s[original_offset+%d*var_%s_stride_%d];' % (stencil.input.name, c, i, stencil.input.name, c, stencil.input.name, stencil.dim))
        p.PrintLine('break;')
        p.UnIndent()
    for dim in range(stencil.dim*2):
        p.UnScope()
    p.PrintLine()

    for param in stencil.extra_params.values():
        p.PrintLine('memcpy(var_%s_buf, var_%s, %s*sizeof(%s));' % (param.name, param.name, '*'.join([str(x) for x in param.size]), param.type))
    p.PrintLine()

    p.PrintLine('err = 0;')
    for idx, param in enumerate(stencil.extra_params.values()):
        p.PrintLine('err |= clEnqueueUnmapMemObject(commands, var_%s_cl, var_%s_buf, 0, nullptr, write_events+%d);' % (param.name, param.name, idx))
    for i in range(max_dram_bank):
        for c in range(stencil.input.chan):
            p.PrintLine('err |= dram_bank>%d ? clEnqueueUnmapMemObject(commands, var_%s_%d_bank_%d_cl, var_%s_%d_buf_bank_%d, 0, nullptr, write_events+%d) : err;' % ((i,)+(stencil.input.name, c, i)*2+(i*stencil.input.chan+c+len(stencil.extra_params),)))
    p.PrintLine()

    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to write to input !\\n");')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('err = 0;')
    p.PrintLine('fprintf(*error_report, "INFO: Using %d DRAM bank%s%s\\n", dram_bank, dram_bank>1 ? "s" : "", dram_separate ? ", separated" : "");')
    for i in range(stencil.dim-1):
        p.PrintLine('fprintf(*error_report, "INFO: tile_num_dim_%d = %%d, TILE_SIZE_DIM_%d = %%d\\n", tile_num_dim_%d, TILE_SIZE_DIM_%d);' % ((i,)*4))
    p.PrintLine('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_extent_%d = %%d' % (stencil.input.name, x) for x in range(stencil.dim)]), ', '.join(['%s_size_dim_%d' % (stencil.input.name, x) for x in range(stencil.dim)])))
    p.PrintLine('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_min_%d = %%d' % (stencil.input.name, x) for x in range(stencil.dim)]), ', '.join(['var_%s_min_%d' % (stencil.input.name, x) for x in range(stencil.dim)])))
    p.PrintLine('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_extent_%d = %%d' % (stencil.output.name, x) for x in range(stencil.dim)]), ', '.join(['%s_size_dim_%d' % (stencil.output.name, x) for x in range(stencil.dim)])))
    p.PrintLine('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_min_%d = %%d' % (stencil.output.name, x) for x in range(stencil.dim)]), ', '.join(['var_%s_min_%d' % (stencil.output.name, x) for x in range(stencil.dim)])))
    p.PrintLine()

    p.PrintLine('int kernel_arg = 0;')
    p.PrintLine('int64_t tile_data_num = ((int64_t(%s_size_dim_%d)%s-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1)*BURST_WIDTH/PIXEL_WIDTH_I*dram_bank/UNROLL_FACTOR;' % (stencil.input.name, stencil.dim-1, ''.join(['*TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)])))
    p.PrintLine('int64_t coalesced_data_num = ((int64_t(%s_size_dim_%d)%s+STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1)*tile_num_dim_0;' % (stencil.input.name, stencil.dim-1, ''.join(['*TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)])))
    p.PrintLine('fprintf(*error_report, "INFO: tile_data_num = %ld, coalesced_data_num = %ld\\n", tile_data_num, coalesced_data_num);')
    p.PrintLine()

    for c in range(stencil.output.chan):
        for i in range(max_dram_bank):
            p.PrintLine('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_%d_bank_%d_cl);' % (i, stencil.output.name, c, i))
    for c in range(stencil.input.chan):
        for i in range(max_dram_bank):
            p.PrintLine('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_%d_bank_%d_cl);' % (i, stencil.input.name, c, i))
    for param in stencil.extra_params.values():
        p.PrintLine('err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_cl);' % param.name)
    for variable in ['coalesced_data_num', 'tile_data_num']+['tile_num_dim_%d'%x for x in range(stencil.dim-1)]+['%s_size_dim_%d' % (stencil.input.name, d) for d in range(stencil.dim)]:
        p.PrintLine('err |= clSetKernelArg(kernel, kernel_arg++, sizeof(%s), &%s);' % ((variable,)*2))
    p.PrintLine('if(err != CL_SUCCESS)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "ERROR: Failed to set kernel arguments %d\\n", err);')
    p.PrintLine('exit(EXIT_FAILURE);')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('timespec execute_begin, execute_end;')
    p.PrintLine('cl_event execute_event;')
    p.PrintLine('err = clEnqueueTask(commands, kernel, dram_bank, write_events, &execute_event);')
    p.PrintLine('if(nullptr==getenv("XCL_EMULATION_MODE"))')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "INFO: FPGA warm up\\n");')
    p.PrintLine('clWaitForEvents(1, &execute_event);')
    p.PrintLine()
    p.PrintLine('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_begin);')
    p.PrintLine('err = clEnqueueTask(commands, kernel, 0, nullptr, &execute_event);')
    p.PrintLine('if(err)')
    p.PrintLine('{')
    p.PrintLine('    fprintf(*error_report, "ERROR: Failed to execute kernel %d\\n", err);')
    p.PrintLine('    exit(EXIT_FAILURE);')
    p.PrintLine('}')
    p.PrintLine('clWaitForEvents(1, &execute_event);')
    p.PrintLine('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_end);')
    p.PrintLine()
    p.PrintLine('double elapsed_time = 0.;')
    p.PrintLine('elapsed_time = (double(execute_end.tv_sec-execute_begin.tv_sec)+(execute_end.tv_nsec-execute_begin.tv_nsec)/1e9)*1e6;')
    p.PrintLine('printf("Kernel execution time: %lf us\\n", elapsed_time);')
    p.PrintLine('printf("Kernel throughput:     %%lf pixel/ns\\n", %s/elapsed_time/1e3);' % '*'.join(['%s_size_dim_%d' % (stencil.input.name, x) for x in range(stencil.dim)]))
    p.UnScope()
    p.PrintLine('else')
    p.DoScope()
    p.PrintLine('clWaitForEvents(1, &execute_event);')
    p.PrintLine('fprintf(*error_report, "INFO: Emulation mode\\n");')
    p.UnScope()
    p.PrintLine()

    p.PrintLine('cl_event read_events[%d];' % (max_dram_bank*stencil.output.chan))
    for i in range(max_dram_bank):
        for c in range(stencil.output.chan):
            p.PrintLine('%s* var_%s_%d_buf_bank_%d = dram_bank>%d ? (%s*)clEnqueueMapBuffer(commands, var_%s_%d_bank_%d_cl, CL_FALSE, CL_MAP_READ, 0, var_%s_buf_size, 0, nullptr, read_events+%d, &err) : nullptr;' % (stencil.output.type, stencil.output.name, c, i, i, stencil.output.type, stencil.output.name, c, i, stencil.output.name, i*stencil.output.chan+c))
    p.PrintLine('clWaitForEvents(dram_bank*%d, read_events);' % stencil.output.chan)
    p.PrintLine()

    for dim in range(stencil.dim-2, -1, -1):
        p.PrintLine('for(int32_t tile_index_dim_%d = 0; tile_index_dim_%d < tile_num_dim_%d; ++tile_index_dim_%d)' % ((dim,)*4))
        p.DoScope()
        p.PrintLine('int32_t actual_tile_size_dim_%d = (tile_index_dim_%d==tile_num_dim_%d-1) ? %s_size_dim_%d-(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)*tile_index_dim_%d : TILE_SIZE_DIM_%d;' % ((dim,)*3+(stencil.input.name,)+(dim,)*5))

    if stencil.iterate:
        p.PrintLine('for(int32_t %c = 0; %c < %s_size_dim_%d; ++%c)' % (coords_in_tile[stencil.dim-1], coords_in_tile[stencil.dim-1], stencil.input.name, stencil.dim-1, coords_in_tile[stencil.dim-1]))
        p.DoScope()
        for dim in range(stencil.dim-2, -1, -1):
            p.PrintLine('for(int32_t %c = 0; %c < actual_tile_size_dim_%d; ++%c)' % (coords_in_tile[dim], coords_in_tile[dim], dim, coords_in_tile[dim]))
            p.DoScope()
    else:
        p.PrintLine('for(int32_t %c = 0; %c < %s_size_dim_%d-STENCIL_DIM_%d+1; ++%c)' % (coords_in_tile[stencil.dim-1], coords_in_tile[stencil.dim-1], stencil.input.name, stencil.dim-1, stencil.dim-1, coords_in_tile[stencil.dim-1]))
        p.DoScope()
        for dim in range(stencil.dim-2, -1, -1):
            p.PrintLine('for(int32_t %c = 0; %c < actual_tile_size_dim_%d-STENCIL_DIM_%d+1; ++%c)' % (coords_in_tile[dim], coords_in_tile[dim], dim, dim, coords_in_tile[dim]))
            p.DoScope()

    p.PrintLine('// (%s) is coordinates in tiled image' % ', '.join(coords_tiled))
    p.PrintLine('// (%s) is coordinates in original image' % ', '.join(coords_in_orig))
    p.PrintLine('// (%s) is coordinates in a tile' % ', '.join(coords_in_tile))
    offset_in_tile = '+'.join(['%c%s' % (coords_in_tile[x], ''.join(['*TILE_SIZE_DIM_%d'%xx for xx in range(x)])) for x in range(stencil.dim)])
    p.PrintLine('int32_t burst_index = (%s+STENCIL_OFFSET)/(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);' % offset_in_tile)
    p.PrintLine('int32_t burst_residue = (%s+STENCIL_OFFSET)%%(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);' % offset_in_tile)
    for dim in range(stencil.dim-1):
        p.PrintLine('int32_t %c = tile_index_dim_%d*(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)+%c;' % (coords_in_orig[dim], dim, dim, dim, coords_in_tile[dim]))
    p.PrintLine('int32_t %c = %c;' % (coords_in_orig[stencil.dim-1], coords_in_tile[stencil.dim-1]))
    p.PrintLine('int64_t tiled_offset = (%s)*tile_size_linearized_o+burst_index*(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank)+burst_residue;' % '+'.join(['%stile_index_dim_%d' % (''.join(['tile_num_dim_%d*'%xx for xx in range(x)]), x) for x in range(stencil.dim-1)]))
    p.PrintLine('int64_t original_offset = %s;' % '+'.join(['%c*var_%s_stride_%d' % (coords_in_orig[x], stencil.output.name, x) for x in range(stencil.dim)]))
    p.PrintLine('switch(tiled_offset%dram_bank)')
    p.DoScope()
    for i in range(max_dram_bank):
        p.PrintLine('case %d:' % i)
        p.DoIndent()
        for c in range(stencil.output.chan):
            p.PrintLine('var_%s[original_offset+%d*var_%s_stride_%d] = var_%s_%d_buf_bank_%d[tiled_offset/dram_bank];' % (stencil.output.name, c, stencil.output.name, stencil.dim, stencil.output.name, c, i))
        p.PrintLine('break;')
        p.UnIndent()
    for dim in range(stencil.dim*2):
        p.UnScope()

    for i in range(max_dram_bank):
        p.PrintLine()
        p.PrintLine('if(dram_bank>%d)' % i)
        p.DoScope()
        for c in range(stencil.output.chan):
            p.PrintLine('clEnqueueUnmapMemObject(commands, var_%s_%d_bank_%d_cl, var_%s_%d_buf_bank_%d, 0, nullptr, read_events+%d);' % ((stencil.output.name, c, i)*2+(i*stencil.output.chan+c,)))
        for c in range(stencil.input.chan):
            p.PrintLine('clReleaseMemObject(var_%s_%d_bank_%d_cl);' % (stencil.input.name, c, i))
        for c in range(stencil.output.chan):
            p.PrintLine('clReleaseMemObject(var_%s_%d_bank_%d_cl);' % (stencil.output.name, c, i))
        p.UnScope()
    p.PrintLine('clWaitForEvents(dram_bank*%d, read_events);' % stencil.output.chan)
    p.PrintLine()

    for param in stencil.extra_params.values():
        p.PrintLine('clReleaseMemObject(var_%s_cl);' % param.name)
    p.PrintLine()

    p.PrintLine('clReleaseProgram(program);')
    p.PrintLine('clReleaseKernel(kernel);')
    p.PrintLine('clReleaseCommandQueue(commands);')
    p.PrintLine('clReleaseContext(context);')

    p.UnScope()
    p.PrintLine('return 0;')
    p.UnScope()
    p.PrintLine()

def PrintEntrance(p, stencil):
    buffers = [[stencil.input.name, stencil.input.type], [stencil.output.name, stencil.output.type]]+[[p.name, p.type] for p in stencil.extra_params.values()]
    p.PrintLine('int %s(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % (stencil.app_name, ''.join([('buffer_t *var_%s_buffer, ') % x[0] for x in buffers])))
    p.DoScope()
    for b in buffers:
        PrintUnloadBuffer(p, b[0], b[1])
    p.PrintLine('return %s_wrapped(%sxclbin);' % (stencil.app_name, ''.join([('var_%s_buffer, ') % x[0] for x in buffers])))
    p.UnScope()
    p.PrintLine()

def PrintUnloadBuffer(p, buffer_name, buffer_type):
    p.PrintLine('%s *var_%s = (%s *)(var_%s_buffer->host);' % (buffer_type, buffer_name, buffer_type, buffer_name))
    p.PrintLine('(void)var_%s;' % buffer_name)
    p.PrintLine('const bool var_%s_host_and_dev_are_null = (var_%s_buffer->host == nullptr) && (var_%s_buffer->dev == 0);' % ((buffer_name,)*3))
    p.PrintLine('(void)var_%s_host_and_dev_are_null;' % buffer_name)
    for item in ['min', 'extent', 'stride']:
        for i in range(4):
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
    p.PrintLine('if(!%s)' % p.LastVar())
    p.DoScope()
    p.PrintLine('int32_t %s = halide_error_bad_elem_size(nullptr, "Buffer %s", "%s", var_%s_elem_size, %d);' % (p.NewVar(), buffer_name, buffer_type, buffer_name, type_width[buffer_type]/8))
    p.PrintLine('return %s;' % p.LastVar())
    p.UnScope()

def PrintTest(p, stencil):
    input_dim = stencil.dim
    if stencil.input.chan>1:
        input_dim += 1
    output_dim = stencil.dim
    if stencil.output.chan>1:
        output_dim += 1

    p.PrintLine('int %s_test(const char* xclbin, const int dims[4])' % stencil.app_name)
    p.DoScope()
    p.PrintLine('buffer_t %s;' % ', '.join([b.name for b in stencil.buffers.values()]))
    for param in stencil.extra_params.values():
        p.PrintLine('buffer_t %s;' % param.name)

    p.PrintLine('memset(&%s, 0, sizeof(buffer_t));' % stencil.input.name)
    p.PrintLine('memset(&%s, 0, sizeof(buffer_t));' % stencil.output.name)
    for param in stencil.extra_params.values():
        p.PrintLine('memset(&%s, 0, sizeof(buffer_t));' % param.name)
    p.PrintLine()

    for b in stencil.buffers.values():
        p.PrintLine('%s* %s_img = new %s[%s]();' % (b.type, b.name, b.type, '*'.join(['dims[%d]' % x for x in range(stencil.dim)]+[str(b.chan)])))
    for param in stencil.extra_params.values():
        p.PrintLine('%s %s_img%s;' % (param.type, param.name, reduce(operator.add, ['[%s]'%x for x in param.size])))
    p.PrintLine()

    for b in stencil.buffers.values():
        for d in range(stencil.dim):
            p.PrintLine('%s.extent[%d] = dims[%d];' % (b.name, d, d))
        if b.chan>1:
            p.PrintLine('%s.extent[%d] = %d;' % (b.name, stencil.dim, b.chan))
        p.PrintLine('%s.stride[0] = 1;' % b.name)
        for d in range(1, stencil.dim + (1 if b.chan>1 else 0)):
            p.PrintLine('%s.stride[%d] = %s;' % (b.name, d, '*'.join(['dims[%d]' % x for x in range(d)])))
        p.PrintLine('%s.elem_size = sizeof(%s);' % (b.name, b.type))
        p.PrintLine('%s.host = (uint8_t*)%s_img;' % (b.name, b.name))
        p.PrintLine()

    for param in stencil.extra_params.values():
        for d, size in enumerate(param.size):
            p.PrintLine('%s.extent[%d] = %d;' % (param.name, d, size))
        p.PrintLine('%s.stride[0] = 1;' % param.name)
        for d in range(1, len(param.size)):
            p.PrintLine('%s.stride[%d] = %s;' % (param.name, d, '*'.join([str(x) for x in param.size[:d]])))
        p.PrintLine('%s.elem_size = sizeof(%s);' % (param.name, param.type))
        p.PrintLine('%s.host = (uint8_t*)%s_img;' % (param.name, param.name))
        p.PrintLine()

    p.PrintLine('// initialization can be parallelized with -fopenmp')
    p.PrintLine('#pragma omp parallel for', 0)
    if stencil.input.chan>1:
        p.PrintLine('for(int32_t %c = 0; %c<%d; ++%c)' % (coords_in_orig[stencil.dim], coords_in_orig[stencil.dim], stencil.input.chan, coords_in_orig[stencil.dim]))
        p.DoScope()
    for d in range(0, stencil.dim):
        dim = stencil.dim-d-1
        p.PrintLine('for(int32_t %c = 0; %c<dims[%d]; ++%c)' % (coords_in_orig[dim], coords_in_orig[dim], dim, coords_in_orig[dim]))
        p.DoScope()
    init_val = '+'.join(coords_in_orig[0:input_dim])
    if IsFloat(stencil.input.type):
        init_val = '%s(%s)/%s(%s%s)' % (stencil.input.type, init_val, stencil.input.type, '%d+' % stencil.input.chan if stencil.input.chan>1 else '', '+'.join('dims[%d]' % d for d in range(stencil.dim)))
    p.PrintLine('%s_img[%s] = %s;' % (stencil.input.name, '+'.join('%c*%s.stride[%d]' % (coords_in_orig[d], stencil.input.name, d) for d in range(input_dim)), init_val))
    for d in range(0, input_dim):
        p.UnScope()
    p.PrintLine()

    for param in stencil.extra_params.values():
        p.PrintLine('#pragma omp parallel for', 0)
        for dim, size in enumerate(param.size):
            p.PrintLine('for(int32_t %c = 0; %c<%d; ++%c)' % (coords_in_orig[dim], coords_in_orig[dim], size, coords_in_orig[dim]))
            p.DoScope()
        p.PrintLine('%s_img%s = %s;' % (param.name, reduce(operator.add, ['[%c]' % (coords_in_orig[d]) for d in range(len(param.size))]), '+'.join(coords_in_orig[0:len(param.size)])))
        for d in param.size:
            p.UnScope()
        p.PrintLine()

    p.PrintLine('%s(&%s, &%s, %sxclbin);' % (stencil.app_name, stencil.input.name, stencil.output.name, ''.join(['&%s, ' % param.name for param in stencil.extra_params.values()])))
    p.PrintLine()

    p.PrintLine('int error_count = 0;')
    p.PrintLine()

    LoadPrinter = lambda node: '%s_img%s' % (node.name, ''.join(['[%d]'%x for x in node.idx])) if node.name in stencil.extra_params else '%s_img[%s+%d*%s.stride[%d]]' % (node.name, '+'.join(['(%c%+d)*%s.stride[%d]' % (coords_in_orig[d], node.idx[d] - s.idx[d], stencil.input.name, d) for d in range(stencil.dim)]), node.chan, node.name, stencil.dim)
    StorePrinter = lambda node: '%s result_chan_%d' % (stencil.output.type, node.chan) if node.name == stencil.output.name else '%s_img[%s+%d*%s.stride[%d]]' % (node.name, '+'.join(['%c*%s.stride[%d]' % (coords_in_orig[d], stencil.input.name, d) for d in range(stencil.dim)]), node.chan, node.name, stencil.dim)

    for s in stencil.GetStagesChronologically():
        logger.debug('emit code to produce %s_img' % s.output.name)
        p.PrintLine('// produce %s, can be parallelized with -fopenmp' % s.output.name)
        p.PrintLine('#pragma omp parallel for', 0)
        stencil_window = GetOverallStencilWindow(stencil.input, s.output)
        stencil_dim = GetStencilDim(stencil_window)
        output_idx = GetStencilWindowOffset(stencil_window)
        for d in range(0, stencil.dim):
            dim = stencil.dim-d-1
            p.PrintLine('for(int32_t %c = %d; %c<dims[%d]-%d; ++%c)' % (coords_in_orig[dim], output_idx[dim], coords_in_orig[dim], dim, stencil_dim[dim]-output_idx[dim]-1, coords_in_orig[dim]))
            p.DoScope()
        for e in s.expr:
            e.PrintCode(p, stencil.buffers, LoadPrinter, StorePrinter)
        p.PrintLine()
        if len(s.output.children)==0:
            for c in range(stencil.output.chan):
                run_result = '%s_img[%s+%d*%s.stride[%d]]' % (stencil.output.name, '+'.join(['%c*%s.stride[%d]' % (coords_in_orig[d], stencil.output.name, d) for d in range(stencil.dim)]), c, stencil.output.name, stencil.dim)
                p.PrintLine('%s val_fpga = %s;' % (stencil.output.type, run_result))
                p.PrintLine('%s val_cpu = result_chan_%d;' % (stencil.output.type, c))
                p.PrintLine('double threshold = 0.00001;')
                p.PrintLine('if(nullptr!=getenv("THRESHOLD"))')
                p.DoScope()
                p.PrintLine('threshold = atof(getenv("THRESHOLD"));')
                p.UnScope()
                p.PrintLine('threshold *= threshold;')
                p.PrintLine('if(double(val_fpga-val_cpu)*double(val_fpga-val_cpu)/(double(val_cpu)*double(val_cpu)) > threshold)')
                p.DoScope()
                params = (c, ', '.join(['%d']*stencil.dim), run_result, c, ', '.join(coords_in_orig[:stencil.dim]))
                if stencil.output.type[-2:]=='_t':
                    p.PrintLine('fprintf(*error_report, "%%ld != %%ld @[%d](%s)\\n", int64_t(%s), int64_t(result_chan_%d), %s);' % params)
                else:
                    p.PrintLine('fprintf(*error_report, "%%lf != %%lf @[%d](%s)\\n", double(%s), double(result_chan_%d), %s);' % params)
                p.PrintLine('++error_count;')
                p.UnScope()
        for d in range(0, stencil.dim):
            p.UnScope()
        p.PrintLine()

        if s.PreserveBorderFrom():
            p.PrintLine('// handle borders for iterative stencil')
            p.PrintLine('#pragma omp parallel for', 0)
            bb = s.PreserveBorderFrom()
            stencil_window = GetOverallStencilWindow(bb, s.output)
            stencil_dim = GetStencilDim(stencil_window)
            output_idx = GetStencilWindowOffset(stencil_window)
            for d in range(0, stencil.dim):
                dim = stencil.dim-d-1
                p.PrintLine('for(int32_t %c = 0; %c<dims[%d]; ++%c)' % (coords_in_orig[dim], coords_in_orig[dim], dim, coords_in_orig[dim]))
                p.DoScope()
            p.PrintLine('if(!(%s))' % ' && '.join('%c>=%d && %c<dims[%d]-%d' % (coords_in_orig[d], output_idx[d], coords_in_orig[d], d, stencil_dim[d]-output_idx[d]-1) for d in range(stencil.dim)))
            p.DoScope()
            GroudTruth = lambda c: '%s_img[%s+%d*%s.stride[%d]]' % (bb.name, '+'.join(['%c*%s.stride[%d]' % (coords_in_orig[d], bb.name, d) for d in range(stencil.dim)]), c, bb.name, stencil.dim)
            for e in s.expr:
                p.PrintLine('%s = %s;' % (StorePrinter(e), GroudTruth(e.chan)))
            if len(s.output.children)==0:
                for c in range(stencil.output.chan):
                    run_result = '%s_img[%s+%d*%s.stride[%d]]' % (stencil.output.name, '+'.join(['%c*%s.stride[%d]' % (coords_in_orig[d], stencil.output.name, d) for d in range(stencil.dim)]), c, stencil.output.name, stencil.dim)
                    p.PrintLine('if(%s != result_chan_%d)' % (run_result, c))
                    p.DoScope()
                    params = (c, ', '.join(['%d']*stencil.dim), run_result, c, ', '.join(coords_in_orig[:stencil.dim]))
                    if stencil.output.type[-2:]=='_t':
                        p.PrintLine('fprintf(*error_report, "%%ld != %%ld @[%d](%s)\\n", int64_t(%s), int64_t(result_chan_%d), %s);' % params)
                    else:
                        p.PrintLine('fprintf(*error_report, "%%lf != %%lf @[%d](%s)\\n", double(%s), double(result_chan_%d), %s);' % params)
                    p.PrintLine('++error_count;')
                    p.UnScope()
            p.UnScope()
            for d in range(0, stencil.dim):
                p.UnScope()
            p.PrintLine()

    p.PrintLine('if(error_count==0)')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "INFO: PASS!\\n");')
    p.UnScope()
    p.PrintLine('else')
    p.DoScope()
    p.PrintLine('fprintf(*error_report, "INFO: FAIL!\\n");')
    p.UnScope()
    p.PrintLine()

    for var in [stencil.input.name, stencil.output.name]:
        p.PrintLine('delete[] %s_img;' % var)
    p.PrintLine()

    p.PrintLine('return error_count;')
    p.UnScope()

def PrintCode(stencil, host_file):
    logger.info('generate host source code as %s' % host_file.name)
    p = Printer(host_file)
    PrintHeader(p)
    p.PrintLine('#include"%s.h"' % stencil.app_name)
    p.PrintLine()

    PrintDefine(p, 'BURST_WIDTH', stencil.burst_width)
    PrintDefine(p, 'PIXEL_WIDTH_I', type_width[stencil.input.type])
    PrintDefine(p, 'PIXEL_WIDTH_O', type_width[stencil.output.type])
    overall_stencil_window = GetOverallStencilWindow(stencil.input, stencil.output)
    overall_stencil_distance = GetStencilDistance(overall_stencil_window, stencil.tile_size)
    for i, dim in enumerate(GetStencilDim(overall_stencil_window)):
        PrintDefine(p, 'STENCIL_DIM_%d' % i, dim)
    PrintDefine(p, 'STENCIL_OFFSET', overall_stencil_distance - Serialize(GetStencilWindowOffset(overall_stencil_window), stencil.tile_size))
    PrintDefine(p, 'STENCIL_DISTANCE', overall_stencil_distance)
    p.PrintLine()

    PrintLoadXCLBIN2(p)
    PrintHalideRewriteBuffer(p)
    PrintHalideErrorCodes(p)
    PrintHalideErrorReport(p)
    PrintWrapped(p, stencil)
    PrintEntrance(p, stencil)
    PrintTest(p, stencil)

