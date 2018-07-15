from collections import deque
from functools import reduce
import logging
import operator

from soda import core

logger = logging.getLogger('__main__').getChild(__name__)

def print_header(p):
  for header in ['assert', 'float', 'math', 'stdbool', 'stdint', 'stdio', 'stdlib', 'string', 'fcntl', 'time', 'unistd', 'sys/types', 'sys/stat', 'CL/opencl']:
    p.println('#include<%s.h>' % header)
  p.println()

def print_load_xclbin2(p):
  p.println('FILE* const* error_report = &stderr;')
  p.println()
  p.println('int load_xclbin2_to_memory(const char *filename, char **kernel_binary, char** device)')
  p.do_scope()
  p.println('uint64_t size = 0;')
  p.println('FILE *f = fopen(filename, "rb");')
  p.println('if(nullptr == f)')
  p.do_scope()
  p.println('*kernel_binary = nullptr;')
  p.println('fprintf(*error_report, "ERROR: cannot open %s\\n", filename);')
  p.println('return -1;')
  p.un_scope()
  p.println('char magic[8];')
  p.println('unsigned char cipher[32];')
  p.println('unsigned char key_block[256];')
  p.println('uint64_t unique_id;')
  p.println('fread(magic, sizeof(magic), 1, f);')
  p.println('if(strcmp(magic, "xclbin2")!=0)')
  p.do_scope()
  p.println('*kernel_binary = nullptr;')
  p.println('fprintf(*error_report, "ERROR: %s is not a valid xclbin2 file\\n", filename);')
  p.println('return -2;')
  p.un_scope()
  p.println('fread(cipher, sizeof(cipher), 1, f);')
  p.println('fread(key_block, sizeof(key_block), 1, f);')
  p.println('fread(&unique_id, sizeof(unique_id), 1, f);')
  p.println('fread(&size, sizeof(size), 1, f);')
  p.println('char* p = new char[size+1]();')
  p.println('*kernel_binary = p;')
  p.println('memcpy(p, magic, sizeof(magic));')
  p.println('p += sizeof(magic);')
  p.println('memcpy(p, cipher, sizeof(cipher));')
  p.println('p += sizeof(cipher);')
  p.println('memcpy(p, key_block, sizeof(key_block));')
  p.println('p += sizeof(key_block);')
  p.println('memcpy(p, &unique_id, sizeof(unique_id));')
  p.println('p += sizeof(unique_id);')
  p.println('memcpy(p, &size, sizeof(size));')
  p.println('p += sizeof(size);')
  p.println('uint64_t size_left = size - sizeof(magic) - sizeof(cipher) - sizeof(key_block) - sizeof(unique_id) - sizeof(size);')
  p.println('if(size_left != fread(p, sizeof(char), size_left, f))')
  p.do_scope()
  p.println('delete[] p;')
  p.println('fprintf(*error_report, "ERROR: %s is corrupted\\n", filename);')
  p.println('return -3;')
  p.un_scope()
  p.println('*device = p + 5*8;')
  p.println('fclose(f);')
  p.println('return size;')
  p.un_scope()
  p.println()

def print_halide_rewrite_buffer(p):
  p.println('static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,')
  for i in range(4):
    p.println('                  int32_t min%d, int32_t extent%d, int32_t stride%d%c' % (i, i, i, ",,,)"[i]))
  p.do_scope()
  for item in ['min', 'extent', 'stride']:
    for i in range(4):
      p.println('b->%s[%d] = %s%d;' % (item, i, item, i))
  p.println('return true;')
  p.un_scope()
  p.println()

def print_halide_error_codes(p):
  i = 0
  for item in ['success', 'generic_error', 'explicit_bounds_too_small', 'bad_elem_size', 'access_out_of_bounds', 'buffer_allocation_too_large', 'buffer_extents_too_large', 'constraints_make_required_region_smaller', 'constraint_violated', 'param_too_small', 'param_too_large', 'out_of_memory', 'buffer_argument_is_null', 'debug_to_file_failed', 'copy_to_host_failed', 'copy_to_device_failed', 'device_malloc_failed', 'device_sync_failed', 'device_free_failed', 'no_device_interface', 'matlab_init_failed', 'matlab_bad_param_type', 'internal_error', 'device_run_failed', 'unaligned_host_ptr', 'bad_fold', 'fold_factor_too_small']:
    p.println('int halide_error_code_%s = %d;' % (item, i))
    i -= 1
  p.println()

def print_halide_error_report(p):
  p.println('int halide_error_bad_elem_size(void *user_context, const char *func_name,')
  p.println('                 const char *type_name, int elem_size_given, int correct_elem_size) {')
  p.println('  fprintf(*error_report, "%s has type %s but elem_size of the buffer passed in is %d instead of %d",')
  p.println('      func_name, type_name, elem_size_given, correct_elem_size);')
  p.println('  return halide_error_code_bad_elem_size;')
  p.println('}')
  p.println('int halide_error_constraint_violated(void *user_context, const char *var, int val,')
  p.println('                   const char *constrained_var, int constrained_val) {')
  p.println('  fprintf(*error_report, "Constraint violated: %s (%d) == %s (%d)",')
  p.println('      var, val, constrained_var, constrained_val);')
  p.println('  return halide_error_code_constraint_violated;')
  p.println('}')
  p.println('int halide_error_buffer_allocation_too_large(void *user_context, const char *buffer_name, uint64_t allocation_size, uint64_t max_size) {')
  p.println('  fprintf(*error_report, "Total allocation for buffer %s is %lu, which exceeds the maximum size of %lu",')
  p.println('      buffer_name, allocation_size, max_size);')
  p.println('  return halide_error_code_buffer_allocation_too_large;')
  p.println('}')
  p.println('int halide_error_buffer_extents_too_large(void *user_context, const char *buffer_name, int64_t actual_size, int64_t max_size) {')
  p.println('  fprintf(*error_report, "Product of extents for buffer %s is %ld, which exceeds the maximum size of %ld",')
  p.println('      buffer_name, actual_size, max_size);')
  p.println('  return halide_error_code_buffer_extents_too_large;')
  p.println('}')
  p.println('int halide_error_access_out_of_bounds(void *user_context, const char *func_name, int dimension, int min_touched, int max_touched, int min_valid, int max_valid) {')
  p.println('  if(min_touched < min_valid) {')
  p.println('    fprintf(*error_report, "%s is accessed at %d, which is before the min (%d) in dimension %d", func_name, min_touched, min_valid, dimension);')
  p.println('  } else if(max_touched > max_valid) {')
  p.println('    fprintf(*error_report, "%s is acccessed at %d, which is beyond the max (%d) in dimension %d", func_name, max_touched, max_valid, dimension);')
  p.println('  }')
  p.println('  return halide_error_code_access_out_of_bounds;')
  p.println('}')
  p.println()

def print_wrapped(p, stencil):
  tensors = [[stencil.input.name, stencil.input.type], [stencil.output.name, stencil.output.type]]+[[p.name, p.type] for p in stencil.extra_params.values()]
  p.println('static int %s_wrapped(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % (stencil.app_name, ''.join([('buffer_t *var_%s_buffer, ') % x[0] for x in tensors])))
  p.do_scope()
  for b in tensors:
    print_unload_buffer(p, b[0], b[1])

  p.println('if(var_%s_host_and_dev_are_null)' % stencil.output.name)
  p.do_scope()
  output_str = [", 0, 0, 0"]*4
  for dim in range(stencil.dim):
    stride  = '1'
    if dim > 0:
      stride = '*'.join([('%s_size_dim_%d' % (stencil.output.name, x)) for x in range(dim)])
    output_str[dim] = (', var_%s_min_%d, %s_size_dim_%d, %s' % (stencil.output.name, dim, stencil.output.name, dim, stride))
  p.println('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.new_var(), stencil.output.name, core.TYPE_WIDTH[stencil.output.type]/8, output_str[0], output_str[1], output_str[2], output_str[3]))
  p.println('(void)%s;' % p.last_var())
  p.un_scope('if(var_%s_host_and_dev_are_null)' % stencil.output.name)

  p.println('if(var_%s_host_and_dev_are_null)' % stencil.input.name)
  p.do_scope()
  input_size = ['0']*4
  for dim in range(stencil.dim):
    p.println('int32_t %s = %s_size_dim_%d + %d;' % (p.new_var(), stencil.output.name, dim, core.get_stencil_dim(core.get_overall_stencil_window(stencil.input, stencil.output))[dim]-1))

  input_str = [', 0, 0, 0']*4
  for dim in range(stencil.dim):
    stride  = '1'
    if dim > 0:
      stride = '*'.join([p.last_var(x-stencil.dim) for x in range(dim)])
    input_str[dim] = (', var_%s_min_%d, %s, %s' % (stencil.output.name, dim, p.last_var(dim-stencil.dim), stride))
  p.println('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' % (p.new_var(), stencil.input.name, core.TYPE_WIDTH[stencil.input.type]/8, input_str[0], input_str[1], input_str[2], input_str[3]))
  p.println('(void)%s;' % p.last_var())
  p.un_scope('if(var_%s_host_and_dev_are_null)' % stencil.input.name)

  p.println('bool %s = %s;' % (p.new_var(), ' || '.join(['var_%s_host_and_dev_are_null' % x for x in [stencil.output.name, stencil.input.name]])))
  p.println('bool %s = !(%s);' % (p.new_var(), p.last_var(-2)))
  p.println('if(%s)' % p.last_var())
  p.do_scope('if(%s)' % p.last_var())

  print_check_elem_size(p, stencil.output.name, stencil.output.type)
  print_check_elem_size(p, stencil.input.name, stencil.input.type)
  for param in stencil.extra_params.values():
    print_check_elem_size(p, param.name, param.type)
  p.println()

  p.println('// allocate buffer for tiled input/output')
  for i in range(stencil.dim-1):
    p.println('int32_t tile_num_dim_%d = (%s_size_dim_%d-STENCIL_DIM_%d+1+TILE_SIZE_DIM_%d-STENCIL_DIM_%d)/(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1);' % (i, stencil.input.name, i, i, i, i, i, i))
  p.println()

  p.println('// change #bank if there is a env var defined')
  p.println('int dram_bank = %d;' % core.MAX_DRAM_BANK)
  p.println('bool dram_separate = false;')
  p.println('if(nullptr!=getenv("DRAM_BANK"))')
  p.do_scope()
  p.println('dram_bank = atoi(getenv("DRAM_BANK"));')
  p.un_scope()
  p.println('if(nullptr!=getenv("DRAM_SEPARATE"))')
  p.do_scope()
  p.println('dram_separate = true;')
  p.println('dram_bank /= 2;')
  p.un_scope()
  p.println()

  p.println('// align each linearized tile to multiples of BURST_WIDTH')
  p.println('int64_t tile_pixel_num = %s*%s_size_dim_%d;' % ('*'.join(['TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)]), stencil.input.name, stencil.dim-1))
  p.println('int64_t tile_burst_num = (tile_pixel_num-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1;')
  p.println('int64_t tile_size_linearized_i = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);')
  p.println('int64_t tile_size_linearized_o = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);')
  p.println()

  p.println('// prepare for opencl')
  p.println('int err;')
  p.println()
  p.println('cl_platform_id platforms[16];')
  p.println('cl_platform_id platform_id;')
  p.println('cl_uint platform_count;')
  p.println('cl_device_id device_id;')
  p.println('cl_context context;')
  p.println('cl_command_queue commands;')
  p.println('cl_program program;')
  p.println('cl_kernel kernel;')
  p.println()
  p.println('char cl_platform_vendor[1001];')
  p.println()

  for c in range(stencil.input.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('cl_mem var_%s_%d_bank_%d_cl;' % (stencil.input.name, c, i))
  for c in range(stencil.output.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('cl_mem var_%s_%d_bank_%d_cl;' % (stencil.output.name, c, i))
  p.println()

  for param in stencil.extra_params.values():
    p.println('cl_mem var_%s_cl;' % param.name)
  p.println()

  p.println('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_i/dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_I));' % (stencil.input.name, stencil.input.type, '*'.join(['tile_num_dim_%d'%x for x in range(stencil.dim-1)])))
  p.println('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_o/dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_O));' % (stencil.output.name, stencil.output.type, '*'.join(['tile_num_dim_%d'%x for x in range(stencil.dim-1)])))
  p.println()

  p.println('unsigned char *kernel_binary;')
  p.println('const char *device_name;')
  p.println('char target_device_name[64];')
  p.println('fprintf(*error_report, "INFO: Loading %s\\n", xclbin);')
  p.println('int kernel_binary_size = load_xclbin2_to_memory(xclbin, (char **) &kernel_binary, (char**)&device_name);')
  p.println('if(kernel_binary_size < 0)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to load kernel from xclbin: %s\\n", xclbin);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println('for(int i = 0; i<64; ++i)')
  p.do_scope()
  p.println("target_device_name[i] = (device_name[i]==':'||device_name[i]=='.') ? '_' : device_name[i];")
  p.un_scope()
  p.println()

  p.println('err = clGetPlatformIDs(16, platforms, &platform_count);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to find an OpenCL platform\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println('fprintf(*error_report, "INFO: Found %d platforms\\n", platform_count);')
  p.println()

  p.println('int platform_found = 0;')
  p.println('for (unsigned iplat = 0; iplat<platform_count; iplat++)')
  p.do_scope()
  p.println('err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,nullptr);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println('if(strcmp(cl_platform_vendor, "Xilinx") == 0)')
  p.do_scope()
  p.println('fprintf(*error_report, "INFO: Selected platform %d from %s\\n", iplat, cl_platform_vendor);')
  p.println('platform_id = platforms[iplat];')
  p.println('platform_found = 1;')
  p.un_scope()
  p.un_scope()
  p.println('if(!platform_found)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Platform Xilinx not found\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('cl_device_id devices[16];')
  p.println('cl_uint device_count;')
  p.println('unsigned int device_found = 0;')
  p.println('char cl_device_name[1001];')
  p.println('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
  p.println('           16, devices, &device_count);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('for (unsigned int i=0; i<device_count; ++i)')
  p.do_scope()
  p.println('err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to get device name for device %d\\n", i);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println('printf("INFO: Find device %s\\n", cl_device_name);')
  p.println('if(strcmp(cl_device_name, target_device_name) == 0 || strcmp(cl_device_name, device_name) == 0)')
  p.do_scope()
  p.println('device_id = devices[i];')
  p.println('device_found = 1;')
  p.println('fprintf(*error_report, "INFO: Selected %s as the target device\\n", device_name);')
  p.un_scope()
  p.un_scope()
  p.println()

  p.println('if(!device_found)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Target device %s not found\\n", target_device_name);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
  p.println('           1, &device_id, nullptr);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);')
  p.println('if(!context)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create a compute context\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('commands = clCreateCommandQueue(context, device_id, 0, &err);')
  p.println('if(!commands)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create a command commands %i\\n",err);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('int status;')
  p.println('size_t kernel_binary_sizes[1] = {kernel_binary_size};')
  p.println('program = clCreateProgramWithBinary(context, 1, &device_id, kernel_binary_sizes,')
  p.println('                   (const unsigned char **) &kernel_binary, &status, &err);')
  p.println('if((!program) || (err!=CL_SUCCESS))')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create compute program from binary %d\\n", err);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println('delete[] kernel_binary;')
  p.println()

  p.println('err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);')
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('size_t len;')
  p.println('char buffer[2048];')
  p.println('fprintf(*error_report, "ERROR: Failed to build program executable\\n");')
  p.println('clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);')
  p.println('fprintf(*error_report, "%s\\n", buffer);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('kernel = clCreateKernel(program, "%s_kernel", &err);' % stencil.app_name)
  p.println('if(!kernel || err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to create compute kernel %d\\n", err);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  for i in range(core.MAX_DRAM_BANK):
    p.println('cl_mem_ext_ptr_t %s_ext_bank_%d, %s_ext_bank_%d;' % (stencil.input.name, i, stencil.output.name, i))
  p.println('if(dram_separate)')
  p.do_scope()
  p.println('switch(dram_bank)')
  p.do_scope()
  for i in range(1, core.MAX_DRAM_BANK//2+1):
    p.println('case %d:' % i)
    p.do_indent()
    for c in range(i):
      p.println('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.output.name, c%i, c))
    for c in range(i):
      p.println('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.input.name, c%i, c+i))
    p.println('break;')
    p.un_indent()
  p.un_scope()
  p.un_scope()
  p.println('else')
  p.do_scope()
  for i in range(core.MAX_DRAM_BANK):
    p.println('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.output.name, i, i))
  for i in range(core.MAX_DRAM_BANK):
    p.println('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (stencil.input.name, i, i))
  p.un_scope()

  for i in range(core.MAX_DRAM_BANK):
    p.println('%s_ext_bank_%d.obj = 0;' % (stencil.input.name, i))
    p.println('%s_ext_bank_%d.param = 0;' % (stencil.input.name, i))
  for i in range(core.MAX_DRAM_BANK):
    p.println('%s_ext_bank_%d.obj = 0;' % (stencil.output.name, i))
    p.println('%s_ext_bank_%d.param = 0;' % (stencil.output.name, i))
  p.println()

  for c in range(stencil.input.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('var_%s_%d_bank_%d_cl = dram_bank > %d ? clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, var_%s_buf_size, &%s_ext_bank_%d, nullptr) : nullptr;' % (stencil.input.name, c, i, i, stencil.input.name, stencil.input.name, i))
  for c in range(stencil.output.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('var_%s_%d_bank_%d_cl = dram_bank > %d ? clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, var_%s_buf_size, &%s_ext_bank_%d, nullptr) : nullptr;' % (stencil.output.name, c, i, i, stencil.output.name, stencil.output.name, i))

  for param in stencil.extra_params.values():
    p.println('var_%s_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, %s*sizeof(%s), nullptr, nullptr);' % (param.name, '*'.join([str(x) for x in param.size]), param.type))

  p.println('if(%s)' % ' || '.join(['(dram_bank>%d && (%s || %s))' % (x, ' || '.join(['!var_%s_%d_bank_%d_cl' % (stencil.input.name, c, x) for c in range(stencil.input.chan)]), ' || '.join(['!var_%s_%d_bank_%d_cl' % (stencil.output.name, c, x) for c in range(stencil.output.chan)])) for x in range(core.MAX_DRAM_BANK)]+['!var_%s_cl' % param.name for param in stencil.extra_params.values()]))
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to allocate device memory\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('cl_event write_events[%d];' % (core.MAX_DRAM_BANK*stencil.input.chan+len(stencil.extra_params)))
  for idx, param in enumerate(stencil.extra_params.values()):
    p.println('%s* var_%s_buf = (%s*)clEnqueueMapBuffer(commands, var_%s_cl, CL_FALSE, CL_MAP_WRITE, 0, %s*sizeof(%s), 0, nullptr, write_events+%d, &err);' % (param.type, param.name, param.type, param.name, '*'.join([str(x) for x in param.size]), param.type, idx))
  for i in range(core.MAX_DRAM_BANK):
    for c in range(stencil.input.chan):
      p.println('%s* var_%s_%d_buf_bank_%d = dram_bank>%d ? (%s*)clEnqueueMapBuffer(commands, var_%s_%d_bank_%d_cl, CL_FALSE, CL_MAP_WRITE, 0, var_%s_buf_size, 0, nullptr, write_events+%d, &err) : nullptr;' % (stencil.input.type, stencil.input.name, c, i, i, stencil.input.type, stencil.input.name, c, i, stencil.input.name, i*stencil.input.chan+c+len(stencil.extra_params)))
  p.println('clWaitForEvents(dram_bank*%d+%d, write_events);' % (stencil.input.chan, len(stencil.extra_params)))
  p.println()

  p.println('// tiling')
  for dim in range(stencil.dim-2, -1, -1):
    p.println('for(int32_t tile_index_dim_%d = 0; tile_index_dim_%d < tile_num_dim_%d; ++tile_index_dim_%d)' % ((dim,)*4))
    p.do_scope()
    p.println('int32_t actual_tile_size_dim_%d = (tile_index_dim_%d==tile_num_dim_%d-1) ? %s_size_dim_%d-(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)*tile_index_dim_%d : TILE_SIZE_DIM_%d;' % ((dim,)*3+(stencil.input.name,)+(dim,)*5))

  p.println('for(int32_t %c = 0; %c < %s_size_dim_%d; ++%c)' % (core.COORDS_IN_TILE[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1], stencil.input.name, stencil.dim-1, core.COORDS_IN_TILE[stencil.dim-1]))
  p.do_scope()
  for dim in range(stencil.dim-2, -1, -1):
    p.println('for(int32_t %c = 0; %c < actual_tile_size_dim_%d; ++%c)' % (core.COORDS_IN_TILE[dim], core.COORDS_IN_TILE[dim], dim, core.COORDS_IN_TILE[dim]))
    p.do_scope()

  p.println('// (%s) is coordinates in tiled image' % ', '.join(core.COORDS_TILED))
  p.println('// (%s) is coordinates in original image' % ', '.join(core.COORDS_IN_ORIG))
  p.println('// (%s) is coordinates in a tile' % ', '.join(core.COORDS_IN_TILE))
  offset_in_tile = '+'.join(['%c%s' % (core.COORDS_IN_TILE[x], ''.join(['*TILE_SIZE_DIM_%d'%xx for xx in range(x)])) for x in range(stencil.dim)])
  p.println('int32_t burst_index = (%s)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);' % offset_in_tile)
  p.println('int32_t burst_residue = (%s)%%(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);' % offset_in_tile)
  for dim in range(stencil.dim-1):
    p.println('int32_t %c = tile_index_dim_%d*(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)+%c;' % (core.COORDS_IN_ORIG[dim], dim, dim, dim, core.COORDS_IN_TILE[dim]))
  p.println('int32_t %c = %c;' % (core.COORDS_IN_ORIG[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1]))
  p.println('int64_t tiled_offset = (%s)*tile_size_linearized_i+burst_index*(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+burst_residue;' % '+'.join(['%stile_index_dim_%d' % (''.join(['tile_num_dim_%d*'%xx for xx in range(x)]), x) for x in range(stencil.dim-1)]))
  p.println('int64_t original_offset = %s;' % '+'.join(['%c*var_%s_stride_%d' % (core.COORDS_IN_ORIG[x], stencil.input.name, x) for x in range(stencil.dim)]))
  p.println('switch(tiled_offset%dram_bank)')
  p.do_scope()
  for i in range(core.MAX_DRAM_BANK):
    p.println('case %d:' % i)
    p.do_indent()
    for c in range(stencil.input.chan):
      p.println('var_%s_%d_buf_bank_%d[tiled_offset/dram_bank] = var_%s[original_offset+%d*var_%s_stride_%d];' % (stencil.input.name, c, i, stencil.input.name, c, stencil.input.name, stencil.dim))
    p.println('break;')
    p.un_indent()
  for dim in range(stencil.dim*2):
    p.un_scope()
  p.println()

  for param in stencil.extra_params.values():
    p.println('memcpy(var_%s_buf, var_%s, %s*sizeof(%s));' % (param.name, param.name, '*'.join([str(x) for x in param.size]), param.type))
  p.println()

  p.println('err = 0;')
  for idx, param in enumerate(stencil.extra_params.values()):
    p.println('err |= clEnqueueUnmapMemObject(commands, var_%s_cl, var_%s_buf, 0, nullptr, write_events+%d);' % (param.name, param.name, idx))
  for i in range(core.MAX_DRAM_BANK):
    for c in range(stencil.input.chan):
      p.println('err |= dram_bank>%d ? clEnqueueUnmapMemObject(commands, var_%s_%d_bank_%d_cl, var_%s_%d_buf_bank_%d, 0, nullptr, write_events+%d) : err;' % ((i,)+(stencil.input.name, c, i)*2+(i*stencil.input.chan+c+len(stencil.extra_params),)))
  p.println()

  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to write to input !\\n");')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('err = 0;')
  p.println('fprintf(*error_report, "INFO: Using %d DRAM bank%s%s\\n", dram_bank, dram_bank>1 ? "s" : "", dram_separate ? ", separated" : "");')
  for i in range(stencil.dim-1):
    p.println('fprintf(*error_report, "INFO: tile_num_dim_%d = %%d, TILE_SIZE_DIM_%d = %%d\\n", tile_num_dim_%d, TILE_SIZE_DIM_%d);' % ((i,)*4))
  p.println('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_extent_%d = %%d' % (stencil.input.name, x) for x in range(stencil.dim)]), ', '.join(['%s_size_dim_%d' % (stencil.input.name, x) for x in range(stencil.dim)])))
  p.println('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_min_%d = %%d' % (stencil.input.name, x) for x in range(stencil.dim)]), ', '.join(['var_%s_min_%d' % (stencil.input.name, x) for x in range(stencil.dim)])))
  p.println('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_extent_%d = %%d' % (stencil.output.name, x) for x in range(stencil.dim)]), ', '.join(['%s_size_dim_%d' % (stencil.output.name, x) for x in range(stencil.dim)])))
  p.println('fprintf(*error_report, "INFO: %s\\n", %s);' % (', '.join(['%s_min_%d = %%d' % (stencil.output.name, x) for x in range(stencil.dim)]), ', '.join(['var_%s_min_%d' % (stencil.output.name, x) for x in range(stencil.dim)])))
  p.println()

  p.println('int kernel_arg = 0;')
  p.println('int64_t tile_data_num = ((int64_t(%s_size_dim_%d)%s-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1)*BURST_WIDTH/PIXEL_WIDTH_I*dram_bank/UNROLL_FACTOR;' % (stencil.input.name, stencil.dim-1, ''.join(['*TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)])))
  p.println('int64_t coalesced_data_num = ((int64_t(%s_size_dim_%d)%s*%s+STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1);' % (stencil.input.name, stencil.dim-1, ''.join(['*TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)]), '*'.join('tile_num_dim_%d'%d for d in range(stencil.dim-1))))
  for d in range(stencil.dim-1):
    p.println('uint32_t input_bound_dim_%d = tile_num_dim_%d*(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1);' % (d, d, d, d))
  p.println('fprintf(*error_report, "INFO: tile_data_num = %ld, coalesced_data_num = %ld\\n", tile_data_num, coalesced_data_num);')
  p.println()

  for c in range(stencil.output.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_%d_bank_%d_cl);' % (i, stencil.output.name, c, i))
  for c in range(stencil.input.chan):
    for i in range(core.MAX_DRAM_BANK):
      p.println('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_%d_bank_%d_cl);' % (i, stencil.input.name, c, i))
  for param in stencil.extra_params.values():
    p.println('err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_%s_cl);' % param.name)
  for variable in ['coalesced_data_num', 'tile_data_num']+['input_bound_dim_%d'%x for x in range(stencil.dim-1)]+['%s_size_dim_%d' % (stencil.input.name, d) for d in range(stencil.dim)]:
    p.println('err |= clSetKernelArg(kernel, kernel_arg++, sizeof(%s), &%s);' % ((variable,)*2))
  p.println('if(err != CL_SUCCESS)')
  p.do_scope()
  p.println('fprintf(*error_report, "ERROR: Failed to set kernel arguments %d\\n", err);')
  p.println('exit(EXIT_FAILURE);')
  p.un_scope()
  p.println()

  p.println('timespec execute_begin, execute_end;')
  p.println('cl_event execute_event;')
  p.println('err = clEnqueueTask(commands, kernel, dram_bank, write_events, &execute_event);')
  p.println('if(nullptr==getenv("XCL_EMULATION_MODE"))')
  p.do_scope()
  p.println('fprintf(*error_report, "INFO: FPGA warm up\\n");')
  p.println('clWaitForEvents(1, &execute_event);')
  p.println()
  p.println('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_begin);')
  p.println('err = clEnqueueTask(commands, kernel, 0, nullptr, &execute_event);')
  p.println('if(err)')
  p.println('{')
  p.println('  fprintf(*error_report, "ERROR: Failed to execute kernel %d\\n", err);')
  p.println('  exit(EXIT_FAILURE);')
  p.println('}')
  p.println('clWaitForEvents(1, &execute_event);')
  p.println('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_end);')
  p.println()
  p.println('double elapsed_time = 0.;')
  p.println('elapsed_time = (double(execute_end.tv_sec-execute_begin.tv_sec)+(execute_end.tv_nsec-execute_begin.tv_nsec)/1e9)*1e6;')
  p.println('printf("Kernel execution time: %lf us\\n", elapsed_time);')
  p.println('printf("Kernel throughput:   %%lf pixel/ns\\n", %s/elapsed_time/1e3);' % '*'.join(['%s_size_dim_%d' % (stencil.input.name, x) for x in range(stencil.dim)]))
  p.un_scope()
  p.println('else')
  p.do_scope()
  p.println('clWaitForEvents(1, &execute_event);')
  p.println('fprintf(*error_report, "INFO: Emulation mode\\n");')
  p.un_scope()
  p.println()

  p.println('cl_event read_events[%d];' % (core.MAX_DRAM_BANK*stencil.output.chan))
  for i in range(core.MAX_DRAM_BANK):
    for c in range(stencil.output.chan):
      p.println('%s* var_%s_%d_buf_bank_%d = dram_bank>%d ? (%s*)clEnqueueMapBuffer(commands, var_%s_%d_bank_%d_cl, CL_FALSE, CL_MAP_READ, 0, var_%s_buf_size, 0, nullptr, read_events+%d, &err) : nullptr;' % (stencil.output.type, stencil.output.name, c, i, i, stencil.output.type, stencil.output.name, c, i, stencil.output.name, i*stencil.output.chan+c))
  p.println('clWaitForEvents(dram_bank*%d, read_events);' % stencil.output.chan)
  p.println()

  for dim in range(stencil.dim-2, -1, -1):
    p.println('for(int32_t tile_index_dim_%d = 0; tile_index_dim_%d < tile_num_dim_%d; ++tile_index_dim_%d)' % ((dim,)*4))
    p.do_scope()
    p.println('int32_t actual_tile_size_dim_%d = (tile_index_dim_%d==tile_num_dim_%d-1) ? %s_size_dim_%d-(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)*tile_index_dim_%d : TILE_SIZE_DIM_%d;' % ((dim,)*3+(stencil.input.name,)+(dim,)*5))

  if stencil.preserve_border:
    overall_stencil_window = core.get_overall_stencil_window(stencil.output.parent.preserve_border_from(), stencil.output)
    overall_stencil_offset = core.get_stencil_window_offset(overall_stencil_window)
    overall_stencil_dim = core.get_stencil_dim(overall_stencil_window)
    p.println('for(int32_t %c = 0; %c < %s_size_dim_%d; ++%c)' % (core.COORDS_IN_TILE[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1], stencil.input.name, stencil.dim-1, core.COORDS_IN_TILE[stencil.dim-1]))
    p.do_scope()
    for dim in range(stencil.dim-2, -1, -1):
      p.println('for(int32_t %c = tile_index_dim_%d==0 ? 0 : %d; %c < actual_tile_size_dim_%d-(tile_index_dim_%d==tile_num_dim_%d-1 ? 0 : %d); ++%c)' % (core.COORDS_IN_TILE[dim], dim, overall_stencil_offset[dim], core.COORDS_IN_TILE[dim], dim, dim, dim, overall_stencil_dim[dim]-1-overall_stencil_offset[dim], core.COORDS_IN_TILE[dim]))
      p.do_scope()
  else:
    overall_stencil_window = core.get_overall_stencil_window(stencil.input, stencil.output)
    overall_stencil_offset = core.get_stencil_window_offset(overall_stencil_window)
    overall_stencil_dim = core.get_stencil_dim(overall_stencil_window)
    p.println('for(int32_t %c = %d; %c < %s_size_dim_%d-%d; ++%c)' % (core.COORDS_IN_TILE[stencil.dim-1], overall_stencil_offset[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1], stencil.output.name, stencil.dim-1, overall_stencil_dim[stencil.dim-1]-1-overall_stencil_offset[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1]))
    p.do_scope()
    for dim in range(stencil.dim-2, -1, -1):
      p.println('for(int32_t %c = %d; %c < actual_tile_size_dim_%d-%d; ++%c)' % (core.COORDS_IN_TILE[dim], overall_stencil_offset[dim], core.COORDS_IN_TILE[dim], dim, overall_stencil_dim[dim]-1-overall_stencil_offset[dim], core.COORDS_IN_TILE[dim]))
      p.do_scope()

  p.println('// (%s) is coordinates in tiled image' % ', '.join(core.COORDS_TILED))
  p.println('// (%s) is coordinates in original image' % ', '.join(core.COORDS_IN_ORIG))
  p.println('// (%s) is coordinates in a tile' % ', '.join(core.COORDS_IN_TILE))
  offset_in_tile = '+'.join(['%c%s' % (core.COORDS_IN_TILE[x], ''.join(['*TILE_SIZE_DIM_%d'%xx for xx in range(x)])) for x in range(stencil.dim)])
  p.println('int32_t burst_index = (%s+STENCIL_OFFSET)/(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);' % offset_in_tile)
  p.println('int32_t burst_residue = (%s+STENCIL_OFFSET)%%(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);' % offset_in_tile)
  for dim in range(stencil.dim-1):
    p.println('int32_t %c = tile_index_dim_%d*(TILE_SIZE_DIM_%d-STENCIL_DIM_%d+1)+%c;' % (core.COORDS_IN_ORIG[dim], dim, dim, dim, core.COORDS_IN_TILE[dim]))
  p.println('int32_t %c = %c;' % (core.COORDS_IN_ORIG[stencil.dim-1], core.COORDS_IN_TILE[stencil.dim-1]))
  p.println('int64_t tiled_offset = (%s)*tile_size_linearized_o+burst_index*(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank)+burst_residue;' % '+'.join(['%stile_index_dim_%d' % (''.join(['tile_num_dim_%d*'%xx for xx in range(x)]), x) for x in range(stencil.dim-1)]))
  p.println('int64_t original_offset = %s;' % '+'.join(['%c*var_%s_stride_%d' % (core.COORDS_IN_ORIG[x], stencil.output.name, x) for x in range(stencil.dim)]))
  p.println('switch(tiled_offset%dram_bank)')
  p.do_scope()
  for i in range(core.MAX_DRAM_BANK):
    p.println('case %d:' % i)
    p.do_indent()
    for c in range(stencil.output.chan):
      p.println('var_%s[original_offset+%d*var_%s_stride_%d] = var_%s_%d_buf_bank_%d[tiled_offset/dram_bank];' % (stencil.output.name, c, stencil.output.name, stencil.dim, stencil.output.name, c, i))
    p.println('break;')
    p.un_indent()
  for dim in range(stencil.dim*2):
    p.un_scope()

  for i in range(core.MAX_DRAM_BANK):
    p.println()
    p.println('if(dram_bank>%d)' % i)
    p.do_scope()
    for c in range(stencil.output.chan):
      p.println('clEnqueueUnmapMemObject(commands, var_%s_%d_bank_%d_cl, var_%s_%d_buf_bank_%d, 0, nullptr, read_events+%d);' % ((stencil.output.name, c, i)*2+(i*stencil.output.chan+c,)))
    for c in range(stencil.input.chan):
      p.println('clReleaseMemObject(var_%s_%d_bank_%d_cl);' % (stencil.input.name, c, i))
    for c in range(stencil.output.chan):
      p.println('clReleaseMemObject(var_%s_%d_bank_%d_cl);' % (stencil.output.name, c, i))
    p.un_scope()
  p.println('clWaitForEvents(dram_bank*%d, read_events);' % stencil.output.chan)
  p.println()

  for param in stencil.extra_params.values():
    p.println('clReleaseMemObject(var_%s_cl);' % param.name)
  p.println()

  p.println('clReleaseProgram(program);')
  p.println('clReleaseKernel(kernel);')
  p.println('clReleaseCommandQueue(commands);')
  p.println('clReleaseContext(context);')

  p.un_scope()
  p.println('return 0;')
  p.un_scope()
  p.println()

def print_entrance(p, stencil):
  tensors = [[stencil.input.name, stencil.input.type], [stencil.output.name, stencil.output.type]]+[[p.name, p.type] for p in stencil.extra_params.values()]
  p.println('int %s(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % (stencil.app_name, ''.join([('buffer_t *var_%s_buffer, ') % x[0] for x in tensors])))
  p.do_scope()
  for b in tensors:
    print_unload_buffer(p, b[0], b[1])
  p.println('return %s_wrapped(%sxclbin);' % (stencil.app_name, ''.join([('var_%s_buffer, ') % x[0] for x in tensors])))
  p.un_scope()
  p.println()

def print_unload_buffer(p, buffer_name, buffer_type):
  p.println('%s *var_%s = (%s *)(var_%s_buffer->host);' % (buffer_type, buffer_name, buffer_type, buffer_name))
  p.println('(void)var_%s;' % buffer_name)
  p.println('const bool var_%s_host_and_dev_are_null = (var_%s_buffer->host == nullptr) && (var_%s_buffer->dev == 0);' % ((buffer_name,)*3))
  p.println('(void)var_%s_host_and_dev_are_null;' % buffer_name)
  for item in ['min', 'extent', 'stride']:
    for i in range(4):
      if item == 'extent':
        p.println('int32_t %s_size_dim_%d = var_%s_buffer->%s[%d];' % (buffer_name, i, buffer_name, item, i))
        p.println('(void)%s_size_dim_%d;' % (buffer_name, i))
      else:
        p.println('int32_t var_%s_%s_%d = var_%s_buffer->%s[%d];' % (buffer_name, item, i, buffer_name, item, i))
        p.println('(void)var_%s_%s_%d;' % (buffer_name, item, i))
  p.println('int32_t var_%s_elem_size = var_%s_buffer->elem_size;' % ((buffer_name,)*2))
  p.println('(void)var_%s_elem_size;' % buffer_name)

def print_check_elem_size(p, buffer_name, buffer_type):
  p.println('bool %s = var_%s_elem_size == %d;' % (p.new_var(), buffer_name, core.TYPE_WIDTH[buffer_type]/8))
  p.println('if(!%s)' % p.last_var())
  p.do_scope()
  p.println('int32_t %s = halide_error_bad_elem_size(nullptr, "Buffer %s", "%s", var_%s_elem_size, %d);' % (p.new_var(), buffer_name, buffer_type, buffer_name, core.TYPE_WIDTH[buffer_type]/8))
  p.println('return %s;' % p.last_var())
  p.un_scope()

def print_test(p, stencil):
  input_dim = stencil.dim
  if stencil.input.chan>1:
    input_dim += 1
  output_dim = stencil.dim
  if stencil.output.chan>1:
    output_dim += 1

  p.println('int %s_test(const char* xclbin, const int dims[4])' % stencil.app_name)
  p.do_scope()
  p.println('buffer_t %s;' % ', '.join([b.name for b in stencil.tensors.values()]))
  for param in stencil.extra_params.values():
    p.println('buffer_t %s;' % param.name)

  p.println('memset(&%s, 0, sizeof(buffer_t));' % stencil.input.name)
  p.println('memset(&%s, 0, sizeof(buffer_t));' % stencil.output.name)
  for param in stencil.extra_params.values():
    p.println('memset(&%s, 0, sizeof(buffer_t));' % param.name)
  p.println()

  for b in stencil.tensors.values():
    p.println('%s* %s_img = new %s[%s]();' % (b.type, b.name, b.type, '*'.join(['dims[%d]' % x for x in range(stencil.dim)]+[str(b.chan)])))
  for param in stencil.extra_params.values():
    p.println('%s %s_img%s;' % (param.type, param.name, reduce(operator.add, ['[%s]'%x for x in param.size])))
  p.println()

  for b in stencil.tensors.values():
    for d in range(stencil.dim):
      p.println('%s.extent[%d] = dims[%d];' % (b.name, d, d))
    if b.chan>1:
      p.println('%s.extent[%d] = %d;' % (b.name, stencil.dim, b.chan))
    p.println('%s.stride[0] = 1;' % b.name)
    for d in range(1, stencil.dim + (1 if b.chan>1 else 0)):
      p.println('%s.stride[%d] = %s;' % (b.name, d, '*'.join(['dims[%d]' % x for x in range(d)])))
    p.println('%s.elem_size = sizeof(%s);' % (b.name, b.type))
    p.println('%s.host = (uint8_t*)%s_img;' % (b.name, b.name))
    p.println()

  for param in stencil.extra_params.values():
    for d, size in enumerate(param.size):
      p.println('%s.extent[%d] = %d;' % (param.name, d, size))
    p.println('%s.stride[0] = 1;' % param.name)
    for d in range(1, len(param.size)):
      p.println('%s.stride[%d] = %s;' % (param.name, d, '*'.join([str(x) for x in param.size[:d]])))
    p.println('%s.elem_size = sizeof(%s);' % (param.name, param.type))
    p.println('%s.host = (uint8_t*)%s_img;' % (param.name, param.name))
    p.println()

  p.println('// initialization can be parallelized with -fopenmp')
  p.println('#pragma omp parallel for', 0)
  if stencil.input.chan>1:
    p.println('for(int32_t %c = 0; %c<%d; ++%c)' % (core.COORDS_IN_ORIG[stencil.dim], core.COORDS_IN_ORIG[stencil.dim], stencil.input.chan, core.COORDS_IN_ORIG[stencil.dim]))
    p.do_scope()
  for d in range(0, stencil.dim):
    dim = stencil.dim-d-1
    p.println('for(int32_t %c = 0; %c<dims[%d]; ++%c)' % (core.COORDS_IN_ORIG[dim], core.COORDS_IN_ORIG[dim], dim, core.COORDS_IN_ORIG[dim]))
    p.do_scope()
  init_val = '+'.join(core.COORDS_IN_ORIG[0:input_dim])
  if core.is_float(stencil.input.type):
    init_val = '%s(%s)/%s(%s%s)' % (stencil.input.type, init_val, stencil.input.type, '%d+' % stencil.input.chan if stencil.input.chan>1 else '', '+'.join('dims[%d]' % d for d in range(stencil.dim)))
  p.println('%s_img[%s] = %s;' % (stencil.input.name, '+'.join('%c*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], stencil.input.name, d) for d in range(input_dim)), init_val))
  for d in range(0, input_dim):
    p.un_scope()
  p.println()

  for param in stencil.extra_params.values():
    p.println('#pragma omp parallel for', 0)
    for dim, size in enumerate(param.size):
      p.println('for(int32_t %c = 0; %c<%d; ++%c)' % (core.COORDS_IN_ORIG[dim], core.COORDS_IN_ORIG[dim], size, core.COORDS_IN_ORIG[dim]))
      p.do_scope()
    p.println('%s_img%s = %s;' % (param.name, reduce(operator.add, ['[%c]' % (core.COORDS_IN_ORIG[d]) for d in range(len(param.size))]), '+'.join(core.COORDS_IN_ORIG[0:len(param.size)])))
    for d in param.size:
      p.un_scope()
    p.println()

  p.println('%s(&%s, &%s, %sxclbin);' % (stencil.app_name, stencil.input.name, stencil.output.name, ''.join(['&%s, ' % param.name for param in stencil.extra_params.values()])))
  p.println()

  p.println('int error_count = 0;')
  p.println()

  LoadPrinter = lambda node: '%s_img%s' % (node.name, ''.join(['[%d]'%x for x in node.idx])) if node.name in stencil.extra_params else '%s_img[%s+%d*%s.stride[%d]]' % (node.name, '+'.join(['(%c%+d)*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], node.idx[d] - s.idx[d], stencil.input.name, d) for d in range(stencil.dim)]), node.chan, node.name, stencil.dim)
  StorePrinter = lambda node: '%s result_chan_%d' % (stencil.output.type, node.chan) if node.name == stencil.output.name else '%s_img[%s+%d*%s.stride[%d]]' % (node.name, '+'.join(['%c*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], stencil.input.name, d) for d in range(stencil.dim)]), node.chan, node.name, stencil.dim)

  for s in stencil.get_stages_chronologically():
    logger.debug('emit code to produce %s_img' % s.output.name)
    p.println('// produce %s, can be parallelized with -fopenmp' % s.output.name)
    p.println('#pragma omp parallel for', 0)
    if stencil.iterate:
      stencil_window_input_queue = deque([s.output])
      while len(stencil_window_input_queue)>0:
        stencil_window_input = stencil_window_input_queue.popleft()
        if stencil_window_input.parent is None:
          break
        parent_tensors = stencil_window_input.parent.inputs
        if len(parent_tensors)==1 and next(iter(parent_tensors.values())).parent is not None and next(iter(parent_tensors.values())).parent.preserve_border_from() is not None:
          stencil_window_input = next(iter(parent_tensors.values()))
          break
        stencil_window_input_queue += parent_tensors.values()
        stencil_window_input = stencil_window_input.parent.output
      logger.debug('preserving border from %s' % stencil_window_input.name)
    else:
      stencil_window_input = stencil.input
    stencil_window = core.get_overall_stencil_window(stencil_window_input, s.output)
    stencil_dim = core.get_stencil_dim(stencil_window)
    output_idx = core.get_stencil_window_offset(stencil_window)
    for d in range(0, stencil.dim):
      dim = stencil.dim-d-1
      p.println('for(int32_t %c = %d; %c<dims[%d]-%d; ++%c)' % (core.COORDS_IN_ORIG[dim], output_idx[dim], core.COORDS_IN_ORIG[dim], dim, stencil_dim[dim]-output_idx[dim]-1, core.COORDS_IN_ORIG[dim]))
      p.do_scope()
    for e in s.expr:
      e.print_code(p, stencil.tensors, LoadPrinter, StorePrinter)
    p.println()
    if len(s.output.children)==0:
      for c in range(stencil.output.chan):
        run_result = '%s_img[%s+%d*%s.stride[%d]]' % (stencil.output.name, '+'.join(['%c*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], stencil.output.name, d) for d in range(stencil.dim)]), c, stencil.output.name, stencil.dim)
        p.println('%s val_fpga = %s;' % (stencil.output.type, run_result))
        p.println('%s val_cpu = result_chan_%d;' % (stencil.output.type, c))
        if core.is_float(stencil.output.type):
          p.println('double threshold = 0.00001;')
          p.println('if(nullptr!=getenv("THRESHOLD"))')
          p.do_scope()
          p.println('threshold = atof(getenv("THRESHOLD"));')
          p.un_scope()
          p.println('threshold *= threshold;')
          p.println('if(double(val_fpga-val_cpu)*double(val_fpga-val_cpu)/(double(val_cpu)*double(val_cpu)) > threshold)')
          p.do_scope()
          params = (c, ', '.join(['%d']*stencil.dim), ', '.join(core.COORDS_IN_ORIG[:stencil.dim]))
          p.println('fprintf(*error_report, "%%lf != %%lf @[%d](%s)\\n", double(val_fpga), double(val_cpu), %s);' % params)
        else:
          p.println('if(val_fpga!=val_cpu)')
          p.do_scope()
          params = (c, ', '.join(['%d']*stencil.dim), ', '.join(core.COORDS_IN_ORIG[:stencil.dim]))
          p.println('fprintf(*error_report, "%%ld != %%ld @[%d](%s)\\n", int64_t(val_fpga), int64_t(val_cpu), %s);' % params)
        p.println('++error_count;')
        p.un_scope()
    for d in range(0, stencil.dim):
      p.un_scope()
    p.println()

    if s.preserve_border_from():
      p.println('// handle borders for iterative stencil')
      p.println('#pragma omp parallel for', 0)
      bb = s.preserve_border_from()
      stencil_window = core.get_overall_stencil_window(bb, s.output)
      stencil_dim = core.get_stencil_dim(stencil_window)
      output_idx = core.get_stencil_window_offset(stencil_window)
      for d in range(0, stencil.dim):
        dim = stencil.dim-d-1
        p.println('for(int32_t %c = 0; %c<dims[%d]; ++%c)' % (core.COORDS_IN_ORIG[dim], core.COORDS_IN_ORIG[dim], dim, core.COORDS_IN_ORIG[dim]))
        p.do_scope()
      p.println('if(!(%s))' % ' && '.join('%c>=%d && %c<dims[%d]-%d' % (core.COORDS_IN_ORIG[d], output_idx[d], core.COORDS_IN_ORIG[d], d, stencil_dim[d]-output_idx[d]-1) for d in range(stencil.dim)))
      p.do_scope()
      GroudTruth = lambda c: '%s_img[%s+%d*%s.stride[%d]]' % (bb.name, '+'.join(['%c*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], bb.name, d) for d in range(stencil.dim)]), c, bb.name, stencil.dim)
      for e in s.expr:
        p.println('%s = %s;' % (StorePrinter(e), GroudTruth(e.chan)))
      if len(s.output.children)==0:
        for c in range(stencil.output.chan):
          run_result = '%s_img[%s+%d*%s.stride[%d]]' % (stencil.output.name, '+'.join(['%c*%s.stride[%d]' % (core.COORDS_IN_ORIG[d], stencil.output.name, d) for d in range(stencil.dim)]), c, stencil.output.name, stencil.dim)
          p.println('%s val_fpga = %s;' % (stencil.output.type, run_result))
          p.println('%s val_cpu = result_chan_%d;' % (stencil.output.type, c))
          if core.is_float(stencil.output.type):
            p.println('double threshold = 0.00001;')
            p.println('if(nullptr!=getenv("THRESHOLD"))')
            p.do_scope()
            p.println('threshold = atof(getenv("THRESHOLD"));')
            p.un_scope()
            p.println('threshold *= threshold;')
            p.println('if(double(val_fpga-val_cpu)*double(val_fpga-val_cpu)/(double(val_cpu)*double(val_cpu)) > threshold)')
            p.do_scope()
            params = (c, ', '.join(['%d']*stencil.dim), ', '.join(core.COORDS_IN_ORIG[:stencil.dim]))
            p.println('fprintf(*error_report, "%%lf != %%lf @[%d](%s)\\n", double(val_fpga), double(val_cpu), %s);' % params)
          else:
            p.println('if(val_fpga!=val_cpu)')
            p.do_scope()
            params = (c, ', '.join(['%d']*stencil.dim), ', '.join(core.COORDS_IN_ORIG[:stencil.dim]))
            p.println('fprintf(*error_report, "%%ld != %%ld @[%d](%s)\\n", int64_t(val_fpga), int64_t(val_cpu), %s);' % params)
          p.println('++error_count;')
          p.un_scope()
      p.un_scope()
      for d in range(0, stencil.dim):
        p.un_scope()
      p.println()

  p.println('if(error_count==0)')
  p.do_scope()
  p.println('fprintf(*error_report, "INFO: PASS!\\n");')
  p.un_scope()
  p.println('else')
  p.do_scope()
  p.println('fprintf(*error_report, "INFO: FAIL!\\n");')
  p.un_scope()
  p.println()

  for var in [stencil.input.name, stencil.output.name]:
    p.println('delete[] %s_img;' % var)
  p.println()

  p.println('return error_count;')
  p.un_scope()

def print_code(stencil, host_file):
  logger.info('generate host source code as %s' % host_file.name)
  p = core.Printer(host_file)
  print_header(p)
  p.println('#include"%s.h"' % stencil.app_name)
  p.println()

  core.print_define(p, 'BURST_WIDTH', stencil.burst_width)
  core.print_define(p, 'PIXEL_WIDTH_I', core.TYPE_WIDTH[stencil.input.type])
  core.print_define(p, 'PIXEL_WIDTH_O', core.TYPE_WIDTH[stencil.output.type])

  if stencil.preserve_border:
    overall_stencil_window = core.get_overall_stencil_window(
      stencil.output.parent.preserve_border_from(), stencil.output)
  else:
    overall_stencil_window = core.get_overall_stencil_window(
      stencil.input, stencil.output)

  overall_stencil_distance = core.get_stencil_distance(overall_stencil_window,
    stencil.tile_size)

  for i, dim in enumerate(core.get_stencil_dim(overall_stencil_window)):
    core.print_define(p, 'STENCIL_DIM_%d' % i, dim)
  stencil_offset = overall_stencil_distance - core.serialize(
    core.get_stencil_window_offset(overall_stencil_window), stencil.tile_size)
  if stencil.preserve_border:
    stencil_offset *= stencil.iterate

  overall_stencil_distance = max(overall_stencil_distance, stencil_offset)

  core.print_define(p, 'STENCIL_OFFSET', stencil_offset)
  core.print_define(p, 'STENCIL_DISTANCE', overall_stencil_distance)
  p.println()

  print_load_xclbin2(p)
  print_halide_rewrite_buffer(p)
  print_halide_error_codes(p)
  print_halide_error_report(p)
  print_wrapped(p, stencil)
  print_entrance(p, stencil)
  print_test(p, stencil)

