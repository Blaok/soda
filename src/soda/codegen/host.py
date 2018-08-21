from collections import deque
from functools import reduce
import copy
import logging
import operator

from soda import core
from soda import grammar
from soda import util

logger = logging.getLogger('__main__').getChild(__name__)

def print_header(printer):
  for header in ['assert', 'float', 'math', 'stdbool', 'stdint', 'stdio',
                 'stdlib', 'string', 'fcntl', 'time', 'unistd', 'sys/types',
                 'sys/stat', 'CL/opencl']:
    printer.println('#include<%s.h>' % header)
  printer.println()

def print_load_xclbin2(printer):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope

  println('FILE* const* error_report = &stderr;')
  println()
  println('int64_t load_xclbin2_to_memory(const char *filename, '
          'char **kernel_binary, char** device)')
  do_scope()
  println('uint64_t size = 0;')
  println('FILE *f = fopen(filename, "rb");')
  println('if(nullptr == f)')
  do_scope()
  println('*kernel_binary = nullptr;')
  println('fprintf(*error_report, "ERROR: cannot open %s\\n", filename);')
  println('return -1;')
  un_scope()
  println('char magic[8];')
  println('unsigned char cipher[32];')
  println('unsigned char key_block[256];')
  println('uint64_t unique_id;')
  println('fread(magic, sizeof(magic), 1, f);')
  println('if(strcmp(magic, "xclbin2")!=0)')
  do_scope()
  println('*kernel_binary = nullptr;')
  println('fprintf(*error_report, "ERROR: %s is not a valid xclbin2 file\\n", '
          'filename);')
  println('return -2;')
  un_scope()
  println('fread(cipher, sizeof(cipher), 1, f);')
  println('fread(key_block, sizeof(key_block), 1, f);')
  println('fread(&unique_id, sizeof(unique_id), 1, f);')
  println('fread(&size, sizeof(size), 1, f);')
  println('char* p = new char[size+1]();')
  println('*kernel_binary = p;')
  println('memcpy(p, magic, sizeof(magic));')
  println('p += sizeof(magic);')
  println('memcpy(p, cipher, sizeof(cipher));')
  println('p += sizeof(cipher);')
  println('memcpy(p, key_block, sizeof(key_block));')
  println('p += sizeof(key_block);')
  println('memcpy(p, &unique_id, sizeof(unique_id));')
  println('p += sizeof(unique_id);')
  println('memcpy(p, &size, sizeof(size));')
  println('p += sizeof(size);')
  println('uint64_t size_left = size - sizeof(magic) - sizeof(cipher) - '
          'sizeof(key_block) - sizeof(unique_id) - sizeof(size);')
  println('if(size_left != fread(p, sizeof(char), size_left, f))')
  do_scope()
  println('delete[] p;')
  println('fprintf(*error_report, "ERROR: %s is corrupted\\n", filename);')
  println('return -3;')
  un_scope()
  println('*device = p + 5*8;')
  println('fclose(f);')
  println('return size;')
  un_scope()
  println()

def print_halide_rewrite_buffer(printer):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope

  println('static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,')
  for dim in range(4):
    println('                  int32_t min{0}, int32_t extent{0}, '
            'int32_t stride{0}{1}'.format(dim, ',,,)'[dim]))
  do_scope()
  for item in 'min', 'extent', 'stride':
    for dim in range(4):
      println('b->{0}[{1}] = {0}{1};'.format(item, dim))
  println('return true;')
  un_scope()
  println()

def print_halide_error_codes(printer):
  i = 0
  for item in [
      'success', 'generic_error', 'explicit_bounds_too_small', 'bad_elem_size',
      'access_out_of_bounds', 'buffer_allocation_too_large',
      'buffer_extents_too_large', 'constraints_make_required_region_smaller',
      'constraint_violated', 'param_too_small', 'param_too_large',
      'out_of_memory', 'buffer_argument_is_null', 'debug_to_file_failed',
      'copy_to_host_failed', 'copy_to_device_failed', 'device_malloc_failed',
      'device_sync_failed', 'device_free_failed', 'no_device_interface',
      'matlab_init_failed', 'matlab_bad_param_type', 'internal_error',
      'device_run_failed', 'unaligned_host_ptr', 'bad_fold',
      'fold_factor_too_small']:
    printer.println('int halide_error_code_%s = %d;' % (item, i))
    i -= 1
  printer.println()

def print_halide_error_report(printer):
  println = printer.println
  println('int halide_error_bad_elem_size(void *user_context, '
          'const char *func_name,')
  println('                 const char *type_name, int elem_size_given, '
          'int correct_elem_size) {')
  println('  fprintf(*error_report, "%s has type %s but elem_size of the '
          'buffer passed in is %d instead of %d",')
  println('      func_name, type_name, elem_size_given, correct_elem_size);')
  println('  return halide_error_code_bad_elem_size;')
  println('}')
  println('int halide_error_constraint_violated(void *user_context, '
          'const char *var, int val,')
  println('                   const char *constrained_var, '
          'int constrained_val) {')
  println('  fprintf(*error_report, "Constraint violated: %s (%d) == %s (%d)",')
  println('      var, val, constrained_var, constrained_val);')
  println('  return halide_error_code_constraint_violated;')
  println('}')
  println('int halide_error_buffer_allocation_too_large(void *user_context, '
          'const char *buffer_name, uint64_t allocation_size, '
          'uint64_t max_size) {')
  println('  fprintf(*error_report, "Total allocation for buffer %s is %lu, '
          'which exceeds the maximum size of %lu",')
  println('      buffer_name, allocation_size, max_size);')
  println('  return halide_error_code_buffer_allocation_too_large;')
  println('}')
  println('int halide_error_buffer_extents_too_large(void *user_context, '
          'const char *buffer_name, int64_t actual_size, int64_t max_size) {')
  println('  fprintf(*error_report, "Product of extents for buffer %s is %ld, '
          'which exceeds the maximum size of %ld",')
  println('      buffer_name, actual_size, max_size);')
  println('  return halide_error_code_buffer_extents_too_large;')
  println('}')
  println('int halide_error_access_out_of_bounds(void *user_context, '
          'const char *func_name, int dimension, int min_touched, '
          'int max_touched, int min_valid, int max_valid) {')
  println('  if(min_touched < min_valid) {')
  println('    fprintf(*error_report, "%s is accessed at %d, which is before '
          'the min (%d) in dimension %d", func_name, min_touched, min_valid, '
          'dimension);')
  println('  } else if(max_touched > max_valid) {')
  println('    fprintf(*error_report, "%s is acccessed at %d, which is beyond '
          'the max (%d) in dimension %d", func_name, max_touched, max_valid, '
          'dimension);')
  println('  }')
  println('  return halide_error_code_access_out_of_bounds;')
  println('}')
  println()

def print_wrapped(printer, stencil):
  println = printer.println
  do_indent = printer.do_indent
  un_indent = printer.un_indent
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  new_var = printer.new_var
  last_var = printer.last_var
  tensors = [(_.name, _.c_type) for _ in stencil.input_stmts] + \
            [(_.name, _.c_type) for _ in stencil.output_stmts] + \
            [(_.name, _.c_type) for _ in stencil.param_stmts]
  println('static int %s_wrapped(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' %
          (stencil.app_name,
           ''.join([('buffer_t *var_%s_buffer, ') % _[0] for _ in tensors])))
  do_scope()
  for tensor in tensors:
    print_unload_buffer(printer, *tensor)

  for output_name in stencil.output_names:
    output_type = stencil.tensors[output_name].soda_type
    println('if(var_%s_host_and_dev_are_null)' % output_name)
    do_scope('var_%s_host_and_dev_are_null' % output_name)
    output_str = [", 0, 0, 0"]*4
    for dim in range(stencil.dim):
      stride  = '1'
      if dim > 0:
        stride = '*'.join(('%s_size_dim_%d' % (output_name, d))
                          for d in range(dim))
      output_str[dim] = (', var_{0}_min_{1}, {0}_size_dim_{1}, {2}'.format(
          output_name, dim, stride))
    println('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' %
            (new_var(), output_name, util.get_width_in_bytes(output_type),
             *output_str))
    println('(void)%s;' % last_var())
    un_scope()

  for input_name in stencil.input_names:
    input_type = stencil.tensors[input_name].soda_type
    println('if(var_%s_host_and_dev_are_null)' % input_name)
    do_scope('var_%s_host_and_dev_are_null' % input_name)
    input_size = ['0']*4
    for dim in range(stencil.dim):
      println('int32_t %s = %s_size_dim_%d + %d;' % (
          new_var(), stencil.output_names[0], dim,
          core.get_stencil_dim(core.get_overall_stencil_window(
              stencil.tensors[stencil.input_names[0]],
              stencil.tensors[stencil.output_names[0]]))[dim]-1))
    input_str = [', 0, 0, 0']*4
    for dim in range(stencil.dim):
      stride  = '1'
      if dim > 0:
        stride = '*'.join([last_var(x-stencil.dim) for x in range(dim)])
      input_str[dim] = (
          ', var_%s_min_%d, %s, %s' % (stencil.output_names[0], dim,
                                       last_var(dim-stencil.dim), stride))
    println('bool %s = halide_rewrite_buffer(var_%s_buffer, %d%s%s%s%s);' %
            (new_var(), input_name, util.get_width_in_bytes(input_type),
             *input_str))
    println('(void)%s;' % last_var())
    un_scope()

  println('bool %s = %s;' % (new_var(), ' || '.join(
      'var_%s_host_and_dev_are_null' % _
      for _ in stencil.output_names + stencil.input_names)))
  println('bool %s = !(%s);' % (new_var(), last_var(-2)))
  println('if(%s)' % last_var())
  do_scope('if(%s)' % last_var())

  for name in stencil.output_names + stencil.input_names + stencil.param_names:
    print_check_elem_size(printer, name, stencil.tensors[name].soda_type)
  println()

  println('// allocate buffer for tiled input/output')
  for d in range(stencil.dim-1):
    println('int32_t tile_num_dim_{d} = ({}_size_dim_{d}-STENCIL_DIM_{d}+1+'
            'TILE_SIZE_DIM_{d}-STENCIL_DIM_{d})/(TILE_SIZE_DIM_{d}-'
            'STENCIL_DIM_{d}+1);'.format(
        stencil.input_names[0], d = d))
  println()

  println('// change #bank if there is a env var defined')
  println('int dram_bank = %d;' % util.MAX_DRAM_BANK)
  println('bool dram_separate = false;')
  println('if(nullptr!=getenv("DRAM_BANK"))')
  do_scope()
  println('dram_bank = atoi(getenv("DRAM_BANK"));')
  un_scope()
  println('if(nullptr!=getenv("DRAM_SEPARATE"))')
  do_scope()
  println('dram_separate = true;')
  println('dram_bank /= 2;')
  un_scope()
  println()

  println('// align each linearized tile to multiples of BURST_WIDTH')
  println('int64_t tile_pixel_num = %s*%s_size_dim_%d;' % (
      '*'.join('TILE_SIZE_DIM_%d'%x for x in range(stencil.dim-1)),
      stencil.input_names[0], stencil.dim-1))
  println('int64_t tile_burst_num = '
          '(tile_pixel_num-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1;')
  println('int64_t tile_size_linearized_i = '
          'tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);')
  println('int64_t tile_size_linearized_o = '
          'tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_O*dram_bank);')
  println()

  println('// prepare for opencl')
  println('int err;')
  println()
  println('cl_platform_id platforms[16];')
  println('cl_platform_id platform_id;')
  println('cl_uint platform_count;')
  println('cl_device_id device_id;')
  println('cl_context context;')
  println('cl_command_queue commands;')
  println('cl_program program;')
  println('cl_kernel kernel;')
  println()
  println('char cl_platform_vendor[1001];')
  println()

  for name in stencil.input_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('cl_mem var_%s_bank_%d_cl;' % (name, bank))
  for name in stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('cl_mem var_%s_bank_%d_cl;' % (name, bank))
  println()

  for name in stencil.param_names:
    println('cl_mem var_%s_cl;' % name)
  println()

  for stmt in stencil.input_stmts:
    println('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_i/'
            'dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*'
            'dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_I));' % (
                stmt.name, stmt.c_type,
                '*'.join('tile_num_dim_%d'%_ for _ in range(stencil.dim-1))))
  for stmt in stencil.output_stmts:
    println('uint64_t var_%s_buf_size = sizeof(%s)*(%s*tile_size_linearized_o/'
            'dram_bank+((STENCIL_DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_O*'
            'dram_bank)+1)*(BURST_WIDTH/PIXEL_WIDTH_O));' % (
                stmt.name, stmt.c_type,
                '*'.join('tile_num_dim_%d'%x for x in range(stencil.dim-1))))
  println()

  println('unsigned char *kernel_binary;')
  println('const char *device_name;')
  println('char target_device_name[64];')
  println('fprintf(*error_report, "INFO: Loading %s\\n", xclbin);')
  println('int64_t kernel_binary_size = load_xclbin2_to_memory(xclbin, (char **) &kernel_binary, (char**)&device_name);')
  println('if(kernel_binary_size < 0)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to load kernel from xclbin: %s\\n", xclbin);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println('for(int i = 0; i<64; ++i)')
  do_scope()
  println("target_device_name[i] = (device_name[i]==':'||device_name[i]=='.') ? '_' : device_name[i];")
  un_scope()
  println()

  println('err = clGetPlatformIDs(16, platforms, &platform_count);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to find an OpenCL platform\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println('fprintf(*error_report, "INFO: Found %d platforms\\n", platform_count);')
  println()

  println('int platform_found = 0;')
  println('for (unsigned iplat = 0; iplat<platform_count; iplat++)')
  do_scope()
  println('err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,nullptr);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println('if(strcmp(cl_platform_vendor, "Xilinx") == 0)')
  do_scope()
  println('fprintf(*error_report, "INFO: Selected platform %d from %s\\n", iplat, cl_platform_vendor);')
  println('platform_id = platforms[iplat];')
  println('platform_found = 1;')
  un_scope()
  un_scope()
  println('if(!platform_found)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Platform Xilinx not found\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('cl_device_id devices[16];')
  println('cl_uint device_count;')
  println('unsigned int device_found = 0;')
  println('char cl_device_name[1001];')
  println('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
  println('           16, devices, &device_count);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('for (unsigned int i=0; i<device_count; ++i)')
  do_scope()
  println('err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to get device name for device %d\\n", i);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println('printf("INFO: Find device %s\\n", cl_device_name);')
  println('if(strcmp(cl_device_name, target_device_name) == 0 || strcmp(cl_device_name, device_name) == 0)')
  do_scope()
  println('device_id = devices[i];')
  println('device_found = 1;')
  println('fprintf(*error_report, "INFO: Selected %s as the target device\\n", device_name);')
  un_scope()
  un_scope()
  println()

  println('if(!device_found)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Target device %s not found\\n", target_device_name);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,')
  println('           1, &device_id, nullptr);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create a device group\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);')
  println('if(!context)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create a compute context\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('commands = clCreateCommandQueue(context, device_id, 0, &err);')
  println('if(!commands)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create a command commands %i\\n",err);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('int status;')
  println('size_t kernel_binary_sizes[1] = {static_cast<size_t>('
          'kernel_binary_size)};')
  println('program = clCreateProgramWithBinary(context, 1, &device_id, kernel_binary_sizes,')
  println('                   (const unsigned char **) &kernel_binary, &status, &err);')
  println('if((!program) || (err!=CL_SUCCESS))')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create compute program from binary %d\\n", err);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println('delete[] kernel_binary;')
  println()

  println('err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);')
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('size_t len;')
  println('char buffer[2048];')
  println('fprintf(*error_report, "ERROR: Failed to build program executable\\n");')
  println('clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);')
  println('fprintf(*error_report, "%s\\n", buffer);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('kernel = clCreateKernel(program, "%s_kernel", &err);' % stencil.app_name)
  println('if(!kernel || err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to create compute kernel %d\\n", err);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  for name in stencil.input_names + stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('cl_mem_ext_ptr_t %s_ext_bank_%d;' % (name, bank))
  println('if(dram_separate)')
  do_scope()
  println('switch(dram_bank)')
  do_scope()
  for num_banks in range(1, util.MAX_DRAM_BANK//2+1):
    println('case %d:' % num_banks)
    do_indent()
    for name in stencil.input_names + stencil.output_names:
      for bank in range(num_banks):
        println('%s_ext_bank_%d.flags = XCL_MEM_DDR_BANK%d;' % (
            name, bank%num_banks, bank))
    println('break;')
    un_indent()
  un_scope()
  un_scope()
  println('else')
  do_scope()
  for name in stencil.input_names + stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('{0}_ext_bank_{1}.flags = XCL_MEM_DDR_BANK{1};'.format(
          name, bank))
  un_scope()

  for name in stencil.input_names + stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('%s_ext_bank_%d.obj = 0;' % (name, bank))
      println('%s_ext_bank_%d.param = 0;' % (name, bank))
  println()

  for name in stencil.input_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('var_{name}_bank_{bank}_cl = dram_bank > {bank} ? '
              'clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX,'
              ' var_{name}_buf_size, &{name}_ext_bank_{bank}, nullptr) : '
              'nullptr;'.format(name=name, bank=bank))
  for name in stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('var_{name}_bank_{bank}_cl = dram_bank > {bank} ? '
              'clCreateBuffer(context, '
              'CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, var_{name}_buf_size, '
              '&{name}_ext_bank_{bank}, nullptr) : nullptr;'.format(
                  name=name, bank=bank))

  for param in stencil.param_stmts:
    println('var_%s_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, '
            '%s*sizeof(%s), nullptr, nullptr);' % (
                param.name, '*'.join(map(str, param.size)), param.type))

  println('if(%s)' % ' || '.join(['(dram_bank > %d && (%s || %s))' % (
          bank, ' || '.join('!var_%s_bank_%d_cl' % (name, bank)
                            for name in stencil.input_names),
          ' || '.join('!var_%s_bank_%d_cl' % (name, bank)
                      for name in stencil.output_names))
      for bank in range(util.MAX_DRAM_BANK)] + [
          '!var_%s_cl' % _ for _ in stencil.param_names]))
  do_scope()
  println('fprintf(*error_report, '
          '"ERROR: Failed to allocate device memory\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('cl_event write_events[%d];' % (
      util.MAX_DRAM_BANK*len(stencil.input_stmts)+len(stencil.param_stmts)))
  idx = 0
  for param in stencil.param_stmts:
    println('%s* var_%s_buf = (%s*)clEnqueueMapBuffer(commands, var_%s_cl, '
            'CL_FALSE, CL_MAP_WRITE, 0, %s*sizeof(%s), 0, nullptr, '
            'write_events+%d, &err);' % (
                param.c_type, param.name, param.c_type, param.name,
                '*'.join(map(str, param.size)), param.c_type, idx))
    idx += 1
  for bank in range(util.MAX_DRAM_BANK):
    for stmt in stencil.input_stmts:
      println('{stmt.c_type}* var_{stmt.name}_buf_bank_{bank} = '
              'dram_bank>{bank} ? ({stmt.c_type}*)clEnqueueMapBuffer(commands,'
              ' var_{stmt.name}_bank_{bank}_cl, CL_FALSE, CL_MAP_WRITE, 0, '
              'var_{stmt.name}_buf_size, 0, nullptr, write_events+{idx}, &err)'
              ' : nullptr;'.format(stmt=stmt, bank=bank, idx=idx))
      idx += 1
  println('clWaitForEvents(dram_bank*%d+%d, write_events);' % (
      len(stencil.input_names), len(stencil.param_stmts)))
  println()

  println('// tiling')
  for dim in range(stencil.dim-2, -1, -1):
    println('for(int32_t tile_index_dim_{0} = 0; tile_index_dim_{0} < '
            'tile_num_dim_{0}; ++tile_index_dim_{0})'.format(dim))
    do_scope()
    println('int32_t actual_tile_size_dim_{0} = (tile_index_dim_{0}==tile_num_'
            'dim_{0}-1) ? {1}_size_dim_{0}-(TILE_SIZE_DIM_{0}-STENCIL_DIM_{0}+'
            '1)*tile_index_dim_{0} : TILE_SIZE_DIM_{0};'.format(
                dim, stencil.input_names[0]))

  println('for(int32_t {0} = 0; {0} < {1}_size_dim_{2}; ++{0})'.format(
      util.COORDS_IN_TILE[stencil.dim-1], stencil.input_names[0],
      stencil.dim-1))
  do_scope()
  for dim in range(stencil.dim-2, -1, -1):
    println('for(int32_t {0} = 0; {0} < actual_tile_size_dim_{1}; '
            '++{0})'.format(util.COORDS_IN_TILE[dim], dim))
    do_scope()

  println('// (%s) is coordinates in tiled image' %
          ', '.join(util.COORDS_TILED))
  println('// (%s) is coordinates in original image' %
          ', '.join(util.COORDS_IN_ORIG))
  println('// (%s) is coordinates in a tile' % ', '.join(util.COORDS_IN_TILE))
  offset_in_tile = '+'.join('%c%s' % (
      util.COORDS_IN_TILE[x],
      ''.join('*TILE_SIZE_DIM_%d' % xx for xx in range(x)))
                            for x in range(stencil.dim))
  println('int32_t burst_index = (%s)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank);' %
          offset_in_tile)
  println('int32_t burst_residue = (%s)%%(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)'
          ';' % offset_in_tile)
  for dim in range(stencil.dim-1):
    println('int32_t {0} = tile_index_dim_{1}*(TILE_SIZE_DIM_{1}-STENCIL_DIM_'
            '{1}+1)+{2};'.format(util.COORDS_IN_ORIG[dim], dim,
                                 util.COORDS_IN_TILE[dim]))
  println('int32_t %c = %c;' % (util.COORDS_IN_ORIG[stencil.dim-1],
                                util.COORDS_IN_TILE[stencil.dim-1]))
  println('int64_t tiled_offset = (%s)*tile_size_linearized_i+burst_index*(BURS'
          'T_WIDTH/PIXEL_WIDTH_I*dram_bank)+burst_residue;' % '+'.join(
              '%stile_index_dim_%d' % (''.join(
                  'tile_num_dim_%d*'%xx for xx in range(x)), x)
              for x in range(stencil.dim-1)))
  println('int64_t original_offset = %s;' % '+'.join(
      '%c*var_%s_stride_%d' % (util.COORDS_IN_ORIG[x], stencil.input_names[0],
                               x) for x in range(stencil.dim)))
  println('switch(tiled_offset%dram_bank)')
  do_scope()
  for bank in range(util.MAX_DRAM_BANK):
    println('case %d:' % bank)
    do_indent()
    for name in stencil.input_names:
      println('var_{0}_buf_bank_{1}[tiled_offset/dram_bank] = var_{0}[original_'
              'offset];'.format(name, bank))
    println('break;')
    un_indent()
  for dim in range(stencil.dim*2):
    un_scope()
  println()

  for param in stencil.param_stmts:
    println('memcpy(var_{0}_buf, var_{0}, {1}*sizeof({2}));' % (
        param.name, '*'.join(map(str, param.size)), param.c_type))
  println()

  println('err = 0;')
  for idx, param in enumerate(stencil.param_stmts):
    println('err |= clEnqueueUnmapMemObject(commands, var_{0}_cl, var_{0}_buf, '
            '0, nullptr, write_events+{1});'.format(param.name, idx))
  idx = 0
  for bank in range(util.MAX_DRAM_BANK):
    for name in stencil.input_names:
      println('err |= dram_bank>{bank} ? clEnqueueUnmapMemObject(commands, var_'
              '{name}_bank_{bank}_cl, var_{name}_buf_bank_{bank}, 0, nullptr, w'
              'rite_events+{}) : err;'.format(idx, bank=bank, name=name))
      idx += 1
  println()

  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to write to input !\\n");')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('err = 0;')
  println('fprintf(*error_report, "INFO: Using %d DRAM bank%s%s\\n", dram_bank,'
          ' dram_bank>1 ? "s" : "", dram_separate ? ", separated" : "");')
  for d in range(stencil.dim-1):
    println('fprintf(*error_report, "INFO: tile_num_dim_{0} = %d, '
            'TILE_SIZE_DIM_{0} = %d\\n", tile_num_dim_{0}, '
            'TILE_SIZE_DIM_{0});'.format(d))
  for name in stencil.input_names:
    println('fprintf(*error_report, "INFO: %s\\n", %s);' % (
        ', '.join('%s_extent_%d = %%d' % (name, _) for _ in range(stencil.dim)),
        ', '.join('%s_size_dim_%d' % (name, _) for _ in range(stencil.dim))))
    println('fprintf(*error_report, "INFO: %s\\n", %s);' % (
        ', '.join('%s_min_%d = %%d' % (name, _) for _ in range(stencil.dim)),
        ', '.join('var_%s_min_%d' % (name, _) for _ in range(stencil.dim))))
  for name in stencil.output_names:
    println('fprintf(*error_report, "INFO: %s\\n", %s);' % (
        ', '.join('%s_extent_%d = %%d' % (name, _) for _ in range(stencil.dim)),
        ', '.join('%s_size_dim_%d' % (name, _) for _ in range(stencil.dim))))
    println('fprintf(*error_report, "INFO: %s\\n", %s);' % (
        ', '.join('%s_min_%d = %%d' % (name, _) for _ in range(stencil.dim)),
        ', '.join('var_%s_min_%d' % (name, _) for _ in range(stencil.dim))))
  println()

  println('int kernel_arg = 0;')
  println('int64_t tile_data_num = ((int64_t(%s_size_dim_%d)%s-1)/(BURST_WIDTH/'
          'PIXEL_WIDTH_I*dram_bank)+1)*BURST_WIDTH/PIXEL_WIDTH_I*dram_bank/UNRO'
          'LL_FACTOR;' % (
              stencil.input_names[0], stencil.dim-1,
              ''.join('*TILE_SIZE_DIM_%d'%_ for _ in range(stencil.dim-1))))
  println('int64_t coalesced_data_num = ((int64_t(%s_size_dim_%d)%s*%s+STENCIL_'
          'DISTANCE-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_bank)+1);' % (
              stencil.input_names[0], stencil.dim-1,
              ''.join('*TILE_SIZE_DIM_%d'%_ for _ in range(stencil.dim-1)),
              '*'.join('tile_num_dim_%d'%_ for _ in range(stencil.dim-1))))
  for d in range(stencil.dim-1):
    println('uint32_t input_bound_dim_{0} = tile_num_dim_{0}*(TILE_SIZE_DIM_{0}'
            '-STENCIL_DIM_{0}+1);'.format(d))
  println('fprintf(*error_report, "INFO: tile_data_num = %ld, coalesced_data_nu'
          'm = %ld\\n", tile_data_num, coalesced_data_num);')
  println()

  for name in stencil.output_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, '
              'sizeof(cl_mem), &var_%s_bank_%d_cl);' % (bank, name, bank))
  for name in stencil.input_names:
    for bank in range(util.MAX_DRAM_BANK):
      println('if(dram_bank>%d) err |= clSetKernelArg(kernel, kernel_arg++, '
              'sizeof(cl_mem), &var_%s_bank_%d_cl);' % (bank, name, bank))
  for name in stencil.param_names:
    println('err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), '
            '&var_%s_cl);' % name)
  for name in stencil.input_names:
    for variable in (
        ['coalesced_data_num', 'tile_data_num']+
        ['input_bound_dim_%d'%x for x in range(stencil.dim-1)]+
        ['%s_size_dim_%d' % (name, _) for _ in range(stencil.dim)]):
      println('err |= clSetKernelArg(kernel, kernel_arg++, sizeof({0}), '
              '&{0});'.format(variable))
  println('if(err != CL_SUCCESS)')
  do_scope()
  println('fprintf(*error_report, "ERROR: Failed to set kernel arguments %d'
          '\\n", err);')
  println('exit(EXIT_FAILURE);')
  un_scope()
  println()

  println('timespec execute_begin, execute_end;')
  println('cl_event execute_event;')
  println('err = clEnqueueTask(commands, kernel, dram_bank, write_events, '
          '&execute_event);')
  println('if(nullptr==getenv("XCL_EMULATION_MODE"))')
  do_scope()
  println('fprintf(*error_report, "INFO: FPGA warm up\\n");')
  println('clWaitForEvents(1, &execute_event);')
  println()
  println('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_begin);')
  println('err = clEnqueueTask(commands, kernel, 0, nullptr, &execute_event);')
  println('if(err)')
  println('{')
  println('  fprintf(*error_report, "ERROR: Failed to execute kernel %d\\n", '
          'err);')
  println('  exit(EXIT_FAILURE);')
  println('}')
  println('clWaitForEvents(1, &execute_event);')
  println('clock_gettime(CLOCK_MONOTONIC_RAW, &execute_end);')
  println()
  println('double elapsed_time = 0.;')
  println('elapsed_time = (double(execute_end.tv_sec-execute_begin.tv_sec)+'
          '(execute_end.tv_nsec-execute_begin.tv_nsec)/1e9)*1e6;')
  println('printf("Kernel execution time: %lf us\\n", elapsed_time);')
  println('printf("Kernel throughput:   %%lf pixel/ns\\n", %s/elapsed_time/1e3)'
          ';' % '*'.join('%s_size_dim_%d' % (stencil.input_names[0], _)
                         for _ in range(stencil.dim)))
  un_scope()
  println('else')
  do_scope()
  println('clWaitForEvents(1, &execute_event);')
  println('fprintf(*error_report, "INFO: Emulation mode\\n");')
  un_scope()
  println()

  println('cl_event read_events[%d];' % (
      util.MAX_DRAM_BANK*len(stencil.output_stmts)))
  idx = 0
  for bank in range(util.MAX_DRAM_BANK):
    for stmt in stencil.output_stmts:
      println('{stmt.c_type}* var_{stmt.name}_buf_bank_{bank} = dram_bank>{bank'
              '} ? ({stmt.c_type}*)clEnqueueMapBuffer(commands, var_{stmt.name}_b'
              'ank_{bank}_cl, CL_FALSE, CL_MAP_READ, 0, var_{stmt.name}_buf_siz'
              'e, 0, nullptr, read_events+{idx}, &err) : nullptr;'.format(
                  stmt=stmt, bank=bank, idx=idx))
      idx += 1
  println('clWaitForEvents(dram_bank*%d, read_events);' %
          len(stencil.output_names))
  println()

  for dim in range(stencil.dim-2, -1, -1):
    println('for(int32_t tile_index_dim_{0} = 0; tile_index_dim_{0} < '
            'tile_num_dim_{0}; ++tile_index_dim_{0})'.format(dim))
    do_scope()
    println('int32_t actual_tile_size_dim_{0} = (tile_index_dim_{0}==tile_num_d'
            'im_{0}-1) ? {1}_size_dim_{0}-(TILE_SIZE_DIM_{0}-STENCIL_DIM_{0}+1)'
            '*tile_index_dim_{0} : TILE_SIZE_DIM_{0};'.format(
                dim, stencil.input_names[0]))

  if stencil.preserve_border:
    overall_stencil_window = util.get_overall_stencil_window(
        stencil.output.parent.preserve_border_from(), stencil.output)
    overall_stencil_offset = util.get_stencil_window_offset(
        overall_stencil_window)
    overall_stencil_dim = util.get_stencil_dim(overall_stencil_window)
    println('for(int32_t {var} = 0; {var} < {0}_size_dim_{1}; ++{var})'.format(
        stencil.input.name, stencil.dim-1,
        var=util.COORDS_IN_TILE[stencil.dim-1]))
    do_scope()
    for dim in range(stencil.dim-2, -1, -1):
      println('for(int32_t {var} = tile_index_dim_{dim}==0 ? 0 : {0}; {var} < a'
              'ctual_tile_size_dim_{dim}-(tile_index_dim_{dim}==tile_num_dim_{d'
              'im}-1 ? 0 : {1}); ++{var})'.format(
                  overall_stencil_offset[dim],
                  overall_stencil_dim[dim]-1-overall_stencil_offset[dim],
                  dim=dim, var=util.COORDS_IN_TILE[dim]))
      do_scope()
  else:
    overall_stencil_window = core.get_overall_stencil_window(
        stencil.tensors[stencil.input_names[0]],
        stencil.tensors[stencil.output_names[0]])
    overall_stencil_offset = core.get_stencil_window_offset(
        overall_stencil_window)
    overall_stencil_dim = core.get_stencil_dim(overall_stencil_window)
    println('for(int32_t {var} = {}; {var} < {}_size_dim_{}-{}; ++{var})'
            ''.format(overall_stencil_offset[stencil.dim-1],
                      stencil.output_names[0], stencil.dim-1,
                      overall_stencil_dim[stencil.dim-1]-1-
                      overall_stencil_offset[stencil.dim-1],
                      var=util.COORDS_IN_TILE[stencil.dim-1]))
    do_scope()
    for dim in range(stencil.dim-2, -1, -1):
      println('for(int32_t {var} = {}; {var} < actual_tile_size_dim_{}-{}; '
              '++{var})'.format(
          overall_stencil_offset[dim], dim,
          overall_stencil_dim[dim]-1-overall_stencil_offset[dim],
          var=util.COORDS_IN_TILE[dim]))
      do_scope()

  println('// (%s) is coordinates in tiled image' %
          ', '.join(util.COORDS_TILED))
  println('// (%s) is coordinates in original image' %
          ', '.join(util.COORDS_IN_ORIG))
  println('// (%s) is coordinates in a tile' %
          ', '.join(util.COORDS_IN_TILE))
  offset_in_tile = '+'.join(
      '%c%s' % (util.COORDS_IN_TILE[x],
                ''.join('*TILE_SIZE_DIM_%d'%xx for xx in range(x)))
      for x in range(stencil.dim))
  println('int32_t burst_index = (%s+STENCIL_OFFSET)/(BURST_WIDTH/PIXEL_WIDTH_O'
          '*dram_bank);' % offset_in_tile)
  println('int32_t burst_residue = (%s+STENCIL_OFFSET)%%(BURST_WIDTH/PIXEL_WIDT'
          'H_O*dram_bank);' % offset_in_tile)
  for dim in range(stencil.dim-1):
    println('int32_t {0} = tile_index_dim_{1}*(TILE_SIZE_DIM_{1}-STENCIL_DIM_{1'
            '}+1)+{2};'.format(util.COORDS_IN_ORIG[dim], dim,
                               util.COORDS_IN_TILE[dim]))
  println('int32_t %c = %c;' % (util.COORDS_IN_ORIG[stencil.dim-1],
                                util.COORDS_IN_TILE[stencil.dim-1]))
  println('int64_t tiled_offset = (%s)*tile_size_linearized_o+burst_index*(BURS'
          'T_WIDTH/PIXEL_WIDTH_O*dram_bank)+burst_residue;' % '+'.join(
              '%stile_index_dim_%d' % (''.join(
                  'tile_num_dim_%d*'%xx for xx in range(x)), x)
              for x in range(stencil.dim-1)))
  println('int64_t original_offset = %s;' % '+'.join(
      '%c*var_%s_stride_%d' % (
          util.COORDS_IN_ORIG[x], stencil.output_names[0], x)
      for x in range(stencil.dim)))
  println('switch(tiled_offset%dram_bank)')
  do_scope()
  for bank in range(util.MAX_DRAM_BANK):
    println('case %d:' % bank)
    do_indent()
    for name in stencil.output_names:
      println('var_%s[original_offset] = var_%s_buf_bank_%d[tiled_offset/'
              'dram_bank];' % (name, name, bank))
    println('break;')
    un_indent()
  for dim in range(stencil.dim*2):
    un_scope()

  for bank in range(util.MAX_DRAM_BANK):
    println()
    println('if(dram_bank>%d)' % bank)
    do_scope()
    for name in stencil.output_names:
      println('clEnqueueUnmapMemObject(commands, var_{name}_bank_{bank}_cl, var'
              '_{name}_buf_bank_{bank}, 0, nullptr, read_events+{bank});'
              ''.format(name=name, bank=bank))
    for name in stencil.input_names + stencil.output_names:
      println('clReleaseMemObject(var_%s_bank_%d_cl);' % (name, bank))
    un_scope()
  println('clWaitForEvents(dram_bank*%d, read_events);' %
          len(stencil.output_stmts))
  println()

  for name in stencil.param_names:
    println('clReleaseMemObject(var_%s_cl);' % name)
  println()

  println('clReleaseProgram(program);')
  println('clReleaseKernel(kernel);')
  println('clReleaseCommandQueue(commands);')
  println('clReleaseContext(context);')

  un_scope()
  println('return 0;')
  un_scope()
  println()

def print_entrance(printer, stencil):
  println = printer.println
  tensors = [(stmt.name, stmt.c_type)
             for stmt in stencil.input_stmts + stencil.output_stmts +
             stencil.param_stmts]
  println('int %s(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS' % (
      stencil.app_name, ''.join('buffer_t *var_%s_buffer, ' % x[0]
                                for x in tensors)))
  printer.do_scope()
  for tensor in tensors:
    print_unload_buffer(printer, *tensor)
  println('return %s_wrapped(%sxclbin);' % (
      stencil.app_name, ''.join(('var_%s_buffer, ') % x[0] for x in tensors)))
  printer.un_scope()
  println()

def print_unload_buffer(printer, buffer_name, buffer_type):
  println = printer.println
  println('%s *var_%s = (%s *)(var_%s_buffer->host);' % (
      buffer_type, buffer_name, buffer_type, buffer_name))
  println('(void)var_%s;' % buffer_name)
  println('const bool var_{0}_host_and_dev_are_null = (var_{0}_buffer->host == '
          'nullptr) && (var_{0}_buffer->dev == 0);'.format(buffer_name))
  println('(void)var_%s_host_and_dev_are_null;' % buffer_name)
  for item in ['min', 'extent', 'stride']:
    for i in range(4):
      if item == 'extent':
        println('int32_t %s_size_dim_%d = var_%s_buffer->%s[%d];' % (
            buffer_name, i, buffer_name, item, i))
        println('(void)%s_size_dim_%d;' % (buffer_name, i))
      else:
        println('int32_t var_%s_%s_%d = var_%s_buffer->%s[%d];' % (
            buffer_name, item, i, buffer_name, item, i))
        println('(void)var_%s_%s_%d;' % (buffer_name, item, i))
  println('int32_t var_{0}_elem_size = var_{0}_buffer->elem_size;'.format(
      buffer_name))
  println('(void)var_%s_elem_size;' % buffer_name)

def print_check_elem_size(printer, buffer_name, buffer_type):
  println = printer.println
  new_var = printer.new_var
  last_var = printer.last_var
  println('bool %s = var_%s_elem_size == %d;' % (
      new_var(), buffer_name, util.get_width_in_bytes(buffer_type)))
  println('if(!%s)' % last_var())
  printer.do_scope()
  println('int32_t %s = halide_error_bad_elem_size(nullptr, "Buffer %s", "%s",'
          ' var_%s_elem_size, %d);' % (
              new_var(), buffer_name, buffer_type, buffer_name,
              util.get_width_in_bytes(buffer_type)))
  println('return %s;' % last_var())
  printer.un_scope()

def print_test(printer, stencil):
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope

  input_dim = stencil.dim
  output_dim = stencil.dim

  println('int %s_test(const char* xclbin, const int dims[4])' %
          stencil.app_name)
  do_scope()
  for name in tuple(stencil.tensors) + stencil.param_names:
    println('buffer_t %s;' % name)
  for name in tuple(stencil.tensors) + stencil.param_names:
    println('memset(&%s, 0, sizeof(buffer_t));' % name)
  println()

  for tensor in stencil.tensors.values():
    println('%s* %s_img = new %s[%s]();' % (
        tensor.c_type, tensor.name, tensor.c_type,
        '*'.join('dims[%d]' % x for x in range(stencil.dim))))
  for param in stencil.param_stmts:
    println('%s %s_img%s;' % (
        param.c_type, param.name,
        reduce(operator.add, ['[%s]'%x for x in param.size])))
  println()

  for tensor in stencil.tensors.values():
    for d in range(stencil.dim):
      println('%s.extent[%d] = dims[%d];' % (tensor.name, d, d))
    println('%s.stride[0] = 1;' % tensor.name)
    for d in range(1, stencil.dim):
      println('%s.stride[%d] = %s;' % (
          tensor.name, d, '*'.join(['dims[%d]' % x for x in range(d)])))
    println('%s.elem_size = sizeof(%s);' % (tensor.name, tensor.c_type))
    println('{0}.host = (uint8_t*){0}_img;'.format(tensor.name))
    println()

  for param in stencil.param_stmts:
    for d, size in enumerate(param.size):
      println('%s.extent[%d] = %d;' % (param.name, d, size))
    println('%s.stride[0] = 1;' % param.name)
    for d in range(1, len(param.size)):
      println('%s.stride[%d] = %s;' % (
          param.name, d, '*'.join([str(x) for x in param.size[:d]])))
    println('%s.elem_size = sizeof(%s);' % (param.name, param.type))
    println('%s.host = (uint8_t*)%s_img;' % (param.name, param.name))
    println()

  for name in stencil.input_names:
    println('// initialization can be parallelized with -fopenmp')
    println('#pragma omp parallel for', 0)
    for d in range(0, stencil.dim):
      dim = stencil.dim-d-1
      println('for(int32_t {var} = 0; {var}<dims[{}]; ++{var})'.format(
          dim, var=util.COORDS_IN_ORIG[dim]))
      do_scope()
    init_val = '+'.join(util.COORDS_IN_ORIG[0:input_dim])
    if util.is_float(stencil.input_types[0]):
      init_val = '%s(%s)/%s(%s%s)' % (
          stencil.input.type, init_val, stencil.input.type,
          '%d+' % stencil.input.chan if stencil.input.chan>1 else '',
          '+'.join('dims[%d]' % d for d in range(stencil.dim)))
    println('%s_img[%s] = %s;' % (
        name, '+'.join('%c*%s.stride[%d]' % (util.COORDS_IN_ORIG[d], name, d)
                       for d in range(input_dim)), init_val))
    for d in range(0, input_dim):
      un_scope()
    println()

  for param in stencil.param_stmts:
    println('#pragma omp parallel for', 0)
    for dim, size in enumerate(param.size):
      println('for(int32_t {var} = 0; {var}<{}; ++{var})'.format(
          size, var=util.COORDS_IN_ORIG[dim]))
      do_scope()
    println('%s_img%s = %s;' % (
        param.name,
        sum(('[%c]' % (util.COORDS_IN_ORIG[d])
             for d in range(len(param.size))), ''),
        '+'.join(util.COORDS_IN_ORIG[0:len(param.size)])))
    for d in param.size:
      un_scope()
    println()

  println('%s(&%s, &%s, %sxclbin);' % (
      stencil.app_name, *stencil.input_names, *stencil.output_names,
      ''.join('&%s, ' % param.name for param in stencil.param_stmts)))
  println()

  println('int error_count = 0;')
  println()

  for tensor in stencil.chronological_tensors:
    if tensor.is_input():
      continue
    logger.debug('emit code to produce %s_img' % tensor.name)
    println('// produce %s, can be parallelized with -fopenmp' % tensor.name)
    println('#pragma omp parallel for', 0)
    stencil_window = core.get_overall_stencil_window(
        next(iter(tensor.parents.values())), tensor)
    stencil_window = core.get_overall_stencil_window(
        stencil.tensors[next(iter(stencil.input_names))], tensor)
    stencil_dim = core.get_stencil_dim(stencil_window)
    output_idx = core.get_stencil_window_offset(stencil_window)
    for d in range(0, stencil.dim):
      dim = stencil.dim-d-1
      println('for(int32_t {var} = {}; {var}<dims[{}]-{}; ++{var})'.format(
          output_idx[dim], dim, stencil_dim[dim]-output_idx[dim]-1,
          var=util.COORDS_IN_ORIG[dim]))
      do_scope()
    def mutate_load_for_host(obj, args):
      if isinstance(obj, grammar.Ref):
        if obj.name in stencil.param_names:
          return '%s_img%s' % (obj.name, ''.join('[%d]' % _ for _ in obj.idx))
        return '%s_img[%s]' % (obj.name, '+'.join('(%c%+d)*%s.stride[%d]' % (
                util.COORDS_IN_ORIG[d], obj.idx[d] - tensor.st_ref.idx[d],
                obj.name, d) for d in range(stencil.dim)))
      return obj
    def mutate_store_for_host(obj, args):
      if isinstance(obj, grammar.Ref):
        if obj.name in stencil.output_names:
          return'%s result_%s' % (obj.c_type, obj.name)
        return '%s_img[%s]' % (obj.name, '+'.join('%c*%s.stride[%d]' % (
            util.COORDS_IN_ORIG[d], obj.name, d) for d in range(stencil.dim)))
      return obj
    println('// {} = {}'.format(tensor.st_ref, tensor.expr))
    println('{} = {};'.format(tensor.st_ref.visit(mutate_store_for_host),
                              tensor.expr.visit(mutate_load_for_host)))
    if tensor.is_output():
      run_result = '%s_img[%s]' % (tensor.name, '+'.join(
          '%c*%s.stride[%d]' % (util.COORDS_IN_ORIG[d], tensor.name, d)
          for d in range(stencil.dim)))
      println('%s val_fpga = %s;' % (tensor.c_type, run_result))
      println('%s val_cpu = result_%s;' % (tensor.c_type, tensor.name))
      if util.is_float(tensor.soda_type):
        println('double threshold = 0.00001;')
        println('if(nullptr!=getenv("THRESHOLD"))')
        do_scope()
        println('threshold = atof(getenv("THRESHOLD"));')
        un_scope()
        println('threshold *= threshold;')
        println('if(double(val_fpga-val_cpu)*double(val_fpga-val_cpu)/(double'
                '(val_cpu)*double(val_cpu)) > threshold)')
        do_scope()
        params = (', '.join(['%d']*stencil.dim),
                  ', '.join(util.COORDS_IN_ORIG[:stencil.dim]))
        println('fprintf(*error_report, "%%lf != %%lf @(%s)\\n", double'
                '(val_fpga), double(val_cpu), %s);' % params)
      else:
        println('if(val_fpga!=val_cpu)')
        do_scope()
        params = (', '.join(['%d']*stencil.dim),
                  ', '.join(util.COORDS_IN_ORIG[:stencil.dim]))
        println('fprintf(*error_report, "%%ld != %%ld @(%s)\\n", int64_t'
                '(val_fpga), int64_t(val_cpu), %s);' % params)
      println('++error_count;')
      un_scope()
    for d in range(0, stencil.dim):
      un_scope()
    println()

    if False and s.preserve_border_from():
      println('// handle borders for iterative stencil')
      println('#pragma omp parallel for', 0)
      bb = s.preserve_border_from()
      stencil_window = util.get_overall_stencil_window(bb, s.output)
      stencil_dim = util.get_stencil_dim(stencil_window)
      output_idx = util.get_stencil_window_offset(stencil_window)
      for d in range(0, stencil.dim):
        dim = stencil.dim-d-1
        println('for(int32_t %c = 0; %c<dims[%d]; ++%c)' % (util.COORDS_IN_ORIG[dim], util.COORDS_IN_ORIG[dim], dim, util.COORDS_IN_ORIG[dim]))
        do_scope()
      println('if(!(%s))' % ' && '.join('%c>=%d && %c<dims[%d]-%d' % (util.COORDS_IN_ORIG[d], output_idx[d], util.COORDS_IN_ORIG[d], d, stencil_dim[d]-output_idx[d]-1) for d in range(stencil.dim)))
      do_scope()
      GroudTruth = lambda c: '%s_img[%s+%d*%s.stride[%d]]' % (bb.name, '+'.join(['%c*%s.stride[%d]' % (util.COORDS_IN_ORIG[d], bb.name, d) for d in range(stencil.dim)]), c, bb.name, stencil.dim)
      for e in s.expr:
        println('%s = %s;' % (StorePrinter(e), GroudTruth(e.chan)))
      if len(s.output.children)==0:
        for c in range(stencil.output.chan):
          run_result = '%s_img[%s+%d*%s.stride[%d]]' % (stencil.output.name, '+'.join(['%c*%s.stride[%d]' % (util.COORDS_IN_ORIG[d], stencil.output.name, d) for d in range(stencil.dim)]), c, stencil.output.name, stencil.dim)
          println('%s val_fpga = %s;' % (stencil.output.type, run_result))
          println('%s val_cpu = result_chan_%d;' % (stencil.output.type, c))
          if util.is_float(stencil.output.type):
            println('double threshold = 0.00001;')
            println('if(nullptr!=getenv("THRESHOLD"))')
            do_scope()
            println('threshold = atof(getenv("THRESHOLD"));')
            un_scope()
            println('threshold *= threshold;')
            println('if(double(val_fpga-val_cpu)*double(val_fpga-val_cpu)/(double(val_cpu)*double(val_cpu)) > threshold)')
            do_scope()
            params = (c, ', '.join(['%d']*stencil.dim), ', '.join(util.COORDS_IN_ORIG[:stencil.dim]))
            println('fprintf(*error_report, "%%lf != %%lf @[%d](%s)\\n", double(val_fpga), double(val_cpu), %s);' % params)
          else:
            println('if(val_fpga!=val_cpu)')
            do_scope()
            params = (c, ', '.join(['%d']*stencil.dim), ', '.join(util.COORDS_IN_ORIG[:stencil.dim]))
            println('fprintf(*error_report, "%%ld != %%ld @[%d](%s)\\n", int64_t(val_fpga), int64_t(val_cpu), %s);' % params)
          println('++error_count;')
          un_scope()
      un_scope()
      for d in range(0, stencil.dim):
        un_scope()
      println()

  println('if(error_count==0)')
  do_scope()
  println('fprintf(*error_report, "INFO: PASS!\\n");')
  un_scope()
  println('else')
  do_scope()
  println('fprintf(*error_report, "INFO: FAIL!\\n");')
  un_scope()
  println()

  for var in (stencil.input_names + stencil.local_names +
              stencil.output_names + stencil.param_names):
    println('delete[] %s_img;' % var)
  println()

  println('return error_count;')
  un_scope()

def print_code(stencil, host_file):
  logger.info('generate host source code as %s' % host_file.name)
  printer = util.Printer(host_file)
  println = printer.println
  do_scope = printer.do_scope
  un_scope = printer.un_scope
  print_define = lambda key, value: util.print_define(printer, key, value)

  print_header(printer)
  println('#include"%s.h"' % stencil.app_name)
  println()

  print_define('BURST_WIDTH', stencil.burst_width)
  print_define('PIXEL_WIDTH_I', sum(map(util.get_width_in_bits,
                                        stencil.input_types)))
  print_define('PIXEL_WIDTH_O', sum(map(util.get_width_in_bits,
                                        stencil.output_types)))

  if stencil.preserve_border:
    overall_stencil_window = core.get_overall_stencil_window(
      stencil.output.parent.preserve_border_from(), stencil.output)
  else:
    overall_stencil_window = core.get_overall_stencil_window(
      stencil.tensors[stencil.input_names[0]],
      stencil.tensors[stencil.output_names[0]])

  overall_stencil_distance = core.get_stencil_distance(overall_stencil_window,
    stencil.tile_size)

  for i, dim in enumerate(core.get_stencil_dim(overall_stencil_window)):
    print_define('STENCIL_DIM_%d' % i, dim)
  stencil_offset = overall_stencil_distance - util.serialize(
    core.get_stencil_window_offset(overall_stencil_window), stencil.tile_size)
  if stencil.preserve_border:
    stencil_offset *= stencil.iterate

  overall_stencil_distance = max(overall_stencil_distance, stencil_offset)

  print_define('STENCIL_OFFSET', stencil_offset)
  print_define('STENCIL_DISTANCE', overall_stencil_distance)
  println()

  print_load_xclbin2(printer)
  print_halide_rewrite_buffer(printer)
  print_halide_error_codes(printer)
  print_halide_error_report(printer)
  print_wrapped(printer, stencil)
  print_entrance(printer, stencil)
  print_test(printer, stencil)
