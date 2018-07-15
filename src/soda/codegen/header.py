import logging

from soda import core

logger = logging.getLogger('__main__').getChild(__name__)

def print_code(stencil, header_file):
  logger.info('generate host header code as %s' % header_file.name)
  p = core.Printer(header_file)
  p.println('#ifndef HALIDE_%s_H_' % stencil.app_name.upper())
  p.println('#define HALIDE_%s_H_' % stencil.app_name.upper())
  p.println()

  p.println('#ifndef HALIDE_ATTRIBUTE_ALIGN')
  p.do_indent()
  p.println('#ifdef _MSC_VER')
  p.do_indent()
  p.println('#define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))')
  p.un_indent()
  p.println('#else')
  p.do_indent()
  p.println('#define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))')
  p.un_indent()
  p.println('#endif')
  p.un_indent()
  p.println('#endif//HALIDE_ATTRIBUTE_ALIGN')
  p.println()

  p.println('#ifndef BUFFER_T_DEFINED')
  p.println('#define BUFFER_T_DEFINED')
  p.println('#include<stdbool.h>')
  p.println('#include<stdint.h>')
  p.println('typedef struct buffer_t {')
  p.do_indent()
  p.println('uint64_t dev;')
  p.println('uint8_t* host;')
  p.println('int32_t extent[4];')
  p.println('int32_t stride[4];')
  p.println('int32_t min[4];')
  p.println('int32_t elem_size;')
  p.println('HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;')
  p.println('HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;')
  p.println('HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];')
  p.un_indent()
  p.println('} buffer_t;')
  p.println('#endif//BUFFER_T_DEFINED')
  p.println()

  p.println('#ifndef HALIDE_FUNCTION_ATTRS')
  p.println('#define HALIDE_FUNCTION_ATTRS')
  p.println('#endif//HALIDE_FUNCTION_ATTRS')
  p.println()

  tensors = [stencil.input.name, stencil.output.name] + [param.name for param in stencil.extra_params.values()]
  p.println('int %s(%sconst char* xclbin) HALIDE_FUNCTION_ATTRS;' % (stencil.app_name, ''.join([('buffer_t *var_%s_buffer, ') % x for x in tensors])))
  p.println()

  p.println('#endif//HALIDE_%s_H_' % stencil.app_name.upper())
  p.println()

