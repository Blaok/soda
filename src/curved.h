#ifndef HALIDE_CURVED_H_
#define HALIDE_CURVED_H_

#ifndef HALIDE_ATTRIBUTE_ALIGN
  #ifdef _MSC_VER
    #define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
  #else
    #define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
  #endif
#endif//HALIDE_ATTRIBUTE_ALIGN

#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdbool.h>
#include <stdint.h>
typedef struct buffer_t {
    uint64_t dev;
    uint8_t* host;
    int32_t extent[4];
    int32_t stride[4];
    int32_t min[4];
    int32_t elem_size;
    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;
#endif

#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif//HALIDE_FUNCTION_ATTRS

int curved(float _color_temp, float _gamma, float _contrast, int32_t _blackLevel, int32_t _whiteLevel, buffer_t *_input_buffer, buffer_t *_m3200_buffer, buffer_t *_m7000_buffer, buffer_t *_processed_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS;

#endif//HALIDE_CURVED_H_
