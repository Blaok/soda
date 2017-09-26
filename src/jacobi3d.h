#ifndef HALIDE_JACOBI2D_H_
#define HALIDE_JACOBI2D_H_

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
#endif//BUFFER_T_DEFINED

#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif//HALIDE_FUNCTION_ATTRS


int jacobi3d(buffer_t *var_p0_buffer, buffer_t *var_jacobi3d_y_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS;

#endif//HALIDE_JACOBI2D_H_

