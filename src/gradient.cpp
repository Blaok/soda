#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#include "gradient.h"
#include "gradient_params.h"

int load_file_to_memory(const char *filename, char **result)
{ 
    size_t size = 0;
    FILE *f = fopen(filename, "rb");
    if (NULL == f)
    {
        *result = NULL;
        return -1;
    } 
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    *result = (char *)malloc(size+1);
    if (size != fread(*result, sizeof(char), size, f))
    {
        free(*result);
        return -2;
    } 
    fclose(f);
    (*result)[size] = 0;
    return size;
}

static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,
                                  int32_t min0, int32_t extent0, int32_t stride0,
                                  int32_t min1, int32_t extent1, int32_t stride1,
                                  int32_t min2, int32_t extent2, int32_t stride2,
                                  int32_t min3, int32_t extent3, int32_t stride3) {
    b->min[0] = min0;
    b->min[1] = min1;
    b->min[2] = min2;
    b->min[3] = min3;
    b->extent[0] = extent0;
    b->extent[1] = extent1;
    b->extent[2] = extent2;
    b->extent[3] = extent3;
    b->stride[0] = stride0;
    b->stride[1] = stride1;
    b->stride[2] = stride2;
    b->stride[3] = stride3;
    return true;
}

int halide_error_code_success = 0;
int halide_error_code_generic_error = -1;
int halide_error_code_explicit_bounds_too_small = -2;
int halide_error_code_bad_elem_size = -3;
int halide_error_code_access_out_of_bounds = -4;
int halide_error_code_buffer_allocation_too_large = -5;
int halide_error_code_buffer_extents_too_large = -6;
int halide_error_code_constraints_make_required_region_smaller = -7;
int halide_error_code_constraint_violated = -8;
int halide_error_code_param_too_small = -9;
int halide_error_code_param_too_large = -10;
int halide_error_code_out_of_memory = -11;
int halide_error_code_buffer_argument_is_null = -12;
int halide_error_code_debug_to_file_failed = -13;
int halide_error_code_copy_to_host_failed = -14;
int halide_error_code_copy_to_device_failed = -15;
int halide_error_code_device_malloc_failed = -16;
int halide_error_code_device_sync_failed = -17;
int halide_error_code_device_free_failed = -18;
int halide_error_code_no_device_interface = -19;
int halide_error_code_matlab_init_failed = -20;
int halide_error_code_matlab_bad_param_type = -21;
int halide_error_code_internal_error = -22;
int halide_error_code_device_run_failed = -23;
int halide_error_code_unaligned_host_ptr = -24;
int halide_error_code_bad_fold = -25;
int halide_error_code_fold_factor_too_small = -26;

FILE* const* error_report = &stderr;

int halide_error_bad_elem_size(void *user_context, const char *func_name,
                               const char *type_name, int elem_size_given, int correct_elem_size) {
    fprintf(*error_report, "%s has type %s but elem_size of the buffer passed in is %d instead of %d",
            func_name, type_name, elem_size_given, correct_elem_size);
    return halide_error_code_bad_elem_size;
}
int halide_error_constraint_violated(void *user_context, const char *var, int val,
                                     const char *constrained_var, int constrained_val) {
    fprintf(*error_report, "Constraint violated: %s (%d) == %s (%d)",
            var, val, constrained_var, constrained_val);
    return halide_error_code_constraint_violated;
}
int halide_error_buffer_allocation_too_large(void *user_context, const char *buffer_name, uint64_t allocation_size, uint64_t max_size) {
    fprintf(*error_report, "Total allocation for buffer %s is %lu, which exceeds the maximum size of %lu",
            buffer_name, allocation_size, max_size);
    return halide_error_code_buffer_allocation_too_large;
}
int halide_error_buffer_extents_too_large(void *user_context, const char *buffer_name, int64_t actual_size, int64_t max_size) {
    fprintf(*error_report, "Product of extents for buffer %s is %ld, which exceeds the maximum size of %ld",
            buffer_name, actual_size, max_size);
    return halide_error_code_buffer_extents_too_large;
}
int halide_error_access_out_of_bounds(void *user_context, const char *func_name, int dimension, int min_touched, int max_touched, int min_valid, int max_valid) {
    if (min_touched < min_valid) {
        fprintf(*error_report, "%s is accessed at %d, which is before the min (%d) in dimension %d", func_name, min_touched, min_valid, dimension);
    } else if (max_touched > max_valid) {
        fprintf(*error_report, "%s is acccessed at %d, which is beyond the max (%d) in dimension %d", func_name, max_touched, max_valid, dimension);
    }
    return halide_error_code_access_out_of_bounds;
}

static int gradient_wrapped(buffer_t *var_input_buffer, buffer_t *var_output_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS {
    float *var_input = (float *)(var_input_buffer->host);
    (void)var_input;
    const bool var_input_host_and_dev_are_null = (var_input_buffer->host == NULL) && (var_input_buffer->dev == 0);
    (void)var_input_host_and_dev_are_null;
    int32_t var_input_min_0 = var_input_buffer->min[0];
    (void)var_input_min_0;
    int32_t var_input_min_1 = var_input_buffer->min[1];
    (void)var_input_min_1;
    int32_t var_input_min_2 = var_input_buffer->min[2];
    (void)var_input_min_2;
    int32_t var_input_min_3 = var_input_buffer->min[3];
    (void)var_input_min_3;
    int32_t input_size_dim_0 = var_input_buffer->extent[0];
    (void)input_size_dim_0;
    int32_t input_size_dim_1 = var_input_buffer->extent[1];
    (void)input_size_dim_1;
    int32_t var_input_extent_2 = var_input_buffer->extent[2];
    (void)var_input_extent_2;
    int32_t var_input_extent_3 = var_input_buffer->extent[3];
    (void)var_input_extent_3;
    int32_t var_input_stride_0 = var_input_buffer->stride[0];
    (void)var_input_stride_0;
    int32_t var_input_stride_1 = var_input_buffer->stride[1];
    (void)var_input_stride_1;
    int32_t var_input_stride_2 = var_input_buffer->stride[2];
    (void)var_input_stride_2;
    int32_t var_input_stride_3 = var_input_buffer->stride[3];
    (void)var_input_stride_3;
    int32_t var_input_elem_size = var_input_buffer->elem_size;
    (void)var_input_elem_size;
    float *var_output = (float *)(var_output_buffer->host);
    (void)var_output;
    const bool var_output_host_and_dev_are_null = (var_output_buffer->host == NULL) && (var_output_buffer->dev == 0);
    (void)var_output_host_and_dev_are_null;
    int32_t var_output_min_0 = var_output_buffer->min[0];
    (void)var_output_min_0;
    int32_t var_output_min_1 = var_output_buffer->min[1];
    (void)var_output_min_1;
    int32_t var_output_min_2 = var_output_buffer->min[2];
    (void)var_output_min_2;
    int32_t var_output_min_3 = var_output_buffer->min[3];
    (void)var_output_min_3;
    int32_t output_size_dim_0 = var_output_buffer->extent[0];
    (void)output_size_dim_0;
    int32_t output_size_dim_1 = var_output_buffer->extent[1];
    (void)output_size_dim_1;
    int32_t var_output_extent_2 = var_output_buffer->extent[2];
    (void)var_output_extent_2;
    int32_t var_output_extent_3 = var_output_buffer->extent[3];
    (void)var_output_extent_3;
    int32_t var_output_stride_0 = var_output_buffer->stride[0];
    (void)var_output_stride_0;
    int32_t var_output_stride_1 = var_output_buffer->stride[1];
    (void)var_output_stride_1;
    int32_t var_output_stride_2 = var_output_buffer->stride[2];
    (void)var_output_stride_2;
    int32_t var_output_stride_3 = var_output_buffer->stride[3];
    (void)var_output_stride_3;
    int32_t var_output_elem_size = var_output_buffer->elem_size;
    (void)var_output_elem_size;
    if (var_output_host_and_dev_are_null)
    {
        bool assign_0 = halide_rewrite_buffer(var_output_buffer, 4, var_output_min_0, output_size_dim_0, 1, var_output_min_1, output_size_dim_1, output_size_dim_0, 0, 0, 0, 0, 0, 0);
        (void)assign_0;
    } // if var_output_host_and_dev_are_null
    if (var_input_host_and_dev_are_null)
    {
        int32_t assign_1 = output_size_dim_0 + 2;
        int32_t assign_2 = output_size_dim_1 + 2;
        bool assign_3 = halide_rewrite_buffer(var_input_buffer, 4, var_output_min_0, assign_1, 1, var_output_min_1, assign_2, assign_1, 0, 0, 0, 0, 0, 0);
        (void)assign_3;
    } // if var_input_host_and_dev_are_null
    bool assign_4 = var_output_host_and_dev_are_null || var_input_host_and_dev_are_null;
    bool assign_5 = !(assign_4);
    if (assign_5)
    {
        bool assign_6 = var_output_elem_size == 4;
        if (!assign_6)         {
            int32_t assign_7 = halide_error_bad_elem_size(NULL, "Output buffer output", "uint16", var_output_elem_size, 4);
            return assign_7;
        }
        bool assign_8 = var_input_elem_size == 4;
        if (!assign_8)         {
            int32_t assign_9 = halide_error_bad_elem_size(NULL, "Input buffer input", "uint16", var_input_elem_size, 4);
            return assign_9;
        }
        bool assign_10 = var_input_min_0 <= var_output_min_0;
        int32_t assign_11 = var_output_min_0 + output_size_dim_0;
        int32_t assign_12 = assign_11 - input_size_dim_0;
        int32_t assign_13 = assign_12 + 2;
        bool assign_14 = assign_13 <= var_input_min_0;
        bool assign_15 = assign_10 && assign_14;
        if (!assign_15)         {
            int32_t assign_16 = var_output_min_0 + output_size_dim_0;
            int32_t assign_17 = assign_16 + 1;
            int32_t assign_18 = var_input_min_0 + input_size_dim_0;
            int32_t assign_19 = assign_18 + -1;
            int32_t assign_20 = halide_error_access_out_of_bounds(NULL, "Input buffer input", 0, var_output_min_0, assign_17, var_input_min_0, assign_19);
            return assign_20;
        }
        bool assign_21 = var_input_min_1 <= var_output_min_1;
        int32_t assign_22 = var_output_min_1 + output_size_dim_1;
        int32_t assign_23 = assign_22 - input_size_dim_1;
        int32_t assign_24 = assign_23 + 2;
        bool assign_25 = assign_24 <= var_input_min_1;
        bool assign_26 = assign_21 && assign_25;
        if (!assign_26)         {
            int32_t assign_27 = var_output_min_1 + output_size_dim_1;
            int32_t assign_28 = assign_27 + 1;
            int32_t assign_29 = var_input_min_1 + input_size_dim_1;
            int32_t assign_30 = assign_29 + -1;
            int32_t assign_31 = halide_error_access_out_of_bounds(NULL, "Input buffer input", 1, var_output_min_1, assign_28, var_input_min_1, assign_30);
            return assign_31;
        }
        bool assign_32 = var_output_stride_0 == 1;
        if (!assign_32)         {
            int32_t assign_33 = halide_error_constraint_violated(NULL, "output.stride.0", var_output_stride_0, "1", 1);
            return assign_33;
        }
        bool assign_34 = var_input_stride_0 == 1;
        if (!assign_34)         {
            int32_t assign_35 = halide_error_constraint_violated(NULL, "input.stride.0", var_input_stride_0, "1", 1);
            return assign_35;
        }
        int64_t assign_36 = (int64_t)(output_size_dim_1);
        int64_t assign_37 = (int64_t)(output_size_dim_0);
        int64_t assign_38 = assign_36 * assign_37;
        int64_t assign_39 = (int64_t)(input_size_dim_1);
        int64_t assign_40 = (int64_t)(input_size_dim_0);
        int64_t assign_41 = assign_39 * assign_40;
        int64_t assign_42 = (int64_t)(2147483647);
        bool assign_43 = assign_37 <= assign_42;
        if (!assign_43)         {
            int64_t assign_44 = (int64_t)(output_size_dim_0);
            int64_t assign_45 = (int64_t)(2147483647);
            int32_t assign_46 = halide_error_buffer_allocation_too_large(NULL, "output", assign_44, assign_45);
            return assign_46;
        }
        int64_t assign_47 = (int64_t)(output_size_dim_1);
        int64_t assign_48 = (int64_t)(var_output_stride_1);
        int64_t assign_49 = assign_47 * assign_48;
        int64_t assign_50 = (int64_t)(2147483647);
        bool assign_51 = assign_49 <= assign_50;
        if (!assign_51)         {
            int64_t assign_52 = (int64_t)(output_size_dim_1);
            int64_t assign_53 = (int64_t)(var_output_stride_1);
            int64_t assign_54 = assign_52 * assign_53;
            int64_t assign_55 = (int64_t)(2147483647);
            int32_t assign_56 = halide_error_buffer_allocation_too_large(NULL, "output", assign_54, assign_55);
            return assign_56;
        }
        int64_t assign_57 = (int64_t)(2147483647);
        bool assign_58 = assign_38 <= assign_57;
        if (!assign_58)         {
            int64_t assign_59 = (int64_t)(2147483647);
            int32_t assign_60 = halide_error_buffer_extents_too_large(NULL, "output", assign_38, assign_59);
            return assign_60;
        }
        int64_t assign_61 = (int64_t)(input_size_dim_0);
        int64_t assign_62 = (int64_t)(2147483647);
        bool assign_63 = assign_61 <= assign_62;
        if (!assign_63)         {
            int64_t assign_64 = (int64_t)(input_size_dim_0);
            int64_t assign_65 = (int64_t)(2147483647);
            int32_t assign_66 = halide_error_buffer_allocation_too_large(NULL, "input", assign_64, assign_65);
            return assign_66;
        }
        int64_t assign_67 = (int64_t)(input_size_dim_1);
        int64_t assign_68 = (int64_t)(var_input_stride_1);
        int64_t assign_69 = assign_67 * assign_68;
        int64_t assign_70 = (int64_t)(2147483647);
        bool assign_71 = assign_69 <= assign_70;
        if (!assign_71)         {
            int64_t assign_72 = (int64_t)(input_size_dim_1);
            int64_t assign_73 = (int64_t)(var_input_stride_1);
            int64_t assign_74 = assign_72 * assign_73;
            int64_t assign_75 = (int64_t)(2147483647);
            int32_t assign_76 = halide_error_buffer_allocation_too_large(NULL, "input", assign_74, assign_75);
            return assign_76;
        }
        int64_t assign_77 = (int64_t)(2147483647);
        bool assign_78 = assign_41 <= assign_77;
        if (!assign_78)         {
            int64_t assign_79 = (int64_t)(2147483647);
            int32_t assign_80 = halide_error_buffer_extents_too_large(NULL, "input", assign_41, assign_79);
            return assign_80;
        }

        // allocate buffer for tiled input/output
        int32_t tile_num_dim_0 = (output_size_dim_0+TILE_SIZE_DIM_0-STENCIL_DIM_0)/(TILE_SIZE_DIM_0-STENCIL_DIM_0+1);

        // align each linearized tile to multiples of BURST_WIDTH
        int64_t tile_pixel_num = TILE_SIZE_DIM_0*input_size_dim_1;
        int64_t tile_burst_num = (tile_pixel_num-1)/BURST_LENGTH+1;
        int64_t tile_size_linearized_i = (CHANNEL_NUM_I == 1 ? tile_pixel_num : tile_burst_num*BURST_LENGTH*CHANNEL_NUM_I);
        int64_t tile_size_linearized_o = (CHANNEL_NUM_O == 1 ? tile_pixel_num : tile_burst_num*BURST_LENGTH*CHANNEL_NUM_O);
        int64_t extra_space_i = (CHANNEL_NUM_I == 1 ? tile_burst_num*BURST_LENGTH-tile_pixel_num : 0);
        int64_t extra_space_o = (CHANNEL_NUM_O == 1 ? tile_burst_num*BURST_LENGTH-tile_pixel_num : 0);

        float* var_input_buf = new float[tile_num_dim_0*tile_size_linearized_i];
        float* var_output_buf = new float[tile_num_dim_0*tile_size_linearized_o];

        // tiling
        for(int32_t tile_index_dim_0 = 0; tile_index_dim_0 < tile_num_dim_0; ++tile_index_dim_0)
        {
            int32_t actual_tile_size_dim_0 = (tile_index_dim_0==tile_num_dim_0-1) ? input_size_dim_0-(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)*tile_index_dim_0 : TILE_SIZE_DIM_0;
            for(int32_t c = 0; c < CHANNEL_NUM_I; ++c)
            {
                for(int32_t j = 0; j < input_size_dim_1; ++j)
                {
                    for(int32_t i = 0; i < actual_tile_size_dim_0; ++i)
                    {
                        // (x, y, z, w) is coordinates in tiled image
                        // (p, q, r, s) is coordinates in original image
                        // (i, j, k, l) is coordinates in a tile
                        int32_t burst_index = (j*TILE_SIZE_DIM_0+i)/BURST_LENGTH;
                        int32_t burst_residue = (j*TILE_SIZE_DIM_0+i)%BURST_LENGTH;
                        int32_t p = tile_index_dim_0*(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)+i;
                        int32_t q = j;
                        int64_t tiled_offset = (tile_index_dim_0*tile_pixel_num+burst_index*BURST_LENGTH)*CHANNEL_NUM_I+c*BURST_LENGTH+burst_residue;
                        int64_t original_offset = (q*var_input_stride_1+p)*CHANNEL_NUM_I+c;
                        var_input_buf[tiled_offset] = var_input[original_offset];
                    }
                }
            }
        }

        // prepare for opencl
#if defined(SDA_PLATFORM) && !defined(TARGET_DEVICE)
#define STR_VALUE(arg)      #arg
#define GET_STRING(name) STR_VALUE(name)
#define TARGET_DEVICE GET_STRING(SDA_PLATFORM)
#endif
        const char *target_device_name = TARGET_DEVICE;
        int err;                            // error code returned from api calls

        cl_platform_id platforms[16];       // platform id
        cl_platform_id platform_id;         // platform id
        cl_uint platform_count;
        cl_device_id device_id;             // compute device id 
        cl_context context;                 // compute context
        cl_command_queue commands;          // compute command queue
        cl_program program;                 // compute program
        cl_kernel kernel;                   // compute kernel
       
        char cl_platform_vendor[1001];
       
        cl_mem var_input_cl;                   // device memory used for the input array
        cl_mem var_output_cl;               // device memory used for the output array
   
        err = clGetPlatformIDs(16, platforms, &platform_count);
        if (err != CL_SUCCESS)
            {
                printf("FATAL: Failed to find an OpenCL platform\n");
                exit(EXIT_FAILURE);
            }
        printf("INFO: Found %d platforms\n", platform_count);

        int platform_found = 0;
        for (unsigned iplat = 0; iplat<platform_count; iplat++) {
            err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
            if (err != CL_SUCCESS) {
                printf("FATAL: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed\n");
                exit(EXIT_FAILURE);
            }
            if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
                printf("INFO: Selected platform %d from %s\n", iplat, cl_platform_vendor);
                platform_id = platforms[iplat];
                platform_found = 1;
            }
        }
        if (!platform_found) {
            printf("FATAL: Platform Xilinx not found\n");
            exit(EXIT_FAILURE);
        }
      
        cl_device_id devices[16];
        cl_uint device_count;
        unsigned int device_found = 0;
        char cl_device_name[1001];
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                             16, devices, &device_count);
        if (err != CL_SUCCESS) {
            printf("FATAL: Failed to create a device group\n");
            exit(EXIT_FAILURE);
        }

        for (unsigned int i=0; i<device_count; i++) {
            err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 1024, cl_device_name, 0);
            if (err != CL_SUCCESS) {
                printf("FATAL: Failed to get device name for device %d\n", i);
                exit(EXIT_FAILURE);
            }
            //printf("CL_DEVICE_NAME %s\n", cl_device_name);
            if(strcmp(cl_device_name, target_device_name) == 0) {
                device_id = devices[i];
                device_found = 1;
                printf("INFO: Selected %s as the target device\n", cl_device_name);
            }
        }
        
        if (!device_found) {
            printf("FATAL: Target device %s not found\n", target_device_name);
            exit(EXIT_FAILURE);
        }


        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                             1, &device_id, NULL);
        if (err != CL_SUCCESS) {
            printf("FATAL: Failed to create a device group\n");
            exit(EXIT_FAILURE);
        }
      
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if (!context) {
            printf("FATAL: Failed to create a compute context\n");
            exit(EXIT_FAILURE);
        }

        commands = clCreateCommandQueue(context, device_id, 0, &err);
        if (!commands) {
            printf("FATAL: Failed to create a command commands %i\n",err);
            exit(EXIT_FAILURE);
        }

        int status;

        unsigned char *kernelbinary;
        printf("INFO: Loading %s\n", xclbin);
        int n_i = load_file_to_memory(xclbin, (char **) &kernelbinary);
        if (n_i < 0) {
            printf("FATAL: Failed to load kernel from xclbin: %s\n", xclbin);
            exit(EXIT_FAILURE);
        }
        size_t n = n_i;
        program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                           (const unsigned char **) &kernelbinary, &status, &err);
        if ((!program) || (err!=CL_SUCCESS)) {
            printf("FATAL: Failed to create compute program from binary %d\n", err);
            exit(EXIT_FAILURE);
        }
        free(kernelbinary);

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t len;
            char buffer[2048];
            printf("FATAL: Failed to build program executable\n");
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            exit(EXIT_FAILURE);
        }

        kernel = clCreateKernel(program, "gradient_kernel", &err);
        if (!kernel || err != CL_SUCCESS) {
            printf("FATAL: Failed to create compute kernel %d\n", err);
            exit(EXIT_FAILURE);
        }

        var_input_cl  = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * (tile_num_dim_0*tile_size_linearized_i+extra_space_i), NULL, NULL);
        var_output_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * (tile_num_dim_0*tile_size_linearized_o+extra_space_o), NULL, NULL);
        if (!var_input_cl || !var_output_cl)
        {
            printf("FATAL: Failed to allocate device memory\n");
            exit(EXIT_FAILURE);
        }
        
        timespec write_begin, write_end;
        cl_event writeevent;
        clock_gettime(CLOCK_REALTIME, &write_begin);
        err = clEnqueueWriteBuffer(commands, var_input_cl, CL_FALSE, 0, sizeof(float) * tile_num_dim_0*tile_size_linearized_i, var_input_buf, 0, NULL, &writeevent);
        if (err != CL_SUCCESS)
        {
            printf("FATAL: Failed to write to input !\n");
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &writeevent);
        clock_gettime(CLOCK_REALTIME, &write_end);

        err = 0;
        printf("HOST: tile_num_dim_0 = %d, TILE_SIZE_DIM_0 = %d\n", tile_num_dim_0, TILE_SIZE_DIM_0);
        printf("HOST: var_input_extent_0 = %d, var_input_extent_1 = %d\n", input_size_dim_0, input_size_dim_1);
        printf("HOST: var_input_min_0 = %d, var_input_min_1 = %d\n", var_input_min_0, var_input_min_1);
        printf("HOST: output_size_dim_0 = %d, output_size_dim_1 = %d\n", output_size_dim_0, output_size_dim_1);
        printf("HOST: var_output_min_0 = %d, var_output_min_1 = %d\n", var_output_min_0, var_output_min_1);

        int64_t extra_space_i_coalesed = extra_space_i/(BURST_WIDTH/PIXEL_WIDTH_I);
        int64_t extra_space_o_coalesed = extra_space_o/(BURST_WIDTH/PIXEL_WIDTH_O);
        int32_t total_burst_num = tile_num_dim_0*tile_burst_num;

        err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &var_output_cl);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &var_input_cl);
        err |= clSetKernelArg(kernel, 2, sizeof(tile_num_dim_0), &tile_num_dim_0);
        err |= clSetKernelArg(kernel, 3, sizeof(input_size_dim_1), &input_size_dim_1);
        err |= clSetKernelArg(kernel, 4, sizeof(tile_burst_num), &tile_burst_num);
        err |= clSetKernelArg(kernel, 5, sizeof(extra_space_i_coalesed), &extra_space_i_coalesed);
        err |= clSetKernelArg(kernel, 6, sizeof(extra_space_o_coalesed), &extra_space_o_coalesed);
        err |= clSetKernelArg(kernel, 7, sizeof(total_burst_num), &total_burst_num);
        if (err != CL_SUCCESS)
        {
            printf("FATAL: Failed to set kernel arguments\n");
            printf("ERROR code: %d\n", err);
            exit(EXIT_FAILURE);
        }

        cl_event execute;
        if(NULL==getenv("XCL_EMULATION_MODE")) {
            printf("INFO: FPGA warm up\n");
            for(int i = 0; i<3; ++i)
            {
                err = clEnqueueTask(commands, kernel, 0, NULL, &execute);
            }
            clWaitForEvents(1, &execute);
        }
        else
        {
            printf("INFO: Emulation mode.\n");
        }

        // Execute the kernel over the entire range of our 1d input data set
        // using the maximum number of work group items for this device
        //
        timespec execute_begin, execute_end;
        clock_gettime(CLOCK_REALTIME, &execute_begin);
        err = clEnqueueTask(commands, kernel, 0, NULL, &execute);
        if (err)
        {
            printf("ERROR: Failed to execute kernel %d\n", err);
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &execute);
        clock_gettime(CLOCK_REALTIME, &execute_end);

        // Read back the results from the device to verify the output
        //
        timespec read_begin, read_end;
        cl_event readevent;
        clock_gettime(CLOCK_REALTIME, &read_begin);
        err = clEnqueueReadBuffer(commands, var_output_cl, CL_FALSE, 0, sizeof(float) * tile_num_dim_0*tile_size_linearized_o, var_output_buf, 0, NULL, &readevent );
        if (err != CL_SUCCESS)
        {
            printf("ERROR: Failed to read output %d\n", err);
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &readevent);
        clock_gettime(CLOCK_REALTIME, &read_end);

        double elapsed_time = 0.;
        elapsed_time = (double(write_end.tv_sec-write_begin.tv_sec)+(write_end.tv_nsec-write_begin.tv_nsec)/1e9)*1e6;
        printf("PCIe write time:       %lf us\n", elapsed_time);
        printf("PCIe write throughput: %lf GB/s\n", input_size_dim_0*input_size_dim_1*var_input_elem_size/elapsed_time/1e3);
        elapsed_time = (double(execute_end.tv_sec-execute_begin.tv_sec)+(execute_end.tv_nsec-execute_begin.tv_nsec)/1e9)*1e6;
        printf("Kernel run time:       %lf us\n", elapsed_time);
        printf("Kernel throughput:     %lf pixel/ns\n", input_size_dim_0*input_size_dim_1/elapsed_time/1e3);
        elapsed_time = (double(read_end.tv_sec-read_begin.tv_sec)+(read_end.tv_nsec-read_begin.tv_nsec)/1e9)*1e6;
        printf("PCIe read time:        %lf us\n", elapsed_time);
        printf("PCIe read throughput:  %lf GB/s\n", output_size_dim_0*output_size_dim_1*var_output_elem_size/elapsed_time/1e3);

        clReleaseMemObject(var_input_cl);
        clReleaseMemObject(var_output_cl);
        clReleaseProgram(program);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(commands);
        clReleaseContext(context);

        for(int32_t tile_index_dim_0 = 0; tile_index_dim_0 < tile_num_dim_0; ++tile_index_dim_0)
        {
            int32_t actual_tile_size_dim_0 = (tile_index_dim_0==tile_num_dim_0-1) ? input_size_dim_0-(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)*tile_index_dim_0 : TILE_SIZE_DIM_0;
            for(int32_t c = 0; c < CHANNEL_NUM_O; ++c)
            {
                for(int32_t j = 0; j < input_size_dim_1-STENCIL_DIM_1+1; ++j)
                {
                    for(int32_t i = 0; i < actual_tile_size_dim_0-STENCIL_DIM_0+1; ++i)
                    {
                        // (x, y, z, w) is coordinates in tiled image
                        // (p, q, r, s) is coordinates in original image
                        // (i, j, k, l) is coordinates in a tile
                        int32_t burst_index = (j*TILE_SIZE_DIM_0+i+STENCIL_DISTANCE)/BURST_LENGTH;
                        int32_t burst_residue = (j*TILE_SIZE_DIM_0+i+STENCIL_DISTANCE)%BURST_LENGTH;
                        int32_t p = tile_index_dim_0*(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)+i;
                        int32_t q = j;
                        int64_t tiled_offset = (tile_index_dim_0*tile_pixel_num+burst_index*BURST_LENGTH)*CHANNEL_NUM_O+c*BURST_LENGTH+burst_residue;
                        int64_t original_offset = (q*var_output_stride_1+p)*CHANNEL_NUM_O+c;
                        //printf("p=%d, q=%d, i=%d, j=%d, tiled_offset = %d, original_offset = %d, var_output_buf[tiled_offset] = %d\n",p,q,i,j,tiled_offset, original_offset, var_output_buf[tiled_offset]);
                        var_output[original_offset] = var_output_buf[tiled_offset];
                    }
                }
            }
        }
        delete[] var_output_buf;
        delete[] var_input_buf;
    } // if assign_5
    return 0;
}

int gradient(buffer_t *var_input_buffer, buffer_t *var_output_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS {
    float *var_input = (float *)(var_input_buffer->host);
    (void)var_input;
    const bool var_input_host_and_dev_are_null = (var_input_buffer->host == NULL) && (var_input_buffer->dev == 0);
    (void)var_input_host_and_dev_are_null;
    int32_t var_input_min_0 = var_input_buffer->min[0];
    (void)var_input_min_0;
    int32_t var_input_min_1 = var_input_buffer->min[1];
    (void)var_input_min_1;
    int32_t var_input_min_2 = var_input_buffer->min[2];
    (void)var_input_min_2;
    int32_t var_input_min_3 = var_input_buffer->min[3];
    (void)var_input_min_3;
    int32_t input_size_dim_0 = var_input_buffer->extent[0];
    (void)input_size_dim_0;
    int32_t input_size_dim_1 = var_input_buffer->extent[1];
    (void)input_size_dim_1;
    int32_t var_input_extent_2 = var_input_buffer->extent[2];
    (void)var_input_extent_2;
    int32_t var_input_extent_3 = var_input_buffer->extent[3];
    (void)var_input_extent_3;
    int32_t var_input_stride_0 = var_input_buffer->stride[0];
    (void)var_input_stride_0;
    int32_t var_input_stride_1 = var_input_buffer->stride[1];
    (void)var_input_stride_1;
    int32_t var_input_stride_2 = var_input_buffer->stride[2];
    (void)var_input_stride_2;
    int32_t var_input_stride_3 = var_input_buffer->stride[3];
    (void)var_input_stride_3;
    int32_t var_input_elem_size = var_input_buffer->elem_size;
    (void)var_input_elem_size;
    float *var_output = (float *)(var_output_buffer->host);
    (void)var_output;
    const bool var_output_host_and_dev_are_null = (var_output_buffer->host == NULL) && (var_output_buffer->dev == 0);
    (void)var_output_host_and_dev_are_null;
    int32_t var_output_min_0 = var_output_buffer->min[0];
    (void)var_output_min_0;
    int32_t var_output_min_1 = var_output_buffer->min[1];
    (void)var_output_min_1;
    int32_t var_output_min_2 = var_output_buffer->min[2];
    (void)var_output_min_2;
    int32_t var_output_min_3 = var_output_buffer->min[3];
    (void)var_output_min_3;
    int32_t output_size_dim_0 = var_output_buffer->extent[0];
    (void)output_size_dim_0;
    int32_t output_size_dim_1 = var_output_buffer->extent[1];
    (void)output_size_dim_1;
    int32_t var_output_extent_2 = var_output_buffer->extent[2];
    (void)var_output_extent_2;
    int32_t var_output_extent_3 = var_output_buffer->extent[3];
    (void)var_output_extent_3;
    int32_t var_output_stride_0 = var_output_buffer->stride[0];
    (void)var_output_stride_0;
    int32_t var_output_stride_1 = var_output_buffer->stride[1];
    (void)var_output_stride_1;
    int32_t var_output_stride_2 = var_output_buffer->stride[2];
    (void)var_output_stride_2;
    int32_t var_output_stride_3 = var_output_buffer->stride[3];
    (void)var_output_stride_3;
    int32_t var_output_elem_size = var_output_buffer->elem_size;
    (void)var_output_elem_size;
    int32_t assign_81 = gradient_wrapped(var_input_buffer, var_output_buffer, xclbin);
    bool assign_82 = assign_81 == 0;
    if (!assign_82)     {
        return assign_81;
    }
    return 0;
}

