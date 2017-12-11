#include<assert.h>
#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<time.h>
#include<unistd.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<CL/opencl.h>

#include"blur.h"

#ifndef BURST_WIDTH
#define BURST_WIDTH 512
#endif//BURST_WIDTH
#ifndef PIXEL_WIDTH_I
#define PIXEL_WIDTH_I 16
#endif//PIXEL_WIDTH_I
#ifndef PIXEL_WIDTH_O
#define PIXEL_WIDTH_O 16
#endif//PIXEL_WIDTH_O
#ifndef STENCIL_DIM_0
#define STENCIL_DIM_0 3
#endif//STENCIL_DIM_0
#ifndef STENCIL_DIM_1
#define STENCIL_DIM_1 3
#endif//STENCIL_DIM_1
#ifndef STENCIL_DISTANCE
#define STENCIL_DISTANCE 16002
#endif//STENCIL_DISTANCE
#ifndef CHANNEL_NUM_I
#define CHANNEL_NUM_I 1
#endif//CHANNEL_NUM_I
#ifndef CHANNEL_NUM_O
#define CHANNEL_NUM_O 1
#endif//CHANNEL_NUM_O

int load_xclbin2_to_memory(const char *filename, char **result, char** device)
{
    uint64_t size = 0;
    FILE *f = fopen(filename, "rb");
    if(nullptr == f)
    {
        *result = nullptr;
        fprintf(stderr, "ERROR: cannot open %s\n", filename);
        return -1;
    }
    char magic[8];
    unsigned char cipher[32];
    unsigned char key_block[256];
    uint64_t unique_id;
    fread(magic, sizeof(magic), 1, f);
    if(strcmp(magic, "xclbin2")!=0)
    {
        *result = nullptr;
        fprintf(stderr, "ERROR: %s is not a valid xclbin2 file\n", filename);
        return -2;
    }
    fread(cipher, sizeof(cipher), 1, f);
    fread(key_block, sizeof(key_block), 1, f);
    fread(&unique_id, sizeof(unique_id), 1, f);
    fread(&size, sizeof(size), 1, f);
    char* p = new char[size+1]();
    *result = p;
    memcpy(p, magic, sizeof(magic));
    p += sizeof(magic);
    memcpy(p, cipher, sizeof(cipher));
    p += sizeof(cipher);
    memcpy(p, key_block, sizeof(key_block));
    p += sizeof(key_block);
    memcpy(p, &unique_id, sizeof(unique_id));
    p += sizeof(unique_id);
    memcpy(p, &size, sizeof(size));
    p += sizeof(size);
    uint64_t size_left = size - sizeof(magic) - sizeof(cipher) - sizeof(key_block) - sizeof(unique_id) - sizeof(size);
    if(size_left != fread(p, sizeof(char), size_left, f))
    {
        delete[] p;
        fprintf(stderr, "ERROR: %s is corrupted\n", filename);
        return -3;
    }
    *device = p + 5*8;
    printf("%lu %s\n", size, *device);
    fclose(f);
    return size;
}

static bool halide_rewrite_buffer(buffer_t *b, int32_t elem_size,
                                  int32_t min0, int32_t extent0, int32_t stride0,
                                  int32_t min1, int32_t extent1, int32_t stride1,
                                  int32_t min2, int32_t extent2, int32_t stride2,
                                  int32_t min3, int32_t extent3, int32_t stride3)
{
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

static int blur_wrapped(buffer_t *var_input_buffer, buffer_t *var_output_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS
{
    uint16_t *var_input = (uint16_t *)(var_input_buffer->host);
    (void)var_input;
    const bool var_input_host_and_dev_are_null = (var_input_buffer->host == nullptr) && (var_input_buffer->dev == 0);
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
    int32_t input_size_dim_2 = var_input_buffer->extent[2];
    (void)input_size_dim_2;
    int32_t input_size_dim_3 = var_input_buffer->extent[3];
    (void)input_size_dim_3;
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
    uint16_t *var_output = (uint16_t *)(var_output_buffer->host);
    (void)var_output;
    const bool var_output_host_and_dev_are_null = (var_output_buffer->host == nullptr) && (var_output_buffer->dev == 0);
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
    int32_t output_size_dim_2 = var_output_buffer->extent[2];
    (void)output_size_dim_2;
    int32_t output_size_dim_3 = var_output_buffer->extent[3];
    (void)output_size_dim_3;
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
        bool assign_0 = halide_rewrite_buffer(var_output_buffer, 2, var_output_min_0, output_size_dim_0, 1, var_output_min_1, output_size_dim_1, output_size_dim_0, 0, 0, 0, 0, 0, 0);
        (void)assign_0;
    } // if (var_output_host_and_dev_are_null)
    if (var_input_host_and_dev_are_null)
    {
        int32_t assign_1 = output_size_dim_0 + 2;
        int32_t assign_2 = output_size_dim_1 + 2;
        bool assign_3 = halide_rewrite_buffer(var_input_buffer, 2, var_output_min_0, assign_1, 1, var_output_min_1, assign_2, assign_1, 0, 0, 0, 0, 0, 0);
        (void)assign_3;
    } // if (var_input_host_and_dev_are_null)
    bool assign_4 = var_output_host_and_dev_are_null || var_input_host_and_dev_are_null;
    bool assign_5 = !(assign_4);
    if (assign_5)
    {
        bool assign_6 = var_output_elem_size == 2;
        if (!assign_6)
        {
            int32_t assign_7 = halide_error_bad_elem_size(nullptr, "Buffer output", "uint16_t", var_output_elem_size, 2);
            return assign_7;
        }
        bool assign_8 = var_input_elem_size == 2;
        if (!assign_8)
        {
            int32_t assign_9 = halide_error_bad_elem_size(nullptr, "Buffer input", "uint16_t", var_input_elem_size, 2);
            return assign_9;
        }

        // allocate buffer for tiled input/output
        int32_t tile_num_dim_0 = (output_size_dim_0+TILE_SIZE_DIM_0-STENCIL_DIM_0)/(TILE_SIZE_DIM_0-STENCIL_DIM_0+1);

        // change #chan if there is a env var defined
        int dram_chan = 4;
        bool dram_separate = false;
        if(nullptr!=getenv("DRAM_CHAN"))
        {
            dram_chan = atoi(getenv("DRAM_CHAN"));
        }
        if(nullptr!=getenv("DRAM_SEPARATE"))
        {
            dram_separate = true;
            dram_chan /= 2;
        }

        // align each linearized tile to multiples of BURST_WIDTH
        int64_t tile_pixel_num = TILE_SIZE_DIM_0*input_size_dim_1;
        int64_t tile_burst_num = (tile_pixel_num-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan*CHANNEL_NUM_I)+1;
        int64_t tile_size_linearized_i = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan*CHANNEL_NUM_I);
        int64_t tile_size_linearized_o = tile_burst_num*(BURST_WIDTH/PIXEL_WIDTH_O*dram_chan*CHANNEL_NUM_O);

        uint16_t* var_input_buf_chan_0 = new uint16_t[tile_num_dim_0*tile_size_linearized_i/dram_chan];
        uint16_t* var_input_buf_chan_1 = new uint16_t[tile_num_dim_0*tile_size_linearized_i/dram_chan];
        uint16_t* var_input_buf_chan_2 = new uint16_t[tile_num_dim_0*tile_size_linearized_i/dram_chan];
        uint16_t* var_input_buf_chan_3 = new uint16_t[tile_num_dim_0*tile_size_linearized_i/dram_chan];
        uint16_t* var_output_buf_chan_0 = new uint16_t[tile_num_dim_0*tile_size_linearized_o/dram_chan];
        uint16_t* var_output_buf_chan_1 = new uint16_t[tile_num_dim_0*tile_size_linearized_o/dram_chan];
        uint16_t* var_output_buf_chan_2 = new uint16_t[tile_num_dim_0*tile_size_linearized_o/dram_chan];
        uint16_t* var_output_buf_chan_3 = new uint16_t[tile_num_dim_0*tile_size_linearized_o/dram_chan];

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
                        int32_t burst_index = (j*TILE_SIZE_DIM_0+i)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan);
                        int32_t burst_residue = (j*TILE_SIZE_DIM_0+i)%(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan);
                        int32_t p = tile_index_dim_0*(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)+i;
                        int32_t q = j;
                        int64_t tiled_offset = tile_index_dim_0*tile_size_linearized_i+(burst_index*(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan))*CHANNEL_NUM_I+c*(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan)+burst_residue;
                        int64_t original_offset = (q*var_input_stride_1+p)*CHANNEL_NUM_I+c;
                        switch(tiled_offset%dram_chan)
                        {
                            case 0:
                                var_input_buf_chan_0[tiled_offset/dram_chan] = var_input[original_offset];
                                break;
                            case 1:
                                var_input_buf_chan_1[tiled_offset/dram_chan] = var_input[original_offset];
                                break;
                            case 2:
                                var_input_buf_chan_2[tiled_offset/dram_chan] = var_input[original_offset];
                                break;
                            case 3:
                                var_input_buf_chan_3[tiled_offset/dram_chan] = var_input[original_offset];
                                break;
                        }
                    }
                }
            }
        }

        // prepare for opencl
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
       
        cl_mem var_input_chan_0_cl;                   // device memory used for the input array
        cl_mem var_input_chan_1_cl;                   // device memory used for the input array
        cl_mem var_input_chan_2_cl;                   // device memory used for the input array
        cl_mem var_input_chan_3_cl;                   // device memory used for the input array
        cl_mem var_output_chan_0_cl;               // device memory used for the output array
        cl_mem var_output_chan_1_cl;               // device memory used for the output array
        cl_mem var_output_chan_2_cl;               // device memory used for the output array
        cl_mem var_output_chan_3_cl;               // device memory used for the output array

        unsigned char *kernelbinary;
        const char *device_name;
        char target_device_name[64];
        printf("INFO: Loading %s\n", xclbin);
        int n_i = load_xclbin2_to_memory(xclbin, (char **) &kernelbinary, (char**)&device_name);
        if (n_i < 0) {
            printf("FATAL: Failed to load kernel from xclbin: %s\n", xclbin);
            exit(EXIT_FAILURE);
        }
        for(int i = 0; i<64; ++i)
        {
            if(device_name[i]==':' || device_name[i]=='.')
            {
                target_device_name[i] = '_';
            }
            else
            {
                target_device_name[i] = device_name[i];
            }
        }
   
        err = clGetPlatformIDs(16, platforms, &platform_count);
        if (err != CL_SUCCESS)
        {
            printf("FATAL: Failed to find an OpenCL platform\n");
            exit(EXIT_FAILURE);
        }
        printf("INFO: Found %d platforms\n", platform_count);

        int platform_found = 0;
        for (unsigned iplat = 0; iplat<platform_count; iplat++) {
            err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,nullptr);
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
            //printf("INFO: Find device %s\n", cl_device_name);
            if(strcmp(cl_device_name, target_device_name) == 0 || strcmp(cl_device_name, device_name) == 0) {
                device_id = devices[i];
                device_found = 1;
                printf("INFO: Selected %s as the target device\n", device_name);
            }
        }
        
        if (!device_found) {
            printf("FATAL: Target device %s not found\n", target_device_name);
            exit(EXIT_FAILURE);
        }


        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR,
                             1, &device_id, nullptr);
        if (err != CL_SUCCESS) {
            printf("FATAL: Failed to create a device group\n");
            exit(EXIT_FAILURE);
        }
      
        context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &err);
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

        size_t n = n_i;
        program = clCreateProgramWithBinary(context, 1, &device_id, &n,
                                           (const unsigned char **) &kernelbinary, &status, &err);
        if ((!program) || (err!=CL_SUCCESS)) {
            printf("FATAL: Failed to create compute program from binary %d\n", err);
            exit(EXIT_FAILURE);
        }
        free(kernelbinary);

        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len;
            char buffer[2048];
            printf("FATAL: Failed to build program executable\n");
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            exit(EXIT_FAILURE);
        }

        kernel = clCreateKernel(program, "blur_kernel", &err);
        if (!kernel || err != CL_SUCCESS) {
            printf("FATAL: Failed to create compute kernel %d\n", err);
            exit(EXIT_FAILURE);
        }

        cl_mem_ext_ptr_t input_ext_chan_0, output_ext_chan_0;
        cl_mem_ext_ptr_t input_ext_chan_1, output_ext_chan_1;
        cl_mem_ext_ptr_t input_ext_chan_2, output_ext_chan_2;
        cl_mem_ext_ptr_t input_ext_chan_3, output_ext_chan_3;
        if(dram_separate)
        {
            switch(dram_chan)
            {
                case 2:
                    output_ext_chan_0.flags = XCL_MEM_DDR_BANK0;
                    output_ext_chan_1.flags = XCL_MEM_DDR_BANK1;
                    input_ext_chan_0.flags = XCL_MEM_DDR_BANK2;
                    input_ext_chan_1.flags = XCL_MEM_DDR_BANK3;
                    break;
                case 1:
                    output_ext_chan_0.flags = XCL_MEM_DDR_BANK0;
                    input_ext_chan_0.flags = XCL_MEM_DDR_BANK1;
                    break;
            }
        }
        else
        {
            output_ext_chan_0.flags = XCL_MEM_DDR_BANK0;
            output_ext_chan_1.flags = XCL_MEM_DDR_BANK1;
            output_ext_chan_2.flags = XCL_MEM_DDR_BANK2;
            output_ext_chan_3.flags = XCL_MEM_DDR_BANK3;
            input_ext_chan_0.flags = XCL_MEM_DDR_BANK0;
            input_ext_chan_1.flags = XCL_MEM_DDR_BANK1;
            input_ext_chan_2.flags = XCL_MEM_DDR_BANK2;
            input_ext_chan_3.flags = XCL_MEM_DDR_BANK3;
        }
        input_ext_chan_0.obj = 0;
        input_ext_chan_0.param = 0;
        input_ext_chan_1.obj = 0;
        input_ext_chan_1.param = 0;
        input_ext_chan_2.obj = 0;
        input_ext_chan_2.param = 0;
        input_ext_chan_3.obj = 0;
        input_ext_chan_3.param = 0;
        output_ext_chan_0.obj = 0;
        output_ext_chan_0.param = 0;
        output_ext_chan_1.obj = 0;
        output_ext_chan_1.param = 0;
        output_ext_chan_2.obj = 0;
        output_ext_chan_2.param = 0;
        output_ext_chan_3.obj = 0;
        output_ext_chan_3.param = 0;

        var_input_chan_0_cl  = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_i)/dram_chan, & input_ext_chan_0, nullptr);
        var_input_chan_1_cl  = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_i)/dram_chan, & input_ext_chan_1, nullptr);
        var_input_chan_2_cl  = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_i)/dram_chan, & input_ext_chan_2, nullptr);
        var_input_chan_3_cl  = clCreateBuffer(context,  CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_i)/dram_chan, & input_ext_chan_3, nullptr);
        var_output_chan_0_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_o)/dram_chan, &output_ext_chan_0, nullptr);
        var_output_chan_1_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_o)/dram_chan, &output_ext_chan_1, nullptr);
        var_output_chan_2_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_o)/dram_chan, &output_ext_chan_2, nullptr);
        var_output_chan_3_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(uint16_t) * (tile_num_dim_0*tile_size_linearized_o)/dram_chan, &output_ext_chan_3, nullptr);
        if (!var_input_chan_0_cl || !var_output_chan_0_cl || !var_input_chan_1_cl || !var_output_chan_1_cl || !var_input_chan_2_cl || !var_output_chan_2_cl|| !var_input_chan_3_cl || !var_output_chan_3_cl)
        {
            printf("FATAL: Failed to allocate device memory\n");
            exit(EXIT_FAILURE);
        }
        
        timespec write_begin, write_end;
        cl_event writeevent;
        clock_gettime(CLOCK_MONOTONIC_RAW, &write_begin);
        err = clEnqueueWriteBuffer(commands, var_input_chan_0_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_i/dram_chan, var_input_buf_chan_0, 0, nullptr, &writeevent);
        err = clEnqueueWriteBuffer(commands, var_input_chan_1_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_i/dram_chan, var_input_buf_chan_1, 0, nullptr, &writeevent);
        err = clEnqueueWriteBuffer(commands, var_input_chan_2_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_i/dram_chan, var_input_buf_chan_2, 0, nullptr, &writeevent);
        err = clEnqueueWriteBuffer(commands, var_input_chan_3_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_i/dram_chan, var_input_buf_chan_3, 0, nullptr, &writeevent);
        if (err != CL_SUCCESS)
        {
            printf("FATAL: Failed to write to input !\n");
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &writeevent);
        clock_gettime(CLOCK_MONOTONIC_RAW, &write_end);

        err = 0;
        printf("HOST: Using %d DRAM channels%s\n", dram_chan, dram_separate ? ", separated" : "");
        printf("HOST: tile_num_dim_0 = %d, TILE_SIZE_DIM_0 = %d\n", tile_num_dim_0, TILE_SIZE_DIM_0);
        printf("HOST: var_input_extent_0 = %d, var_input_extent_1 = %d\n", input_size_dim_0, input_size_dim_1);
        printf("HOST: var_input_min_0 = %d, var_input_min_1 = %d\n", var_input_min_0, var_input_min_1);
        printf("HOST: output_size_dim_0 = %d, output_size_dim_1 = %d\n", output_size_dim_0, output_size_dim_1);
        printf("HOST: var_output_min_0 = %d, var_output_min_1 = %d\n", var_output_min_0, var_output_min_1);

        int kernel_arg = 0;
        int64_t tile_data_num = ((int64_t(input_size_dim_1)*TILE_SIZE_DIM_0-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan)+1)*BURST_WIDTH/PIXEL_WIDTH_I*dram_chan/UNROLL_FACTOR;
        int64_t coalesced_data_num = ((int64_t(input_size_dim_1)*TILE_SIZE_DIM_0-1)/(BURST_WIDTH/PIXEL_WIDTH_I*dram_chan)+1)*tile_num_dim_0;
        printf("HOST: tile_data_num = %ld, coalesced_data_num = %ld\n", tile_data_num, coalesced_data_num);

        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_output_chan_0_cl);
        if(dram_chan>1)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_output_chan_1_cl);
        if(dram_chan>2)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_output_chan_2_cl);
        if(dram_chan>3)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_output_chan_3_cl);
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_input_chan_0_cl);
        if(dram_chan>1)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_input_chan_1_cl);
        if(dram_chan>2)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_input_chan_2_cl);
        if(dram_chan>3)
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(cl_mem), &var_input_chan_3_cl);
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(coalesced_data_num), &coalesced_data_num);
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(tile_data_num), &tile_data_num);
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(tile_num_dim_0), &tile_num_dim_0);
        err |= clSetKernelArg(kernel, kernel_arg++, sizeof(input_size_dim_1), &input_size_dim_1);
        if (err != CL_SUCCESS)
        {
            printf("FATAL: Failed to set kernel arguments\n");
            printf("ERROR code: %d\n", err);
            exit(EXIT_FAILURE);
        }

        cl_event execute;
        if(nullptr==getenv("XCL_EMULATION_MODE")) {
            printf("INFO: FPGA warm up\n");
            for(int i = 0; i<3; ++i)
            {
                err = clEnqueueTask(commands, kernel, 0, nullptr, &execute);
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
        clock_gettime(CLOCK_MONOTONIC_RAW, &execute_begin);
        err = clEnqueueTask(commands, kernel, 0, nullptr, &execute);
        if (err)
        {
            printf("ERROR: Failed to execute kernel %d\n", err);
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &execute);
        clock_gettime(CLOCK_MONOTONIC_RAW, &execute_end);

        // Read back the results from the device to verify the output
        //
        timespec read_begin, read_end;
        cl_event readevent;
        clock_gettime(CLOCK_MONOTONIC_RAW, &read_begin);
        err = clEnqueueReadBuffer(commands, var_output_chan_0_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_o/dram_chan, var_output_buf_chan_0, 0, nullptr, &readevent );
        err = clEnqueueReadBuffer(commands, var_output_chan_1_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_o/dram_chan, var_output_buf_chan_1, 0, nullptr, &readevent );
        err = clEnqueueReadBuffer(commands, var_output_chan_2_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_o/dram_chan, var_output_buf_chan_2, 0, nullptr, &readevent );
        err = clEnqueueReadBuffer(commands, var_output_chan_3_cl, CL_FALSE, 0, sizeof(uint16_t) * tile_num_dim_0*tile_size_linearized_o/dram_chan, var_output_buf_chan_3, 0, nullptr, &readevent );
        if (err != CL_SUCCESS)
        {
            printf("ERROR: Failed to read output %d\n", err);
            exit(EXIT_FAILURE);
        }

        clWaitForEvents(1, &readevent);
        clock_gettime(CLOCK_MONOTONIC_RAW, &read_end);

        double elapsed_time = 0.;
        elapsed_time = (double(write_end.tv_sec-write_begin.tv_sec)+(write_end.tv_nsec-write_begin.tv_nsec)/1e9)*1e6;
        elapsed_time = (double(execute_end.tv_sec-execute_begin.tv_sec)+(execute_end.tv_nsec-execute_begin.tv_nsec)/1e9)*1e6;
        printf("Kernel run time:       %lf us\n", elapsed_time);
        printf("Kernel throughput:     %lf pixel/ns\n", input_size_dim_0*input_size_dim_1/elapsed_time/1e3);
        elapsed_time = (double(read_end.tv_sec-read_begin.tv_sec)+(read_end.tv_nsec-read_begin.tv_nsec)/1e9)*1e6;

        clReleaseMemObject(var_input_chan_0_cl);
        clReleaseMemObject(var_input_chan_1_cl);
        clReleaseMemObject(var_output_chan_0_cl);
        clReleaseMemObject(var_output_chan_1_cl);
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
                        int32_t burst_index = (j*TILE_SIZE_DIM_0+i+STENCIL_DISTANCE)/(BURST_WIDTH/PIXEL_WIDTH_O*dram_chan);
                        int32_t burst_residue = (j*TILE_SIZE_DIM_0+i+STENCIL_DISTANCE)%(BURST_WIDTH/PIXEL_WIDTH_O*dram_chan);
                        int32_t p = tile_index_dim_0*(TILE_SIZE_DIM_0-STENCIL_DIM_0+1)+i;
                        int32_t q = j;
                        //int64_t tiled_offset = (tile_index_dim_0*tile_pixel_num+burst_index*BURST_LENGTH*4)*CHANNEL_NUM_O+c*BURST_LENGTH*4+burst_residue;
                        int64_t tiled_offset = tile_index_dim_0*tile_size_linearized_o+(burst_index*(BURST_WIDTH/PIXEL_WIDTH_O*dram_chan))*CHANNEL_NUM_O+c*(BURST_WIDTH/PIXEL_WIDTH_O*dram_chan)+burst_residue;
                        int64_t original_offset = (q*var_output_stride_1+p)*CHANNEL_NUM_O+c;
                        //printf("p=%d, q=%d, i=%d, j=%d, tiled_offset = %d, original_offset = %d, var_output_buf[tiled_offset] = %d\n",p,q,i,j,tiled_offset, original_offset, var_output_buf[tiled_offset]);
                        switch(tiled_offset%dram_chan)
                        {
                            case 0:
                                var_output[original_offset] = var_output_buf_chan_0[tiled_offset/dram_chan];
                                break;
                            case 1:
                                var_output[original_offset] = var_output_buf_chan_1[tiled_offset/dram_chan];
                                break;
                            case 2:
                                var_output[original_offset] = var_output_buf_chan_2[tiled_offset/dram_chan];
                                break;
                            case 3:
                                var_output[original_offset] = var_output_buf_chan_3[tiled_offset/dram_chan];
                                break;
                        }
                    }
                }
            }
        }
        delete[] var_output_buf_chan_0;
        delete[] var_output_buf_chan_1;
        delete[] var_output_buf_chan_2;
        delete[] var_output_buf_chan_3;
        delete[] var_input_buf_chan_0;
        delete[] var_input_buf_chan_1;
        delete[] var_input_buf_chan_2;
        delete[] var_input_buf_chan_3;
    } // if (assign_5)
    return 0;
}

int blur(buffer_t *var_input_buffer, buffer_t *var_output_buffer, const char* xclbin) HALIDE_FUNCTION_ATTRS {
    uint16_t *var_input = (uint16_t *)(var_input_buffer->host);
    (void)var_input;
    const bool var_input_host_and_dev_are_null = (var_input_buffer->host == nullptr) && (var_input_buffer->dev == 0);
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
    int32_t input_size_dim_2 = var_input_buffer->extent[2];
    (void)input_size_dim_2;
    int32_t input_size_dim_3 = var_input_buffer->extent[3];
    (void)input_size_dim_3;
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
    uint16_t *var_output = (uint16_t *)(var_output_buffer->host);
    (void)var_output;
    const bool var_output_host_and_dev_are_null = (var_output_buffer->host == nullptr) && (var_output_buffer->dev == 0);
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
    int32_t output_size_dim_2 = var_output_buffer->extent[2];
    (void)output_size_dim_2;
    int32_t output_size_dim_3 = var_output_buffer->extent[3];
    (void)output_size_dim_3;
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
    int32_t assign_81 = blur_wrapped(var_input_buffer, var_output_buffer, xclbin);
    bool assign_82 = assign_81 == 0;
    if (!assign_82)     {
        return assign_81;
    }
    return 0;
}

