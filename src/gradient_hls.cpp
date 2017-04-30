#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stddef.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>

#include"ap_int.h"

#ifndef TILE_SIZE_DIM0
#define TILE_SIZE_DIM0 (128)
#endif//TILE_SIZE_DIM0
#ifndef TILE_SIZE_DIM1
#define TILE_SIZE_DIM1 (128)
#endif//TILE_SIZE_DIM1
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR (64)
#endif//UNROLL_FACTOR
#define STENCIL_DIM0 (3)
#define STENCIL_DIM1 (3)
#define STENCIL_DISTANCE ((TILE_SIZE_DIM0)*2+1)
#define BURST_WIDTH (512)
#define PIXEL_WIDTH (sizeof(float)*8)

#define TILE_INDEX_DIM0(tile_index) ((tile_index)%(tile_num_dim0))
#define TILE_INDEX_DIM1(tile_index) ((tile_index)/(tile_num_dim0))
#define TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/(PIXEL_WIDTH)))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

void load(bool load_flag, float to[TILE_SIZE_DIM0*TILE_SIZE_DIM1], ap_uint<BURST_WIDTH>* from, int32_t tile_index)
{
    if(load_flag)
    {
        for(int i = 0; i < TILE_SIZE_BURST; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp(from[i+tile_index*TILE_SIZE_BURST]);
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                uint32_t raw_bits = tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH);
                to[i*BURST_WIDTH/PIXEL_WIDTH+j] = *(float*)(&raw_bits);
            }
        }
    }
}

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, float from[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index)
{
    if(store_flag)
    {
        for(int i = 0; i < TILE_SIZE_BURST; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp;
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                float raw_bits = from[i*BURST_WIDTH/PIXEL_WIDTH+j];
                tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH) = *(uint32_t*)(&raw_bits);
            }
            to[i+tile_index*TILE_SIZE_BURST] = tmp;
        }
    }
}

void compute(bool compute_flag, float output[TILE_SIZE_DIM0*TILE_SIZE_DIM1], float input[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index, int32_t tile_num_dim0, int32_t var_f_extent_0, int32_t var_f_extent_1, int32_t var_f_min_0, int32_t var_f_min_1)
{
    if(compute_flag)
    {
        int32_t tile_index_dim0 = TILE_INDEX_DIM0(tile_index);
        int32_t tile_index_dim1 = TILE_INDEX_DIM1(tile_index);

        for(int32_t j = 0; j < TILE_SIZE_DIM1; ++j)
        {
#pragma HLS pipeline II=1
            for(int32_t i = 0; i < TILE_SIZE_DIM0; ++i)
            {
#pragma HLS unroll factor=UNROLL_FACTOR
                int32_t q = Q(tile_index_dim1, j);
                int32_t p = P(tile_index_dim0, i);
                float input_c = input[TILE_SIZE_DIM0*(j+1)+i+1];
                float input_u = input[TILE_SIZE_DIM0*(j+0)+i+1];
                float input_d = input[TILE_SIZE_DIM0*(j+2)+i+1];
                float input_l = input[TILE_SIZE_DIM0*(j+1)+i+0];
                float input_r = input[TILE_SIZE_DIM0*(j+1)+i+2];
                if(p >= var_f_min_0 &&
                   q >= var_f_min_1 &&
                   p < var_f_min_0 + var_f_extent_0 &&
                   q < var_f_min_1 + var_f_extent_1)
                {
                    float assign_95 = input_c;
                    float assign_100 = input_u;
                    float assign_101 = assign_95 - assign_100;
                    float assign_102 = input_l;
                    float assign_103 = assign_95 - assign_102;
                    float assign_109 = input_d;
                    float assign_110 = assign_95 - assign_109;
                    float assign_112 = input_r;
                    float assign_113 = assign_95 - assign_112;
                    float assign_114 = assign_101 * assign_101;
                    float assign_115 = assign_103 * assign_103;
                    float assign_116 = assign_114 + assign_115;
                    float assign_117 = assign_110 * assign_110;
                    float assign_118 = assign_116 + assign_117;
                    float assign_119 = assign_113 * assign_113;
                    float assign_120 = assign_118 + assign_119;
                    float assign_121 = sqrt(assign_120);
                    float assign_122 = 1.f / assign_121;
                    output[TILE_SIZE_DIM0*j+i] = assign_122;
                } // if input_index >= STENCIL_DISTANCE
            } // for output_index
        } // for input_index
        // consume f
    }
}

extern "C"
{

void gradient_kernel(ap_uint<BURST_WIDTH>* var_f, ap_uint<BURST_WIDTH>* var_input, int32_t tile_num_dim0, int32_t tile_num_dim1, int32_t var_f_extent_0, int32_t var_f_extent_1, int32_t var_f_min_0, int32_t var_f_min_1)
{
#pragma HLS INTERFACE m_axi port=var_f offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_input offset=slave depth=65536 bundle=gmem2 latency=120

#pragma HLS INTERFACE s_axilite port=var_f bundle=control
#pragma HLS INTERFACE s_axilite port=var_input bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_f_extent_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_f_extent_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_f_min_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_f_min_1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float  input_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    float  input_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    float output_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    float output_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
#pragma HLS array_partition variable=input_0  cyclic factor=UNROLL_FACTOR dim=1
#pragma HLS array_partition variable=input_1  cyclic factor=UNROLL_FACTOR dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=UNROLL_FACTOR dim=1
#pragma HLS array_partition variable=output_1 cyclic factor=UNROLL_FACTOR dim=1

    int32_t total_tile_num = tile_num_dim0*tile_num_dim1;
    int32_t tile_index;
    bool    load_flag;
    bool compute_flag;
    bool   store_flag;

    for (tile_index = 0; tile_index < total_tile_num+2; ++tile_index)
    {
           load_flag =                   tile_index < total_tile_num;
        compute_flag = tile_index > 0 && tile_index < total_tile_num+1;
          store_flag = tile_index > 1;
        if(tile_index%2==0)
        {
            load(load_flag, input_0, var_input, tile_index);
            compute(compute_flag, output_1, input_1, tile_index-1, tile_num_dim0, var_f_extent_0, var_f_extent_1, var_f_min_0, var_f_min_1);
            store(store_flag, var_f, output_0, tile_index-2);
        }
        else
        {
            load(load_flag, input_1, var_input, tile_index);
            compute(compute_flag, output_0, input_0, tile_index-1, tile_num_dim0, var_f_extent_0, var_f_extent_1, var_f_min_0, var_f_min_1);
            store(store_flag, var_f, output_1, tile_index-2);
        }
    }
}

}
