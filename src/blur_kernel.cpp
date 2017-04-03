#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stddef.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>

#include"ap_int.h"

#define TILE_SIZE_DIM0 (128)
#define TILE_SIZE_DIM1 (128)
#define STENCIL_DIM0 (3)
#define STENCIL_DIM1 (3)
#define STENCIL_DISTANCE ((TILE_SIZE_DIM0)*2+2)
#define UNROLL_FACTOR (64)
#define BURST_WIDTH (512)
#define PIXEL_WIDTH (sizeof(uint16_t)*8)

#define TILE_INDEX_DIM0(tile_index) ((tile_index)%(tile_num_dim0))
#define TILE_INDEX_DIM1(tile_index) ((tile_index)/(tile_num_dim0))
#define TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/(PIXEL_WIDTH)))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

void load(bool load_flag, ap_uint<BURST_WIDTH> to[TILE_SIZE_BURST], ap_uint<BURST_WIDTH>* from, size_t tile_index)
{
    if(load_flag)
    {
        memcpy(to, from + tile_index*TILE_SIZE_BURST, TILE_SIZE_DIM0*TILE_SIZE_DIM1*sizeof(uint16_t));
    }
}

void unpack(bool unpack_flag, uint16_t to[TILE_SIZE_DIM0*TILE_SIZE_DIM1], ap_uint<BURST_WIDTH> from[TILE_SIZE_BURST])
{
    if(unpack_flag)
    {
        for(int i = 0; i < TILE_SIZE_BURST; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp(from[i]);
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                to[i*BURST_WIDTH/PIXEL_WIDTH+j] = tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH);
            }
        }
    }
}

void pack(bool pack_flag, ap_uint<BURST_WIDTH> to[TILE_SIZE_BURST], uint16_t from[TILE_SIZE_DIM0*TILE_SIZE_DIM1])
{
    if(pack_flag)
    {
        for(int i = 0; i < TILE_SIZE_BURST; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp;
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH) = from[i*BURST_WIDTH/PIXEL_WIDTH+j];
            }
            to[i] = tmp;
        }
    }
}

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, ap_uint<BURST_WIDTH> from[TILE_SIZE_BURST], size_t tile_index)
{
    if(store_flag)
    {
        memcpy(to + tile_index*TILE_SIZE_BURST, from, TILE_SIZE_DIM0*TILE_SIZE_DIM1*sizeof(uint16_t));
    }
}

void compute(bool compute_flag, uint16_t output[TILE_SIZE_DIM0*TILE_SIZE_DIM1], uint16_t input[TILE_SIZE_DIM0*TILE_SIZE_DIM1], size_t tile_index, size_t tile_num_dim0, int var_blur_y_extent_0, int var_blur_y_extent_1, int var_blur_y_min_0, int var_blur_y_min_1)
{
    if(compute_flag)
    {
        size_t tile_index_dim0 = TILE_INDEX_DIM0(tile_index);
        size_t tile_index_dim1 = TILE_INDEX_DIM1(tile_index);

        uint16_t stencil_buf[STENCIL_DISTANCE+UNROLL_FACTOR];
#pragma HLS array_partition variable=stencil_buf cyclic factor=64 dim=1

        // produce blur_y
        for (size_t input_index = 0; input_index < (STENCIL_DISTANCE+TILE_SIZE_DIM0*TILE_SIZE_DIM1+UNROLL_FACTOR-1)/UNROLL_FACTOR; ++input_index)
        {
#pragma HLS pipeline II=1
            for (size_t stencil_index = 0; stencil_index < UNROLL_FACTOR; ++stencil_index)
            {
#pragma HLS unroll
                if(input_index*UNROLL_FACTOR+stencil_index < TILE_SIZE_DIM0*TILE_SIZE_DIM1)
                {
                    stencil_buf[STENCIL_DISTANCE+stencil_index] = input[input_index*UNROLL_FACTOR+stencil_index];
                }
            }
            if(input_index >= STENCIL_DISTANCE/UNROLL_FACTOR)
            {
                for(size_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)
                {
#pragma HLS unroll
                    size_t output_index = (input_index-STENCIL_DISTANCE/UNROLL_FACTOR)*UNROLL_FACTOR+unroll_index-STENCIL_DISTANCE%UNROLL_FACTOR;
                    if((input_index-STENCIL_DISTANCE/UNROLL_FACTOR)*UNROLL_FACTOR+unroll_index >= STENCIL_DISTANCE%UNROLL_FACTOR &&
                        output_index < TILE_SIZE_DIM0*TILE_SIZE_DIM1)
                    {
                        size_t i = output_index%TILE_SIZE_DIM0;
                        size_t j = output_index/TILE_SIZE_DIM0;
                        size_t q = Q(tile_index_dim1, j);
                        size_t p = P(tile_index_dim0, i);
                        uint16_t input_0 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+0];
                        uint16_t input_1 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+1];
                        uint16_t input_2 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+2];
                        uint16_t input_3 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+0];
                        uint16_t input_4 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+1];
                        uint16_t input_5 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+2];
                        uint16_t input_6 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+0];
                        uint16_t input_7 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+1];
                        uint16_t input_8 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+2];
                        if(p >= var_blur_y_min_0 &&
                           q >= var_blur_y_min_1 &&
                           p < var_blur_y_min_0 + var_blur_y_extent_0 &&
                           q < var_blur_y_min_1 + var_blur_y_extent_1)
                        {
                            uint16_t assign_99 = input_0;
                            uint16_t assign_101 = input_1;
                            uint16_t assign_102 = assign_99 + assign_101;
                            uint16_t assign_104 = input_2;
                            uint16_t assign_105 = assign_102 + assign_104;
                            uint16_t assign_106 = (uint16_t)(3);
                            uint16_t assign_107 = assign_105 / assign_106;
                            uint16_t assign_109 = input_3;
                            uint16_t assign_111 = input_4;
                            uint16_t assign_112 = assign_109 + assign_111;
                            uint16_t assign_114 = input_5;
                            uint16_t assign_115 = assign_112 + assign_114;
                            uint16_t assign_116 = assign_115 / assign_106;
                            uint16_t assign_117 = assign_107 + assign_116;
                            uint16_t assign_119 = input_6;
                            uint16_t assign_121 = input_7;
                            uint16_t assign_122 = assign_119 + assign_121;
                            uint16_t assign_124 = input_8;
                            uint16_t assign_125 = assign_122 + assign_124;
                            uint16_t assign_126 = assign_125 / assign_106;
                            uint16_t assign_127 = assign_117 + assign_126;
                            uint16_t assign_128 = assign_127 / assign_106;
                            output[output_index] = assign_128;
                        }
                    }
                } // for output_index
            } // if input_index >= STENCIL_DISTANCE
            for(size_t stencil_index = 0; stencil_index < STENCIL_DISTANCE; ++stencil_index)
            {
#pragma HLS unroll
                stencil_buf[stencil_index] = stencil_buf[stencil_index+UNROLL_FACTOR];
            }
        } // for input_index
        // consume blur_y
    }
}

extern "C"
{

void blur_kernel(size_t tile_num_dim0_ptr[1], size_t tile_num_dim1_ptr[1], size_t var_blur_y_extent_0_ptr[1], size_t var_blur_y_extent_1_ptr[1], size_t var_blur_y_min_0_ptr[1], size_t var_blur_y_min_1_ptr[1], ap_uint<BURST_WIDTH>* var_blur_y, ap_uint<BURST_WIDTH>* var_p0)
{
#pragma HLS INTERFACE m_axi port=var_blur_y offset=slave depth=512 bundle=gmem1
#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth=512 bundle=gmem2

#pragma HLS INTERFACE s_axilite port=tile_num_dim0_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_extent_0_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_extent_1_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_min_0_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_min_1_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    size_t tile_num_dim0 = tile_num_dim0_ptr[0];
    size_t tile_num_dim1 = tile_num_dim1_ptr[0];
    printf("tile_num_dim0 = %lu, tile_num_dim1 = %lu\n", tile_num_dim0, tile_num_dim1);
    size_t var_blur_y_extent_0 = var_blur_y_extent_0_ptr[0];
    size_t var_blur_y_extent_1 = var_blur_y_extent_1_ptr[0];
    printf("var_blur_y_extent_0 = %lu, var_blur_y_extent_1 = %lu\n", var_blur_y_extent_0, var_blur_y_extent_1);
    size_t var_blur_y_min_0 = var_blur_y_min_0_ptr[0];
    size_t var_blur_y_min_1 = var_blur_y_min_1_ptr[0];
    printf("var_blur_y_min_0 = %lu, var_blur_y_min_1 = %lu\n", var_blur_y_min_0, var_blur_y_min_1);

    ap_uint<BURST_WIDTH>  load_0[TILE_SIZE_BURST];
    ap_uint<BURST_WIDTH>  load_1[TILE_SIZE_BURST];
    ap_uint<BURST_WIDTH> store_0[TILE_SIZE_BURST];
    ap_uint<BURST_WIDTH> store_1[TILE_SIZE_BURST];
    uint16_t  input_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t  input_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t output_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t output_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
#pragma HLS array_partition variable=input_0 cyclic factor=64 dim=1
#pragma HLS array_partition variable=input_1 cyclic factor=64 dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=64 dim=1
#pragma HLS array_partition variable=output_1 cyclic factor=64 dim=1

    int total_tile_num = tile_num_dim0*tile_num_dim1;
    int tile_index;
    bool    load_flag;
    bool  unpack_flag;
    bool compute_flag;
    bool    pack_flag;
    bool   store_flag;

    for (tile_index = 0; tile_index < total_tile_num+4; ++tile_index)
    {
           load_flag =                   tile_index < total_tile_num;
         unpack_flag = tile_index > 0 && tile_index < total_tile_num+1;
        compute_flag = tile_index > 1 && tile_index < total_tile_num+2;
           pack_flag = tile_index > 2 && tile_index < total_tile_num+3;
          store_flag = tile_index > 3;
        if(tile_index%2==0)
        {
            load(load_flag, load_0, var_p0, tile_index);
            unpack(unpack_flag, input_1, load_1);
            compute(compute_flag, output_0, input_0, tile_index-2, tile_num_dim0, var_blur_y_extent_0, var_blur_y_extent_1, var_blur_y_min_0, var_blur_y_min_1);
            pack(pack_flag, store_1, output_1);
            store(store_flag, var_blur_y, store_0, tile_index-4);
        }
        else
        {
            load(load_flag, load_1, var_p0, tile_index);
            unpack(unpack_flag, input_0, load_0);
            compute(compute_flag, output_1, input_1, tile_index-2, tile_num_dim0, var_blur_y_extent_0, var_blur_y_extent_1, var_blur_y_min_0, var_blur_y_min_1);
            pack(pack_flag, store_0, output_0);
            store(store_flag, var_blur_y, store_1, tile_index-4);
        }
    }
}

}
