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
#define STENCIL_DISTANCE ((TILE_SIZE_DIM0)*2+2)
#define BURST_WIDTH (512)
#define PIXEL_WIDTH (sizeof(uint8_t)*8)

#define TILE_INDEX_DIM0(tile_index) ((tile_index)%(tile_num_dim0))
#define TILE_INDEX_DIM1(tile_index) ((tile_index)/(tile_num_dim0))
#define TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/(PIXEL_WIDTH)))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

void load(bool load_flag, uint8_t to[TILE_SIZE_DIM0*TILE_SIZE_DIM1], ap_uint<BURST_WIDTH>* from, int32_t tile_index)
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
                to[i*BURST_WIDTH/PIXEL_WIDTH+j] = tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH);
            }
        }
    }
}

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, uint8_t from[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index)
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
                tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH) = from[i*BURST_WIDTH/PIXEL_WIDTH+j];
            }
            to[i+tile_index*TILE_SIZE_BURST] = tmp;
        }
    }
}

void compute(bool compute_flag, uint8_t output[TILE_SIZE_DIM0*TILE_SIZE_DIM1], uint8_t input[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index, int32_t tile_num_dim0, int32_t var_edge_y_extent_0, int32_t var_edge_y_extent_1, int32_t var_edge_y_min_0, int32_t var_edge_y_min_1)
{
    if(compute_flag)
    {
        int32_t tile_index_dim0 = TILE_INDEX_DIM0(tile_index);
        int32_t tile_index_dim1 = TILE_INDEX_DIM1(tile_index);

        uint8_t stencil_buf[STENCIL_DISTANCE+UNROLL_FACTOR];
#pragma HLS array_partition variable=stencil_buf complete dim=1

        // produce edge_y
        for (int32_t input_index = 0; input_index < (STENCIL_DISTANCE+TILE_SIZE_DIM0*TILE_SIZE_DIM1+UNROLL_FACTOR-1)/UNROLL_FACTOR; ++input_index)
        {
#pragma HLS pipeline II=1
            for (int32_t stencil_index = 0; stencil_index < UNROLL_FACTOR; ++stencil_index)
            {
#pragma HLS unroll
                if(input_index*UNROLL_FACTOR+stencil_index < TILE_SIZE_DIM0*TILE_SIZE_DIM1)
                {
                    stencil_buf[STENCIL_DISTANCE+stencil_index] = input[input_index*UNROLL_FACTOR+stencil_index];
                }
            }
            for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)
            {
#pragma HLS unroll
                if(input_index >= STENCIL_DISTANCE/UNROLL_FACTOR)
                {
                    int32_t output_index = (input_index-STENCIL_DISTANCE/UNROLL_FACTOR)*UNROLL_FACTOR+unroll_index-STENCIL_DISTANCE%UNROLL_FACTOR;
                    if((input_index-STENCIL_DISTANCE/UNROLL_FACTOR)*UNROLL_FACTOR+unroll_index >= STENCIL_DISTANCE%UNROLL_FACTOR &&
                        output_index < TILE_SIZE_DIM0*TILE_SIZE_DIM1)
                    {
                        int32_t i = output_index%TILE_SIZE_DIM0;
                        int32_t j = output_index/TILE_SIZE_DIM0;
                        int32_t q = Q(tile_index_dim1, j);
                        int32_t p = P(tile_index_dim0, i);
                        uint8_t input_0 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+0];
                        uint8_t input_1 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+1];
                        uint8_t input_2 = stencil_buf[unroll_index+TILE_SIZE_DIM0*0+2];
                        uint8_t input_3 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+0];
                        uint8_t input_4 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+1];
                        uint8_t input_5 = stencil_buf[unroll_index+TILE_SIZE_DIM0*1+2];
                        uint8_t input_6 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+0];
                        uint8_t input_7 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+1];
                        uint8_t input_8 = stencil_buf[unroll_index+TILE_SIZE_DIM0*2+2];
                        if(p >= var_edge_y_min_0 &&
                           q >= var_edge_y_min_1 &&
                           p < var_edge_y_min_0 + var_edge_y_extent_0 &&
                           q < var_edge_y_min_1 + var_edge_y_extent_1)
                        {
                            uint16_t assign_100 = (uint16_t)input_4;
                            uint16_t assign_102 = assign_100 << 3;           
                            uint16_t assign_104 = (uint16_t)input_0;
                            uint16_t assign_105 = assign_102 - assign_104;           
                            uint16_t assign_106 = (uint16_t)input_1; 
                            uint16_t assign_107 = assign_105 - assign_106;           
                            uint16_t assign_109 = (uint16_t)input_2;
                            uint16_t assign_110 = assign_107 - assign_109;           
                            uint16_t assign_112 = (uint16_t)input_3;
                            uint16_t assign_113 = assign_110 - assign_112;           
                            uint16_t assign_115 = (uint16_t)input_5;
                            uint16_t assign_116 = assign_113 - assign_115;           
                            uint16_t assign_118 = (uint16_t)input_6;
                            uint16_t assign_119 = assign_116 - assign_118;           
                            uint16_t assign_121 = (uint16_t)input_7;
                            uint16_t assign_122 = assign_119 - assign_121;           
                            uint16_t assign_124 = (uint16_t)input_8;
                            uint16_t assign_125 = assign_122 - assign_124;           
                            output[output_index] = (uint8_t)assign_125;
                        }
                    }
                } // if input_index >= STENCIL_DISTANCE
            } // for output_index
            for(int32_t stencil_index = 0; stencil_index < STENCIL_DISTANCE; ++stencil_index)
            {
#pragma HLS unroll
                stencil_buf[stencil_index] = stencil_buf[stencil_index+UNROLL_FACTOR];
            }
        } // for input_index
        // consume edge_y
    }
}

extern "C"
{

void edge_kernel(ap_uint<BURST_WIDTH>* var_edge_y, ap_uint<BURST_WIDTH>* var_p0, int32_t tile_num_dim0, int32_t tile_num_dim1, int32_t var_edge_y_extent_0, int32_t var_edge_y_extent_1, int32_t var_edge_y_min_0, int32_t var_edge_y_min_1)
{
#pragma HLS INTERFACE m_axi port=var_edge_y offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth=65536 bundle=gmem2 latency=120

#pragma HLS INTERFACE s_axilite port=var_edge_y bundle=control
#pragma HLS INTERFACE s_axilite port=var_p0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_edge_y_extent_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_edge_y_extent_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_edge_y_min_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_edge_y_min_1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    uint8_t  input_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint8_t  input_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint8_t output_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint8_t output_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
#pragma HLS array_partition variable=input_0 cyclic factor=KI dim=1
#pragma HLS array_partition variable=input_1 cyclic factor=KI dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=KO dim=1
#pragma HLS array_partition variable=output_1 cyclic factor=KO dim=1

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
            load(load_flag, input_0, var_p0, tile_index);
            compute(compute_flag, output_1, input_1, tile_index-1, tile_num_dim0, var_edge_y_extent_0, var_edge_y_extent_1, var_edge_y_min_0, var_edge_y_min_1);
            store(store_flag, var_edge_y, output_0, tile_index-2);
        }
        else
        {
            load(load_flag, input_1, var_p0, tile_index);
            compute(compute_flag, output_0, input_0, tile_index-1, tile_num_dim0, var_edge_y_extent_0, var_edge_y_extent_1, var_edge_y_min_0, var_edge_y_min_1);
            store(store_flag, var_edge_y, output_1, tile_index-2);
        }
    }
}

}
