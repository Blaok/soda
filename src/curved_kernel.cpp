#include <iostream>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

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
#define UNROLL_FACTOR (32)
#endif//UNROLL_FACTOR

#define STENCIL_DIM0 (23)
#define STENCIL_DIM1 (19)
#define STENCIL_OFFSET ((TILE_SIZE_DIM0)*7+15)
#define STENCIL_DISTANCE (((TILE_SIZE_DIM0)*17+17) - (STENCIL_OFFSET))
#define BURST_WIDTH (512)
#define  INPUT_PIXEL_WIDTH (sizeof(uint16_t)*8)
#define OUTPUT_PIXEL_WIDTH (sizeof(uint8_t )*8)

#define TILE_INDEX_DIM0(tile_index) ((tile_index)%(tile_num_dim0))
#define TILE_INDEX_DIM1(tile_index) ((tile_index)/(tile_num_dim0))
#define  INPUT_TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/( INPUT_PIXEL_WIDTH)))
#define OUTPUT_TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/(OUTPUT_PIXEL_WIDTH)))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

#define min(a, b) ((a) < (b) ? (a) : (b) )
#define max(a, b) ((a) > (b) ? (a) : (b) )

void load(bool load_flag, uint16_t to[TILE_SIZE_DIM0*TILE_SIZE_DIM1], ap_uint<BURST_WIDTH>* from, int32_t tile_index)
{
    if(load_flag)
    {
        for(int i = 0; i < INPUT_TILE_SIZE_BURST; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp(from[i+tile_index*INPUT_TILE_SIZE_BURST]);
            for(int j = 0; j < BURST_WIDTH/INPUT_PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                to[i*BURST_WIDTH/INPUT_PIXEL_WIDTH+j] = tmp((j+1)*INPUT_PIXEL_WIDTH-1, j*INPUT_PIXEL_WIDTH);
            }
        }
    }
}

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, uint8_t from[3][TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index)
{
    if(store_flag)
    {
        for(int i = 0; i < OUTPUT_TILE_SIZE_BURST*3; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp;
            for(int j = 0; j < BURST_WIDTH/OUTPUT_PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                tmp((j+1)*OUTPUT_PIXEL_WIDTH-1, j*OUTPUT_PIXEL_WIDTH) = from[i/OUTPUT_TILE_SIZE_BURST][(i%OUTPUT_TILE_SIZE_BURST)*(BURST_WIDTH/OUTPUT_PIXEL_WIDTH)+j];
            }
            to[i+tile_index*OUTPUT_TILE_SIZE_BURST*3] = tmp;
        }
    }
}

void compute(bool compute_flag, uint8_t output[3][TILE_SIZE_DIM0*TILE_SIZE_DIM1], uint16_t input[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int16_t var_matrix[UNROLL_FACTOR][12], uint8_t var_curve[UNROLL_FACTOR*3][1024], int32_t tile_index, int32_t tile_num_dim0, int32_t var_processed_extent_0, int32_t var_processed_extent_1, int32_t var_processed_min_0, int32_t var_processed_min_1)
{
    if(compute_flag)
    {
        int32_t tile_index_dim0 = TILE_INDEX_DIM0(tile_index);
        int32_t tile_index_dim1 = TILE_INDEX_DIM1(tile_index);

        uint16_t stencil_buf[STENCIL_DISTANCE+UNROLL_FACTOR];
#pragma HLS array_partition variable=stencil_buf complete

        // produce processed
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
                    int32_t output_index = (input_index-STENCIL_DISTANCE/UNROLL_FACTOR)*UNROLL_FACTOR+unroll_index-STENCIL_DISTANCE%UNROLL_FACTOR - STENCIL_OFFSET;
                    if(output_index >= 0 && output_index < TILE_SIZE_DIM0*TILE_SIZE_DIM1)
                    {
                        int32_t i = output_index%TILE_SIZE_DIM0;
                        int32_t j = output_index/TILE_SIZE_DIM0;
                        int32_t q = Q(tile_index_dim1, j);
                        int32_t p = P(tile_index_dim0, i);

                        if(p >= var_processed_min_0 &&
                           q >= var_processed_min_1 &&
                           p < var_processed_min_0 + var_processed_extent_0 &&
                           q < var_processed_min_1 + var_processed_extent_1)
                        {
                            uint16_t var_651_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+16-STENCIL_OFFSET];
                            uint16_t var_651_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+15-STENCIL_OFFSET];
                            uint16_t var_651_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 7+15-STENCIL_OFFSET];
                            uint16_t var_651_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 7+16-STENCIL_OFFSET];

                            uint16_t var_740_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+18-STENCIL_OFFSET];
                            uint16_t var_740_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+17-STENCIL_OFFSET];
                            uint16_t var_740_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 7+17-STENCIL_OFFSET];
                            uint16_t var_740_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 7+18-STENCIL_OFFSET];

                            uint16_t var_470_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+15-STENCIL_OFFSET];
                            uint16_t var_470_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+14-STENCIL_OFFSET];
                            uint16_t var_470_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+14-STENCIL_OFFSET];
                            uint16_t var_470_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+15-STENCIL_OFFSET];

                            uint16_t var_700_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+16-STENCIL_OFFSET];
                            uint16_t var_700_01 = var_470_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+15-STENCIL_OFFSET];
                            uint16_t var_700_11 = var_651_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+15-STENCIL_OFFSET];
                            uint16_t var_700_10 = var_651_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+16-STENCIL_OFFSET];

                            uint16_t var_411_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+17-STENCIL_OFFSET];
                            uint16_t var_411_01 = var_700_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+16-STENCIL_OFFSET];
                            uint16_t var_411_11 = var_651_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+16-STENCIL_OFFSET];
                            uint16_t var_411_10 = var_740_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+17-STENCIL_OFFSET];

                            uint16_t var_710_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+18-STENCIL_OFFSET];
                            uint16_t var_710_01 = var_411_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+17-STENCIL_OFFSET];
                            uint16_t var_710_11 = var_740_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+17-STENCIL_OFFSET];
                            uint16_t var_710_10 = var_740_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+18-STENCIL_OFFSET];

                            uint16_t var_730_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+19-STENCIL_OFFSET];
                            uint16_t var_730_01 = var_710_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+18-STENCIL_OFFSET];
                            uint16_t var_730_11 = var_740_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+18-STENCIL_OFFSET];
                            uint16_t var_730_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 8+19-STENCIL_OFFSET];

                            uint16_t var_386_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+14-STENCIL_OFFSET];
                            uint16_t var_386_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+13-STENCIL_OFFSET];
                            uint16_t var_386_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+13-STENCIL_OFFSET];
                            uint16_t var_386_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+14-STENCIL_OFFSET];

                            uint16_t var_569_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+15-STENCIL_OFFSET];
                            uint16_t var_569_01 = var_386_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+14-STENCIL_OFFSET];
                            uint16_t var_569_11 = var_386_10;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+14-STENCIL_OFFSET];
                            uint16_t var_569_10 = var_470_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+15-STENCIL_OFFSET];

                            uint16_t var_311_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+16-STENCIL_OFFSET];
                            uint16_t var_311_01 = var_569_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+15-STENCIL_OFFSET];
                            uint16_t var_311_11 = var_470_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+15-STENCIL_OFFSET];
                            uint16_t var_311_10 = var_700_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+16-STENCIL_OFFSET];

                            uint16_t var_579_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+17-STENCIL_OFFSET];
                            uint16_t var_579_01 = var_311_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+16-STENCIL_OFFSET];
                            uint16_t var_579_11 = var_700_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+16-STENCIL_OFFSET];
                            uint16_t var_579_10 = var_411_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+17-STENCIL_OFFSET];

                            uint16_t var_371_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+18-STENCIL_OFFSET];
                            uint16_t var_371_01 = var_579_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+17-STENCIL_OFFSET];
                            uint16_t var_371_11 = var_411_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+17-STENCIL_OFFSET];
                            uint16_t var_371_10 = var_710_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+18-STENCIL_OFFSET];

                            uint16_t var_736_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+20-STENCIL_OFFSET];
                            uint16_t var_736_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+19-STENCIL_OFFSET];
                            uint16_t var_736_11 = var_730_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+19-STENCIL_OFFSET];
                            uint16_t var_736_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0* 9+20-STENCIL_OFFSET];

                            uint16_t var_466_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+13-STENCIL_OFFSET];
                            uint16_t var_466_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+12-STENCIL_OFFSET];
                            uint16_t var_466_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+12-STENCIL_OFFSET];
                            uint16_t var_466_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+13-STENCIL_OFFSET];

                            uint16_t var_696_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+14-STENCIL_OFFSET];
                            uint16_t var_696_01 = var_466_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+13-STENCIL_OFFSET];
                            uint16_t var_696_11 = var_386_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+13-STENCIL_OFFSET];
                            uint16_t var_696_10 = var_386_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+14-STENCIL_OFFSET];

                            uint16_t var_395_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+15-STENCIL_OFFSET];
                            uint16_t var_395_01 = var_696_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+14-STENCIL_OFFSET];
                            uint16_t var_395_11 = var_386_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+14-STENCIL_OFFSET];
                            uint16_t var_395_10 = var_569_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+15-STENCIL_OFFSET];

                            uint16_t var_344_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+16-STENCIL_OFFSET];
                            uint16_t var_344_01 = var_395_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+15-STENCIL_OFFSET];
                            uint16_t var_344_11 = var_569_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+15-STENCIL_OFFSET];
                            uint16_t var_344_10 = var_311_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+16-STENCIL_OFFSET];

                            uint16_t var_359_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+17-STENCIL_OFFSET];
                            uint16_t var_359_01 = var_344_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+16-STENCIL_OFFSET];
                            uint16_t var_359_11 = var_311_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+16-STENCIL_OFFSET];
                            uint16_t var_359_10 = var_579_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+17-STENCIL_OFFSET];

                            uint16_t var_377_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+18-STENCIL_OFFSET];
                            uint16_t var_377_01 = var_359_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+17-STENCIL_OFFSET];
                            uint16_t var_377_11 = var_579_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+17-STENCIL_OFFSET];
                            uint16_t var_377_10 = var_371_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+18-STENCIL_OFFSET];

                            uint16_t var_380_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+19-STENCIL_OFFSET];
                            uint16_t var_380_01 = var_377_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+18-STENCIL_OFFSET];
                            uint16_t var_380_11 = var_371_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+18-STENCIL_OFFSET];
                            uint16_t var_380_10 = var_736_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+19-STENCIL_OFFSET];

                            uint16_t var_706_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+20-STENCIL_OFFSET];
                            uint16_t var_706_01 = var_380_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+19-STENCIL_OFFSET];
                            uint16_t var_706_11 = var_736_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+19-STENCIL_OFFSET];
                            uint16_t var_706_10 = var_736_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+20-STENCIL_OFFSET];

                            uint16_t var_726_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+21-STENCIL_OFFSET];
                            uint16_t var_726_01 = var_706_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+20-STENCIL_OFFSET];
                            uint16_t var_726_11 = var_736_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*10+20-STENCIL_OFFSET];
                            uint16_t var_726_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0*10+21-STENCIL_OFFSET];

                            uint16_t var_459_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+12-STENCIL_OFFSET];
                            uint16_t var_459_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+11-STENCIL_OFFSET];
                            uint16_t var_459_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+11-STENCIL_OFFSET];
                            uint16_t var_459_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0*11+12-STENCIL_OFFSET];

                            uint16_t var_565_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+13-STENCIL_OFFSET];
                            uint16_t var_565_01 = var_459_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+12-STENCIL_OFFSET];
                            uint16_t var_565_11 = var_466_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+12-STENCIL_OFFSET];
                            uint16_t var_565_10 = var_466_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+13-STENCIL_OFFSET];

                            uint16_t var_301_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+14-STENCIL_OFFSET];
                            uint16_t var_301_01 = var_565_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+13-STENCIL_OFFSET];
                            uint16_t var_301_11 = var_466_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+13-STENCIL_OFFSET];
                            uint16_t var_301_10 = var_696_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+14-STENCIL_OFFSET];

                            uint16_t var_324_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+15-STENCIL_OFFSET];
                            uint16_t var_324_01 = var_301_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+14-STENCIL_OFFSET];
                            uint16_t var_324_11 = var_696_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+14-STENCIL_OFFSET];
                            uint16_t var_324_10 = var_395_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+15-STENCIL_OFFSET];

                            uint16_t var_321_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+16-STENCIL_OFFSET];
                            uint16_t var_321_01 = var_324_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+15-STENCIL_OFFSET];
                            uint16_t var_321_11 = var_395_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+15-STENCIL_OFFSET];
                            uint16_t var_321_10 = var_344_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+16-STENCIL_OFFSET];

                            uint16_t var_330_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+17-STENCIL_OFFSET];
                            uint16_t var_330_01 = var_321_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+16-STENCIL_OFFSET];
                            uint16_t var_330_11 = var_344_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+16-STENCIL_OFFSET];
                            uint16_t var_330_10 = var_359_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+17-STENCIL_OFFSET];

                            uint16_t var_304_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+18-STENCIL_OFFSET];
                            uint16_t var_304_01 = var_330_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+17-STENCIL_OFFSET];
                            uint16_t var_304_11 = var_359_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+17-STENCIL_OFFSET];
                            uint16_t var_304_10 = var_377_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+18-STENCIL_OFFSET];

                            uint16_t var_575_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+19-STENCIL_OFFSET];
                            uint16_t var_575_01 = var_304_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+18-STENCIL_OFFSET];
                            uint16_t var_575_11 = var_377_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+18-STENCIL_OFFSET];
                            uint16_t var_575_10 = var_380_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+19-STENCIL_OFFSET];

                            uint16_t var_398_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+20-STENCIL_OFFSET];
                            uint16_t var_398_01 = var_575_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+19-STENCIL_OFFSET];
                            uint16_t var_398_11 = var_380_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+19-STENCIL_OFFSET];
                            uint16_t var_398_10 = var_706_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*11+20-STENCIL_OFFSET];

                            uint16_t var_476_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+13-STENCIL_OFFSET];
                            uint16_t var_476_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+12-STENCIL_OFFSET];
                            uint16_t var_476_11 = var_459_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+12-STENCIL_OFFSET];
                            uint16_t var_476_10 = var_565_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+13-STENCIL_OFFSET];

                            uint16_t var_686_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+14-STENCIL_OFFSET];
                            uint16_t var_686_01 = var_476_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+13-STENCIL_OFFSET];
                            uint16_t var_686_11 = var_565_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+13-STENCIL_OFFSET];
                            uint16_t var_686_10 = var_301_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+14-STENCIL_OFFSET];

                            uint16_t var_353_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+15-STENCIL_OFFSET];
                            uint16_t var_353_01 = var_686_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+14-STENCIL_OFFSET];
                            uint16_t var_353_11 = var_301_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+14-STENCIL_OFFSET];
                            uint16_t var_353_10 = var_324_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+15-STENCIL_OFFSET];

                            uint16_t var_350_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+16-STENCIL_OFFSET];
                            uint16_t var_350_01 = var_353_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+15-STENCIL_OFFSET];
                            uint16_t var_350_11 = var_324_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+15-STENCIL_OFFSET];
                            uint16_t var_350_10 = var_321_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+16-STENCIL_OFFSET];

                            uint16_t var_366_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+17-STENCIL_OFFSET];
                            uint16_t var_366_01 = var_350_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+16-STENCIL_OFFSET];
                            uint16_t var_366_11 = var_321_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+16-STENCIL_OFFSET];
                            uint16_t var_366_10 = var_330_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+17-STENCIL_OFFSET];

                            uint16_t var_337_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+18-STENCIL_OFFSET];
                            uint16_t var_337_01 = var_366_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+17-STENCIL_OFFSET];
                            uint16_t var_337_11 = var_330_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+17-STENCIL_OFFSET];
                            uint16_t var_337_10 = var_304_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+18-STENCIL_OFFSET];

                            uint16_t var_356_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+19-STENCIL_OFFSET];
                            uint16_t var_356_01 = var_337_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+18-STENCIL_OFFSET];
                            uint16_t var_356_11 = var_304_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+18-STENCIL_OFFSET];
                            uint16_t var_356_10 = var_575_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+19-STENCIL_OFFSET];

                            uint16_t var_716_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+20-STENCIL_OFFSET];
                            uint16_t var_716_01 = var_356_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+19-STENCIL_OFFSET];
                            uint16_t var_716_11 = var_575_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+19-STENCIL_OFFSET];
                            uint16_t var_716_10 = var_398_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+20-STENCIL_OFFSET];

                            uint16_t var_657_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+21-STENCIL_OFFSET];
                            uint16_t var_657_01 = var_716_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+20-STENCIL_OFFSET];
                            uint16_t var_657_11 = var_398_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*12+20-STENCIL_OFFSET];
                            uint16_t var_657_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0*12+21-STENCIL_OFFSET];

                            uint16_t var_605_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+12-STENCIL_OFFSET];
                            uint16_t var_605_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+11-STENCIL_OFFSET];
                            uint16_t var_605_11 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+11-STENCIL_OFFSET];
                            uint16_t var_605_10 = stencil_buf[unroll_index+TILE_SIZE_DIM0*13+12-STENCIL_OFFSET];

                            uint16_t var_585_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+13-STENCIL_OFFSET];
                            uint16_t var_585_01 = var_605_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+12-STENCIL_OFFSET];
                            uint16_t var_585_11 = var_476_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+12-STENCIL_OFFSET];
                            uint16_t var_585_10 = var_476_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+13-STENCIL_OFFSET];

                            uint16_t var_450_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+14-STENCIL_OFFSET];
                            uint16_t var_450_01 = var_585_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+13-STENCIL_OFFSET];
                            uint16_t var_450_11 = var_476_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+13-STENCIL_OFFSET];
                            uint16_t var_450_10 = var_686_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+14-STENCIL_OFFSET];

                            uint16_t var_453_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+15-STENCIL_OFFSET];
                            uint16_t var_453_01 = var_450_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+14-STENCIL_OFFSET];
                            uint16_t var_453_11 = var_686_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+14-STENCIL_OFFSET];
                            uint16_t var_453_10 = var_353_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+15-STENCIL_OFFSET];

                            uint16_t var_318_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+16-STENCIL_OFFSET];
                            uint16_t var_318_01 = var_453_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+15-STENCIL_OFFSET];
                            uint16_t var_318_11 = var_353_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+15-STENCIL_OFFSET];
                            uint16_t var_318_10 = var_350_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+16-STENCIL_OFFSET];

                            uint16_t var_327_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+17-STENCIL_OFFSET];
                            uint16_t var_327_01 = var_318_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+16-STENCIL_OFFSET];
                            uint16_t var_327_11 = var_350_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+16-STENCIL_OFFSET];
                            uint16_t var_327_10 = var_366_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+17-STENCIL_OFFSET];

                            uint16_t var_374_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+18-STENCIL_OFFSET];
                            uint16_t var_374_01 = var_327_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+17-STENCIL_OFFSET];
                            uint16_t var_374_11 = var_366_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+17-STENCIL_OFFSET];
                            uint16_t var_374_10 = var_337_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+18-STENCIL_OFFSET];

                            uint16_t var_595_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+19-STENCIL_OFFSET];
                            uint16_t var_595_01 = var_374_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+18-STENCIL_OFFSET];
                            uint16_t var_595_11 = var_337_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+18-STENCIL_OFFSET];
                            uint16_t var_595_10 = var_356_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+19-STENCIL_OFFSET];

                            uint16_t var_521_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*14+20-STENCIL_OFFSET];
                            uint16_t var_521_01 = var_595_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+19-STENCIL_OFFSET];
                            uint16_t var_521_11 = var_356_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+19-STENCIL_OFFSET];
                            uint16_t var_521_10 = var_716_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*13+20-STENCIL_OFFSET];

                            uint16_t var_615_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+13-STENCIL_OFFSET];
                            uint16_t var_615_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+12-STENCIL_OFFSET];
                            uint16_t var_615_11 = var_605_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+12-STENCIL_OFFSET];
                            uint16_t var_615_10 = var_585_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+13-STENCIL_OFFSET];

                            uint16_t var_456_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+15-STENCIL_OFFSET];
                            uint16_t var_456_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+14-STENCIL_OFFSET];
                            uint16_t var_456_11 = var_450_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+14-STENCIL_OFFSET];
                            uint16_t var_456_10 = var_453_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+15-STENCIL_OFFSET];

                            uint16_t var_690_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+16-STENCIL_OFFSET];
                            uint16_t var_690_01 = var_456_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+15-STENCIL_OFFSET];
                            uint16_t var_690_11 = var_453_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+15-STENCIL_OFFSET];
                            uint16_t var_690_10 = var_318_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+16-STENCIL_OFFSET];

                            uint16_t var_363_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+17-STENCIL_OFFSET];
                            uint16_t var_363_01 = var_690_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+16-STENCIL_OFFSET];
                            uint16_t var_363_11 = var_318_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+16-STENCIL_OFFSET];
                            uint16_t var_363_10 = var_327_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+17-STENCIL_OFFSET];

                            uint16_t var_720_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+18-STENCIL_OFFSET];
                            uint16_t var_720_01 = var_363_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+17-STENCIL_OFFSET];
                            uint16_t var_720_11 = var_327_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+17-STENCIL_OFFSET];
                            uint16_t var_720_10 = var_374_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+18-STENCIL_OFFSET];

                            uint16_t var_383_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*15+19-STENCIL_OFFSET];
                            uint16_t var_383_01 = var_720_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+18-STENCIL_OFFSET];
                            uint16_t var_383_11 = var_374_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+18-STENCIL_OFFSET];
                            uint16_t var_383_10 = var_595_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*14+19-STENCIL_OFFSET];

                            uint16_t var_609_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+14-STENCIL_OFFSET];
                            uint16_t var_609_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+13-STENCIL_OFFSET];
                            uint16_t var_609_11 = var_615_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+13-STENCIL_OFFSET];
                            uint16_t var_609_10 = var_456_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+14-STENCIL_OFFSET];

                            uint16_t var_589_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+15-STENCIL_OFFSET];
                            uint16_t var_589_01 = var_609_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+14-STENCIL_OFFSET];
                            uint16_t var_589_11 = var_456_01;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+14-STENCIL_OFFSET];
                            uint16_t var_589_10 = var_456_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+15-STENCIL_OFFSET];

                            uint16_t var_491_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+16-STENCIL_OFFSET];
                            uint16_t var_491_01 = var_589_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+15-STENCIL_OFFSET];
                            uint16_t var_491_11 = var_456_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+15-STENCIL_OFFSET];
                            uint16_t var_491_10 = var_690_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+16-STENCIL_OFFSET];

                            uint16_t var_599_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+17-STENCIL_OFFSET];
                            uint16_t var_599_01 = var_491_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+16-STENCIL_OFFSET];
                            uint16_t var_599_11 = var_690_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+16-STENCIL_OFFSET];
                            uint16_t var_599_10 = var_363_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+17-STENCIL_OFFSET];

                            uint16_t var_525_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*16+18-STENCIL_OFFSET];
                            uint16_t var_525_01 = var_599_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+17-STENCIL_OFFSET];
                            uint16_t var_525_11 = var_363_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+17-STENCIL_OFFSET];
                            uint16_t var_525_10 = var_720_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*15+18-STENCIL_OFFSET];

                            uint16_t var_619_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*17+15-STENCIL_OFFSET];
                            uint16_t var_619_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*17+14-STENCIL_OFFSET];
                            uint16_t var_619_11 = var_609_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+14-STENCIL_OFFSET];
                            uint16_t var_619_10 = var_589_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+15-STENCIL_OFFSET];

                            uint16_t var_533_00 = stencil_buf[unroll_index+TILE_SIZE_DIM0*17+17-STENCIL_OFFSET];
                            uint16_t var_533_01 = stencil_buf[unroll_index+TILE_SIZE_DIM0*17+16-STENCIL_OFFSET];
                            uint16_t var_533_11 = var_491_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+16-STENCIL_OFFSET];
                            uint16_t var_533_10 = var_599_00;//stencil_buf[unroll_index+TILE_SIZE_DIM0*16+17-STENCIL_OFFSET];

                            uint16_t var_651 =  (q&1)==0 ? ((p&1)==0? var_651_00 : var_651_01 ) : ((p&1)==0 ? var_651_10 : var_651_11);
                            uint16_t var_740 =  (q&1)==0 ? ((p&1)==0? var_740_00 : var_740_01 ) : ((p&1)==0 ? var_740_10 : var_740_11);
                            uint16_t var_470 =  (q&1)==0 ? ((p&1)==0? var_470_00 : var_470_01 ) : ((p&1)==0 ? var_470_10 : var_470_11);
                            uint16_t var_700 =  (q&1)==0 ? ((p&1)==0? var_700_00 : var_700_01 ) : ((p&1)==0 ? var_700_10 : var_700_11);
                            uint16_t var_411 =  (q&1)==0 ? ((p&1)==0? var_411_00 : var_411_01 ) : ((p&1)==0 ? var_411_10 : var_411_11);
                            uint16_t var_710 =  (q&1)==0 ? ((p&1)==0? var_710_00 : var_710_01 ) : ((p&1)==0 ? var_710_10 : var_710_11);
                            uint16_t var_730 =  (q&1)==0 ? ((p&1)==0? var_730_00 : var_730_01 ) : ((p&1)==0 ? var_730_10 : var_730_11);
                            uint16_t var_386 =  (q&1)==0 ? ((p&1)==0? var_386_00 : var_386_01 ) : ((p&1)==0 ? var_386_10 : var_386_11);
                            uint16_t var_569 =  (q&1)==0 ? ((p&1)==0? var_569_00 : var_569_01 ) : ((p&1)==0 ? var_569_10 : var_569_11);
                            uint16_t var_311 =  (q&1)==0 ? ((p&1)==0? var_311_00 : var_311_01 ) : ((p&1)==0 ? var_311_10 : var_311_11);
                            uint16_t var_579 =  (q&1)==0 ? ((p&1)==0? var_579_00 : var_579_01 ) : ((p&1)==0 ? var_579_10 : var_579_11);
                            uint16_t var_371 =  (q&1)==0 ? ((p&1)==0? var_371_00 : var_371_01 ) : ((p&1)==0 ? var_371_10 : var_371_11);
                            uint16_t var_736 =  (q&1)==0 ? ((p&1)==0? var_736_00 : var_736_01 ) : ((p&1)==0 ? var_736_10 : var_736_11);
                            uint16_t var_466 =  (q&1)==0 ? ((p&1)==0? var_466_00 : var_466_01 ) : ((p&1)==0 ? var_466_10 : var_466_11);
                            uint16_t var_696 =  (q&1)==0 ? ((p&1)==0? var_696_00 : var_696_01 ) : ((p&1)==0 ? var_696_10 : var_696_11);
                            uint16_t var_395 =  (q&1)==0 ? ((p&1)==0? var_395_00 : var_395_01 ) : ((p&1)==0 ? var_395_10 : var_395_11);
                            uint16_t var_344 =  (q&1)==0 ? ((p&1)==0? var_344_00 : var_344_01 ) : ((p&1)==0 ? var_344_10 : var_344_11);
                            uint16_t var_359 =  (q&1)==0 ? ((p&1)==0? var_359_00 : var_359_01 ) : ((p&1)==0 ? var_359_10 : var_359_11);
                            uint16_t var_377 =  (q&1)==0 ? ((p&1)==0? var_377_00 : var_377_01 ) : ((p&1)==0 ? var_377_10 : var_377_11);
                            uint16_t var_380 =  (q&1)==0 ? ((p&1)==0? var_380_00 : var_380_01 ) : ((p&1)==0 ? var_380_10 : var_380_11);
                            uint16_t var_706 =  (q&1)==0 ? ((p&1)==0? var_706_00 : var_706_01 ) : ((p&1)==0 ? var_706_10 : var_706_11);
                            uint16_t var_726 =  (q&1)==0 ? ((p&1)==0? var_726_00 : var_726_01 ) : ((p&1)==0 ? var_726_10 : var_726_11);
                            uint16_t var_459 =  (q&1)==0 ? ((p&1)==0? var_459_00 : var_459_01 ) : ((p&1)==0 ? var_459_10 : var_459_11);
                            uint16_t var_565 =  (q&1)==0 ? ((p&1)==0? var_565_00 : var_565_01 ) : ((p&1)==0 ? var_565_10 : var_565_11);
                            uint16_t var_301 =  (q&1)==0 ? ((p&1)==0? var_301_00 : var_301_01 ) : ((p&1)==0 ? var_301_10 : var_301_11);
                            uint16_t var_324 =  (q&1)==0 ? ((p&1)==0? var_324_00 : var_324_01 ) : ((p&1)==0 ? var_324_10 : var_324_11);
                            uint16_t var_321 =  (q&1)==0 ? ((p&1)==0? var_321_00 : var_321_01 ) : ((p&1)==0 ? var_321_10 : var_321_11);
                            uint16_t var_330 =  (q&1)==0 ? ((p&1)==0? var_330_00 : var_330_01 ) : ((p&1)==0 ? var_330_10 : var_330_11);
                            uint16_t var_304 =  (q&1)==0 ? ((p&1)==0? var_304_00 : var_304_01 ) : ((p&1)==0 ? var_304_10 : var_304_11);
                            uint16_t var_575 =  (q&1)==0 ? ((p&1)==0? var_575_00 : var_575_01 ) : ((p&1)==0 ? var_575_10 : var_575_11);
                            uint16_t var_398 =  (q&1)==0 ? ((p&1)==0? var_398_00 : var_398_01 ) : ((p&1)==0 ? var_398_10 : var_398_11);
                            uint16_t var_476 =  (q&1)==0 ? ((p&1)==0? var_476_00 : var_476_01 ) : ((p&1)==0 ? var_476_10 : var_476_11);
                            uint16_t var_686 =  (q&1)==0 ? ((p&1)==0? var_686_00 : var_686_01 ) : ((p&1)==0 ? var_686_10 : var_686_11);
                            uint16_t var_353 =  (q&1)==0 ? ((p&1)==0? var_353_00 : var_353_01 ) : ((p&1)==0 ? var_353_10 : var_353_11);
                            uint16_t var_350 =  (q&1)==0 ? ((p&1)==0? var_350_00 : var_350_01 ) : ((p&1)==0 ? var_350_10 : var_350_11);
                            uint16_t var_366 =  (q&1)==0 ? ((p&1)==0? var_366_00 : var_366_01 ) : ((p&1)==0 ? var_366_10 : var_366_11);
                            uint16_t var_337 =  (q&1)==0 ? ((p&1)==0? var_337_00 : var_337_01 ) : ((p&1)==0 ? var_337_10 : var_337_11);
                            uint16_t var_356 =  (q&1)==0 ? ((p&1)==0? var_356_00 : var_356_01 ) : ((p&1)==0 ? var_356_10 : var_356_11);
                            uint16_t var_716 =  (q&1)==0 ? ((p&1)==0? var_716_00 : var_716_01 ) : ((p&1)==0 ? var_716_10 : var_716_11);
                            uint16_t var_657 =  (q&1)==0 ? ((p&1)==0? var_657_00 : var_657_01 ) : ((p&1)==0 ? var_657_10 : var_657_11);
                            uint16_t var_605 =  (q&1)==0 ? ((p&1)==0? var_605_00 : var_605_01 ) : ((p&1)==0 ? var_605_10 : var_605_11);
                            uint16_t var_585 =  (q&1)==0 ? ((p&1)==0? var_585_00 : var_585_01 ) : ((p&1)==0 ? var_585_10 : var_585_11);
                            uint16_t var_450 =  (q&1)==0 ? ((p&1)==0? var_450_00 : var_450_01 ) : ((p&1)==0 ? var_450_10 : var_450_11);
                            uint16_t var_453 =  (q&1)==0 ? ((p&1)==0? var_453_00 : var_453_01 ) : ((p&1)==0 ? var_453_10 : var_453_11);
                            uint16_t var_318 =  (q&1)==0 ? ((p&1)==0? var_318_00 : var_318_01 ) : ((p&1)==0 ? var_318_10 : var_318_11);
                            uint16_t var_327 =  (q&1)==0 ? ((p&1)==0? var_327_00 : var_327_01 ) : ((p&1)==0 ? var_327_10 : var_327_11);
                            uint16_t var_374 =  (q&1)==0 ? ((p&1)==0? var_374_00 : var_374_01 ) : ((p&1)==0 ? var_374_10 : var_374_11);
                            uint16_t var_595 =  (q&1)==0 ? ((p&1)==0? var_595_00 : var_595_01 ) : ((p&1)==0 ? var_595_10 : var_595_11);
                            uint16_t var_521 =  (q&1)==0 ? ((p&1)==0? var_521_00 : var_521_01 ) : ((p&1)==0 ? var_521_10 : var_521_11);
                            uint16_t var_615 =  (q&1)==0 ? ((p&1)==0? var_615_00 : var_615_01 ) : ((p&1)==0 ? var_615_10 : var_615_11);
                            uint16_t var_456 =  (q&1)==0 ? ((p&1)==0? var_456_00 : var_456_01 ) : ((p&1)==0 ? var_456_10 : var_456_11);
                            uint16_t var_690 =  (q&1)==0 ? ((p&1)==0? var_690_00 : var_690_01 ) : ((p&1)==0 ? var_690_10 : var_690_11);
                            uint16_t var_363 =  (q&1)==0 ? ((p&1)==0? var_363_00 : var_363_01 ) : ((p&1)==0 ? var_363_10 : var_363_11);
                            uint16_t var_720 =  (q&1)==0 ? ((p&1)==0? var_720_00 : var_720_01 ) : ((p&1)==0 ? var_720_10 : var_720_11);
                            uint16_t var_383 =  (q&1)==0 ? ((p&1)==0? var_383_00 : var_383_01 ) : ((p&1)==0 ? var_383_10 : var_383_11);
                            uint16_t var_609 =  (q&1)==0 ? ((p&1)==0? var_609_00 : var_609_01 ) : ((p&1)==0 ? var_609_10 : var_609_11);
                            uint16_t var_589 =  (q&1)==0 ? ((p&1)==0? var_589_00 : var_589_01 ) : ((p&1)==0 ? var_589_10 : var_589_11);
                            uint16_t var_491 =  (q&1)==0 ? ((p&1)==0? var_491_00 : var_491_01 ) : ((p&1)==0 ? var_491_10 : var_491_11);
                            uint16_t var_599 =  (q&1)==0 ? ((p&1)==0? var_599_00 : var_599_01 ) : ((p&1)==0 ? var_599_10 : var_599_11);
                            uint16_t var_525 =  (q&1)==0 ? ((p&1)==0? var_525_00 : var_525_01 ) : ((p&1)==0 ? var_525_10 : var_525_11);
                            uint16_t var_619 =  (q&1)==0 ? ((p&1)==0? var_619_00 : var_619_01 ) : ((p&1)==0 ? var_619_10 : var_619_11);
                            uint16_t var_533 =  (q&1)==0 ? ((p&1)==0? var_533_00 : var_533_01 ) : ((p&1)==0 ? var_533_10 : var_533_11);

                            //int32_t var_292 = var_processed_s0_v0 >> 1;
                            //int32_t var_293 = var_292 * 2;
                            //int32_t var_294 = var_processed_s0_y >> 1;
                            //int32_t var_295 = var_294 * 2;
                            //int32_t var_296 = var_295 + 12;
                            //int32_t var_297 = var_296 * var_input_stride_1;
                            //int32_t var_298 = var_293 + var_297;
                            //int32_t var_299 = var_298 - var_291;
                            //int32_t var_300 = var_299 + 14;
                            //uint16_t var_301 = var_input[var_300];//[12][14]
                            int16_t var_302 = (int16_t)(var_301);
                            //int32_t var_303 = var_299 + 18;
                            //uint16_t var_304 = var_input[var_303];//[12][18]
                            int16_t var_305 = (int16_t)(var_304);
                            //int32_t var_306 = var_295 + 10;
                            //int32_t var_307 = var_306 * var_input_stride_1;
                            //int32_t var_308 = var_293 + var_307;
                            //int32_t var_309 = var_308 - var_291;
                            //int32_t var_310 = var_309 + 16;
                            //uint16_t var_311 = var_input[var_310];//[10][16]
                            int16_t var_312 = (int16_t)(var_311);
                            //int32_t var_313 = var_295 + 14;
                            //int32_t var_314 = var_313 * var_input_stride_1;
                            //int32_t var_315 = var_293 + var_314;
                            //int32_t var_316 = var_315 - var_291;
                            //int32_t var_317 = var_316 + 16;
                            //uint16_t var_318 = var_input[var_317];//[14][16]
                            int16_t var_319 = (int16_t)(var_318);
                            //int32_t var_320 = var_299 + 16;
                            //uint16_t var_321 = var_input[var_320];//[12][16]
                            int16_t var_322 = (int16_t)(var_321);
                            //int32_t var_323 = var_299 + 15;
                            //uint16_t var_324 = var_input[var_323];//[12][15]
                            int16_t var_325 = (int16_t)(var_324);
                            //int32_t var_326 = var_316 + 17;
                            //uint16_t var_327 = var_input[var_326];//[14][17]
                            int16_t var_328 = (int16_t)(var_327);
                            //int32_t var_329 = var_299 + 17;
                            //uint16_t var_330 = var_input[var_329];//[12][17]
                            int16_t var_331 = (int16_t)(var_330);
                            //int32_t var_332 = var_295 + 13;
                            //nt32_t var_333 = var_332 * var_input_stride_1;
                            //nt32_t var_334 = var_293 + var_333;
                            //nt32_t var_335 = var_334 - var_291;
                            //nt32_t var_336 = var_335 + 18;
                            //int16_t var_337 = var_input[var_336];//[13][18]
                            int16_t var_338 = (int16_t)(var_337);
                            //int32_t var_339 = var_295 + 11;
                            //nt32_t var_340 = var_339 * var_input_stride_1;
                            //nt32_t var_341 = var_293 + var_340;
                            //nt32_t var_342 = var_341 - var_291;
                            //nt32_t var_343 = var_342 + 16;
                            //int16_t var_344 = var_input[var_343];//[11][16]
                            int16_t var_345 = (int16_t)(var_344);
                            //int32_t var_346 = var_295 + 15;
                            //int32_t var_347 = var_346 * var_input_stride_1;
                            //int32_t var_348 = var_293 + var_347;
                            //int32_t var_349 = var_335 + 16;
                            //uint16_t var_350 = var_input[var_349];//[13][16]
                            int16_t var_351 = (int16_t)(var_350);
                            //int32_t var_352 = var_335 + 15;
                            //uint16_t var_353 = var_input[var_352];//[13][15]
                            int16_t var_354 = (int16_t)(var_353);
                            //int32_t var_355 = var_335 + 19;
                            //uint16_t var_356 = var_input[var_355];//[13][19]
                            int16_t var_357 = (int16_t)(var_356);
                            //int32_t var_358 = var_342 + 17;
                            //uint16_t var_359 = var_input[var_358];//[11][17]
                            int16_t var_360 = (int16_t)(var_359);
                            //int32_t var_361 = var_348 - var_291;
                            //int32_t var_362 = var_361 + 17;
                            //uint16_t var_363 = var_input[var_362];//[15][17]
                            int16_t var_364 = (int16_t)(var_363);
                            //int32_t var_365 = var_335 + 17;
                            //uint16_t var_366 = var_input[var_365];//[13][17]
                            int16_t var_367 = (int16_t)(var_366);
                            int32_t var_368 = i & 1;
                            bool var_369 = var_368 == 0;
                            //int32_t var_370 = var_309 + 18;
                            //uint16_t var_371 = var_input[var_370];//[10][18]
                            int16_t var_372 = (int16_t)(var_371);
                            //int32_t var_373 = var_316 + 18;
                            //uint16_t var_374 = var_input[var_373];//[14][18]
                            int16_t var_375 = (int16_t)(var_374);
                            //int32_t var_376 = var_342 + 18;
                            //uint16_t var_377 = var_input[var_376];//[11][18]
                            int16_t var_378 = (int16_t)(var_377);
                            //int32_t var_379 = var_342 + 19;
                            //uint16_t var_380 = var_input[var_379];//[11][19]
                            int16_t var_381 = (int16_t)(var_380);
                            //int32_t var_382 = var_361 + 19;
                            //uint16_t var_383 = var_input[var_382];//[15][19]
                            int16_t var_384 = (int16_t)(var_383);
                            //int32_t var_385 = var_309 + 14;
                            //uint16_t var_386 = var_input[var_385];//[10][14]
                            int16_t var_387 = (int16_t)(var_386);
                            //int32_t var_388 = var_295 + 8;
                            //int32_t var_389 = var_388 * var_input_stride_1;
                            //int32_t var_390 = var_293 + var_389;
                            //int32_t var_391 = var_295 + 9;
                            //int32_t var_392 = var_391 * var_input_stride_1;
                            //int32_t var_393 = var_293 + var_392;
                            //int32_t var_394 = var_342 + 15;
                            //uint16_t var_395 = var_input[var_394];//[11][15]
                            int16_t var_396 = (int16_t)(var_395);
                            //int32_t var_397 = var_299 + 20;
                            //uint16_t var_398 = var_input[var_397];//[12][20]
                            int16_t var_399 = (int16_t)(var_398);
                            int16_t var_400 = max(var_322, var_399);
                            int16_t var_401 = max(var_372, var_375);
                            int16_t var_402 = max(var_400, var_401);
                            int16_t var_403 = min(var_402, var_305);
                            int16_t var_404 = max(var_302, var_305);
                            int16_t var_405 = max(var_312, var_319);
                            int16_t var_406 = max(var_404, var_405);
                            int16_t var_407 = min(var_406, var_322);
                            int16_t var_408 = max(var_396, var_381);
                            //int32_t var_409 = var_393 - var_291;
                            //int32_t var_410 = var_409 + 17;
                            //uint16_t var_411 = var_input[var_410];//[9][17]
                            int16_t var_412 = (int16_t)(var_411);
                            int16_t var_413 = max(var_412, var_367);
                            int16_t var_414 = max(var_408, var_413);
                            int16_t var_415 = min(var_414, var_360);
                            int16_t var_416 = max(var_354, var_357);
                            int16_t var_417 = max(var_360, var_364);
                            int16_t var_418 = max(var_416, var_417);
                            int16_t var_419 = min(var_418, var_367);
                            int16_t var_420 = (int16_t)(0);
                            int16_t var_421 = max(var_403, var_420);
                            int32_t var_422 = (int32_t)(var_421);
                            int16_t var_423 = max(var_407, var_420);
                            int32_t var_424 = (int32_t)(var_423);
                            int32_t var_425 = var_422 + var_424;
                            int32_t var_426 = var_425 + 1;
                            int32_t var_427 = var_426 >> 1;
                            int16_t var_428 = (int16_t)(var_427);
                            int16_t var_429 = max(var_415, var_420);
                            int32_t var_430 = (int32_t)(var_429);
                            int16_t var_431 = max(var_419, var_420);
                            int32_t var_432 = (int32_t)(var_431);
                            int32_t var_433 = var_430 + var_432;
                            int32_t var_434 = var_433 + 1;
                            int32_t var_435 = var_434 >> 1;
                            int16_t var_436 = (int16_t)(var_435);
                            int16_t var_437 = var_423 - var_421;
                            int16_t var_438 = var_421 - var_423;
                            bool var_439 = var_421 < var_423;
                            int16_t var_440 = (int16_t)(var_439 ? var_437 : var_438);
                            uint16_t var_441 = var_440;
                            int16_t var_442 = var_431 - var_429;
                            int16_t var_443 = var_429 - var_431;
                            bool var_444 = var_429 < var_431;
                            int16_t var_445 = (int16_t)(var_444 ? var_442 : var_443);
                            uint16_t var_446 = var_445;
                            bool var_447 = var_441 < var_446;
                            int16_t var_448 = (int16_t)(var_447 ? var_428 : var_436);
                            //int32_t var_449 = var_316 + 14;
                            //uint16_t var_450 = var_input[var_449];//[14][14]
                            int16_t var_451 = (int16_t)(var_450);
                            //int32_t var_452 = var_316 + 15;
                            //uint16_t var_453 = var_input[var_452];//[14][15]
                            int16_t var_454 = (int16_t)(var_453);
                            //int32_t var_455 = var_361 + 15;
                            //uint16_t var_456 = var_input[var_455];//[15][15]
                            int16_t var_457 = (int16_t)(var_456);
                            //int32_t var_458 = var_299 + 12;
                            //uint16_t var_459 = var_input[var_458];//[12][12]
                            int16_t var_460 = (int16_t)(var_459);
                            int16_t var_461 = max(var_460, var_322);
                            int16_t var_462 = max(var_387, var_451);
                            int16_t var_463 = max(var_461, var_462);
                            int16_t var_464 = min(var_463, var_302);
                            //int32_t var_465 = var_342 + 13;
                            //uint16_t var_466 = var_input[var_465];//[11][13]
                            int16_t var_467 = (int16_t)(var_466);
                            int16_t var_468 = max(var_467, var_360);
                            //int32_t var_469 = var_409 + 15;
                            //uint16_t var_470 = var_input[var_469];//[9][15]
                            int16_t var_471 = (int16_t)(var_470);
                            int16_t var_472 = max(var_471, var_354);
                            int16_t var_473 = max(var_468, var_472);
                            int16_t var_474 = min(var_473, var_396);
                            //int32_t var_475 = var_335 + 13;
                            //uint16_t var_476 = var_input[var_475];//[13][13]
                            int16_t var_477 = (int16_t)(var_476);
                            int16_t var_478 = max(var_477, var_367);
                            int16_t var_479 = max(var_396, var_457);
                            int16_t var_480 = max(var_478, var_479);
                            int16_t var_481 = min(var_480, var_354);
                            //int32_t var_482 = var_295 + 16;
                            //int32_t var_483 = var_482 * var_input_stride_1;
                            //int32_t var_484 = var_293 + var_483;
                            //int32_t var_485 = var_295 + 17;
                            //int32_t var_486 = var_485 * var_input_stride_1;
                            //int32_t var_487 = var_293 + var_486;
                            int16_t var_488 = max(var_451, var_375);
                            //int32_t var_489 = var_484 - var_291;
                            //int32_t var_490 = var_489 + 16;
                            //uint16_t var_491 = var_input[var_490];//[16][16]
                            int16_t var_492 = (int16_t)(var_491);
                            int16_t var_493 = max(var_322, var_492);
                            int16_t var_494 = max(var_488, var_493);
                            int16_t var_495 = min(var_494, var_319);
                            int16_t var_496 = max(var_481, var_420);
                            int32_t var_497 = (int32_t)(var_496);
                            int32_t var_498 = var_497 + var_432;
                            int32_t var_499 = var_498 + 1;
                            int32_t var_500 = var_499 >> 1;
                            int16_t var_501 = (int16_t)(var_500);
                            int16_t var_502 = max(var_495, var_420);
                            int32_t var_503 = (int32_t)(var_502);
                            int32_t var_504 = var_503 + var_424;
                            int32_t var_505 = var_504 + 1;
                            int32_t var_506 = var_505 >> 1;
                            int16_t var_507 = (int16_t)(var_506);
                            int16_t var_508 = var_431 - var_496;
                            int16_t var_509 = var_496 - var_431;
                            bool var_510 = var_496 < var_431;
                            int16_t var_511 = (int16_t)(var_510 ? var_508 : var_509);
                            uint16_t var_512 = var_511;
                            int16_t var_513 = var_423 - var_502;
                            int16_t var_514 = var_502 - var_423;
                            bool var_515 = var_502 < var_423;
                            int16_t var_516 = (int16_t)(var_515 ? var_513 : var_514);
                            uint16_t var_517 = var_516;
                            bool var_518 = var_512 < var_517;
                            int16_t var_519 = (int16_t)(var_518 ? var_501 : var_507);
                            //int32_t var_520 = var_316 + 20;
                            //uint16_t var_521 = var_input[var_520];//[14][20]
                            int16_t var_522 = (int16_t)(var_521);
                            int16_t var_523 = max(var_319, var_522);
                            //int32_t var_524 = var_489 + 18;
                            //uint16_t var_525 = var_input[var_524];//[16][18]
                            int16_t var_526 = (int16_t)(var_525);
                            int16_t var_527 = max(var_305, var_526);
                            int16_t var_528 = max(var_523, var_527);
                            int16_t var_529 = min(var_528, var_375);
                            int16_t var_530 = max(var_457, var_384);
                            //int32_t var_531 = var_487 - var_291;
                            //int32_t var_532 = var_531 + 17;
                            //uint16_t var_533 = var_input[var_532];//[17][17]
                            int16_t var_534 = (int16_t)(var_533);
                            int16_t var_535 = max(var_367, var_534);
                            int16_t var_536 = max(var_530, var_535);
                            int16_t var_537 = min(var_536, var_364);
                            int32_t var_538 = j & 1;
                            bool var_539 = var_538 == 0;
                            int16_t var_540 = max(var_464, var_420);
                            int32_t var_541 = (int32_t)(var_540);
                            int32_t var_542 = var_424 + var_541;
                            int32_t var_543 = var_542 + 1;
                            int32_t var_544 = var_543 >> 1;
                            int16_t var_545 = (int16_t)(var_544);
                            int16_t var_546 = max(var_474, var_420);
                            int32_t var_547 = (int32_t)(var_546);
                            int32_t var_548 = var_547 + var_497;
                            int32_t var_549 = var_548 + 1;
                            int32_t var_550 = var_549 >> 1;
                            int16_t var_551 = (int16_t)(var_550);
                            int16_t var_552 = var_540 - var_423;
                            int16_t var_553 = var_423 - var_540;
                            bool var_554 = var_423 < var_540;
                            int16_t var_555 = (int16_t)(var_554 ? var_552 : var_553);
                            uint16_t var_556 = var_555;
                            int16_t var_557 = var_496 - var_546;
                            int16_t var_558 = var_546 - var_496;
                            bool var_559 = var_546 < var_496;
                            int16_t var_560 = (int16_t)(var_559 ? var_557 : var_558);
                            uint16_t var_561 = var_560;
                            bool var_562 = var_556 < var_561;
                            int16_t var_563 = (int16_t)(var_562 ? var_545 : var_551);
                            //int32_t var_564 = var_299 + 13;
                            //uint16_t var_565 = var_input[var_564];//[12][13]
                            int16_t var_566 = (int16_t)(var_565);
                            int16_t var_567 = max(var_566, var_331);
                            //int32_t var_568 = var_309 + 15;
                            //uint16_t var_569 = var_input[var_568];//[10][15]
                            int16_t var_570 = (int16_t)(var_569);
                            int16_t var_571 = max(var_570, var_454);
                            int16_t var_572 = max(var_567, var_571);
                            int16_t var_573 = min(var_572, var_325);
                            //int32_t var_574 = var_299 + 19;
                            //uint16_t var_575 = var_input[var_574];//[12][19]
                            int16_t var_576 = (int16_t)(var_575);
                            int16_t var_577 = max(var_325, var_576);
                            //int32_t var_578 = var_309 + 17;
                            //uint16_t var_579 = var_input[var_578];//[10][17]
                            int16_t var_580 = (int16_t)(var_579);
                            int16_t var_581 = max(var_580, var_328);
                            int16_t var_582 = max(var_577, var_581);
                            int16_t var_583 = min(var_582, var_331);
                            //int32_t var_584 = var_316 + 13;
                            //uint16_t var_585 = var_input[var_584];//[14][13]
                            int16_t var_586 = (int16_t)(var_585);
                            int16_t var_587 = max(var_586, var_328);
                            //int32_t var_588 = var_489 + 15;
                            //uint16_t var_589 = var_input[var_588];//[16][15]
                            int16_t var_590 = (int16_t)(var_589);
                            int16_t var_591 = max(var_325, var_590);
                            int16_t var_592 = max(var_587, var_591);
                            int16_t var_593 = min(var_592, var_454);
                            //int32_t var_594 = var_316 + 19;
                            //uint16_t var_595 = var_input[var_594];//[14][19]
                            int16_t var_596 = (int16_t)(var_595);
                            int16_t var_597 = max(var_454, var_596);
                            //int32_t var_598 = var_489 + 17;
                            //uint16_t var_599 = var_input[var_598];//[16][17]
                            int16_t var_600 = (int16_t)(var_599);
                            int16_t var_601 = max(var_331, var_600);
                            int16_t var_602 = max(var_597, var_601);
                            int16_t var_603 = min(var_602, var_328);
                            //int32_t var_604 = var_316 + 12;
                            //uint16_t var_605 = var_input[var_604];//[14][12]
                            int16_t var_606 = (int16_t)(var_605);
                            int16_t var_607 = max(var_606, var_319);
                            //int32_t var_608 = var_489 + 14;
                            //uint16_t var_609 = var_input[var_608];//[16][14]
                            int16_t var_610 = (int16_t)(var_609);
                            int16_t var_611 = max(var_302, var_610);
                            int16_t var_612 = max(var_607, var_611);
                            int16_t var_613 = min(var_612, var_451);
                            //int32_t var_614 = var_361 + 13;
                            //uint16_t var_615 = var_input[var_614];//[15][13]
                            int16_t var_616 = (int16_t)(var_615);
                            int16_t var_617 = max(var_616, var_364);
                            //int32_t var_618 = var_531 + 15;
                            //uint16_t var_619 = var_input[var_618];//[17][15]
                            int16_t var_620 = (int16_t)(var_619);
                            int16_t var_621 = max(var_354, var_620);
                            int16_t var_622 = max(var_617, var_621);
                            int16_t var_623 = min(var_622, var_457);
                            int16_t var_624 = max(var_529, var_420);
                            int32_t var_625 = (int32_t)(var_624);
                            int32_t var_626 = var_625 + var_503;
                            int32_t var_627 = var_626 + 1;
                            int32_t var_628 = var_627 >> 1;
                            int16_t var_629 = (int16_t)(var_628);
                            int16_t var_630 = max(var_537, var_420);
                            int32_t var_631 = (int32_t)(var_630);
                            int32_t var_632 = var_432 + var_631;
                            int32_t var_633 = var_632 + 1;
                            int32_t var_634 = var_633 >> 1;
                            int16_t var_635 = (int16_t)(var_634);
                            int16_t var_636 = var_502 - var_624;
                            int16_t var_637 = var_624 - var_502;
                            bool var_638 = var_624 < var_502;
                            int16_t var_639 = (int16_t)(var_638 ? var_636 : var_637);
                            uint16_t var_640 = var_639;
                            int16_t var_641 = var_630 - var_431;
                            int16_t var_642 = var_431 - var_630;
                            bool var_643 = var_431 < var_630;
                            int16_t var_644 = (int16_t)(var_643 ? var_641 : var_642);
                            uint16_t var_645 = var_644;
                            bool var_646 = var_640 < var_645;
                            int16_t var_647 = (int16_t)(var_646 ? var_629 : var_635);
                            int16_t var_648 = max(var_387, var_372);
                            //int32_t var_649 = var_390 - var_291;
                            //int32_t var_650 = var_649 + 16;
                            //uint16_t var_651 = var_input[var_650];//[8][16]
                            int16_t var_652 = (int16_t)(var_651);
                            int16_t var_653 = max(var_652, var_322);
                            int16_t var_654 = max(var_648, var_653);
                            int16_t var_655 = min(var_654, var_312);
                            //int32_t var_656 = var_335 + 21;
                            //uint16_t var_657 = var_input[var_656];//[13][21]
                            int16_t var_658 = (int16_t)(var_657);
                            int16_t var_659 = max(var_367, var_658);
                            int16_t var_660 = max(var_381, var_384);
                            int16_t var_661 = max(var_659, var_660);
                            int16_t var_662 = min(var_661, var_357);
                            int32_t var_663 = var_547 + var_430;
                            int32_t var_664 = var_663 + 1;
                            int32_t var_665 = var_664 >> 1;
                            int16_t var_666 = (int16_t)(var_665);
                            int16_t var_667 = max(var_655, var_420);
                            int32_t var_668 = (int32_t)(var_667);
                            int32_t var_669 = var_424 + var_668;
                            int32_t var_670 = var_669 + 1;
                            int32_t var_671 = var_670 >> 1;
                            int16_t var_672 = (int16_t)(var_671);
                            int16_t var_673 = var_429 - var_546;
                            int16_t var_674 = var_546 - var_429;
                            bool var_675 = var_546 < var_429;
                            int16_t var_676 = (int16_t)(var_675 ? var_673 : var_674);
                            uint16_t var_677 = var_676;
                            int16_t var_678 = var_667 - var_423;
                            int16_t var_679 = var_423 - var_667;
                            bool var_680 = var_423 < var_667;
                            int16_t var_681 = (int16_t)(var_680 ? var_678 : var_679);
                            uint16_t var_682 = var_681;
                            bool var_683 = var_677 < var_682;
                            int16_t var_684 = (int16_t)(var_683 ? var_666 : var_672);
                            //int32_t var_685 = var_335 + 14;
                            //uint16_t var_686 = var_input[var_685];//[13][14]
                            int16_t var_687 = (int16_t)(var_686);
                            int16_t var_688 = max(var_687, var_338);
                            //int32_t var_689 = var_361 + 16;
                            //uint16_t var_690 = var_input[var_689];//[15][16]
                            int16_t var_691 = (int16_t)(var_690);
                            int16_t var_692 = max(var_345, var_691);
                            int16_t var_693 = max(var_688, var_692);
                            int16_t var_694 = min(var_693, var_351);
                            //int32_t var_695 = var_342 + 14;
                            //uint16_t var_696 = var_input[var_695];//[11][14]
                            int16_t var_697 = (int16_t)(var_696);
                            int16_t var_698 = max(var_697, var_378);
                            //int32_t var_699 = var_409 + 16;
                            //uint16_t var_700 = var_input[var_699];//[9][16]
                            int16_t var_701 = (int16_t)(var_700);
                            int16_t var_702 = max(var_701, var_351);
                            int16_t var_703 = max(var_698, var_702);
                            int16_t var_704 = min(var_703, var_345);
                            //int32_t var_705 = var_342 + 20;
                            //uint16_t var_706 = var_input[var_705];//[11][20]
                            int16_t var_707 = (int16_t)(var_706);
                            int16_t var_708 = max(var_345, var_707);
                            //int32_t var_709 = var_409 + 18;
                            //uint16_t var_710 = var_input[var_709];//[9][18]
                            int16_t var_711 = (int16_t)(var_710);
                            int16_t var_712 = max(var_711, var_338);
                            int16_t var_713 = max(var_708, var_712);
                            int16_t var_714 = min(var_713, var_378);
                            //int32_t var_715 = var_335 + 20;
                            //uint16_t var_716 = var_input[var_715];//[13][20]
                            int16_t var_717 = (int16_t)(var_716);
                            int16_t var_718 = max(var_351, var_717);
                            //int32_t var_719 = var_361 + 18;
                            //uint16_t var_720 = var_input[var_719];//[15][18]
                            int16_t var_721 = (int16_t)(var_720);
                            int16_t var_722 = max(var_378, var_721);
                            int16_t var_723 = max(var_718, var_722);
                            int16_t var_724 = min(var_723, var_338);
                            //int32_t var_725 = var_342 + 21;
                            //uint16_t var_726 = var_input[var_725];//[11][21]
                            int16_t var_727 = (int16_t)(var_726);
                            int16_t var_728 = max(var_360, var_727);
                            //int32_t var_729 = var_409 + 19;
                            //uint16_t var_730 = var_input[var_729];//[9][19]
                            int16_t var_731 = (int16_t)(var_730);
                            int16_t var_732 = max(var_731, var_357);
                            int16_t var_733 = max(var_728, var_732);
                            int16_t var_734 = min(var_733, var_381);
                            //int32_t var_735 = var_309 + 20;
                            //uint16_t var_736 = var_input[var_735];//[10][20]
                            int16_t var_737 = (int16_t)(var_736);
                            int16_t var_738 = max(var_312, var_737);
                            //int32_t var_739 = var_649 + 18;
                            //uint16_t var_740 = var_input[var_739];//[8][18]
                            int16_t var_741 = (int16_t)(var_740);
                            int16_t var_742 = max(var_741, var_305);
                            int16_t var_743 = max(var_738, var_742);
                            int16_t var_744 = min(var_743, var_372);
                            int16_t var_745 = max(var_662, var_420);
                            int32_t var_746 = (int32_t)(var_745);
                            int32_t var_747 = var_432 + var_746;
                            int32_t var_748 = var_747 + 1;
                            int32_t var_749 = var_748 >> 1;
                            int16_t var_750 = (int16_t)(var_749);
                            int32_t var_751 = var_625 + var_422;
                            int32_t var_752 = var_751 + 1;
                            int32_t var_753 = var_752 >> 1;
                            int16_t var_754 = (int16_t)(var_753);
                            int16_t var_755 = var_745 - var_431;
                            int16_t var_756 = var_431 - var_745;
                            bool var_757 = var_431 < var_745;
                            int16_t var_758 = (int16_t)(var_757 ? var_755 : var_756);
                            uint16_t var_759 = var_758;
                            int16_t var_760 = var_421 - var_624;
                            int16_t var_761 = var_624 - var_421;
                            bool var_762 = var_624 < var_421;
                            int16_t var_763 = (int16_t)(var_762 ? var_760 : var_761);
                            uint16_t var_764 = var_763;
                            bool var_765 = var_759 < var_764;
                            int16_t var_766 = (int16_t)(var_765 ? var_750 : var_754);
                            int32_t var_767 = (int32_t)(var_448);
                            int32_t var_768 = (int32_t)(var_563);
                            int32_t var_769 = var_767 + var_768;
                            int32_t var_770 = var_769 + 1;
                            int32_t var_771 = var_770 >> 1;
                            int16_t var_772 = (int16_t)(var_771);
                            int16_t var_773 = var_423 - var_772;
                            int16_t var_774 = max(var_573, var_420);
                            int32_t var_775 = (int32_t)(var_774);
                            int16_t var_776 = max(var_583, var_420);
                            int32_t var_777 = (int32_t)(var_776);
                            int32_t var_778 = var_775 + var_777;
                            int32_t var_779 = var_778 + 1;
                            int32_t var_780 = var_779 >> 1;
                            int16_t var_781 = (int16_t)(var_780);
                            int16_t var_782 = var_773 + var_781;
                            int16_t var_783 = (int16_t)(var_369 ? var_782 : var_776);
                            int16_t var_784 = max(var_613, var_420);
                            int32_t var_785 = (int32_t)(var_784);
                            int32_t var_786 = var_503 + var_785;
                            int32_t var_787 = var_786 + 1;
                            int32_t var_788 = var_787 >> 1;
                            int16_t var_789 = (int16_t)(var_788);
                            int16_t var_790 = max(var_623, var_420);
                            int32_t var_791 = (int32_t)(var_790);
                            int32_t var_792 = var_497 + var_791;
                            int32_t var_793 = var_792 + 1;
                            int32_t var_794 = var_793 >> 1;
                            int16_t var_795 = (int16_t)(var_794);
                            int16_t var_796 = var_784 - var_502;
                            int16_t var_797 = var_502 - var_784;
                            bool var_798 = var_502 < var_784;
                            int16_t var_799 = (int16_t)(var_798 ? var_796 : var_797);
                            uint16_t var_800 = var_799;
                            int16_t var_801 = var_790 - var_496;
                            int16_t var_802 = var_496 - var_790;
                            bool var_803 = var_496 < var_790;
                            int16_t var_804 = (int16_t)(var_803 ? var_801 : var_802);
                            uint16_t var_805 = var_804;
                            bool var_806 = var_800 < var_805;
                            int16_t var_807 = (int16_t)(var_806 ? var_789 : var_795);
                            int32_t var_808 = (int32_t)(var_807);
                            int32_t var_809 = var_767 + var_808;
                            int32_t var_810 = var_809 + 1;
                            int32_t var_811 = var_810 >> 1;
                            int16_t var_812 = (int16_t)(var_811);
                            int16_t var_813 = var_519 - var_812;
                            int16_t var_814 = max(var_593, var_420);
                            int32_t var_815 = (int32_t)(var_814);
                            int32_t var_816 = var_777 + var_815;
                            int32_t var_817 = var_816 + 1;
                            int32_t var_818 = var_817 >> 1;
                            int16_t var_819 = (int16_t)(var_818);
                            int16_t var_820 = var_813 + var_819;
                            int32_t var_821 = (int32_t)(var_647);
                            int32_t var_822 = var_768 + var_821;
                            int32_t var_823 = var_822 + 1;
                            int32_t var_824 = var_823 >> 1;
                            int16_t var_825 = (int16_t)(var_824);
                            int16_t var_826 = var_519 - var_825;
                            int16_t var_827 = max(var_603, var_420);
                            int32_t var_828 = (int32_t)(var_827);
                            int32_t var_829 = var_775 + var_828;
                            int32_t var_830 = var_829 + 1;
                            int32_t var_831 = var_830 >> 1;
                            int16_t var_832 = (int16_t)(var_831);
                            int16_t var_833 = var_826 + var_832;
                            int16_t var_834 = var_814 - var_776;
                            int16_t var_835 = var_776 - var_814;
                            bool var_836 = var_776 < var_814;
                            int16_t var_837 = (int16_t)(var_836 ? var_834 : var_835);
                            uint16_t var_838 = var_837;
                            int16_t var_839 = var_827 - var_774;
                            int16_t var_840 = var_774 - var_827;
                            bool var_841 = var_774 < var_827;
                            int16_t var_842 = (int16_t)(var_841 ? var_839 : var_840);
                            uint16_t var_843 = var_842;
                            bool var_844 = var_838 < var_843;
                            int16_t var_845 = (int16_t)(var_844 ? var_820 : var_833);
                            int32_t var_846 = var_767 + var_821;
                            int32_t var_847 = var_846 + 1;
                            int32_t var_848 = var_847 >> 1;
                            int16_t var_849 = (int16_t)(var_848);
                            int16_t var_850 = var_431 - var_849;
                            int32_t var_851 = var_777 + var_828;
                            int32_t var_852 = var_851 + 1;
                            int32_t var_853 = var_852 >> 1;
                            int16_t var_854 = (int16_t)(var_853);
                            int16_t var_855 = var_850 + var_854;
                            int16_t var_856 = (int16_t)(var_369 ? var_845 : var_855);
                            int16_t var_857 = (int16_t)(var_539 ? var_783 : var_856);
                            int16_t var_858 = (int16_t)(var_369 ? var_423 : var_448);
                            int16_t var_859 = (int16_t)(var_369 ? var_519 : var_431);
                            int16_t var_860 = (int16_t)(var_539 ? var_858 : var_859);
                            int32_t var_861 = (int32_t)(var_519);
                            int32_t var_862 = (int32_t)(var_684);
                            int32_t var_863 = var_861 + var_862;
                            int32_t var_864 = var_863 + 1;
                            int32_t var_865 = var_864 >> 1;
                            int16_t var_866 = (int16_t)(var_865);
                            int16_t var_867 = var_423 - var_866;
                            int16_t var_868 = max(var_694, var_420);
                            int32_t var_869 = (int32_t)(var_868);
                            int16_t var_870 = max(var_704, var_420);
                            int32_t var_871 = (int32_t)(var_870);
                            int32_t var_872 = var_869 + var_871;
                            int32_t var_873 = var_872 + 1;
                            int32_t var_874 = var_873 >> 1;
                            int16_t var_875 = (int16_t)(var_874);
                            int16_t var_876 = var_867 + var_875;
                            int16_t var_877 = max(var_734, var_420);
                            int32_t var_878 = (int32_t)(var_877);
                            int32_t var_879 = var_430 + var_878;
                            int32_t var_880 = var_879 + 1;
                            int32_t var_881 = var_880 >> 1;
                            int16_t var_882 = (int16_t)(var_881);
                            int16_t var_883 = max(var_744, var_420);
                            int32_t var_884 = (int32_t)(var_883);
                            int32_t var_885 = var_422 + var_884;
                            int32_t var_886 = var_885 + 1;
                            int32_t var_887 = var_886 >> 1;
                            int16_t var_888 = (int16_t)(var_887);
                            int16_t var_889 = var_877 - var_429;
                            int16_t var_890 = var_429 - var_877;
                            bool var_891 = var_429 < var_877;
                            int16_t var_892 = (int16_t)(var_891 ? var_889 : var_890);
                            uint16_t var_893 = var_892;
                            int16_t var_894 = var_883 - var_421;
                            int16_t var_895 = var_421 - var_883;
                            bool var_896 = var_421 < var_883;
                            int16_t var_897 = (int16_t)(var_896 ? var_894 : var_895);
                            uint16_t var_898 = var_897;
                            bool var_899 = var_893 < var_898;
                            int16_t var_900 = (int16_t)(var_899 ? var_882 : var_888);
                            int32_t var_901 = (int32_t)(var_900);
                            int32_t var_902 = var_861 + var_901;
                            int32_t var_903 = var_902 + 1;
                            int32_t var_904 = var_903 >> 1;
                            int16_t var_905 = (int16_t)(var_904);
                            int16_t var_906 = var_448 - var_905;
                            int16_t var_907 = max(var_714, var_420);
                            int32_t var_908 = (int32_t)(var_907);
                            int32_t var_909 = var_869 + var_908;
                            int32_t var_910 = var_909 + 1;
                            int32_t var_911 = var_910 >> 1;
                            int16_t var_912 = (int16_t)(var_911);
                            int16_t var_913 = var_906 + var_912;
                            int32_t var_914 = (int32_t)(var_766);
                            int32_t var_915 = var_914 + var_862;
                            int32_t var_916 = var_915 + 1;
                            int32_t var_917 = var_916 >> 1;
                            int16_t var_918 = (int16_t)(var_917);
                            int16_t var_919 = var_448 - var_918;
                            int16_t var_920 = max(var_724, var_420);
                            int32_t var_921 = (int32_t)(var_920);
                            int32_t var_922 = var_921 + var_871;
                            int32_t var_923 = var_922 + 1;
                            int32_t var_924 = var_923 >> 1;
                            int16_t var_925 = (int16_t)(var_924);
                            int16_t var_926 = var_919 + var_925;
                            int16_t var_927 = var_907 - var_868;
                            int16_t var_928 = var_868 - var_907;
                            bool var_929 = var_868 < var_907;
                            int16_t var_930 = (int16_t)(var_929 ? var_927 : var_928);
                            uint16_t var_931 = var_930;
                            int16_t var_932 = var_870 - var_920;
                            int16_t var_933 = var_920 - var_870;
                            bool var_934 = var_920 < var_870;
                            int16_t var_935 = (int16_t)(var_934 ? var_932 : var_933);
                            uint16_t var_936 = var_935;
                            bool var_937 = var_931 < var_936;
                            int16_t var_938 = (int16_t)(var_937 ? var_913 : var_926);
                            int16_t var_939 = (int16_t)(var_369 ? var_876 : var_938);
                            int32_t var_940 = var_861 + var_914;
                            int32_t var_941 = var_940 + 1;
                            int32_t var_942 = var_941 >> 1;
                            int16_t var_943 = (int16_t)(var_942);
                            int16_t var_944 = var_431 - var_943;
                            int32_t var_945 = var_869 + var_921;
                            int32_t var_946 = var_945 + 1;
                            int32_t var_947 = var_946 >> 1;
                            int16_t var_948 = (int16_t)(var_947);
                            int16_t var_949 = var_944 + var_948;
                            int16_t var_950 = (int16_t)(var_369 ? var_868 : var_949);
                            int16_t var_951 = (int16_t)(var_539 ? var_939 : var_950);
                            int16_t var_952 = var_matrix[unroll_index][3];
                            int32_t var_953 = (int32_t)(var_952);
                            int16_t var_954 = var_matrix[unroll_index][0];
                            int32_t var_955 = (int32_t)(var_954);
                            int32_t var_956 = (int32_t)(var_857);
                            int32_t var_957 = var_955 * var_956;
                            int32_t var_958 = var_953 + var_957;
                            int16_t var_959 = var_matrix[unroll_index][1];
                            int32_t var_960 = (int32_t)(var_959);
                            int32_t var_961 = (int32_t)(var_860);
                            int32_t var_962 = var_960 * var_961;
                            int32_t var_963 = var_958 + var_962;
                            int16_t var_964 = var_matrix[unroll_index][2];
                            int32_t var_965 = (int32_t)(var_964);
                            int32_t var_966 = (int32_t)(var_951);
                            int32_t var_967 = var_965 * var_966;
                            int32_t var_968 = var_963 + var_967;
                            int32_t var_969 = var_968 >> 8;
                            int16_t var_970 = (int16_t)(var_969);
                            int16_t var_971 = var_matrix[unroll_index][7];
                            int32_t var_972 = (int32_t)(var_971);
                            int16_t var_973 = var_matrix[unroll_index][4];
                            int32_t var_974 = (int32_t)(var_973);
                            int32_t var_975 = var_974 * var_956;
                            int32_t var_976 = var_972 + var_975;
                            int16_t var_977 = var_matrix[unroll_index][5];
                            int32_t var_978 = (int32_t)(var_977);
                            int32_t var_979 = var_978 * var_961;
                            int32_t var_980 = var_976 + var_979;
                            int16_t var_981 = var_matrix[unroll_index][6];
                            int32_t var_982 = (int32_t)(var_981);
                            int32_t var_983 = var_982 * var_966;
                            int32_t var_984 = var_980 + var_983;
                            int32_t var_985 = var_984 >> 8;
                            int16_t var_986 = (int16_t)(var_985);
                            int16_t var_987 = var_matrix[unroll_index][11];
                            int32_t var_988 = (int32_t)(var_987);
                            int16_t var_989 = var_matrix[unroll_index][8];
                            int32_t var_990 = (int32_t)(var_989);
                            int32_t var_991 = var_990 * var_956;
                            int32_t var_992 = var_988 + var_991;
                            int16_t var_993 = var_matrix[unroll_index][9];
                            int32_t var_994 = (int32_t)(var_993);
                            int32_t var_995 = var_994 * var_961;
                            int32_t var_996 = var_992 + var_995;
                            int16_t var_997 = var_matrix[unroll_index][10];
                            int32_t var_998 = (int32_t)(var_997);
                            int32_t var_999 = var_998 * var_966;
                            int32_t var_1000 = var_996 + var_999;
                            int32_t var_1001 = var_1000 >> 8;
                            int16_t var_1002 = (int16_t)(var_1001);
                            //bool var_1003 = var_processed_s0_c == 1;
                            //int16_t var_1004 = (int16_t)(var_1003 ? var_986 : var_1002);
                            //bool var_1005 = var_processed_s0_c == 0;
                            //int16_t var_1006 = (int16_t)(var_1005 ? var_970 : var_1004);
                            int16_t var_1006_c0 = var_970;
                            int16_t var_1006_c1 = var_986;
                            int16_t var_1006_c2 = var_1002;
                            int16_t var_1007 = (int16_t)(1023);
                            //int16_t var_1008 = min(var_1006, var_1007);
                            int16_t var_1008_c0 = min(var_1006_c0, var_1007);
                            int16_t var_1008_c1 = min(var_1006_c1, var_1007);
                            int16_t var_1008_c2 = min(var_1006_c2, var_1007);
                            //int16_t var_1009 = max(var_1008, var_420);
                            int16_t var_1009_c0 = max(var_1008_c0, var_420);
                            int16_t var_1009_c1 = max(var_1008_c1, var_420);
                            int16_t var_1009_c2 = max(var_1008_c2, var_420);
                            //int32_t var_1010 = (int32_t)(var_1009);
                            //uint8_t var_1011 = var_curve[var_1010];
                            //var_processed[var_289] = var_1011;
                            output[0][output_index] = var_curve[UNROLL_FACTOR*0+unroll_index][var_1009_c0];
                            output[1][output_index] = var_curve[UNROLL_FACTOR*1+unroll_index][var_1009_c1];
                            output[2][output_index] = var_curve[UNROLL_FACTOR*2+unroll_index][var_1009_c2];
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
        // consume processed
    }
}

extern "C"
{

void curved_kernel(volatile int16_t* var_matrix, volatile uint8_t* var_curve, ap_uint<BURST_WIDTH>* var_processed, ap_uint<BURST_WIDTH>* var_input, int32_t tile_num_dim0, int32_t tile_num_dim1, int32_t var_processed_extent_0, int32_t var_processed_extent_1, int32_t var_processed_min_0, int32_t var_processed_min_1)
{
#pragma HLS INTERFACE m_axi port=var_processed offset=slave depth=351000 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_input offset=slave depth=115200 bundle=gmem2 latency=120
#pragma HLS INTERFACE m_axi port=var_matrix offset=slave depth=12 bundle=gmem3 latency=120
#pragma HLS INTERFACE m_axi port=var_curve offset=slave depth=1024 bundle=gmem3 latency=120

#pragma HLS INTERFACE s_axilite port=var_matrix bundle=control
#pragma HLS INTERFACE s_axilite port=var_curve bundle=control
#pragma HLS INTERFACE s_axilite port=var_processed bundle=control
#pragma HLS INTERFACE s_axilite port=var_input bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_processed_extent_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_processed_extent_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_processed_min_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_processed_min_1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    uint16_t  input_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t  input_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint8_t output_0[3][TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint8_t output_1[3][TILE_SIZE_DIM0*TILE_SIZE_DIM1];
#pragma HLS array_partition variable=input_0 cyclic factor=KI
#pragma HLS array_partition variable=input_1 cyclic factor=KI
#pragma HLS array_partition variable=output_0 complete dim=1
#pragma HLS array_partition variable=output_1 complete dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=KO dim=2
#pragma HLS array_partition variable=output_1 cyclic factor=KO dim=2

    int32_t total_tile_num = tile_num_dim0*tile_num_dim1;
    int32_t tile_index;
    bool    load_flag;
    bool compute_flag;
    bool   store_flag;

    int16_t matrix[UNROLL_FACTOR][12];
    uint8_t curve[UNROLL_FACTOR*3][1024];
#pragma HLS array_partition variable=matrix complete dim=1
#pragma HLS array_partition variable=matrix complete dim=2
#pragma HLS array_partition variable=curve complete dim=1

    for(int unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)
    {
#pragma HLS unroll
        for(int matrix_index = 0; matrix_index < 12; ++matrix_index)
        {
#pragma HLS pipeline II=1
            matrix[unroll_index][matrix_index] = var_matrix[matrix_index];
        }
        for(int curve_index = 0; curve_index < 1024; ++curve_index)
        {
#pragma HLS pipeline II=1
            uint8_t curve_data = var_curve[curve_index];
            curve[UNROLL_FACTOR*0+unroll_index][curve_index] = curve_data;
            curve[UNROLL_FACTOR*1+unroll_index][curve_index] = curve_data;
            curve[UNROLL_FACTOR*2+unroll_index][curve_index] = curve_data;
        }
    }

    for (tile_index = 0; tile_index < total_tile_num+2; ++tile_index)
    {
           load_flag =                   tile_index < total_tile_num;
        compute_flag = tile_index > 0 && tile_index < total_tile_num+1;
          store_flag = tile_index > 1;
        if(tile_index%2==0)
        {
            load(load_flag, input_0, var_input, tile_index);
            compute(compute_flag, output_1, input_1, matrix, curve, tile_index-1, tile_num_dim0, var_processed_extent_0, var_processed_extent_1, var_processed_min_0, var_processed_min_1);
            store(store_flag, var_processed, output_0, tile_index-2);
        }
        else
        {
            load(load_flag, input_1, var_input, tile_index);
            compute(compute_flag, output_0, input_0, matrix, curve, tile_index-1, tile_num_dim0, var_processed_extent_0, var_processed_extent_1, var_processed_min_0, var_processed_min_1);
            store(store_flag, var_processed, output_1, tile_index-2);
        }
    }
}

}
