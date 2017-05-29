#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stddef.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>

#include"ap_int.h"

//#ifndef TILE_SIZE_DIM0
//#define TILE_SIZE_DIM0 (2048)
//#endif//TILE_SIZE_DIM0
//#ifndef TILE_SIZE_DIM1
//#define TILE_SIZE_DIM1 (2048)
//#endif//TILE_SIZE_DIM1
//#ifndef BURST_LENGTH
//#define BURST_LENGTH (2048)
//#endif//BURST_LENGTH
//#ifndef UNROLL_FACTOR
//#define UNROLL_FACTOR (32)
//#endif//UNROLL_FACTOR

#define STENCIL_DIM0 (3)
#define STENCIL_DIM1 (3)
#define STENCIL_DISTANCE ((TILE_SIZE_DIM0)*2+2)
#define BURST_WIDTH (512)
#define PIXEL_WIDTH (16)

#define TILE_INDEX_DIM0(burst_index) ((burst_index/((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/(BURST_LENGTH)))%(tile_num_dim0))
#define TILE_INDEX_DIM1(burst_index) ((burst_index/((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/(BURST_LENGTH)))/(tile_num_dim0))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

#if 16!=(UNROLL_FACTOR)
#error UNROLL_FACTOR != 16
#endif

#if 2000!=(TILE_SIZE_DIM0)
#error TILE_SIZE_DIM0 != 2000
#endif

#if 2000!=(BURST_LENGTH)
#error BURST_LENGTH != 2000
#endif

#if ((BURST_WIDTH)/(PIXEL_WIDTH)*(BURST_LENGTH))%(UNROLL_FACTOR)!=0
#error BURST_LENGTH % UNROLL_FACTOR != 0
#endif

#if (((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1))%((BURST_LENGTH)*(BURST_WIDTH)/(PIXEL_WIDTH)))!=0
#error TILE_SIZE % BURST_LENGTH != 0
#endif

void load(bool load_flag, uint16_t to[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH], ap_uint<BURST_WIDTH>* from, int32_t burst_index)
{
    if(load_flag)
    {
        //fprintf(stderr, "Load: %d\n", burst_index);
load_epoch:
        for(int i = 0; i < BURST_LENGTH; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp(from[i+burst_index*BURST_LENGTH]);
load_coalesced:
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                to[i*BURST_WIDTH/PIXEL_WIDTH+j] = tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH);
            }
        }
        //fprintf(stderr, "Load done: %d\n",burst_index);
    }
}

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, uint16_t from[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH], int32_t burst_index)
{
    if(store_flag)
    {
        //fprintf(stderr, "Store: %d\n",burst_index);
store_epoch:
        for(int i = 0; i < BURST_LENGTH; ++i)
        {
#pragma HLS pipeline II=1
            ap_uint<BURST_WIDTH> tmp;
store_coalesced:
            for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH; ++j)
            {
#pragma HLS unroll
                tmp((j+1)*PIXEL_WIDTH-1, j*PIXEL_WIDTH) = from[i*BURST_WIDTH/PIXEL_WIDTH+j];
            }
            to[i+burst_index*BURST_LENGTH] = tmp;
        }
        //fprintf(stderr, "Store done: %d\n",burst_index);
    }
}

void compute(bool compute_flag, uint16_t output[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH], uint16_t input[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH], uint16_t FF[6], uint16_t FIFO_124[4][124], uint16_t FIFO_125[28][125], int32_t burst_index_in_total, int32_t tile_num_dim0, int32_t var_blur_y_extent_0, int32_t var_blur_y_extent_1, int32_t var_blur_y_min_0, int32_t var_blur_y_min_1)
{
    if(compute_flag)
    {
        int32_t burst_index = burst_index_in_total % (TILE_SIZE_DIM0*TILE_SIZE_DIM1/(BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH));
        //fprintf(stderr, "Compute: %d %d\n", burst_index, burst_index_in_total);
        int32_t tile_index_dim0 = TILE_INDEX_DIM0(burst_index_in_total);
        int32_t tile_index_dim1 = TILE_INDEX_DIM1(burst_index_in_total);

        int32_t FIFO_124_ptr = (BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH/UNROLL_FACTOR*burst_index)%124;
        int32_t FIFO_125_ptr = (BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH/UNROLL_FACTOR*burst_index)%125;

        uint16_t input_points[(UNROLL_FACTOR)][9];
        //         input_points[(UNROLL_FACTOR)][0] <=> (0, 0)
        //         input_points[(UNROLL_FACTOR)][1] <=> (1, 0)
        //         input_points[(UNROLL_FACTOR)][2] <=> (2, 0)
        //         input_points[(UNROLL_FACTOR)][3] <=> (0, 1)
        //         input_points[(UNROLL_FACTOR)][4] <=> (1, 1)
        //         input_points[(UNROLL_FACTOR)][5] <=> (2, 1)
        //         input_points[(UNROLL_FACTOR)][6] <=> (0, 2)
        //         input_points[(UNROLL_FACTOR)][7] <=> (1, 2)
        //         input_points[(UNROLL_FACTOR)][8] <=> (2, 2)
        uint16_t input_buffer[16];
#pragma HLS array_partition variable=input_points complete dim=0
#pragma HLS array_partition variable=input_buffer complete

        // produce blur_y
compute_epoch:
        for (int32_t epoch = 0; epoch < BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH/UNROLL_FACTOR; ++epoch)
        {
#pragma HLS dependence variable=FF inter false
#pragma HLS dependence variable=FIFO_124 inter false
#pragma HLS dependence variable=FIFO_125 inter false
            int32_t input_index = epoch + BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH/UNROLL_FACTOR*burst_index;
#pragma HLS pipeline II=1
compute_load_unrolled:
            for(uint32_t unroll_index = 0; unroll_index<(UNROLL_FACTOR); ++unroll_index)
            {
#pragma HLS unroll
                if(input_index*(UNROLL_FACTOR)+unroll_index < (TILE_SIZE_DIM0)*(TILE_SIZE_DIM1))
                {
/*                    if(input[epoch*(UNROLL_FACTOR)+unroll_index] != (input_index*UNROLL_FACTOR+unroll_index)%TILE_SIZE_DIM0+(input_index*UNROLL_FACTOR+unroll_index)/TILE_SIZE_DIM0)
                    {
                        printf("burst_index:%d input_index*UNROLL_FACTOR+unroll_index:%d input[epoch*(UNROLL_FACTOR)+unroll_index]:%d\n", burst_index, input_index*UNROLL_FACTOR+unroll_index, input[epoch*(UNROLL_FACTOR)+unroll_index]);
                    }*/
                    input_buffer[unroll_index] = input[epoch*(UNROLL_FACTOR)+unroll_index];
                }
            }

            input_points[0][8] = input_buffer[0]; // (2, 2)
            input_points[1][7] = input_buffer[0]; // (1, 2)
            input_points[2][6] = input_buffer[0]; // (0, 2)
            input_points[1][8] = input_buffer[1]; // (2, 2)
            input_points[2][7] = input_buffer[1]; // (1, 2)
            input_points[3][6] = input_buffer[1]; // (0, 2)
            input_points[2][8] = input_buffer[2]; // (2, 2)
            input_points[3][7] = input_buffer[2]; // (1, 2)
            input_points[4][6] = input_buffer[2]; // (0, 2)
            input_points[3][8] = input_buffer[3]; // (2, 2)
            input_points[4][7] = input_buffer[3]; // (1, 2)
            input_points[5][6] = input_buffer[3]; // (0, 2)
            input_points[4][8] = input_buffer[4]; // (2, 2)
            input_points[5][7] = input_buffer[4]; // (1, 2)
            input_points[6][6] = input_buffer[4]; // (0, 2)
            input_points[5][8] = input_buffer[5]; // (2, 2)
            input_points[6][7] = input_buffer[5]; // (1, 2)
            input_points[7][6] = input_buffer[5]; // (0, 2)
            input_points[8][6] = input_buffer[6]; // (0, 2)
            input_points[6][8] = input_buffer[6]; // (2, 2)
            input_points[7][7] = input_buffer[6]; // (1, 2)
            input_points[8][7] = input_buffer[7]; // (1, 2)
            input_points[9][6] = input_buffer[7]; // (0, 2)
            input_points[7][8] = input_buffer[7]; // (2, 2)
            input_points[8][8] = input_buffer[8]; // (2, 2)
            input_points[9][7] = input_buffer[8]; // (1, 2)
            input_points[10][6] = input_buffer[8]; // (0, 2)
            input_points[9][8] = input_buffer[9]; // (2, 2)
            input_points[10][7] = input_buffer[9]; // (1, 2)
            input_points[11][6] = input_buffer[9]; // (0, 2)
            input_points[10][8] = input_buffer[10]; // (2, 2)
            input_points[11][7] = input_buffer[10]; // (1, 2)
            input_points[12][6] = input_buffer[10]; // (0, 2)
            input_points[11][8] = input_buffer[11]; // (2, 2)
            input_points[12][7] = input_buffer[11]; // (1, 2)
            input_points[13][6] = input_buffer[11]; // (0, 2)
            input_points[12][8] = input_buffer[12]; // (2, 2)
            input_points[13][7] = input_buffer[12]; // (1, 2)
            input_points[14][6] = input_buffer[12]; // (0, 2)
            input_points[13][8] = input_buffer[13]; // (2, 2)
            input_points[14][7] = input_buffer[13]; // (1, 2)
            input_points[15][6] = input_buffer[13]; // (0, 2)
            input_points[14][8] = input_buffer[14]; // (2, 2)
            input_points[15][7] = input_buffer[14]; // (1, 2)
            input_points[15][8] = input_buffer[15]; // (2, 2)
            input_points[0][0] = FF[0]; // (0, 0)
            input_points[0][1] = FF[1]; // (1, 0)
            input_points[1][0] = FF[1]; // (0, 0)
            input_points[0][3] = FF[2]; // (0, 1)
            input_points[0][4] = FF[3]; // (1, 1)
            input_points[1][3] = FF[3]; // (0, 1)
            input_points[0][6] = FF[4]; // (0, 2)
            input_points[0][7] = FF[5]; // (1, 2)
            input_points[1][6] = FF[5]; // (0, 2)
            input_points[14][2] = FIFO_124[0][FIFO_124_ptr]; // (2, 0)
            input_points[15][1] = FIFO_124[0][FIFO_124_ptr]; // (1, 0)
            input_points[15][2] = FIFO_124[1][FIFO_124_ptr]; // (2, 0)
            input_points[14][5] = FIFO_124[2][FIFO_124_ptr]; // (2, 1)
            input_points[15][4] = FIFO_124[2][FIFO_124_ptr]; // (1, 1)
            input_points[15][5] = FIFO_124[3][FIFO_124_ptr]; // (2, 1)
            input_points[0][2] = FIFO_125[0][FIFO_125_ptr]; // (2, 0)
            input_points[1][1] = FIFO_125[0][FIFO_125_ptr]; // (1, 0)
            input_points[2][0] = FIFO_125[0][FIFO_125_ptr]; // (0, 0)
            input_points[1][2] = FIFO_125[1][FIFO_125_ptr]; // (2, 0)
            input_points[2][1] = FIFO_125[1][FIFO_125_ptr]; // (1, 0)
            input_points[3][0] = FIFO_125[1][FIFO_125_ptr]; // (0, 0)
            input_points[2][2] = FIFO_125[2][FIFO_125_ptr]; // (2, 0)
            input_points[3][1] = FIFO_125[2][FIFO_125_ptr]; // (1, 0)
            input_points[4][0] = FIFO_125[2][FIFO_125_ptr]; // (0, 0)
            input_points[3][2] = FIFO_125[3][FIFO_125_ptr]; // (2, 0)
            input_points[4][1] = FIFO_125[3][FIFO_125_ptr]; // (1, 0)
            input_points[5][0] = FIFO_125[3][FIFO_125_ptr]; // (0, 0)
            input_points[4][2] = FIFO_125[4][FIFO_125_ptr]; // (2, 0)
            input_points[5][1] = FIFO_125[4][FIFO_125_ptr]; // (1, 0)
            input_points[6][0] = FIFO_125[4][FIFO_125_ptr]; // (0, 0)
            input_points[5][2] = FIFO_125[5][FIFO_125_ptr]; // (2, 0)
            input_points[6][1] = FIFO_125[5][FIFO_125_ptr]; // (1, 0)
            input_points[7][0] = FIFO_125[5][FIFO_125_ptr]; // (0, 0)
            input_points[8][0] = FIFO_125[6][FIFO_125_ptr]; // (0, 0)
            input_points[6][2] = FIFO_125[6][FIFO_125_ptr]; // (2, 0)
            input_points[7][1] = FIFO_125[6][FIFO_125_ptr]; // (1, 0)
            input_points[8][1] = FIFO_125[7][FIFO_125_ptr]; // (1, 0)
            input_points[9][0] = FIFO_125[7][FIFO_125_ptr]; // (0, 0)
            input_points[7][2] = FIFO_125[7][FIFO_125_ptr]; // (2, 0)
            input_points[8][2] = FIFO_125[8][FIFO_125_ptr]; // (2, 0)
            input_points[9][1] = FIFO_125[8][FIFO_125_ptr]; // (1, 0)
            input_points[10][0] = FIFO_125[8][FIFO_125_ptr]; // (0, 0)
            input_points[9][2] = FIFO_125[9][FIFO_125_ptr]; // (2, 0)
            input_points[10][1] = FIFO_125[9][FIFO_125_ptr]; // (1, 0)
            input_points[11][0] = FIFO_125[9][FIFO_125_ptr]; // (0, 0)
            input_points[10][2] = FIFO_125[10][FIFO_125_ptr]; // (2, 0)
            input_points[11][1] = FIFO_125[10][FIFO_125_ptr]; // (1, 0)
            input_points[12][0] = FIFO_125[10][FIFO_125_ptr]; // (0, 0)
            input_points[11][2] = FIFO_125[11][FIFO_125_ptr]; // (2, 0)
            input_points[12][1] = FIFO_125[11][FIFO_125_ptr]; // (1, 0)
            input_points[13][0] = FIFO_125[11][FIFO_125_ptr]; // (0, 0)
            input_points[12][2] = FIFO_125[12][FIFO_125_ptr]; // (2, 0)
            input_points[13][1] = FIFO_125[12][FIFO_125_ptr]; // (1, 0)
            input_points[14][0] = FIFO_125[12][FIFO_125_ptr]; // (0, 0)
            input_points[13][2] = FIFO_125[13][FIFO_125_ptr]; // (2, 0)
            input_points[14][1] = FIFO_125[13][FIFO_125_ptr]; // (1, 0)
            input_points[15][0] = FIFO_125[13][FIFO_125_ptr]; // (0, 0)
            input_points[0][5] = FIFO_125[14][FIFO_125_ptr]; // (2, 1)
            input_points[1][4] = FIFO_125[14][FIFO_125_ptr]; // (1, 1)
            input_points[2][3] = FIFO_125[14][FIFO_125_ptr]; // (0, 1)
            input_points[1][5] = FIFO_125[15][FIFO_125_ptr]; // (2, 1)
            input_points[2][4] = FIFO_125[15][FIFO_125_ptr]; // (1, 1)
            input_points[3][3] = FIFO_125[15][FIFO_125_ptr]; // (0, 1)
            input_points[2][5] = FIFO_125[16][FIFO_125_ptr]; // (2, 1)
            input_points[3][4] = FIFO_125[16][FIFO_125_ptr]; // (1, 1)
            input_points[4][3] = FIFO_125[16][FIFO_125_ptr]; // (0, 1)
            input_points[3][5] = FIFO_125[17][FIFO_125_ptr]; // (2, 1)
            input_points[4][4] = FIFO_125[17][FIFO_125_ptr]; // (1, 1)
            input_points[5][3] = FIFO_125[17][FIFO_125_ptr]; // (0, 1)
            input_points[4][5] = FIFO_125[18][FIFO_125_ptr]; // (2, 1)
            input_points[5][4] = FIFO_125[18][FIFO_125_ptr]; // (1, 1)
            input_points[6][3] = FIFO_125[18][FIFO_125_ptr]; // (0, 1)
            input_points[5][5] = FIFO_125[19][FIFO_125_ptr]; // (2, 1)
            input_points[6][4] = FIFO_125[19][FIFO_125_ptr]; // (1, 1)
            input_points[7][3] = FIFO_125[19][FIFO_125_ptr]; // (0, 1)
            input_points[8][3] = FIFO_125[20][FIFO_125_ptr]; // (0, 1)
            input_points[6][5] = FIFO_125[20][FIFO_125_ptr]; // (2, 1)
            input_points[7][4] = FIFO_125[20][FIFO_125_ptr]; // (1, 1)
            input_points[8][4] = FIFO_125[21][FIFO_125_ptr]; // (1, 1)
            input_points[9][3] = FIFO_125[21][FIFO_125_ptr]; // (0, 1)
            input_points[7][5] = FIFO_125[21][FIFO_125_ptr]; // (2, 1)
            input_points[8][5] = FIFO_125[22][FIFO_125_ptr]; // (2, 1)
            input_points[9][4] = FIFO_125[22][FIFO_125_ptr]; // (1, 1)
            input_points[10][3] = FIFO_125[22][FIFO_125_ptr]; // (0, 1)
            input_points[9][5] = FIFO_125[23][FIFO_125_ptr]; // (2, 1)
            input_points[10][4] = FIFO_125[23][FIFO_125_ptr]; // (1, 1)
            input_points[11][3] = FIFO_125[23][FIFO_125_ptr]; // (0, 1)
            input_points[10][5] = FIFO_125[24][FIFO_125_ptr]; // (2, 1)
            input_points[11][4] = FIFO_125[24][FIFO_125_ptr]; // (1, 1)
            input_points[12][3] = FIFO_125[24][FIFO_125_ptr]; // (0, 1)
            input_points[11][5] = FIFO_125[25][FIFO_125_ptr]; // (2, 1)
            input_points[12][4] = FIFO_125[25][FIFO_125_ptr]; // (1, 1)
            input_points[13][3] = FIFO_125[25][FIFO_125_ptr]; // (0, 1)
            input_points[12][5] = FIFO_125[26][FIFO_125_ptr]; // (2, 1)
            input_points[13][4] = FIFO_125[26][FIFO_125_ptr]; // (1, 1)
            input_points[14][3] = FIFO_125[26][FIFO_125_ptr]; // (0, 1)
            input_points[13][5] = FIFO_125[27][FIFO_125_ptr]; // (2, 1)
            input_points[14][4] = FIFO_125[27][FIFO_125_ptr]; // (1, 1)
            input_points[15][3] = FIFO_125[27][FIFO_125_ptr]; // (0, 1)

compute_unrolled:
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
                        uint16_t input_0 = input_points[unroll_index][0];
                        uint16_t input_1 = input_points[unroll_index][1];
                        uint16_t input_2 = input_points[unroll_index][2];
                        uint16_t input_3 = input_points[unroll_index][3];
                        uint16_t input_4 = input_points[unroll_index][4];
                        uint16_t input_5 = input_points[unroll_index][5];
                        uint16_t input_6 = input_points[unroll_index][6];
                        uint16_t input_7 = input_points[unroll_index][7];
                        uint16_t input_8 = input_points[unroll_index][8];
                        if(p >= var_blur_y_min_0 &&
                           q >= var_blur_y_min_1 &&
                           p < var_blur_y_min_0 + var_blur_y_extent_0 &&
                           q < var_blur_y_min_1 + var_blur_y_extent_1 &&
                           i < TILE_SIZE_DIM0-STENCIL_DIM0+1 &&
                           j < TILE_SIZE_DIM1-STENCIL_DIM1+1 &&
                           1)
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
                            output[output_index+STENCIL_DISTANCE-BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH*burst_index] = assign_128;
                            //printf("out offset: %d - %d = %d\n", output_index+STENCIL_DISTANCE, BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH*burst_index, output_index+STENCIL_DISTANCE-BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH*burst_index);
/*                            if(output_index+STENCIL_DISTANCE-BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH*burst_index<0 || output_index+STENCIL_DISTANCE-BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH*burst_index>=BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH)
                            {
                                fprintf(stderr, "Something wrong.\n");
                            }
                            if(true || assign_128!=p+q+2)
                            {
                                printf("input_index=%d output_index=%d unroll_index=%d burst_index=%d i=%d j=%d p=%d q=%d output=%d input_0=%d input_buffer[unroll_index]=%d\n", input_index, output_index, unroll_index, burst_index, i, j, p, q, int(assign_128), int(input_0), int(input_buffer[unroll_index]));
                            }*/
                        }
                        /*else
                        {
                            printf("if branch: tile_num_dim0:%d tile_index_dim0:%d tile_index_dim1:%d input_index:%d, output_index:%d, i:%d, j:%d, p:%d, q:%d\n", tile_num_dim0, tile_index_dim0, tile_index_dim1, input_index, output_index, i, j, p, q);
                        }*/
                    }
                } // if input_index >= STENCIL_DISTANCE
            } // for unroll_index

            FIFO_125[0][FIFO_125_ptr] = FIFO_125[14][FIFO_125_ptr];
            FIFO_125[14][FIFO_125_ptr] = input_buffer[0];

            FIFO_125[1][FIFO_125_ptr] = FIFO_125[15][FIFO_125_ptr];
            FIFO_125[15][FIFO_125_ptr] = input_buffer[1];

            FIFO_125[2][FIFO_125_ptr] = FIFO_125[16][FIFO_125_ptr];
            FIFO_125[16][FIFO_125_ptr] = input_buffer[2];

            FIFO_125[3][FIFO_125_ptr] = FIFO_125[17][FIFO_125_ptr];
            FIFO_125[17][FIFO_125_ptr] = input_buffer[3];

            FIFO_125[4][FIFO_125_ptr] = FIFO_125[18][FIFO_125_ptr];
            FIFO_125[18][FIFO_125_ptr] = input_buffer[4];

            FIFO_125[5][FIFO_125_ptr] = FIFO_125[19][FIFO_125_ptr];
            FIFO_125[19][FIFO_125_ptr] = input_buffer[5];

            FIFO_125[6][FIFO_125_ptr] = FIFO_125[20][FIFO_125_ptr];
            FIFO_125[20][FIFO_125_ptr] = input_buffer[6];

            FIFO_125[7][FIFO_125_ptr] = FIFO_125[21][FIFO_125_ptr];
            FIFO_125[21][FIFO_125_ptr] = input_buffer[7];

            FIFO_125[8][FIFO_125_ptr] = FIFO_125[22][FIFO_125_ptr];
            FIFO_125[22][FIFO_125_ptr] = input_buffer[8];

            FIFO_125[9][FIFO_125_ptr] = FIFO_125[23][FIFO_125_ptr];
            FIFO_125[23][FIFO_125_ptr] = input_buffer[9];

            FIFO_125[10][FIFO_125_ptr] = FIFO_125[24][FIFO_125_ptr];
            FIFO_125[24][FIFO_125_ptr] = input_buffer[10];

            FIFO_125[11][FIFO_125_ptr] = FIFO_125[25][FIFO_125_ptr];
            FIFO_125[25][FIFO_125_ptr] = input_buffer[11];

            FIFO_125[12][FIFO_125_ptr] = FIFO_125[26][FIFO_125_ptr];
            FIFO_125[26][FIFO_125_ptr] = input_buffer[12];

            FIFO_125[13][FIFO_125_ptr] = FIFO_125[27][FIFO_125_ptr];
            FIFO_125[27][FIFO_125_ptr] = input_buffer[13];

            FF[0] = FIFO_124[0][FIFO_124_ptr];
            FIFO_124[0][FIFO_124_ptr] = FF[2];
            FF[2] = FIFO_124[2][FIFO_124_ptr];
            FIFO_124[2][FIFO_124_ptr] = FF[4];
            FF[4] = input_buffer[14];

            FF[1] = FIFO_124[1][FIFO_124_ptr];
            FIFO_124[1][FIFO_124_ptr] = FF[3];
            FF[3] = FIFO_124[3][FIFO_124_ptr];
            FIFO_124[3][FIFO_124_ptr] = FF[5];
            FF[5] = input_buffer[15];

            FIFO_124_ptr = FIFO_124_ptr==(124-1) ? 0 : FIFO_124_ptr+1;
            FIFO_125_ptr = FIFO_125_ptr==(125-1) ? 0 : FIFO_125_ptr+1;
        } // for input_index
        // consume blur_y
        //fprintf(stderr, "Compute done: %d\n",burst_index);
    }
}

extern "C"
{

void blur_kernel(ap_uint<BURST_WIDTH>* var_blur_y, ap_uint<BURST_WIDTH>* var_p0, int32_t tile_num_dim0, int32_t tile_num_dim1, int32_t var_blur_y_extent_0, int32_t var_blur_y_extent_1, int32_t var_blur_y_min_0, int32_t var_blur_y_min_1)
{
#pragma HLS INTERFACE m_axi port=var_blur_y offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth=65536 bundle=gmem2 latency=120

#pragma HLS INTERFACE s_axilite port=var_blur_y bundle=control
#pragma HLS INTERFACE s_axilite port=var_p0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_extent_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_extent_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_min_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_blur_y_min_1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    uint16_t  input_0[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH];
    uint16_t  input_1[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH];
    uint16_t output_0[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH];
    uint16_t output_1[BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH];
    uint16_t FF[6];
    uint16_t FIFO_124[4][124];
    uint16_t FIFO_125[28][125];
#pragma HLS array_partition variable=input_0 cyclic factor=KI dim=1
#pragma HLS array_partition variable=input_1 cyclic factor=KI dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=KO dim=1
#pragma HLS array_partition variable=output_1 cyclic factor=KO dim=1
#pragma HLS array_partition variable=FF complete
#pragma HLS array_partition variable=FIFO_124 complete dim=1
#pragma HLS array_partition variable=FIFO_125 complete dim=1

    int32_t total_burst_num = TILE_SIZE_DIM0*TILE_SIZE_DIM1/(BURST_WIDTH/PIXEL_WIDTH*BURST_LENGTH)*tile_num_dim0*tile_num_dim1;
    int32_t tile_index_dim0 = 0;
    int32_t tile_index_dim1 = 0;
    bool    load_flag;
    bool compute_flag;
    bool   store_flag;

    //fprintf(stderr, "Checkpoint 1\n");
burst:
    for(int32_t burst_index = 0; burst_index < total_burst_num+2; ++burst_index)
    {
        //fprintf(stderr, "\n%dCheckpoint 2:%d\n", total_burst_num, burst_index);
           load_flag =                    burst_index < total_burst_num;
        compute_flag = burst_index > 0 && burst_index < total_burst_num+1;
          store_flag = burst_index > 1;
        if(burst_index%2==0)
        {
            load(load_flag, input_0, var_p0, burst_index);
            compute(compute_flag, output_1, input_1, FF, FIFO_124, FIFO_125, burst_index-1, tile_num_dim0, var_blur_y_extent_0, var_blur_y_extent_1, var_blur_y_min_0, var_blur_y_min_1);
            store(store_flag, var_blur_y, output_0, burst_index-2);
        }
        else
        {
            load(load_flag, input_1, var_p0, burst_index);
            compute(compute_flag, output_0, input_0, FF, FIFO_124, FIFO_125, burst_index-1, tile_num_dim0, var_blur_y_extent_0, var_blur_y_extent_1, var_blur_y_min_0, var_blur_y_min_1);
            store(store_flag, var_blur_y, output_1, burst_index-2);
        }
        //fprintf(stderr, "burst_index: %d, TILE_SIZE_DIM0*TILE_SIZE_DIM1/BURST_LENGTH: %d\n", burst_index, TILE_SIZE_DIM0*TILE_SIZE_DIM1/BURST_LENGTH);
        /*
        if(burst_index == TILE_SIZE_DIM0*TILE_SIZE_DIM1/BURST_LENGTH)
        {
            tile_index_dim0 += 1;
            if(tile_index_dim0==tile_num_dim0)
            {
                tile_index_dim0 = 0;
                tile_index_dim1 += 1;
            }
            var_p0 += TILE_SIZE_DIM0*TILE_SIZE_DIM1/(BURST_WIDTH/PIXEL_WIDTH);
            var_blur_y += TILE_SIZE_DIM0*TILE_SIZE_DIM1/(BURST_WIDTH/PIXEL_WIDTH);
        }*/
    }
}

}
