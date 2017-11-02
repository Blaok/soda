#include<float.h>
#include<math.h>
#include<stdbool.h>
#include<stddef.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>
#include<ap_int.h>

#include"blur_params.h"

typedef uint16_t input_type;
typedef uint16_t output_type;
#if UNROLL_FACTOR != 64
#error UNROLL_FACTOR != 64
#endif//UNROLL_FACTOR != 64
#if TILE_SIZE_DIM_0 != 2000
#error TILE_SIZE_DIM_0 != 2000
#endif//TILE_SIZE_DIM_0 != 2000

template<int chan_idx, int chan_tot>
void load(bool load_flag, input_type to[CHANNEL_NUM_I][BURST_LENGTH*chan_tot], ap_uint<BURST_WIDTH>* from)
{
    if(load_flag)
    {
load_channel:
        for(int c = 0; c < CHANNEL_NUM_I; ++c)
        {
load_epoch:
            for(int i = 0; i < BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I); ++i)
            {
#pragma HLS pipeline II=1
                ap_uint<BURST_WIDTH> tmp(from[c*(BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O))+i]);
load_coalesced:
                for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_I; ++j)
                {
#pragma HLS unroll
                    to[c][(i*BURST_WIDTH/PIXEL_WIDTH_I+j)*chan_tot+chan_idx] = tmp((j+1)*PIXEL_WIDTH_I-1, j*PIXEL_WIDTH_I);
                }
            }
        }
    }
}

template<int chan_idx, int chan_tot>
void store(bool store_flag, ap_uint<BURST_WIDTH>* to, output_type from[CHANNEL_NUM_O][BURST_LENGTH*chan_tot])
{
    if(store_flag)
    {
store_channel:
        for(int c = 0; c < CHANNEL_NUM_O; ++c)
        {
store_epoch:
            for(int i = 0; i < BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O); ++i)
            {
#pragma HLS pipeline II=1
                ap_uint<BURST_WIDTH> tmp;
store_coalesced:
                for(int j = 0; j < BURST_WIDTH/PIXEL_WIDTH_O; ++j)
                {
#pragma HLS unroll
                    tmp((j+1)*PIXEL_WIDTH_O-1, j*PIXEL_WIDTH_O) = from[c][(i*BURST_WIDTH/PIXEL_WIDTH_O+j)*chan_tot+chan_idx];
                }
                to[c*(BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O))+i] = tmp;
            }
        }
    }
}

void compute(bool compute_flag, output_type output[CHANNEL_NUM_O][BURST_LENGTH*4],
    input_type input[CHANNEL_NUM_I][BURST_LENGTH*4],
    input_type FF[CHANNEL_NUM_I][6],
    input_type FIFO_31[CHANNEL_NUM_I][100][31],
    input_type FIFO_32[CHANNEL_NUM_I][28][32],
    int32_t FIFO_ptrs[2],
    int32_t i_base[UNROLL_FACTOR],
    int32_t j_base[UNROLL_FACTOR],
    int32_t p_base,
    int32_t input_index_base)
{
    if(compute_flag)
    {
        int32_t& FIFO_31_ptr = FIFO_ptrs[0];
        int32_t& FIFO_32_ptr = FIFO_ptrs[1];

        input_type input_points[CHANNEL_NUM_I][UNROLL_FACTOR][9];
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][0] <=> (0, 0)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][1] <=> (1, 0)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][2] <=> (2, 0)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][3] <=> (0, 1)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][4] <=> (1, 1)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][5] <=> (2, 1)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][6] <=> (0, 2)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][7] <=> (1, 2)
        //         input_points[CHANNEL_NUM_I][UNROLL_FACTOR][8] <=> (2, 2)
        input_type input_buffer[CHANNEL_NUM_I][UNROLL_FACTOR];
#pragma HLS array_partition variable=input_points complete dim=0
#pragma HLS array_partition variable=input_buffer complete dim=0

        // produce output
compute_epoch:
        for(int32_t epoch = 0; epoch < BURST_LENGTH*4/UNROLL_FACTOR; ++epoch)
        {
#pragma HLS dependence variable=FF inter false
#pragma HLS dependence variable=FIFO_31 inter false
#pragma HLS dependence variable=FIFO_32 inter false
            int32_t input_index = epoch + input_index_base;
#pragma HLS pipeline II=1
compute_load_channel:
            for(int32_t c = 0; c<CHANNEL_NUM_I; ++c)
            {
#pragma HLS unroll
compute_load_unrolled:
                for(int32_t unroll_index = 0; unroll_index<UNROLL_FACTOR; ++unroll_index)
                {
#pragma HLS unroll
                    input_buffer[c][unroll_index] = input[c][epoch*UNROLL_FACTOR+unroll_index];
                }
            }

            input_type FIFO_31_0_0 = FIFO_31[0][0][FIFO_31_ptr];
            input_type FIFO_31_0_1 = FIFO_31[0][1][FIFO_31_ptr];
            input_type FIFO_31_0_2 = FIFO_31[0][2][FIFO_31_ptr];
            input_type FIFO_31_0_3 = FIFO_31[0][3][FIFO_31_ptr];
            input_type FIFO_31_0_4 = FIFO_31[0][4][FIFO_31_ptr];
            input_type FIFO_31_0_5 = FIFO_31[0][5][FIFO_31_ptr];
            input_type FIFO_31_0_6 = FIFO_31[0][6][FIFO_31_ptr];
            input_type FIFO_31_0_7 = FIFO_31[0][7][FIFO_31_ptr];
            input_type FIFO_31_0_8 = FIFO_31[0][8][FIFO_31_ptr];
            input_type FIFO_31_0_9 = FIFO_31[0][9][FIFO_31_ptr];
            input_type FIFO_31_0_10 = FIFO_31[0][10][FIFO_31_ptr];
            input_type FIFO_31_0_11 = FIFO_31[0][11][FIFO_31_ptr];
            input_type FIFO_31_0_12 = FIFO_31[0][12][FIFO_31_ptr];
            input_type FIFO_31_0_13 = FIFO_31[0][13][FIFO_31_ptr];
            input_type FIFO_31_0_14 = FIFO_31[0][14][FIFO_31_ptr];
            input_type FIFO_31_0_15 = FIFO_31[0][15][FIFO_31_ptr];
            input_type FIFO_31_0_16 = FIFO_31[0][16][FIFO_31_ptr];
            input_type FIFO_31_0_17 = FIFO_31[0][17][FIFO_31_ptr];
            input_type FIFO_31_0_18 = FIFO_31[0][18][FIFO_31_ptr];
            input_type FIFO_31_0_19 = FIFO_31[0][19][FIFO_31_ptr];
            input_type FIFO_31_0_20 = FIFO_31[0][20][FIFO_31_ptr];
            input_type FIFO_31_0_21 = FIFO_31[0][21][FIFO_31_ptr];
            input_type FIFO_31_0_22 = FIFO_31[0][22][FIFO_31_ptr];
            input_type FIFO_31_0_23 = FIFO_31[0][23][FIFO_31_ptr];
            input_type FIFO_31_0_24 = FIFO_31[0][24][FIFO_31_ptr];
            input_type FIFO_31_0_25 = FIFO_31[0][25][FIFO_31_ptr];
            input_type FIFO_31_0_26 = FIFO_31[0][26][FIFO_31_ptr];
            input_type FIFO_31_0_27 = FIFO_31[0][27][FIFO_31_ptr];
            input_type FIFO_31_0_28 = FIFO_31[0][28][FIFO_31_ptr];
            input_type FIFO_31_0_29 = FIFO_31[0][29][FIFO_31_ptr];
            input_type FIFO_31_0_30 = FIFO_31[0][30][FIFO_31_ptr];
            input_type FIFO_31_0_31 = FIFO_31[0][31][FIFO_31_ptr];
            input_type FIFO_31_0_32 = FIFO_31[0][32][FIFO_31_ptr];
            input_type FIFO_31_0_33 = FIFO_31[0][33][FIFO_31_ptr];
            input_type FIFO_31_0_34 = FIFO_31[0][34][FIFO_31_ptr];
            input_type FIFO_31_0_35 = FIFO_31[0][35][FIFO_31_ptr];
            input_type FIFO_31_0_36 = FIFO_31[0][36][FIFO_31_ptr];
            input_type FIFO_31_0_37 = FIFO_31[0][37][FIFO_31_ptr];
            input_type FIFO_31_0_38 = FIFO_31[0][38][FIFO_31_ptr];
            input_type FIFO_31_0_39 = FIFO_31[0][39][FIFO_31_ptr];
            input_type FIFO_31_0_40 = FIFO_31[0][40][FIFO_31_ptr];
            input_type FIFO_31_0_41 = FIFO_31[0][41][FIFO_31_ptr];
            input_type FIFO_31_0_42 = FIFO_31[0][42][FIFO_31_ptr];
            input_type FIFO_31_0_43 = FIFO_31[0][43][FIFO_31_ptr];
            input_type FIFO_31_0_44 = FIFO_31[0][44][FIFO_31_ptr];
            input_type FIFO_31_0_45 = FIFO_31[0][45][FIFO_31_ptr];
            input_type FIFO_31_0_46 = FIFO_31[0][46][FIFO_31_ptr];
            input_type FIFO_31_0_47 = FIFO_31[0][47][FIFO_31_ptr];
            input_type FIFO_31_0_48 = FIFO_31[0][48][FIFO_31_ptr];
            input_type FIFO_31_0_49 = FIFO_31[0][49][FIFO_31_ptr];
            input_type FIFO_31_0_50 = FIFO_31[0][50][FIFO_31_ptr];
            input_type FIFO_31_0_51 = FIFO_31[0][51][FIFO_31_ptr];
            input_type FIFO_31_0_52 = FIFO_31[0][52][FIFO_31_ptr];
            input_type FIFO_31_0_53 = FIFO_31[0][53][FIFO_31_ptr];
            input_type FIFO_31_0_54 = FIFO_31[0][54][FIFO_31_ptr];
            input_type FIFO_31_0_55 = FIFO_31[0][55][FIFO_31_ptr];
            input_type FIFO_31_0_56 = FIFO_31[0][56][FIFO_31_ptr];
            input_type FIFO_31_0_57 = FIFO_31[0][57][FIFO_31_ptr];
            input_type FIFO_31_0_58 = FIFO_31[0][58][FIFO_31_ptr];
            input_type FIFO_31_0_59 = FIFO_31[0][59][FIFO_31_ptr];
            input_type FIFO_31_0_60 = FIFO_31[0][60][FIFO_31_ptr];
            input_type FIFO_31_0_61 = FIFO_31[0][61][FIFO_31_ptr];
            input_type FIFO_31_0_62 = FIFO_31[0][62][FIFO_31_ptr];
            input_type FIFO_31_0_63 = FIFO_31[0][63][FIFO_31_ptr];
            input_type FIFO_31_0_64 = FIFO_31[0][64][FIFO_31_ptr];
            input_type FIFO_31_0_65 = FIFO_31[0][65][FIFO_31_ptr];
            input_type FIFO_31_0_66 = FIFO_31[0][66][FIFO_31_ptr];
            input_type FIFO_31_0_67 = FIFO_31[0][67][FIFO_31_ptr];
            input_type FIFO_31_0_68 = FIFO_31[0][68][FIFO_31_ptr];
            input_type FIFO_31_0_69 = FIFO_31[0][69][FIFO_31_ptr];
            input_type FIFO_31_0_70 = FIFO_31[0][70][FIFO_31_ptr];
            input_type FIFO_31_0_71 = FIFO_31[0][71][FIFO_31_ptr];
            input_type FIFO_31_0_72 = FIFO_31[0][72][FIFO_31_ptr];
            input_type FIFO_31_0_73 = FIFO_31[0][73][FIFO_31_ptr];
            input_type FIFO_31_0_74 = FIFO_31[0][74][FIFO_31_ptr];
            input_type FIFO_31_0_75 = FIFO_31[0][75][FIFO_31_ptr];
            input_type FIFO_31_0_76 = FIFO_31[0][76][FIFO_31_ptr];
            input_type FIFO_31_0_77 = FIFO_31[0][77][FIFO_31_ptr];
            input_type FIFO_31_0_78 = FIFO_31[0][78][FIFO_31_ptr];
            input_type FIFO_31_0_79 = FIFO_31[0][79][FIFO_31_ptr];
            input_type FIFO_31_0_80 = FIFO_31[0][80][FIFO_31_ptr];
            input_type FIFO_31_0_81 = FIFO_31[0][81][FIFO_31_ptr];
            input_type FIFO_31_0_82 = FIFO_31[0][82][FIFO_31_ptr];
            input_type FIFO_31_0_83 = FIFO_31[0][83][FIFO_31_ptr];
            input_type FIFO_31_0_84 = FIFO_31[0][84][FIFO_31_ptr];
            input_type FIFO_31_0_85 = FIFO_31[0][85][FIFO_31_ptr];
            input_type FIFO_31_0_86 = FIFO_31[0][86][FIFO_31_ptr];
            input_type FIFO_31_0_87 = FIFO_31[0][87][FIFO_31_ptr];
            input_type FIFO_31_0_88 = FIFO_31[0][88][FIFO_31_ptr];
            input_type FIFO_31_0_89 = FIFO_31[0][89][FIFO_31_ptr];
            input_type FIFO_31_0_90 = FIFO_31[0][90][FIFO_31_ptr];
            input_type FIFO_31_0_91 = FIFO_31[0][91][FIFO_31_ptr];
            input_type FIFO_31_0_92 = FIFO_31[0][92][FIFO_31_ptr];
            input_type FIFO_31_0_93 = FIFO_31[0][93][FIFO_31_ptr];
            input_type FIFO_31_0_94 = FIFO_31[0][94][FIFO_31_ptr];
            input_type FIFO_31_0_95 = FIFO_31[0][95][FIFO_31_ptr];
            input_type FIFO_31_0_96 = FIFO_31[0][96][FIFO_31_ptr];
            input_type FIFO_31_0_97 = FIFO_31[0][97][FIFO_31_ptr];
            input_type FIFO_31_0_98 = FIFO_31[0][98][FIFO_31_ptr];
            input_type FIFO_31_0_99 = FIFO_31[0][99][FIFO_31_ptr];
            input_type FIFO_32_0_0 = FIFO_32[0][0][FIFO_32_ptr];
            input_type FIFO_32_0_1 = FIFO_32[0][1][FIFO_32_ptr];
            input_type FIFO_32_0_2 = FIFO_32[0][2][FIFO_32_ptr];
            input_type FIFO_32_0_3 = FIFO_32[0][3][FIFO_32_ptr];
            input_type FIFO_32_0_4 = FIFO_32[0][4][FIFO_32_ptr];
            input_type FIFO_32_0_5 = FIFO_32[0][5][FIFO_32_ptr];
            input_type FIFO_32_0_6 = FIFO_32[0][6][FIFO_32_ptr];
            input_type FIFO_32_0_7 = FIFO_32[0][7][FIFO_32_ptr];
            input_type FIFO_32_0_8 = FIFO_32[0][8][FIFO_32_ptr];
            input_type FIFO_32_0_9 = FIFO_32[0][9][FIFO_32_ptr];
            input_type FIFO_32_0_10 = FIFO_32[0][10][FIFO_32_ptr];
            input_type FIFO_32_0_11 = FIFO_32[0][11][FIFO_32_ptr];
            input_type FIFO_32_0_12 = FIFO_32[0][12][FIFO_32_ptr];
            input_type FIFO_32_0_13 = FIFO_32[0][13][FIFO_32_ptr];
            input_type FIFO_32_0_14 = FIFO_32[0][14][FIFO_32_ptr];
            input_type FIFO_32_0_15 = FIFO_32[0][15][FIFO_32_ptr];
            input_type FIFO_32_0_16 = FIFO_32[0][16][FIFO_32_ptr];
            input_type FIFO_32_0_17 = FIFO_32[0][17][FIFO_32_ptr];
            input_type FIFO_32_0_18 = FIFO_32[0][18][FIFO_32_ptr];
            input_type FIFO_32_0_19 = FIFO_32[0][19][FIFO_32_ptr];
            input_type FIFO_32_0_20 = FIFO_32[0][20][FIFO_32_ptr];
            input_type FIFO_32_0_21 = FIFO_32[0][21][FIFO_32_ptr];
            input_type FIFO_32_0_22 = FIFO_32[0][22][FIFO_32_ptr];
            input_type FIFO_32_0_23 = FIFO_32[0][23][FIFO_32_ptr];
            input_type FIFO_32_0_24 = FIFO_32[0][24][FIFO_32_ptr];
            input_type FIFO_32_0_25 = FIFO_32[0][25][FIFO_32_ptr];
            input_type FIFO_32_0_26 = FIFO_32[0][26][FIFO_32_ptr];
            input_type FIFO_32_0_27 = FIFO_32[0][27][FIFO_32_ptr];

            input_points[0][0][8] = input_buffer[0][0]; // (2, 2)
            input_points[0][1][7] = input_buffer[0][0]; // (1, 2)
            input_points[0][2][6] = input_buffer[0][0]; // (0, 2)
            input_points[0][1][8] = input_buffer[0][1]; // (2, 2)
            input_points[0][2][7] = input_buffer[0][1]; // (1, 2)
            input_points[0][3][6] = input_buffer[0][1]; // (0, 2)
            input_points[0][2][8] = input_buffer[0][2]; // (2, 2)
            input_points[0][3][7] = input_buffer[0][2]; // (1, 2)
            input_points[0][4][6] = input_buffer[0][2]; // (0, 2)
            input_points[0][3][8] = input_buffer[0][3]; // (2, 2)
            input_points[0][4][7] = input_buffer[0][3]; // (1, 2)
            input_points[0][5][6] = input_buffer[0][3]; // (0, 2)
            input_points[0][4][8] = input_buffer[0][4]; // (2, 2)
            input_points[0][5][7] = input_buffer[0][4]; // (1, 2)
            input_points[0][6][6] = input_buffer[0][4]; // (0, 2)
            input_points[0][5][8] = input_buffer[0][5]; // (2, 2)
            input_points[0][6][7] = input_buffer[0][5]; // (1, 2)
            input_points[0][7][6] = input_buffer[0][5]; // (0, 2)
            input_points[0][6][8] = input_buffer[0][6]; // (2, 2)
            input_points[0][7][7] = input_buffer[0][6]; // (1, 2)
            input_points[0][8][6] = input_buffer[0][6]; // (0, 2)
            input_points[0][7][8] = input_buffer[0][7]; // (2, 2)
            input_points[0][8][7] = input_buffer[0][7]; // (1, 2)
            input_points[0][9][6] = input_buffer[0][7]; // (0, 2)
            input_points[0][8][8] = input_buffer[0][8]; // (2, 2)
            input_points[0][9][7] = input_buffer[0][8]; // (1, 2)
            input_points[0][10][6] = input_buffer[0][8]; // (0, 2)
            input_points[0][9][8] = input_buffer[0][9]; // (2, 2)
            input_points[0][10][7] = input_buffer[0][9]; // (1, 2)
            input_points[0][11][6] = input_buffer[0][9]; // (0, 2)
            input_points[0][10][8] = input_buffer[0][10]; // (2, 2)
            input_points[0][11][7] = input_buffer[0][10]; // (1, 2)
            input_points[0][12][6] = input_buffer[0][10]; // (0, 2)
            input_points[0][11][8] = input_buffer[0][11]; // (2, 2)
            input_points[0][12][7] = input_buffer[0][11]; // (1, 2)
            input_points[0][13][6] = input_buffer[0][11]; // (0, 2)
            input_points[0][12][8] = input_buffer[0][12]; // (2, 2)
            input_points[0][13][7] = input_buffer[0][12]; // (1, 2)
            input_points[0][14][6] = input_buffer[0][12]; // (0, 2)
            input_points[0][13][8] = input_buffer[0][13]; // (2, 2)
            input_points[0][14][7] = input_buffer[0][13]; // (1, 2)
            input_points[0][15][6] = input_buffer[0][13]; // (0, 2)
            input_points[0][14][8] = input_buffer[0][14]; // (2, 2)
            input_points[0][15][7] = input_buffer[0][14]; // (1, 2)
            input_points[0][16][6] = input_buffer[0][14]; // (0, 2)
            input_points[0][15][8] = input_buffer[0][15]; // (2, 2)
            input_points[0][16][7] = input_buffer[0][15]; // (1, 2)
            input_points[0][17][6] = input_buffer[0][15]; // (0, 2)
            input_points[0][16][8] = input_buffer[0][16]; // (2, 2)
            input_points[0][17][7] = input_buffer[0][16]; // (1, 2)
            input_points[0][18][6] = input_buffer[0][16]; // (0, 2)
            input_points[0][17][8] = input_buffer[0][17]; // (2, 2)
            input_points[0][18][7] = input_buffer[0][17]; // (1, 2)
            input_points[0][19][6] = input_buffer[0][17]; // (0, 2)
            input_points[0][18][8] = input_buffer[0][18]; // (2, 2)
            input_points[0][19][7] = input_buffer[0][18]; // (1, 2)
            input_points[0][20][6] = input_buffer[0][18]; // (0, 2)
            input_points[0][19][8] = input_buffer[0][19]; // (2, 2)
            input_points[0][20][7] = input_buffer[0][19]; // (1, 2)
            input_points[0][21][6] = input_buffer[0][19]; // (0, 2)
            input_points[0][20][8] = input_buffer[0][20]; // (2, 2)
            input_points[0][21][7] = input_buffer[0][20]; // (1, 2)
            input_points[0][22][6] = input_buffer[0][20]; // (0, 2)
            input_points[0][21][8] = input_buffer[0][21]; // (2, 2)
            input_points[0][22][7] = input_buffer[0][21]; // (1, 2)
            input_points[0][23][6] = input_buffer[0][21]; // (0, 2)
            input_points[0][22][8] = input_buffer[0][22]; // (2, 2)
            input_points[0][23][7] = input_buffer[0][22]; // (1, 2)
            input_points[0][24][6] = input_buffer[0][22]; // (0, 2)
            input_points[0][23][8] = input_buffer[0][23]; // (2, 2)
            input_points[0][24][7] = input_buffer[0][23]; // (1, 2)
            input_points[0][25][6] = input_buffer[0][23]; // (0, 2)
            input_points[0][24][8] = input_buffer[0][24]; // (2, 2)
            input_points[0][25][7] = input_buffer[0][24]; // (1, 2)
            input_points[0][26][6] = input_buffer[0][24]; // (0, 2)
            input_points[0][25][8] = input_buffer[0][25]; // (2, 2)
            input_points[0][26][7] = input_buffer[0][25]; // (1, 2)
            input_points[0][27][6] = input_buffer[0][25]; // (0, 2)
            input_points[0][26][8] = input_buffer[0][26]; // (2, 2)
            input_points[0][27][7] = input_buffer[0][26]; // (1, 2)
            input_points[0][28][6] = input_buffer[0][26]; // (0, 2)
            input_points[0][27][8] = input_buffer[0][27]; // (2, 2)
            input_points[0][28][7] = input_buffer[0][27]; // (1, 2)
            input_points[0][29][6] = input_buffer[0][27]; // (0, 2)
            input_points[0][28][8] = input_buffer[0][28]; // (2, 2)
            input_points[0][29][7] = input_buffer[0][28]; // (1, 2)
            input_points[0][30][6] = input_buffer[0][28]; // (0, 2)
            input_points[0][29][8] = input_buffer[0][29]; // (2, 2)
            input_points[0][30][7] = input_buffer[0][29]; // (1, 2)
            input_points[0][31][6] = input_buffer[0][29]; // (0, 2)
            input_points[0][30][8] = input_buffer[0][30]; // (2, 2)
            input_points[0][31][7] = input_buffer[0][30]; // (1, 2)
            input_points[0][32][6] = input_buffer[0][30]; // (0, 2)
            input_points[0][31][8] = input_buffer[0][31]; // (2, 2)
            input_points[0][32][7] = input_buffer[0][31]; // (1, 2)
            input_points[0][33][6] = input_buffer[0][31]; // (0, 2)
            input_points[0][32][8] = input_buffer[0][32]; // (2, 2)
            input_points[0][33][7] = input_buffer[0][32]; // (1, 2)
            input_points[0][34][6] = input_buffer[0][32]; // (0, 2)
            input_points[0][33][8] = input_buffer[0][33]; // (2, 2)
            input_points[0][34][7] = input_buffer[0][33]; // (1, 2)
            input_points[0][35][6] = input_buffer[0][33]; // (0, 2)
            input_points[0][34][8] = input_buffer[0][34]; // (2, 2)
            input_points[0][35][7] = input_buffer[0][34]; // (1, 2)
            input_points[0][36][6] = input_buffer[0][34]; // (0, 2)
            input_points[0][35][8] = input_buffer[0][35]; // (2, 2)
            input_points[0][36][7] = input_buffer[0][35]; // (1, 2)
            input_points[0][37][6] = input_buffer[0][35]; // (0, 2)
            input_points[0][36][8] = input_buffer[0][36]; // (2, 2)
            input_points[0][37][7] = input_buffer[0][36]; // (1, 2)
            input_points[0][38][6] = input_buffer[0][36]; // (0, 2)
            input_points[0][37][8] = input_buffer[0][37]; // (2, 2)
            input_points[0][38][7] = input_buffer[0][37]; // (1, 2)
            input_points[0][39][6] = input_buffer[0][37]; // (0, 2)
            input_points[0][38][8] = input_buffer[0][38]; // (2, 2)
            input_points[0][39][7] = input_buffer[0][38]; // (1, 2)
            input_points[0][40][6] = input_buffer[0][38]; // (0, 2)
            input_points[0][39][8] = input_buffer[0][39]; // (2, 2)
            input_points[0][40][7] = input_buffer[0][39]; // (1, 2)
            input_points[0][41][6] = input_buffer[0][39]; // (0, 2)
            input_points[0][40][8] = input_buffer[0][40]; // (2, 2)
            input_points[0][41][7] = input_buffer[0][40]; // (1, 2)
            input_points[0][42][6] = input_buffer[0][40]; // (0, 2)
            input_points[0][41][8] = input_buffer[0][41]; // (2, 2)
            input_points[0][42][7] = input_buffer[0][41]; // (1, 2)
            input_points[0][43][6] = input_buffer[0][41]; // (0, 2)
            input_points[0][42][8] = input_buffer[0][42]; // (2, 2)
            input_points[0][43][7] = input_buffer[0][42]; // (1, 2)
            input_points[0][44][6] = input_buffer[0][42]; // (0, 2)
            input_points[0][43][8] = input_buffer[0][43]; // (2, 2)
            input_points[0][44][7] = input_buffer[0][43]; // (1, 2)
            input_points[0][45][6] = input_buffer[0][43]; // (0, 2)
            input_points[0][44][8] = input_buffer[0][44]; // (2, 2)
            input_points[0][45][7] = input_buffer[0][44]; // (1, 2)
            input_points[0][46][6] = input_buffer[0][44]; // (0, 2)
            input_points[0][45][8] = input_buffer[0][45]; // (2, 2)
            input_points[0][46][7] = input_buffer[0][45]; // (1, 2)
            input_points[0][47][6] = input_buffer[0][45]; // (0, 2)
            input_points[0][46][8] = input_buffer[0][46]; // (2, 2)
            input_points[0][47][7] = input_buffer[0][46]; // (1, 2)
            input_points[0][48][6] = input_buffer[0][46]; // (0, 2)
            input_points[0][47][8] = input_buffer[0][47]; // (2, 2)
            input_points[0][48][7] = input_buffer[0][47]; // (1, 2)
            input_points[0][49][6] = input_buffer[0][47]; // (0, 2)
            input_points[0][48][8] = input_buffer[0][48]; // (2, 2)
            input_points[0][49][7] = input_buffer[0][48]; // (1, 2)
            input_points[0][50][6] = input_buffer[0][48]; // (0, 2)
            input_points[0][49][8] = input_buffer[0][49]; // (2, 2)
            input_points[0][50][7] = input_buffer[0][49]; // (1, 2)
            input_points[0][51][6] = input_buffer[0][49]; // (0, 2)
            input_points[0][50][8] = input_buffer[0][50]; // (2, 2)
            input_points[0][51][7] = input_buffer[0][50]; // (1, 2)
            input_points[0][52][6] = input_buffer[0][50]; // (0, 2)
            input_points[0][51][8] = input_buffer[0][51]; // (2, 2)
            input_points[0][52][7] = input_buffer[0][51]; // (1, 2)
            input_points[0][53][6] = input_buffer[0][51]; // (0, 2)
            input_points[0][52][8] = input_buffer[0][52]; // (2, 2)
            input_points[0][53][7] = input_buffer[0][52]; // (1, 2)
            input_points[0][54][6] = input_buffer[0][52]; // (0, 2)
            input_points[0][53][8] = input_buffer[0][53]; // (2, 2)
            input_points[0][54][7] = input_buffer[0][53]; // (1, 2)
            input_points[0][55][6] = input_buffer[0][53]; // (0, 2)
            input_points[0][54][8] = input_buffer[0][54]; // (2, 2)
            input_points[0][55][7] = input_buffer[0][54]; // (1, 2)
            input_points[0][56][6] = input_buffer[0][54]; // (0, 2)
            input_points[0][55][8] = input_buffer[0][55]; // (2, 2)
            input_points[0][56][7] = input_buffer[0][55]; // (1, 2)
            input_points[0][57][6] = input_buffer[0][55]; // (0, 2)
            input_points[0][56][8] = input_buffer[0][56]; // (2, 2)
            input_points[0][57][7] = input_buffer[0][56]; // (1, 2)
            input_points[0][58][6] = input_buffer[0][56]; // (0, 2)
            input_points[0][57][8] = input_buffer[0][57]; // (2, 2)
            input_points[0][58][7] = input_buffer[0][57]; // (1, 2)
            input_points[0][59][6] = input_buffer[0][57]; // (0, 2)
            input_points[0][58][8] = input_buffer[0][58]; // (2, 2)
            input_points[0][59][7] = input_buffer[0][58]; // (1, 2)
            input_points[0][60][6] = input_buffer[0][58]; // (0, 2)
            input_points[0][59][8] = input_buffer[0][59]; // (2, 2)
            input_points[0][60][7] = input_buffer[0][59]; // (1, 2)
            input_points[0][61][6] = input_buffer[0][59]; // (0, 2)
            input_points[0][60][8] = input_buffer[0][60]; // (2, 2)
            input_points[0][61][7] = input_buffer[0][60]; // (1, 2)
            input_points[0][62][6] = input_buffer[0][60]; // (0, 2)
            input_points[0][61][8] = input_buffer[0][61]; // (2, 2)
            input_points[0][62][7] = input_buffer[0][61]; // (1, 2)
            input_points[0][63][6] = input_buffer[0][61]; // (0, 2)
            input_points[0][62][8] = input_buffer[0][62]; // (2, 2)
            input_points[0][63][7] = input_buffer[0][62]; // (1, 2)
            input_points[0][63][8] = input_buffer[0][63]; // (2, 2)
            input_points[0][0][0] = FF[0][0]; // (0, 0)
            input_points[0][0][1] = FF[0][1]; // (1, 0)
            input_points[0][1][0] = FF[0][1]; // (0, 0)
            input_points[0][0][3] = FF[0][2]; // (0, 1)
            input_points[0][0][4] = FF[0][3]; // (1, 1)
            input_points[0][1][3] = FF[0][3]; // (0, 1)
            input_points[0][0][6] = FF[0][4]; // (0, 2)
            input_points[0][0][7] = FF[0][5]; // (1, 2)
            input_points[0][1][6] = FF[0][5]; // (0, 2)
            input_points[0][14][2] = FIFO_31_0_0; // (2, 0)
            input_points[0][15][1] = FIFO_31_0_0; // (1, 0)
            input_points[0][16][0] = FIFO_31_0_0; // (0, 0)
            input_points[0][15][2] = FIFO_31_0_1; // (2, 0)
            input_points[0][16][1] = FIFO_31_0_1; // (1, 0)
            input_points[0][17][0] = FIFO_31_0_1; // (0, 0)
            input_points[0][16][2] = FIFO_31_0_2; // (2, 0)
            input_points[0][17][1] = FIFO_31_0_2; // (1, 0)
            input_points[0][18][0] = FIFO_31_0_2; // (0, 0)
            input_points[0][17][2] = FIFO_31_0_3; // (2, 0)
            input_points[0][18][1] = FIFO_31_0_3; // (1, 0)
            input_points[0][19][0] = FIFO_31_0_3; // (0, 0)
            input_points[0][18][2] = FIFO_31_0_4; // (2, 0)
            input_points[0][19][1] = FIFO_31_0_4; // (1, 0)
            input_points[0][20][0] = FIFO_31_0_4; // (0, 0)
            input_points[0][19][2] = FIFO_31_0_5; // (2, 0)
            input_points[0][20][1] = FIFO_31_0_5; // (1, 0)
            input_points[0][21][0] = FIFO_31_0_5; // (0, 0)
            input_points[0][20][2] = FIFO_31_0_6; // (2, 0)
            input_points[0][21][1] = FIFO_31_0_6; // (1, 0)
            input_points[0][22][0] = FIFO_31_0_6; // (0, 0)
            input_points[0][21][2] = FIFO_31_0_7; // (2, 0)
            input_points[0][22][1] = FIFO_31_0_7; // (1, 0)
            input_points[0][23][0] = FIFO_31_0_7; // (0, 0)
            input_points[0][22][2] = FIFO_31_0_8; // (2, 0)
            input_points[0][23][1] = FIFO_31_0_8; // (1, 0)
            input_points[0][24][0] = FIFO_31_0_8; // (0, 0)
            input_points[0][23][2] = FIFO_31_0_9; // (2, 0)
            input_points[0][24][1] = FIFO_31_0_9; // (1, 0)
            input_points[0][25][0] = FIFO_31_0_9; // (0, 0)
            input_points[0][24][2] = FIFO_31_0_10; // (2, 0)
            input_points[0][25][1] = FIFO_31_0_10; // (1, 0)
            input_points[0][26][0] = FIFO_31_0_10; // (0, 0)
            input_points[0][25][2] = FIFO_31_0_11; // (2, 0)
            input_points[0][26][1] = FIFO_31_0_11; // (1, 0)
            input_points[0][27][0] = FIFO_31_0_11; // (0, 0)
            input_points[0][26][2] = FIFO_31_0_12; // (2, 0)
            input_points[0][27][1] = FIFO_31_0_12; // (1, 0)
            input_points[0][28][0] = FIFO_31_0_12; // (0, 0)
            input_points[0][27][2] = FIFO_31_0_13; // (2, 0)
            input_points[0][28][1] = FIFO_31_0_13; // (1, 0)
            input_points[0][29][0] = FIFO_31_0_13; // (0, 0)
            input_points[0][28][2] = FIFO_31_0_14; // (2, 0)
            input_points[0][29][1] = FIFO_31_0_14; // (1, 0)
            input_points[0][30][0] = FIFO_31_0_14; // (0, 0)
            input_points[0][29][2] = FIFO_31_0_15; // (2, 0)
            input_points[0][30][1] = FIFO_31_0_15; // (1, 0)
            input_points[0][31][0] = FIFO_31_0_15; // (0, 0)
            input_points[0][30][2] = FIFO_31_0_16; // (2, 0)
            input_points[0][31][1] = FIFO_31_0_16; // (1, 0)
            input_points[0][32][0] = FIFO_31_0_16; // (0, 0)
            input_points[0][31][2] = FIFO_31_0_17; // (2, 0)
            input_points[0][32][1] = FIFO_31_0_17; // (1, 0)
            input_points[0][33][0] = FIFO_31_0_17; // (0, 0)
            input_points[0][32][2] = FIFO_31_0_18; // (2, 0)
            input_points[0][33][1] = FIFO_31_0_18; // (1, 0)
            input_points[0][34][0] = FIFO_31_0_18; // (0, 0)
            input_points[0][33][2] = FIFO_31_0_19; // (2, 0)
            input_points[0][34][1] = FIFO_31_0_19; // (1, 0)
            input_points[0][35][0] = FIFO_31_0_19; // (0, 0)
            input_points[0][34][2] = FIFO_31_0_20; // (2, 0)
            input_points[0][35][1] = FIFO_31_0_20; // (1, 0)
            input_points[0][36][0] = FIFO_31_0_20; // (0, 0)
            input_points[0][35][2] = FIFO_31_0_21; // (2, 0)
            input_points[0][36][1] = FIFO_31_0_21; // (1, 0)
            input_points[0][37][0] = FIFO_31_0_21; // (0, 0)
            input_points[0][36][2] = FIFO_31_0_22; // (2, 0)
            input_points[0][37][1] = FIFO_31_0_22; // (1, 0)
            input_points[0][38][0] = FIFO_31_0_22; // (0, 0)
            input_points[0][37][2] = FIFO_31_0_23; // (2, 0)
            input_points[0][38][1] = FIFO_31_0_23; // (1, 0)
            input_points[0][39][0] = FIFO_31_0_23; // (0, 0)
            input_points[0][38][2] = FIFO_31_0_24; // (2, 0)
            input_points[0][39][1] = FIFO_31_0_24; // (1, 0)
            input_points[0][40][0] = FIFO_31_0_24; // (0, 0)
            input_points[0][39][2] = FIFO_31_0_25; // (2, 0)
            input_points[0][40][1] = FIFO_31_0_25; // (1, 0)
            input_points[0][41][0] = FIFO_31_0_25; // (0, 0)
            input_points[0][40][2] = FIFO_31_0_26; // (2, 0)
            input_points[0][41][1] = FIFO_31_0_26; // (1, 0)
            input_points[0][42][0] = FIFO_31_0_26; // (0, 0)
            input_points[0][41][2] = FIFO_31_0_27; // (2, 0)
            input_points[0][42][1] = FIFO_31_0_27; // (1, 0)
            input_points[0][43][0] = FIFO_31_0_27; // (0, 0)
            input_points[0][42][2] = FIFO_31_0_28; // (2, 0)
            input_points[0][43][1] = FIFO_31_0_28; // (1, 0)
            input_points[0][44][0] = FIFO_31_0_28; // (0, 0)
            input_points[0][43][2] = FIFO_31_0_29; // (2, 0)
            input_points[0][44][1] = FIFO_31_0_29; // (1, 0)
            input_points[0][45][0] = FIFO_31_0_29; // (0, 0)
            input_points[0][44][2] = FIFO_31_0_30; // (2, 0)
            input_points[0][45][1] = FIFO_31_0_30; // (1, 0)
            input_points[0][46][0] = FIFO_31_0_30; // (0, 0)
            input_points[0][45][2] = FIFO_31_0_31; // (2, 0)
            input_points[0][46][1] = FIFO_31_0_31; // (1, 0)
            input_points[0][47][0] = FIFO_31_0_31; // (0, 0)
            input_points[0][46][2] = FIFO_31_0_32; // (2, 0)
            input_points[0][47][1] = FIFO_31_0_32; // (1, 0)
            input_points[0][48][0] = FIFO_31_0_32; // (0, 0)
            input_points[0][47][2] = FIFO_31_0_33; // (2, 0)
            input_points[0][48][1] = FIFO_31_0_33; // (1, 0)
            input_points[0][49][0] = FIFO_31_0_33; // (0, 0)
            input_points[0][48][2] = FIFO_31_0_34; // (2, 0)
            input_points[0][49][1] = FIFO_31_0_34; // (1, 0)
            input_points[0][50][0] = FIFO_31_0_34; // (0, 0)
            input_points[0][49][2] = FIFO_31_0_35; // (2, 0)
            input_points[0][50][1] = FIFO_31_0_35; // (1, 0)
            input_points[0][51][0] = FIFO_31_0_35; // (0, 0)
            input_points[0][50][2] = FIFO_31_0_36; // (2, 0)
            input_points[0][51][1] = FIFO_31_0_36; // (1, 0)
            input_points[0][52][0] = FIFO_31_0_36; // (0, 0)
            input_points[0][51][2] = FIFO_31_0_37; // (2, 0)
            input_points[0][52][1] = FIFO_31_0_37; // (1, 0)
            input_points[0][53][0] = FIFO_31_0_37; // (0, 0)
            input_points[0][52][2] = FIFO_31_0_38; // (2, 0)
            input_points[0][53][1] = FIFO_31_0_38; // (1, 0)
            input_points[0][54][0] = FIFO_31_0_38; // (0, 0)
            input_points[0][53][2] = FIFO_31_0_39; // (2, 0)
            input_points[0][54][1] = FIFO_31_0_39; // (1, 0)
            input_points[0][55][0] = FIFO_31_0_39; // (0, 0)
            input_points[0][54][2] = FIFO_31_0_40; // (2, 0)
            input_points[0][55][1] = FIFO_31_0_40; // (1, 0)
            input_points[0][56][0] = FIFO_31_0_40; // (0, 0)
            input_points[0][55][2] = FIFO_31_0_41; // (2, 0)
            input_points[0][56][1] = FIFO_31_0_41; // (1, 0)
            input_points[0][57][0] = FIFO_31_0_41; // (0, 0)
            input_points[0][56][2] = FIFO_31_0_42; // (2, 0)
            input_points[0][57][1] = FIFO_31_0_42; // (1, 0)
            input_points[0][58][0] = FIFO_31_0_42; // (0, 0)
            input_points[0][57][2] = FIFO_31_0_43; // (2, 0)
            input_points[0][58][1] = FIFO_31_0_43; // (1, 0)
            input_points[0][59][0] = FIFO_31_0_43; // (0, 0)
            input_points[0][58][2] = FIFO_31_0_44; // (2, 0)
            input_points[0][59][1] = FIFO_31_0_44; // (1, 0)
            input_points[0][60][0] = FIFO_31_0_44; // (0, 0)
            input_points[0][59][2] = FIFO_31_0_45; // (2, 0)
            input_points[0][60][1] = FIFO_31_0_45; // (1, 0)
            input_points[0][61][0] = FIFO_31_0_45; // (0, 0)
            input_points[0][60][2] = FIFO_31_0_46; // (2, 0)
            input_points[0][61][1] = FIFO_31_0_46; // (1, 0)
            input_points[0][62][0] = FIFO_31_0_46; // (0, 0)
            input_points[0][61][2] = FIFO_31_0_47; // (2, 0)
            input_points[0][62][1] = FIFO_31_0_47; // (1, 0)
            input_points[0][63][0] = FIFO_31_0_47; // (0, 0)
            input_points[0][62][2] = FIFO_31_0_48; // (2, 0)
            input_points[0][63][1] = FIFO_31_0_48; // (1, 0)
            input_points[0][63][2] = FIFO_31_0_49; // (2, 0)
            input_points[0][14][5] = FIFO_31_0_50; // (2, 1)
            input_points[0][15][4] = FIFO_31_0_50; // (1, 1)
            input_points[0][16][3] = FIFO_31_0_50; // (0, 1)
            input_points[0][15][5] = FIFO_31_0_51; // (2, 1)
            input_points[0][16][4] = FIFO_31_0_51; // (1, 1)
            input_points[0][17][3] = FIFO_31_0_51; // (0, 1)
            input_points[0][16][5] = FIFO_31_0_52; // (2, 1)
            input_points[0][17][4] = FIFO_31_0_52; // (1, 1)
            input_points[0][18][3] = FIFO_31_0_52; // (0, 1)
            input_points[0][17][5] = FIFO_31_0_53; // (2, 1)
            input_points[0][18][4] = FIFO_31_0_53; // (1, 1)
            input_points[0][19][3] = FIFO_31_0_53; // (0, 1)
            input_points[0][18][5] = FIFO_31_0_54; // (2, 1)
            input_points[0][19][4] = FIFO_31_0_54; // (1, 1)
            input_points[0][20][3] = FIFO_31_0_54; // (0, 1)
            input_points[0][19][5] = FIFO_31_0_55; // (2, 1)
            input_points[0][20][4] = FIFO_31_0_55; // (1, 1)
            input_points[0][21][3] = FIFO_31_0_55; // (0, 1)
            input_points[0][20][5] = FIFO_31_0_56; // (2, 1)
            input_points[0][21][4] = FIFO_31_0_56; // (1, 1)
            input_points[0][22][3] = FIFO_31_0_56; // (0, 1)
            input_points[0][21][5] = FIFO_31_0_57; // (2, 1)
            input_points[0][22][4] = FIFO_31_0_57; // (1, 1)
            input_points[0][23][3] = FIFO_31_0_57; // (0, 1)
            input_points[0][22][5] = FIFO_31_0_58; // (2, 1)
            input_points[0][23][4] = FIFO_31_0_58; // (1, 1)
            input_points[0][24][3] = FIFO_31_0_58; // (0, 1)
            input_points[0][23][5] = FIFO_31_0_59; // (2, 1)
            input_points[0][24][4] = FIFO_31_0_59; // (1, 1)
            input_points[0][25][3] = FIFO_31_0_59; // (0, 1)
            input_points[0][24][5] = FIFO_31_0_60; // (2, 1)
            input_points[0][25][4] = FIFO_31_0_60; // (1, 1)
            input_points[0][26][3] = FIFO_31_0_60; // (0, 1)
            input_points[0][25][5] = FIFO_31_0_61; // (2, 1)
            input_points[0][26][4] = FIFO_31_0_61; // (1, 1)
            input_points[0][27][3] = FIFO_31_0_61; // (0, 1)
            input_points[0][26][5] = FIFO_31_0_62; // (2, 1)
            input_points[0][27][4] = FIFO_31_0_62; // (1, 1)
            input_points[0][28][3] = FIFO_31_0_62; // (0, 1)
            input_points[0][27][5] = FIFO_31_0_63; // (2, 1)
            input_points[0][28][4] = FIFO_31_0_63; // (1, 1)
            input_points[0][29][3] = FIFO_31_0_63; // (0, 1)
            input_points[0][28][5] = FIFO_31_0_64; // (2, 1)
            input_points[0][29][4] = FIFO_31_0_64; // (1, 1)
            input_points[0][30][3] = FIFO_31_0_64; // (0, 1)
            input_points[0][29][5] = FIFO_31_0_65; // (2, 1)
            input_points[0][30][4] = FIFO_31_0_65; // (1, 1)
            input_points[0][31][3] = FIFO_31_0_65; // (0, 1)
            input_points[0][30][5] = FIFO_31_0_66; // (2, 1)
            input_points[0][31][4] = FIFO_31_0_66; // (1, 1)
            input_points[0][32][3] = FIFO_31_0_66; // (0, 1)
            input_points[0][31][5] = FIFO_31_0_67; // (2, 1)
            input_points[0][32][4] = FIFO_31_0_67; // (1, 1)
            input_points[0][33][3] = FIFO_31_0_67; // (0, 1)
            input_points[0][32][5] = FIFO_31_0_68; // (2, 1)
            input_points[0][33][4] = FIFO_31_0_68; // (1, 1)
            input_points[0][34][3] = FIFO_31_0_68; // (0, 1)
            input_points[0][33][5] = FIFO_31_0_69; // (2, 1)
            input_points[0][34][4] = FIFO_31_0_69; // (1, 1)
            input_points[0][35][3] = FIFO_31_0_69; // (0, 1)
            input_points[0][34][5] = FIFO_31_0_70; // (2, 1)
            input_points[0][35][4] = FIFO_31_0_70; // (1, 1)
            input_points[0][36][3] = FIFO_31_0_70; // (0, 1)
            input_points[0][35][5] = FIFO_31_0_71; // (2, 1)
            input_points[0][36][4] = FIFO_31_0_71; // (1, 1)
            input_points[0][37][3] = FIFO_31_0_71; // (0, 1)
            input_points[0][36][5] = FIFO_31_0_72; // (2, 1)
            input_points[0][37][4] = FIFO_31_0_72; // (1, 1)
            input_points[0][38][3] = FIFO_31_0_72; // (0, 1)
            input_points[0][37][5] = FIFO_31_0_73; // (2, 1)
            input_points[0][38][4] = FIFO_31_0_73; // (1, 1)
            input_points[0][39][3] = FIFO_31_0_73; // (0, 1)
            input_points[0][38][5] = FIFO_31_0_74; // (2, 1)
            input_points[0][39][4] = FIFO_31_0_74; // (1, 1)
            input_points[0][40][3] = FIFO_31_0_74; // (0, 1)
            input_points[0][39][5] = FIFO_31_0_75; // (2, 1)
            input_points[0][40][4] = FIFO_31_0_75; // (1, 1)
            input_points[0][41][3] = FIFO_31_0_75; // (0, 1)
            input_points[0][40][5] = FIFO_31_0_76; // (2, 1)
            input_points[0][41][4] = FIFO_31_0_76; // (1, 1)
            input_points[0][42][3] = FIFO_31_0_76; // (0, 1)
            input_points[0][41][5] = FIFO_31_0_77; // (2, 1)
            input_points[0][42][4] = FIFO_31_0_77; // (1, 1)
            input_points[0][43][3] = FIFO_31_0_77; // (0, 1)
            input_points[0][42][5] = FIFO_31_0_78; // (2, 1)
            input_points[0][43][4] = FIFO_31_0_78; // (1, 1)
            input_points[0][44][3] = FIFO_31_0_78; // (0, 1)
            input_points[0][43][5] = FIFO_31_0_79; // (2, 1)
            input_points[0][44][4] = FIFO_31_0_79; // (1, 1)
            input_points[0][45][3] = FIFO_31_0_79; // (0, 1)
            input_points[0][44][5] = FIFO_31_0_80; // (2, 1)
            input_points[0][45][4] = FIFO_31_0_80; // (1, 1)
            input_points[0][46][3] = FIFO_31_0_80; // (0, 1)
            input_points[0][45][5] = FIFO_31_0_81; // (2, 1)
            input_points[0][46][4] = FIFO_31_0_81; // (1, 1)
            input_points[0][47][3] = FIFO_31_0_81; // (0, 1)
            input_points[0][46][5] = FIFO_31_0_82; // (2, 1)
            input_points[0][47][4] = FIFO_31_0_82; // (1, 1)
            input_points[0][48][3] = FIFO_31_0_82; // (0, 1)
            input_points[0][47][5] = FIFO_31_0_83; // (2, 1)
            input_points[0][48][4] = FIFO_31_0_83; // (1, 1)
            input_points[0][49][3] = FIFO_31_0_83; // (0, 1)
            input_points[0][48][5] = FIFO_31_0_84; // (2, 1)
            input_points[0][49][4] = FIFO_31_0_84; // (1, 1)
            input_points[0][50][3] = FIFO_31_0_84; // (0, 1)
            input_points[0][49][5] = FIFO_31_0_85; // (2, 1)
            input_points[0][50][4] = FIFO_31_0_85; // (1, 1)
            input_points[0][51][3] = FIFO_31_0_85; // (0, 1)
            input_points[0][50][5] = FIFO_31_0_86; // (2, 1)
            input_points[0][51][4] = FIFO_31_0_86; // (1, 1)
            input_points[0][52][3] = FIFO_31_0_86; // (0, 1)
            input_points[0][51][5] = FIFO_31_0_87; // (2, 1)
            input_points[0][52][4] = FIFO_31_0_87; // (1, 1)
            input_points[0][53][3] = FIFO_31_0_87; // (0, 1)
            input_points[0][52][5] = FIFO_31_0_88; // (2, 1)
            input_points[0][53][4] = FIFO_31_0_88; // (1, 1)
            input_points[0][54][3] = FIFO_31_0_88; // (0, 1)
            input_points[0][53][5] = FIFO_31_0_89; // (2, 1)
            input_points[0][54][4] = FIFO_31_0_89; // (1, 1)
            input_points[0][55][3] = FIFO_31_0_89; // (0, 1)
            input_points[0][54][5] = FIFO_31_0_90; // (2, 1)
            input_points[0][55][4] = FIFO_31_0_90; // (1, 1)
            input_points[0][56][3] = FIFO_31_0_90; // (0, 1)
            input_points[0][55][5] = FIFO_31_0_91; // (2, 1)
            input_points[0][56][4] = FIFO_31_0_91; // (1, 1)
            input_points[0][57][3] = FIFO_31_0_91; // (0, 1)
            input_points[0][56][5] = FIFO_31_0_92; // (2, 1)
            input_points[0][57][4] = FIFO_31_0_92; // (1, 1)
            input_points[0][58][3] = FIFO_31_0_92; // (0, 1)
            input_points[0][57][5] = FIFO_31_0_93; // (2, 1)
            input_points[0][58][4] = FIFO_31_0_93; // (1, 1)
            input_points[0][59][3] = FIFO_31_0_93; // (0, 1)
            input_points[0][58][5] = FIFO_31_0_94; // (2, 1)
            input_points[0][59][4] = FIFO_31_0_94; // (1, 1)
            input_points[0][60][3] = FIFO_31_0_94; // (0, 1)
            input_points[0][59][5] = FIFO_31_0_95; // (2, 1)
            input_points[0][60][4] = FIFO_31_0_95; // (1, 1)
            input_points[0][61][3] = FIFO_31_0_95; // (0, 1)
            input_points[0][60][5] = FIFO_31_0_96; // (2, 1)
            input_points[0][61][4] = FIFO_31_0_96; // (1, 1)
            input_points[0][62][3] = FIFO_31_0_96; // (0, 1)
            input_points[0][61][5] = FIFO_31_0_97; // (2, 1)
            input_points[0][62][4] = FIFO_31_0_97; // (1, 1)
            input_points[0][63][3] = FIFO_31_0_97; // (0, 1)
            input_points[0][62][5] = FIFO_31_0_98; // (2, 1)
            input_points[0][63][4] = FIFO_31_0_98; // (1, 1)
            input_points[0][63][5] = FIFO_31_0_99; // (2, 1)
            input_points[0][0][2] = FIFO_32_0_0; // (2, 0)
            input_points[0][1][1] = FIFO_32_0_0; // (1, 0)
            input_points[0][2][0] = FIFO_32_0_0; // (0, 0)
            input_points[0][1][2] = FIFO_32_0_1; // (2, 0)
            input_points[0][2][1] = FIFO_32_0_1; // (1, 0)
            input_points[0][3][0] = FIFO_32_0_1; // (0, 0)
            input_points[0][2][2] = FIFO_32_0_2; // (2, 0)
            input_points[0][3][1] = FIFO_32_0_2; // (1, 0)
            input_points[0][4][0] = FIFO_32_0_2; // (0, 0)
            input_points[0][3][2] = FIFO_32_0_3; // (2, 0)
            input_points[0][4][1] = FIFO_32_0_3; // (1, 0)
            input_points[0][5][0] = FIFO_32_0_3; // (0, 0)
            input_points[0][4][2] = FIFO_32_0_4; // (2, 0)
            input_points[0][5][1] = FIFO_32_0_4; // (1, 0)
            input_points[0][6][0] = FIFO_32_0_4; // (0, 0)
            input_points[0][5][2] = FIFO_32_0_5; // (2, 0)
            input_points[0][6][1] = FIFO_32_0_5; // (1, 0)
            input_points[0][7][0] = FIFO_32_0_5; // (0, 0)
            input_points[0][6][2] = FIFO_32_0_6; // (2, 0)
            input_points[0][7][1] = FIFO_32_0_6; // (1, 0)
            input_points[0][8][0] = FIFO_32_0_6; // (0, 0)
            input_points[0][7][2] = FIFO_32_0_7; // (2, 0)
            input_points[0][8][1] = FIFO_32_0_7; // (1, 0)
            input_points[0][9][0] = FIFO_32_0_7; // (0, 0)
            input_points[0][8][2] = FIFO_32_0_8; // (2, 0)
            input_points[0][9][1] = FIFO_32_0_8; // (1, 0)
            input_points[0][10][0] = FIFO_32_0_8; // (0, 0)
            input_points[0][9][2] = FIFO_32_0_9; // (2, 0)
            input_points[0][10][1] = FIFO_32_0_9; // (1, 0)
            input_points[0][11][0] = FIFO_32_0_9; // (0, 0)
            input_points[0][10][2] = FIFO_32_0_10; // (2, 0)
            input_points[0][11][1] = FIFO_32_0_10; // (1, 0)
            input_points[0][12][0] = FIFO_32_0_10; // (0, 0)
            input_points[0][11][2] = FIFO_32_0_11; // (2, 0)
            input_points[0][12][1] = FIFO_32_0_11; // (1, 0)
            input_points[0][13][0] = FIFO_32_0_11; // (0, 0)
            input_points[0][12][2] = FIFO_32_0_12; // (2, 0)
            input_points[0][13][1] = FIFO_32_0_12; // (1, 0)
            input_points[0][14][0] = FIFO_32_0_12; // (0, 0)
            input_points[0][13][2] = FIFO_32_0_13; // (2, 0)
            input_points[0][14][1] = FIFO_32_0_13; // (1, 0)
            input_points[0][15][0] = FIFO_32_0_13; // (0, 0)
            input_points[0][0][5] = FIFO_32_0_14; // (2, 1)
            input_points[0][1][4] = FIFO_32_0_14; // (1, 1)
            input_points[0][2][3] = FIFO_32_0_14; // (0, 1)
            input_points[0][1][5] = FIFO_32_0_15; // (2, 1)
            input_points[0][2][4] = FIFO_32_0_15; // (1, 1)
            input_points[0][3][3] = FIFO_32_0_15; // (0, 1)
            input_points[0][2][5] = FIFO_32_0_16; // (2, 1)
            input_points[0][3][4] = FIFO_32_0_16; // (1, 1)
            input_points[0][4][3] = FIFO_32_0_16; // (0, 1)
            input_points[0][3][5] = FIFO_32_0_17; // (2, 1)
            input_points[0][4][4] = FIFO_32_0_17; // (1, 1)
            input_points[0][5][3] = FIFO_32_0_17; // (0, 1)
            input_points[0][4][5] = FIFO_32_0_18; // (2, 1)
            input_points[0][5][4] = FIFO_32_0_18; // (1, 1)
            input_points[0][6][3] = FIFO_32_0_18; // (0, 1)
            input_points[0][5][5] = FIFO_32_0_19; // (2, 1)
            input_points[0][6][4] = FIFO_32_0_19; // (1, 1)
            input_points[0][7][3] = FIFO_32_0_19; // (0, 1)
            input_points[0][6][5] = FIFO_32_0_20; // (2, 1)
            input_points[0][7][4] = FIFO_32_0_20; // (1, 1)
            input_points[0][8][3] = FIFO_32_0_20; // (0, 1)
            input_points[0][7][5] = FIFO_32_0_21; // (2, 1)
            input_points[0][8][4] = FIFO_32_0_21; // (1, 1)
            input_points[0][9][3] = FIFO_32_0_21; // (0, 1)
            input_points[0][8][5] = FIFO_32_0_22; // (2, 1)
            input_points[0][9][4] = FIFO_32_0_22; // (1, 1)
            input_points[0][10][3] = FIFO_32_0_22; // (0, 1)
            input_points[0][9][5] = FIFO_32_0_23; // (2, 1)
            input_points[0][10][4] = FIFO_32_0_23; // (1, 1)
            input_points[0][11][3] = FIFO_32_0_23; // (0, 1)
            input_points[0][10][5] = FIFO_32_0_24; // (2, 1)
            input_points[0][11][4] = FIFO_32_0_24; // (1, 1)
            input_points[0][12][3] = FIFO_32_0_24; // (0, 1)
            input_points[0][11][5] = FIFO_32_0_25; // (2, 1)
            input_points[0][12][4] = FIFO_32_0_25; // (1, 1)
            input_points[0][13][3] = FIFO_32_0_25; // (0, 1)
            input_points[0][12][5] = FIFO_32_0_26; // (2, 1)
            input_points[0][13][4] = FIFO_32_0_26; // (1, 1)
            input_points[0][14][3] = FIFO_32_0_26; // (0, 1)
            input_points[0][13][5] = FIFO_32_0_27; // (2, 1)
            input_points[0][14][4] = FIFO_32_0_27; // (1, 1)
            input_points[0][15][3] = FIFO_32_0_27; // (0, 1)

compute_unrolled:
            for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)
            {
#pragma HLS unroll
                int32_t& i = i_base[unroll_index];
                int32_t& j = j_base[unroll_index];
                int32_t p = p_base+i;
                int32_t q = j;
                int32_t output_index_offset = epoch*UNROLL_FACTOR+unroll_index;

                input_type input_0_0 = input_points[0][unroll_index][0];
                input_type input_1_0 = input_points[0][unroll_index][1];
                input_type input_2_0 = input_points[0][unroll_index][2];
                input_type input_0_1 = input_points[0][unroll_index][3];
                input_type input_1_1 = input_points[0][unroll_index][4];
                input_type input_2_1 = input_points[0][unroll_index][5];
                input_type input_0_2 = input_points[0][unroll_index][6];
                input_type input_1_2 = input_points[0][unroll_index][7];
                input_type input_2_2 = input_points[0][unroll_index][8];

                input_type input_0 = input_points[0][unroll_index][0];
                input_type input_1 = input_points[0][unroll_index][1];
                input_type input_2 = input_points[0][unroll_index][2];
                input_type input_3 = input_points[0][unroll_index][3];
                input_type input_4 = input_points[0][unroll_index][4];
                input_type input_5 = input_points[0][unroll_index][5];
                input_type input_6 = input_points[0][unroll_index][6];
                input_type input_7 = input_points[0][unroll_index][7];
                input_type input_8 = input_points[0][unroll_index][8];
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
                output[0][output_index_offset] = assign_128;

                i += UNROLL_FACTOR;
                if(i>=TILE_SIZE_DIM_0)
                {
                    i -= TILE_SIZE_DIM_0;
                    ++j;
                }
            } // for unroll_index

            FIFO_31[0][18][FIFO_31_ptr] = FIFO_31_0_52;
            FIFO_31[0][52][FIFO_31_ptr] = input_buffer[0][0];

            FIFO_31[0][19][FIFO_31_ptr] = FIFO_31_0_53;
            FIFO_31[0][53][FIFO_31_ptr] = input_buffer[0][1];

            FIFO_31[0][20][FIFO_31_ptr] = FIFO_31_0_54;
            FIFO_31[0][54][FIFO_31_ptr] = input_buffer[0][2];

            FIFO_31[0][21][FIFO_31_ptr] = FIFO_31_0_55;
            FIFO_31[0][55][FIFO_31_ptr] = input_buffer[0][3];

            FIFO_31[0][22][FIFO_31_ptr] = FIFO_31_0_56;
            FIFO_31[0][56][FIFO_31_ptr] = input_buffer[0][4];

            FIFO_31[0][23][FIFO_31_ptr] = FIFO_31_0_57;
            FIFO_31[0][57][FIFO_31_ptr] = input_buffer[0][5];

            FIFO_31[0][24][FIFO_31_ptr] = FIFO_31_0_58;
            FIFO_31[0][58][FIFO_31_ptr] = input_buffer[0][6];

            FIFO_31[0][25][FIFO_31_ptr] = FIFO_31_0_59;
            FIFO_31[0][59][FIFO_31_ptr] = input_buffer[0][7];

            FIFO_31[0][26][FIFO_31_ptr] = FIFO_31_0_60;
            FIFO_31[0][60][FIFO_31_ptr] = input_buffer[0][8];

            FIFO_31[0][27][FIFO_31_ptr] = FIFO_31_0_61;
            FIFO_31[0][61][FIFO_31_ptr] = input_buffer[0][9];

            FIFO_31[0][28][FIFO_31_ptr] = FIFO_31_0_62;
            FIFO_31[0][62][FIFO_31_ptr] = input_buffer[0][10];

            FIFO_31[0][29][FIFO_31_ptr] = FIFO_31_0_63;
            FIFO_31[0][63][FIFO_31_ptr] = input_buffer[0][11];

            FIFO_31[0][30][FIFO_31_ptr] = FIFO_31_0_64;
            FIFO_31[0][64][FIFO_31_ptr] = input_buffer[0][12];

            FIFO_31[0][31][FIFO_31_ptr] = FIFO_31_0_65;
            FIFO_31[0][65][FIFO_31_ptr] = input_buffer[0][13];

            FIFO_31[0][32][FIFO_31_ptr] = FIFO_31_0_66;
            FIFO_31[0][66][FIFO_31_ptr] = input_buffer[0][14];

            FIFO_31[0][33][FIFO_31_ptr] = FIFO_31_0_67;
            FIFO_31[0][67][FIFO_31_ptr] = input_buffer[0][15];

            FIFO_31[0][34][FIFO_31_ptr] = FIFO_31_0_68;
            FIFO_31[0][68][FIFO_31_ptr] = input_buffer[0][16];

            FIFO_31[0][35][FIFO_31_ptr] = FIFO_31_0_69;
            FIFO_31[0][69][FIFO_31_ptr] = input_buffer[0][17];

            FIFO_31[0][36][FIFO_31_ptr] = FIFO_31_0_70;
            FIFO_31[0][70][FIFO_31_ptr] = input_buffer[0][18];

            FIFO_31[0][37][FIFO_31_ptr] = FIFO_31_0_71;
            FIFO_31[0][71][FIFO_31_ptr] = input_buffer[0][19];

            FIFO_31[0][38][FIFO_31_ptr] = FIFO_31_0_72;
            FIFO_31[0][72][FIFO_31_ptr] = input_buffer[0][20];

            FIFO_31[0][39][FIFO_31_ptr] = FIFO_31_0_73;
            FIFO_31[0][73][FIFO_31_ptr] = input_buffer[0][21];

            FIFO_31[0][40][FIFO_31_ptr] = FIFO_31_0_74;
            FIFO_31[0][74][FIFO_31_ptr] = input_buffer[0][22];

            FIFO_31[0][41][FIFO_31_ptr] = FIFO_31_0_75;
            FIFO_31[0][75][FIFO_31_ptr] = input_buffer[0][23];

            FIFO_31[0][42][FIFO_31_ptr] = FIFO_31_0_76;
            FIFO_31[0][76][FIFO_31_ptr] = input_buffer[0][24];

            FIFO_31[0][43][FIFO_31_ptr] = FIFO_31_0_77;
            FIFO_31[0][77][FIFO_31_ptr] = input_buffer[0][25];

            FIFO_31[0][44][FIFO_31_ptr] = FIFO_31_0_78;
            FIFO_31[0][78][FIFO_31_ptr] = input_buffer[0][26];

            FIFO_31[0][45][FIFO_31_ptr] = FIFO_31_0_79;
            FIFO_31[0][79][FIFO_31_ptr] = input_buffer[0][27];

            FIFO_31[0][46][FIFO_31_ptr] = FIFO_31_0_80;
            FIFO_31[0][80][FIFO_31_ptr] = input_buffer[0][28];

            FIFO_31[0][47][FIFO_31_ptr] = FIFO_31_0_81;
            FIFO_31[0][81][FIFO_31_ptr] = input_buffer[0][29];

            FF[0][0] = FIFO_31_0_48;
            FIFO_31[0][48][FIFO_31_ptr] = FIFO_31_0_82;
            FIFO_31[0][82][FIFO_31_ptr] = input_buffer[0][30];

            FF[0][1] = FIFO_31_0_49;
            FIFO_31[0][49][FIFO_31_ptr] = FIFO_31_0_83;
            FIFO_31[0][83][FIFO_31_ptr] = input_buffer[0][31];

            FIFO_32[0][0][FIFO_32_ptr] = FIFO_31_0_84;
            FIFO_31[0][84][FIFO_31_ptr] = input_buffer[0][32];

            FIFO_32[0][1][FIFO_32_ptr] = FIFO_31_0_85;
            FIFO_31[0][85][FIFO_31_ptr] = input_buffer[0][33];

            FIFO_32[0][2][FIFO_32_ptr] = FIFO_31_0_86;
            FIFO_31[0][86][FIFO_31_ptr] = input_buffer[0][34];

            FIFO_32[0][3][FIFO_32_ptr] = FIFO_31_0_87;
            FIFO_31[0][87][FIFO_31_ptr] = input_buffer[0][35];

            FIFO_32[0][4][FIFO_32_ptr] = FIFO_31_0_88;
            FIFO_31[0][88][FIFO_31_ptr] = input_buffer[0][36];

            FIFO_32[0][5][FIFO_32_ptr] = FIFO_31_0_89;
            FIFO_31[0][89][FIFO_31_ptr] = input_buffer[0][37];

            FIFO_32[0][6][FIFO_32_ptr] = FIFO_31_0_90;
            FIFO_31[0][90][FIFO_31_ptr] = input_buffer[0][38];

            FIFO_32[0][7][FIFO_32_ptr] = FIFO_31_0_91;
            FIFO_31[0][91][FIFO_31_ptr] = input_buffer[0][39];

            FIFO_32[0][8][FIFO_32_ptr] = FIFO_31_0_92;
            FIFO_31[0][92][FIFO_31_ptr] = input_buffer[0][40];

            FIFO_32[0][9][FIFO_32_ptr] = FIFO_31_0_93;
            FIFO_31[0][93][FIFO_31_ptr] = input_buffer[0][41];

            FIFO_32[0][10][FIFO_32_ptr] = FIFO_31_0_94;
            FIFO_31[0][94][FIFO_31_ptr] = input_buffer[0][42];

            FIFO_32[0][11][FIFO_32_ptr] = FIFO_31_0_95;
            FIFO_31[0][95][FIFO_31_ptr] = input_buffer[0][43];

            FIFO_32[0][12][FIFO_32_ptr] = FIFO_31_0_96;
            FIFO_31[0][96][FIFO_31_ptr] = input_buffer[0][44];

            FIFO_32[0][13][FIFO_32_ptr] = FIFO_31_0_97;
            FIFO_31[0][97][FIFO_31_ptr] = input_buffer[0][45];

            FIFO_31[0][0][FIFO_31_ptr] = FF[0][2];
            FF[0][2] = FIFO_31_0_98;
            FIFO_31[0][98][FIFO_31_ptr] = input_buffer[0][46];

            FIFO_31[0][1][FIFO_31_ptr] = FF[0][3];
            FF[0][3] = FIFO_31_0_99;
            FIFO_31[0][99][FIFO_31_ptr] = input_buffer[0][47];

            FIFO_31[0][2][FIFO_31_ptr] = FIFO_32_0_14;
            FIFO_32[0][14][FIFO_32_ptr] = input_buffer[0][48];

            FIFO_31[0][3][FIFO_31_ptr] = FIFO_32_0_15;
            FIFO_32[0][15][FIFO_32_ptr] = input_buffer[0][49];

            FIFO_31[0][4][FIFO_31_ptr] = FIFO_32_0_16;
            FIFO_32[0][16][FIFO_32_ptr] = input_buffer[0][50];

            FIFO_31[0][5][FIFO_31_ptr] = FIFO_32_0_17;
            FIFO_32[0][17][FIFO_32_ptr] = input_buffer[0][51];

            FIFO_31[0][6][FIFO_31_ptr] = FIFO_32_0_18;
            FIFO_32[0][18][FIFO_32_ptr] = input_buffer[0][52];

            FIFO_31[0][7][FIFO_31_ptr] = FIFO_32_0_19;
            FIFO_32[0][19][FIFO_32_ptr] = input_buffer[0][53];

            FIFO_31[0][8][FIFO_31_ptr] = FIFO_32_0_20;
            FIFO_32[0][20][FIFO_32_ptr] = input_buffer[0][54];

            FIFO_31[0][9][FIFO_31_ptr] = FIFO_32_0_21;
            FIFO_32[0][21][FIFO_32_ptr] = input_buffer[0][55];

            FIFO_31[0][10][FIFO_31_ptr] = FIFO_32_0_22;
            FIFO_32[0][22][FIFO_32_ptr] = input_buffer[0][56];

            FIFO_31[0][11][FIFO_31_ptr] = FIFO_32_0_23;
            FIFO_32[0][23][FIFO_32_ptr] = input_buffer[0][57];

            FIFO_31[0][12][FIFO_31_ptr] = FIFO_32_0_24;
            FIFO_32[0][24][FIFO_32_ptr] = input_buffer[0][58];

            FIFO_31[0][13][FIFO_31_ptr] = FIFO_32_0_25;
            FIFO_32[0][25][FIFO_32_ptr] = input_buffer[0][59];

            FIFO_31[0][14][FIFO_31_ptr] = FIFO_32_0_26;
            FIFO_32[0][26][FIFO_32_ptr] = input_buffer[0][60];

            FIFO_31[0][15][FIFO_31_ptr] = FIFO_32_0_27;
            FIFO_32[0][27][FIFO_32_ptr] = input_buffer[0][61];

            FIFO_31[0][16][FIFO_31_ptr] = FIFO_31_0_50;
            FIFO_31[0][50][FIFO_31_ptr] = FF[0][4];
            FF[0][4] = input_buffer[0][62];

            FIFO_31[0][17][FIFO_31_ptr] = FIFO_31_0_51;
            FIFO_31[0][51][FIFO_31_ptr] = FF[0][5];
            FF[0][5] = input_buffer[0][63];

            FIFO_31_ptr = FIFO_31_ptr==(31-1) ? 0 : FIFO_31_ptr+1;
            FIFO_32_ptr = FIFO_32_ptr==(32-1) ? 0 : FIFO_32_ptr+1;
        }
    }
}

extern "C"
{

void blur_kernel(
    ap_uint<BURST_WIDTH>* var_output_chan_0,
    ap_uint<BURST_WIDTH>* var_output_chan_1,
    ap_uint<BURST_WIDTH>* var_output_chan_2,
    ap_uint<BURST_WIDTH>* var_output_chan_3,
    ap_uint<BURST_WIDTH>* var_input_chan_0,
    ap_uint<BURST_WIDTH>* var_input_chan_1,
    ap_uint<BURST_WIDTH>* var_input_chan_2,
    ap_uint<BURST_WIDTH>* var_input_chan_3,
    int32_t tile_num_dim_0,
    int32_t input_size_dim_1,
    int64_t tile_burst_num,
    int64_t extra_space_i_coalesed,
    int64_t extra_space_o_coalesed,
    int32_t total_burst_num)
{
#pragma HLS INTERFACE m_axi port=var_output_chan_0 offset=slave depth=65536 bundle=gmem0 latency=120
#pragma HLS INTERFACE m_axi port=var_output_chan_1 offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_output_chan_2 offset=slave depth=65536 bundle=gmem2 latency=120
#pragma HLS INTERFACE m_axi port=var_output_chan_3 offset=slave depth=65536 bundle=gmem3 latency=120
#pragma HLS INTERFACE m_axi port=var_input_chan_0 offset=slave depth=65536 bundle=gmem0 latency=120
#pragma HLS INTERFACE m_axi port=var_input_chan_1 offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_input_chan_2 offset=slave depth=65536 bundle=gmem2 latency=120
#pragma HLS INTERFACE m_axi port=var_input_chan_3 offset=slave depth=65536 bundle=gmem3 latency=120

#pragma HLS INTERFACE s_axilite port=var_output_chan_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_output_chan_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_output_chan_2 bundle=control
#pragma HLS INTERFACE s_axilite port=var_output_chan_3 bundle=control
#pragma HLS INTERFACE s_axilite port=var_input_chan_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_input_chan_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_input_chan_2 bundle=control
#pragma HLS INTERFACE s_axilite port=var_input_chan_3 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim_0 bundle=control
#pragma HLS INTERFACE s_axilite port=input_size_dim_1 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_burst_num bundle=control
#pragma HLS INTERFACE s_axilite port=extra_space_i_coalesed bundle=control
#pragma HLS INTERFACE s_axilite port=extra_space_o_coalesed bundle=control
#pragma HLS INTERFACE s_axilite port=total_burst_num bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    input_type  input_0[CHANNEL_NUM_I][BURST_LENGTH*4];
    input_type  input_1[CHANNEL_NUM_I][BURST_LENGTH*4];
    output_type output_0[CHANNEL_NUM_O][BURST_LENGTH*4];
    output_type output_1[CHANNEL_NUM_O][BURST_LENGTH*4];
    input_type FF[CHANNEL_NUM_I][6];
    input_type FIFO_31[CHANNEL_NUM_I][100][31];
    input_type FIFO_32[CHANNEL_NUM_I][28][32];
#pragma HLS array_partition variable=input_0 complete dim=1
#pragma HLS array_partition variable=input_0 cyclic factor=64 dim=2
#pragma HLS array_partition variable=input_1 complete dim=1
#pragma HLS array_partition variable=input_1 cyclic factor=64 dim=2
#pragma HLS array_partition variable=output_0 complete dim=1
#pragma HLS array_partition variable=output_0 cyclic factor=64 dim=2
#pragma HLS array_partition variable=output_1 complete dim=1
#pragma HLS array_partition variable=output_1 cyclic factor=64 dim=2
#pragma HLS array_partition variable=FF complete dim=0
#pragma HLS array_partition variable=FIFO_31 complete dim=1
#pragma HLS array_partition variable=FIFO_31 complete dim=2
#pragma HLS array_partition variable=FIFO_32 complete dim=1
#pragma HLS array_partition variable=FIFO_32 complete dim=2

    int32_t tile_index_dim_0 = 0;
    bool    load_flag;
    bool compute_flag;
    bool   store_flag;
    int32_t burst_index_load = 0;
    int32_t burst_index_compute = 0;
    int32_t burst_index_store = 0;
    int32_t FIFO_ptrs[2] = {0};
    int32_t input_index_base = 0;

    int32_t i_base[UNROLL_FACTOR];
    int32_t j_base[UNROLL_FACTOR];
    int32_t p_base = 0;

#pragma HLS array_partition variable=FIFO_ptrs complete
#pragma HLS array_partition variable=i_base complete
#pragma HLS array_partition variable=j_base complete

bases_init:
    for(int32_t unroll_index = 0; unroll_index < UNROLL_FACTOR; ++unroll_index)
    {
#pragma HLS unroll
        i_base[unroll_index] = unroll_index - STENCIL_DISTANCE;
        j_base[unroll_index] = 0;
    }

burst:
    for(int32_t burst_index_in_total = 0; burst_index_in_total < total_burst_num+2; ++burst_index_in_total)
    {
        load_flag = burst_index_in_total < total_burst_num;
        compute_flag = burst_index_in_total > 0 && burst_index_in_total < total_burst_num+1;
        store_flag = burst_index_in_total > 1;
        if(burst_index_in_total%2==0)
        {
            load<0, 4>(load_flag, input_0, var_input_chan_0);
            load<1, 4>(load_flag, input_0, var_input_chan_1);
            load<2, 4>(load_flag, input_0, var_input_chan_2);
            load<3, 4>(load_flag, input_0, var_input_chan_3);
            compute(compute_flag, output_1, input_1, FF, FIFO_31, FIFO_32, FIFO_ptrs, i_base, j_base, p_base, input_index_base);
            store<0, 4>(store_flag, var_output_chan_0, output_0);
            store<1, 4>(store_flag, var_output_chan_1, output_0);
            store<2, 4>(store_flag, var_output_chan_2, output_0);
            store<3, 4>(store_flag, var_output_chan_3, output_0);
        }
        else
        {
            load<0, 4>(load_flag, input_1, var_input_chan_0);
            load<1, 4>(load_flag, input_1, var_input_chan_1);
            load<2, 4>(load_flag, input_1, var_input_chan_2);
            load<3, 4>(load_flag, input_1, var_input_chan_3);
            compute(compute_flag, output_0, input_0, FF, FIFO_31, FIFO_32, FIFO_ptrs, i_base, j_base, p_base, input_index_base);
            store<0, 4>(store_flag, var_output_chan_0, output_1);
            store<1, 4>(store_flag, var_output_chan_1, output_1);
            store<2, 4>(store_flag, var_output_chan_2, output_1);
            store<3, 4>(store_flag, var_output_chan_3, output_1);
        }
        if(load_flag)
        {
            var_input_chan_0 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I)*CHANNEL_NUM_I;
            var_input_chan_1 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I)*CHANNEL_NUM_I;
            var_input_chan_2 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I)*CHANNEL_NUM_I;
            var_input_chan_3 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_I)*CHANNEL_NUM_I;
            burst_index_load += 1;
            if(burst_index_load == tile_burst_num)
            {
                burst_index_load = 0;
                var_input_chan_0 -= extra_space_i_coalesed;
                var_input_chan_1 -= extra_space_i_coalesed;
                var_input_chan_2 -= extra_space_i_coalesed;
                var_input_chan_3 -= extra_space_i_coalesed;
            }
        }
        if(compute_flag)
        {
            burst_index_compute += 1;
            input_index_base  += BURST_LENGTH*2/UNROLL_FACTOR;
            if(burst_index_compute == tile_burst_num)
            {
                burst_index_compute = 0;
                input_index_base = 0;
                tile_index_dim_0 += 1;
                p_base += (TILE_SIZE_DIM_0-STENCIL_DIM_0+1);
                if(tile_index_dim_0==tile_num_dim_0)
                {
                    tile_index_dim_0 = 0;
                }
            }
        }
        if(store_flag)
        {
            var_output_chan_0 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O)*CHANNEL_NUM_O;
            var_output_chan_1 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O)*CHANNEL_NUM_O;
            var_output_chan_2 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O)*CHANNEL_NUM_O;
            var_output_chan_3 += BURST_LENGTH/(BURST_WIDTH/PIXEL_WIDTH_O)*CHANNEL_NUM_O;
            burst_index_store += 1;
            if(burst_index_store == tile_burst_num)
            {
                burst_index_store = 0;
                var_output_chan_0 -= extra_space_o_coalesed;
                var_output_chan_1 -= extra_space_o_coalesed;
                var_output_chan_2 -= extra_space_o_coalesed;
                var_output_chan_3 -= extra_space_o_coalesed;
            }
        }
    }
}

}//extern "C"
