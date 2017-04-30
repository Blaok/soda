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
#define STENCIL_DIM0 (5)
#define STENCIL_DIM1 (5)
#define STENCIL_DISTANCE ((TILE_SIZE_DIM0)*4+4)
#define BURST_WIDTH (512)
#define PIXEL_WIDTH (sizeof(uint16_t)*8)

#define TILE_INDEX_DIM0(tile_index) ((tile_index)%(tile_num_dim0))
#define TILE_INDEX_DIM1(tile_index) ((tile_index)/(tile_num_dim0))
#define TILE_SIZE_BURST ((TILE_SIZE_DIM0)*(TILE_SIZE_DIM1)/((BURST_WIDTH)/(PIXEL_WIDTH)))
#define P(tile_index_dim0,i) ((tile_index_dim0)*((TILE_SIZE_DIM0)-(STENCIL_DIM0)+1)+(i))
#define Q(tile_index_dim1,j) ((tile_index_dim1)*((TILE_SIZE_DIM1)-(STENCIL_DIM1)+1)+(j))

void load(bool load_flag, uint16_t to[TILE_SIZE_DIM0*TILE_SIZE_DIM1], ap_uint<BURST_WIDTH>* from, int32_t tile_index)
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

void store(bool store_flag, ap_uint<BURST_WIDTH>* to, uint16_t from[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index)
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

void compute(bool compute_flag, uint16_t output[TILE_SIZE_DIM0*TILE_SIZE_DIM1], uint16_t input[TILE_SIZE_DIM0*TILE_SIZE_DIM1], int32_t tile_index, int32_t tile_num_dim0, int32_t var_gaussian_y_extent_0, int32_t var_gaussian_y_extent_1, int32_t var_gaussian_y_min_0, int32_t var_gaussian_y_min_1)
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
                uint16_t input_0_0 = input[TILE_SIZE_DIM0*(j+0)+i+0];
                uint16_t input_0_1 = input[TILE_SIZE_DIM0*(j+0)+i+1];
                uint16_t input_0_2 = input[TILE_SIZE_DIM0*(j+0)+i+2];
                uint16_t input_0_3 = input[TILE_SIZE_DIM0*(j+0)+i+3];
                uint16_t input_0_4 = input[TILE_SIZE_DIM0*(j+0)+i+4];
                uint16_t input_1_0 = input[TILE_SIZE_DIM0*(j+1)+i+0];
                uint16_t input_1_1 = input[TILE_SIZE_DIM0*(j+1)+i+1];
                uint16_t input_1_2 = input[TILE_SIZE_DIM0*(j+1)+i+2];
                uint16_t input_1_3 = input[TILE_SIZE_DIM0*(j+1)+i+3];
                uint16_t input_1_4 = input[TILE_SIZE_DIM0*(j+1)+i+4];
                uint16_t input_2_0 = input[TILE_SIZE_DIM0*(j+2)+i+0];
                uint16_t input_2_1 = input[TILE_SIZE_DIM0*(j+2)+i+1];
                uint16_t input_2_2 = input[TILE_SIZE_DIM0*(j+2)+i+2];
                uint16_t input_2_3 = input[TILE_SIZE_DIM0*(j+2)+i+3];
                uint16_t input_2_4 = input[TILE_SIZE_DIM0*(j+2)+i+4];
                uint16_t input_3_0 = input[TILE_SIZE_DIM0*(j+3)+i+0];
                uint16_t input_3_1 = input[TILE_SIZE_DIM0*(j+3)+i+1];
                uint16_t input_3_2 = input[TILE_SIZE_DIM0*(j+3)+i+2];
                uint16_t input_3_3 = input[TILE_SIZE_DIM0*(j+3)+i+3];
                uint16_t input_3_4 = input[TILE_SIZE_DIM0*(j+3)+i+4];
                uint16_t input_4_0 = input[TILE_SIZE_DIM0*(j+4)+i+0];
                uint16_t input_4_1 = input[TILE_SIZE_DIM0*(j+4)+i+1];
                uint16_t input_4_2 = input[TILE_SIZE_DIM0*(j+4)+i+2];
                uint16_t input_4_3 = input[TILE_SIZE_DIM0*(j+4)+i+3];
                uint16_t input_4_4 = input[TILE_SIZE_DIM0*(j+4)+i+4];
                if(p >= var_gaussian_y_min_0 &&
                   q >= var_gaussian_y_min_1 &&
                   p < var_gaussian_y_min_0 + var_gaussian_y_extent_0 &&
                   q < var_gaussian_y_min_1 + var_gaussian_y_extent_1)
                {
                    uint16_t assign_105 = input_0_0;//((uint16_t *)var_input)[assign_104];
                    uint16_t assign_107 = input_0_1;//((uint16_t *)var_input)[assign_106];
                    uint16_t assign_108 = (uint16_t)(4);
                    uint16_t assign_109 = assign_107 * assign_108;
                    uint16_t assign_110 = assign_105 + assign_109;
                    uint16_t assign_112 = input_0_2;//((uint16_t *)var_input)[assign_111];
                    uint16_t assign_113 = (uint16_t)(6);
                    uint16_t assign_114 = assign_112 * assign_113;
                    uint16_t assign_115 = assign_110 + assign_114;
                    uint16_t assign_117 = input_0_3;//((uint16_t *)var_input)[assign_116];
                    uint16_t assign_118 = assign_117 * assign_108;
                    uint16_t assign_119 = assign_115 + assign_118;
                    uint16_t assign_121 = input_0_4;//((uint16_t *)var_input)[assign_120];
                    uint16_t assign_122 = assign_119 + assign_121;
                    uint16_t assign_123 = assign_122 >> 4;
                    uint16_t assign_125 = input_1_0;//((uint16_t *)var_input)[assign_124];
                    uint16_t assign_127 = input_1_1;//((uint16_t *)var_input)[assign_126];
                    uint16_t assign_128 = assign_127 * assign_108;
                    uint16_t assign_129 = assign_125 + assign_128;
                    uint16_t assign_131 = input_1_2;//((uint16_t *)var_input)[assign_130];
                    uint16_t assign_132 = assign_131 * assign_113;
                    uint16_t assign_133 = assign_129 + assign_132;
                    uint16_t assign_135 = input_1_3;//((uint16_t *)var_input)[assign_134];
                    uint16_t assign_136 = assign_135 * assign_108;
                    uint16_t assign_137 = assign_133 + assign_136;
                    uint16_t assign_139 = input_1_4;//((uint16_t *)var_input)[assign_138];
                    uint16_t assign_140 = assign_137 + assign_139;
                    uint16_t assign_141 = assign_140 >> 4;
                    uint16_t assign_142 = assign_141 * assign_108;
                    uint16_t assign_143 = assign_123 + assign_142;
                    uint16_t assign_145 = input_2_0;//((uint16_t *)var_input)[assign_144];
                    uint16_t assign_147 = input_2_1;//((uint16_t *)var_input)[assign_146];
                    uint16_t assign_148 = assign_147 * assign_108;
                    uint16_t assign_149 = assign_145 + assign_148;
                    uint16_t assign_151 = input_2_2;//((uint16_t *)var_input)[assign_150];
                    uint16_t assign_152 = assign_151 * assign_113;
                    uint16_t assign_153 = assign_149 + assign_152;
                    uint16_t assign_155 = input_2_3;//((uint16_t *)var_input)[assign_154];
                    uint16_t assign_156 = assign_155 * assign_108;
                    uint16_t assign_157 = assign_153 + assign_156;
                    uint16_t assign_159 = input_2_4;//((uint16_t *)var_input)[assign_158];
                    uint16_t assign_160 = assign_157 + assign_159;
                    uint16_t assign_161 = assign_160 >> 4;
                    uint16_t assign_162 = assign_161 * assign_113;
                    uint16_t assign_163 = assign_143 + assign_162;
                    uint16_t assign_165 = input_3_0;//((uint16_t *)var_input)[assign_164];
                    uint16_t assign_167 = input_3_1;//((uint16_t *)var_input)[assign_166];
                    uint16_t assign_168 = assign_167 * assign_108;
                    uint16_t assign_169 = assign_165 + assign_168;
                    uint16_t assign_171 = input_3_2;//((uint16_t *)var_input)[assign_170];
                    uint16_t assign_172 = assign_171 * assign_113;
                    uint16_t assign_173 = assign_169 + assign_172;
                    uint16_t assign_175 = input_3_3;//((uint16_t *)var_input)[assign_174];
                    uint16_t assign_176 = assign_175 * assign_108;
                    uint16_t assign_177 = assign_173 + assign_176;
                    uint16_t assign_179 = input_3_4;//((uint16_t *)var_input)[assign_178];
                    uint16_t assign_180 = assign_177 + assign_179;
                    uint16_t assign_181 = assign_180 >> 4;
                    uint16_t assign_182 = assign_181 * assign_108;
                    uint16_t assign_183 = assign_163 + assign_182;
                    uint16_t assign_185 = input_4_0;//((uint16_t *)var_input)[assign_184];
                    uint16_t assign_187 = input_4_1;//((uint16_t *)var_input)[assign_186];
                    uint16_t assign_188 = assign_187 * assign_108;
                    uint16_t assign_189 = assign_185 + assign_188;
                    uint16_t assign_191 = input_4_2;//((uint16_t *)var_input)[assign_190];
                    uint16_t assign_192 = assign_191 * assign_113;
                    uint16_t assign_193 = assign_189 + assign_192;
                    uint16_t assign_195 = input_4_3;//((uint16_t *)var_input)[assign_194];
                    uint16_t assign_196 = assign_195 * assign_108;
                    uint16_t assign_197 = assign_193 + assign_196;
                    uint16_t assign_199 = input_4_4;//((uint16_t *)var_input)[assign_198];
                    uint16_t assign_200 = assign_197 + assign_199;
                    uint16_t assign_201 = assign_200 >> 4;
                    uint16_t assign_202 = assign_183 + assign_201;
                    uint16_t assign_203 = assign_202 >> 4;
                    output[TILE_SIZE_DIM0*j+i] = assign_203;
                } // if input_index >= STENCIL_DISTANCE
            } // for output_index
        } // for input_index
        // consume gaussian_y
    }
}

extern "C"
{

void gaussian_kernel(ap_uint<BURST_WIDTH>* var_gaussian_y, ap_uint<BURST_WIDTH>* var_p0, int32_t tile_num_dim0, int32_t tile_num_dim1, int32_t var_gaussian_y_extent_0, int32_t var_gaussian_y_extent_1, int32_t var_gaussian_y_min_0, int32_t var_gaussian_y_min_1)
{
#pragma HLS INTERFACE m_axi port=var_gaussian_y offset=slave depth=65536 bundle=gmem1 latency=120
#pragma HLS INTERFACE m_axi port=var_p0 offset=slave depth=65536 bundle=gmem2 latency=120

#pragma HLS INTERFACE s_axilite port=var_gaussian_y bundle=control
#pragma HLS INTERFACE s_axilite port=var_p0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=tile_num_dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_gaussian_y_extent_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_gaussian_y_extent_1 bundle=control
#pragma HLS INTERFACE s_axilite port=var_gaussian_y_min_0 bundle=control
#pragma HLS INTERFACE s_axilite port=var_gaussian_y_min_1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    uint16_t  input_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t  input_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t output_0[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
    uint16_t output_1[TILE_SIZE_DIM0*TILE_SIZE_DIM1];
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
            load(load_flag, input_0, var_p0, tile_index);
            compute(compute_flag, output_1, input_1, tile_index-1, tile_num_dim0, var_gaussian_y_extent_0, var_gaussian_y_extent_1, var_gaussian_y_min_0, var_gaussian_y_min_1);
            store(store_flag, var_gaussian_y, output_0, tile_index-2);
        }
        else
        {
            load(load_flag, input_1, var_p0, tile_index);
            compute(compute_flag, output_0, input_0, tile_index-1, tile_num_dim0, var_gaussian_y_extent_0, var_gaussian_y_extent_1, var_gaussian_y_min_0, var_gaussian_y_min_1);
            store(store_flag, var_gaussian_y, output_1, tile_index-2);
        }
    }
}

}// extern "C"
