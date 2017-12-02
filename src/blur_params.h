#ifndef BLUR_PARAMS_H_
#define BLUR_PARAMS_H_

// determined arbitrarily
#ifndef TILE_SIZE_DIM_0
#define TILE_SIZE_DIM_0 2000    // in pixels
#endif//TILE_SIZE_DIM_0
#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 16
#endif//UNROLL_FACTOR

// determined by platform
#define BURST_WIDTH 512         // in bits

// determined by application
#define PIXEL_WIDTH_I 16        // in bits
#define PIXEL_WIDTH_O 16        // in bits
#define CHANNEL_NUM_I 1
#define CHANNEL_NUM_O 1
#define STENCIL_DIM_0 3         // in pixels
#define STENCIL_DIM_1 3         // in pixels
#define STENCIL_DISTANCE ((TILE_SIZE_DIM_0)*2+2)

// PIXEL_WIDTH_I and PIXEL_WIDTH_O can only be 8, 16, 32, 64
// BURST_WIDTH must be power of 2

#endif//BLUR_PARAMS_H_
