#ifndef BLUR_PARAMS_H_
#define BLUR_PARAMS_H_

// determined arbitrarily
#ifndef TILE_SIZE_DIM_0
#define TILE_SIZE_DIM_0 2000    // in pixels
#endif//TILE_SIZE_DIM_0
#ifndef BURST_LENGTH
#define BURST_LENGTH 100000     // in pixels
#endif//BURST_LENGTH
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
#define STENCIL_DIM_0 5         // in pixels
#define STENCIL_DIM_1 5         // in pixels
#define STENCIL_DISTANCE ((TILE_SIZE_DIM_0)*4+3)

// determined by params above
#define BURST_EPOCH_I (BURST_LENGTH*PIXEL_WIDTH_I/BUSRT_WIDTH)
#define BURST_EPOCH_O (BURST_LENGTH*PIXEL_WIDTH_O/BUSRT_WIDTH)

// BURST_LENGTH % UNROLL_FACTOR must be 0
// BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_I) must be 0
// BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_O) must be 0
// PIXEL_WIDTH_I and PIXEL_WIDTH_O can only be 8, 16, 32, 64
// BURST_WIDTH must be power of 2
#if BURST_LENGTH % UNROLL_FACTOR
#error BURST_LENGTH % UNROLL_FACTOR must be 0
#endif//BURST_LENGTH % UNROLL_FACTOR
#if BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_I)
#error BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_I) must be 0
#endif//BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_I)
#if BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_O)
#error BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_O) must be 0
#endif//BURST_LENGTH % (BURST_WIDTH/PIXEL_WIDTH_O)

#endif//BLUR_PARAMS_H_
