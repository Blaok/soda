#include "camera.h"
#include "halide_image.h"
#include "halide_image_io.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace Halide::Tools;

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        fprintf(stderr, "Usage: \n    %s <xclbin> <input width> <input height>\n", argv[0]);
        return 1;
    }
    buffer_t input, output;
    memset(&input, 0, sizeof(buffer_t));
    memset(&output, 0, sizeof(buffer_t));

    const int32_t  input_width  = strtoul(argv[2], nullptr, 10);
    const int32_t  input_height = strtoul(argv[3], nullptr, 10);
    const int32_t output_width  = ((input_width - 32)/32)*32;
    const int32_t output_height = ((input_height - 24)/32)*32;

    uint16_t* input_img = new uint16_t[ input_width* input_height];
    uint8_t* output_img = new  uint8_t[output_width*output_height*3];

    input.extent[0] = input_width; // Width.
    input.extent[1] = input_height; // Height.
    input.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    input.stride[1] = input_width; // Spacing in memory between adjacent values of y;
    input.elem_size = sizeof(uint16_t); // Bytes per element.
    input.host = (uint8_t *)input_img;

    output.extent[0] = output_width; // Width.
    output.extent[1] = output_height; // Height.
    output.extent[2] = 3;
    output.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    output.stride[1] = output_width; // Spacing in memory between adjacent values of y;
    output.stride[2] = output_width*output_height; // Spacing in memory between adjacent values of y;
    output.elem_size = sizeof(uint8_t); // Bytes per element.
    output.host = (uint8_t *)output_img;

    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            input_img[y * input_width + x] = x + y;
        }
    }

    /*
    if (argc < 2)
    {
        printf("Usage:   %s xclbin [raw.png color_temp gamma contrast timing_iterations output.png]\n"
               "Default: %s xclbin ~/git/Halide/apps/images/bayer_raw.png 3700 2.0 50 1 out.png\n", argv[0], argv[0]);
        return 0;
    }

    fprintf(stderr, "input: %s\n", argc < 3 ? "/curr/blaok/git/Halide/apps/images/bayer_raw.png" : argv[2]);
    Image<uint16_t> input = load_image(argc < 3 ? "/curr/blaok/git/Halide/apps/images/bayer_raw.png" : argv[2]);
    fprintf(stderr, "       %d %d\n", input.width(), input.height());
    Image<uint8_t> output(((input.width() - 32)/32)*32, ((input.height() - 24)/32)*32, 3);
    */

    // These color matrices are for the sensor in the Nokia N900 and are
    // taken from the FCam source.
    float _matrix_3200[][4] = {{ 1.6697f, -0.2693f, -0.4004f, -42.4346f},
                               {-0.3576f,  1.0615f,  1.5949f, -37.1158f},
                               {-0.2175f, -1.8751f,  6.9640f, -26.6970f}};

    float _matrix_7000[][4] = {{ 2.2997f, -0.4478f,  0.1706f, -39.0923f},
                               {-0.3826f,  1.5906f, -0.2080f, -25.4311f},
                               {-0.0888f, -0.7344f,  2.2832f, -20.0826f}};
    Image<float> matrix_3200(4, 3), matrix_7000(4, 3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++) {
            matrix_3200(j, i) = _matrix_3200[i][j];
            matrix_7000(j, i) = _matrix_7000[i][j];
        }
    }

    float color_temp = atof( argc < 4 ? "3700" : argv[3]);
    float gamma = atof(argc < 5 ? "2.0" : argv[4]);
    float contrast = atof( argc < 6 ? "50" : argv[5]);
    int timing_iterations = atoi( argc < 7 ? "1" : argv[6]);
    int blackLevel = 25;
    int whiteLevel = 1023;

    double best = 0;

    fprintf(stderr, "Launch\n");
    for(int i = 0; i<timing_iterations; ++i)
    {
        camera(color_temp, gamma, contrast, blackLevel, whiteLevel,
               &input, matrix_3200, matrix_7000,
               &output, argv[1]);
    }
    fprintf(stderr, "Halide:\t%gus\n", best * 1e6);
    fprintf(stderr, "output: %s\n", argc < 8 ? "out.png" : argv[7]);
    //save_image(output, argc < 8 ? "out.png" : argv[7]);
    //fprintf(stderr, "        %d %d\n", output.width(), output.height());

    return 0;
}
