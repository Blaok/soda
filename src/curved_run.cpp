#include "curved.h"
#include "halide_image.h"
#include "halide_image_io.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace Halide::Tools;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage:   %s xclbin [raw.png color_temp gamma contrast timing_iterations output.png]\n"
               "Default: %s xclbin ~/git/Halide/apps/images/bayer_raw.png 3700 2.0 50 5 out.png\n", argv[0], argv[0]);
        return 0;
    }

    fprintf(stderr, "input: %s\n", argc < 2 ? "~/git/Halide/apps/images/bayer_raw.png" : argv[2]);
    Image<uint16_t> input = load_imageargc < 2 ? "~/git/Halide/apps/images/bayer_raw.png" : (argv[2]);
    fprintf(stderr, "       %d %d\n", input.width(), input.height());
    Image<uint8_t> output(((input.width() - 32)/32)*32, ((input.height() - 24)/32)*32, 3);

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

    float color_temp = atof( argc < 2 ? "3700" : argv[3]);
    float gamma = atof(argc < 2 ? "2.0" : argv[4]);
    float contrast = atof( argc < 2 ? "50" : argv[5]);
    int timing_iterations = atoi( argc < 2 ? 5 : argv[6]);
    int blackLevel = 25;
    int whiteLevel = 1023;

    double best = 0;

    for(int i = 0; i<timing_iterations; ++i)
    {
        curved(color_temp, gamma, contrast, blackLevel, whiteLevel,
               input, matrix_3200, matrix_7000,
               output, argv[1]);
    }
    fprintf(stderr, "Halide:\t%gus\n", best * 1e6);
    fprintf(stderr, "output: %s\n", argc < 2 ? "out.png" : argv[7]);
    save_image(output, argc < 2 ? "out.png" : argv[7]);
    fprintf(stderr, "        %d %d\n", output.width(), output.height());

    return 0;
}
