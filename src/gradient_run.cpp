#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include"gradient.h"

int main(int argc, char **argv)
{
    if(argc != 4)
    {fprintf(stderr, "Usage: \n    %s <xclbin> <input width> <input height>\n", argv[0]);
        return 1;
    }
    buffer_t input, output;
    memset(&input, 0, sizeof(buffer_t));
    memset(&output, 0, sizeof(buffer_t));

    const uint32_t  input_width  = strtoul(argv[2], nullptr, 10);
    const uint32_t  input_height = strtoul(argv[3], nullptr, 10);
    const uint32_t output_width  = input_width -2;
    const uint32_t output_height = input_height-2;

    float*  input_img = new float[ input_width* input_height];
    float* output_img = new float[output_width*output_height];

    input.extent[0] = input_width; // Width.
    input.extent[1] = input_height; // Height.
    input.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    input.stride[1] = input_width; // Spacing in memory between adjacent values of y;
    input.elem_size = sizeof(float); // Bytes per element.
    input.host = (uint8_t *)input_img;

    output.extent[0] = output_width; // Width.
    output.extent[1] = output_height; // Height.
    output.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    output.stride[1] = output_width; // Spacing in memory between adjacent values of y;
    output.elem_size = sizeof(float); // Bytes per element.
    output.host = (uint8_t *)output_img;

    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            input_img[y * input_width + x] = x + y;
        }
    }

    // Run the pipeline
    timespec t1,t2;
    double elapsed_time = 0.;
    clock_gettime(CLOCK_REALTIME, &t1);
    gradient(&input, &output, argv[1]);
    clock_gettime(CLOCK_REALTIME, &t2);
    elapsed_time += (double(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9)*1e6;
    printf("Kernel runtime: %lf us\n", elapsed_time);

    // Print the output_img.
    int error_count = 0;
    for (int y = 0; y < output_height; y++)
    {
        for (int x = 0; x < output_width; x++)
        {
            if(x<10 && y<20)
            {
                printf("%3f ", output_img[y * output_width + x]);
            }
            float assign_95 = input_img[(y+1)*input_width + (x+1)];
            float assign_100 = input_img[(y+0)*input_width + (x+1)];
            float assign_101 = assign_95 - assign_100;
            float assign_102 = input_img[(y+1)*input_width + (x+0)];
            float assign_103 = assign_95 - assign_102;
            float assign_109 = input_img[(y+2)*input_width + (x+1)];
            float assign_110 = assign_95 - assign_109;
            float assign_112 = input_img[(y+1)*input_width + (x+2)];
            float assign_113 = assign_95 - assign_112;
            float assign_114 = assign_101 * assign_101;
            float assign_115 = assign_103 * assign_103;
            float assign_116 = assign_114 + assign_115;
            float assign_117 = assign_110 * assign_110;
            float assign_118 = assign_116 + assign_117;
            float assign_119 = assign_113 * assign_113;
            float assign_120 = assign_118 + assign_119;
            float assign_121 = sqrt(assign_120);
            float assign_122 = 1.f / assign_121;
            float ground_truth = assign_122;
            if(output_img[y * output_width + x] != ground_truth)
            {
                printf("%f != %f @(%d, %d)\n", output_img[y * output_width + x], ground_truth, x, y);
                ++error_count;
/*                if(error_count > 30)
                {
                    return -1;
                }*/
            }
        }
        if(y<20)
        {
            printf("\n");
        }
    }
    printf("Kernel runtime: %lf us\n", elapsed_time);

    delete[] input_img;
    delete[] output_img;

    return 0;
}
