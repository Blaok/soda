#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include"erode.h"

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
    const int32_t output_width  = input_width -2;
    const int32_t output_height = input_height-2;

    uint8_t*  input_img = new uint8_t[ input_width* input_height];
    uint8_t* output_img = new uint8_t[output_width*output_height];

    input.extent[0] = input_width; // Width.
    input.extent[1] = input_height; // Height.
    input.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    input.stride[1] = input_width; // Spacing in memory between adjacent values of y;
    input.elem_size = sizeof(uint8_t); // Bytes per element.
    input.host = (uint8_t *)input_img;

    output.extent[0] = output_width; // Width.
    output.extent[1] = output_height; // Height.
    output.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    output.stride[1] = output_width; // Spacing in memory between adjacent values of y;
    output.elem_size = sizeof(uint8_t); // Bytes per element.
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
    erode(&input, &output, argv[1]);
    clock_gettime(CLOCK_REALTIME, &t2);
    elapsed_time += (double(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9)*1e6;
    printf("Kernel runtime: %lf us\n", elapsed_time);

    return 0;

    // Print the output_img.
    int error_count = 0;
    for (int y = 0; y < output_height; y++)
    {
        for (int x = 0; x < output_width; x++)
        {
            if(x<10 && y<20)
            {
                printf("%3d ", output_img[y * output_width + x]);
            }
            uint8_t ground_truth = (
                uint8_t(input_img[ y   *input_width+x]+input_img[ y   *input_width+x+1]+input_img[ y   *input_width+x+2])/3+
                uint8_t(input_img[(y+1)*input_width+x]+input_img[(y+1)*input_width+x+1]+input_img[(y+1)*input_width+x+2])/3+
                uint8_t(input_img[(y+2)*input_width+x]+input_img[(y+2)*input_width+x+1]+input_img[(y+2)*input_width+x+2])/3)/3;
            if(output_img[y * output_width + x] != ground_truth)
            {
                printf("%d != %d @(%d, %d)\n", output_img[y * output_width + x], ground_truth, x, y);
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
