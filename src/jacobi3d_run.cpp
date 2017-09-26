#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include"jacobi3d.h"

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        fprintf(stderr, "Usage: \n    %s <xclbin> <input width> <input height> <input length>\n", argv[0]);
        return 1;
    }
    buffer_t input, output;
    memset(&input, 0, sizeof(buffer_t));
    memset(&output, 0, sizeof(buffer_t));

    const int32_t  input_width  = strtoul(argv[2], nullptr, 10);
    const int32_t  input_height = strtoul(argv[3], nullptr, 10);
    const int32_t  input_length = strtoul(argv[4], nullptr, 10);
    const int32_t output_width  = input_width -12;
    const int32_t output_height = input_height-12;
    const int32_t output_length = input_length-12;

    float*  input_img = new float[ input_width* input_height* input_length];
    float* output_img = new float[output_width*output_height*output_length];

    input.extent[0] = input_width; // Width.
    input.extent[1] = input_height; // Height.
    input.extent[2] = input_length;
    input.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    input.stride[1] = input_width; // Spacing in memory between adjacent values of y;
    input.stride[2] = input_width*input_height;
    input.elem_size = sizeof(float); // Bytes per element.
    input.host = (uint8_t *)input_img;

    output.extent[0] = output_width; // Width.
    output.extent[1] = output_height; // Height.
    output.extent[2] = output_length;
    output.stride[0] = 1;  // Spacing in memory between adjacent values of x.
    output.stride[1] = output_width; // Spacing in memory between adjacent values of y;
    output.stride[2] = output_width*output_height;
    output.elem_size = sizeof(float); // Bytes per element.
    output.host = (uint8_t *)output_img;

    for (int z = 0; z < input_length; z++) {
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                input_img[z * input.stride[1] + y * input_width + x] = x + y + z;
            }
        }
    }

    // Run the pipeline
    timespec t1,t2;
    double elapsed_time = 0.;
    clock_gettime(CLOCK_REALTIME, &t1);
    jacobi3d(&input, &output, argv[1]);
    clock_gettime(CLOCK_REALTIME, &t2);
    elapsed_time += (double(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1e9)*1e6;
    printf("Kernel runtime: %lf us\n", elapsed_time);

    /*

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
            float ground_truth = (
                float(input_img[ y   *input_width+x]+input_img[ y   *input_width+x+1]+input_img[ y   *input_width+x+2])/3+
                float(input_img[(y+1)*input_width+x]+input_img[(y+1)*input_width+x+1]+input_img[(y+1)*input_width+x+2])/3+
                float(input_img[(y+2)*input_width+x]+input_img[(y+2)*input_width+x+1]+input_img[(y+2)*input_width+x+2])/3)/3;
            if(output_img[y * output_width + x] != ground_truth)
            {
                printf("%d != %d @(%d, %d)\n", output_img[y * output_width + x], ground_truth, x, y);
                ++error_count;
                if(error_count > 30)
                {
                    return -1;
                }
            }
        }
        if(y<20)
        {
            printf("\n");
        }
    }
    printf("Kernel runtime: %lf us\n", elapsed_time);
*/

    delete[] input_img;
    delete[] output_img;

    return 0;
}
