#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

#include"jacobi3d.h"

int jacobi3d_test(const char*, const int dims[4]);
int main(int argc, char **argv)
{
    if(argc != 5)
    {
        fprintf(stderr, "Usage: \n    %s <xclbin> <input width> <input height> <input depth>\n", argv[0]);
        return 1;
    }
    int dims[4] = {atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), 0};
    return jacobi3d_test(argv[1], dims);
}
