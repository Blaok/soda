#!/bin/sh

# blur
echo -n "blur "
bin/blur|grep '^Halide run time:'|awk '{print $4}'

# curved
echo -n "curved "
bin/curved img/curved-input.png 3700 2.0 50 10 img/curved-output.png 2>&1 |grep '^Halide run time:'|awk '{print $4}'

