# SODA: Stencil with Optimized Dataflow Architecture

## Publication

+ Yuze Chi, Jason Cong. Exploiting Computation Reuse for Stencil Accelerators. In DAC, 2020. [[PDF]](https://about.blaok.me/pub/dac20-soda-cr.pdf) [[Code]](https://github.com/Blaok/soda-cr)
+ Yuze Chi, Jason Cong, Peng Wei, Peipei Zhou. [SODA: Stencil with Optimized Dataflow Architecture](https://doi.org/10.1145/3240765.3240850). In ICCAD, 2018. (Best Paper Candidate) [[PDF]](https://about.blaok.me/pub/iccad18.pdf) [[Slides]](https://about.blaok.me/pub/iccad18.slides.pdf) [[Code]](https://github.com/Blaok/soda)

## SODA DSL Example

```soda
# comments start with hashtag(#)

kernel: blur      # the kernel name, will be used as the kernel name in HLS
burst width: 512  # I/O width in bits, for Xilinx platform 512 works the best
unroll factor: 16 # how many pixels are generated per cycle

# specify the dram bank, type, name, and dimension of the input tile
# the last dimension is not needed and a placeholder '*' must be given
# dram bank is optional
# multiple inputs can be specified but 1 and only 1 must specify the dimensions
input dram 0 uint16: input(2000, *)

# specify an intermediate stage of computation, may appear 0 or more times
local uint16: tmp(0, 0) = (input(-1, 0) + input(0, 0) + input(1, 0)) / 3

# specify the output
# dram bank is optional
output dram 1 uint16: output(0, 0) = (tmp(0, -1) + tmp(0, 0) + tmp(0, 1)) / 3

# how many times the whole computation is repeated (only works if input matches output)
iterate: 2
```

## Getting Started

### Prerequisites

+ Python 3.6+ and corresponding `pip`

<details><summary>How to install Python 3.6+ on Ubuntu 16.04+ and CentOS 7?</summary>

#### Ubuntu 16.04

```bash
sudo apt install software-properties-common python3-pip
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
```

#### Ubuntu 18.04+

```bash
sudo apt install python3 python3-pip
```

#### CentOS 7

```bash
sudo yum install python3 python3-pip
```

</details>

### Install SODA

#### Install from PyPI

```bash
python3 -m pip install --user --upgrade sodac
```

+ Replace `python3` with a more specific Python version higher than or equal to
    `python3.6`, if necessary.
+ Make sure `~/.local/bin` is in your `PATH`,
    or replace `sodac` with `python3 -m soda.sodac` below.

### Generate Vivado HLS kernel code

```bash
sodac tests/src/blur.soda --xocl-kernel blur_kernel.cpp
```

### Generate Intel OpenCL kernel code

```bash
sodac tests/src/blur.soda --iocl-kernel blur_kernel.cl
```

### Generate Xilinx Object file with AXI MMAP Master Interface

Requries `vivado_hls`.

```bash
sodac tests/src/blur.soda --xocl-hw-xo blur_kernel.hw.xo
```

### Generate Xilinx Object file with AXI Stream Interface

Requries `vivado_hls`.

```bash
sodac tests/src/blur.soda --xocl-hw-xo blur_kernel.hw.xo --xocl-interface axis
```

### Apply Computation Reuse

```bash
sodac tests/src/blur.soda --computation-reuse --xocl-kernel blur_kernel.cpp
```

## Code Snippets

### Configuration

+ 5-point 2D Jacobi: `t0(0, 0) = (t1(0, 1) + t1(1, 0) + t1(0, 0) + t1(0, -1) + t1(-1, 0)) * 0.2f`
+ tile size is `(2000, *)`

Each function in the below code snippets is synthesized into an RTL module.
Their arguments are all `hls::stream` FIFOs; Without unrolling, a simple line-buffer pipeline is generated, producing 1 pixel per cycle.
With unrolling, a SODA microarchitecture pipeline is generated, procuding 2 pixeles per cycle.

### Without Unrolling

```c++
#pragma HLS dataflow
Module1Func(
  /*output*/ &from_t1_offset_0_to_t1_offset_1999,
  /*output*/ &from_t1_offset_0_to_t0_pe_0,
  /* input*/ &from_super_source_to_t1_offset_0);
Module2Func(
  /*output*/ &from_t1_offset_1999_to_t1_offset_2000,
  /*output*/ &from_t1_offset_1999_to_t0_pe_0,
  /* input*/ &from_t1_offset_0_to_t1_offset_1999);
Module3Func(
  /*output*/ &from_t1_offset_2000_to_t1_offset_2001,
  /*output*/ &from_t1_offset_2000_to_t0_pe_0,
  /* input*/ &from_t1_offset_1999_to_t1_offset_2000);
Module3Func(
  /*output*/ &from_t1_offset_2001_to_t1_offset_4000,
  /*output*/ &from_t1_offset_2001_to_t0_pe_0,
  /* input*/ &from_t1_offset_2000_to_t1_offset_2001);
Module4Func(
  /*output*/ &from_t1_offset_4000_to_t0_pe_0,
  /* input*/ &from_t1_offset_2001_to_t1_offset_4000);
Module5Func(
  /*output*/ &from_t0_pe_0_to_super_sink,
  /* input*/ &from_t1_offset_0_to_t0_pe_0,
  /* input*/ &from_t1_offset_1999_to_t0_pe_0,
  /* input*/ &from_t1_offset_2000_to_t0_pe_0,
  /* input*/ &from_t1_offset_4000_to_t0_pe_0,
  /* input*/ &from_t1_offset_2001_to_t0_pe_0);
```

In the above code snippet, `Module1Func` to `Module4Func` are forwarding modules; they constitute the line buffer.
The line buffer size is approximately two lines of pixels, i.e. 4000 pixels.
`Module5Func` is a computing module; it implements the computation kernel.
The whole design is fully pipelined; however, with only 1 computing module, it can only produce 1 pixel per cycle.

### Unroll 2 Times

```c++
#pragma HLS dataflow
Module1Func(
  /*output*/ &from_t1_offset_1_to_t1_offset_1999,
  /*output*/ &from_t1_offset_1_to_t0_pe_0,
  /* input*/ &from_super_source_to_t1_offset_1);
Module1Func(
  /*output*/ &from_t1_offset_0_to_t1_offset_2000,
  /*output*/ &from_t1_offset_0_to_t0_pe_1,
  /* input*/ &from_super_source_to_t1_offset_0);
Module2Func(
  /*output*/ &from_t1_offset_1999_to_t1_offset_2001,
  /*output*/ &from_t1_offset_1999_to_t0_pe_1,
  /* input*/ &from_t1_offset_1_to_t1_offset_1999);
Module3Func(
  /*output*/ &from_t1_offset_2000_to_t1_offset_2002,
  /*output*/ &from_t1_offset_2000_to_t0_pe_1,
  /*output*/ &from_t1_offset_2000_to_t0_pe_0,
  /* input*/ &from_t1_offset_0_to_t1_offset_2000);
Module4Func(
  /*output*/ &from_t1_offset_2001_to_t1_offset_4001,
  /*output*/ &from_t1_offset_2001_to_t0_pe_1,
  /*output*/ &from_t1_offset_2001_to_t0_pe_0,
  /* input*/ &from_t1_offset_1999_to_t1_offset_2001);
Module5Func(
  /*output*/ &from_t1_offset_2002_to_t1_offset_4000,
  /*output*/ &from_t1_offset_2002_to_t0_pe_0,
  /* input*/ &from_t1_offset_2000_to_t1_offset_2002);
Module6Func(
  /*output*/ &from_t1_offset_4001_to_t0_pe_0,
  /* input*/ &from_t1_offset_2001_to_t1_offset_4001);
Module7Func(
  /*output*/ &from_t0_pe_0_to_super_sink,
  /* input*/ &from_t1_offset_1_to_t0_pe_0,
  /* input*/ &from_t1_offset_2000_to_t0_pe_0,
  /* input*/ &from_t1_offset_2001_to_t0_pe_0,
  /* input*/ &from_t1_offset_4001_to_t0_pe_0,
  /* input*/ &from_t1_offset_2002_to_t0_pe_0);
Module8Func(
  /*output*/ &from_t1_offset_4000_to_t0_pe_1,
  /* input*/ &from_t1_offset_2002_to_t1_offset_4000);
Module7Func(
  /*output*/ &from_t0_pe_1_to_super_sink,
  /* input*/ &from_t1_offset_0_to_t0_pe_1,
  /* input*/ &from_t1_offset_1999_to_t0_pe_1,
  /* input*/ &from_t1_offset_2000_to_t0_pe_1,
  /* input*/ &from_t1_offset_4000_to_t0_pe_1,
  /* input*/ &from_t1_offset_2001_to_t0_pe_1);
```

In the above code snippet, `Module1Func` to `Module6Func` and `Module8Func` are forwarding modules; they constitute the line buffers of the SODA microarchitecture.
Although unrolled, the line buffer size is still approximately two lines of pixels, i.e. 4000 pixels.
`Module7Func` is a computing module; it is instanciated twice.
The whole design is fully pipelined and can produce 2 pixel per cycle.
In general, the unroll factor can be set to any number that satisfies the throughput requirement.

## Best Practices

+ For non-iterative stencil, `unroll factor` shall be determined by the DRAM bandwidth, i.e. saturate the external bandwidth, since the resource is usually not the bottleneck
+ For iterative stencil, to use more PEs in a single iteration or to implement more iterations is yet to be explored
+ Note that `2.0` will be a `double` number. To generate `float`, use `2.0f`. This may help reduce DSP usage
+ SODA is tiling-based and the size of the tile is specified in the `input` keyword. The last dimension is omitted because it is not needed in the reuse buffer generation

## Projects Using SODA

+ Licheng Guo, Jason Lau, Yuze Chi, Jie Wang, Cody Hao Yu, Zhe Chen, Zhiru Zhang, Jason Cong . Analysis and Optimization of the Implicit Broadcasts in FPGA HLS to Improve Maximum Frequency. In DAC, 2020. [[PDF]](https://about.blaok.me/pub/dac20-hls-timing.pdf)
+ Jiajie Li, Yuze Chi, Jason Cong. [HeteroHalide: From Image Processing DSL to Efficient FPGA Acceleration](https://doi.org/10.1145/3373087.3375320). In FPGA, 2020. [[PDF]](https://about.blaok.me/pub/fpga20-heterohalide.pdf) [[Slides]](https://about.blaok.me/pub/fpga20-heterohalide.slides.pdf)
+ Young-kyu Choi, Yuze Chi, Jie Wang, Jason Cong. [FLASH: Fast, ParalleL, and Accurate Simulator for HLS](https://doi.org/10.1109/TCAD.2020.2970597). In TCAD, 2020. [[PDF]](https://about.blaok.me/pub/tcad20-flash.pdf)
+ Yi-Hsiang Lai, Yuze Chi, Yuwei Hu, Jie Wang, Cody Hao Yu, Yuan Zhou, Jason Cong, Zhiru Zhang. [HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing](https://doi.org/10.1145/3289602.3293910). In FPGA, 2019. (Best Paper Candidate) [[PDF]](https://about.blaok.me/pub/fpga19-heterocl.pdf) [[Slides]](https://about.blaok.me/pub/fpga19-heterocl.slides.pdf) [[Code]](https://github.com/cornell-zhang/heterocl)
+ Yuze Chi, Young-kyu Choi, Jason Cong, Jie Wang. [Rapid Cycle-Accurate Simulator for High-Level Synthesis](https://doi.org/10.1145/3289602.3293918). In FPGA, 2019. [[PDF]](https://about.blaok.me/pub/fpga19-flash.pdf) [[Slides]](https://about.blaok.me/pub/fpga19-flash.slides.pdf)
