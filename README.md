# SUPO: Stencil with Unrolling and Pipelining Optimizations

## SUPO DSL Example

    # comments start with hashtag(#)
    
    kernel: blur      # the kernel name, will be used as the kernel name in HLS
    burst width: 512  # DRAM burst I/O width in bits, for Xilinx platform by default is 512
    dram separate: no # whether to use separated DRAM banks for read and write
    dram bank: 1      # how many DRAM banks to use in total
    unroll factor: 16 # how many pixels are generated per cycle
    
    # specify the type, name, dimension, and channel of input tile, the last dimension is not needed
    input uint16: input[1](2000,)
    
    # specify an intermediate stage of computation, may appear 0 or more times
    buffer uint16: tmp[0](0,0) = (input[0](-1,0)+input[0](0,0)+input[0](1,0))/3
    
    # specify the output
    # channel field is optional and defaults to channel 0
    output uint16: output(0,0) = (tmp(0,-1)+tmp(0,0)+tmp(0,1))/3
    
    # how many times the whole computation is repeated (if input has the same type and channel as output)
    iterate: 2
    
    # how to deal with border, currently 'preserve' and 'ignore' are available ('preserve' doesn't support tiling yet)
    border: preserve
    
    # constant values that may be referenced as coefficients or lookup tables (implementation currently broken)
    # array partitioning information can be passed to HLS code
    param uint16, partition cyclic factor=2 dim=1, partition cyclic factor=2 dim=2: p1[20][30]
    # keyword 'dup' allows simultaneous access to the same parameter
    param uint16, dup 3, partition complete: p2[20]
    
## TODOs

+ [ ] support multiple inputs & outputs

## Design Considerations

+ All keywords are mandatory except intermediate `buffer` and extra `param` are optional
+ If `dram separate: no`, the same DRAM bank will be used for read and write at the same time, bandwidth will be slightly less than `dram separate: yes`, but in the case where input and output doesn't have the same data width, using the same bank can alleviate the imbalance
+ For non-iterative stencil, `unroll factor` shall be determined by the DRAM bandwidth, i.e. saturate the external bandwidth, since the resource is usually not the bottleneck
+ For iterative stencil, to use more PEs in a single iteration or to implement more iterations is yet to be explored
+ Currently only `{{u,}int{8,16,32,64}, float, double}` types are implemented
+ Currently only `+,-,*,/%` operators are implemented
+ Currently `math.h` functions can be parsed but type induction is not fully implemented
+ Note that `2.0` will be a `double` number. To generate `float`, use `2.0f`. This may help reduce DSP usage
+ SUPO is tiling-based and the size of the tile is specified in the `input` keyword. The last dimension is omitted because it is not needed in the reuse buffer generation
+ The input, output, or the intermediate buffer can have more than 1 channels, which is helpful when dealing with things like RGB images


## Getting Started

### Prerequisites

+ SDAccel 2017.1 or 2017.2 (other versions not tested)
+ A script `with-sdaccel` in `PATH`, setting up environment variables for SDAccel
  - `with-sdaccel xocc` shall invoke `xocc` properly, with the arguments properly passed
  - Or comment out `WITH_SDACCEL` in `makefile` and setup environment variables before invoking `make`
+ A working target platform for `xocc`
  - `xilinx:adm-pcie-7v3:1ddr:3.0` (default) and `xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0` are tested

### Clone the Repo
    git clone https://github.com/Blaok/supo.git
    cd supo

### Generate HLS kernel code
    make kernel

### Run C-Sim
    make csim

### Generate HDL code
    make hls
    
### Run Co-Sim
    make cosim
    
### Generate FPGA Bitstream
    make bitstream
    
### Run Bitstream
    make hw # requires actual FPGA hardware and driver

## Code Snippets

### Configuration

+ 5-point 2D Jacobi
+ tile size is (2000,)

### Without Unrolling

    #pragma HLS dataflow
        load(input_stream_chan_0_bank_0, var_input_chan_0_bank_0, coalesced_data_num);
        unpack_float(
            t1_offset_0_chan_0,
            input_stream_chan_0_bank_0, coalesced_data_num);

        forward_2<float, 0>(from_t1_to_t0_param_4_chan_0_pe_0, t1_offset_1999_chan_0, t1_offset_0_chan_0, epoch_num);
        forward_2<float, 1999>(from_t1_to_t0_param_3_chan_0_pe_0, t1_offset_2000_chan_0, t1_offset_1999_chan_0, epoch_num);
        forward_2<float, 1>(from_t1_to_t0_param_2_chan_0_pe_0, t1_offset_2001_chan_0, t1_offset_2000_chan_0, epoch_num);
        forward_2<float, 1>(from_t1_to_t0_param_1_chan_0_pe_0, t1_offset_4000_chan_0, t1_offset_2001_chan_0, epoch_num);
        forward_1<float, 1999>(from_t1_to_t0_param_0_chan_0_pe_0, t1_offset_4000_chan_0, epoch_num);

        compute_t0<0>(t0_offset_0_chan_0, from_t1_to_t0_param_0_chan_0_pe_0, from_t1_to_t0_param_1_chan_0_pe_0, from_t1_to_t0_param_2_chan_0_pe_0, from_t1_to_t0_param_3_chan_0_pe_0, from_t1_to_t0_param_4_chan_0_pe_0, input_size_dim_0, input_
    size_dim_1, epoch_num);

        pack_float(output_stream_chan_0_bank_0,
            t0_offset_0_chan_0,
            coalesced_data_num);
        store(var_output_chan_0_bank_0, output_stream_chan_0_bank_0, coalesced_data_num);

### Unroll 2 Times

    #pragma HLS dataflow
        load(input_stream_chan_0_bank_0, var_input_chan_0_bank_0, coalesced_data_num);
        unpack_float(
            t1_offset_1_chan_0,
            t1_offset_0_chan_0,
            input_stream_chan_0_bank_0, coalesced_data_num);

        forward_2<float, 0>(from_t1_to_t0_param_4_chan_0_pe_1, t1_offset_2000_chan_0, t1_offset_0_chan_0, epoch_num);
        forward_2<float, 0>(from_t1_to_t0_param_4_chan_0_pe_0, t1_offset_1999_chan_0, t1_offset_1_chan_0, epoch_num);
        forward_2<float, 999>(from_t1_to_t0_param_3_chan_0_pe_1, t1_offset_2001_chan_0, t1_offset_1999_chan_0, epoch_num);
        forward_3<float, 1000>(from_t1_to_t0_param_2_chan_0_pe_1, from_t1_to_t0_param_3_chan_0_pe_0, t1_offset_2002_chan_0, t1_offset_2000_chan_0, epoch_num);
        forward_3<float, 1>(from_t1_to_t0_param_1_chan_0_pe_1, from_t1_to_t0_param_2_chan_0_pe_0, t1_offset_4001_chan_0, t1_offset_2001_chan_0, epoch_num);
        forward_2<float, 1>(from_t1_to_t0_param_1_chan_0_pe_0, t1_offset_4000_chan_0, t1_offset_2002_chan_0, epoch_num);
        forward_1<float, 999>(from_t1_to_t0_param_0_chan_0_pe_1, t1_offset_4000_chan_0, epoch_num);
        forward_1<float, 1000>(from_t1_to_t0_param_0_chan_0_pe_0, t1_offset_4001_chan_0, epoch_num);

        compute_t0<0>(t0_offset_1_chan_0, from_t1_to_t0_param_0_chan_0_pe_0, from_t1_to_t0_param_1_chan_0_pe_0, from_t1_to_t0_param_2_chan_0_pe_0, from_t1_to_t0_param_3_chan_0_pe_0, from_t1_to_t0_param_4_chan_0_pe_0, input_size_dim_0, input_
    size_dim_1, epoch_num);
        compute_t0<1>(t0_offset_0_chan_0, from_t1_to_t0_param_0_chan_0_pe_1, from_t1_to_t0_param_1_chan_0_pe_1, from_t1_to_t0_param_2_chan_0_pe_1, from_t1_to_t0_param_3_chan_0_pe_1, from_t1_to_t0_param_4_chan_0_pe_1, input_size_dim_0, input_
    size_dim_1, epoch_num);

        pack_float(output_stream_chan_0_bank_0,
            t0_offset_1_chan_0,
            t0_offset_0_chan_0,
            coalesced_data_num);
        store(var_output_chan_0_bank_0, output_stream_chan_0_bank_0, coalesced_data_num);
    
### Functions Explained

+ `load`/`store` performs burst DRAM I/O
+ `unpack`/`pack` performs memory coalescing
+ `forward` forwards data from source to multiple destintaions, with a certain number of cycles' delay
  - They form the reuse buffers
+ `compute` computes output using the inputs
  - They are the PEs

