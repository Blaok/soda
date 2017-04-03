.PHONY: csim cosim hw mktemp

CSIM_XCLBIN = blur-csim.xclbin
COSIM_XCLBIN = blur-cosim.xclbin
HW_XCLBIN = blur-hw.xclbin

KERNEL_SRCS = blur_kernel.cpp
KERNEL_NAME = blur_kernel
HOST_SRCS = blur.cpp halide_run.cpp
HOST_BIN = blur

SRC = src
OBJ = obj
BIN = bin
BIT = bit
RPT = rpt

CXX ?= g++
CLCXX ?= xocc

SDA_VER ?= 2016.3
XILINX_SDACCEL ?= /opt/tools/SDx/$(SDA_VER)
WITH_SDACCEL = SDA_VER=$(SDA_VER) with-sdaccel

HOST_CFLAGS = -g -Wall -DFPGA_DEVICE -DC_KERNEL -I$(XILINX_SDACCEL)/runtime/include/1_2
HOST_LFLAGS = -L$(XILINX_SDACCEL)/runtime/lib/x86_64 -lxilinxopencl -llmx6.0

XDEVICE = xilinx:adm-pcie-7v3:1ddr:3.0
HOST_CFLAGS += -DTARGET_DEVICE=\"$(XDEVICE)\"

CLCXX_OPT = $(CLCXX_OPT_LEVEL) $(DEVICE_REPO_OPT) --xdevice $(XDEVICE) $(KERNEL_DEFS) $(KERNEL_INCS)
CLCXX_OPT += --kernel $(KERNEL_NAME)
CLCXX_OPT += -s -g
CLCXX_CSIM_OPT = -t sw_emu
CLCXX_COSIM_OPT = -t hw_emu
CLCXX_HW_OPT = -t hw

csim: $(BIN)/$(HOST_BIN) $(BIT)/$(CSIM_XCLBIN)
	XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^

cosim: $(BIN)/$(HOST_BIN) $(BIT)/$(COSIM_XCLBIN)
	XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^

hw: $(BIN)/$(HOST_BIN) $(BIT)/$(HW_XCLBIN)
	$(WITH_SDACCEL) $^

mktemp:
	@TMP=$$(mktemp -d);mkdir $${TMP}/src;cp -r $(SRC)/* $${TMP}/src;cp makefile $${TMP};echo $${TMP}

$(BIN)/$(HOST_BIN): $(HOST_SRCS:%.cpp=$(OBJ)/%.o)
	@mkdir -p $(BIN)
	$(WITH_SDACCEL) $(CXX) $(HOST_LFLAGS) $^ -o $@

$(OBJ)/%.o: $(SRC)/%.cpp
	@mkdir -p $(OBJ)
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -MM -MP -MT $@ -MF $(@:.o=.d) $<
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -c $< -o $@

-include $(OBJ)/$(HOST_SRCS:%.cpp=%.d)

$(BIT)/$(CSIM_XCLBIN): $(SRC)/$(KERNEL_SRCS) $(BIN)/emconfig.json
	@mkdir -p $(BIT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_CSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@$(RM) -rf .Xil

$(BIT)/$(COSIM_XCLBIN): $(SRC)/$(KERNEL_SRCS) $(BIN)/emconfig.json
	@mkdir -p $(BIT)
	@mkdir -p $(RPT)
	@ln -sf ../_xocc_$(KERNEL_NAME)_$(COSIM_XCLBIN:%.xclbin=%.dir)/impl/kernels/$(KERNEL_NAME)/$(KERNEL_NAME)/solution_OCL_REGION_0/syn/report $(RPT)/cosim
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_COSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@$(RM) -rf .Xil

$(BIT)/$(HW_XCLBIN): $(SRC)/$(KERNEL_SRCS)
	@mkdir -p $(BIT)
	@mkdir -p $(RPT)
	@ln -sf ../_xocc_$(KERNEL_NAME)_$(HW_XCLBIN:%.xclbin=%.dir)/impl/kernels/$(KERNEL_NAME)/$(KERNEL_NAME)/solution_OCL_REGION_0/syn/report $(RPT)/hw
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_HW_OPT) $(CLCXX_OPT) -o $@ $<
	@$(RM) -rf .Xil

$(BIN)/emconfig.json:
	cd $(BIN);$(WITH_SDACCEL) emconfigutil --xdevice $(XDEVICE) $(DEVICE_REPO_OPT) --od .
