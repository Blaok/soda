.PHONY: csim cosim hw hls mktemp

APP ?= blur
SDA_VER ?= 2016.3
TILE_SIZE_DIM0 ?= 2000
TILE_SIZE_DIM1 ?= 1024
BURST_LENGTH ?= 2000
UNROLL_FACTOR ?= 16
KI ?= 16
KO ?= 16

CSIM_XCLBIN ?= $(APP)-csim-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1)-unroll$(UNROLL_FACTOR).xclbin
COSIM_XCLBIN ?= $(APP)-cosim-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1)-unroll$(UNROLL_FACTOR)-ki$(KI)-ko$(KO).xclbin
HW_XCLBIN ?= $(APP)-hw-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1)-unroll$(UNROLL_FACTOR)-ki$(KI)-ko$(KO).xclbin

KERNEL_SRCS ?= $(APP)_kernel.cpp
KERNEL_NAME ?= $(APP)_kernel
HOST_SRCS ?= $(APP)_run.cpp $(APP).cpp
HOST_ARGS ?=
HOST_BIN ?= $(APP)-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1)

SRC ?= src
OBJ ?= obj
BIN ?= bin
BIT ?= bit
RPT ?= rpt

CXX ?= g++
CLCXX ?= xocc

XILINX_SDACCEL ?= /opt/tools/xilinx/SDx/$(SDA_VER)
WITH_SDACCEL = SDA_VER=$(SDA_VER) with-sdaccel

HOST_CFLAGS = -std=c++0x -g -Wall -DFPGA_DEVICE -DC_KERNEL -I$(XILINX_SDACCEL)/runtime/include/1_2
HOST_LFLAGS = -L$(XILINX_SDACCEL)/runtime/lib/x86_64 -lxilinxopencl -llmx6.0 -ldl -lpthread -lz $(shell libpng-config --ldflags)

XDEVICE = xilinx:adm-pcie-7v3:1ddr:3.0
HOST_CFLAGS += -DTARGET_DEVICE=\"$(XDEVICE)\"
HOST_CFLAGS += -DTILE_SIZE_DIM0=$(TILE_SIZE_DIM0) -DTILE_SIZE_DIM1=$(TILE_SIZE_DIM1)

CLCXX_OPT = $(CLCXX_OPT_LEVEL) $(DEVICE_REPO_OPT) --xdevice $(XDEVICE) $(KERNEL_DEFS) $(KERNEL_INCS)
CLCXX_OPT += --kernel $(KERNEL_NAME)
CLCXX_OPT += -s -g
CLCXX_OPT += -DTILE_SIZE_DIM0=$(TILE_SIZE_DIM0) -DTILE_SIZE_DIM1=$(TILE_SIZE_DIM1) -DBURST_LENGTH=$(BURST_LENGTH) -DUNROLL_FACTOR=$(UNROLL_FACTOR) -DKI=${KI} -DKO=${KO}
CLCXX_CSIM_OPT = -t sw_emu
CLCXX_COSIM_OPT = -t hw_emu
CLCXX_HW_OPT = -t hw

csim: $(BIN)/$(HOST_BIN) $(BIT)/$(CSIM_XCLBIN)
	XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^ $(HOST_ARGS)

cosim: $(BIN)/$(HOST_BIN) $(BIT)/$(COSIM_XCLBIN)
	XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^ $(HOST_ARGS)

hw: $(BIN)/$(HOST_BIN) $(BIT)/$(HW_XCLBIN)
	$(WITH_SDACCEL) $^ $(HOST_ARGS)

hls: $(OBJ)/$(HW_XCLBIN:.xclbin=.xo)

mktemp:
	@TMP=$$(mktemp -d --suffix=-sdaccel-2016.3-halide1-tmp);mkdir $${TMP}/src;cp -r $(SRC)/* $${TMP}/src;cp makefile $${TMP};echo -e "#!$${SHELL}\nrm \$$0;cd $${TMP}\n$${SHELL} \$$@ && rm -r $${TMP}" > mktemp.sh;chmod +x mktemp.sh

$(BIN)/$(HOST_BIN): $(HOST_SRCS:%.cpp=$(OBJ)/%-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1).o)
	@mkdir -p $(BIN)
	$(WITH_SDACCEL) $(CXX) $(HOST_LFLAGS) $^ -o $@

$(OBJ)/%-tile$(TILE_SIZE_DIM0)x$(TILE_SIZE_DIM1).o: $(SRC)/%.cpp
	@mkdir -p $(OBJ)
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -MM -MP -MT $@ -MF $(@:.o=.d) $<
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -c $< -o $@

-include $(OBJ)/$(HOST_SRCS:%.cpp=%.d)

$(BIT)/$(CSIM_XCLBIN): $(SRC)/$(KERNEL_SRCS) $(BIN)/emconfig.json
	@mkdir -p $(BIT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_CSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@getchild(){ for PID in $$(ps --ppid $$1 --no-headers -o pid);do grep 'Name:\s*xocc' /proc/$$PID/status -qs && echo -n "$$PID ";getchild $$PID;done; };EXCLUDED=$$(getchild $$PPID);rm -rf $$(ls -d .Xil/*|grep -vE "\-($${EXCLUDED// /|})-" 2>/dev/null)
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null

$(BIT)/$(COSIM_XCLBIN): $(SRC)/$(KERNEL_SRCS) $(BIN)/emconfig.json
	@mkdir -p $(BIT)
	@mkdir -p $(RPT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_COSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@getchild(){ for PID in $$(ps --ppid $$1 --no-headers -o pid);do grep 'Name:\s*xocc' /proc/$$PID/status -qs && echo -n "$$PID ";getchild $$PID;done; };EXCLUDED=$$(getchild $$PPID);rm -rf $$(ls -d .Xil/*|grep -vE "\-($${EXCLUDED// /|})-" 2>/dev/null)
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null

$(BIT)/$(HW_XCLBIN): $(OBJ)/$(HW_XCLBIN:.xclbin=.xo)
	@mkdir -p $(BIT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_HW_OPT) $(CLCXX_OPT) -l -o $@ $<
	@getchild(){ for PID in $$(ps --ppid $$1 --no-headers -o pid);do grep 'Name:\s*xocc' /proc/$$PID/status -qs && echo -n "$$PID ";getchild $$PID;done; };EXCLUDED=$$(getchild $$PPID);rm -rf $$(ls -d .Xil/*|grep -vE "\-($${EXCLUDED// /|})-" 2>/dev/null)
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null

$(OBJ)/$(HW_XCLBIN:.xclbin=.xo): $(SRC)/$(KERNEL_SRCS)
	@mkdir -p $(OBJ)
	@mkdir -p $(RPT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_HW_OPT) $(CLCXX_OPT) -c -o $@ $<
	@cp _xocc_compile_$(KERNEL_SRCS:%.cpp=%)_$(HW_XCLBIN:%.xclbin=%.dir)/impl/kernels/$(KERNEL_NAME)/$(KERNEL_NAME)/solution_OCL_REGION_0/syn/report/*.rpt $(RPT)/
	@getchild(){ for PID in $$(ps --ppid $$1 --no-headers -o pid);do grep 'Name:\s*xocc' /proc/$$PID/status -qs && echo -n "$$PID ";getchild $$PID;done; };EXCLUDED=$$(getchild $$PPID);rm -rf $$(ls -d .Xil/*|grep -vE "\-($${EXCLUDED// /|})-" 2>/dev/null)
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null

$(BIN)/emconfig.json:
	@mkdir -p $(BIN)
	cd $(BIN);$(WITH_SDACCEL) emconfigutil --xdevice $(XDEVICE) $(DEVICE_REPO_OPT) --od .

