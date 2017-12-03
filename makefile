.PHONY: csim cosim hw hls exe bitstream check-afi-status check-aws-bucket mktemp

APP ?= blur
SDA_VER := 2017.1
TILE_SIZE_DIM_0 ?= 8000
#TILE_SIZE_DIM_1 ?= 1024
UNROLL_FACTOR ?= 64
HOST_ARGS ?= 8000 8000
DRAM_CHAN ?= 1

CSIM_XCLBIN ?= $(APP)-csim-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-unroll$(UNROLL_FACTOR)-$(DRAM_CHAN)ddr$(if $(DRAM_SEPARATE),-separated).xclbin
COSIM_XCLBIN ?= $(APP)-cosim-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-unroll$(UNROLL_FACTOR)-$(DRAM_CHAN)ddr$(if $(DRAM_SEPARATE),-separated).xclbin
HW_XCLBIN ?= $(APP)-hw-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-unroll$(UNROLL_FACTOR)-$(DRAM_CHAN)ddr$(if $(DRAM_SEPARATE),-separated).xclbin

KERNEL_SRCS ?= $(APP)_kernel-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-unroll$(UNROLL_FACTOR)-$(DRAM_CHAN)ddr$(if $(DRAM_SEPARATE),-separated).cpp
KERNEL_NAME ?= $(APP)_kernel
HOST_SRCS ?= $(APP)_run.cpp $(APP).cpp
HOST_BIN ?= $(APP)-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))

SRC ?= src
OBJ ?= obj/$(word 2,$(subst :, ,$(XDEVICE)))
BIN ?= bin/$(word 2,$(subst :, ,$(XDEVICE)))
BIT ?= bit/$(word 2,$(subst :, ,$(XDEVICE)))
RPT ?= rpt/$(word 2,$(subst :, ,$(XDEVICE)))
TMP ?= tmp/$(word 2,$(subst :, ,$(XDEVICE)))

AWS_AFI_DIR ?= afis
AWS_AFI_LOG ?= logs
CXX ?= g++
CLCXX ?= xocc

XILINX_SDACCEL ?= /opt/tools/xilinx/SDx/$(SDA_VER)
WITH_SDACCEL = SDA_VER=$(SDA_VER) with-sdaccel

HOST_CFLAGS = -std=c++0x -g -Wall -DFPGA_DEVICE -DC_KERNEL -I$(XILINX_SDACCEL)/runtime/include/1_2
HOST_LFLAGS = -L$(XILINX_SDACCEL)/runtime/lib/x86_64 -lxilinxopencl -lrt -ldl -lpthread -lz $(shell libpng-config --ldflags)

XDEVICE ?= xilinx:adm-pcie-7v3:1ddr:3.0
XDEVICE := xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0
ifeq ($(SDA_VER),2017.2)
	XILINX_SDX ?= /opt/tools/xilinx/SDx/$(SDA_VER)
	HOST_CFLAGS += -DTARGET_DEVICE=\"$(subst :,_,$(subst .,_,$(XDEVICE)))\"
else
	HOST_CFLAGS += -DTARGET_DEVICE=\"$(XDEVICE)\"
endif
HOST_CFLAGS += -DTILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),-DTILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) -DUNROLL_FACTOR=$(UNROLL_FACTOR)

CLCXX_OPT = $(CLCXX_OPT_LEVEL) $(DEVICE_REPO_OPT) --platform $(XDEVICE) $(KERNEL_DEFS) $(KERNEL_INCS)
CLCXX_OPT += --kernel $(KERNEL_NAME)
CLCXX_OPT += -s -g --temp_dir $(TMP)
CLCXX_OPT += -DTILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),-DTILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) -DUNROLL_FACTOR=$(UNROLL_FACTOR)
CLCXX_OPT += $(if $(shell grep -E "^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=gmem0" $(TMP)/$(KERNEL_SRCS)),--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_GMEM0.core.OCL_REGION_0.M00_AXI)
CLCXX_OPT += $(if $(shell grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=gmem1' $(TMP)/$(KERNEL_SRCS)),--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_GMEM1.core.OCL_REGION_0.M01_AXI)
CLCXX_OPT += $(if $(shell grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=gmem2' $(TMP)/$(KERNEL_SRCS)),--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_GMEM2.core.OCL_REGION_0.M02_AXI)
CLCXX_OPT += $(if $(shell grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=gmem3' $(TMP)/$(KERNEL_SRCS)),--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_GMEM3.core.OCL_REGION_0.M03_AXI)
CLCXX_CSIM_OPT = -t sw_emu
CLCXX_COSIM_OPT = -t hw_emu
CLCXX_HW_OPT = -t hw

csim: $(BIN)/$(HOST_BIN) $(BIT)/$(CSIM_XCLBIN)
	@echo DRAM_CHAN=$(DRAM_CHAN) $(if $(DRAM_SEPARATE),DRAM_SEPARATE=) XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^ $(HOST_ARGS)
	@ulimit -s unlimited;DRAM_CHAN=$(DRAM_CHAN) $(if $(DRAM_SEPARATE),DRAM_SEPARATE=) XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^ $(HOST_ARGS)

cosim: $(BIN)/$(HOST_BIN) $(BIT)/$(COSIM_XCLBIN)
	XCL_EMULATION_MODE=true $(WITH_SDACCEL) $^ $(HOST_ARGS)

ifeq ($(XDEVICE),"xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0")
bitstream: $(BIT)/$(HW_XCLBIN:.xclbin=.awsxclbin)

hw: $(BIN)/$(HOST_BIN) $(BIT)/$(HW_XCLBIN:.xclbin=.awsxclbin)
	$(WITH_SDACCEL) $^ $(HOST_ARGS)
else
bitstream: $(BIT)/$(HW_XCLBIN)

hw: $(BIN)/$(HOST_BIN) $(BIT)/$(HW_XCLBIN)
	$(WITH_SDACCEL) $^ $(HOST_ARGS)
endif

hls: $(OBJ)/$(HW_XCLBIN:.xclbin=.xo)

exe: $(BIN)/$(HOST_BIN)

check-afi-status:
	@echo -n 'AFI state: ';aws ec2 describe-fpga-images --fpga-image-ids $$(jq -r '.FpgaImageId' $(BIT)/$(HW_XCLBIN:.xclbin=.afi))|jq '.FpgaImages[0].State.Code' -r

mktemp:
	@TMP=$$(mktemp -d --suffix=-sdaccel-stencil-tmp);mkdir $${TMP}/src;cp -r $(SRC)/* $${TMP}/src;cp makefile generate-kernel.py $${TMP};echo -e "#!$${SHELL}\nrm \$$0;cd $${TMP}\n$${SHELL} \$$@ && rm -r $${TMP}" > mktemp.sh;chmod +x mktemp.sh

$(TMP)/$(KERNEL_SRCS): $(SRC)/$(APP).json
	@echo DRAM_CHAN=$(DRAM_CHAN) $(if $(DRAM_SEPARATE),DRAM_SEPARATE=) UNROLL_FACTOR=$(UNROLL_FACTOR) TILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),TILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) ./generate-kernel.py\<$^\>$@
	@TMP=$$(mktemp --suffix='generate-kernel.py');if DRAM_CHAN=$(DRAM_CHAN) $(if $(DRAM_SEPARATE),DRAM_SEPARATE=) UNROLL_FACTOR=$(UNROLL_FACTOR) TILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),TILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) ./generate-kernel.py<$^>$${TMP};then mv $${TMP} $@;else rm $${TMP};exit 1;fi

$(BIN)/$(HOST_BIN): $(HOST_SRCS:%.cpp=$(OBJ)/%-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1)).o)
	@mkdir -p $(BIN)
	$(WITH_SDACCEL) $(CXX) $(HOST_LFLAGS) $^ -o $@

$(OBJ)/%-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1)).o: $(SRC)/%.cpp $(SRC)/$(APP)_params.h
	@mkdir -p $(OBJ)
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -MM -MP -MT $@ -MF $(@:.o=.d) $<
	$(WITH_SDACCEL) $(CXX) $(HOST_CFLAGS) -c $< -o $@

-include $(OBJ)/$(HOST_SRCS:%.cpp=%.d)

$(BIT)/$(CSIM_XCLBIN): $(TMP)/$(KERNEL_SRCS) $(BIN)/emconfig.json $(SRC)/$(APP)_params.h
	@mkdir -p $(BIT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_CSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@rm -rf $$(ls -d .Xil/xocc-*-$$(cat /etc/hostname) 2>/dev/null|grep -vE "\-($$(pgrep xocc|tr '\n' '|'))-")
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null; exit 0

$(BIT)/$(COSIM_XCLBIN): $(TMP)/$(KERNEL_SRCS) $(BIN)/emconfig.json $(SRC)/$(APP)_params.h
	@mkdir -p $(BIT)
	@mkdir -p $(RPT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_COSIM_OPT) $(CLCXX_OPT) -o $@ $<
	@rm -rf $$(ls -d .Xil/xocc-*-$$(cat /etc/hostname) 2>/dev/null|grep -vE "\-($$(pgrep xocc|tr '\n' '|'))-")
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null; exit 0

$(BIT)/$(HW_XCLBIN): $(OBJ)/$(HW_XCLBIN:.xclbin=.xo)
	@mkdir -p $(BIT)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_HW_OPT) $(CLCXX_OPT) -l -o $@ $<
	@mkdir -p $(RPT)/$(HW_XCLBIN:.xclbin=)
	@cp $(TMP)/_xocc_link_$(HW_XCLBIN:.xclbin=)_$(HW_XCLBIN:%.xclbin=%.dir)/impl/build/system/$(HW_XCLBIN:.xclbin=)/bitstream/$(HW_XCLBIN:%.xclbin=%_ipi)/{vivado.log,ipiimpl/ipiimpl.runs/impl_1/{*_timing_summary,kernel_util}_routed.rpt} $(RPT)/$(HW_XCLBIN:.xclbin=)
	@rm -rf $$(ls -d .Xil/xocc-*-$$(cat /etc/hostname) 2>/dev/null|grep -vE "\-($$(pgrep xocc|tr '\n' '|'))-")
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null; exit 0

check-aws-bucket:
ifndef AWS_BUCKET
	$(error AWS_BUCKET must be set to an available AWS S3 bucket)
endif

$(BIT)/$(HW_XCLBIN:.xclbin=.awsxclbin): check-aws-bucket $(BIT)/$(HW_XCLBIN)
	@TMP=$$(mktemp -d);ln -rs ${BIT}/$(HW_XCLBIN) $${TMP};pushd $${TMP} >/dev/null;create-sdaccel-afi -xclbin=$(HW_XCLBIN) -o=$(HW_XCLBIN:.xclbin=) -s3_bucket=$(AWS_BUCKET) -s3_dcp_key=$(AWS_AFI_DIR) -s3_logs_key=$(AWS_AFI_LOG);popd >/dev/null;mv $${TMP}/$(HW_XCLBIN:.xclbin=.awsxclbin) $(BIT);mv $${TMP}/*afi_id.txt $(BIT)/$(HW_XCLBIN:.xclbin=.afi);rm -rf $${TMP}

$(OBJ)/$(HW_XCLBIN:.xclbin=.xo): $(TMP)/$(KERNEL_SRCS) $(SRC)/$(APP)_params.h
	@mkdir -p $(OBJ)
	$(WITH_SDACCEL) $(CLCXX) $(CLCXX_HW_OPT) $(CLCXX_OPT) -c -o $@ $<
	@mkdir -p $(RPT)/$(HW_XCLBIN:.xclbin=)
	@cp $(TMP)/_xocc_compile_$(KERNEL_SRCS:%.cpp=%)_$(HW_XCLBIN:%.xclbin=%.dir)/impl/kernels/$(KERNEL_NAME)/vivado_hls.log $(RPT)/$(HW_XCLBIN:.xclbin=)
	@cp $(TMP)/_xocc_compile_$(KERNEL_SRCS:%.cpp=%)_$(HW_XCLBIN:%.xclbin=%.dir)/impl/kernels/$(KERNEL_NAME)/$(KERNEL_NAME)/solution_OCL_REGION_0/syn/report/*.rpt $(RPT)/$(HW_XCLBIN:.xclbin=)
	@rm -rf $$(ls -d .Xil/xocc-*-$$(cat /etc/hostname) 2>/dev/null|grep -vE "\-($$(pgrep xocc|tr '\n' '|'))-")
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null; exit 0

$(BIN)/emconfig.json:
	@mkdir -p $(BIN)
	$(WITH_SDACCEL) emconfigutil --platform $(XDEVICE) $(DEVICE_REPO_OPT) --od $(BIN)
	@rm -rf $$(ls -d .Xil/configutil-*-$$(cat /etc/hostname) 2>/dev/null|grep -vE "\-($$(pgrep emconfigutil|tr '\n' '|'))-")
	@rmdir .Xil --ignore-fail-on-non-empty 2>/dev/null; exit 0

