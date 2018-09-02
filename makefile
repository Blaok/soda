.PHONY: csim cosim hw hls exe kernel bitstream check-afi-status mktemp check-git-status

APP ?= blur
SDA_VER ?= 2017.4
TILE_SIZE_DIM_0 ?= 32
#TILE_SIZE_DIM_1 ?= 32
UNROLL_FACTOR ?= 2
HOST_ARGS ?= 32 32
HOST_SRCS ?= $(APP)_run.cpp
DRAM_BANK ?= 1
ITERATE ?= 1
CLUSTER ?= none
BORDER ?= ignore

ifneq ("$(REPLICATION_FACTOR)","")
FACTOR_ARGUMENT := --replication-factor $(REPLICATION_FACTOR)
FACTOR_SUFFIX := replicate$(REPLICATION_FACTOR)
else
FACTOR_ARGUMENT := --unroll-factor $(UNROLL_FACTOR)
FACTOR_SUFFIX := unroll$(UNROLL_FACTOR)
endif

CSIM_XCLBIN ?= $(APP)-csim-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-$(FACTOR_SUFFIX)-$(DRAM_BANK)ddr$(if $(DRAM_SEPARATE),-separated)-iterate$(ITERATE)-border-$(BORDER)d-$(CLUSTER)-clustered.xclbin
COSIM_XCLBIN ?= $(APP)-cosim-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-$(FACTOR_SUFFIX)-$(DRAM_BANK)ddr$(if $(DRAM_SEPARATE),-separated)-iterate$(ITERATE)-border-$(BORDER)d-$(CLUSTER)-clustered.xclbin
HW_XCLBIN ?= $(APP)-hw-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-$(FACTOR_SUFFIX)-$(DRAM_BANK)ddr$(if $(DRAM_SEPARATE),-separated)-iterate$(ITERATE)-border-$(BORDER)d-$(CLUSTER)-clustered.xclbin

KERNEL_SRCS ?= $(APP)_kernel-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-$(FACTOR_SUFFIX)-$(DRAM_BANK)ddr$(if $(DRAM_SEPARATE),-separated)-iterate$(ITERATE)-border-$(BORDER)d-$(CLUSTER)-clustered.cpp
KERNEL_NAME ?= $(APP)_kernel
HOST_BIN ?= $(APP)-tile$(TILE_SIZE_DIM_0)$(if $(TILE_SIZE_DIM_1),x$(TILE_SIZE_DIM_1))-iterate$(ITERATE)-border-$(BORDER)d
HOST_SRCS += $(HOST_BIN).cpp

SHELL = /bin/bash
SODA_SRC ?= src
LABEL ?= $(COMMIT)
OBJ ?= obj/$(LABEL)/$(word 2,$(subst :, ,$(XDEVICE)))
BIN ?= bin/$(LABEL)/$(word 2,$(subst :, ,$(XDEVICE)))
BIT ?= bit/$(LABEL)/$(word 2,$(subst :, ,$(XDEVICE)))
RPT ?= rpt/$(LABEL)/$(word 2,$(subst :, ,$(XDEVICE)))
TMP ?= tmp/$(LABEL)/$(word 2,$(subst :, ,$(XDEVICE)))
SRC ?= $(TMP)

AWS_AFI_DIR ?= afis
AWS_AFI_LOG ?= logs
CXX ?= g++
CLCXX ?= xocc
COMMIT := $(shell git rev-parse --short HEAD)$(shell git diff --exit-code --quiet || echo '-dirty')

XILINX_SDACCEL ?= /opt/tools/xilinx/SDx/$(SDA_VER)
WITH_SDACCEL = SDA_VER=$(SDA_VER) with-sdaccel
SUPPORTED_XDEVICES = xilinx:adm-pcie-7v3:1ddr:3.0 xilinx:aws-vu9p-f1:4ddr-xpr-2pr:4.0 xilinx:adm-pcie-ku3:2ddr:3.3 xilinx:adm-pcie-ku3:2ddr-xpr:3.3 xilinx:adm-pcie-ku3:2ddr-xpr:4.0 xilinx:vcu1525:dynamic:5.0 xilinx:aws-vu9p-f1-04261818:dynamic:5.0

HOST_CFLAGS ?= -fopenmp -I$(TMP)
HOST_LFLAGS ?= -fopenmp

ifdef AWS_BUCKET
XDEVICE ?= xilinx:aws-vu9p-f1-04261818:dynamic:5.0
else # AWS_BUCKET
XDEVICE ?= xilinx:adm-pcie-7v3:1ddr:3.0
endif # AWS_BUCKET

ifeq (,$(findstring $(XDEVICE),$(SUPPORTED_XDEVICES)))
$(error $(XDEVICE) is not supported)
endif

HOST_CFLAGS += -DTILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),-DTILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) -DUNROLL_FACTOR=$(UNROLL_FACTOR)

CLCXX_OPT += -DTILE_SIZE_DIM_0=$(TILE_SIZE_DIM_0) $(if $(TILE_SIZE_DIM_1),-DTILE_SIZE_DIM_1=$(TILE_SIZE_DIM_1)) -DUNROLL_FACTOR=$(UNROLL_FACTOR)
ifneq (,$(findstring aws-vu9p-f1,$(XDEVICE)))
CLCXX_OPT += --xp "vivado_param:project.runs.noReportGeneration=false"
endif
ifneq ("$(XDEVICE)","xilinx:adm-pcie-7v3:1ddr:3.0")
CLCXX_OPT += $(shell for bundle in $$(grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=chan[0-9]+bank0+[io]' $(addprefix $(TMP)/,$(KERNEL_SRCS))|grep -oE 'chan[0-9]+bank[0-9]+[io]'|sort -u);do echo -n "--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_$${bundle^^}.core.OCL_REGION_0.M00_AXI ";done)
CLCXX_OPT += $(shell for bundle in $$(grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=chan[0-9]+bank1+[io]' $(addprefix $(TMP)/,$(KERNEL_SRCS))|grep -oE 'chan[0-9]+bank[0-9]+[io]'|sort -u);do echo -n "--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_$${bundle^^}.core.OCL_REGION_0.M01_AXI ";done)
CLCXX_OPT += $(shell for bundle in $$(grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=chan[0-9]+bank2+[io]' $(addprefix $(TMP)/,$(KERNEL_SRCS))|grep -oE 'chan[0-9]+bank[0-9]+[io]'|sort -u);do echo -n "--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_$${bundle^^}.core.OCL_REGION_0.M02_AXI ";done)
CLCXX_OPT += $(shell for bundle in $$(grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=chan[0-9]+bank3+[io]' $(addprefix $(TMP)/,$(KERNEL_SRCS))|grep -oE 'chan[0-9]+bank[0-9]+[io]'|sort -u);do echo -n "--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_$${bundle^^}.core.OCL_REGION_0.M03_AXI ";done)
CLCXX_OPT += $(shell for bundle in $$(grep -E '^\s*\#pragma\s+[Hh][Ll][Ss]\s+[Ii][Nn][Tt][Ee][Rr][Ff][Aa][Cc][Ee]\s+.*bundle=gmem[0-9]+'       $(addprefix $(TMP)/,$(KERNEL_SRCS))|grep -oE 'gmem[0-9]+'                  );do echo -n "--xp misc:map_connect=add.kernel.$(APP)_kernel_1.M_AXI_$${bundle^^}.core.OCL_REGION_0.M00_AXI ";done)
endif # ifneq ("$(XDEVICE)","xilinx:adm-pcie-7v3:1ddr:3.0")

include sdaccel-examples/makefile

check-git-status:
	@echo $(COMMIT)

$(TMP)/$(KERNEL_SRCS): $(SODA_SRC)/$(APP).soda
	@mkdir -p $(TMP)
	src/sodac $(FACTOR_ARGUMENT) --tile-size $(TILE_SIZE_DIM_0) $(TILE_SIZE_DIM_1) --dram-in $(DRAM_IN) --dram-out $(DRAM_OUT) --iterate $(ITERATE) --border $(BORDER) --cluster $(CLUSTER) --kernel-file $@ $^

$(TMP)/$(HOST_BIN).cpp: $(SODA_SRC)/$(APP).soda
	@mkdir -p $(TMP)
	src/sodac $(FACTOR_ARGUMENT) --tile-size $(TILE_SIZE_DIM_0) $(TILE_SIZE_DIM_1) --dram-in $(DRAM_IN) --dram-out $(DRAM_OUT) --iterate $(ITERATE) --border $(BORDER) --cluster $(CLUSTER) --source-file $@ $^

$(TMP)/$(APP).h: $(SODA_SRC)/$(APP).soda
	@mkdir -p $(TMP)
	src/sodac $(FACTOR_ARGUMENT) --tile-size $(TILE_SIZE_DIM_0) $(TILE_SIZE_DIM_1) --dram-in $(DRAM_IN) --dram-out $(DRAM_OUT) --iterate $(ITERATE) --border $(BORDER) --cluster $(CLUSTER) --header-file $@ $^

$(OBJ)/%.o: $(TMP)/%.cpp $(TMP)/$(APP).h
	@mkdir -p $(OBJ)
	$(CXX) $(HOST_CFLAGS) -MM -MP -MT $@ -MF $(@:.o=.d) $<
	$(CXX) $(HOST_CFLAGS) -c $< -o $@

$(SRC)/$(APP)_run.cpp: $(SODA_SRC)/$(APP)_run.cpp
	ln -sf $(abspath $<) $@
