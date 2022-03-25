LIB_A   = libca3dmm.a
LIB_SO  = libca3dmm.so

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
CU_SRCS = $(wildcard *.cu)
CU_OBJS = $(CU_SRCS:.cu=.cu.o)
OBJS    = $(C_OBJS)

DEFS    = 
INCS    = 
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)

GENCODE_SM60  = -gencode arch=compute_60,code=sm_60
GENCODE_SM70  = -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS = $(GENCODE_SM60) $(GENCODE_SM70)

CUDA_PATH   ?= /usr/local/cuda-10.0
NVCC        = nvcc
NVCCFLAGS   = -O3 -g --compiler-options -fPIC $(GENCODE_FLAGS)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
AR      = xiar rcs
CFLAGS += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS   += -DUSE_MKL
CFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
OPENBLAS_INSTALL_DIR = ../../OpenBLAS-git/install
DEFS   += -DUSE_OPENBLAS
INCS   += -I$(OPENBLAS_INSTALL_DIR)/include
endif

ifeq ($(strip $(USE_CUDA)), 1)
OBJS   += $(CU_OBJS)
DEFS   += -DUSE_CUDA
endif

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(OBJS)

all: install

install: $(LIB_A) $(LIB_SO)
	mkdir -p ../lib
	mkdir -p ../include
	cp -u $(LIB_A)  ../lib/$(LIB_A)
	cp -u $(LIB_SO) ../lib/$(LIB_SO)
	cp -u *.h ../include/

$(LIB_A): $(OBJS) 
	$(AR) $@ $^

$(LIB_SO): $(OBJS) 
	$(CC) -shared -o $@ $^

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $^

clean:
	rm $(OBJS) $(LIB_A) $(LIB_SO)