LIB_A   = libca3dmm.a
LIB_SO  = libca3dmm.so

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
LIB_OBJS = $(filter-out utils.c.o memory.c.o, $(C_OBJS))

DEFS    = 
INCS    = 
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS) -DDEBUG=0

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

ifeq ($(strip $(USE_GPU)), 1)
DEFS     += -DUSE_GPU=1
LIB     += -lcublas
INCS     += -I$(CUDA_ROOT)/include

#TODO: Handle gcc case
LDFLAGS = -ccbin=mpicc -Xcompiler -std=gnu++98,-mkl,-O3,-xHost,-g,-fPIC -G -lcuda -lcudart -lcublas -arch=sm_70 -gencode=arch=compute_70,code=sm_70
CUFLAGS = $(DEFS) -O3 -g -Xcompiler -std=gnu++98,-O3,-g,-fPIC -G -arch=sm_70 -gencode=arch=compute_70,code=sm_70

LINALG_OBJ := linalg_gpu.cu.o linalg_cpu.c.o

LINKER = $(NVCC) -ccbin=$(CC)
EXES = carma_test2 
else
LINALG_OBJ := linalg_cpu.c.o 

LINKER = $(CC)
EXES = carma_test carma_test2 carma_test2-ATA
endif



# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: install $(LIB_OBJS)

#gpu.cu.o memory.c.o utils.c.o
install: $(LIB_A) $(LIB_SO) 
	mkdir -p ../lib
	mkdir -p ../include
	cp -u $(LIB_A)  ../lib/$(LIB_A)
	cp -u $(LIB_SO) ../lib/$(LIB_SO)
	cp -u *.h ../include/

$(LIB_A): $(LIB_OBJS)  $(LINALG_OBJ)
	$(AR) $@ $^

$(LIB_SO): $(LIB_OBJS)  $(LINALG_OBJ)
	$(LINKER) -shared -o $@ $^

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.cu.o: %.cu
	$(NVCC) $(CUFLAGS) -c $^ -o $@


clean:
	rm -f $(C_OBJS) $(LIB_A) $(LIB_SO)
