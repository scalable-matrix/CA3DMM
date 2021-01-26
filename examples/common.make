CA3DMM_INSTALL_DIR = ..

DEFS    = 
INCS    = -I$(CA3DMM_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS) -DDEBUG=0
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(CA3DMM_INSTALL_DIR)/lib/libca3dmm.a

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
OPENBLAS_INSTALL_DIR = ../../OpenBLAS-git/install
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

ifeq ($(strip $(USE_GPU)), 1)
DEFS     += -DUSE_GPU=1
LIBS     += -lcublas
INC     += -I$(OPENBLAS_INSTALL_DIR)/include

#TODO: Handle gcc case
LDFLAGS = -ccbin=mpicc -Xcompiler -std=gnu++98,-O3,-g,-fPIC -G -lcuda -lcudart -lcublas
CUFLAGS = $(DEFS) -O3 -g -Xcompiler -std=gnu++98,-O3,-g,-fPIC -G

LINALG_OBJ := linalg_gpu.cu.o linalg_cpu.c.o

LINKER = $(NVCC)
else
LINALG_OBJ := linalg_cpu.c.o 

LINKER = $(CC)
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = $(C_SRCS:.c=.exe)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.exe: %.c.o 
	$(NVCC) ../src/memory.c.o ../src/gpu.cu.o $(LDFLAGS) -o $@ $^ $(LIBS) 

clean:
	rm -f $(EXES) $(C_OBJS)
