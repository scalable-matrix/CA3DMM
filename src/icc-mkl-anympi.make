CC           = mpicc
USE_MKL      = 1
USE_OPENBLAS = 0

#If USE_GPU is not defined, set it
ifeq ($(strip $(USE_GPU)),)
USE_GPU     := 0
endif

NVCC         = nvcc

include common.make
