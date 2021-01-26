#ifndef GPU_H
#define GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasXt.h>
#include <stdio.h>

#if TRY_CAM
#include <mpi.h>
#include <mpi-ext.h>
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#define USE_CAM 1
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BDIMX 16 // tile (and threadblock) size in x
#define BDIMY 16 // tile (and threadblock) size in y

void gpuAssert(cudaError_t code, const char *file, int line);
double gpu_sum(const double* vec, int n);
void gpu_print(const double* vec, int n);
void gpu_print_mat(const double* vec, int m, int n);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void cublasAssert(cublasStatus_t code, const char *file, int line);
#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }

#ifdef __cplusplus
}
#endif

#endif
