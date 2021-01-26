#ifndef __LINALG_LIB_WRAPPER_H__
#define __LINALG_LIB_WRAPPER_H__

// Wrapper for linear algebra library (BLAS, LAPACK)

// #if !defined(USE_MKL) && !defined(USE_OPENBLAS) && !defined(USE_GPU)
// #define USE_OPENBLAS
// #endif

// #if (defined(USE_MKL) + defined(USE_OPENBLAS) + defined(USE_GPU) > 1)
// #error "Multiple BLAS libraries asked to be used"
// #endif

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_SET_NUM_THREADS mkl_set_num_threads
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#include <lapacke.h>
#define BLAS_SET_NUM_THREADS openblas_set_num_threads
#endif

#include "enum.h"

#ifdef USE_GPU
#include <cuda.h>
#include <cublasXt.h>
#endif

#ifdef USE_GPU
typedef cublasHandle_t linalg_handle_t;
#else
typedef int linalg_handle_t;
#endif

//ColMajor only!
void local_AB(linalg_handle_t handle, int m,  int n, int k, double alpha,  const double* A, int lda,
     const double* B, int ldb, double beta, double* C, int ldc,
     device_type storage_device,
     device_type compute_device);

void init_linalg_handle(linalg_handle_t* handle, device_type compute_device);


#endif

