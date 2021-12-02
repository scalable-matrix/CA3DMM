// Wrap up some CUDA operations for non-CUDA modules
#ifndef __CUDA_PROXY_H__
#define __CUDA_PROXY_H__

#include <stdint.h>

typedef enum
{
    // cuBLAS only supports column-major
    CublasNoTrans,
    CublasTrans,
    CublasConjTrans,
    CublasLeft,
    CublasRight,
    CublasUpper,
    CublasLower,
    CublasUnit,
    CublasNonUnit
} cublas_enum_t;

#ifdef __cplusplus
extern "C" {
#endif

// This can be called before calling MPI_Init()
void select_cuda_device_by_mpi_local_rank();

// ========== Memory operations ========== //

void cuda_set_rt_dev_id(const int dev_id);

void cuda_memcpy_h2d(const void *hptr, void *dptr, const size_t bytes);

void cuda_memcpy_d2h(const void *dptr, void *hptr, const size_t bytes);

void cuda_memcpy_d2d(const void *dptr_src, void *dptr_dst, const size_t bytes);

void cuda_memcpy_auto(const void *src, void *dst, const size_t bytes);

// Note: for memcpy_2d, ld* are in unit of bytes, and col_bytes should <= ld*
void cuda_memcpy_2d_h2d(const void *hptr, const size_t ldh, void *dptr, const size_t ldd, const size_t nrow, const size_t col_bytes);

void cuda_memcpy_2d_d2h(const void *dptr, const size_t ldd, void *hptr, const size_t ldh, const size_t nrow, const size_t col_bytes);

void cuda_memcpy_2d_d2d(const void *dptr_src, const size_t lds, void *dptr_dst, const size_t ldd, const size_t nrow, const size_t col_bytes);

void cuda_memcpy_2d_auto(const void *src, const size_t lds, void *dst, const size_t ldd, const size_t nrow, const size_t col_bytes);

void cuda_malloc_dev(void **dptr_, const size_t bytes);

void cuda_malloc_host(void **hptr_, const size_t bytes);

void cuda_memset_dev(void *dptr, const int value, const size_t bytes);

void cuda_free_dev(void *dptr);

void cuda_free_host(void *hptr);

void cuda_device_sync();

void cuda_stream_sync(void *stream_p);

// Copy a row-major matrix block to another row-major matrix
// Input parameters:
//   dt_size  : Size of matrix element data type, in bytes, must be 4 or 8
//   nrow     : Number of rows to be copied
//   ncol     : Number of columns to be copied
//   src      : Size >= lds * nrow, source matrix
//   lds      : Leading dimension of src, >= ncol
//   ldd      : Leading dimension of dst, >= ncol
// Output parameter:
//   dst : Size >= ldd * nrow, destination matrix
void cuda_copy_matrix_block(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd
);

// ========== cuBLAS and cuSOLVER functions ========== //

// y(1 : incy : n) := x(1 : incx : n)
void cuda_cublas_dcopy(const int n, const double *x, const int incx, double *y, const int incy);

// x(1 : incx : n) *= alpha
void cuda_cublas_dscal(const int n, const double alpha, double *x, const int incx);

// ret = dot(x(1 : incx : n), y(1 : incy : n))
double cuda_cublas_ddot(const int n, const double *x, const int incx, const double *y, const int incy);

// C := alpha * op(A) * op(B) + beta * C
void cuda_cublas_dgemm(
    cublas_enum_t transA, cublas_enum_t transB,
    const int m, const int n, const int k, const double alpha, 
    const double *A, const int ldA, const double *B, const int ldB,
    const double beta, double *C, const int ldC
);

// C := alpha * op(A) + beta * op(B)
void cuda_cublas_dgeam(
    cublas_enum_t transA, cublas_enum_t transB,
    const int m, const int n, 
    const double alpha, const double *A, const int ldA,
    const double beta,  const double *B, const int ldB,
    double *C, const int ldC
);

// Solve X for op(A) * X = alpha * B or X * op(A) = alpha * B
void cuda_cublas_dtrsm(
    cublas_enum_t side, cublas_enum_t uplo,
    cublas_enum_t trans, cublas_enum_t diag,
    const int m, const int n, const double alpha,
    const double *A, const int ldA, double *B, const int ldB
);

// Cholesky factorization A = L * L^T or A = U^T * U
int cuda_cusolver_dpotrf(const char uplo, const int n, double *A, const int ldA);

// Standard eigen solver using divided-and-conquer algorithm
int cuda_cusolver_dsyevd(const char jobz, const char uplo, const int n, double *A, const int ldA, double *W);

// ========== BLAS-like operations ========== //

// x := alpha * x + beta
void cuda_daxpb(const int n, const double alpha, double *x, const double beta);

// y := alpha * x + beta * y
void cuda_daxpby(const int n, const double alpha, const double *x, const double beta, double *y);

// ========== Random number generator ========== //

// Uniformly distributed n random numbers in range (0, 1]
void cuda_curand_uniform_double(const int n, double *x, const uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif 