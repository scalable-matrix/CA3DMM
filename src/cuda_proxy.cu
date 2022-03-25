#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

#include "cuda_utils.h"
#include "cuda_proxy.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

static int get_mpi_local_rank_env()
{
    int local_rank = -1;
    char *env_p;

    // MPICH
    env_p = getenv("MPI_LOCALRANKID");
    if (env_p != NULL) return atoi(env_p);

    // MVAPICH2
    env_p = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (env_p != NULL) return atoi(env_p);

    // OpenMPI
    env_p = getenv("OMPI_COMM_WORLD_NODE_RANK");
    if (env_p != NULL) return atoi(env_p);

    // SLURM or PBS/Torque
    env_p = getenv("SLURM_LOCALID");
    if (env_p != NULL) return atoi(env_p);

    env_p = getenv("PBS_O_VNODENUM");
    if (env_p != NULL) return atoi(env_p);

    return local_rank;
}

void select_cuda_device_by_mpi_local_rank()
{
    int local_rank = get_mpi_local_rank_env();
    if (local_rank == -1) local_rank = 0;
    int num_gpu, dev_id;
    CUDA_RUNTIME_CHECK( cudaGetDeviceCount(&num_gpu) );
    dev_id = local_rank % num_gpu;
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void cuda_set_rt_dev_id(const int dev_id)
{
    CUDA_RUNTIME_CHECK( cudaSetDevice(dev_id) );
}

void cuda_memcpy_h2d(const void *hptr, void *dptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr, hptr, bytes, cudaMemcpyHostToDevice) );
}

void cuda_memcpy_d2h(const void *dptr, void *hptr, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(hptr, dptr, bytes, cudaMemcpyDeviceToHost) );
}

void cuda_memcpy_d2d(const void *dptr_src, void *dptr_dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dptr_dst, dptr_src, bytes, cudaMemcpyDeviceToDevice) );
}

void cuda_memcpy_auto(const void *src, void *dst, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy(dst, src, bytes, cudaMemcpyDefault) );
}

void cuda_memcpy_2d_h2d(const void *hptr, const size_t ldh, void *dptr, const size_t ldd, const size_t nrow, const size_t col_bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(dptr, ldd, hptr, ldh, col_bytes, nrow, cudaMemcpyHostToDevice) );
}

void cuda_memcpy_2d_d2h(const void *dptr, const size_t ldd, void *hptr, const size_t ldh, const size_t nrow, const size_t col_bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(hptr, ldh, dptr, ldd, col_bytes, nrow, cudaMemcpyDeviceToHost) );
}

void cuda_memcpy_2d_d2d(const void *dptr_src, const size_t lds, void *dptr_dst, const size_t ldd, const size_t nrow, const size_t col_bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(dptr_dst, ldd, dptr_src, lds, col_bytes, nrow, cudaMemcpyDeviceToDevice) );
}

void cuda_memcpy_2d_auto(const void *src, const size_t lds, void *dst, const size_t ldd, const size_t nrow, const size_t col_bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemcpy2D(dst, ldd, src, lds, col_bytes, nrow, cudaMemcpyDefault) );
}

void cuda_malloc_dev(void **dptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMalloc(dptr_, bytes) );
}

void cuda_malloc_host(void **hptr_, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMallocHost(hptr_, bytes) );
}

void cuda_memset_dev(void *dptr, const int value, const size_t bytes)
{
    CUDA_RUNTIME_CHECK( cudaMemset(dptr, value, bytes) );    
}

void cuda_free_dev(void *dptr)
{
    CUDA_RUNTIME_CHECK( cudaFree(dptr) );
}

void cuda_free_host(void *hptr)
{
    CUDA_RUNTIME_CHECK( cudaFreeHost(hptr) );
}

void cuda_device_sync()
{
    CUDA_RUNTIME_CHECK( cudaDeviceSynchronize() );
}

void cuda_stream_sync(void *stream_p)
{
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(*((cudaStream_t *) stream_p)) );
}

__global__ void copy_matrix_block_kernel(
    const int nrow, const int ncol, const float *src, const int lds,
    float *dst, const int ldd
)
{
    const float *src_ptr = src + lds * blockIdx.x;
    float *dst_ptr = dst + ldd * blockIdx.x;
    for (int i = threadIdx.x; i < ncol; i += blockDim.x) dst_ptr[i] = src_ptr[i];
    __syncthreads();
}

void cuda_copy_matrix_block(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd
)
{
    int k = dt_size / 4;
    copy_matrix_block_kernel<<<nrow, 64>>>(nrow, ncol * k, (float *) src, lds * k, (float *) dst, ldd * k);
    CUDA_RUNTIME_CHECK( cudaPeekAtLastError() );
}

static int            cublas_initialzed = 0;
static cublasHandle_t cublas_handle;
static cublasStatus_t cublas_stat;
static cudaStream_t   cublas_stream;

static int init_cublas()
{
    int ret = 1;
    if (cublas_initialzed == 0)
    {
        cublas_stat = cublasCreate(&cublas_handle);
        if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "[ERROR] %s, %d: CUBLAS initialization failed\n", __FILE__, __LINE__);
            fflush(stderr);
            ret = 0;
        } else {
            cublas_initialzed = 1;
            ret = 1;
        }
        if (ret == 1) cublas_stat = cublasGetStream(cublas_handle, &cublas_stream);
    }
    return ret;
}

void cuda_cublas_dcopy(const int n, const double *x, const int incx, double *y, const int incy)
{
    if (init_cublas() == 0) return;

    cublas_stat = cublasDcopy(cublas_handle, n, x, incx, y, incy);
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDcopy failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since x and y may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
}

void cuda_cublas_dscal(const int n, const double alpha, double *x, const int incx)
{
    if (init_cublas() == 0) return;

    cublas_stat = cublasDscal(cublas_handle, n, &alpha, x, incx);
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDscal failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since x may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
}

double cuda_cublas_ddot(const int n, const double *x, const int incx, const double *y, const int incy)
{
    if (init_cublas() == 0) return 0.0;

    double res;
    cublas_stat = cublasDdot(cublas_handle, n, x, incx, y, incy, &res);
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDdot failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since x may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
    return res;
}

void cuda_cublas_dgemm(
    cublas_enum_t transA, cublas_enum_t transB,
    const int m, const int n, const int k, const double alpha, 
    const double *A, const int ldA, const double *B, const int ldB,
    const double beta, double *C, const int ldC
)
{
    if (init_cublas() == 0) return;

    cublasOperation_t transA_ = (transA == CublasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_ = (transB == CublasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublas_stat = cublasDgemm(
        cublas_handle, transA_, transB_, m, n, k, 
        &alpha, A, ldA, B, ldB, &beta, C, ldC
    );
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDgemm failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since a and b may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
}

void cuda_cublas_dgeam(
    cublas_enum_t transA, cublas_enum_t transB,
    const int m, const int n, 
    const double alpha, const double *A, const int ldA,
    const double beta,  const double *B, const int ldB,
    double *C, const int ldC
)
{
    if (init_cublas() == 0) return;

    cublasOperation_t transA_ = (transA == CublasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB_ = (transB == CublasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublas_stat = cublasDgeam(
        cublas_handle, transA_, transB_, m, n,
        &alpha, A, ldA, &beta, B, ldB, C, ldC
    );
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDgeam failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since a and b may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
}

void cuda_cublas_dtrsm(
    cublas_enum_t side, cublas_enum_t uplo,
    cublas_enum_t trans, cublas_enum_t diag,
    const int m, const int n, const double alpha,
    const double *A, const int ldA, double *B, const int ldB
)
{
    if (init_cublas() == 0) return;

    cublasSideMode_t  side_  = (side  == CublasLeft)  ? CUBLAS_SIDE_LEFT       : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t  uplo_  = (uplo  == CublasUpper) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t trans_ = (trans == CublasTrans) ? CUBLAS_OP_T            : CUBLAS_OP_N;
    cublasDiagType_t  diag_  = (diag  == CublasUnit)  ? CUBLAS_DIAG_UNIT       : CUBLAS_DIAG_NON_UNIT;
    cublas_stat = cublasDtrsm(
        cublas_handle, side_, uplo_, trans_, diag_,
        m, n, &alpha, A, ldA, B, ldB
    );
    if (cublas_stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cublasDtrsm failed\n", __FILE__, __LINE__);
    // Wait the compute to finish, since a and b may be rewritten on exist
    CUDA_RUNTIME_CHECK( cudaStreamSynchronize(cublas_stream) );
}

static int    cusolver_initialzed    = 0;
static size_t cusolver_workbuf_bytes = 0;
static int    *cusolver_retval       = NULL;
static void   *cusolver_workbuf      = NULL;
static cusolverDnHandle_t cusolver_dn_handle;
static cusolverStatus_t   cusolver_stat;
static cudaStream_t       cusolver_stream;

static int init_cusolver()
{
    int ret = 1;
    if (cusolver_initialzed == 0)
    {
        cusolver_stat = cusolverDnCreate(&cusolver_dn_handle);
        if (cusolver_stat != CUSOLVER_STATUS_SUCCESS)
        {
            fprintf(stderr, "[ERROR] %s, %d: CUSOLVE initialization failed\n", __FILE__, __LINE__);
            fflush(stderr);
            ret = 0;
        }
        cusolver_initialzed = 1;
        ret = 1;
        cusolver_stat = cusolverDnGetStream(cusolver_dn_handle, &cusolver_stream);
        cuda_malloc_dev((void **) &cusolver_retval, sizeof(int));
        cusolver_workbuf_bytes = 0;
    }
    return ret;
}

static void realloc_cusolver_workbuf(const int req_bytes)
{
    if (req_bytes <= cusolver_workbuf_bytes) return;
    cuda_free_dev(cusolver_workbuf);
    cusolver_workbuf_bytes = 0;
    cuda_malloc_dev((void **) &cusolver_workbuf, req_bytes);
    if (cusolver_workbuf != NULL) cusolver_workbuf_bytes = req_bytes;
}

int cuda_cusolver_dpotrf(const char uplo, const int n, double *A, const int ldA)
{
    if (init_cusolver() == 0) return -1;

    cublasFillMode_t uplo_ = (uplo == 'U') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    int lwork;
    cusolver_stat = cusolverDnDpotrf_bufferSize(cusolver_dn_handle, uplo_, n, A, ldA, &lwork);
    if (cusolver_stat != CUSOLVER_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDpotrf_bufferSize failed\n", __FILE__, __LINE__);
    realloc_cusolver_workbuf(sizeof(double) * lwork);

    cusolver_stat = cusolverDnDpotrf(
        cusolver_dn_handle, uplo_, n, A, ldA,  
        (double *) cusolver_workbuf, lwork, cusolver_retval
    );
    if (cusolver_stat != CUSOLVER_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDpotrf failed\n", __FILE__, __LINE__);

    int info = 0;
    cuda_memcpy_d2h(cusolver_retval, &info, sizeof(int));
    if (info != 0)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDpotrf returned %d\n", __FILE__, __LINE__, info);
    return info;
}

int cuda_cusolver_dsyevd(const char jobz, const char uplo, const int n, double *A, const int ldA, double *W)
{
    if (init_cusolver() == 0) return -1;

    cusolverEigMode_t jobz_ = (jobz == 'V') ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t  uplo_ = (uplo == 'U') ? CUBLAS_FILL_MODE_UPPER   : CUBLAS_FILL_MODE_LOWER;
    
    int lwork;
    cusolver_stat = cusolverDnDsyevd_bufferSize(cusolver_dn_handle, jobz_, uplo_, n, A, ldA, W, &lwork);
    if (cusolver_stat != CUSOLVER_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDsyevd_bufferSize failed\n", __FILE__, __LINE__);
    realloc_cusolver_workbuf(sizeof(double) * lwork);

    cusolver_stat = cusolverDnDsyevd(
        cusolver_dn_handle, jobz_, uplo_, n, A, ldA, W, 
        (double *) cusolver_workbuf, lwork, cusolver_retval
    );
    if (cusolver_stat != CUSOLVER_STATUS_SUCCESS)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDsyevd failed\n", __FILE__, __LINE__);

    int info = 0;
    cuda_memcpy_d2h(cusolver_retval, &info, sizeof(int));
    if (info != 0)
        fprintf(stderr, "[ERROR] %s, %d: cusolverDnDsyevd returned %d\n", __FILE__, __LINE__, info);
    return info;
}

__global__ void daxpb_kernel(const int n, const double alpha, double *x, const double beta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_thread = gridDim.x * blockDim.x;
    for (; tid < n; tid += n_thread)
        x[tid] = alpha * x[tid] + beta;
    __syncthreads();
}

void cuda_daxpb(const int n, const double alpha, double *x, const double beta)
{
    int blk_size = 64;
    int num_blk  = (n + blk_size - 1) / blk_size / 4;
    if (num_blk < 1) num_blk = 1;
    daxpb_kernel<<<num_blk, blk_size>>>(n, alpha, x, beta);
    CUDA_RUNTIME_CHECK( cudaPeekAtLastError() );
}

__global__ void daxpby_kernel(const int n, const double alpha, const double *x, const double beta, double *y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_thread = gridDim.x * blockDim.x;
    for (; tid < n; tid += n_thread)
        y[tid] = alpha * x[tid] + beta * y[tid];
    __syncthreads();
}

void cuda_daxpby(const int n, const double alpha, const double *x, const double beta, double *y)
{
    int blk_size = 64;
    int num_blk  = (n + blk_size - 1) / blk_size / 4;
    if (num_blk < 1) num_blk = 1;
    daxpby_kernel<<<num_blk, blk_size>>>(n, alpha, x, beta, y);
    CUDA_RUNTIME_CHECK( cudaPeekAtLastError() );
}

void cuda_curand_uniform_double(const int n, double *x, const uint64_t seed)
{
    if (n < 1) return;

    curandGenerator_t gen;
    CURAND_CHECK( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CHECK( curandSetPseudoRandomGeneratorSeed(gen, seed) );
    CURAND_CHECK( curandGenerateUniformDouble(gen, x, n) );
    CURAND_CHECK( curandDestroyGenerator(gen) );
}