#include <assert.h>
#include <unistd.h>

#include <cuda.h>
#include <cublasXt.h>
#include <cuda_runtime.h>

#include "gpu.h"
#include "memory.h"
#include "linalg_lib_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

void local_AB_gpu(linalg_handle_t handle, int m,  int n, int k, double alpha,  const double* A, int lda,
     const double* B, int ldb, double beta, double* C, int ldc, 
     device_type storage_device) {

    //printf("m: %i, n: %i, k: %i, lda: %i, ldb: %i, ldc: %i\n", m,n,k,lda,ldb,ldc);

    assert((m == lda) && (k == ldb) && (m == ldc)  && "GPU Impl currently assumes simple leading dimensions, A in col major");
    double* a_d_t;
    double* b_d_t;
    double* c_d_t;
    if(storage_device != DEVICE_TYPE_DEVICE) {
        printf("STORAGE IS NOT DEVICE, CPYING\n");
        a_d_t = OUR_MALLOC(m*k, double,  DEVICE_TYPE_DEVICE);
        b_d_t = OUR_MALLOC(n*k, double,  DEVICE_TYPE_DEVICE);
        c_d_t = OUR_MALLOC(m*n, double,  DEVICE_TYPE_DEVICE);
        OUR_MEMCPY((void*) a_d_t, A, m*k*sizeof(double), DEVICE_TYPE_DEVICE, storage_device);
        OUR_MEMCPY((void*) b_d_t, B, n*k*sizeof(double), DEVICE_TYPE_DEVICE, storage_device);
        OUR_MEMCPY((void*) c_d_t, C, n*m*sizeof(double), DEVICE_TYPE_DEVICE, storage_device);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    } 
    const double* a_d = (storage_device != DEVICE_TYPE_DEVICE)? a_d_t:A;
    const double* b_d = (storage_device != DEVICE_TYPE_DEVICE)? b_d_t:B;
    double* c_d = (storage_device != DEVICE_TYPE_DEVICE)? c_d_t:C;

    //printf("A: %p \n", a_d);
    //gpu_print_mat(a_d, 1, 1);

    //printf("B: %p \n", b_d);
    //gpu_print_mat(b_d, k, n);

    cublasErrchk(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a_d, lda, b_d, ldb, &beta, c_d, ldc));

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );



    // gpu_print_mat(c_d, m, n);
    //usleep(50000);

    if(storage_device != DEVICE_TYPE_DEVICE) {
        OUR_FREE(a_d_t, DEVICE_TYPE_DEVICE);
        OUR_FREE(b_d_t, DEVICE_TYPE_DEVICE);
        OUR_MEMCPY(C, c_d, n*m*sizeof(double), storage_device, DEVICE_TYPE_DEVICE);
        OUR_FREE(c_d_t, DEVICE_TYPE_DEVICE);
    }

    //Is this needed?
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

void init_linalg_handle_gpu(linalg_handle_t* handle) {
  printf("CREATE HANDL\n");
  cublasErrchk(cublasCreate(handle));
}


void gpu_transpose(
    const int A_nrow, const int A_ncol, const double *A, const int ldA,
    double *A_trans, const int ldAT, linalg_handle_t handle, device_type dev) {
    //printf("A_nrow: %i, A_ncol: %i, A: %p, ldA: %i, A_trans: %p, ldAT: %i\n",
    A_nrow, A_ncol, A, ldA, A_trans, ldAT);

    double alpha = 1;
    double beta = 0;
    int opA_nrow = A_nrow;
    cublasErrchk(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, A_nrow, A_ncol, &alpha, A, ldA, &beta, A,  ldAT, A_trans, ldAT));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    }


#ifdef __cplusplus
}
#endif
