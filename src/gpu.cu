#include "gpu.h"
#include "enum.h"
#include "memory.h"
#include "utils.h"
#include "linalg_lib_wrapper.h"
#include <cassert>
#include <cuda.h>


__global__
    void gpu_sum_impl(const double* vec, int n, double* res) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        atomicAdd(res, vec[i]);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

__global__
void gpu_print_impl(const double* vec, int n) {
    for(int i = 0; i < n; i++) {
    printf("%f, ", vec[i]);
    }
    printf("\n");
}

//Col major
__global__
void gpu_print_mat_impl(const double* vec, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f, ", vec[j*m + i]);
        }
    printf("\n");
    }
}

void gpu_print(const double* vec, int n)  {
    printf("Printing %i gpu els: \n", n);
    dim3 blockDims(1,1,1);
    dim3 gridDims(1,1,1);
    gpuErrchk( cudaDeviceSynchronize() );

    gpu_print_impl<<<blockDims, gridDims>>>(vec, n);

    gpuErrchk( cudaDeviceSynchronize() );
}

void gpu_print_mat(const double* vec, int m, int n)  {
    printf("Printing %i gpu els %p: \n", m * n, vec);
    dim3 blockDims(1,1,1);
    dim3 gridDims(1,1,1);
    gpuErrchk( cudaDeviceSynchronize() );

    gpu_print_mat_impl<<<blockDims, gridDims>>>(vec, m, n);

    gpuErrchk( cudaDeviceSynchronize() );
}

double gpu_sum(const double* vec, int n)  {
    double sum;

#if DEBUG
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif

    double* sum_d = OUR_CALLOC(1, double, DEVICE_TYPE_DEVICE);

#if DEBUG
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif

    dim3 blockDims(256,1,1);
    dim3 gridDims(((n) + blockDims.x-1) / blockDims.x,1,1);

    //gpu_sum_impl<<<1,1>>>(vec, n, sum_d);
    gpu_sum_impl<<<gridDims,blockDims>>>(vec, n, sum_d);
#if DEBUG
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
#endif
    OUR_MEMCPY((void*) &sum, (void*) sum_d, sizeof(double), DEVICE_TYPE_HOST, DEVICE_TYPE_DEVICE);
#if DEBUG
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
#endif
    return sum;
}
void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (1) {
            double* x = NULL;
            assert((int) x[0]);
        assert(code);
        assert(-1);
        exit(code);
      }
   }
}

void cublasAssert(cublasStatus_t code, const char *file, int line)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"CUBLASassert: %i %s %d\n", code,  file, line);
      if (1) exit(code);
   }
}

#ifdef __cplusplus
}
#endif
