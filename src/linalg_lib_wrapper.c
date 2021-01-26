#include "linalg_lib_wrapper.h"
#include "enum.h"

#ifdef USE_GPU
#include "linalg_gpu.h"
#endif

#include "linalg_cpu.h"

// Right now, assumes A,B,C all on CPU, will transfer to GPU if needed for GPU
void local_AB(linalg_handle_t handle, int m,  int n, int k, double alpha,  const double* A, int lda,
     const double* B, int ldb, double beta, double* C, int ldc,
     device_type storage_device, device_type compute_device) {
#if USE_GPU
if(compute_device == DEVICE_TYPE_DEVICE) {
  local_AB_gpu(handle, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, storage_device);
  } else {
#endif
  local_AB_cpu(handle, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#if USE_GPU
  }
#endif
}

void init_linalg_handle(linalg_handle_t* handle, device_type compute_device) {
#if USE_GPU
if(compute_device == DEVICE_TYPE_DEVICE) {
  init_linalg_handle_gpu(handle);
  }
#endif
}


