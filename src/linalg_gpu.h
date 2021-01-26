#include "linalg_lib_wrapper.h"

void local_AB_gpu(linalg_handle_t handle, int m,  int n, int k, double alpha,  const double* A, int lda,
     const double* B, int ldb, double beta, double* C, int ldc, device_type storage_device);

void init_linalg_handle_gpu(linalg_handle_t* handle);
