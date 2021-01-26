#include "linalg_lib_wrapper.h"
void local_AB_cpu(linalg_handle_t handle, int m,  int n, int k, double alpha,  const double* A, int lda,
     const double* B, int ldb, double beta, double* C, int ldc) {
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
            alpha, A, lda, B, ldb, beta, C, ldc 
        );
}

