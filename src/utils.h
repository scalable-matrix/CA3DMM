#ifndef UTILS_H

#include "enum.h"

#ifdef __cplusplus
extern "C" {
#endif

double monotonic_seconds();
void scale_vector(int n, double* x, double alpha, device_type dev);
void chebyshev_recurrence_relation(int ntotal, double * x, double* y, double* ynew, double vscale, double vscale2, device_type dev);
void print_mat(const double* vec, int m, int n, device_type dev);
double sum_vec(const double* vec, int n, device_type dev);

#ifdef __cplusplus
}
#endif

#endif
