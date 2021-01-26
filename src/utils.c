#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include <time.h>
#include <stdlib.h>
#include "utils.h"

#include "memory.h"

#if USE_GPU
#include "gpu.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

double monotonic_seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double sum_vec(const double* vec, int n, device_type dev) {
#if USE_GPU
    if(dev == DEVICE_TYPE_DEVICE) {
        return gpu_sum(vec, n);
    } else {
#endif
        double sum = 0;
        for(int i =0; i < n; i++) {
            sum += vec[i];
        }
        return sum;
#if USE_GPU
    }
#endif
}


void print_vec(const double* vec, int n, device_type dev) {
#if USE_GPU
    printf("Printing\n");
    if(dev == DEVICE_TYPE_DEVICE) {
    printf("Printing GPU\n");
        gpu_print(vec, n);
    } else {
#endif
    printf("Printing CPU\n");
        for(int i = 0; i < n; i++) {
            printf("%f, ", vec[i]);
        }
        printf("\n");
#if USE_GPU
    }
#endif
}

void scale_vector(int n, double* x, double alpha, device_type dev) {
#if USE_GPU
    if(dev == DEVICE_TYPE_DEVICE) {
        scale_vector_gpu(n,x,alpha);
    } else {
#endif
        for (int ix = 0; ix < n; ix++)
        {
            x[ix] *= alpha;
        }
#if USE_GPU
    }
#endif
}

void chebyshev_recurrence_relation(int ntotal, double * x, double* y, double* ynew, double vscale, double vscale2, device_type dev) {
#if USE_GPU
    if (dev == DEVICE_TYPE_DEVICE) {
        chebyshev_recurrence_relation_gpu(ntotal,x,y,ynew,vscale, vscale2);
    } else {
#endif
        // Three-term recurrence relation
        for (int ix = 0; ix < ntotal; ix++)
        {
            ynew[ix] *= vscale;
            ynew[ix] -= vscale2 * x[ix];
            x[ix] = y[ix];
            y[ix] = ynew[ix];
        }
#if USE_GPU
    }
#endif
}

void print_mat(const double* vec, int m, int n, device_type dev) {
#if USE_GPU
    printf("(%i x %i): ", m, n);
    if(dev == DEVICE_TYPE_DEVICE) {
        gpu_print_mat(vec, m, n);
    } else {
#endif
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%f, ", vec[j*m + i]);
        }
    printf("\n");
    }
#if USE_GPU
    }
#endif
}



#ifdef __cplusplus
}
#endif
