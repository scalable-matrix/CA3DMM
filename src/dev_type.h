#ifndef __DEV_TYPE_H__
#define __DEV_TYPE_H__

#include <stdint.h>
#include "utils.h"

#include <mpi.h>

typedef enum 
{
    DEV_TYPE_HOST = 0,          // CPU  memory
    DEV_TYPE_CUDA,              // CUDA memory
    DEV_TYPE_CUDA_MPI_DIRECT    // CUDA memory, MPI is CUDA-aware
} dev_type_t;

#ifdef __cplusplus
extern "C" {
#endif

// Check if a device type is valid
int is_dev_type_valid(dev_type_t dev_type);

// Allocate memory on specified device 
void *dev_type_malloc(size_t bytes, dev_type_t dev_type);

// Free memory on specified device
void dev_type_free(void *mem, dev_type_t dev_type);

// Re-allocate memory on specified device if required size larger than current size
void dev_type_realloc(size_t *curr_bytes, size_t req_bytes, dev_type_t dev_type, void **mem);

// Memset on specified device
void dev_type_memset(void *mem, int value, size_t bytes, dev_type_t dev_type);

// Memcpy on specified device
void dev_type_memcpy(
    void *dst, const void *src, size_t bytes, 
    dev_type_t dst_dev_type, dev_type_t src_dev_type
);

// Copy a row-major matrix block to another row-major matrix
// Input parameters:
//   dt_size  : Size of matrix element data type, in bytes
//   nrow     : Number of rows to be copied
//   ncol     : Number of columns to be copied
//   src      : Size >= lds * nrow, source matrix
//   lds      : Leading dimension of src, >= ncol
//   ldd      : Leading dimension of dst, >= ncol
//   dev_type : Memory device type
// Output parameter:
//   dst : Size >= ldd * nrow, destination matrix
// Note: if dev_type == TYPE_CUDA or TYPE_CUDA_MPI_DIRECT, dt_size should be 4 or 8
void dev_type_copy_mat_blk(
    size_t dt_size, const int nrow, const int ncol,
    const void *src, const int lds, void *dst, const int ldd,
    dev_type_t dev_type
);

// Perform an in-place MPI allreduce
double dev_type_allreduce_inplace(
    MPI_Comm comm, void *mem_h, void *mem_d, size_t unit_bytes, 
    int n_elem, MPI_Datatype mpi_dtype, MPI_Op mpi_op, dev_type_t dev_type
);

// Perform DGEMM for column-major matrices
void dev_type_dgemm_cm(
    const int transA, const int transB, const int m, const int n, const int k,
    const double alpha, const double *A, const int ldA, const double *B, const int ldB,
    const double beta, double *C, const int ldC, dev_type_t dev_type
);

#ifdef __cplusplus
}
#endif

#define MALLOC_ATTACH_WORKBUF(attach_func, free_func, engine, dev_type, workbuf_bytes, workbuf_h, workbuf_d) \
    do {                                                                                \
        workbuf_h = NULL;                                                               \
        workbuf_d = NULL;                                                               \
        if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))                 \
        {                                                                               \
            workbuf_h = dev_type_malloc(workbuf_bytes, DEV_TYPE_HOST);                  \
            if (workbuf_h == NULL)                                                      \
            {                                                                           \
                ERROR_PRINTF("Allocate host workbuf failed\n");                         \
                free_func(&engine);                                                     \
                break;                                                                  \
            }                                                                           \
        }                                                                               \
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))      \
        {                                                                               \
            workbuf_d = dev_type_malloc(workbuf_bytes, DEV_TYPE_CUDA);                  \
            if (workbuf_d == NULL)                                                      \
            {                                                                           \
                ERROR_PRINTF("Allocate CUDA workbuf failed\n");                         \
                free_func(&engine);                                                     \
                break;                                                                  \
            }                                                                           \
        }                                                                               \
        attach_func(engine, workbuf_h, workbuf_d);                                      \
    } while (0)

#endif
