#ifndef __CANNON_H__
#define __CANNON_H__

#include <mpi.h>
#include "dev_type.h"

struct cannon_engine
{
    int  m, n, k;               // Size of A (m * k), B (k * n), C (m * n) matrices
    int  my_rank;               // Rank of this process in the Target MPI communicator
    int  rank_row;              // Row rank of this process in the 2D process grid
    int  rank_col;              // Column rank of this process in the 2D process grid
    int  np_dim;                // Number of processes on each dimension
    int  A_srow, A_scol;        // 1st row & col of A matrix block needed by this MPI process
    int  B_srow, B_scol;        // 1st row & col of B matrix block needed by this MPI process
    int  C_srow, C_scol;        // 1st row & col of C matrix block calculated by this MPI process
    int  A_nrow, A_ncol;        // Number of rows & cols of A matrix block needed by this MPI process
    int  B_nrow, B_ncol;        // Number of rows & cols of B matrix block needed by this MPI process
    int  C_nrow, C_ncol;        // Number of rows & cols of C matrix block calculated by this MPI process
    int  gemm_cycle;            // Number of P2P shift steps before a local GEMM compute
    int  max_A_blk_size;        // Maximum A block size (number of elements), == ceil(m/np_dim) * ceil(k/np_dim)
    int  max_B_blk_size;        // Maximum B block size (number of elements), == ceil(k/np_dim) * ceil(n/np_dim)
    int  max_C_blk_size;        // Maximum C block size (number of elements), == ceil(m/np_dim) * ceil(n/np_dim)
    int  alloc_workbuf;         // If work_buf is allocated by cannon_engine
    int  *m_displs;             // Size np_dim+1, partitioning displacements on the m dimension
    int  *n_displs;             // Size np_dim+1, partitioning displacements on the n dimension
    int  *k_displs;             // Size np_dim+1, partitioning displacements on the k dimension
    void *A_recv_h;             // Size ceil(m/np_dim) * ceil(k/np_dim), A block receive host buffer
    void *A_stack_h;            // Size ceil(m/np_dim) * ceil(k/np_dim) * gemm_cycle, stacked A blocks on host
    void *B_recv_h;             // Size ceil(k/np_dim) * ceil(n/np_dim), B block receive host buffer
    void *B_stack_h;            // Size ceil(k/np_dim) * ceil(n/np_dim) * gemm_cycle, stacked B blocks on host
    void *A_recv_d;             // Size ceil(m/np_dim) * ceil(k/np_dim), A block receive device buffer
    void *A_stack_d;            // Size ceil(m/np_dim) * ceil(k/np_dim) * gemm_cycle, stacked A blocks on device
    void *B_recv_d;             // Size ceil(k/np_dim) * ceil(n/np_dim), B block receive device buffer
    void *B_stack_d;            // Size ceil(k/np_dim) * ceil(n/np_dim) * gemm_cycle, stacked B blocks on device
    void *workbuf_h;            // Work buffer, all void* above with _h suffix are aliases to workbuf_h
    void *workbuf_d;            // Work buffer, all void* above with _d suffix are aliases to workbuf_d
    void *A_gemm_h;             // Size ceil(m/np_dim) * ceil(k/np_dim), A block GEMM host buffer for DEV_TYPE_CUDA
    void *B_gemm_h;             // Size ceil(k/np_dim) * ceil(n/np_dim), B block GEMM host buffer for DEV_TYPE_CUDE
    dev_type_t  dev_type;       // Data resident device type
    MPI_Comm    comm;           // Target MPI communicator

    // Statistic data
    double init_ms;             // Time (milliseconds) used in initialization
    double shift0_ms;           // Time (milliseconds) used in initial shift
    double lshift_ms;           // Time (milliseconds) used in loop shift
    double gemm_ms;             // Time (milliseconds) used in local DGEMM
    double exec_ms;             // Time (milliseconds) used in the whole Cannon algorithm execution
    double hd_trans_ms;         // Time (milliseconds) used in host-device data transfer
    int    n_exec;              // Number of Cannon algorithm execution
};
typedef struct cannon_engine  cannon_engine_s;
typedef struct cannon_engine* cannon_engine_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a cannon_engine for 2D Cannon matrix multiplication algorithm
// Input parameters:
//   m, n, k  : Size of A (m * k), B (k * n), C (m * n) matrices
//   comm     : Target MPI communicator
//   dev_type : Data resident device type
// Output parameters: 
//   *engine_        : An initialized cannon_engine
//   *workbuf_bytes  : Optional. If pointer is not NULL, the returning value is the size 
//                     of work buffer, and cannon_engine will not allocate work buffer.
//                     If pointer is NULL, cannon_engine will allocate and free work buffer.
void cannon_engine_init(
    const int m, const int n, const int k, MPI_Comm comm, 
    dev_type_t dev_type, cannon_engine_p *engine_, size_t *workbuf_bytes
);

// Attach an external work buffer for cannon_engine
// Input parameters:
//   engine   : Initialized cannon_engine_p
//   workbuf_h : Work buffer on host, size >= *workbuf_bytes returned by cannon_engine_init()
//   workbuf_d : Work buffer on device, size >= *workbuf_bytes returned by cannon_engine_init()
// Note:
//   1. workbuf_d can be NULL if dev_type == DEV_TYPE_HOST
//   2. workbuf_h can be NULL if dev_type == DEV_TYPE_CUDA_MPI_DIRECT
void cannon_engine_attach_workbuf(cannon_engine_p engine, void *workbuf_h, void *workbuf_d);

// Free a cannon_engine
void cannon_engine_free(cannon_engine_p *engine_);

// Compute C := alpha * A * B + beta * C using 2D Cannon matrix multiplication algorithm
// Input parameters:
//   engine : An initialized cannon_engine
//   alpha  : Scaling factor of A * B
//   beta   : Scaling factor of C
//   A_blk  : Size >= ceil(m/np_dim) * ceil(k/np_dim), a column-major A matrix block  
//            starting at (engine->m_displs[row_rank], engine->k_displs[col_rank])
//   B_blk  : Size >= ceil(k/np_dim) * ceil(n/np_dim), a column-major B matrix block 
//            starting at (engine->k_displs[row_rank], engine->n_displs[col_rank])
// Output parameters:
//   C_blk : Size depends on process rank, a column-major C matrix block starting at 
//           (engine->m_displs[row_rank], engine->n_displs[col_rank])
// Note: A_blk and B_blk will be overwritten on exit
void cannon_engine_exec(
    cannon_engine_p engine, const double alpha, const double beta, 
    double *A_blk, double *B_blk, double *C_blk
);

// Reset the statistic data of a cannon_engine (not a collective call)
void cannon_engine_reset_stat(cannon_engine_p engine);

// Print the statistic data of a cannon_engine (not a collective call)
void cannon_engine_print_stat(cannon_engine_p engine);

#ifdef __cplusplus
}
#endif

#endif
