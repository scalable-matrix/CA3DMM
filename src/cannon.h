#ifndef __CANNON_H__
#define __CANNON_H__

#include <mpi.h>

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
    int  max_A_blk_size;        // Maximum A block size (number of elements)
    int  max_B_blk_size;        // Maximum B block size (number of elements)
    int  max_C_blk_size;        // Maximum C block size (number of elements)
    int  alloc_workbuf;         // If work_buf is allocated by cannon_engine
    int  *m_displs;             // Size np_dim+1, partitioning displacements on the m dimension
    int  *n_displs;             // Size np_dim+1, partitioning displacements on the n dimension
    int  *k_displs;             // Size np_dim+1, partitioning displacements on the k dimension
    void *A_gemm;               // Size (m/np_dim + 1) * (k/np_dim + 1), A block GEMM buffer
    void *A_recv;               // Size (m/np_dim + 1) * (k/np_dim + 1), A block receive buffer
    void *A_stack;              // Size (m/np_dim + 1) * (k/np_dim + 1) * gemm_cycle, stacked A blocks
    void *B_gemm;               // Size (k/np_dim + 1) * (n/np_dim + 1), B block GEMM buffer
    void *B_recv;               // Size (k/np_dim + 1) * (n/np_dim + 1), B block receive buffer
    void *B_stack;              // Size (k/np_dim + 1) * (n/np_dim + 1) * gemm_cycle, stacked B blocks
    void *C_buff;               // Size (m/np_dim + 1) * (n/np_dim + 1), C block result buffer
    void *work_buf;             // Work buffer, all void * above are alias to work_buf
    MPI_Comm    comm;           // Target MPI communicator
    MPI_Request req_send_A[2];  // MPI requests for sending   A to   left  rank
    MPI_Request req_send_B[2];  // MPI requests for sending   B to   upper rank
    MPI_Request req_recv_A[2];  // MPI requests for receiving A from right rank
    MPI_Request req_recv_B[2];  // MPI requests for receiving B from lower rank

    // Statistic data
    double init_ms;             // Time (milliseconds) used in initialization
    double shift0_ms;           // Time (milliseconds) used in initial shift
    double lshift_ms;           // Time (milliseconds) used in loop shift
    double gemm_ms;             // Time (milliseconds) used in local DGEMM
    double exec_ms;             // Time (milliseconds) used in the whole Cannon algorithm execution
    int    n_exec;              // Number of Cannon algorithm execution
};
typedef struct cannon_engine  cannon_engine_s;
typedef struct cannon_engine* cannon_engine_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a cannon_engine for 2D Cannon matrix multiplication algorithm
// Input parameters: see the cannon_engine structure above
// Output parameters: 
//   *engine_        : An initialized cannon_engine
//   *workbuf_bytes  : Optional. If pointer is not NULL, the returning value is the size 
//                     of work buffer, and cannon_engine will not allocate work buffer.
//                     If pointer is NULL, cannon_engine will allocate and free work buffer.
void cannon_engine_init(
    const int m, const int n, const int k, 
    MPI_Comm comm, cannon_engine_p *engine_, size_t *workbuf_bytes
);

// Attach an external work buffer for cannon_engine
// Input parameters:
//   engine   : Initialized cannon_engine_p
//   work_buf : Work buffer, size >= *workbuf_bytes returned by cannon_engine_init()
void cannon_engine_attach_workbuf(cannon_engine_p engine, void *work_buf);

// Free a cannon_engine
void cannon_engine_free(cannon_engine_p *engine_);

// Compute C := alpha * A * B + beta * C using 2D Cannon matrix multiplication algorithm
// Input parameters:
//   alpha  : Scaling factor of A * B
//   A_blk  : Size depends on process rank, a column-major A matrix block starting at 
//            (engine->m_displs[row_rank], engine->k_displs[col_rank])
//   B_blk  : Size depends on process rank, a column-major B matrix block starting at 
//            (engine->k_displs[row_rank], engine->n_displs[col_rank])
//   beta   : Scaling factor of C
//   engine : An initialized cannon_engine
// Output parameters:
//   C_blk : Size depends on process rank, a column-major C matrix block starting at 
//           (engine->m_displs[row_rank], engine->n_displs[col_rank])
void cannon_engine_exec(
    const double alpha, const double *A_blk, const double *B_blk, 
    const double beta, double *C_blk, cannon_engine_p engine
);

// Reset the statistic data of a cannon_engine (not a collective call)
void cannon_engine_reset_stat(cannon_engine_p engine);

// Print the statistic data of a cannon_engine (not a collective call)
void cannon_engine_print_stat(cannon_engine_p engine);

#ifdef __cplusplus
}
#endif

#endif
