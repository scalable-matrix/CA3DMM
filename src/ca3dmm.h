#ifndef __CA3DMM_H__
#define __CA3DMM_H__

#include <mpi.h>
#include "mat_redist.h"
#include "cannon.h"
#include "partition.h"
#include "linalg_lib_wrapper.h"

struct ca3dmm_engine
{
    int  m, n, k;                   // Size of matrix op(A) (m * k), op(B) (k * n), and C (m * n), op can be transpose or no-transpose
    int  mp, np, kp, rp;            // CA3DMM process grid size on m, n, and k dimensions, and idle process
    int  my_rank;                   // MPI rank of this process
    int  n_proc;                    // Total number of MPI processes in the global communicator
    int  is_active;                 // If this MPI process is active in computation
    int  rank_m, rank_n, rank_k;    // Coordinate of this process in the CA3DMM process grid
    int  trans_A, trans_B;          // If A / B matrix need to be transposed
    int  task_m_num, task_m_id;     // Total number and index of m dimension task (task_m_num and task_n_num cannot both > 1)
    int  task_n_num, task_n_id;     // Total number and index of n dimension task (task_m_id  and task_n_id  cannot both > 0)
    int  task_k_num, task_k_id;     // Total number and index of k dimension task
    int  is_BTB;                    // If this ca3dmm_engine is to compute B^T * B
    int  use_ag, use_rsb;           // If we can use MPI_Allgather/MPI_Reduce_scatter_block instead of MPI_Allgatherv/MPI_Reduce_scatter
    int  A_rd_srow,   A_rd_scol;    // 1st row & col of op(A) matrix block needed by this MPI process in redistribution
    int  B_rd_srow,   B_rd_scol;    // 1st row & col of op(B) matrix block needed by this MPI process in redistribution
    int  A_2dmm_srow, A_2dmm_scol;  // 1st row & col of op(A) matrix block needed by this MPI process in 2D matmul
    int  B_2dmm_srow, B_2dmm_scol;  // 1st row & col of op(B) matrix block needed by this MPI process in 2D matmul
    int  C_2dmm_srow, C_2dmm_scol;  // 1st row & col of C matrix block calculated by this MPI process in 2D matmul
    int  C_out_srow,  C_out_scol;   // 1st row & col of output C matrix block stored on this MPI process
    int  A_rd_nrow,   A_rd_ncol;    // Number of rows & cols of op(A) matrix block needed by this MPI process in redistribution 
    int  B_rd_nrow,   B_rd_ncol;    // Number of rows & cols of op(B) matrix block needed by this MPI process in redistribution
    int  A_2dmm_nrow, A_2dmm_ncol;  // Number of rows & cols of op(A) matrix block needed by this MPI process in 2D matmul
    int  B_2dmm_nrow, B_2dmm_ncol;  // Number of rows & cols of op(B) matrix block needed by this MPI process in 2D matmul
    int  C_2dmm_nrow, C_2dmm_ncol;  // Number of rows & cols of C matrix block calculated by this MPI process in 2D matmul
    int  C_out_nrow,  C_out_ncol;   // Number of rows & cols of output C matrix block owned by this MPI process
    int  alloc_workbuf;             // If work_buf is allocated by cannon_engine
    int  *AB_agv_recvcnts;          // Size unknown, recvcounts array used in MPI_Allgatherv after redistribution for A or B matrix
    int  *AB_agv_displs;            // Size unknown, displs array used in MPI_Allgatherv after redistribution for A or B matrix
    int  *C_rs_recvcnts;            // Size task_k_num, output C matrix block reduce-scatter receive count array
    void *A_rd_recv;                // Size A_rd_nrow   * A_rd_ncol,   op(A) matrix block received in redistribution
    void *A_trans;                  // Size A_2dmm_nrow * A_2dmm_ncol, A_2dmm transpose buffer
    void *A_2dmm;                   // Size A_2dmm_nrow * A_2dmm_ncol, initial op(A) matrix block required in 2D matmul
    void *B_rd_recv;                // Size B_rd_nrow   * B_rd_ncol,   op(A) matrix block received in redistribution
    void *B_trans;                  // Size B_2dmm_nrow * B_2dmm_ncol, B_2dmm transpose buffer
    void *B_2dmm;                   // Size B_2dmm_nrow * B_2dmm_ncol, initial op(B) matrix block required in 2D matmul
    void *C_2dmm;                   // Size C_2dmm_nrow * C_2dmm_ncol, 2D matmul result C matrix block
    void *C_out;                    // Size C_out_nrow  * C_out_ncol,  output C matrix block
    void *work_buf;                 // Work buffer, all void * above are alias to work_buf
    size_t   rdA_workbuf_bytes;     // redist_A work buffer size in bytes
    size_t   rdB_workbuf_bytes;     // redist_B work buffer size in bytes
    size_t   rdC_workbuf_bytes;     // redist_C work buffer size in bytes
    size_t   cannon_workbuf_bytes;  // cannon_engine work buffer size in bytes
    size_t   self_workbuf_bytes;    // Self work buffer size in bytes
    MPI_Comm comm_AB_agv;           // Communicator for A or B matrix block MPI_Allgatherv
    MPI_Comm comm_C_rs;             // Communicator for C matrix reduce-scatter
    MPI_Comm comm_2dmm;             // Communicator for 2D matmul in each k_task
    mat_redist_engine_p redist_A;   // Redistribution of A matrix from its initial layout to CA3DMM required layout
    mat_redist_engine_p redist_B;   // Redistribution of B matrix from its initial layout to CA3DMM required layout
    mat_redist_engine_p redist_C;   // Redistribution of C matrix from CA3DMM output layout to required layout
    cannon_engine_p cannon_engine;  // cannon_engine for 2D matmul

    // Statistic data
    int    print_timing;            // If rank 0 should print timing in each ca3dmm_engine_exec
    double init_ms;                 // Time (milliseconds) used in initialization
    double redist_ms;               // Time (milliseconds) used in redistribution of A and B
    double agvAB_ms;                // Time (milliseconds) used in MPI_Allgatherv of A or B
    double cannon_ms;               // Time (milliseconds) used in 2D matmul
    double reduce_ms;               // Time (milliseconds) used in k dimension reduction
    double exec_ms;                 // Time (milliseconds) used in the whole CA3DMM algorithm execution
    int    n_exec;                  // Number of CA3DMM algorithm execution
};
typedef struct ca3dmm_engine  ca3dmm_engine_s;
typedef struct ca3dmm_engine* ca3dmm_engine_p;

#ifdef __cplusplus
extern "C" {
#endif

// Initialize a camm3d_engine structure for C := op(A) * op(B), op can be transpose or no-transpose
// Input parameters:
//   m, n, k         : Size of matrix op(A) (m * k), op(B) (k * n), and C (m * n)
//   trans_{A, B}    : If A / B matrix need to be transposed
//   src_{A, B}_srow : First row         of input A/B matrix on this MPI process
//   src_{A, B}_nrow : Number of rows    of input A/B matrix on this MPI process
//   src_{A, B}_scol : First column      of input A/B matrix on this MPI process
//   src_{A, B}_ncol : Number of columns of input A/B matrix on this MPI process
//   dst_C_srow      : First row         of output C  matrix on this MPI process
//   dst_C_nrow      : Number of rows    of output C  matrix on this MPI process
//   dst_C_scol      : First column      of output C  matrix on this MPI process
//   dst_C_ncol      : Number of columns of output C  matrix on this MPI process
//   proc_grid       : MPI process grid [mp, np, kp], max(mp, np) must be a multiplier of min(np, mp).
//                     If proc_grid == NULL, CA3DMM will find a process grid solution.
//   comm            : MPI communicator of all MPI processes participating CA3DMM
// Output parameter:
//   *engine_       : Pointer to an initialized camm3d_engine structure
//   *workbuf_bytes : Optional. If pointer is not NULL, the returning value is the size 
//                    of work buffer, and ca3dmm_engine will not allocate work buffer.
//                    If pointer is NULL, ca3dmm_engine will allocate and free work buffer.
// Note: 
//   (1) CA3DMM does not check the correctness of src_{A, B}_{s, n}{row, col} and dst_C_{s, n}{row, col}
//   (2) If dst_C_{s, n}{row, col} are all -1, CA3DMM will not redistribute output C matrix
void ca3dmm_engine_init(
    const int m, const int n, const int k, const int trans_A, const int trans_B, 
    const int src_A_srow, const int src_A_nrow, 
    const int src_A_scol, const int src_A_ncol,
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, 
    ca3dmm_engine_p *engine_, size_t *workbuf_bytes
);

// Initialize a camm3d_engine structure for C := B^T * B
// Input parameters:
//   n, k       : Size of matrix B (k * n) and C (n * n)
//   src_B_srow : First row         of input  B matrix on this MPI process
//   src_B_nrow : Number of rows    of input  B matrix on this MPI process
//   src_B_scol : First column      of input  B matrix on this MPI process
//   src_B_ncol : Number of columns of input  B matrix on this MPI process
//   dst_C_srow : First row         of output C matrix on this MPI process
//   dst_C_nrow : Number of rows    of output C matrix on this MPI process
//   dst_C_scol : First column      of output C matrix on this MPI process
//   dst_C_ncol : Number of columns of output C matrix on this MPI process
//   proc_grid  : MPI process grid [mp, np, kp], mp must == np.
//                If proc_grid == NULL, CA3DMM will find a process grid solution.
//   comm       : MPI communicator of all MPI processes participating CA3DMM
// Output parameter:
//   *engine_       : Pointer to an initialized camm3d_engine structure
//   *workbuf_bytes : Optional. If pointer is not NULL, the returning value is the size 
//                    of work buffer, and ca3dmm_engine will not allocate work buffer.
//                    If pointer is NULL, ca3dmm_engine will allocate and free work buffer.
// Note: 
//   (1) CA3DMM does not check the correctness of src_B_{s, n}{row, col} and dst_C_{s, n}{row, col}
//   (2) If dst_C_{s, n}{row, col} are all -1, CA3DMM will not redistribute output C matrix
void ca3dmm_engine_init_BTB(
    const int n, const int k, 
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, 
    ca3dmm_engine_p *engine_, size_t *workbuf_bytes
);

// Attach an external work buffer for camm3d_engine
// Input parameters:
//   engine   : Initialized camm3d_engine_p
//   work_buf : Work buffer, size >= *workbuf_bytes returned by 
//              ca3dmm_engine_init() or ca3dmm_engine_init_BTB()
void ca3dmm_engine_attach_workbuf(ca3dmm_engine_p engine, void *work_buf);

// Free a camm3d_engine structure
void ca3dmm_engine_free(ca3dmm_engine_p *engine_);

// Perform Communication-Avoiding 3D Matrix Multiplication (CA3DMM)
// Input parameters:
//   engine      : An initialize camm3d_engine structure
//   src_{A, B}  : Size unknown, input A/B matrix block (col-major) on this MPI process
//   ld{A, B, C} : Leading dimension of src_{A, B} and dst_C
//                 If A/B is not transposed, ldA >= m and ldB >= k. 
//                 If A/B is     transposed, ldA >= k and ldB >= n.
//                 If C block is specified in ca3dmm_engine_{init, init_BTB}(), ldC >= dst_C_nrow.
//                 If C block is not specified in ca3dmm_engine_{init, init_BTB}(), ldC will be ignored.
// Output parameters:
//   dst_C       : If C block is specified in ca3dmm_engine_{init, init_BTB}(), dst_C is a 
//                 pointer to a buffer of size >= ldC * dst_C_ncol for storing the required C block.
//                 If C block is not specified in ca3dmm_engine_{init, init_BTB}(), dst_C will be ignored.
// In engine->:
//   C_out       : Size C_out_nrow-by-C_out_ncol, output C matrix block on this MPI process
//   C_out_srow  : First row         of output C matrix on this MPI process
//   C_out_nrow  : Number of rows    of output C matrix on this MPI process
//   C_out_scol  : First column      of output C matrix on this MPI process
//   C_out_ncol  : Number of columns of output C matrix on this MPI process
void ca3dmm_engine_exec(
    ca3dmm_engine_p engine,
    const void *src_A, const int ldA,
    const void *src_B, const int ldB,
    void *dst_C, const int ldC
);

// Reset statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_reset_stat(ca3dmm_engine_p engine);

// Print statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_print_stat(ca3dmm_engine_p engine);

#ifdef __cplusplus
}
#endif

#endif
