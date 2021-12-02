#ifndef __MAT_REDIST_H__
#define __MAT_REDIST_H__

#include "mpi.h"
#include "dev_type.h"

struct mat_redist_engine
{
    MPI_Comm graph_comm;    // New MPI communicator for neighbor alltoall
    MPI_Datatype dtype;     // Matrix element MPI data type
    size_t  dt_size;        // Matrix element MPI data type size in bytes
    int     nproc;          // Number of MPI processes in comm
    int     rank;           // Rank of this MPI process in comm
    int     src_srow;       // The starting row of this process's source matrix block
    int     src_scol;       // Number of rows of this process's source matrix block
    int     src_nrow;       // The starting columns of this process's source matrix block
    int     src_ncol;       // Number of columns of this process's source matrix block
    int     req_srow;       // The starting row this process requires
    int     req_scol;       // Number of rows this process requires
    int     req_nrow;       // The starting columns this process requires
    int     req_ncol;       // Number of columns this process requires
    int     n_proc_send;    // Number of processes this process needs to send its original block to
    int     n_proc_recv;    // Number of processes this process needs to receive its required block from
    int     send_cnt;       // Number of elements to send
    int     recv_cnt;       // Number of elements to receive
    int     alloc_workbuf;  // If work_buf is allocated by mat_redist_engine
    int     *send_ranks;    // Size n_proc_send, MPI ranks this process need to send a block to 
    int     *send_sizes;    // Size n_proc_send, sizes of blocks this process need to send
    int     *send_displs;   // Size n_proc_send+1, send block displacements in send_buf
    int     *sblk_sizes;    // Size n_proc_send*4, each row describes a send block's srow, scol, nrow, ncol
    int     *recv_ranks;    // Size n_proc_recv, MPI ranks this process need to receive a block from
    int     *recv_sizes;    // Size n_proc_recv, sizes of blocks this process need to receive
    int     *recv_displs;   // Size n_proc_recv+1, receive block displacements in recv_buf
    int     *rblk_sizes;    // Size n_proc_recv*4, each row describes a receive block's srow, scol, nrow, ncol
    int     *send_info0;    // Temporary work array for determining send and receive pairs
    int     *recv_info0;    // Temporary work array for determining send and receive pairs
    void    *sendbuf_h;     // Send buffer on host
    void    *recvbuf_h;     // Receive buffer on host
    void    *sendbuf_d;     // Send buffer on device
    void    *recvbuf_d;     // Receive buffer on device
    void    *workbuf_h;     // Work buffer, all void* with _h suffix above are aliases to workbuf_h
    void    *workbuf_d;     // Work buffer, all void* with _d suffix above are aliases to workbuf_d
    double  hd_trans_ms;    // Time (milliseconds) used in host-device data transfer
    dev_type_t dev_type;    // Data resident device type
};
typedef struct mat_redist_engine  mat_redist_engine_s;
typedef struct mat_redist_engine* mat_redist_engine_p;

#ifdef __cplusplus
extern "C" {
#endif

// Set up a mat_redist_engine_s for redistributing a 2D partitioned matrix
// Note: the source blocks of any two processes should not overlap with each other
// Input parameters:
//   src_s{row, col} : The starting row / column of this process's source matrix block
//   src_n{row, col} : Number of rows / columns of this process's source matrix block
//   req_s{row, col} : The starting row / column this process requires
//   req_n{row, col} : Number of rows / columns this process requires
//   comm            : MPI communicator
//   dtype           : Matrix element MPI data type
//   dt_size         : Matrix element MPI data type size in bytes
//   dev_type        : Data resident device type
// Output parameters:
//   *engine_        : Initialized mat_redist_engine_p
//   *workbuf_bytes  : Optional. If pointer is not NULL, the returning value is the size 
//                     of work buffer, and mat_redist_engine will not allocate work buffer.
//                     If pointer is NULL, mat_redist_engine will allocate and free work buffer.
void mat_redist_engine_init(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    MPI_Comm comm, MPI_Datatype dtype, const size_t dt_size, dev_type_t dev_type,
    mat_redist_engine_p *engine_, size_t *workbuf_bytes
);

// Attach an external work buffer for mat_redist_engine
// Input parameters:
//   engine    : Initialized mat_redist_engine_p
//   workbuf_h : Work buffer on host, size >= *workbuf_bytes returned by mat_redist_engine_init()
//   workbuf_d : Work buffer on device, size >= *workbuf_bytes returned by mat_redist_engine_init()
// Note:
//   1. workbuf_d can be NULL if dev_type == DEV_TYPE_HOST
//   2. workbuf_h can be NULL if dev_type == DEV_TYPE_CUDA_MPI_DIRECT
void mat_redist_engine_attach_workbuf(mat_redist_engine_p engine, void *workbuf_h, void *workbuf_d);

// Perform matrix data redistribution
// Input parameters:
//   engine  : Initialized mat_redist_engine_p
//   src_blk : Source matrix block of this process
//   src_ld  : Leading dimension of src_blk
//   dst_ld  : Leading dimension of dst_blk
// Output parameter:
//   dst_blk : Destination (required) matrix block of this process
void mat_redist_engine_exec(
    mat_redist_engine_p engine, const void *src_blk, const int src_ld, 
    void *dst_blk, const int dst_ld
);

// Free a mat_redist_engine_s
void mat_redist_engine_free(mat_redist_engine_p *engine_);

#ifdef __cplusplus
}
#endif

#endif
