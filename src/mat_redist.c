#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "memory.h"
#include "mat_redist.h"

static void calc_seg_intersection(
    int s0, int e0, int s1, int e1, 
    int *is_intersect, int *is, int *ie
)
{
    if (s0 > s1)
    {
        int swap;
        swap = s0; s0 = s1; s1 = swap;
        swap = e0; e0 = e1; e1 = swap;
    }
    if (s1 > e0 || s1 > e1 || s0 > e0)
    {
        *is_intersect = 0;
        *is = -1;
        *ie = -1;
        return;
    }
    *is_intersect = 1;
    *is = s1;
    *ie = (e0 < e1) ? e0 : e1;
}

static void calc_rect_intersection(
    int xs0, int xe0, int ys0, int ye0,
    int xs1, int xe1, int ys1, int ye1,
    int *is_intersect, int *ixs, int *ixe, int *iys, int *iye
)
{
    calc_seg_intersection(xs0, xe0, xs1, xe1, is_intersect, ixs, ixe);
    if (*is_intersect == 0) return;
    calc_seg_intersection(ys0, ye0, ys1, ye1, is_intersect, iys, iye);
}

static void copy_matrix_block(
    const size_t dt_size, const int nrow, const int ncol, 
    const void *src, const int lds, void *dst, const int ldd,
    device_type device
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    for (int irow = 0; irow < nrow; irow++)
    {
        size_t src_offset = (size_t) irow * lds_;
        size_t dst_offset = (size_t) irow * ldd_;
        OUR_MEMCPY(dst_ + dst_offset, src_ + src_offset, row_msize, device, device);
    }
}

void mat_redist_engine_init(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    MPI_Comm comm, MPI_Datatype dtype, const size_t dt_size, mat_redist_engine_p *engine_
)
{
    mat_redist_engine_init_ex(src_srow, src_scol, src_nrow, src_ncol, req_srow,
    req_scol, req_nrow, req_ncol, DEVICE_TYPE_HOST, comm, dtype, dt_size, engine_);
}

// Set up a mat_redist_engine_s for redistributing a 2D partitioned matrix
void mat_redist_engine_init_ex(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    device_type communication_device,
    MPI_Comm comm, MPI_Datatype dtype, const size_t dt_size, mat_redist_engine_p *engine_
)
{
    mat_redist_engine_p engine = (mat_redist_engine_p) malloc(sizeof(mat_redist_engine_s));

    // Set up basic MPI and source block info
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
    //engine->comm     = comm;
    engine->rank     = rank;
    engine->nproc    = nproc;
    engine->dtype    = dtype;
    engine->dt_size  = dt_size;
    engine->src_srow = src_srow;
    engine->src_nrow = src_nrow;
    engine->src_scol = src_scol;
    engine->src_ncol = src_ncol;
    engine->req_srow = req_srow;
    engine->req_nrow = req_nrow;
    engine->req_scol = req_scol;
    engine->req_ncol = req_ncol;
    engine->communication_device = communication_device;

    // Gather all processes' source and required block info
    int src_erow = src_srow + src_nrow - 1;
    int src_ecol = src_scol + src_ncol - 1;
    int req_erow = req_srow + req_nrow - 1;
    int req_ecol = req_scol + req_ncol - 1;
    int my_src_req_info[8] = {
        src_srow, src_scol, src_erow, src_ecol, 
        req_srow, req_scol, req_erow, req_ecol
    };
    int *all_src_req_info = (int*) malloc(sizeof(int) * 8 * nproc);
    MPI_Allgather(my_src_req_info, 8, MPI_INT, all_src_req_info, 8, MPI_INT, comm);

    // Calculate send_info
    int send_cnt = 0, n_proc_send = 0;
    int is_intersect, int_srow, int_erow, int_scol, int_ecol;
    int *send_info0 = (int*) malloc(sizeof(int) * 6 * nproc);
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *i_req_info = all_src_req_info + iproc * 8 + 4;
        int i_req_srow = i_req_info[0];
        int i_req_scol = i_req_info[1];
        int i_req_erow = i_req_info[2];
        int i_req_ecol = i_req_info[3];
        calc_rect_intersection(
            src_srow, src_erow, src_scol, src_ecol,
            i_req_srow, i_req_erow, i_req_scol, i_req_ecol,
            &is_intersect, &int_srow, &int_erow, &int_scol, &int_ecol
        );
        if (is_intersect)
        {
            int *send_info0_i = send_info0 + n_proc_send * 6;
            send_info0_i[0] = int_srow;
            send_info0_i[1] = int_scol;
            send_info0_i[2] = int_erow - int_srow + 1;
            send_info0_i[3] = int_ecol - int_scol + 1;
            send_info0_i[4] = iproc;
            send_info0_i[5] = send_cnt;
            n_proc_send++;
            send_cnt += send_info0_i[2] * send_info0_i[3];
        }
    }  // End of iproc loop
    int  *send_ranks  = (int*)  malloc(sizeof(int) * n_proc_send);
    int  *send_sizes  = (int*)  malloc(sizeof(int) * n_proc_send);
    int  *send_displs = (int*)  malloc(sizeof(int) * (n_proc_send + 1));
    int  *sblk_sizes  = (int*)  malloc(sizeof(int) * n_proc_send * 4);
    void *send_buf    = (void*) _OUR_MALLOC(dt_size     * send_cnt, communication_device);
    // printf("n_proc_send: %i, send_ranks: %p, sblk_sizes: %p, send_sizes: %p, send_displs: %p, send_buf: %p, send_cnt: %i\n",
    // n_proc_send, send_ranks, sblk_sizes, send_sizes, send_displs, send_buf, send_cnt);
    if ((((n_proc_send != 0) && (send_ranks == NULL || sblk_sizes == NULL || send_sizes == NULL))  || send_displs == NULL || ((send_buf == NULL) && (send_cnt != 0))))
    {
        fprintf(stderr, "[ERROR] Failed to allocate send_info (size %d) or send_buf (size %d)\n", 7 * n_proc_send, send_cnt);
        free(engine);
        *engine_ = NULL;
        return;
    }
    for (int i = 0; i < n_proc_send; i++)
    {
        int *send_info0_i = send_info0 + i * 6;
        int *sblk_size_i  = sblk_sizes + i * 4;
        sblk_size_i[0] = send_info0_i[0];
        sblk_size_i[1] = send_info0_i[1];
        sblk_size_i[2] = send_info0_i[2];
        sblk_size_i[3] = send_info0_i[3];
        send_ranks[i]  = send_info0_i[4];
        send_displs[i] = send_info0_i[5];
        send_sizes[i]  = sblk_size_i[2] * sblk_size_i[3];
    }
    send_displs[n_proc_send] = send_cnt;
    engine->n_proc_send = n_proc_send;
    engine->send_ranks  = send_ranks;
    engine->send_sizes  = send_sizes;
    engine->send_displs = send_displs;
    engine->sblk_sizes  = sblk_sizes;
    engine->send_buf    = send_buf;
    free(send_info0);

    // Calculate recv_info
    int recv_cnt = 0, n_proc_recv = 0;
    int *recv_info0 = (int*) malloc(sizeof(int) * 6 * nproc);
    for (int iproc = 0; iproc < nproc; iproc++)
    {
        int *i_src_info = all_src_req_info + iproc * 8;
        int i_src_srow = i_src_info[0];
        int i_src_scol = i_src_info[1];
        int i_src_erow = i_src_info[2];
        int i_src_ecol = i_src_info[3];
        calc_rect_intersection(
            req_srow, req_erow, req_scol, req_ecol,
            i_src_srow, i_src_erow, i_src_scol, i_src_ecol,
            &is_intersect, &int_srow, &int_erow, &int_scol, &int_ecol
        );
        if (is_intersect)
        {
            int *recv_info0_i = recv_info0 + n_proc_recv * 6;
            recv_info0_i[0] = int_srow;
            recv_info0_i[1] = int_scol;
            recv_info0_i[2] = int_erow - int_srow + 1;
            recv_info0_i[3] = int_ecol - int_scol + 1;
            recv_info0_i[4] = iproc;
            recv_info0_i[5] = recv_cnt;
            n_proc_recv++;
            recv_cnt += recv_info0_i[2] * recv_info0_i[3];
        }
    }  // End of iproc loop
    int  *recv_ranks  = (int*)  malloc(sizeof(int) * n_proc_recv);
    int  *recv_sizes  = (int*)  malloc(sizeof(int) * n_proc_recv);
    int  *recv_displs = (int*)  malloc(sizeof(int) * (n_proc_recv + 1));
    int  *rblk_sizes  = (int*)  malloc(sizeof(int) * n_proc_recv * 4);
    void *recv_buf    = (void*) _OUR_MALLOC(dt_size     * recv_cnt, communication_device);
    if (((((n_proc_recv != 0) && (recv_ranks == NULL || recv_sizes == NULL ||  rblk_sizes == NULL)) || recv_displs == NULL || ((recv_buf == NULL) && (recv_cnt != 0))))) {
        fprintf(stderr, "[ERROR] Failed to allocate recv_info (size %d) or recv_buf (size %d)\n", 7 * n_proc_recv, recv_cnt);
        free(engine);
        *engine_ = NULL;
        return;
    }
    for (int i = 0; i < n_proc_recv; i++)
    {
        int *recv_info0_i = recv_info0 + i * 6;
        int *rblk_size_i  = rblk_sizes + i * 4;
        rblk_size_i[0] = recv_info0_i[0];
        rblk_size_i[1] = recv_info0_i[1];
        rblk_size_i[2] = recv_info0_i[2];
        rblk_size_i[3] = recv_info0_i[3];
        recv_ranks[i]  = recv_info0_i[4];
        recv_displs[i] = recv_info0_i[5];
        recv_sizes[i]  = rblk_size_i[2] * rblk_size_i[3];
    }
    recv_displs[n_proc_recv] = recv_cnt;
    engine->n_proc_recv = n_proc_recv;
    engine->recv_ranks  = recv_ranks;
    engine->recv_sizes  = recv_sizes;
    engine->recv_displs = recv_displs;
    engine->rblk_sizes  = rblk_sizes;
    engine->recv_buf    = recv_buf;
    free(recv_info0);

    // Build a new communicator with graph info
    int reorder = 0;
    MPI_Info mpi_info;
    MPI_Info_create(&mpi_info);
    MPI_Dist_graph_create_adjacent(
        comm, n_proc_recv, recv_ranks, MPI_UNWEIGHTED, n_proc_send, 
        send_ranks, MPI_UNWEIGHTED, mpi_info, reorder, &engine->comm
    );
    MPI_Info_free(&mpi_info);

    free(all_src_req_info);
    *engine_ = engine;

    MPI_Barrier(engine->comm);
}

// Destroy a mat_redist_engine_s
void mat_redist_engine_free(mat_redist_engine_p *engine_)
{
    mat_redist_engine_p engine = *engine_;
    if (engine == NULL) return;
    free(engine->send_ranks);
    free(engine->send_sizes);
    free(engine->send_displs);
    free(engine->sblk_sizes);
    free(engine->recv_ranks);
    free(engine->recv_sizes);
    free(engine->recv_displs);
    free(engine->rblk_sizes);
    OUR_FREE(engine->send_buf, engine->communication_device);
    OUR_FREE(engine->recv_buf, engine->communication_device);
    free(engine);
    *engine_ = NULL;
}

// Perform matrix data redistribution
void mat_redist_engine_exec(
    mat_redist_engine_p engine, const void *src_blk, const int src_ld, 
    void *dst_blk, const int dst_ld
)
{
    if (engine == NULL)
    {
        fprintf(stderr, "[ERROR] engine == NULL\n");
        return;
    }

    size_t dt_size = engine->dt_size;

    // Pack the send_buf
    int  src_srow     = engine->src_srow;
    int  src_scol     = engine->src_scol;
    int  n_proc_send  = engine->n_proc_send;
    int  *send_sizes  = engine->send_sizes;
    int  *send_displs = engine->send_displs;
    int  *sblk_sizes  = engine->sblk_sizes;
    char *send_buf    = (char*) engine->send_buf;
    char *src_blk_    = (char*) src_blk;
    const device_type dev = engine->communication_device;
    for (int isend = 0; isend < n_proc_send; isend++)
    {
        int *i_sblk_size = sblk_sizes + isend * 4;
        int i_send_srow = i_sblk_size[0];
        int i_send_scol = i_sblk_size[1];
        int i_send_nrow = i_sblk_size[2];
        int i_send_ncol = i_sblk_size[3];
        int local_srow  = i_send_srow - src_srow;
        int local_scol  = i_send_scol - src_scol;
        char *i_send_buf  = send_buf + dt_size * send_displs[isend];
        const char *i_send_src = src_blk_ + dt_size * (local_srow * src_ld + local_scol);
        copy_matrix_block(dt_size, i_send_nrow, i_send_ncol, i_send_src, src_ld, i_send_buf, i_send_ncol, dev);
    }  // End of isend loop
#if USE_GPU
    if(dev == DEVICE_TYPE_DEVICE) {
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#endif

    // Redistribute data using MPI_Neighbor_alltoallv
    int  *recv_sizes  = engine->recv_sizes;
    int  *recv_displs = engine->recv_displs;
    void *recv_buf    = engine->recv_buf;
    MPI_Neighbor_alltoallv(
        send_buf, send_sizes, send_displs, engine->dtype, 
        recv_buf, recv_sizes, recv_displs, engine->dtype, engine->comm
    );
#if USE_GPU
    if(dev == DEVICE_TYPE_DEVICE) {
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#endif

    // Repack received blocks
    int  req_srow    = engine->req_srow;
    int  req_scol    = engine->req_scol;
    int  n_proc_recv = engine->n_proc_recv;
    int  *rblk_sizes = engine->rblk_sizes;
    char *dst_blk_   = (char*) dst_blk;
    for (int irecv = 0; irecv < n_proc_recv; irecv++)
    {
        int *i_rblk_size = rblk_sizes + irecv * 4;
        int i_recv_srow = i_rblk_size[0];
        int i_recv_scol = i_rblk_size[1];
        int i_recv_nrow = i_rblk_size[2];
        int i_recv_ncol = i_rblk_size[3];
        int local_srow  = i_recv_srow - req_srow;
        int local_scol  = i_recv_scol - req_scol;
        char *i_recv_buf = recv_buf + dt_size * recv_displs[irecv];
        char *i_recv_dst = dst_blk_ + dt_size * (local_srow * dst_ld + local_scol);
        copy_matrix_block(dt_size, i_recv_nrow, i_recv_ncol, i_recv_buf, i_recv_ncol, i_recv_dst, dst_ld, dev);
    }  // End of recv_cnt loop

    MPI_Barrier(engine->comm);
}
