#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

#include "mat_redist.h"
#include "utils.h"

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

// Set up a mat_redist_engine_s for redistributing a 2D partitioned matrix
void mat_redist_engine_init(
    const int src_srow, const int src_scol, const int src_nrow, const int src_ncol, 
    const int req_srow, const int req_scol, const int req_nrow, const int req_ncol,
    MPI_Comm comm, MPI_Datatype dtype, const size_t dt_size, dev_type_t dev_type,
    mat_redist_engine_p *engine_, size_t *workbuf_bytes
)
{
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    mat_redist_engine_p engine = (mat_redist_engine_p) malloc(sizeof(mat_redist_engine_s));
    memset(engine, 0, sizeof(mat_redist_engine_s));

    // Set up basic MPI and source block info
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);
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
    engine->dev_type = dev_type;

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
    engine->n_proc_send = n_proc_send;
    engine->send_cnt    = send_cnt;
    engine->send_info0  = send_info0;

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
    engine->n_proc_recv = n_proc_recv;
    engine->recv_cnt    = recv_cnt;
    engine->recv_info0  = recv_info0;

    // No need to use external work buffer for integer arrays,
    // these arrays should be accessed on host
    engine->send_ranks  = (int *) malloc(sizeof(int) * n_proc_send);
    engine->send_sizes  = (int *) malloc(sizeof(int) * n_proc_send);
    engine->send_displs = (int *) malloc(sizeof(int) * (n_proc_send + 1));
    engine->sblk_sizes  = (int *) malloc(sizeof(int) * n_proc_send * 4);
    engine->recv_ranks  = (int *) malloc(sizeof(int) * n_proc_recv);
    engine->recv_sizes  = (int *) malloc(sizeof(int) * n_proc_recv);
    engine->recv_displs = (int *) malloc(sizeof(int) * (n_proc_recv + 1));
    engine->rblk_sizes  = (int *) malloc(sizeof(int) * n_proc_recv * 4);

    // Set up send blocks metadata
    int *sblk_sizes  = engine->sblk_sizes;
    int *send_ranks  = engine->send_ranks;
    int *send_displs = engine->send_displs;
    int *send_sizes  = engine->send_sizes;
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
    free(engine->send_info0);
    engine->send_info0 = NULL;

    // Set up receive blocks metadata
    int *rblk_sizes  = engine->rblk_sizes;
    int *recv_ranks  = engine->recv_ranks;
    int *recv_displs = engine->recv_displs;
    int *recv_sizes  = engine->recv_sizes;
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
    free(engine->recv_info0);
    engine->recv_info0 = NULL;

    // Build a new communicator with graph info
    int reorder = 0;
    MPI_Dist_graph_create_adjacent(
        comm, n_proc_recv, recv_ranks, MPI_UNWEIGHTED, n_proc_send, 
        send_ranks, MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, &engine->graph_comm
    );

    // Calculate work buffer size and allocate it if needed
    size_t workbuf_bytes_ = 0;
    workbuf_bytes_ += dt_size * send_cnt;  // send_buf
    workbuf_bytes_ += dt_size * recv_cnt;  // recv_buf
    if (workbuf_bytes != NULL)
    {
        engine->alloc_workbuf = 0;
        *workbuf_bytes = workbuf_bytes_;
    } else {
        engine->alloc_workbuf = 1;
        void *workbuf_h, *workbuf_d;
        MALLOC_ATTACH_WORKBUF(
            mat_redist_engine_attach_workbuf, mat_redist_engine_free, 
            engine, dev_type, workbuf_bytes_, workbuf_h, workbuf_d
        );
    }

    free(all_src_req_info);
    *engine_ = engine;

    MPI_Barrier(comm);
}

// Attach an external work buffer for mat_redist_engine
void mat_redist_engine_attach_workbuf(mat_redist_engine_p engine, void *workbuf_h, void *workbuf_d)
{
    if (engine == NULL)
    {
        WARNING_PRINTF("mat_redist_engine not initialized\n");
        return;
    }

    int dt_size  = engine->dt_size;
    int send_cnt = engine->send_cnt;

    engine->workbuf_h = workbuf_h;
    engine->workbuf_d = workbuf_d;

    dev_type_t dev_type = engine->dev_type;
    if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
    {
        engine->sendbuf_h = (void *) workbuf_h;
        engine->recvbuf_h = (void *) ((char *) engine->sendbuf_h + dt_size * send_cnt);
    }

    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        engine->sendbuf_d = (void *) workbuf_d;
        engine->recvbuf_d = (void *) ((char *) engine->sendbuf_d + dt_size * send_cnt);
    }
    #endif
}

// Destroy a mat_redist_engine_s
void mat_redist_engine_free(mat_redist_engine_p *engine_)
{
    mat_redist_engine_p engine = *engine_;
    if (engine == NULL) return;
    dev_type_t dev_type = engine->dev_type;
    if (engine->alloc_workbuf)
    {
        if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
            dev_type_free(engine->workbuf_h, DEV_TYPE_HOST);
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            dev_type_free(engine->workbuf_d, dev_type);
        #endif
    }
    free(engine->send_ranks);
    free(engine->send_sizes);
    free(engine->send_displs);
    free(engine->sblk_sizes);
    free(engine->recv_ranks);
    free(engine->recv_sizes);
    free(engine->recv_displs);
    free(engine->rblk_sizes);
    MPI_Comm_free(&engine->graph_comm);
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
        WARNING_PRINTF("mat_redist_engine not initialized\n");
        return;
    }

    size_t dt_size = engine->dt_size;
    int send_cnt = engine->send_cnt;
    int recv_cnt = engine->recv_cnt;
    dev_type_t dev_type = engine->dev_type;

    double hd_start_t, hd_stop_t;
    engine->hd_trans_ms = 0.0;

    // Pack the send_buf
    int  src_srow     = engine->src_srow;
    int  src_scol     = engine->src_scol;
    int  n_proc_send  = engine->n_proc_send;
    int  *send_sizes  = engine->send_sizes;
    int  *send_displs = engine->send_displs;
    int  *sblk_sizes  = engine->sblk_sizes;
    char *sendbuf_h   = (char*) engine->sendbuf_h;
    char *sendbuf_d   = (char*) engine->sendbuf_d;
    char *src_blk_    = (char*) src_blk;
    for (int isend = 0; isend < n_proc_send; isend++)
    {
        int *i_sblk_size = sblk_sizes + isend * 4;
        int i_send_srow = i_sblk_size[0];
        int i_send_scol = i_sblk_size[1];
        int i_send_nrow = i_sblk_size[2];
        int i_send_ncol = i_sblk_size[3];
        int local_srow  = i_send_srow - src_srow;
        int local_scol  = i_send_scol - src_scol;
        const char *i_send_src = src_blk_ + dt_size * (local_srow * src_ld + local_scol);
        char *i_send_buf;
        if (dev_type == DEV_TYPE_HOST)
            i_send_buf = sendbuf_h + dt_size * send_displs[isend];
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            i_send_buf = sendbuf_d + dt_size * send_displs[isend];
        #endif
        dev_type_copy_mat_blk(
            dt_size, i_send_nrow, i_send_ncol, 
            i_send_src, src_ld, i_send_buf, i_send_ncol, dev_type
        );
    }  // End of isend loop

    // Redistribute data using MPI_Neighbor_alltoallv
    int  *recv_sizes  = engine->recv_sizes;
    int  *recv_displs = engine->recv_displs;
    void *recvbuf_h   = engine->recvbuf_h;
    void *recvbuf_d   = engine->recvbuf_d;
    if (dev_type == DEV_TYPE_HOST)
    {
        MPI_Neighbor_alltoallv(
            sendbuf_h, send_sizes, send_displs, engine->dtype, 
            recvbuf_h, recv_sizes, recv_displs, engine->dtype, engine->graph_comm
        );
    }
    #ifdef USE_CUDA
    if (dev_type == DEV_TYPE_CUDA)
    {
        hd_start_t = MPI_Wtime();
        dev_type_memcpy(sendbuf_h, sendbuf_d, dt_size * send_cnt, DEV_TYPE_HOST, dev_type);
        hd_stop_t = MPI_Wtime();
        engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
        
        MPI_Neighbor_alltoallv(
            sendbuf_h, send_sizes, send_displs, engine->dtype, 
            recvbuf_h, recv_sizes, recv_displs, engine->dtype, engine->graph_comm
        );
        
        hd_start_t = MPI_Wtime();
        dev_type_memcpy(recvbuf_d, recvbuf_h, dt_size * recv_cnt, dev_type, DEV_TYPE_HOST);
        hd_stop_t = MPI_Wtime();
        engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
    }
    if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
    {
        MPI_Neighbor_alltoallv(
            sendbuf_d, send_sizes, send_displs, engine->dtype, 
            recvbuf_d, recv_sizes, recv_displs, engine->dtype, engine->graph_comm
        );
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
        char *i_recv_dst = dst_blk_ + dt_size * (local_srow * dst_ld + local_scol);
        char *i_recv_buf;
        if (dev_type == DEV_TYPE_HOST)
            i_recv_buf = recvbuf_h + dt_size * recv_displs[irecv];
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            i_recv_buf = recvbuf_d + dt_size * recv_displs[irecv];
        #endif
        dev_type_copy_mat_blk(
            dt_size, i_recv_nrow, i_recv_ncol, 
            i_recv_buf, i_recv_ncol, i_recv_dst, dst_ld, dev_type
        );
    }  // End of recv_cnt loop

    MPI_Barrier(engine->graph_comm);
}