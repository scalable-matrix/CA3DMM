#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "utils.h"
#include "cannon.h"
#include "cpu_linalg_lib_wrapper.h"
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

// Initialize a cannon_engine for 2D Cannon matrix multiplication algorithm
void cannon_engine_init(
    const int m, const int n, const int k, MPI_Comm comm, 
    dev_type_t dev_type, cannon_engine_p *engine_, size_t *workbuf_bytes
)
{
    *engine_ = NULL;
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    int my_rank, n_proc, np_dim;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_proc);
    np_dim = (int) sqrt((double) n_proc);
    if (np_dim * np_dim != n_proc)
    {
        ERROR_PRINTF("Communicator size %d is not a square number\n", n_proc);
        return;
    }

    double start_t = MPI_Wtime();

    MPI_Comm comm_cart;
    int dims[2] = {np_dim, np_dim};
    int periods[2] = {1, 1};
    int rank_cart, coords[2];
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_cart);
    MPI_Comm_rank(comm_cart, &rank_cart);
    MPI_Cart_coords(comm_cart, rank_cart, 2, &coords[0]);
    int rank_row = coords[0];
    int rank_col = coords[1];

    cannon_engine_p engine = (cannon_engine_p) malloc(sizeof(cannon_engine_s));
    memset(engine, 0, sizeof(cannon_engine_s));
    engine->m           = m;
    engine->n           = n;
    engine->k           = k;
    engine->my_rank     = rank_cart;
    engine->rank_row    = rank_row;
    engine->rank_col    = rank_col;
    engine->np_dim      = np_dim;
    engine->comm        = comm_cart;
    engine->shift0_ms   = 0.0;
    engine->lshift_ms   = 0.0;
    engine->gemm_ms     = 0.0;
    engine->exec_ms     = 0.0;
    engine->hd_trans_ms = 0.0;
    engine->n_exec      = 0;
    engine->dev_type    = dev_type;

    calc_block_spos_size(m, np_dim, rank_row, &engine->A_srow, &engine->A_nrow);
    calc_block_spos_size(k, np_dim, rank_col, &engine->A_scol, &engine->A_ncol);
    calc_block_spos_size(k, np_dim, rank_row, &engine->B_srow, &engine->B_nrow);
    calc_block_spos_size(n, np_dim, rank_col, &engine->B_scol, &engine->B_ncol);
    calc_block_spos_size(m, np_dim, rank_row, &engine->C_srow, &engine->C_nrow);
    calc_block_spos_size(n, np_dim, rank_col, &engine->C_scol, &engine->C_ncol);

    int max_A_blk_size = (m / np_dim + 1) * (k / np_dim + 1);
    int max_B_blk_size = (k / np_dim + 1) * (n / np_dim + 1);
    int max_C_blk_size = (m / np_dim + 1) * (n / np_dim + 1);
    engine->max_A_blk_size = max_A_blk_size;
    engine->max_B_blk_size = max_B_blk_size;
    engine->max_C_blk_size = max_C_blk_size;

    size_t workbuf_bytes_ = 0;
    workbuf_bytes_ += sizeof(double) * max_A_blk_size;  // A_recv
    workbuf_bytes_ += sizeof(double) * max_B_blk_size;  // B_recv

    int  min_k_blk_size  = 160;
    int  curr_k_blk_size = engine->A_ncol;
    int  gemm_cycle      = 1;
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        min_k_blk_size = 256;
    GET_ENV_INT_VAR(min_k_blk_size, "CANNON_MIN_KBLK_SIZE", "min_k_blk_size", 160, 16, 8192);
    if (curr_k_blk_size < min_k_blk_size)
    {
        gemm_cycle = (min_k_blk_size + curr_k_blk_size - 1) / curr_k_blk_size;
        if (gemm_cycle > np_dim) gemm_cycle = np_dim;
        workbuf_bytes_ += sizeof(double) * max_A_blk_size * gemm_cycle;  // A_stack
        workbuf_bytes_ += sizeof(double) * max_B_blk_size * gemm_cycle;  // B_stack
    }
    engine->gemm_cycle = gemm_cycle;

    // No need to use external work buffer for integer arrays,
    // these arrays should be accessed on host
    engine->m_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    engine->n_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    engine->k_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    int dummy;
    for (int i = 0; i <= np_dim; i++)
    {
        calc_block_spos_size(m, np_dim, i, engine->m_displs + i, &dummy);
        calc_block_spos_size(n, np_dim, i, engine->n_displs + i, &dummy);
        calc_block_spos_size(k, np_dim, i, engine->k_displs + i, &dummy);
    }

    engine->workbuf_h = NULL;
    engine->workbuf_d = NULL;
    if (workbuf_bytes != NULL)
    {
        engine->alloc_workbuf = 0;
        *workbuf_bytes = workbuf_bytes_;
    } else {
        engine->alloc_workbuf = 1;
        void *workbuf_h, *workbuf_d;
        MALLOC_ATTACH_WORKBUF(
            cannon_engine_attach_workbuf, cannon_engine_free, 
            engine, dev_type, workbuf_bytes_, workbuf_h, workbuf_d
        );
    }

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

// Attach an external work buffer for cannon_engine
void cannon_engine_attach_workbuf(cannon_engine_p engine, void *workbuf_h, void *workbuf_d)
{
    int m = engine->m;
    int n = engine->n;
    int k = engine->k;
    int np_dim     = engine->np_dim;
    int rank_row   = engine->rank_row;
    int rank_col   = engine->rank_col;
    int gemm_cycle = engine->gemm_cycle;
    int max_A_blk_size = engine->max_A_blk_size;
    int max_B_blk_size = engine->max_B_blk_size;
    
    // Assign work buffer
    engine->workbuf_h = workbuf_h;
    engine->workbuf_d = workbuf_d;
    dev_type_t dev_type = engine->dev_type;
    if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
    {
        engine->A_recv_h = workbuf_h;
        engine->B_recv_h = (void *) ((double *) engine->A_recv_h + max_A_blk_size);
        if (gemm_cycle > 1)
        {
            engine->A_stack_h = (void *) ((double *) engine->B_recv_h  + max_B_blk_size);
            engine->B_stack_h = (void *) ((double *) engine->A_stack_h + max_A_blk_size * gemm_cycle);
        } else {
            engine->A_stack_h = NULL;
            engine->B_stack_h = NULL;
        }
    }
    #ifdef USE_CUDA
    if (dev_type == DEV_TYPE_CUDA)  // An extra host buffer for MPI operations
    {
        engine->A_gemm_h = malloc(sizeof(double) * max_A_blk_size);
        engine->B_gemm_h = malloc(sizeof(double) * max_B_blk_size);
    }
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        engine->A_recv_d = workbuf_d;
        engine->B_recv_d = (void *) ((double *) engine->A_recv_d + max_A_blk_size);
        if (gemm_cycle > 1)
        {
            engine->A_stack_d = (void *) ((double *) engine->B_recv_d  + max_B_blk_size);
            engine->B_stack_d = (void *) ((double *) engine->A_stack_d + max_A_blk_size * gemm_cycle);
        } else {
            engine->A_stack_d = NULL;
            engine->B_stack_d = NULL;
        }
    }
    #endif
}

// Free a cannon_engine
void cannon_engine_free(cannon_engine_p *engine_)
{
    cannon_engine_p engine = *engine_;
    if (engine == NULL) return;
    dev_type_t dev_type = engine->dev_type;
    if (engine->alloc_workbuf)
    {
        if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
            dev_type_free(engine->workbuf_h, DEV_TYPE_HOST);
        #ifdef USE_CUDA
        if (dev_type == DEV_TYPE_CUDA)
        {
            free(engine->A_gemm_h);
            free(engine->B_gemm_h);
        }
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            dev_type_free(engine->workbuf_d, dev_type);
        #endif
    }
    free(engine->m_displs);
    free(engine->n_displs);
    free(engine->k_displs);
    MPI_Comm_free(&engine->comm);
    free(engine);
    *engine_ = NULL;
}

#define LOAD_PARAMS \
    int m = engine->m;  \
    int n = engine->n;  \
    int k = engine->k;  \
    int rank_row   = engine->rank_row;      \
    int rank_col   = engine->rank_col;      \
    int np_dim     = engine->np_dim;        \
    int gemm_cycle = engine->gemm_cycle;    \
    int *m_displs  = engine->m_displs;      \
    int *n_displs  = engine->n_displs;      \
    int *k_displs  = engine->k_displs;      \
    int max_A_blk_size = engine->max_A_blk_size;    \
    int max_B_blk_size = engine->max_B_blk_size;    \
    int src_offset = (rank_row + rank_col) % np_dim;            \
    int A_dst_col  = (rank_col - rank_row + np_dim) % np_dim;   \
    int B_dst_row  = (rank_row - rank_col + np_dim) % np_dim;   \
    int A_m        = m_displs[rank_row   + 1] - m_displs[rank_row];     \
    int B_n        = n_displs[rank_col   + 1] - n_displs[rank_col];     \
    int A_send_k   = k_displs[rank_col   + 1] - k_displs[rank_col];     \
    int A_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];   \
    int B_send_k   = k_displs[rank_row   + 1] - k_displs[rank_row];     \
    int B_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];   \
    int A_src_rank = rank_row   * np_dim + src_offset;  \
    int B_src_rank = src_offset * np_dim + rank_col;    \
    int A_dst_rank = rank_row   * np_dim + A_dst_col;   \
    int B_dst_rank = B_dst_row  * np_dim + rank_col;    \
    int ldAs       = A_m;                               \
    int ldBs       = (k / np_dim + 1) * gemm_cycle;     \
    int left_col   = (rank_col - 1 + np_dim) % np_dim;  \
    int right_col  = (rank_col + 1) % np_dim;           \
    int upper_row  = (rank_row - 1 + np_dim) % np_dim;  \
    int lower_row  = (rank_row + 1) % np_dim;           \
    int left_rank  = rank_row  * np_dim + left_col;     \
    int right_rank = rank_row  * np_dim + right_col;    \
    int lower_rank = lower_row * np_dim + rank_col;     \
    int upper_rank = upper_row * np_dim + rank_col;     \
    size_t max_A_blk_bytes = sizeof(double) * max_A_blk_size;   \
    size_t max_B_blk_bytes = sizeof(double) * max_B_blk_size;   \
    double *A_gemm_h  = (double *) engine->A_gemm_h;    \
    double *A_recv_h  = (double *) engine->A_recv_h;    \
    double *A_stack_h = (double *) engine->A_stack_h;   \
    double *B_gemm_h  = (double *) engine->B_gemm_h;    \
    double *B_recv_h  = (double *) engine->B_recv_h;    \
    double *B_stack_h = (double *) engine->B_stack_h;   \
    double *A_recv_d  = (double *) engine->A_recv_d;    \
    double *A_stack_d = (double *) engine->A_stack_d;   \
    double *B_recv_d  = (double *) engine->B_recv_d;    \
    double *B_stack_d = (double *) engine->B_stack_d;   \
    double *A_gemm_d  = NULL;       \
    double *B_gemm_d  = NULL;       \
    MPI_Comm comm = engine->comm;   \
    dev_type_t dev_type = engine->dev_type;


void cannon_engine_init_alignment(cannon_engine_p engine, const double *A_blk, const double *B_blk)
{
    LOAD_PARAMS;

    const double *A_send_ptr, *B_send_ptr;
    double *A_recv_ptr, *B_recv_ptr;
    double hd_start_t, hd_stop_t;
    if (dev_type == DEV_TYPE_HOST)
    {
        A_send_ptr = A_blk;  A_recv_ptr = A_recv_h;
        B_send_ptr = B_blk;  B_recv_ptr = B_recv_h;
    }
    #ifdef USE_CUDA
    if (dev_type == DEV_TYPE_CUDA)
    {
        hd_start_t = MPI_Wtime();
        dev_type_memcpy(A_gemm_h, A_blk, sizeof(double) * A_m * A_send_k, DEV_TYPE_HOST, dev_type);
        dev_type_memcpy(B_gemm_h, B_blk, sizeof(double) * B_send_k * B_n, DEV_TYPE_HOST, dev_type);
        hd_stop_t = MPI_Wtime();
        engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
        A_send_ptr = A_gemm_h;  A_recv_ptr = A_recv_h;
        B_send_ptr = B_gemm_h;  B_recv_ptr = B_recv_h;
    }
    if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
    {
        A_send_ptr = A_blk;  A_recv_ptr = A_recv_d;
        B_send_ptr = B_blk;  B_recv_ptr = B_recv_d;
    }
    #endif
    MPI_Sendrecv(
        A_send_ptr, A_m * A_send_k, MPI_DOUBLE, A_dst_rank, 0, 
        A_recv_ptr, A_m * A_recv_k, MPI_DOUBLE, A_src_rank, 0, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        B_send_ptr, B_send_k * B_n, MPI_DOUBLE, B_dst_rank, 1, 
        B_recv_ptr, B_recv_k * B_n, MPI_DOUBLE, B_src_rank, 1, comm, MPI_STATUS_IGNORE
    );
    #ifdef USE_CUDA
    if (dev_type == DEV_TYPE_CUDA)
    {
        hd_start_t = MPI_Wtime();
        dev_type_memcpy(A_recv_d, A_recv_h, sizeof(double) * A_m * A_recv_k, dev_type, DEV_TYPE_HOST);
        dev_type_memcpy(B_recv_d, B_recv_h, sizeof(double) * B_recv_k * B_n, dev_type, DEV_TYPE_HOST);
        hd_stop_t = MPI_Wtime();
        engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
    }
    #endif
    MPI_Barrier(comm);
}

void cannon_engine_exec_cc1(
    cannon_engine_p engine, const double alpha, const double beta, 
    double *A_blk, double *B_blk, double *C_blk
)
{
    LOAD_PARAMS;

    double exec_start_t, exec_stop_t, hd_start_t, hd_stop_t, start_t, stop_t;
    exec_start_t = MPI_Wtime();

    // Initial alignment
    start_t = MPI_Wtime();
    cannon_engine_init_alignment(engine, A_blk, B_blk);
    stop_t  = MPI_Wtime();
    engine->shift0_ms += 1000.0 * (stop_t - start_t);

    // {A, B}_blk is in {A, B}_recv_{d/h}, we can reuse A_blk and B_blk
    if (dev_type == DEV_TYPE_HOST)
    {
        A_gemm_h = A_blk;
        B_gemm_h = B_blk;
    }
    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        A_gemm_d = A_blk;
        B_gemm_d = B_blk;
    }
    #endif

    // Shift and multiply
    MPI_Request req_send_A, req_send_B, req_recv_A, req_recv_B;
    int local_k = k_displs[src_offset + 1] - k_displs[src_offset];
    double *tmp_ptr;
    for (int i_step = 0; i_step < np_dim; i_step++)
    {
        start_t = MPI_Wtime();
        if (i_step > 0)
        {
            MPI_Wait(&req_send_A, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send_B, MPI_STATUS_IGNORE);
            MPI_Wait(&req_recv_A, MPI_STATUS_IGNORE);
            MPI_Wait(&req_recv_B, MPI_STATUS_IGNORE);

            // DEV_TYPE_HOST: data on host buffer, will be used on host
            // DEV_TYPE_CUDA: data on host buffer, need to copy to CUDA buffer
            // DEV_TYPE_CUDA_MPI_DIRECT: data on CUDA buffer, will be used on GPU
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA)
            {
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(A_recv_d, A_recv_h, max_A_blk_bytes, dev_type, DEV_TYPE_HOST);
                dev_type_memcpy(B_recv_d, B_recv_h, max_B_blk_bytes, dev_type, DEV_TYPE_HOST);
                hd_stop_t = MPI_Wtime();
                engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
        }
        tmp_ptr = A_gemm_h; A_gemm_h = A_recv_h; A_recv_h = tmp_ptr;
        tmp_ptr = B_gemm_h; B_gemm_h = B_recv_h; B_recv_h = tmp_ptr;
        tmp_ptr = A_gemm_d; A_gemm_d = A_recv_d; A_recv_d = tmp_ptr;
        tmp_ptr = B_gemm_d; B_gemm_d = B_recv_d; B_recv_d = tmp_ptr;

        double *A_gemm_ptr, *B_gemm_ptr, *A_recv_ptr, *B_recv_ptr;
        if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
        {
            A_gemm_ptr = A_gemm_h;  A_recv_ptr = A_recv_h;
            B_gemm_ptr = B_gemm_h;  B_recv_ptr = B_recv_h;
        }
        #ifdef USE_CUDA
        if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
        {
            A_gemm_ptr = A_gemm_d;  A_recv_ptr = A_recv_d;
            B_gemm_ptr = B_gemm_d;  B_recv_ptr = B_recv_d;
        }
        #endif
        if (i_step < np_dim - 1)
        {
            MPI_Isend(A_gemm_ptr, max_A_blk_size, MPI_DOUBLE, left_rank,  i_step, comm, &req_send_A);
            MPI_Isend(B_gemm_ptr, max_B_blk_size, MPI_DOUBLE, upper_rank, i_step, comm, &req_send_B);
            MPI_Irecv(A_recv_ptr, max_A_blk_size, MPI_DOUBLE, right_rank, i_step, comm, &req_recv_A);
            MPI_Irecv(B_recv_ptr, max_B_blk_size, MPI_DOUBLE, lower_rank, i_step, comm, &req_recv_B);
        }
        stop_t  = MPI_Wtime();
        engine->lshift_ms += 1000.0 * (stop_t - start_t);

        start_t = MPI_Wtime();
        double beta_  = (i_step == 0) ? beta  : 1.0;
        double alpha_ = alpha;
        if (dev_type == DEV_TYPE_HOST)
        {
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A_m, B_n, local_k, 
                alpha_, A_gemm_h, A_m, B_gemm_h, local_k, beta_, C_blk, A_m
            );
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            cuda_cublas_dgemm(
                CublasNoTrans, CublasNoTrans, A_m, B_n, local_k, 
                alpha_, A_gemm_d, A_m, B_gemm_d, local_k, beta_, C_blk, A_m
            );
        }
        #endif
        src_offset = (src_offset + 1) % np_dim;
        local_k = k_displs[src_offset + 1] - k_displs[src_offset];
        stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
    }

    exec_stop_t = MPI_Wtime();
    engine->exec_ms += 1000.0 * (exec_stop_t - exec_start_t);
    engine->n_exec++;
}

void cannon_engine_exec_cck(
    cannon_engine_p engine, const double alpha, const double beta, 
    double *A_blk, double *B_blk, double *C_blk
)
{
    LOAD_PARAMS;

    double exec_start_t, exec_stop_t, hd_start_t, hd_stop_t, start_t, stop_t;
    exec_start_t = MPI_Wtime();

    // Initial alignment
    int k_stack_size = 0;
    start_t = MPI_Wtime();
    cannon_engine_init_alignment(engine, A_blk, B_blk);
    // dev_type_copy_mat_blk() works on row-major matrix, we have  
    // column-major matrix here, remember to swap row & column parameters
    double *A_stack_ptr, *B_stack_ptr;
    const double *A_recv_ptr, *B_recv_ptr;
    if (dev_type == DEV_TYPE_HOST)
    {
        A_stack_ptr = A_stack_h;  A_recv_ptr = A_recv_h;
        B_stack_ptr = B_stack_h;  B_recv_ptr = B_recv_h;
    }
    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        A_stack_ptr = A_stack_h;  A_recv_ptr = A_recv_d;
        B_stack_ptr = B_stack_h;  B_recv_ptr = B_recv_d;
    }
    #endif
    dev_type_copy_mat_blk(
        sizeof(double), A_recv_k, A_m, 
        A_recv_ptr, A_m, A_stack_ptr + k_stack_size * A_m, ldAs, dev_type
    );
    dev_type_copy_mat_blk(
        sizeof(double), B_n, B_recv_k, 
        B_recv_ptr, B_recv_k, B_stack_ptr + k_stack_size, ldBs, dev_type
    );
    k_stack_size += A_recv_k;
    stop_t  = MPI_Wtime();
    engine->shift0_ms += 1000.0 * (stop_t - start_t);

    // {A, B}_blk is in {A, B}_recv_{d/h}, we can reuse A_blk and B_blk
    if (dev_type == DEV_TYPE_HOST)
    {
        A_gemm_h = A_blk;
        B_gemm_h = B_blk;
    }
    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        A_gemm_d = A_blk;
        B_gemm_d = B_blk;
    }
    #endif

    // Shift and multiply
    MPI_Request req_send_A, req_send_B, req_recv_A, req_recv_B;
    int local_k = k_displs[src_offset + 1] - k_displs[src_offset];
    double *tmp_ptr;
    int gemm_step = 0;
    for (int i_step = 0; i_step < np_dim; i_step++)
    {
        start_t = MPI_Wtime();
        if (i_step > 0)
        {
            MPI_Wait(&req_send_A, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send_B, MPI_STATUS_IGNORE);
            MPI_Wait(&req_recv_A, MPI_STATUS_IGNORE);
            MPI_Wait(&req_recv_B, MPI_STATUS_IGNORE);

            // DEV_TYPE_HOST: data on host buffer, will be used on host
            // DEV_TYPE_CUDA: data on host buffer, need to copy to CUDA buffer
            // DEV_TYPE_CUDA_MPI_DIRECT: data on CUDA buffer, will be used on GPU
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA)
            {
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(A_recv_d, A_recv_h, max_A_blk_bytes, dev_type, DEV_TYPE_HOST);
                dev_type_memcpy(B_recv_d, B_recv_h, max_B_blk_bytes, dev_type, DEV_TYPE_HOST);
                hd_stop_t = MPI_Wtime();
                engine->hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif

            if (dev_type == DEV_TYPE_HOST)
            {
                A_recv_ptr = A_recv_h;
                B_recv_ptr = B_recv_h;
            }
            #ifdef USE_CUDA
            if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            {
                A_recv_ptr = A_recv_d;
                B_recv_ptr = B_recv_d;
            }
            #endif
            dev_type_copy_mat_blk(
                sizeof(double), local_k, A_m, 
                A_recv_ptr, A_m, A_stack_ptr + k_stack_size * A_m, ldAs, dev_type
            );
            dev_type_copy_mat_blk(
                sizeof(double), B_n, local_k, 
                B_recv_ptr, local_k, B_stack_ptr + k_stack_size, ldBs, dev_type
            );

            k_stack_size += local_k;
        }
        tmp_ptr = A_gemm_h; A_gemm_h = A_recv_h; A_recv_h = tmp_ptr;
        tmp_ptr = B_gemm_h; B_gemm_h = B_recv_h; B_recv_h = tmp_ptr;
        tmp_ptr = A_gemm_d; A_gemm_d = A_recv_d; A_recv_d = tmp_ptr;
        tmp_ptr = B_gemm_d; B_gemm_d = B_recv_d; B_recv_d = tmp_ptr;

        double *A_gemm_ptr, *B_gemm_ptr, *A_recv_ptr2, *B_recv_ptr2;
        if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
        {
            A_gemm_ptr = A_gemm_h;  A_recv_ptr2 = A_recv_h;
            B_gemm_ptr = B_gemm_h;  B_recv_ptr2 = B_recv_h;
        }
        #ifdef USE_CUDA
        if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
        {
            A_gemm_ptr = A_gemm_d;  A_recv_ptr2 = A_recv_d;
            B_gemm_ptr = B_gemm_d;  B_recv_ptr2 = B_recv_d;
        }
        #endif
        if (i_step < np_dim - 1)
        {
            MPI_Isend(A_gemm_ptr,  max_A_blk_size, MPI_DOUBLE, left_rank,  i_step, comm, &req_send_A);
            MPI_Isend(B_gemm_ptr,  max_B_blk_size, MPI_DOUBLE, upper_rank, i_step, comm, &req_send_B);
            MPI_Irecv(A_recv_ptr2, max_A_blk_size, MPI_DOUBLE, right_rank, i_step, comm, &req_recv_A);
            MPI_Irecv(B_recv_ptr2, max_B_blk_size, MPI_DOUBLE, lower_rank, i_step, comm, &req_recv_B);
        }
        stop_t  = MPI_Wtime();
        engine->lshift_ms += 1000.0 * (stop_t - start_t);

        start_t = MPI_Wtime();
        if ((i_step + 1) % gemm_cycle == 0)
        {
            double beta_  = (gemm_step == 0) ? beta  : 1.0;
            double alpha_ = alpha;
            if (dev_type == DEV_TYPE_HOST)
            {
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans, A_m, B_n, k_stack_size, 
                    alpha_, A_stack_h, ldAs, B_stack_h, ldBs, beta_, C_blk, A_m
                );
            }
            #ifdef USE_CUDA
            if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            {
                cuda_cublas_dgemm(
                    CublasNoTrans, CublasNoTrans, A_m, B_n, k_stack_size, 
                    alpha_, A_stack_d, ldAs, B_stack_d, ldBs, beta_, C_blk, A_m
                );
            }
            #endif
            gemm_step++;
            k_stack_size = 0;
        }
        src_offset = (src_offset + 1) % np_dim;
        local_k = k_displs[src_offset + 1] - k_displs[src_offset];
        stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
    }

    if (k_stack_size > 0)
    {
        start_t = MPI_Wtime();
        double beta_  = (gemm_step == 0) ? beta : 1.0;
        double alpha_ = alpha;
        if (dev_type == DEV_TYPE_HOST)
        {
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, A_m, B_n, k_stack_size, 
                alpha_, A_stack_h, ldAs, B_stack_h, ldBs, beta_, C_blk, A_m
            );
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            cuda_cublas_dgemm(
                CublasNoTrans, CublasNoTrans, A_m, B_n, k_stack_size, 
                alpha_, A_stack_d, ldAs, B_stack_d, ldBs, beta_, C_blk, A_m
            );
        }
        #endif
        gemm_step++;
        k_stack_size = 0;
        stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
    }

    exec_stop_t = MPI_Wtime();
    engine->exec_ms += 1000.0 * (exec_stop_t - exec_start_t);
    engine->n_exec++;
}

// Compute C := alpha * A * B + beta * C using 2D Cannon matrix multiplication algorithm
void cannon_engine_exec(
    cannon_engine_p engine, const double alpha, const double beta, 
    double *A_blk, double *B_blk, double *C_blk
)
{
    if (engine == NULL)
    {
        ERROR_PRINTF("cannon_engine not initialized\n");
        return;
    }

    int m = engine->m;
    int n = engine->n;
    int k = engine->k;
    
    if (m == 0 || n == 0 || k == 0) return;

    if (engine->np_dim == 1)
    {
        double start_t = MPI_Wtime();
        dev_type_t dev_type = engine->dev_type;
        if (dev_type == DEV_TYPE_HOST)
        {
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
                alpha, A_blk, m, B_blk, k, beta, C_blk, m
            );
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            cuda_cublas_dgemm(
                CublasNoTrans, CublasNoTrans, m, n, k, 
                alpha, A_blk, m, B_blk, k, beta, C_blk, m
            );
        }
        #endif
        double stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
        engine->exec_ms += 1000.0 * (stop_t - start_t);
        engine->n_exec++;
        return;
    }

    if (engine->gemm_cycle == 1)
    {
        cannon_engine_exec_cc1(engine, alpha, beta, A_blk, B_blk, C_blk);
    } else {
        cannon_engine_exec_cck(engine, alpha, beta, A_blk, B_blk, C_blk);
    }
}

// Reset the statistic data of a cannon_engine (not a collective call)
void cannon_engine_reset_stat(cannon_engine_p engine)
{
    if (engine == NULL) return;
    engine->shift0_ms   = 0.0;
    engine->lshift_ms   = 0.0;
    engine->gemm_ms     = 0.0;
    engine->hd_trans_ms = 0.0;
    engine->exec_ms     = 0.0;
    engine->n_exec      = 0;
}

// Print the statistic data of a cannon_engine (not a collective call)
void cannon_engine_print_stat(cannon_engine_p engine)
{
    if (engine == NULL) return;
    if (engine->n_exec == 0)
    {
        WARNING_PRINTF("No cannon_engine statistic data to print\n");
        return;
    }
    double GFlops = (double) engine->C_nrow * (double) engine->C_ncol * (double) engine->k;
    GFlops = GFlops * 2.0 * (double) engine->n_exec / engine->exec_ms * 1e3 / 1e9;
    printf("--------------- 2D Cannon algorithm engine ---------------\n");
    printf("* Initialization : %.2f ms\n", engine->init_ms);
    printf("* Number of executions  : %d\n", engine->n_exec);
    printf("* Execution time (avg)  : %.2f ms\n", engine->exec_ms   / engine->n_exec);
    printf("  * Initial shift       : %.2f ms\n", engine->shift0_ms / engine->n_exec);
    printf("  * Loop shift wait     : %.2f ms\n", engine->lshift_ms / engine->n_exec);
    printf("  * Local DGEMM         : %.2f ms\n", engine->gemm_ms   / engine->n_exec);
    if (engine->dev_type == DEV_TYPE_CUDA)
        printf("  * CUDA H <-> D memcpy : %.2f ms\n", engine->hd_trans_ms / engine->n_exec);
    printf("* Per-rank performance  : %.2f GFlops\n", GFlops);
    printf("----------------------------------------------------------\n");
}
