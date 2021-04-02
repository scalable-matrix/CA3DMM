#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "memory.h"
#include "partition.h"
#include "cannon.h"
#include "linalg_lib_wrapper.h"

void cannon_engine_init(const int m, const int n, const int k, MPI_Comm comm, cannon_engine_p *engine_)
{
    cannon_engine_init_ex(m,n,k,DEVICE_TYPE_HOST,DEVICE_TYPE_HOST,comm,engine_);
}

// Initialize a cannon_engine for 2D Cannon matrix multiplication algorithm
void cannon_engine_init_ex(const int m, const int n, const int k, device_type communication_device, device_type compute_device, MPI_Comm comm, cannon_engine_p *engine_)
{
    *engine_ = NULL;


    int my_rank, n_proc, np_dim;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_proc);
    np_dim = (int) sqrt((double) n_proc);
    if (np_dim * np_dim != n_proc)
    {
        fprintf(stderr, "[ERROR] Communicator size %d is not a square number\n", n_proc);
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
    engine->m         = m;
    engine->n         = n;
    engine->k         = k;
    engine->my_rank   = rank_cart;
    engine->rank_row  = rank_row;
    engine->rank_col  = rank_col;
    engine->np_dim    = np_dim;
    engine->comm      = comm_cart;
    engine->shift0_ms = 0.0;
    engine->lshift_ms = 0.0;
    engine->gemm_ms   = 0.0;
    engine->exec_ms   = 0.0;
    engine->n_exec    = 0;

    init_linalg_handle(&(engine->handle), compute_device);
    engine->communication_device  = communication_device;
    engine->compute_device  = compute_device;

    int *m_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    int *n_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    int *k_displs = (int *) malloc(sizeof(int) * (np_dim + 1));
    if (m_displs == NULL || n_displs == NULL || k_displs == NULL)
    {
        fprintf(stderr, "[ERROR] Failed to allocate cannon_engine displacement arrays\n");
        free(engine);
        return;
    }
    for (int i = 0; i <= np_dim; i++)
    {
        int dummy;
        calc_block_size_pos(m, np_dim, i, &dummy, m_displs + i);
        calc_block_size_pos(n, np_dim, i, &dummy, n_displs + i);
        calc_block_size_pos(k, np_dim, i, &dummy, k_displs + i);
    }
    engine->m_displs = m_displs;
    engine->n_displs = n_displs;
    engine->k_displs = k_displs;

    engine->A_srow = m_displs[rank_row];
    engine->A_scol = k_displs[rank_col];
    engine->A_nrow = m_displs[rank_row + 1] - m_displs[rank_row];
    engine->A_ncol = k_displs[rank_col + 1] - k_displs[rank_col];
    engine->B_srow = k_displs[rank_row];
    engine->B_scol = n_displs[rank_col];
    engine->B_nrow = k_displs[rank_row + 1] - k_displs[rank_row];
    engine->B_ncol = n_displs[rank_col + 1] - n_displs[rank_col];
    engine->C_srow = m_displs[rank_row];
    engine->C_scol = n_displs[rank_col];
    engine->C_nrow = m_displs[rank_row + 1] - m_displs[rank_row];
    engine->C_ncol = n_displs[rank_col + 1] - n_displs[rank_col];

    const int max_A_blk_size = (m / np_dim + 1) * (k / np_dim + 1);
    const int max_B_blk_size = (k / np_dim + 1) * (n / np_dim + 1);
    const int max_C_blk_size = (m / np_dim + 1) * (n / np_dim + 1);
    void *A_gemm = _OUR_MALLOC(sizeof(double) * max_A_blk_size, communication_device);
    void *A_recv = _OUR_MALLOC(sizeof(double) * max_A_blk_size, communication_device);
    void *B_gemm = _OUR_MALLOC(sizeof(double) * max_B_blk_size, communication_device);
    void *B_recv = _OUR_MALLOC(sizeof(double) * max_B_blk_size, communication_device);
    void *C_buff = _OUR_MALLOC(sizeof(double) * max_C_blk_size, communication_device);
    if ((A_gemm == NULL) || (A_recv == NULL) || (B_gemm == NULL) || (B_recv == NULL) || (C_buff == NULL))
    {
        fprintf(stderr, "[ERROR] Failed to allocate cannon_engine matrix buffers\n");
        free(engine);
        return;
    }
    engine->A_gemm = A_gemm;
    engine->A_recv = A_recv;
    engine->B_gemm = B_gemm;
    engine->B_recv = B_recv;
    engine->C_buff = C_buff;

    const int left_col   = (rank_col - 1 + np_dim) % np_dim;
    const int right_col  = (rank_col + 1) % np_dim;
    const int upper_row  = (rank_row - 1 + np_dim) % np_dim;
    const int lower_row  = (rank_row + 1) % np_dim;
    const int left_rank  = rank_row  * np_dim + left_col;
    const int right_rank = rank_row  * np_dim + right_col;
    const int lower_rank = lower_row * np_dim + rank_col;
    const int upper_rank = upper_row * np_dim + rank_col;
    MPI_Send_init(A_gemm, max_A_blk_size, MPI_DOUBLE, left_rank,  0, comm_cart, &engine->req_send_A[0]);
    MPI_Send_init(A_recv, max_A_blk_size, MPI_DOUBLE, left_rank,  1, comm_cart, &engine->req_send_A[1]);
    MPI_Send_init(B_gemm, max_B_blk_size, MPI_DOUBLE, upper_rank, 0, comm_cart, &engine->req_send_B[0]);
    MPI_Send_init(B_recv, max_B_blk_size, MPI_DOUBLE, upper_rank, 1, comm_cart, &engine->req_send_B[1]);
    MPI_Recv_init(A_recv, max_A_blk_size, MPI_DOUBLE, right_rank, 0, comm_cart, &engine->req_recv_A[0]);
    MPI_Recv_init(A_gemm, max_A_blk_size, MPI_DOUBLE, right_rank, 1, comm_cart, &engine->req_recv_A[1]);
    MPI_Recv_init(B_recv, max_B_blk_size, MPI_DOUBLE, lower_rank, 0, comm_cart, &engine->req_recv_B[0]);
    MPI_Recv_init(B_gemm, max_B_blk_size, MPI_DOUBLE, lower_rank, 1, comm_cart, &engine->req_recv_B[1]);

    int  min_k_blk_size  = 140;
    int  curr_k_blk_size = engine->A_ncol;
    int  gemm_cycle   = 1;
    void *A_stack = NULL, *B_stack = NULL;
    char *min_k_blk_size_p = getenv("CANNON_MIN_K_BLK_SIZE");
    if (min_k_blk_size_p != NULL) min_k_blk_size = atoi(min_k_blk_size_p);
    if (min_k_blk_size < 8) min_k_blk_size = 8;
    if (curr_k_blk_size < min_k_blk_size)
    {
        gemm_cycle = (min_k_blk_size + curr_k_blk_size - 1) / curr_k_blk_size;
        if (gemm_cycle > np_dim) gemm_cycle = np_dim;
        A_stack = _OUR_MALLOC(sizeof(double) * max_A_blk_size * gemm_cycle, communication_device);
        B_stack = _OUR_MALLOC(sizeof(double) * max_B_blk_size * gemm_cycle, communication_device);
    }
    engine->gemm_cycle = gemm_cycle;
    engine->A_stack = A_stack;
    engine->B_stack = B_stack;

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

static void copy_matrix_block(
    const size_t dt_size, const int nrow, const int ncol, 
    const void *src, const int lds, void *dst, const int ldd
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
        memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
    }
}

// Free a cannon_engine
void cannon_engine_free(cannon_engine_p *engine_)
{
    cannon_engine_p engine = *engine_;
    if (engine == NULL) return;
    free(engine->m_displs);
    free(engine->n_displs);
    free(engine->k_displs);
    OUR_FREE(engine->A_gemm, engine->communication_device);
    OUR_FREE(engine->A_recv, engine->communication_device);
    OUR_FREE(engine->A_stack, engine->communication_device);
    OUR_FREE(engine->B_gemm, engine->communication_device);
    OUR_FREE(engine->B_recv, engine->communication_device);
    OUR_FREE(engine->B_stack, engine->communication_device);
    OUR_FREE(engine->C_buff, engine->communication_device);
    MPI_Comm_free(&engine->comm);
    for (int i = 0; i < 1; i++)
    {
        MPI_Request_free(&engine->req_send_A[i]);
        MPI_Request_free(&engine->req_send_B[i]);
        MPI_Request_free(&engine->req_recv_A[i]);
        MPI_Request_free(&engine->req_recv_B[i]);
    }
    free(engine);
    *engine_ = NULL;
}

void cannon_engine_exec_cc1(
    const double alpha, const double *A_blk, const double *B_blk, 
    const double beta, double *C_blk, cannon_engine_p engine
)
{
    const int m         = engine->m;
    const int n         = engine->n;
    const int k         = engine->k;
    const int rank_row  = engine->rank_row;
    const int rank_col  = engine->rank_col;
    const int np_dim    = engine->np_dim;
    const int *m_displs = engine->m_displs;
    const int *n_displs = engine->n_displs;
    const int *k_displs = engine->k_displs;
    double *A_gemm = (double *) engine->A_gemm;
    double *A_recv = (double *) engine->A_recv;
    double *B_gemm = (double *) engine->B_gemm;
    double *B_recv = (double *) engine->B_recv;
    double *C_buff = (double *) engine->C_buff;
    MPI_Comm comm = engine->comm;

    double exec_start_t, exec_stop_t, start_t, stop_t;
    exec_start_t = MPI_Wtime();

    // Initial alignment
    int src_offset = (rank_row + rank_col) % np_dim;
    const int A_dst_col  = (rank_col - rank_row + np_dim) % np_dim;
    const int B_dst_row  = (rank_row - rank_col + np_dim) % np_dim;
    const int A_m        = m_displs[rank_row   + 1] - m_displs[rank_row];
    const int A_send_k   = k_displs[rank_col   + 1] - k_displs[rank_col];
    const int A_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];
    const int B_send_k   = k_displs[rank_row   + 1] - k_displs[rank_row];
    const int B_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];
    const int B_n        = n_displs[rank_col   + 1] - n_displs[rank_col];
    const int A_src_rank = rank_row   * np_dim + src_offset;
    const int B_src_rank = src_offset * np_dim + rank_col;
    const int A_dst_rank = rank_row   * np_dim + A_dst_col;
    const int B_dst_rank = B_dst_row  * np_dim + rank_col;
    start_t = MPI_Wtime();
    MPI_Sendrecv(
        A_blk,  A_m * A_send_k, MPI_DOUBLE, A_dst_rank, 0, 
        A_recv, A_m * A_recv_k, MPI_DOUBLE, A_src_rank, 0, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        B_blk,  B_send_k * B_n, MPI_DOUBLE, B_dst_rank, 1, 
        B_recv, B_recv_k * B_n, MPI_DOUBLE, B_src_rank, 1, comm, MPI_STATUS_IGNORE
    );
    MPI_Barrier(comm);

#if USE_GPU
    if(engine->communication_device == DEVICE_TYPE_DEVICE) {
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#endif
    stop_t  = MPI_Wtime();
    engine->shift0_ms += 1000.0 * (stop_t - start_t);

    // Shift and multiply
    MPI_Request *req_send_A_p, *req_send_B_p, *req_recv_A_p, *req_recv_B_p;
    int local_k = k_displs[src_offset + 1] - k_displs[src_offset];
    double *tmp_ptr;
    for (int i_step = 0; i_step < np_dim; i_step++)
    {
        start_t = MPI_Wtime();
        if (i_step > 0)
        {
            MPI_Wait(req_send_A_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_send_B_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_recv_A_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_recv_B_p, MPI_STATUS_IGNORE);
#if USE_GPU
            if(engine->communication_device == DEVICE_TYPE_DEVICE) {
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
            }
#endif
        }
        tmp_ptr = A_gemm; A_gemm = A_recv; A_recv = tmp_ptr;
        tmp_ptr = B_gemm; B_gemm = B_recv; B_recv = tmp_ptr;

        if (i_step < np_dim - 1)
        {
            req_send_A_p = &engine->req_send_A[(i_step + 1) % 2];
            req_send_B_p = &engine->req_send_B[(i_step + 1) % 2];
            req_recv_A_p = &engine->req_recv_A[(i_step + 1) % 2];
            req_recv_B_p = &engine->req_recv_B[(i_step + 1) % 2];
            MPI_Start(req_send_A_p);
            MPI_Start(req_send_B_p);
            MPI_Start(req_recv_A_p);
            MPI_Start(req_recv_B_p);
        }
        stop_t  = MPI_Wtime();
        engine->lshift_ms += 1000.0 * (stop_t - start_t);

        start_t = MPI_Wtime();
        double beta  = (i_step == 0) ? 0.0 : 1.0;
        double alpha = 1.0;
        local_AB(engine->handle,
            A_m, B_n, local_k, 
            alpha, A_gemm, A_m, B_gemm, local_k, beta, C_buff, A_m,
            engine->communication_device, engine->compute_device
        );
        src_offset = (src_offset + 1) % np_dim;
        local_k = k_displs[src_offset + 1] - k_displs[src_offset];
        stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
    }

    // Accumulate to final output
    for (int i = 0; i < A_m * B_n; i++)
        C_blk[i] = alpha * C_buff[i] + beta * C_blk[i];

    exec_stop_t = MPI_Wtime();
    engine->exec_ms += 1000.0 * (exec_stop_t - exec_start_t);
    engine->n_exec++;
}

void cannon_engine_exec_cck(
    const double alpha, const double *A_blk, const double *B_blk, 
    const double beta, double *C_blk, cannon_engine_p engine
)
{
    const int m          = engine->m;
    const int n          = engine->n;
    const int k          = engine->k;
    const int rank_row   = engine->rank_row;
    const int rank_col   = engine->rank_col;
    const int np_dim     = engine->np_dim;
    const int gemm_cycle = engine->gemm_cycle;
    const int *m_displs  = engine->m_displs;
    const int *n_displs  = engine->n_displs;
    const int *k_displs  = engine->k_displs;
    double *A_gemm  = (double *) engine->A_gemm;
    double *A_recv  = (double *) engine->A_recv;
    double *A_stack = (double *) engine->A_stack;
    double *B_gemm  = (double *) engine->B_gemm;
    double *B_recv  = (double *) engine->B_recv;
    double *B_stack = (double *) engine->B_stack;
    double *C_buff  = (double *) engine->C_buff;
    MPI_Comm comm = engine->comm;

    double exec_start_t, exec_stop_t, start_t, stop_t;
    exec_start_t = MPI_Wtime();

    // Initial alignment
    int src_offset = (rank_row + rank_col) % np_dim;
    const int A_dst_col  = (rank_col - rank_row + np_dim) % np_dim;
    const int B_dst_row  = (rank_row - rank_col + np_dim) % np_dim;
    const int A_m        = m_displs[rank_row   + 1] - m_displs[rank_row];
    const int A_send_k   = k_displs[rank_col   + 1] - k_displs[rank_col];
    const int A_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];
    const int B_send_k   = k_displs[rank_row   + 1] - k_displs[rank_row];
    const int B_recv_k   = k_displs[src_offset + 1] - k_displs[src_offset];
    const int B_n        = n_displs[rank_col   + 1] - n_displs[rank_col];
    const int A_src_rank = rank_row   * np_dim + src_offset;
    const int B_src_rank = src_offset * np_dim + rank_col;
    const int A_dst_rank = rank_row   * np_dim + A_dst_col;
    const int B_dst_rank = B_dst_row  * np_dim + rank_col;
    const int ldAs       = A_m;
    const int ldBs       = (k / np_dim + 1) * gemm_cycle;
    int k_stack_size = 0;
    start_t = MPI_Wtime();
    MPI_Sendrecv(
        A_blk,  A_m * A_send_k, MPI_DOUBLE, A_dst_rank, 0, 
        A_recv, A_m * A_recv_k, MPI_DOUBLE, A_src_rank, 0, comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        B_blk,  B_send_k * B_n, MPI_DOUBLE, B_dst_rank, 1, 
        B_recv, B_recv_k * B_n, MPI_DOUBLE, B_src_rank, 1, comm, MPI_STATUS_IGNORE
    );
    copy_matrix_block(
        sizeof(double), A_recv_k, A_m, 
        A_recv, A_m, A_stack + k_stack_size * A_m, ldAs
    );
    copy_matrix_block(
        sizeof(double), B_n, B_recv_k, 
        B_recv, B_recv_k, B_stack + k_stack_size, ldBs
    );
    k_stack_size += A_recv_k;
    MPI_Barrier(comm);
    stop_t  = MPI_Wtime();
    engine->shift0_ms += 1000.0 * (stop_t - start_t);

    // Shift and multiply
    MPI_Request *req_send_A_p, *req_send_B_p, *req_recv_A_p, *req_recv_B_p;
    int local_k = k_displs[src_offset + 1] - k_displs[src_offset];
    double *tmp_ptr;
    int gemm_step = 0;
    for (int i_step = 0; i_step < np_dim; i_step++)
    {
        start_t = MPI_Wtime();
        if (i_step > 0)
        {
            MPI_Wait(req_send_A_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_send_B_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_recv_A_p, MPI_STATUS_IGNORE);
            MPI_Wait(req_recv_B_p, MPI_STATUS_IGNORE);
#if USE_GPU
            if(engine->communication_device == DEVICE_TYPE_DEVICE) {
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
            }
#endif
            copy_matrix_block(
                sizeof(double), local_k, A_m, 
                A_recv, A_m, A_stack + k_stack_size * A_m, ldAs
            );
            copy_matrix_block(
                sizeof(double), B_n, local_k, 
                B_recv, local_k, B_stack + k_stack_size, ldBs
            );
            k_stack_size += local_k;
        }
        tmp_ptr = A_gemm; A_gemm = A_recv; A_recv = tmp_ptr;
        tmp_ptr = B_gemm; B_gemm = B_recv; B_recv = tmp_ptr;

        if (i_step < np_dim - 1)
        {
            req_send_A_p = &engine->req_send_A[(i_step + 1) % 2];
            req_send_B_p = &engine->req_send_B[(i_step + 1) % 2];
            req_recv_A_p = &engine->req_recv_A[(i_step + 1) % 2];
            req_recv_B_p = &engine->req_recv_B[(i_step + 1) % 2];
            MPI_Start(req_send_A_p);
            MPI_Start(req_send_B_p);
            MPI_Start(req_recv_A_p);
            MPI_Start(req_recv_B_p);
        }
        stop_t  = MPI_Wtime();
        engine->lshift_ms += 1000.0 * (stop_t - start_t);

        start_t = MPI_Wtime();
        if ((i_step + 1) % gemm_cycle == 0)
        {
            double beta  = (gemm_step == 0) ? 0.0 : 1.0;
            double alpha = 1.0;
            local_AB(engine->handle,
                A_m, B_n, k_stack_size, 
                alpha, A_stack, ldAs, B_stack, ldBs, beta, C_buff, A_m,
            engine->communication_device, engine->compute_device
            );
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
        double beta  = (gemm_step == 0) ? 0.0 : 1.0;
        double alpha = 1.0;
        local_AB(engine->handle,
            A_m, B_n, k_stack_size, 
            alpha, A_stack, ldAs, B_stack, ldBs, beta, C_buff, A_m,
            engine->communication_device, engine->compute_device
        );
        gemm_step++;
        k_stack_size = 0;
    }

    // Accumulate to final output
    for (int i = 0; i < A_m * B_n; i++)
        C_blk[i] = alpha * C_buff[i] + beta * C_blk[i];

    exec_stop_t = MPI_Wtime();
    engine->exec_ms += 1000.0 * (exec_stop_t - exec_start_t);
    engine->n_exec++;
}

// Compute C := alpha * A * B + beta * C using 2D Cannon matrix multiplication algorithm
void cannon_engine_exec(
    const double alpha, const double *A_blk, const double *B_blk, 
    const double beta, double *C_blk, cannon_engine_p engine
)
{
    if (engine == NULL)
    {
        fprintf(stderr, "[ERROR] canon_engine not initialized\n");
        return;
    }

    const int m = engine->m;
    const int n = engine->n;
    const int k = engine->k;
    
    if (m == 0 || n == 0 || k == 0) return;

    if (engine->np_dim == 1)
    {
        double start_t = MPI_Wtime();
        local_AB(engine->handle,
            m, n, k, 
            alpha, A_blk, m, B_blk, k, beta, C_blk, m,
            engine->communication_device, engine->compute_device
        );
        double stop_t  = MPI_Wtime();
        engine->gemm_ms += 1000.0 * (stop_t - start_t);
        engine->exec_ms += 1000.0 * (stop_t - start_t);
        engine->n_exec++;
        return;
    }

    if (engine->gemm_cycle == 1)
    {
        cannon_engine_exec_cc1(alpha, A_blk, B_blk, beta, C_blk, engine);
    } else {
        cannon_engine_exec_cck(alpha, A_blk, B_blk, beta, C_blk, engine);
    }
}

// Reset the statistic data of a cannon_engine (not a collective call)
void cannon_engine_reset_stat(cannon_engine_p engine)
{
    if (engine == NULL) return;
    engine->shift0_ms = 0.0;
    engine->lshift_ms = 0.0;
    engine->gemm_ms   = 0.0;
    engine->exec_ms   = 0.0;
    engine->n_exec    = 0;
}

// Print the statistic data of a cannon_engine (not a collective call)
void cannon_engine_print_stat(cannon_engine_p engine)
{
    if (engine == NULL) return;
    if (engine->n_exec == 0)
    {
        printf("No cannon_engine statistic data to print\n");
        return;
    }
    double GFlops = (double) engine->C_nrow * (double) engine->C_ncol * (double) engine->k;
    GFlops = GFlops * 2.0 * (double) engine->n_exec / engine->exec_ms * 1e3 / 1e9;
    printf("--------------- 2D Cannon algorithm engine ---------------\n");
    printf("* Initialization       : %.2f ms\n", engine->init_ms);
    printf("* Number of executions : %d\n", engine->n_exec);
    printf("* Execution time (avg) : %.2f ms\n", engine->exec_ms   / engine->n_exec);
    printf("  * Initial shift      : %.2f ms\n", engine->shift0_ms / engine->n_exec);
    printf("  * Loop shift wait    : %.2f ms\n", engine->lshift_ms / engine->n_exec);
    printf("  * Local DGEMM        : %.2f ms\n", engine->gemm_ms   / engine->n_exec);
    printf("* Per-rank performance : %.2f GFlops\n", GFlops);
    printf("----------------------------------------------------------\n");
}
