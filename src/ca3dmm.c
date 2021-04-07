#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#if USE_GPU
#include "gpu.h"
#endif

#include "memory.h"
#include "partition.h"
#include "cannon.h"
#include "ca3dmm.h"
#include "mat_redist.h"

static inline void swap_int(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline void transpose_cm_mat(
    const int A_nrow, const int A_ncol, const double *A, const int ldA,
    double *A_trans, const int ldAT, linalg_handle_t handle, device_type dev
)
{
//printf("TRANSPOSE\n");
#if USE_GPU
if(dev == DEVICE_TYPE_DEVICE) {
//printf("ON GPU\n");
    gpu_transpose(A_nrow, A_ncol, A, ldA, A_trans, ldAT, handle, dev);
} else {
#endif
    // TODO: use multithreading if necessary
    for (int j = 0; j < A_ncol; j++)
    {
        for (int i = 0; i < A_nrow; i++)
        {
            int idx0 = i * ldA  + j;
            int idx1 = j * ldAT + i;
            A_trans[idx1] = A[idx0];
        }
    }
#if USE_GPU
}
#endif
}

void ca3dmm_engine_init(
    const int m, const int n, const int k, const int trans_A, const int trans_B, 
    const int src_A_srow, const int src_A_nrow,
    const int src_A_scol, const int src_A_ncol,
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, ca3dmm_engine_p *engine_
) {
 ca3dmm_engine_init_ex(m,n,k,trans_A,trans_B,
     src_A_srow,  src_A_nrow,
     src_A_scol,  src_A_ncol,
     src_B_srow,  src_B_nrow,
     src_B_scol,  src_B_ncol,
     dst_C_srow,  dst_C_nrow,
     dst_C_scol,  dst_C_ncol,
     DEVICE_TYPE_HOST, DEVICE_TYPE_HOST,
     proc_grid, comm, engine_);
}


// Initialize a camm3d_engine structure for C := op(A) * op(B)
void ca3dmm_engine_init_ex(
    const int m, const int n, const int k, const int trans_A, const int trans_B, 
    const int src_A_srow, const int src_A_nrow,
    const int src_A_scol, const int src_A_ncol,
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    device_type communication_device, device_type compute_device,
    const int *proc_grid, MPI_Comm comm, ca3dmm_engine_p *engine_
)
{
    *engine_ = NULL;
    ca3dmm_engine_p engine = (ca3dmm_engine_p) malloc(sizeof(ca3dmm_engine_s));
    memset(engine, 0, sizeof(ca3dmm_engine_s));

    double start_t = MPI_Wtime();

    engine->communication_device = communication_device;
    engine->compute_device = compute_device;
    init_linalg_handle(&(engine->handle), compute_device);
    //printf("Egine dev: %i, %i\n", engine->communication_device, engine->compute_device);

    // 1. Get the basic process grid
    engine->m = m;
    engine->n = n;
    engine->k = k;
    MPI_Comm_size(comm, &engine->n_proc);
    MPI_Comm_rank(comm, &engine->my_rank);
    int p = engine->n_proc;
    int mp, np, kp, rp, gen_proc_grid = 1;
    if (proc_grid != NULL)
    {
        mp = proc_grid[0];
        np = proc_grid[1];
        kp = proc_grid[2];
        rp = engine->n_proc - mp * np * kp;
        int max_mp_np = (mp > np) ? mp : np;
        int min_mp_np = (mp < np) ? mp : np;
        if ((max_mp_np % min_mp_np) || (rp < 0))
        {
            if (engine->my_rank == 0) printf("[WARNING] Invalid process grid specified: mp, np, kp = %d, %d, %d\n", mp, np, kp);
        } else {
            gen_proc_grid = 0;
        }
    }
    if (gen_proc_grid) calc_3d_decomposition(p, m, n, k, &mp, &np, &kp, &rp);
    if ((mp < 1) || (np < 1) || (kp < 1) || (mp * np * kp > p))
    {
        if (engine->my_rank == 0) 
        {
            fprintf(stderr, "[ERROR] Invalid process grid generated: p = %d, mp, np, kp = %d, %d, %d\n", p, mp, np, kp);
            fflush(stderr);
        }
        ca3dmm_engine_free(&engine);
        return;
    }
    engine->mp = mp;
    engine->np = np;
    engine->kp = kp;
    engine->rp = rp;
    engine->is_active = (engine->my_rank < mp * np * kp) ? 1 : 0;
    engine->is_BTB    = 0;
    engine->redist_ms = 0.0;
    engine->agvAB_ms  = 0.0;
    engine->cannon_ms = 0.0;
    engine->reduce_ms = 0.0;
    engine->n_exec    = 0;

    // 2. Handle the task groups, note that max(mp, np) is a multiplier of min(mp, np)
    int rank_k  = engine->my_rank / (mp * np);
    int rank_mn = engine->my_rank % (mp * np);
    int rank_n  = rank_mn / mp;
    int rank_m  = rank_mn % mp; 
    int task_m_num  = 1,  task_m_id = 0;
    int task_n_num  = 1,  task_n_id = 0;
    int task_k_num  = kp, task_k_id = rank_k;
    int task_k_nproc = 1;
    if (engine->is_active)
    {
        task_k_nproc = mp * np;
        if (mp > np)
        {
            task_m_num   = mp / np;
            task_m_id    = rank_mn / (np * np);
            task_k_nproc = np * np;
        }
        if (np > mp)
        {
            task_n_num   = np / mp;
            task_n_id    = rank_mn / (mp * mp);
            task_k_nproc = mp * mp;
        }
    } else {
        rank_m = 0; rank_n  = 0;
        rank_k = 0; rank_mn = 0;
        task_m_num = 0;  task_m_id = 0;
        task_n_num = 0;  task_n_id = 0;
        task_k_num = 0;  task_k_id = 0;
    }  // End of "if (engine->is_active)"
    engine->rank_m     = rank_m;
    engine->rank_n     = rank_n;
    engine->rank_k     = rank_k;
    engine->task_m_num = task_m_num;
    engine->task_m_id  = task_m_id;
    engine->task_n_num = task_n_num;
    engine->task_n_id  = task_n_id;
    engine->task_k_num = task_k_num;
    engine->task_k_id  = task_k_id;

    // 3. Set up the communicators
    // In the process grid, each mn-plane has (task_m_num * task_n_num) 2D Cannon tasks.
    // (1) If n >= 2 * m, A block is a near square block but B block is a short-fat block, A block needs to be
    // duplicated task_n_num > 1 times, each task_k_nproc == mp * mp processes holds a copy of the A block.
    // (2) If m >= 2 * n, A block is a tall-skinny block but B block is a near square block, B block needs to be
    // duplicated task_m_num > 1 times, each task_k_nproc == np * np processes holds a copy of the B block.
    // All processes in comm_2dmm and comm_AB_agv are in the same mn-plane.
    int color_2dmm, color_C_rs, color_AB_agv;
    int task_mn_num = task_m_num * task_n_num;
    int task_mn_id  = task_m_id  + task_n_id;   // task_m_id and task_n_id cannot > 0 at the same time
    if (engine->is_active == 1)
    {
        color_2dmm   = rank_k * task_mn_num + task_mn_id;
        color_C_rs   = rank_mn;
        color_AB_agv = rank_k * task_k_nproc + rank_mn % task_k_nproc;
    } else {
        // 19241112 should be large enough
        color_2dmm   = 19241112;
        color_C_rs   = 19241112;
        color_AB_agv = 19241112;
    }
    MPI_Comm_split(comm, color_2dmm,   engine->my_rank, &engine->comm_2dmm);
    MPI_Comm_split(comm, color_C_rs,   engine->my_rank, &engine->comm_C_rs);
    MPI_Comm_split(comm, color_AB_agv, engine->my_rank, &engine->comm_AB_agv);

    // 4. Calculate A, B, C block information
    if (engine->is_active)
    {
        int task_m_spos, task_m_size;
        int task_n_spos, task_n_size;
        int task_k_spos, task_k_size;
        calc_block_size_pos(m, task_m_num, task_m_id, &task_m_size, &task_m_spos);
        calc_block_size_pos(n, task_n_num, task_n_id, &task_n_size, &task_n_spos);
        calc_block_size_pos(k, task_k_num, task_k_id, &task_k_size, &task_k_spos);
        cannon_engine_init_ex(task_m_size, task_n_size, task_k_size, engine->communication_device, engine->compute_device, engine->comm_2dmm, &engine->cannon_engine);
        cannon_engine_p ce = engine->cannon_engine;
        if (ce == NULL)
        {
            ca3dmm_engine_free(&engine);
            return;
        }
        engine->A_2dmm_srow = task_m_spos + ce->A_srow;
        engine->A_2dmm_scol = task_k_spos + ce->A_scol;
        engine->A_2dmm_nrow = ce->A_nrow;
        engine->A_2dmm_ncol = ce->A_ncol;
        engine->B_2dmm_srow = task_k_spos + ce->B_srow;
        engine->B_2dmm_scol = task_n_spos + ce->B_scol;
        engine->B_2dmm_nrow = ce->B_nrow;
        engine->B_2dmm_ncol = ce->B_ncol;
        engine->C_2dmm_srow = task_m_spos + ce->C_srow;
        engine->C_2dmm_scol = task_n_spos + ce->C_scol;
        engine->C_2dmm_nrow = ce->C_nrow;
        engine->C_2dmm_ncol = ce->C_ncol;
        engine->C_out_srow  = engine->C_2dmm_srow;
        engine->C_out_nrow  = engine->C_2dmm_nrow;
        int C_out_scol, C_out_ncol, use_rsb = 1, C_out_ncol0 = -1;
        int *C_rs_recvcnts = (int *) malloc(sizeof(int) * task_k_num);
        for (int i = 0; i < task_k_num; i++)
        {
            calc_block_size_pos(engine->C_2dmm_ncol, task_k_num, i, &C_out_ncol, &C_out_scol);
            if (C_out_ncol0 == -1) C_out_ncol0 = C_out_ncol;
            if (C_out_ncol0 != C_out_ncol) use_rsb = 0;
            C_rs_recvcnts[i] = C_out_ncol * engine->C_2dmm_nrow;
            if (i == task_k_id)
            {
                engine->C_out_scol = engine->C_2dmm_scol + C_out_scol;
                engine->C_out_ncol = C_out_ncol;
            }
        }
        engine->C_rs_recvcnts = C_rs_recvcnts;
        engine->use_rsb = use_rsb;
    } else {
        engine->A_2dmm_srow = 0;
        engine->A_2dmm_scol = 0;
        engine->A_2dmm_nrow = 0;
        engine->A_2dmm_ncol = 0;
        engine->B_2dmm_srow = 0;
        engine->B_2dmm_scol = 0;
        engine->B_2dmm_nrow = 0;
        engine->B_2dmm_ncol = 0;
        engine->C_2dmm_srow = 0;
        engine->C_2dmm_scol = 0;
        engine->C_2dmm_nrow = 0;
        engine->C_2dmm_ncol = 0;
        engine->C_out_srow  = 0;
        engine->C_out_scol  = 0;
        engine->C_out_nrow  = 0;
        engine->C_out_ncol  = 0;
        engine->C_rs_recvcnts = NULL;
    }  // End of "if (engine->is_active)"

    // 5. Set up mat_redist_engine
    // (1) src_{A, B}_{s, n}{row, col} describes the source blocks of A & B, not op(A) & op(B),
    //     so we do not need to swap them if A and/or B need to be transposed.
    // (2) engine->{A, B}_rd_{s, n}{row, col} describes the required blocks of op(A) and op(B)
    //     in redistribution, so the row & col of the actual required source blocks need to be
    //     swapped if op == transpose. 
    // (3) The input A and B matrices are column-major, mat_redist_engine uses row-major, so 
    //     we need to swap the parameters when calling mat_redist_engine_init().
    int A_rd_srow = engine->A_2dmm_srow;
    int A_rd_scol = engine->A_2dmm_scol;
    int A_rd_nrow = engine->A_2dmm_nrow;
    int A_rd_ncol = engine->A_2dmm_ncol;
    int B_rd_srow = engine->B_2dmm_srow;
    int B_rd_scol = engine->B_2dmm_scol;
    int B_rd_nrow = engine->B_2dmm_nrow;
    int B_rd_ncol = engine->B_2dmm_ncol;
    int *AB_agv_recvcnts = (int *) malloc(sizeof(int) * task_mn_num);
    int *AB_agv_displs   = (int *) malloc(sizeof(int) * (task_mn_num + 1));
    memset(AB_agv_displs, 0, sizeof(int) * task_mn_num);
    if (task_n_num > 1)  // A block need to be duplicated
    {
        int scol, ncol, use_ag = 1, ncol0 = -1;
        for (int i = 0; i < task_n_num; i++)
        {
            calc_block_size_pos(engine->A_2dmm_ncol, task_n_num, i, &ncol, &scol);
            if (ncol0 == -1) ncol0 = ncol;
            if (ncol0 != ncol) use_ag = 0;
            AB_agv_recvcnts[i] = ncol * engine->A_2dmm_nrow;
            AB_agv_displs[i + 1] = AB_agv_displs[i] + AB_agv_recvcnts[i];
            if (i == task_n_id)
            {
                A_rd_scol = engine->A_2dmm_scol + scol;
                A_rd_ncol = ncol;
            }
        }
        engine->use_ag = use_ag;
    }
    if (task_m_num > 1)  // B block need to be duplicated
    {
        int scol, ncol, use_ag = 1, ncol0 = -1;
        for (int i = 0; i < task_m_num; i++)
        {
            calc_block_size_pos(engine->B_2dmm_ncol, task_m_num, i, &ncol, &scol);
            if (ncol0 == -1) ncol0 = ncol;
            if (ncol0 != ncol) use_ag = 0;
            AB_agv_recvcnts[i] = ncol * engine->B_2dmm_nrow;
            AB_agv_displs[i + 1] = AB_agv_displs[i] + AB_agv_recvcnts[i];
            if (i == task_m_id)
            {
                B_rd_scol = engine->B_2dmm_scol + scol;
                B_rd_ncol = ncol;
            }
        }
        engine->use_ag = use_ag;
    }
    engine->A_rd_srow = A_rd_srow;
    engine->A_rd_scol = A_rd_scol;
    engine->A_rd_nrow = A_rd_nrow;
    engine->A_rd_ncol = A_rd_ncol;
    engine->B_rd_srow = B_rd_srow;
    engine->B_rd_scol = B_rd_scol;
    engine->B_rd_nrow = B_rd_nrow;
    engine->B_rd_ncol = B_rd_ncol;
    engine->trans_A   = trans_A;
    engine->trans_B   = trans_B;
    if (trans_A)
    {
        swap_int(&A_rd_srow, &A_rd_scol);
        swap_int(&A_rd_nrow, &A_rd_ncol);
    }
    if (trans_B)
    {
        swap_int(&B_rd_srow, &B_rd_scol);
        swap_int(&B_rd_nrow, &B_rd_ncol);
    }
    // Non-active processes still need to participate in initial A & B redistribution,
    // but their {A, B}_rd_{s, n}{row, col} == 0.
    mat_redist_engine_init_ex(
        src_A_scol, src_A_srow, src_A_ncol, src_A_nrow, 
        A_rd_scol,  A_rd_srow,  A_rd_ncol,  A_rd_nrow, 
        engine->communication_device,
        comm, MPI_DOUBLE, sizeof(double), &engine->redist_A
    );
    mat_redist_engine_init_ex(
        src_B_scol, src_B_srow, src_B_ncol, src_B_nrow, 
        B_rd_scol,  B_rd_srow,  B_rd_ncol,  B_rd_nrow, 
        engine->communication_device,
        comm, MPI_DOUBLE, sizeof(double), &engine->redist_B
    );
    if ((engine->redist_A == NULL) || (engine->redist_B == NULL))
    {
        ca3dmm_engine_free(&engine);
        return;
    }
    engine->AB_agv_recvcnts = AB_agv_recvcnts;
    engine->AB_agv_displs   = AB_agv_displs;
    if (!((dst_C_srow == -1) || (dst_C_nrow == -1) || (dst_C_scol == -1) || (dst_C_ncol == -1)))
    {
        mat_redist_engine_init_ex(
            engine->C_out_scol, engine->C_out_srow, engine->C_out_ncol, engine->C_out_nrow, 
            dst_C_scol, dst_C_srow, dst_C_ncol, dst_C_nrow, 
            engine->communication_device,
            comm, MPI_DOUBLE, sizeof(double), &engine->redist_C
        );
        if (engine->redist_C == NULL)
        {
            ca3dmm_engine_free(&engine);
            return;
        }
    }

    // 6. Allocate local matrix blocks
    void *A_rd_recv = NULL, *A_2dmm = NULL, *A_trans = NULL;
    void *B_rd_recv = NULL, *B_2dmm = NULL, *B_trans = NULL;
    void *C_2dmm = NULL, *C_out = NULL;
    if (engine->is_active)
    {
        A_rd_recv = _OUR_MALLOC(sizeof(double) * engine->A_rd_nrow   * engine->A_rd_ncol, engine->communication_device);
        B_rd_recv = _OUR_MALLOC(sizeof(double) * engine->B_rd_nrow   * engine->B_rd_ncol, engine->communication_device);
        A_trans   = (trans_A == 1)   ? _OUR_MALLOC(sizeof(double) * engine->A_2dmm_nrow * engine->A_2dmm_ncol, engine->communication_device) : A_rd_recv;
        B_trans   = (trans_B == 1)   ? _OUR_MALLOC(sizeof(double) * engine->B_2dmm_nrow * engine->B_2dmm_ncol, engine->communication_device) : B_rd_recv;
        A_2dmm    = (task_n_num > 1) ? _OUR_MALLOC(sizeof(double) * engine->A_2dmm_nrow * engine->A_2dmm_ncol, engine->communication_device) : A_trans;
        B_2dmm    = (task_m_num > 1) ? _OUR_MALLOC(sizeof(double) * engine->B_2dmm_nrow * engine->B_2dmm_ncol, engine->communication_device) : B_trans;
        C_2dmm    = _OUR_MALLOC(sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol, engine->communication_device);
        C_out     = _OUR_MALLOC(sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol, engine->communication_device);

        //printf("ALLOCED LOTS on def: %i\n", engine->communication_device);
        //gpu_print_mat(A_rd_recv, 1, 1);
        //gpu_print_mat(B_rd_recv, 1, 1);
        //gpu_print_mat(A_trans, 1, 1);
        //gpu_print_mat(B_trans, 1, 1);
        //gpu_print_mat(A_2dmm, 1, 1);
        //gpu_print_mat(B_2dmm, 1, 1);
        if ((A_rd_recv == NULL) || (A_trans == NULL) || (A_2dmm == NULL) ||
            (B_rd_recv == NULL) || (B_trans == NULL) || (B_2dmm == NULL) ||
            (C_2dmm == NULL) || (C_out  == NULL))
        {
            fprintf(stderr, "[ERROR] Failed to allocate ca3dmm_engine matrix buffers\n");
            ca3dmm_engine_free(&engine);
            return;
        }
    }
    engine->A_rd_recv = A_rd_recv;
    engine->A_2dmm    = A_2dmm;
    engine->A_trans   = A_trans;
    engine->B_rd_recv = B_rd_recv;
    engine->B_2dmm    = B_2dmm;
    engine->B_trans   = B_trans;
    engine->C_2dmm    = C_2dmm;
    engine->C_out     = C_out;

    char *print_timing_p = getenv("CA3DMM_PRINT_TIMING");
    int print_timing = 0 ;
    if (print_timing_p != NULL) print_timing = atoi(print_timing_p);
    if (engine->my_rank == 0 && print_timing == 1) engine->print_timing = 1;
    else engine->print_timing = 0;

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

// Initialize a camm3d_engine structure for C := B^T * B
void ca3dmm_engine_init_BTB(
    const int n, const int k, 
    const int src_B_srow, const int src_B_nrow, 
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, ca3dmm_engine_p *engine_
)
{
    ca3dmm_engine_init_BTB_ex(n,k,src_B_srow,src_B_nrow,src_B_scol,src_B_ncol,
    dst_C_srow,dst_C_nrow,dst_C_scol,dst_C_ncol,DEVICE_TYPE_HOST, DEVICE_TYPE_HOST,
    proc_grid,comm,engine_);
}

void ca3dmm_engine_init_BTB_ex(
    const int n, const int k, 
    const int src_B_srow, const int src_B_nrow, 
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    device_type communication_device, device_type compute_device,
    const int *proc_grid, MPI_Comm comm, ca3dmm_engine_p *engine_
)
{
    *engine_ = NULL;
    ca3dmm_engine_p engine = (ca3dmm_engine_p) malloc(sizeof(ca3dmm_engine_s));
    memset(engine, 0, sizeof(ca3dmm_engine_s));

    double start_t = MPI_Wtime();

    engine->communication_device = communication_device;
    engine->compute_device = compute_device;
    init_linalg_handle(&(engine->handle), compute_device);
    //printf("Egine dev: %i, %i\n", engine->communication_device, engine->compute_device);

    // 1. Get the basic process grid
    int m = n;
    engine->m = n;
    engine->n = n;
    engine->k = k;
    MPI_Comm_size(comm, &engine->n_proc);
    MPI_Comm_rank(comm, &engine->my_rank);
    int p = engine->n_proc;
    int mp, np, kp, rp, gen_proc_grid = 1;
    if (proc_grid != NULL)
    {
        mp = proc_grid[0];
        np = proc_grid[1];
        kp = proc_grid[2];
        rp = engine->n_proc - mp * np * kp;
        if ((mp != np) || (rp < 0))
        {
            if (engine->my_rank == 0) printf("[WARNING] Invalid process grid specified: mp, np, kp = %d, %d, %d\n", mp, np, kp);
        } else {
            gen_proc_grid = 0;
        }
    }
    if (gen_proc_grid) calc_3d_decomposition_nk(p, n, k, &np, &kp, &rp);
    mp = np;
    if ((mp < 1) || (np < 1) || (kp < 1) || (mp * np * kp > p))
    {
        if (engine->my_rank == 0) 
        {
            fprintf(stderr, "3D decomposition function error: p = %d, mp, np, kp = %d, %d, %d\n", p, mp, np, kp);
            fflush(stderr);
        }
        ca3dmm_engine_free(&engine);
        return;
    }
    engine->mp = mp;
    engine->np = np;
    engine->kp = kp;
    engine->rp = rp;
    engine->is_active = (engine->my_rank < mp * np * kp) ? 1 : 0;
    engine->is_BTB    = 1;
    engine->redist_ms = 0.0;
    engine->agvAB_ms  = 0.0;
    engine->cannon_ms = 0.0;
    engine->reduce_ms = 0.0;
    engine->n_exec    = 0;

    // 2. Handle the task groups
    // Since np == mp, only task_k_num can > 1
    int rank_k  = engine->my_rank / (mp * np);
    int rank_mn = engine->my_rank % (mp * np);
    int rank_n  = rank_mn / mp;
    int rank_m  = rank_mn % mp; 
    int task_m_num  = 1,  task_m_id = 0;
    int task_n_num  = 1,  task_n_id = 0;
    int task_k_num  = kp, task_k_id = rank_k;
    if (engine->is_active == 0)
    {
        rank_m = 0; rank_n  = 0;
        rank_k = 0; rank_mn = 0;
        task_m_num = 0;  task_m_id = 0;
        task_n_num = 0;  task_n_id = 0;
        task_k_num = 0;  task_k_id = 0;
    }
    engine->rank_m     = rank_m;
    engine->rank_n     = rank_n;
    engine->rank_k     = rank_k;
    engine->task_m_num = task_m_num;
    engine->task_m_id  = task_m_id;
    engine->task_n_num = task_n_num;
    engine->task_n_id  = task_n_id;
    engine->task_k_num = task_k_num;
    engine->task_k_id  = task_k_id;

    // 3. Set up the communicators
    // Since np == mp, only comm_2dmm and comm_C_rs is needed.
    // In the process grid, each mn-plane has exactly one 2D Cannon tasks.
    int color_2dmm, color_C_rs;
    if (engine->is_active == 1)
    {
        color_2dmm = rank_k;
        color_C_rs = rank_mn;
    } else {
        // 19241112 should be large enough
        color_2dmm = 19241112;
        color_C_rs = 19241112;
    }
    MPI_Comm_split(comm, color_2dmm, engine->my_rank, &engine->comm_2dmm);
    MPI_Comm_split(comm, color_C_rs, engine->my_rank, &engine->comm_C_rs);

    // 4. Calculate A, B, C block information
    if (engine->is_active)
    {
        int task_m_spos, task_m_size;
        int task_n_spos, task_n_size;
        int task_k_spos, task_k_size;
        calc_block_size_pos(m, task_m_num, task_m_id, &task_m_size, &task_m_spos);
        calc_block_size_pos(n, task_n_num, task_n_id, &task_n_size, &task_n_spos);
        calc_block_size_pos(k, task_k_num, task_k_id, &task_k_size, &task_k_spos);
        cannon_engine_init_ex(task_m_size, task_n_size, task_k_size, engine->communication_device, engine->compute_device, engine->comm_2dmm, &engine->cannon_engine);
        cannon_engine_p ce = engine->cannon_engine;
        if (ce == NULL)
        {
            ca3dmm_engine_free(&engine);
            return;
        }
        engine->A_2dmm_srow = task_m_spos + ce->A_srow;
        engine->A_2dmm_scol = task_k_spos + ce->A_scol;
        engine->A_2dmm_nrow = ce->A_nrow;
        engine->A_2dmm_ncol = ce->A_ncol;
        engine->B_2dmm_srow = task_k_spos + ce->B_srow;
        engine->B_2dmm_scol = task_n_spos + ce->B_scol;
        engine->B_2dmm_nrow = ce->B_nrow;
        engine->B_2dmm_ncol = ce->B_ncol;
        engine->C_2dmm_srow = task_m_spos + ce->C_srow;
        engine->C_2dmm_scol = task_n_spos + ce->C_scol;
        engine->C_2dmm_nrow = ce->C_nrow;
        engine->C_2dmm_ncol = ce->C_ncol;
        engine->C_out_srow  = engine->C_2dmm_srow;
        engine->C_out_nrow  = engine->C_2dmm_nrow;
        int C_out_scol, C_out_ncol, use_rsb = 1, C_out_ncol0 = -1;
        int *C_rs_recvcnts = (int *) malloc(sizeof(int) * task_k_num);
        for (int i = 0; i < task_k_num; i++)
        {
            calc_block_size_pos(engine->C_2dmm_ncol, task_k_num, i, &C_out_ncol, &C_out_scol);
            if (C_out_ncol0 == -1) C_out_ncol0 = C_out_ncol;
            if (C_out_ncol0 != C_out_ncol) use_rsb = 0;
            C_rs_recvcnts[i] = C_out_ncol * engine->C_2dmm_nrow;
            if (i == task_k_id)
            {
                engine->C_out_scol = engine->C_2dmm_scol + C_out_scol;
                engine->C_out_ncol = C_out_ncol;
            }
        }
        engine->C_rs_recvcnts = C_rs_recvcnts;
        engine->use_rsb = use_rsb;
    } else {
        engine->A_2dmm_srow = 0;
        engine->A_2dmm_scol = 0;
        engine->A_2dmm_nrow = 0;
        engine->A_2dmm_ncol = 0;
        engine->B_2dmm_srow = 0;
        engine->B_2dmm_scol = 0;
        engine->B_2dmm_nrow = 0;
        engine->B_2dmm_ncol = 0;
        engine->C_2dmm_srow = 0;
        engine->C_2dmm_scol = 0;
        engine->C_2dmm_nrow = 0;
        engine->C_2dmm_ncol = 0;
        engine->C_out_srow  = 0;
        engine->C_out_scol  = 0;
        engine->C_out_nrow  = 0;
        engine->C_out_ncol  = 0;
        engine->C_rs_recvcnts = NULL;
    }  // End of "if (engine->is_active)"
    engine->A_rd_srow = engine->A_2dmm_srow;
    engine->A_rd_scol = engine->A_2dmm_scol;
    engine->A_rd_nrow = engine->A_2dmm_nrow;
    engine->A_rd_ncol = engine->A_2dmm_ncol;
    engine->B_rd_srow = engine->B_2dmm_srow;
    engine->B_rd_scol = engine->B_2dmm_scol;
    engine->B_rd_nrow = engine->B_2dmm_nrow;
    engine->B_rd_ncol = engine->B_2dmm_ncol;

    // 5. Set up mat_redist_engine
    // (1) We only need to redistribute B globally, A (== B^T) block can be obtained
    //     using P2P communication efficiently. 
    // (2) Non-active processes still need to participate in global B redistribution, 
    //     but their B_{s, n}{row, col} == 0.
    // (3) The input A and B matrices are column-major, mat_redist_engine uses row-major, so 
    //     we need to swap the parameters when calling mat_redist_engine_init().
    mat_redist_engine_init_ex(
        src_B_scol, src_B_srow, src_B_ncol, src_B_nrow, 
        engine->B_2dmm_scol, engine->B_2dmm_srow, engine->B_2dmm_ncol, engine->B_2dmm_nrow, 
        engine->communication_device,
        comm, MPI_DOUBLE, sizeof(double), &engine->redist_B
    );
    if (engine->redist_B == NULL)
    {
        ca3dmm_engine_free(&engine);
        return;
    }
    if (!((dst_C_srow == -1) || (dst_C_nrow == -1) || (dst_C_scol == -1) || (dst_C_ncol == -1)))
    {
        mat_redist_engine_init_ex(
            engine->C_out_scol, engine->C_out_srow, engine->C_out_ncol, engine->C_out_nrow, 
            dst_C_scol, dst_C_srow, dst_C_ncol, dst_C_nrow, 
            engine->communication_device,
            comm, MPI_DOUBLE, sizeof(double), &engine->redist_C
        );
        if (engine->redist_C == NULL)
        {
            ca3dmm_engine_free(&engine);
            return;
        }
    }

    // 6. Allocate local matrix blocks
    void *A_rd_recv = NULL, *A_2dmm = NULL, *A_trans = NULL;
    void *B_rd_recv = NULL, *B_2dmm = NULL, *B_trans = NULL;
    void *C_2dmm = NULL, *C_out = NULL;
    if (engine->is_active)
    {
        A_2dmm  = _OUR_MALLOC(sizeof(double) * engine->A_2dmm_nrow * engine->A_2dmm_ncol, engine->communication_device);
        A_trans = _OUR_MALLOC(sizeof(double) * engine->A_2dmm_nrow * engine->A_2dmm_ncol, engine->communication_device);
        B_2dmm  = _OUR_MALLOC(sizeof(double) * engine->B_2dmm_nrow * engine->B_2dmm_ncol, engine->communication_device);
        C_2dmm  = _OUR_MALLOC(sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol, engine->communication_device);
        C_out   = _OUR_MALLOC(sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol, engine->communication_device);
        if ((A_2dmm == NULL) || (A_trans == NULL) || (B_2dmm == NULL) || 
            (C_2dmm == NULL) || (C_out == NULL))
        {
            fprintf(stderr, "[ERROR] Failed to allocate ca3dmm_engine matrix buffers\n");
            ca3dmm_engine_free(&engine);
            return;
        }
    }
    engine->A_rd_recv = A_rd_recv;
    engine->A_2dmm    = A_2dmm;
    engine->A_trans   = A_trans;
    engine->B_rd_recv = A_rd_recv;
    engine->B_2dmm    = B_2dmm;
    engine->B_trans   = B_trans;
    engine->C_2dmm    = C_2dmm;
    engine->C_out     = C_out;

    char *print_timing_p = getenv("CA3DMM_PRINT_TIMING");
    int print_timing = 0;
    if (print_timing_p != NULL) print_timing = atoi(print_timing_p);
    if (engine->my_rank == 0 && print_timing == 1) engine->print_timing = 1;
    else engine->print_timing = 0;

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

// Free a camm3d_engine structure
void ca3dmm_engine_free(ca3dmm_engine_p *engine_)
{
    ca3dmm_engine_p engine = *engine_;
    if (engine == NULL) return;
    free(engine->AB_agv_recvcnts);
    free(engine->AB_agv_displs);
    free(engine->C_rs_recvcnts);
    OUR_FREE(engine->A_rd_recv, engine->communication_device);
    OUR_FREE(engine->B_rd_recv, engine->communication_device);
    if (engine->A_trans != engine->A_rd_recv) OUR_FREE(engine->A_trans, engine->communication_device);
    if (engine->B_trans != engine->B_rd_recv) OUR_FREE(engine->B_trans, engine->communication_device);
    if (engine->A_2dmm != engine->A_trans) OUR_FREE(engine->A_2dmm, engine->communication_device);
    if (engine->B_2dmm != engine->B_trans) OUR_FREE(engine->B_2dmm, engine->communication_device);
    OUR_FREE(engine->C_2dmm, engine->communication_device);
    OUR_FREE(engine->C_out, engine->communication_device);
    if (engine->is_BTB == 0) MPI_Comm_free(&engine->comm_AB_agv);
    MPI_Comm_free(&engine->comm_C_rs);
    MPI_Comm_free(&engine->comm_2dmm);
    mat_redist_engine_free(&engine->redist_A);
    mat_redist_engine_free(&engine->redist_B);
    mat_redist_engine_free(&engine->redist_C);
    cannon_engine_free(&engine->cannon_engine);
    free(engine);
    *engine_ = NULL;
}

// Perform Communication-Avoiding 3D Matrix Multiplication (CA3DMM)
void ca3dmm_engine_exec(
    const void *src_A, const int ldA,
    const void *src_B, const int ldB,
    void *dst_C, const int ldC,
    ca3dmm_engine_p engine
)
{
    if (engine == NULL)
    {
        fprintf(stderr, "[ERROR] ca3dmm_engine not initialized\n");
        return;
    }

    int A_rd_nrow   = engine->A_rd_nrow;
    int A_rd_ncol   = engine->A_rd_ncol;
    int A_2dmm_nrow = engine->A_2dmm_nrow;
    int A_2dmm_ncol = engine->A_2dmm_ncol;
    int B_rd_nrow   = engine->B_rd_nrow;
    int B_rd_ncol   = engine->B_rd_ncol;
    int B_2dmm_nrow = engine->B_2dmm_nrow;
    int B_2dmm_ncol = engine->B_2dmm_ncol;
    int C_2dmm_nrow = engine->C_2dmm_nrow;
    int C_2dmm_ncol = engine->C_2dmm_ncol;
    int *AB_agv_recvcnts = engine->AB_agv_recvcnts;
    int *AB_agv_displs   = engine->AB_agv_displs;
    double *A_rd_recv = (double *) engine->A_rd_recv;
    double *A_2dmm    = (double *) engine->A_2dmm;
    double *A_trans   = (double *) engine->A_trans;
    double *B_rd_recv = (double *) engine->B_rd_recv;
    double *B_2dmm    = (double *) engine->B_2dmm;
    double *B_trans   = (double *) engine->B_trans;
    double *C_2dmm    = (double *) engine->C_2dmm;
    double *C_out     = (double *) engine->C_out;

     //gpu_print_mat(A_2dmm, 1, 1);
     //gpu_print_mat(A_trans, 1, 1);

    double start_t, stop_t, exec_start_t, exec_stop_t;
    double redist_ms, agvAB_ms, cannon_ms, reduce_ms, exec_ms;
    
    exec_start_t = MPI_Wtime();

    // Notice again: 
    // (1) mat_redist_engine uses row-major but ca3dmm_engine uses col-major
    // (2) task_m_num and task_n_num cannot both > 1
    if (engine->is_BTB == 0)
    {
        // Redistribute A and B matrices first. If A/B needs to be transposed, 
        // {A/B}_rd_nrow is swapped with {A/B}_rd_ncol before calling mat_redist_engine_init(), 
        // so recv_ld{A/B} should == req_{A/B}_nrow.
        start_t = MPI_Wtime();
        int trans_A  = engine->trans_A;
        int trans_B  = engine->trans_B;
        int recv_ldA = (trans_A) ? A_rd_ncol : A_rd_nrow;
        int recv_ldB = (trans_B) ? B_rd_ncol : B_rd_nrow;   
        mat_redist_engine_exec(engine->redist_A, src_A, ldA, A_rd_recv, recv_ldA);
        mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_rd_recv, recv_ldB);

#if USE_GPU
        if(engine->communication_device == DEVICE_TYPE_DEVICE) {
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
#endif

        stop_t = MPI_Wtime();
        redist_ms = 1000.0 * (stop_t - start_t);
        engine->redist_ms += redist_ms;
        if (engine->print_timing) printf("[INFO] Redistribute A & B time = %.2f ms\n", redist_ms);

        // Local transpose the received A & B blocks before MPI_allgatherv and Cannon2D.
        // Cannon 2D calls DGEMM with the parameters of no-transpose for both A & B block, 
        // we need to manually transpose the received Aji^T before calling Cannon.
        start_t = MPI_Wtime();
        int A_trans_nrow, A_trans_ncol, B_trans_nrow, B_trans_ncol;
        if (engine->task_n_num > 1)
        {
            A_trans_nrow = A_rd_nrow;
            A_trans_ncol = A_rd_ncol;
        } else {
            A_trans_nrow = A_2dmm_nrow;
            A_trans_ncol = A_2dmm_ncol;
        }
        if (engine->task_m_num > 1)
        {
            B_trans_nrow = B_rd_nrow;
            B_trans_ncol = B_rd_ncol;
        } else {
            B_trans_nrow = B_2dmm_nrow;
            B_trans_ncol = B_2dmm_ncol;
        }
        if (trans_A) transpose_cm_mat(A_trans_nrow, A_trans_ncol, A_rd_recv, A_trans_ncol, A_trans, A_2dmm_nrow, engine->handle, engine->communication_device);
        else A_trans = A_rd_recv;
        if (trans_B) transpose_cm_mat(B_trans_nrow, B_trans_ncol, B_rd_recv, B_trans_ncol, B_trans, B_2dmm_nrow, engine->handle, engine->communication_device);
        else B_trans = B_rd_recv;

        // Allgatherv A or B to make it complete
        if (engine->task_m_num > 1)
        {
            if (engine->use_ag)
            {
                MPI_Allgather(
                    B_trans, B_rd_nrow * B_rd_ncol, MPI_DOUBLE, B_2dmm, 
                    AB_agv_recvcnts[0], MPI_DOUBLE, engine->comm_AB_agv
                );
            } else {
                MPI_Allgatherv(
                    B_trans, B_rd_nrow * B_rd_ncol, MPI_DOUBLE, B_2dmm, 
                    AB_agv_recvcnts, AB_agv_displs, MPI_DOUBLE, engine->comm_AB_agv
                );
            }

#if USE_GPU
            if(engine->communication_device == DEVICE_TYPE_DEVICE) {
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
            }
#endif
        } else {
            B_2dmm = B_trans;
        }  // End of "if (engine->task_m_num > 1)"

        if (engine->task_n_num > 1)
        {
            if (engine->use_ag)
            {
                MPI_Allgather(
                    A_trans, A_rd_nrow * A_rd_ncol, MPI_DOUBLE, A_2dmm, 
                    AB_agv_recvcnts[0], MPI_DOUBLE, engine->comm_AB_agv
                );
            } else {
                MPI_Allgatherv(
                    A_trans, A_rd_nrow * A_rd_ncol, MPI_DOUBLE, A_2dmm, 
                    AB_agv_recvcnts, AB_agv_displs, MPI_DOUBLE, engine->comm_AB_agv
                );
            }

#if USE_GPU
            if(engine->communication_device == DEVICE_TYPE_DEVICE) {
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
            }
#endif
        } else {
            A_2dmm = A_trans;
        }  // End of "if (engine->task_n_num > 1)"
        stop_t = MPI_Wtime();
        agvAB_ms = 1000.0 * (stop_t - start_t);
        engine->agvAB_ms += agvAB_ms;
        if (engine->print_timing) printf("[INFO] Allgather A or B time   = %.2f ms\n", agvAB_ms);
    } else {
        start_t = MPI_Wtime();
        mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_2dmm, B_2dmm_nrow);
        stop_t = MPI_Wtime();
        redist_ms = 1000.0 * (stop_t - start_t);
        engine->redist_ms += redist_ms;
        if (engine->print_timing) printf("[INFO] Redistribute A & B time = %.2f ms\n", redist_ms);

        // In each Cannon task group, block Bij hold by process Pij == the transpose of 
        // block Aji required by process Pji. Since Cannon 2D calls DGEMM with the 
        // parameters of no-transpose for both A & B block, we need to manually 
        // transpose the received Aji^T before calling Cannon.
        start_t = MPI_Wtime();
        if (engine->is_active == 1)
        {
            cannon_engine_p ce = engine->cannon_engine;
            int ce_rank_row = ce->rank_row;
            int ce_rank_col = ce->rank_col;
            if (ce_rank_row == ce_rank_col)
            {
                assert(B_2dmm_nrow * B_2dmm_ncol == A_2dmm_nrow * A_2dmm_ncol);
                OUR_MEMCPY(A_trans, B_2dmm, sizeof(double) * B_2dmm_nrow * B_2dmm_ncol, engine->communication_device, engine->communication_device);
            } else {
                int ce_pair_rank = ce_rank_col * ce->np_dim + ce_rank_row;
                MPI_Sendrecv(
                    B_2dmm,  B_2dmm_nrow * B_2dmm_ncol, MPI_DOUBLE, ce_pair_rank, 0,
                    A_trans, A_2dmm_nrow * A_2dmm_ncol, MPI_DOUBLE, ce_pair_rank, 0, ce->comm, MPI_STATUS_IGNORE
                );

#if USE_GPU
                if(engine->communication_device == DEVICE_TYPE_DEVICE) {
                    gpuErrchk( cudaPeekAtLastError() );
                    gpuErrchk( cudaDeviceSynchronize() );
                }
#endif
            }
            //gpu_print_mat(A_2dmm, 1, 1);
            //gpu_print_mat(B_2dmm, 1, 1);
            transpose_cm_mat(A_2dmm_nrow, A_2dmm_ncol, A_trans, A_2dmm_ncol, A_2dmm, A_2dmm_nrow, engine->handle, engine->communication_device);
            //gpu_print_mat(A_2dmm, 1, 1);
            //gpu_print_mat(B_2dmm, 1, 1);
        }  // End of "if (engine->is_active == 1)"
        stop_t = MPI_Wtime();
        agvAB_ms = 1000.0 * (stop_t - start_t);
        engine->agvAB_ms += agvAB_ms;
        if (engine->print_timing) printf("[INFO] Allgather A or B time   = %.2f ms\n", agvAB_ms);
    }  // End of "if (engine->is_BTB == 0)"

    if (engine->is_active == 1)
    {
        start_t = MPI_Wtime();
        if (engine->task_k_num == 1) C_2dmm = C_out;
        //gpu_print_mat(A_2dmm, 1, 1);
        //gpu_print_mat(B_2dmm, 1, 1);
        cannon_engine_exec(1.0, A_2dmm, B_2dmm, 0.0, C_2dmm, engine->cannon_engine);
        stop_t = MPI_Wtime();
        cannon_ms = 1000.0 * (stop_t - start_t);
        engine->cannon_ms += cannon_ms;
        if (engine->print_timing) printf("[INFO] 2D Cannon time          = %.2f ms\n", cannon_ms);

        start_t = MPI_Wtime();
        if (engine->task_k_num > 1)
        {
            if (engine->use_rsb)
            {
                MPI_Reduce_scatter_block(
                    C_2dmm, C_out, engine->C_rs_recvcnts[0], MPI_DOUBLE, 
                    MPI_SUM, engine->comm_C_rs
                );
            } else {
                MPI_Reduce_scatter(
                    C_2dmm, C_out, engine->C_rs_recvcnts, MPI_DOUBLE, 
                    MPI_SUM, engine->comm_C_rs
                );
            }
        }
        stop_t = MPI_Wtime();
        reduce_ms = 1000.0 * (stop_t - start_t);
        engine->reduce_ms += reduce_ms;
        if (engine->print_timing)
        {
            printf("[INFO] Reduce-scatter C time   = %.2f ms\n", reduce_ms);
            printf("[INFO] CA3DMM matmul time      = %.2f ms\n", agvAB_ms + cannon_ms + reduce_ms);
        }
    }

    start_t = MPI_Wtime();
    if (engine->redist_C != NULL)
        mat_redist_engine_exec(engine->redist_C, engine->C_out, engine->C_out_nrow, dst_C, ldC);
    stop_t = MPI_Wtime();
    redist_ms = 1000.0 * (stop_t - start_t);
    engine->redist_ms += redist_ms;
    if (engine->print_timing) printf("[INFO] Redistribute C time     = %.2f ms\n", redist_ms);

    // No need to add a global barrier here since mat_redist_engine_exec() already has one
    exec_stop_t = MPI_Wtime();
    exec_ms = 1000.0 * (exec_stop_t - exec_start_t);
    engine->exec_ms += exec_ms;
    if (engine->print_timing) printf("[INFO] CA3DMM exec time        = %.2f ms\n\n", exec_ms);
    engine->n_exec++;
}


// Reset the statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_reset_stat(ca3dmm_engine_p engine)
{
    if (engine == NULL) return;
    cannon_engine_reset_stat(engine->cannon_engine);
    engine->redist_ms = 0.0;
    engine->agvAB_ms  = 0.0;
    engine->cannon_ms = 0.0;
    engine->reduce_ms = 0.0;
    engine->exec_ms   = 0.0;
    engine->n_exec    = 0;
}

// Print the statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_print_stat(ca3dmm_engine_p engine)
{
    if (engine == NULL) return;
    if (engine->n_exec == 0)
    {
        printf("No ca3dmm_engine statistic data to print\n");
        return;
    }
    printf("================== CA3DMM algorithm engine =================\n");
    printf("* Initialization         : %.2f ms\n", engine->init_ms);
    printf("* Number of executions   : %d\n", engine->n_exec);
    printf("* Execution time (avg)   : %.2f ms\n", engine->exec_ms   / engine->n_exec);
    printf("  * Redistribute A, B, C : %.2f ms\n", engine->redist_ms / engine->n_exec);
    printf("  * Allgather A or B     : %.2f ms\n", engine->agvAB_ms  / engine->n_exec);
    printf("  * 2D Cannon execution  : %.2f ms\n", engine->cannon_ms / engine->n_exec);
    printf("  * Reduce-scatter C     : %.2f ms\n", engine->reduce_ms / engine->n_exec);
    cannon_engine_print_stat(engine->cannon_engine);
    printf("============================================================\n");
}
