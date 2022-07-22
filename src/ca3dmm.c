#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>

#include "utils.h"
#include "ca3dmm.h"
#include "mpi_op_omp.h"
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

static inline void swap_int(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// Initialize a ca3dmm_engine structure for C := op(A) * op(B)
void ca3dmm_engine_init(
    const int m, const int n, const int k, const int trans_A, const int trans_B, 
    const int src_A_srow, const int src_A_nrow,
    const int src_A_scol, const int src_A_ncol,
    const int src_B_srow, const int src_B_nrow,
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, dev_type_t dev_type, 
    ca3dmm_engine_p *engine_, size_t *workbuf_bytes
)
{
    *engine_ = NULL;
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    ca3dmm_engine_p engine = (ca3dmm_engine_p) malloc(sizeof(ca3dmm_engine_s));
    memset(engine, 0, sizeof(ca3dmm_engine_s));

    double start_t = MPI_Wtime();

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
            if (engine->my_rank == 0) 
                WARNING_PRINTF("Invalid process grid specified: mp, np, kp = %d, %d, %d\n", mp, np, kp);
        } else {
            gen_proc_grid = 0;
        }
    }
    if (gen_proc_grid)
    {
        calc_3d_decomposition_cannon(p, m, n, k, &mp, &np, &kp, &rp);
        int reduce_kp;
        GET_ENV_INT_VAR(reduce_kp, "CA3DMM_REDUCE_KP", "reduce_kp", 0, 0, 1);
        if (reduce_kp)
        {
            if ((kp >= 3) && (kp % 3 == 0))
            {
                if (mp <= np)
                {
                    int mp_new = mp * 3;
                    int max_mn = (mp_new > np) ? mp_new : np;
                    int min_mn = (mp_new < np) ? mp_new : np;
                    if (max_mn % min_mn == 0)
                    {
                        kp /= 3;
                        mp *= 3;
                    }
                } else {
                    int np_new = np * 3;
                    int max_mn = (np_new > mp) ? np_new : mp;
                    int min_mn = (np_new < mp) ? np_new : mp;
                    if (max_mn % min_mn == 0)
                    {
                        kp /= 3;
                        np *= 3;
                    }
                }
            }   // End of "if ((kp >= 3) && (kp % 3 == 0))"
            else if ((kp >= 2) && (kp % 2 == 0))
            {
                if (mp <= np)
                {
                    int mp_new = mp * 2;
                    int max_mn = (mp_new > np) ? mp_new : np;
                    int min_mn = (mp_new < np) ? mp_new : np;
                    if (max_mn % min_mn == 0)
                    {
                        kp /= 2;
                        mp *= 2;
                    }
                } else {
                    int np_new = np * 2;
                    int max_mn = (np_new > mp) ? np_new : mp;
                    int min_mn = (np_new < mp) ? np_new : mp;
                    if (max_mn % min_mn == 0)
                    {
                        kp /= 2;
                        np *= 2;
                    }
                }
            }  // End of "if ((kp >= 2) && (kp % 2 == 0))"
        }  // End of "if (reduce_kp)"
    }
    if ((mp < 1) || (np < 1) || (kp < 1) || (mp * np * kp > p))
    {
        if (engine->my_rank == 0)
            ERROR_PRINTF("Invalid process grid generated: p = %d, mp, np, kp = %d, %d, %d\n", p, mp, np, kp);
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
    engine->dev_type  = dev_type;

    // 2. Handle the task groups, note that max(mp, np) is a multiple of min(mp, np)
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
        calc_block_spos_size(m, task_m_num, task_m_id, &task_m_spos, &task_m_size);
        calc_block_spos_size(n, task_n_num, task_n_id, &task_n_spos, &task_n_size);
        calc_block_spos_size(k, task_k_num, task_k_id, &task_k_spos, &task_k_size);
        cannon_engine_init(
            task_m_size, task_n_size, task_k_size, engine->comm_2dmm, 
            dev_type, &engine->cannon_engine, &engine->cannon_workbuf_bytes
        );
        cannon_engine_p ce = engine->cannon_engine;
        if (ce == NULL)
        {
            ERROR_PRINTF("Failed to initialize cannon_engine\n");
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
            calc_block_spos_size(engine->C_2dmm_ncol, task_k_num, i, &C_out_scol, &C_out_ncol);
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
            calc_block_spos_size(engine->A_2dmm_ncol, task_n_num, i, &scol, &ncol);
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
            calc_block_spos_size(engine->B_2dmm_ncol, task_m_num, i, &scol, &ncol);
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
    mat_redist_engine_init(
        src_A_scol, src_A_srow, src_A_ncol, src_A_nrow, 
        A_rd_scol,  A_rd_srow,  A_rd_ncol,  A_rd_nrow, 
        comm, MPI_DOUBLE, sizeof(double), dev_type, 
        &engine->redist_A, &engine->rdA_workbuf_bytes
    );
    mat_redist_engine_init(
        src_B_scol, src_B_srow, src_B_ncol, src_B_nrow, 
        B_rd_scol,  B_rd_srow,  B_rd_ncol,  B_rd_nrow, 
        comm, MPI_DOUBLE, sizeof(double), dev_type,
        &engine->redist_B, &engine->rdB_workbuf_bytes
    );
    if ((engine->redist_A == NULL) || (engine->redist_B == NULL))
    {
        ERROR_PRINTF("Failed to initialize redist_A and redist_B\n");
        mat_redist_engine_free(&engine->redist_A);
        mat_redist_engine_free(&engine->redist_B);
        ca3dmm_engine_free(&engine);
        return;
    }
    engine->AB_agv_recvcnts = AB_agv_recvcnts;
    engine->AB_agv_displs   = AB_agv_displs;
    if (!((dst_C_srow == -1) || (dst_C_nrow == -1) || (dst_C_scol == -1) || (dst_C_ncol == -1)))
    {
        mat_redist_engine_init(
            engine->C_out_scol, engine->C_out_srow, engine->C_out_ncol, engine->C_out_nrow, 
            dst_C_scol, dst_C_srow, dst_C_ncol, dst_C_nrow, 
            comm, MPI_DOUBLE, sizeof(double), dev_type, 
            &engine->redist_C, &engine->rdC_workbuf_bytes
        );
        if (engine->redist_C == NULL)
        {
            ERROR_PRINTF("Failed to initialize redist_C\n");
            mat_redist_engine_free(&engine->redist_C);
            ca3dmm_engine_free(&engine);
            return;
        }
    }

    // 6. Calculate local matrix block sizes
    size_t self_workbuf_bytes = 0;
    if (engine->is_active)
    {
        size_t A_blk_bytes = sizeof(double) * engine->cannon_engine->max_A_blk_size;
        size_t B_blk_bytes = sizeof(double) * engine->cannon_engine->max_B_blk_size;
        self_workbuf_bytes += A_blk_bytes;  // A_rd_recv
        self_workbuf_bytes += B_blk_bytes;  // B_rd_recv
        if (trans_A == 1)   self_workbuf_bytes += A_blk_bytes;  // A_trans
        if (trans_B == 1)   self_workbuf_bytes += B_blk_bytes;  // B_trans
        if (task_n_num > 1) self_workbuf_bytes += A_blk_bytes;  // A_2dmm
        if (task_m_num > 1) self_workbuf_bytes += B_blk_bytes;  // B_2dmm
        self_workbuf_bytes += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;
        self_workbuf_bytes += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;
    }
    engine->self_workbuf_bytes = self_workbuf_bytes;

    // 7. Allocate and attach work buffer if needed
    size_t max_child_workbuf_bytes = engine->rdA_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->rdB_workbuf_bytes) 
        max_child_workbuf_bytes = engine->rdB_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->rdC_workbuf_bytes) 
        max_child_workbuf_bytes = engine->rdC_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->cannon_workbuf_bytes) 
        max_child_workbuf_bytes = engine->cannon_workbuf_bytes;
    size_t total_workbuf_bytes = max_child_workbuf_bytes + self_workbuf_bytes;
    if (workbuf_bytes != NULL)
    {
        engine->alloc_workbuf = 0;
        *workbuf_bytes = total_workbuf_bytes;
    } else {
        engine->alloc_workbuf = 1;
        void *workbuf_h, *workbuf_d;
        MALLOC_ATTACH_WORKBUF(
            ca3dmm_engine_attach_workbuf, ca3dmm_engine_free, 
            engine, dev_type, total_workbuf_bytes, workbuf_h, workbuf_d
        );
    }

    GET_ENV_INT_VAR(engine->print_timing, "CA3DMM_PRINT_TIMING", "engine->print_timing", 0, 0, 1);
    if (engine->my_rank > 0) engine->print_timing = 0;

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

// Initialize a ca3dmm_engine structure for C := B^T * B
void ca3dmm_engine_init_BTB(
    const int n, const int k, 
    const int src_B_srow, const int src_B_nrow, 
    const int src_B_scol, const int src_B_ncol,
    const int dst_C_srow, const int dst_C_nrow,
    const int dst_C_scol, const int dst_C_ncol,
    const int *proc_grid, MPI_Comm comm, dev_type_t dev_type, 
    ca3dmm_engine_p *engine_, size_t *workbuf_bytes
)
{
    *engine_ = NULL;
    if (is_dev_type_valid(dev_type) == 0)
    {
        ERROR_PRINTF("Invalid device type %d\n", dev_type);
        return;
    }

    ca3dmm_engine_p engine = (ca3dmm_engine_p) malloc(sizeof(ca3dmm_engine_s));
    memset(engine, 0, sizeof(ca3dmm_engine_s));

    double start_t = MPI_Wtime();

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
            if (engine->my_rank == 0) 
                WARNING_PRINTF("Invalid process grid specified: mp, np, kp = %d, %d, %d\n", mp, np, kp);
        } else {
            gen_proc_grid = 0;
        }
    }
    if (gen_proc_grid)
    {
        calc_3d_decomposition_nk(p, n, k, &np, &kp, &rp);
        int reduce_kp;
        GET_ENV_INT_VAR(reduce_kp, "CA3DMM_REDUCE_KP", "reduce_kp", 1, 0, 1);
        if ((reduce_kp) && (kp >= 4) && (kp % 4 == 0))
        {
            kp /= 4;
            np *= 2;
        }
    }
    mp = np;
    if ((mp < 1) || (np < 1) || (kp < 1) || (mp * np * kp > p))
    {
        if (engine->my_rank == 0)
            ERROR_PRINTF("3D decomposition function error: p = %d, mp, np, kp = %d, %d, %d\n", p, mp, np, kp);
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
    engine->dev_type  = dev_type;

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
        calc_block_spos_size(m, task_m_num, task_m_id, &task_m_spos, &task_m_size);
        calc_block_spos_size(n, task_n_num, task_n_id, &task_n_spos, &task_n_size);
        calc_block_spos_size(k, task_k_num, task_k_id, &task_k_spos, &task_k_size);
        cannon_engine_init(
            task_m_size, task_n_size, task_k_size, engine->comm_2dmm, 
            dev_type, &engine->cannon_engine, &engine->cannon_workbuf_bytes
        );
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
            calc_block_spos_size(engine->C_2dmm_ncol, task_k_num, i, &C_out_scol, &C_out_ncol);
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
    mat_redist_engine_init(
        src_B_scol, src_B_srow, src_B_ncol, src_B_nrow, 
        engine->B_2dmm_scol, engine->B_2dmm_srow, engine->B_2dmm_ncol, engine->B_2dmm_nrow, 
        comm, MPI_DOUBLE, sizeof(double), dev_type, 
        &engine->redist_B, &engine->rdB_workbuf_bytes
    );
    if (engine->redist_B == NULL)
    {
        ERROR_PRINTF("Failed to initialize redist_B\n");
        ca3dmm_engine_free(&engine);
        return;
    }
    if (!((dst_C_srow == -1) || (dst_C_nrow == -1) || (dst_C_scol == -1) || (dst_C_ncol == -1)))
    {
        mat_redist_engine_init(
            engine->C_out_scol, engine->C_out_srow, engine->C_out_ncol, engine->C_out_nrow, 
            dst_C_scol, dst_C_srow, dst_C_ncol, dst_C_nrow, 
            comm, MPI_DOUBLE, sizeof(double), dev_type, 
            &engine->redist_C, &engine->rdC_workbuf_bytes
        );
        if (engine->redist_C == NULL)
        {
            ERROR_PRINTF("Failed to initialize redist_C\n");
            ca3dmm_engine_free(&engine);
            return;
        }
    }

    // 6. Calculate local matrix block sizes
    size_t self_workbuf_bytes = 0;
    if (engine->is_active)
    {
        size_t A_blk_bytes = sizeof(double) * engine->cannon_engine->max_A_blk_size;
        size_t B_blk_bytes = sizeof(double) * engine->cannon_engine->max_B_blk_size;
        self_workbuf_bytes += A_blk_bytes;  // A_2dmm
        self_workbuf_bytes += A_blk_bytes;  // A_trans
        self_workbuf_bytes += B_blk_bytes;  // B_2dmm
        self_workbuf_bytes += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;  // C_2dmm
        self_workbuf_bytes += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;   // C_out
    }
    engine->self_workbuf_bytes = self_workbuf_bytes;

    // 7. Allocate and attach work buffer if needed
    size_t max_child_workbuf_bytes = engine->rdA_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->rdB_workbuf_bytes) 
        max_child_workbuf_bytes = engine->rdB_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->rdC_workbuf_bytes) 
        max_child_workbuf_bytes = engine->rdC_workbuf_bytes;
    if (max_child_workbuf_bytes < engine->cannon_workbuf_bytes) 
        max_child_workbuf_bytes = engine->cannon_workbuf_bytes;
    size_t total_workbuf_bytes = max_child_workbuf_bytes + self_workbuf_bytes;
    if (workbuf_bytes != NULL)
    {
        engine->alloc_workbuf = 0;
        *workbuf_bytes = total_workbuf_bytes;
    } else {
        engine->alloc_workbuf = 1;
        void *workbuf_h, *workbuf_d;
        MALLOC_ATTACH_WORKBUF(
            ca3dmm_engine_attach_workbuf, ca3dmm_engine_free, 
            engine, dev_type, total_workbuf_bytes, workbuf_h, workbuf_d
        );
    }

    GET_ENV_INT_VAR(engine->print_timing, "CA3DMM_PRINT_TIMING", "engine->print_timing", 0, 0, 1);
    if (engine->my_rank > 0) engine->print_timing = 0;

    double stop_t = MPI_Wtime();
    engine->init_ms = 1000.0 * (stop_t - start_t);

    *engine_ = engine;
}

// Attach an external work buffer for ca3dmm_engine
void ca3dmm_engine_attach_workbuf(ca3dmm_engine_p engine, void *workbuf_h, void *workbuf_d)
{
    size_t rdA_workbuf_bytes    = engine->rdA_workbuf_bytes;
    size_t rdB_workbuf_bytes    = engine->rdB_workbuf_bytes;
    size_t rdC_workbuf_bytes    = engine->rdC_workbuf_bytes;
    size_t cannon_workbuf_bytes = engine->cannon_workbuf_bytes;

    size_t max_child_workbuf_bytes = rdA_workbuf_bytes;
    if (max_child_workbuf_bytes < rdB_workbuf_bytes) 
        max_child_workbuf_bytes = rdB_workbuf_bytes;
    if (max_child_workbuf_bytes < rdC_workbuf_bytes) 
        max_child_workbuf_bytes = rdC_workbuf_bytes;
    if (max_child_workbuf_bytes < cannon_workbuf_bytes) 
        max_child_workbuf_bytes = cannon_workbuf_bytes;

    engine->workbuf_h = workbuf_h;
    engine->workbuf_d = workbuf_d;

    char *child_workbuf_h = (char *) workbuf_h;
    char *child_workbuf_d = (char *) workbuf_d;
    char *self_workbuf_h  = child_workbuf_h + max_child_workbuf_bytes;
    char *self_workbuf_d  = child_workbuf_d + max_child_workbuf_bytes;

    if (engine->redist_A != NULL) mat_redist_engine_attach_workbuf(engine->redist_A, child_workbuf_h, child_workbuf_d);
    if (engine->redist_B != NULL) mat_redist_engine_attach_workbuf(engine->redist_B, child_workbuf_h, child_workbuf_d);
    if (engine->redist_C != NULL) mat_redist_engine_attach_workbuf(engine->redist_C, child_workbuf_h, child_workbuf_d);
    if (engine->cannon_engine != NULL) cannon_engine_attach_workbuf(engine->cannon_engine, child_workbuf_h, child_workbuf_d);

    dev_type_t dev_type = engine->dev_type;
    size_t A_blk_bytes = 0, B_blk_bytes = 0;
    if (engine->is_active)
    {
        A_blk_bytes = sizeof(double) * engine->cannon_engine->max_A_blk_size;
        B_blk_bytes = sizeof(double) * engine->cannon_engine->max_B_blk_size;
    }
    if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
    {
        if ((engine->is_BTB == 0) && (engine->is_active))
        {
            engine->A_rd_recv_h = (void *) self_workbuf_h;
            self_workbuf_h += A_blk_bytes;

            engine->B_rd_recv_h = (void *) self_workbuf_h;
            self_workbuf_h += B_blk_bytes;

            if (engine->trans_A)
            {
                engine->A_trans_h = (void *) self_workbuf_h;
                self_workbuf_h += A_blk_bytes;
            } else {
                engine->A_trans_h = engine->A_rd_recv_h;
            }

            if (engine->trans_B)
            {
                engine->B_trans_h = (void *) self_workbuf_h;
                self_workbuf_h += B_blk_bytes;
            } else {
                engine->B_trans_h = engine->B_rd_recv_h;
            }

            if (engine->task_n_num > 1)
            {
                engine->A_2dmm_h = (void *) self_workbuf_h;
                self_workbuf_h += A_blk_bytes;
            } else {
                engine->A_2dmm_h = engine->A_trans_h;
            }

            if (engine->task_m_num > 1)
            {
                engine->B_2dmm_h = (void *) self_workbuf_h;
                self_workbuf_h += B_blk_bytes;
            } else {
                engine->B_2dmm_h = engine->B_trans_h;
            }

            engine->C_2dmm_h = (void *) self_workbuf_h;
            self_workbuf_h += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;

            engine->C_out_h = (void *) self_workbuf_h;
            self_workbuf_h += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;
        }
        if ((engine->is_BTB == 1) && (engine->is_active)) 
        {
            engine->A_2dmm_h = (void *) self_workbuf_h;
            self_workbuf_h += A_blk_bytes;

            engine->A_trans_h = (void *) self_workbuf_h;
            self_workbuf_h += A_blk_bytes;

            engine->B_2dmm_h = (void *) self_workbuf_h;
            self_workbuf_h += B_blk_bytes;

            engine->C_2dmm_h = (void *) self_workbuf_h;
            self_workbuf_h += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;

            engine->C_out_h = (void *) self_workbuf_h;
            self_workbuf_h += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;
        }
    }
    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
    {
        if ((engine->is_BTB == 0) && (engine->is_active))
        {
            engine->A_rd_recv_d = (void *) self_workbuf_d;
            self_workbuf_d += A_blk_bytes;

            engine->B_rd_recv_d = (void *) self_workbuf_d;
            self_workbuf_d += B_blk_bytes;

            if (engine->trans_A)
            {
                engine->A_trans_d = (void *) self_workbuf_d;
                self_workbuf_d += A_blk_bytes;
            } else {
                engine->A_trans_d = engine->A_rd_recv_d;
            }

            if (engine->trans_B)
            {
                engine->B_trans_d = (void *) self_workbuf_d;
                self_workbuf_d += B_blk_bytes;
            } else {
                engine->B_trans_d = engine->B_rd_recv_d;
            }

            if (engine->task_n_num > 1)
            {
                engine->A_2dmm_d = (void *) self_workbuf_d;
                self_workbuf_d += A_blk_bytes;
            } else {
                engine->A_2dmm_d = engine->A_trans_d;
            }

            if (engine->task_m_num > 1)
            {
                engine->B_2dmm_d = (void *) self_workbuf_d;
                self_workbuf_d += B_blk_bytes;
            } else {
                engine->B_2dmm_d = engine->B_trans_d;
            }

            engine->C_2dmm_d = (void *) self_workbuf_d;
            self_workbuf_d += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;

            engine->C_out_d = (void *) self_workbuf_d;
            self_workbuf_d += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;
        }
        if ((engine->is_BTB == 1) && (engine->is_active)) 
        {
            engine->A_2dmm_d = (void *) self_workbuf_d;
            self_workbuf_d += A_blk_bytes;

            engine->A_trans_d = (void *) self_workbuf_d;
            self_workbuf_d += A_blk_bytes;

            engine->B_2dmm_d = (void *) self_workbuf_d;
            self_workbuf_d += B_blk_bytes;

            engine->C_2dmm_d = (void *) self_workbuf_d;
            self_workbuf_d += sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;

            engine->C_out_d = (void *) self_workbuf_d;
            self_workbuf_d += sizeof(double) * engine->C_out_nrow  * engine->C_out_ncol;
        }
    }
    #endif
}

// Free a ca3dmm_engine structure
void ca3dmm_engine_free(ca3dmm_engine_p *engine_)
{
    ca3dmm_engine_p engine = *engine_;
    if (engine == NULL) return;
    free(engine->AB_agv_recvcnts);
    free(engine->AB_agv_displs);
    free(engine->C_rs_recvcnts);
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
    ca3dmm_engine_p engine, const void *src_A, const int ldA,
    const void *src_B, const int ldB, void *dst_C, const int ldC
)
{
    if (engine == NULL)
    {
        ERROR_PRINTF("ca3dmm_engine not initialized\n");
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
    double *A_rd_recv_h = (double *) engine->A_rd_recv_h;
    double *A_2dmm_h    = (double *) engine->A_2dmm_h;
    double *A_trans_h   = (double *) engine->A_trans_h;
    double *B_rd_recv_h = (double *) engine->B_rd_recv_h;
    double *B_2dmm_h    = (double *) engine->B_2dmm_h;
    double *B_trans_h   = (double *) engine->B_trans_h;
    double *C_2dmm_h    = (double *) engine->C_2dmm_h;
    double *C_out_h     = (double *) engine->C_out_h;
    double *A_rd_recv_d = (double *) engine->A_rd_recv_d;
    double *A_2dmm_d    = (double *) engine->A_2dmm_d;
    double *A_trans_d   = (double *) engine->A_trans_d;
    double *B_rd_recv_d = (double *) engine->B_rd_recv_d;
    double *B_2dmm_d    = (double *) engine->B_2dmm_d;
    double *B_trans_d   = (double *) engine->B_trans_d;
    double *C_2dmm_d    = (double *) engine->C_2dmm_d;
    double *C_out_d     = (double *) engine->C_out_d;
    dev_type_t dev_type = engine->dev_type;

    double start_t, stop_t, exec_start_t, exec_stop_t, hd_start_t, hd_stop_t;
    double redist_ms, agvAB_ms, cannon_ms, reduce_ms, exec_ms, hd_trans_ms;
    hd_trans_ms = 0.0;
    
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
        if (dev_type == DEV_TYPE_HOST)
        {
            mat_redist_engine_exec(engine->redist_A, src_A, ldA, A_rd_recv_h, recv_ldA);
            mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_rd_recv_h, recv_ldB);
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            mat_redist_engine_exec(engine->redist_A, src_A, ldA, A_rd_recv_d, recv_ldA);
            mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_rd_recv_d, recv_ldB);
            hd_trans_ms += engine->redist_A->hd_trans_ms;
            hd_trans_ms += engine->redist_B->hd_trans_ms;
        }
        #endif
        stop_t = MPI_Wtime();
        redist_ms = 1000.0 * (stop_t - start_t);
        engine->redist_ms += redist_ms;
        if (engine->print_timing) INFO_PRINTF("Redistribute A & B time = %.2f ms\n", redist_ms);

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
        if (dev_type == DEV_TYPE_HOST)
        {
            if (trans_A) transpose_cm_mat(A_trans_nrow, A_trans_ncol, A_rd_recv_h, A_trans_ncol, A_trans_h, A_2dmm_nrow);
            else A_trans_h = A_rd_recv_h;
            if (trans_B) transpose_cm_mat(B_trans_nrow, B_trans_ncol, B_rd_recv_h, B_trans_ncol, B_trans_h, B_2dmm_nrow);
            else B_trans_h = B_rd_recv_h;
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            // Use cublasDgeam to transpose a matrix
            if (trans_A)
            {
                cuda_cublas_dgeam(
                    CublasTrans, CublasNoTrans, A_trans_nrow, A_trans_ncol, 
                    1.0, A_rd_recv_d, A_trans_ncol,
                    0.0, NULL, A_trans_nrow,
                    A_trans_d, A_2dmm_nrow
                );
            } else {
                A_trans_d = A_rd_recv_d;
            }
            if (trans_B)
            {
                cuda_cublas_dgeam(
                    CublasTrans, CublasNoTrans, B_trans_nrow, B_trans_ncol, 
                    1.0, B_rd_recv_d, B_trans_ncol,
                    0.0, NULL, B_trans_nrow,
                    B_trans_d, B_trans_nrow
                );
            } else {
                B_trans_d = B_rd_recv_d;
            }
        }
        #endif

        // Allgatherv A or B to make it complete
        if (engine->task_m_num > 1)
        {
            void *B_trans_ptr, *B_2dmm_ptr;
            if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
            {
                B_trans_ptr = B_trans_h;
                B_2dmm_ptr  = B_2dmm_h;
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
            {
                B_trans_ptr = B_trans_d;
                B_2dmm_ptr  = B_2dmm_d;
            }
            if (dev_type == DEV_TYPE_CUDA)
            {
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(B_trans_h, B_trans_d, sizeof(double) * B_rd_nrow * B_rd_ncol, DEV_TYPE_HOST, dev_type);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
            if (engine->use_ag)
            {
                MPI_Allgather(
                    B_trans_ptr, B_rd_nrow * B_rd_ncol, MPI_DOUBLE, B_2dmm_ptr,
                    AB_agv_recvcnts[0], MPI_DOUBLE, engine->comm_AB_agv
                );
            } else {
                MPI_Allgatherv(
                    B_trans_ptr, B_rd_nrow * B_rd_ncol, MPI_DOUBLE, B_2dmm_ptr,
                    AB_agv_recvcnts, AB_agv_displs, MPI_DOUBLE, engine->comm_AB_agv
                );
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA)
            {
                size_t B_2dmm_bytes = sizeof(double) * engine->B_2dmm_nrow * engine->B_2dmm_ncol;
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(B_2dmm_d, B_2dmm_h, B_2dmm_bytes, dev_type, DEV_TYPE_HOST);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
        } else {
            if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA)) 
                B_2dmm_h = B_trans_h;
            #ifdef USE_CUDA
            if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
                B_2dmm_d = B_trans_d;
            #endif
        }  // End of "if (engine->task_m_num > 1)"

        if (engine->task_n_num > 1)
        {
            double *A_trans_ptr, *A_2dmm_ptr;
            if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
            {
                A_trans_ptr = A_trans_h;
                A_2dmm_ptr  = A_2dmm_h;
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
            {
                A_trans_ptr = A_trans_d;
                A_2dmm_ptr  = A_2dmm_d;
            }
            if (dev_type == DEV_TYPE_CUDA)
            {
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(A_trans_h, A_trans_d, sizeof(double) * A_rd_nrow * A_rd_ncol, DEV_TYPE_HOST, dev_type);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
            if (engine->use_ag)
            {
                MPI_Allgather(
                    A_trans_ptr, A_rd_nrow * A_rd_ncol, MPI_DOUBLE, A_2dmm_ptr,
                    AB_agv_recvcnts[0], MPI_DOUBLE, engine->comm_AB_agv
                );
            } else {
                MPI_Allgatherv(
                    A_trans_ptr, A_rd_nrow * A_rd_ncol, MPI_DOUBLE, A_2dmm_ptr,
                    AB_agv_recvcnts, AB_agv_displs, MPI_DOUBLE, engine->comm_AB_agv
                );
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA)
            {
                size_t A_2dmm_bytes = sizeof(double) * engine->A_2dmm_nrow * engine->A_2dmm_ncol;
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(A_2dmm_d, A_2dmm_h, A_2dmm_bytes, dev_type, DEV_TYPE_HOST);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
        } else {
            if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA)) 
                A_2dmm_h = A_trans_h;
            #ifdef USE_CUDA
            if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
                A_2dmm_d = A_trans_d;
            #endif
        }  // End of "if (engine->task_n_num > 1)"
        stop_t = MPI_Wtime();
        agvAB_ms = 1000.0 * (stop_t - start_t);
        engine->agvAB_ms += agvAB_ms;
        if (engine->print_timing) INFO_PRINTF("Allgather A or B time   = %.2f ms\n", agvAB_ms);
    } else {
        start_t = MPI_Wtime();
        if (dev_type == DEV_TYPE_HOST)
            mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_2dmm_h, B_2dmm_nrow);
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            mat_redist_engine_exec(engine->redist_B, src_B, ldB, B_2dmm_d, B_2dmm_nrow);
            hd_trans_ms += engine->redist_B->hd_trans_ms;
        }
        #endif
        stop_t = MPI_Wtime();
        redist_ms = 1000.0 * (stop_t - start_t);
        engine->redist_ms += redist_ms;
        if (engine->print_timing) INFO_PRINTF("Redistribute A & B time = %.2f ms\n", redist_ms);

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
                if (dev_type == DEV_TYPE_HOST)
                {
                    #pragma omp parallel for 
                    for (int i = 0; i < B_2dmm_nrow * B_2dmm_ncol; i++)
                        A_trans_h[i] = B_2dmm_h[i];
                }
                #ifdef USE_CUDA
                if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
                    cuda_cublas_dcopy(B_2dmm_nrow * B_2dmm_ncol, B_2dmm_d, 1, A_trans_d, 1);
                #endif
            } else {
                int ce_pair_rank = ce_rank_col * ce->np_dim + ce_rank_row;
                double *B_2dmm_ptr, *A_trans_ptr;
                if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
                {
                    B_2dmm_ptr  = B_2dmm_h;
                    A_trans_ptr = A_trans_h;
                }
                #ifdef USE_CUDA
                if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
                {
                    B_2dmm_ptr  = B_2dmm_d;
                    A_trans_ptr = A_trans_d;
                }
                if (dev_type == DEV_TYPE_CUDA)
                {
                    hd_start_t = MPI_Wtime();
                    dev_type_memcpy(B_2dmm_h, B_2dmm_d, sizeof(double) * B_2dmm_nrow * B_2dmm_ncol, DEV_TYPE_HOST, dev_type);
                    hd_stop_t = MPI_Wtime();
                    hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
                }
                #endif
                MPI_Sendrecv(
                    B_2dmm_ptr,  B_2dmm_nrow * B_2dmm_ncol, MPI_DOUBLE, ce_pair_rank, 0,
                    A_trans_ptr, A_2dmm_nrow * A_2dmm_ncol, MPI_DOUBLE, ce_pair_rank, 0, ce->comm, MPI_STATUS_IGNORE
                );
                #ifdef USE_CUDA
                if (dev_type == DEV_TYPE_CUDA)
                {
                    hd_start_t = MPI_Wtime();
                    dev_type_memcpy(A_trans_d, A_trans_h, sizeof(double) * A_2dmm_nrow * A_2dmm_ncol, dev_type, DEV_TYPE_HOST);
                    hd_stop_t = MPI_Wtime();
                    hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
                }
                #endif
            }
            if (dev_type == DEV_TYPE_HOST)
                transpose_cm_mat(A_2dmm_nrow, A_2dmm_ncol, A_trans_h, A_2dmm_ncol, A_2dmm_h, A_2dmm_nrow);
            #ifdef USE_CUDA
            if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
            {
                cuda_cublas_dgeam(
                    CublasTrans, CublasNoTrans, A_2dmm_nrow, A_2dmm_ncol, 
                    1.0, A_trans_d, A_2dmm_ncol,
                    0.0, NULL, A_2dmm_nrow,
                    A_2dmm_d, A_2dmm_nrow
                );
            }
            #endif
        }  // End of "if (engine->is_active == 1)"
        stop_t = MPI_Wtime();
        agvAB_ms = 1000.0 * (stop_t - start_t);
        engine->agvAB_ms += agvAB_ms;
        if (engine->print_timing) INFO_PRINTF("Allgather A or B time   = %.2f ms\n", agvAB_ms);
    }  // End of "if (engine->is_BTB == 0)"

    if (engine->is_active == 1)
    {
        start_t = MPI_Wtime();
        if (dev_type == DEV_TYPE_HOST)
        {
            if (engine->task_k_num == 1) C_2dmm_h = C_out_h;
            cannon_engine_exec(engine->cannon_engine, 1.0, 0.0, A_2dmm_h, B_2dmm_h, C_2dmm_h);
        }
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            if (engine->task_k_num == 1) C_2dmm_d = C_out_d;
            cannon_engine_exec(engine->cannon_engine, 1.0, 0.0, A_2dmm_d, B_2dmm_d, C_2dmm_d);
        }
        #endif
        stop_t = MPI_Wtime();
        cannon_ms = 1000.0 * (stop_t - start_t);
        engine->cannon_ms += cannon_ms;
        if (engine->print_timing) INFO_PRINTF("2D Cannon time          = %.2f ms\n", cannon_ms);

        start_t = MPI_Wtime();
        if (engine->task_k_num > 1)
        {
            double *C_2dmm_ptr, *C_out_ptr;
            if ((dev_type == DEV_TYPE_HOST) || (dev_type == DEV_TYPE_CUDA))
            {
                C_2dmm_ptr = C_2dmm_h;
                C_out_ptr  = C_out_h;
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA_MPI_DIRECT)
            {
                C_2dmm_ptr = C_2dmm_d;
                C_out_ptr  = C_out_d;
            }
            if (dev_type == DEV_TYPE_CUDA)
            {
                size_t C_2dmm_bytes = sizeof(double) * engine->C_2dmm_nrow * engine->C_2dmm_ncol;
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(C_2dmm_h, C_2dmm_d, C_2dmm_bytes, DEV_TYPE_HOST, dev_type);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
            MPI_Op op_omp_sum = MPI_SUM;
            MPI_Op_omp_sum_get(&op_omp_sum);
            if (engine->use_rsb)
            {
                MPI_Reduce_scatter_block(
                    C_2dmm_ptr, C_out_ptr, engine->C_rs_recvcnts[0], MPI_DOUBLE, 
                    op_omp_sum, engine->comm_C_rs
                );
            } else {
                MPI_Reduce_scatter(
                    C_2dmm_ptr, C_out_ptr, engine->C_rs_recvcnts, MPI_DOUBLE, 
                    op_omp_sum, engine->comm_C_rs
                );
            }
            #ifdef USE_CUDA
            if (dev_type == DEV_TYPE_CUDA)
            {
                size_t C_out_bytes = sizeof(double) * engine->C_out_nrow * engine->C_out_ncol;
                hd_start_t = MPI_Wtime();
                dev_type_memcpy(C_out_d, C_out_h, C_out_bytes, dev_type, DEV_TYPE_HOST);
                hd_stop_t = MPI_Wtime();
                hd_trans_ms += 1000.0 * (hd_stop_t - hd_start_t);
            }
            #endif
        }
        stop_t = MPI_Wtime();
        reduce_ms = 1000.0 * (stop_t - start_t);
        engine->reduce_ms += reduce_ms;
        if (engine->print_timing)
        {
            INFO_PRINTF("Reduce-scatter C time   = %.2f ms\n", reduce_ms);
            INFO_PRINTF("CA3DMM matmul time      = %.2f ms\n", agvAB_ms + cannon_ms + reduce_ms);
        }
    }

    engine->hd_trans_ms += hd_trans_ms;

    start_t = MPI_Wtime();
    if (engine->redist_C != NULL)
    {
        if (dev_type == DEV_TYPE_HOST)
            mat_redist_engine_exec(engine->redist_C, engine->C_out_h, engine->C_out_nrow, dst_C, ldC);
        #ifdef USE_CUDA
        if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        {
            mat_redist_engine_exec(engine->redist_C, engine->C_out_d, engine->C_out_nrow, dst_C, ldC);
            hd_trans_ms += engine->redist_C->hd_trans_ms;
        }
        #endif
    }
    stop_t = MPI_Wtime();
    redist_ms = 1000.0 * (stop_t - start_t);
    engine->redist_ms += redist_ms;
    if (engine->print_timing) INFO_PRINTF("Redistribute C time     = %.2f ms\n", redist_ms);

    // No need to add a global barrier here since mat_redist_engine_exec() already has one
    exec_stop_t = MPI_Wtime();
    exec_ms = 1000.0 * (exec_stop_t - exec_start_t);
    engine->exec_ms += exec_ms;
    if (engine->print_timing) INFO_PRINTF("CA3DMM exec time        = %.2f ms\n\n", exec_ms);
    engine->n_exec++;
}


// Reset the statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_reset_stat(ca3dmm_engine_p engine)
{
    if (engine == NULL) return;
    cannon_engine_reset_stat(engine->cannon_engine);
    engine->redist_ms   = 0.0;
    engine->agvAB_ms    = 0.0;
    engine->cannon_ms   = 0.0;
    engine->reduce_ms   = 0.0;
    engine->hd_trans_ms = 0.0;
    engine->exec_ms     = 0.0;
    engine->n_exec      = 0;
}

// Print the statistic data of a ca3dmm_engine (not a collective call)
void ca3dmm_engine_print_stat(ca3dmm_engine_p engine)
{
    if (engine == NULL) return;
    if (engine->n_exec == 0)
    {
        WARNING_PRINTF("No ca3dmm_engine statistic data to print\n");
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
    if (engine->dev_type == DEV_TYPE_CUDA)
        printf("  * CUDA H <-> D memcpy  : %.2f ms\n", engine->hd_trans_ms / engine->n_exec);
    cannon_engine_print_stat(engine->cannon_engine);
    printf("============================================================\n");
}
