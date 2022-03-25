#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "ca3dmm.h"
#include "example_utils.h"
#include "utils.h"
#include "cpu_linalg_lib_wrapper.h"
#ifdef USE_CUDA
#include "cuda_proxy.h"
#endif

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int my_rank, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    srand48(time(NULL) + my_rank);

    if ((argc == 2) && ((strcmp(argv[1], "--help") == 0) || (strcmp(argv[1], "-h") == 0)))
    {
        printf("Usage: %s nrow ncol\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int B_nrow = get_int_param(argc, argv, 1, 10000, 1, 16384);
    int B_ncol = get_int_param(argc, argv, 2,   500, 1, 16384);
    if (my_rank == 0) printf("Random matrix B size %d * %d\n", B_nrow, B_ncol);

    dev_type_t dev_type = DEV_TYPE_HOST;

    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        select_cuda_device_by_mpi_local_rank();
    #endif

    // Query the ideal partitioning of B and grid size. External application can 
    // organize all active processes (ce->is_active == 1) into a 2D grid by grouping
    // all ranks with the same src_B_scol into a column communicator and all ranks
    // with the same src_B_srow into a row communicator. 
    int src_B_srow, src_B_nrow, src_B_scol, src_B_ncol;
    int proc_grid[3];
    ca3dmm_engine_p ce;
    size_t ce_workbuf_bytes;
    ca3dmm_engine_init_BTB(
        B_ncol, B_nrow, 0, B_nrow, 0, B_ncol, 
        -1, -1, -1, -1, NULL, MPI_COMM_WORLD, 
        dev_type, &ce, &ce_workbuf_bytes
    );
    // Since we don't need to use ce here, we do not need to attach work buffer for it
    src_B_srow = ce->B_rd_srow;
    src_B_scol = ce->B_rd_scol;
    src_B_nrow = ce->B_rd_nrow;
    src_B_ncol = ce->B_rd_ncol;
    proc_grid[0] = ce->mp;
    proc_grid[1] = ce->np;
    proc_grid[2] = ce->kp;
    #if 0
    printf(
        "Rank %2d initial B block: [%d : %d, %d : %d]\n", my_rank,
        src_B_srow, src_B_srow + src_B_nrow - 1, src_B_scol, src_B_scol + src_B_ncol - 1
    );
    #endif
    if (my_rank == 0) printf("CA3DMM process grid : %d * %d * %d\n", proc_grid[0], proc_grid[1], proc_grid[2]);
    ca3dmm_engine_free(&ce);

    // Allocate local B block and fill it with random number
    size_t B_in_msize = sizeof(double) * src_B_nrow * src_B_ncol;
    double *B_in   = (double *) malloc(B_in_msize);
    double *B_orth = (double *) malloc(B_in_msize);
    for (int j = 0; j < src_B_ncol; j++)
    {
        int global_j = src_B_scol + j;
        size_t jcol_offset = (size_t) j * (size_t) src_B_nrow;
        double *B_in_jcol = B_in + jcol_offset;
        for (int i = 0; i < src_B_nrow; i++)
        {
            int global_i = src_B_srow + i;
            B_in_jcol[i] = drand48();
        }
    }

    // Allocate local S0 and S1 matrix
    int S0_srow, S0_nrow, S0_scol, S0_ncol;
    int S1_srow, S1_nrow, S1_scol, S1_ncol;
    S0_srow = 0;  S0_scol = 0;
    if (my_rank == 0)
    {
        S0_nrow = B_ncol;
        S0_ncol = B_ncol;
    } else {
        S0_nrow = 0;
        S0_ncol = 0;
    }
    S1_srow = 0;  S1_nrow = B_ncol;
    calc_block_spos_size(B_ncol, n_proc, my_rank, &S1_scol, &S1_ncol);
    double *S0 = (double *) malloc(sizeof(double) * S0_nrow * S0_ncol);
    double *S1 = (double *) malloc(sizeof(double) * S1_nrow * S1_ncol);

    // Compute S0 = B^T * B
    ca3dmm_engine_p ce_S0_mat;
    ca3dmm_engine_init_BTB(
        B_ncol, B_nrow, src_B_srow, src_B_nrow, src_B_scol, src_B_ncol, 
        S0_srow, S0_nrow, S0_scol, S0_ncol, &proc_grid[0], MPI_COMM_WORLD, 
        dev_type, &ce_S0_mat, NULL
    );
    ca3dmm_engine_exec(ce_S0_mat, NULL, 0, B_in, src_B_nrow, S0, S0_nrow);
    mat_redist_engine_p S0_rdB = ce_S0_mat->redist_B;
    if ((S0_rdB->n_proc_send > 1) || (S0_rdB->n_proc_recv > 1))
        printf("Oh no, rank %d need to send/recv B_orth to/from other processes in computing S1 = B_orth^T * B_orth!\n", my_rank);

    // Rank 0 calculate Rinv = inv(chol(S0, 'upper')) and store it in S0
    if (my_rank == 0)
    {
        int info;

        info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', S0_nrow, S0, S0_nrow);
        if (info != 0)
        {
            printf("Bad luck, Cholesky factorization for B^T * B failed, error = %d\n", info);
            printf("Try another run or change the size of B\n");
        }
        for (int j = 0; j < S0_nrow; j++)
        {
            size_t jcol_offset = (size_t) j * (size_t) S0_nrow;
            double *S0_jcol = S0 + jcol_offset;
            for (int i = j + 1; i < S0_nrow; i++) S0_jcol[i] = 0.0;
        }

        info = LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', S0_nrow, S0, S0_nrow);
        if (info != 0) printf("Bad luck, cannot inverse the Cholesky output upper matrix, error = %d\n", info);
    }

    // Compute B_orth = B * Rinv, B_orth has the same partition as B
    // If np > 1, we cannot reuse the initial layout of B, so swapping the
    // process grid from [np, np, kp] to [kp, np, np] might be unnecessary
    ca3dmm_engine_p ce_Borth_mat;
    proc_grid[0] = proc_grid[2];
    proc_grid[2] = proc_grid[1];
    ca3dmm_engine_init(
        B_nrow, B_ncol, B_ncol, 0, 0,
        src_B_srow, src_B_nrow, src_B_scol, src_B_ncol,
        S0_srow, S0_nrow, S0_scol, S0_ncol, 
        src_B_srow, src_B_nrow, src_B_scol, src_B_ncol,
        &proc_grid[0], MPI_COMM_WORLD, dev_type, &ce_Borth_mat, NULL
    );
    ca3dmm_engine_exec(ce_Borth_mat, B_in, src_B_nrow, S0, S0_nrow, B_orth, src_B_nrow);

    // Compute S1 = B_orth^T * B_orth and check the correctness
    // Notice: need to swap the proc_grid from [kp, np, np] to [np, np, kp] to 
    // reuse the layout of B_orth layout. The Rayleigh-Ritz procedure does not
    // have a similar step.
    proc_grid[2] = proc_grid[0];
    proc_grid[0] = proc_grid[1];
    ca3dmm_engine_p ce_S1_mat;
    ca3dmm_engine_init_BTB(
        B_ncol, B_nrow, src_B_srow, src_B_nrow, src_B_scol, src_B_ncol, 
        S1_srow, S1_nrow, S1_scol, S1_ncol, &proc_grid[0], MPI_COMM_WORLD, 
        dev_type, &ce_S1_mat, NULL
    );
    ca3dmm_engine_exec(ce_S1_mat, NULL, 0, B_orth, src_B_nrow, S1, S1_nrow);
    mat_redist_engine_p S1_rdB = ce_S1_mat->redist_B;
    if ((S1_rdB->n_proc_send > 1) || (S1_rdB->n_proc_recv > 1))
        printf("Oh no, rank %d need to send/recv B_orth to/from other processes in computing S1 = B_orth^T * B_orth!\n", my_rank);

    double err, local_max_err = 0, global_max_err;
    for (int j = 0; j < S1_ncol; j++)
    {
        int global_j = S1_scol + j;
        size_t jcol_offset = (size_t) j * (size_t) S1_nrow;
        double *S1_jcol = S1 + jcol_offset;
        for (int i = 0; i < S1_nrow; i++)
        {
            int global_i = i;
            if (global_i == global_j)
            {
                err = fabs(S1_jcol[i] - 1.0);
                local_max_err = (err > local_max_err) ? err : local_max_err;
            } else {
                err = fabs(S1_jcol[i]);
                local_max_err = (err > local_max_err) ? err : local_max_err;
            }
        }
    }
    MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0) printf("CA3DMM CholeskyQR max error : %e\n", global_max_err);
    
    ca3dmm_engine_free(&ce_S1_mat);
    ca3dmm_engine_free(&ce_Borth_mat);
    ca3dmm_engine_free(&ce_S0_mat);
    free(S1);
    free(S0);
    free(B_orth);
    free(B_in);
    MPI_Finalize();
    return 0;
}