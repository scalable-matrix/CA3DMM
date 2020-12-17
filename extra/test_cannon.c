#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "linalg_lib_wrapper.h"
#include "cannon.h"
#include "../examples/example_utils.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ((argc == 2) && ((strcmp(argv[1], "--help") == 0) || (strcmp(argv[1], "-h") == 0)))
    {
        printf("Usage: %s m n k check_correct(0 or 1) n_test\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int m = get_int_param(argc, argv, 1, 4096, 1, 65536);
    int n = get_int_param(argc, argv, 2, 4096, 1, 65536);
    int k = get_int_param(argc, argv, 3, 4096, 1, 65536);
    int chk_res = get_int_param(argc, argv, 4, 1, 0, 1);
    int n_test  = get_int_param(argc, argv, 5, 10, 1, 100);

    if (my_rank == 0)
    {
        printf("Test problem size m * n * k : %d * %d * %d\n", m, n, k);
        printf("Number of tests             : %d\n", n_test);
        printf("Check result correctness    : %d\n", chk_res);
        fflush(stdout);
    }

    cannon_engine_p ce;
    cannon_engine_init(m, n, k, MPI_COMM_WORLD, &ce);
    MPI_Barrier(MPI_COMM_WORLD);

    int np_dim   = ce->np_dim;
    int rank_row = ce->rank_row;
    int rank_col = ce->rank_col;
    int max_A_blk_size = (m / np_dim + 1) * (k / np_dim + 1);
    int max_B_blk_size = (k / np_dim + 1) * (n / np_dim + 1);
    int max_C_blk_size = (m / np_dim + 1) * (n / np_dim + 1);
    double *A_blk = (double *) malloc(sizeof(double) * max_A_blk_size);
    double *B_blk = (double *) malloc(sizeof(double) * max_B_blk_size);
    double *C_blk = (double *) malloc(sizeof(double) * max_C_blk_size);
    memset(C_blk, 0, sizeof(double) * max_C_blk_size);
    for (int j = 0; j < ce->A_ncol; j++)
    {
        int global_j = ce->A_scol + j;
        for (int i = 0; i < ce->A_nrow; i++)
        {
            int global_i = ce->A_srow + i;
            A_blk[i + j * ce->A_nrow] = (double) ((global_i + global_j) % 5);
        }
    }
    for (int j = 0; j < ce->B_ncol; j++)
    {
        int global_j = ce->B_scol + j;
        for (int i = 0; i < ce->B_nrow; i++)
        {
            int global_i = ce->B_srow + i;
            B_blk[i + j * ce->B_nrow] = (double) ((global_i + global_j) % 3);
        }
    }
    double alpha = 1.0, beta = 0.0;
    cannon_engine_exec(alpha, A_blk, B_blk, beta, C_blk, ce);
    cannon_engine_reset_stat(ce);
    for (int itest = 0; itest < n_test; itest++)
        cannon_engine_exec(alpha, A_blk, B_blk, beta, C_blk, ce);
    if (my_rank == 0) cannon_engine_print_stat(ce);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    if (chk_res)
    {
        double *ref_A = (double *) malloc(sizeof(double) * ce->A_nrow * k);
        double *ref_B = (double *) malloc(sizeof(double) * k * ce->B_ncol);
        double *ref_C = (double *) malloc(sizeof(double) * max_C_blk_size);
        for (int global_j = 0; global_j < k; global_j++)
        {
            for (int i = 0; i < ce->A_nrow; i++)
            {
                int global_i = ce->A_srow + i;
                ref_A[i + global_j * ce->A_nrow] = (double) ((global_i + global_j) % 5);
            }
        }
        for (int j = 0; j < ce->B_ncol; j++)
        {
            int global_j = ce->B_scol + j;
            for (int global_i = 0; global_i < k; global_i++)
            {
                ref_B[global_i + j * k] = (double) ((global_i + global_j) % 3);
            }
        }
        cblas_dgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, ce->C_nrow, ce->C_ncol, k, 
            alpha, ref_A, ce->A_nrow, ref_B, k, beta, ref_C, ce->C_nrow
        );

        int local_error = 0, total_error;
        for (int i = 0; i < ce->C_nrow * ce->C_ncol; i++)
        {
            double diff = ref_C[i] - C_blk[i];
            double relerr = fabs(diff / ref_C[i]);
            if (relerr > 1e-12) local_error++;
        }
        MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_rank == 0) printf("Cannon output : %d error(s)\n", total_error);
        free(ref_A);
        free(ref_B);
        free(ref_C);
    }

    free(A_blk);
    free(B_blk);
    free(C_blk);
    cannon_engine_free(&ce);
    MPI_Finalize();
    return 0;
}