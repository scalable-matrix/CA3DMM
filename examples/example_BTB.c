#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "ca3dmm.h"
#include "example_utils.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    srand48(time(NULL));

    int my_rank, n_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    if ((argc == 2) && ((strcmp(argv[1], "--help") == 0) || (strcmp(argv[1], "-h") == 0)))
    {
        printf("Usage: %s m n k transA(0 or 1) transB(0 or 1) check_correct(0 or 1) n_test\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int m = get_int_param(argc, argv, 1, 4096, 1, 8388608);
    int n = get_int_param(argc, argv, 2, 4096, 1, 8388608);
    int k = get_int_param(argc, argv, 3, 4096, 1, 8388608);
    int trans_A = get_int_param(argc, argv, 4, 0, 0, 1);
    int trans_B = get_int_param(argc, argv, 5, 0, 0, 1);
    int chk_res = get_int_param(argc, argv, 6, 1, 0, 1);
    int n_test  = get_int_param(argc, argv, 7, 10, 1, 100);

    if (m != n)
    {
        m = (m < n) ? m : n;
        n = (n < m) ? n : m;
        if (my_rank == 0) printf("Forced using m = n = %d for B^T * B\n", m);
    }

    if (my_rank == 0)
    {
        printf("Test problem size m * n * k : %d * %d * %d\n", m, n, k);
        printf("Transpose A / B             : %d / %d\n", trans_A, trans_B);
        printf("Number of tests             : %d\n", n_test);
        printf("Check result correctness    : %d\n", chk_res);
        printf("\n");
        fflush(stdout);
    }

    // Initial distribution: 1D column partition of B
    int n_spos, local_n_size;
    calc_block_size_pos(n, n_proc, my_rank, &local_n_size, &n_spos);
    size_t input_B_msize = sizeof(double) * (size_t) k * (size_t) local_n_size;
    double *B_in = (double *) malloc(input_B_msize);
    for (int j = 0; j < local_n_size; j++)
    {
        int global_j = j + n_spos;
        size_t jcol_offset = (size_t) j * (size_t) k;
        double *B_in_jcol = B_in + jcol_offset;
        for (int global_i = 0; global_i < k; global_i++)
            B_in_jcol[global_i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
    }

    // Initialize ca3dmm_engine
    ca3dmm_engine_p ce;
    ca3dmm_engine_init_BTB(
        n, k, 0, k, n_spos, local_n_size,
        MPI_COMM_WORLD, &ce
    );
    if (ce->my_rank == 0)
    {
        int mb = (m + ce->mp - 1) / ce->mp;
        int nb = (n + ce->np - 1) / ce->np;
        int kb = (k + ce->kp - 1) / ce->kp;
        double min_comm_vol = (double) m * (double) n * (double) k / (double) n_proc;
        min_comm_vol = 3.0 * pow(min_comm_vol, 2.0/3.0) * (double) n_proc;
        double curr_comm_vol = (double) (mb * nb) + (double) (mb * kb) + (double) (nb * kb);
        curr_comm_vol *= (double) n_proc;
        printf("CA3DMM partition info:\n");
        printf("Process grid mp * np * kp  : %d * %d * %d\n", ce->mp, ce->np, ce->kp);
        printf("Work cuboid  mb * nb * kb  : %d * %d * %d\n", mb, nb, kb);
        printf("Process utilization        : %.2f %% \n", 100.0 * (1.0 - (double) ce->rp / (double) n_proc));
        printf("Comm. volume / lower bound : %.2f\n", curr_comm_vol / min_comm_vol);
        printf("\n");
        fflush(stdout);
    }

    // Warm up running
    ca3dmm_engine_exec(NULL, 0, B_in, k, ce);
    ca3dmm_engine_reset_stat(ce);

    // Timing running
    for (int itest = 0; itest < n_test; itest++)
        ca3dmm_engine_exec(NULL, 0, B_in, k, ce);
    if (my_rank == 0) ca3dmm_engine_print_stat(ce);

    // Check the correctness of the result
    if (chk_res)
    {
        int chk_m_spos = ce->C_out_srow;
        int chk_m_size = ce->C_out_nrow;
        int chk_n_spos = ce->C_out_scol;
        int chk_n_size = ce->C_out_ncol;
        size_t chk_BT_msize = sizeof(double) * (size_t) k * (size_t) chk_m_size;
        size_t chk_B_msize  = sizeof(double) * (size_t) k * (size_t) chk_n_size;
        size_t chk_C_msize  = sizeof(double) * (size_t) chk_m_size * (size_t) chk_n_size;
        double *BT_chk = (double *) malloc(chk_BT_msize);
        double *B_chk  = (double *) malloc(chk_B_msize);
        double *C_chk  = (double *) malloc(chk_C_msize);
        for (int j = 0; j < chk_m_size; j++)
        {
            int global_j = j + chk_m_spos;
            size_t jcol_offset = (size_t) j * (size_t) k;
            double *BT_chk_jcol = BT_chk + jcol_offset;
            for (int global_i = 0; global_i < k; global_i++)
                BT_chk_jcol[global_i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }
        for (int j = 0; j < chk_n_size; j++)
        {
            int global_j = j + chk_n_spos;
            size_t jcol_offset = (size_t) j * (size_t) k;
            double *B_chk_jcol = B_chk + jcol_offset;
            for (int global_i = 0; global_i < k; global_i++)
                B_chk_jcol[global_i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }

        if (ce->is_active)
        {
            cblas_dgemm(
                CblasColMajor, CblasTrans, CblasNoTrans, chk_m_size, chk_n_size, k,
                1.0, BT_chk, k, B_chk, k, 0.0, C_chk, chk_m_size
            );
        }
        
        int local_error = 0, total_error = 0;
        if (ce->is_active)
        {
            double *C = ce->C_out;
            for (int j = 0; j < chk_n_size; j++)
            {
                size_t offset = (size_t) j * (size_t) chk_m_size;
                double *C_jcol = C + offset;
                double *C_chk_jcol = C_chk + offset;
                for (int i = 0; i < chk_m_size; i++)
                {
                    double diff = C_jcol[i] - C_chk_jcol[i];
                    double relerr = fabs(diff / C_chk[i]);
                    if (relerr > 1e-12) local_error++;
                }
            }
        }
        MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_rank == 0) printf("CA3DMM output : %d error(s)\n", total_error);

        free(BT_chk);
        free(B_chk);
        free(C_chk);
    }

    free(B_in);
    ca3dmm_engine_free(&ce);
    MPI_Finalize();
    return 0;
}