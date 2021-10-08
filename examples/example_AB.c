#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "ca3dmm.h"
#include "example_utils.h"
#include "utils.h"  // in CA3DMM's include/

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

    if (my_rank == 0)
    {
        printf("Test problem size m * n * k : %d * %d * %d\n", m, n, k);
        printf("Transpose A / B             : %d / %d\n", trans_A, trans_B);
        printf("Number of tests             : %d\n", n_test);
        printf("Check result correctness    : %d\n", chk_res);
        printf("\n");
        fflush(stdout);
    }

    int A_nrow, A_ncol, B_nrow, B_ncol, C_nrow, C_ncol;
    if (trans_A == 0)
    {
        A_nrow = m;
        A_ncol = k;
    } else {
        A_nrow = k;
        A_ncol = m;
    }
    if (trans_B == 0)
    {
        B_nrow = k;
        B_ncol = n;
    } else {
        B_nrow = n;
        B_ncol = k;
    }
    C_nrow = m;
    C_ncol = n;

    // Initial distribution: 1D column partition of A and B
    // Output distribution: 1D column partition of C
    int A_in_srow, A_in_nrow, A_in_scol, A_in_ncol;
    int B_in_srow, B_in_nrow, B_in_scol, B_in_ncol;
    int C_out_srow, C_out_nrow, C_out_scol, C_out_ncol;
    A_in_srow  = 0;
    A_in_nrow  = A_nrow;
    B_in_srow  = 0;
    B_in_nrow  = B_nrow;
    C_out_srow = 0;
    C_out_nrow = C_nrow;
    calc_block_spos_size(A_ncol, n_proc, my_rank,  &A_in_scol, &A_in_ncol);
    calc_block_spos_size(B_ncol, n_proc, my_rank,  &B_in_scol, &B_in_ncol);
    calc_block_spos_size(C_ncol, n_proc, my_rank, &C_out_scol, &C_out_ncol);
    size_t A_in_msize  = sizeof(double) * (size_t) A_in_nrow  * (size_t) A_in_ncol;
    size_t B_in_msize  = sizeof(double) * (size_t) B_in_nrow  * (size_t) B_in_ncol;
    size_t C_out_msize = sizeof(double) * (size_t) C_out_nrow * (size_t) C_out_ncol;
    double *A_in  = (double *) malloc(A_in_msize);
    double *B_in  = (double *) malloc(B_in_msize);
    double *C_out = (double *) malloc(C_out_msize);
    for (int j = 0; j < A_in_ncol; j++)
    {
        int global_j = j + A_in_scol;
        size_t jcol_offset = (size_t) j * (size_t) A_in_nrow;
        double *A_in_jcol = A_in + jcol_offset;
        for (int i = 0; i < A_in_nrow; i++)
        {
            int global_i = i + A_in_srow;
            A_in_jcol[i] = 0.19 * (double) global_i + 0.24 * (double) global_j;
        }
    }
    for (int j = 0; j < B_in_ncol; j++)
    {
        int global_j = j + B_in_scol;
        size_t jcol_offset = (size_t) j * (size_t) B_in_nrow;
        double *B_in_jcol = B_in + jcol_offset;
        for (int i = 0; i < B_in_nrow; i++)
        {
            int global_i = i + B_in_srow;
            B_in_jcol[i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }
    }

    // Initialize ca3dmm_engine
    ca3dmm_engine_p ce;
    size_t ce_workbuf_bytes;
    ca3dmm_engine_init(
        m, n, k, trans_A, trans_B, 
        A_in_srow,  A_in_nrow,  A_in_scol,  A_in_ncol,
        B_in_srow,  B_in_nrow,  B_in_scol,  B_in_ncol,
        C_out_srow, C_out_nrow, C_out_scol, C_out_ncol,
        NULL, MPI_COMM_WORLD, &ce, &ce_workbuf_bytes
    );
    void *ce_work_buf = malloc(ce_workbuf_bytes);
    ca3dmm_engine_attach_workbuf(ce, ce_work_buf);
    if (ce->my_rank == 0)
    {
        int mb = (m + ce->mp - 1) / ce->mp;
        int nb = (n + ce->np - 1) / ce->np;
        int kb = (k + ce->kp - 1) / ce->kp;
        double min_comm_vol = (double) m * (double) n * (double) k / (double) n_proc;
        min_comm_vol = 3.0 * pow(min_comm_vol, 2.0/3.0) * (double) n_proc;
        double curr_comm_vol = (double) (mb * nb) + (double) (mb * kb) + (double) (nb * kb);
        curr_comm_vol *= (double) n_proc;
        double ce_workbuf_mb = (double) ce_workbuf_bytes / 1048576.0;
        printf("CA3DMM partition info:\n");
        printf("Process grid mp * np * kp  : %d * %d * %d\n", ce->mp, ce->np, ce->kp);
        printf("Work cuboid  mb * nb * kb  : %d * %d * %d\n", mb, nb, kb);
        printf("Process utilization        : %.2f %% \n", 100.0 * (1.0 - (double) ce->rp / (double) n_proc));
        printf("Comm. volume / lower bound : %.2f\n", curr_comm_vol / min_comm_vol);
        printf("Rank 0 work buffer size    : %.2f MBytes\n", ce_workbuf_mb);
        printf("\n");
        fflush(stdout);
    }

    // Warm up running
    ca3dmm_engine_exec(ce, A_in, A_in_nrow, B_in, B_in_nrow, C_out, C_out_nrow);
    ca3dmm_engine_reset_stat(ce);

    // Timing running
    double *redist_mss = (double *) malloc(sizeof(double) * n_test);
    double *agvAB_mss  = (double *) malloc(sizeof(double) * n_test);
    double *cannon_mss = (double *) malloc(sizeof(double) * n_test);
    double *reduce_mss = (double *) malloc(sizeof(double) * n_test);
    double *matmul_mss = (double *) malloc(sizeof(double) * n_test);
    double *exec_mss   = (double *) malloc(sizeof(double) * n_test);
    double redist_ms = 0.0, agvAB_ms = 0.0, cannon_ms = 0.0;
    double reduce_ms = 0.0, exec_ms = 0.0;
    for (int i = 0; i < n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        ca3dmm_engine_exec(ce, A_in, A_in_nrow, B_in, B_in_nrow, C_out, C_out_nrow);
        redist_mss[i] = ce->redist_ms - redist_ms;
        agvAB_mss[i]  = ce->agvAB_ms  - agvAB_ms;
        cannon_mss[i] = ce->cannon_ms - cannon_ms;
        reduce_mss[i] = ce->reduce_ms - reduce_ms;
        exec_mss[i]   = ce->exec_ms   - exec_ms;
        matmul_mss[i] = agvAB_mss[i] + cannon_mss[i] + reduce_mss[i];
        redist_ms = ce->redist_ms;
        agvAB_ms  = ce->agvAB_ms;
        cannon_ms = ce->cannon_ms;
        reduce_ms = ce->reduce_ms;
        exec_ms   = ce->exec_ms;

    }
    if (my_rank == 0)
    {
        printf("\nA, B, C redist   : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", redist_mss[i]);
        printf("\nA / B allgather  : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", agvAB_mss[i]);
        printf("\n2D Cannon        : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", cannon_mss[i]);
        printf("\nC reduce-scatter : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", reduce_mss[i]);
        printf("\nmatmul only      : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", matmul_mss[i]);
        printf("\ntotal execution  : ");
        for (int i = 0; i < n_test; i++) printf("%.0lf ", exec_mss[i]);
        printf("\n\n");
        ca3dmm_engine_print_stat(ce);
    }
    free(redist_mss);
    free(agvAB_mss);
    free(cannon_mss);
    free(reduce_mss);
    free(matmul_mss);
    free(exec_mss);

    // Check the correctness of the result
    if (chk_res)
    {
        int C_chk_srow = C_out_srow;
        int C_chk_nrow = C_out_nrow;
        int C_chk_scol = C_out_scol;
        int C_chk_ncol = C_out_ncol;
        int A_chk_srow, A_chk_scol, A_chk_nrow, A_chk_ncol;
        int B_chk_srow, B_chk_scol, B_chk_nrow, B_chk_ncol;
        if (trans_A == 0)
        {
            A_chk_srow = C_chk_srow;
            A_chk_scol = 0;
            A_chk_nrow = C_chk_nrow;
            A_chk_ncol = A_ncol;
        } else {
            A_chk_srow = 0;
            A_chk_scol = C_chk_srow;
            A_chk_nrow = A_nrow;
            A_chk_ncol = C_chk_nrow;
        }
        if (trans_B == 0)
        {
            B_chk_srow = 0;
            B_chk_scol = C_chk_scol;
            B_chk_nrow = B_nrow;
            B_chk_ncol = C_chk_ncol;
        } else {
            B_chk_srow = C_chk_scol;
            B_chk_scol = 0;
            B_chk_nrow = C_chk_ncol;
            B_chk_ncol = B_ncol;
        }
        size_t A_chk_msize = sizeof(double) * (size_t) A_chk_nrow * (size_t) A_chk_ncol;
        size_t B_chk_msize = sizeof(double) * (size_t) B_chk_nrow * (size_t) B_chk_ncol;
        size_t C_chk_msize = sizeof(double) * (size_t) C_chk_nrow * (size_t) C_chk_ncol;
        double *A_chk = (double *) malloc(A_chk_msize);
        double *B_chk = (double *) malloc(B_chk_msize);
        double *C_chk = (double *) malloc(C_chk_msize);
        for (int j = 0; j < A_chk_ncol; j++)
        {
            int global_j = j + A_chk_scol;
            size_t jcol_offset = (size_t) j * (size_t) A_chk_nrow;
            double *A_chk_jcol = A_chk + jcol_offset;
            for (int i = 0; i < A_chk_nrow; i++)
            {
                int global_i = i + A_chk_srow;
                A_chk_jcol[i] = 0.19 * (double) global_i + 0.24 * (double) global_j;
            }
        }
        for (int j = 0; j < B_chk_ncol; j++)
        {
            int global_j = j + B_chk_scol;
            size_t jcol_offset = (size_t) j * (size_t) B_chk_nrow;
            double *B_chk_jcol = B_chk + jcol_offset;
            for (int i = 0; i < B_chk_nrow; i++)
            {
                int global_i = i + B_chk_srow;
                B_chk_jcol[i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
            }
        }

        CBLAS_TRANSPOSE A_trans = (trans_A) ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE B_trans = (trans_B) ? CblasTrans : CblasNoTrans;
        cblas_dgemm(
            CblasColMajor, A_trans, B_trans, C_chk_nrow, C_chk_ncol, k,
            1.0, A_chk, A_chk_nrow, B_chk, B_chk_nrow, 0.0, C_chk, C_chk_nrow
        );
        
        int local_error = 0, total_error = 0;
        for (int j = 0; j < C_chk_ncol; j++)
        {
            size_t out_offset = (size_t) j * (size_t) C_out_nrow;
            size_t chk_offset = (size_t) j * (size_t) C_chk_nrow;
            double *C_out_jcol = C_out + out_offset;
            double *C_chk_jcol = C_chk + chk_offset;
            for (int i = 0; i < C_chk_nrow; i++)
            {
                double diff = C_out_jcol[i] - C_chk_jcol[i];
                double relerr = fabs(diff / C_chk_jcol[i]);
                if (relerr > 1e-12) local_error++;
            }
        }
    MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_rank == 0) printf("CA3DMM output : %d error(s)\n", total_error);
        free(A_chk);
        free(B_chk);
        free(C_chk);
    }

    free(A_in);
    free(B_in);
    free(C_out);
    ca3dmm_engine_free(&ce);
    free(ce_work_buf);
    MPI_Finalize();
    return 0;
}