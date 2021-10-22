#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

#include "memory.h"
#include "enum.h"
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
    int use_gpu  = get_int_param(argc, argv, 8, 0, 0, 1);

    device_type dev = (use_gpu)?DEVICE_TYPE_DEVICE:DEVICE_TYPE_HOST;
    device_type compute_device = dev;
    device_type communication_device = DEVICE_TYPE_HOST;

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
    int B_nrow = k, B_ncol = n;
    int C_nrow = n, C_ncol = n;

    // Initial distribution: 1D column partition of B
    // Output distribution: 1D column partition of C
    int B_in_srow, B_in_nrow, B_in_scol, B_in_ncol;
    int C_out_srow, C_out_nrow, C_out_scol, C_out_ncol;
    B_in_srow  = 0;
    B_in_nrow  = B_nrow;
    C_out_srow = 0;
    C_out_nrow = C_nrow;
    calc_block_size_pos(B_ncol, n_proc, my_rank, &B_in_ncol,  &B_in_scol);
    calc_block_size_pos(C_ncol, n_proc, my_rank, &C_out_ncol, &C_out_scol);
    size_t B_in_msize  = sizeof(double) * (size_t) B_in_nrow  * (size_t) B_in_ncol;
    size_t C_out_msize = sizeof(double) * (size_t) C_out_nrow * (size_t) C_out_ncol;
    double *B_in  = (double *) malloc(B_in_msize);
    double *C_out = (double *) malloc(C_out_msize);

    double *B_in_comm  = _OUR_MALLOC(B_in_msize, communication_device);
    double *C_out_comm  = _OUR_MALLOC(C_out_msize, communication_device);
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
    ca3dmm_engine_init_BTB_ex(
        n, k, B_in_srow, B_in_nrow, B_in_scol, B_in_ncol,
        C_out_srow, C_out_nrow, C_out_scol, C_out_ncol,
        communication_device, compute_device,
        NULL, MPI_COMM_WORLD, &ce
    );
    OUR_MEMCPY(B_in_comm, B_in, B_in_msize, communication_device, DEVICE_TYPE_HOST);
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
    ca3dmm_engine_exec(NULL, 0, B_in_comm, B_in_nrow, C_out_comm, C_out_nrow, ce);
    ca3dmm_engine_reset_stat(ce);

    // Timing running
    for (int itest = 0; itest < n_test; itest++)
        ca3dmm_engine_exec(NULL, 0, B_in_comm, B_in_nrow, C_out_comm, C_out_nrow, ce);
    OUR_MEMCPY(C_out, C_out_comm, C_out_msize, DEVICE_TYPE_HOST, communication_device);
    if (my_rank == 0) ca3dmm_engine_print_stat(ce);

    // Check the correctness of the result
    if (chk_res)
    {
        int chk_m_spos = C_out_srow;
        int chk_m_size = C_out_nrow;
        int chk_n_spos = C_out_scol;
        int chk_n_size = C_out_ncol;
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


        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans, chk_m_size, chk_n_size, k,
            1.0, BT_chk, k, B_chk, k, 0.0, C_chk, chk_m_size
        );
        
        int local_error = 0, total_error = 0;
        for (int j = 0; j < chk_n_size; j++)
        {
            size_t out_offset = (size_t) j * (size_t) C_out_nrow;
            size_t chk_offset = (size_t) j * (size_t) chk_m_size;
            double *C_out_jcol = C_out + out_offset;
            double *C_chk_jcol = C_chk + chk_offset;
            for (int i = 0; i < chk_m_size; i++)
            {
                double diff = C_out_jcol[i] - C_chk_jcol[i];
                double relerr = fabs(diff / C_chk_jcol[i]);
                if (relerr > 1e-12) local_error++;
            }
        }
        MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_rank == 0) printf("CA3DMM output : %d error(s)\n", total_error);

        free(BT_chk);
        free(B_chk);
        free(C_chk);
    }
    OUR_FREE(B_in_comm, communication_device);
    OUR_FREE(C_out_comm, communication_device);

    free(B_in);
    free(C_out);
    ca3dmm_engine_free(&ce);
    MPI_Finalize();
    return 0;
}
