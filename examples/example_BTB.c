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
    if ((argc == 2) && ((strcmp(argv[1], "--help") == 0) || (strcmp(argv[1], "-h") == 0)))
    {
        printf("Usage: %s m k check_correct(0 or 1) n_test dev_type\n", argv[0]);
        return 0;
    }

    int m = get_int_param(argc, argv, 1, 4096, 1, 8388608);
    int n = m;
    int k = get_int_param(argc, argv, 2, 4096, 1, 8388608);
    int chk_res = get_int_param(argc, argv, 3, 1, 0, 1);
    int n_test  = get_int_param(argc, argv, 4, 10, 1, 100);
    dev_type_t dev_type = get_int_param(argc, argv, 5, DEV_TYPE_HOST, DEV_TYPE_HOST, DEV_TYPE_CUDA_MPI_DIRECT);

    #ifdef USE_CUDA
    if ((dev_type == DEV_TYPE_CUDA) || (dev_type == DEV_TYPE_CUDA_MPI_DIRECT))
        select_cuda_device_by_mpi_local_rank();
    #endif

    srand48(time(NULL));
    int my_rank, n_proc;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

    if (my_rank == 0)
    {
        printf("Test problem size m * n * k : %d * %d * %d\n", m, n, k);
        printf("Number of tests             : %d\n", n_test);
        printf("Check result correctness    : %d\n", chk_res);
        printf("Device type                 : %d\n", dev_type);
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
    calc_block_spos_size(B_ncol, n_proc, my_rank,  &B_in_scol, &B_in_ncol);
    calc_block_spos_size(C_ncol, n_proc, my_rank, &C_out_scol, &C_out_ncol);
    size_t B_in_msize  = sizeof(double) * (size_t) B_in_nrow  * (size_t) B_in_ncol;
    size_t C_out_msize = sizeof(double) * (size_t) C_out_nrow * (size_t) C_out_ncol;
    double *B_in_h  = (double *) dev_type_malloc(B_in_msize, DEV_TYPE_HOST);
    double *B_in_d  = (double *) dev_type_malloc(B_in_msize, dev_type);
    double *C_out_d = (double *) dev_type_malloc(C_out_msize, dev_type);
    for (int j = 0; j < B_in_ncol; j++)
    {
        int global_j = j + B_in_scol;
        size_t jcol_offset = (size_t) j * (size_t) B_in_nrow;
        double *B_in_jcol = B_in_h + jcol_offset;
        for (int i = 0; i < B_in_nrow; i++)
        {
            int global_i = i + B_in_srow;
            B_in_jcol[i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }
    }
    dev_type_memcpy(B_in_d, B_in_h, B_in_msize, dev_type, DEV_TYPE_HOST);

    // Initialize ca3dmm_engine
    ca3dmm_engine_p ce;
    size_t ce_workbuf_bytes;
    ca3dmm_engine_init_BTB(
        n, k, B_in_srow, B_in_nrow, B_in_scol, B_in_ncol,
        C_out_srow, C_out_nrow, C_out_scol, C_out_ncol,
        NULL, MPI_COMM_WORLD, dev_type, &ce, &ce_workbuf_bytes
    );
    void *workbuf_h, *workbuf_d;
    MALLOC_ATTACH_WORKBUF(
        ca3dmm_engine_attach_workbuf, ca3dmm_engine_free, 
        ce, dev_type, ce_workbuf_bytes, workbuf_h, workbuf_d
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
    ca3dmm_engine_exec(ce, NULL, 0, B_in_d, B_in_nrow, C_out_d, C_out_nrow);
    ca3dmm_engine_reset_stat(ce);

    // Timing running
    for (int itest = 0; itest < n_test; itest++)
        ca3dmm_engine_exec(ce, NULL, 0, B_in_d, B_in_nrow, C_out_d, C_out_nrow);
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
        double *BT_chk_h = (double *) dev_type_malloc(chk_BT_msize, DEV_TYPE_HOST);
        double *B_chk_h  = (double *) dev_type_malloc(chk_B_msize, DEV_TYPE_HOST);
        double *C_chk_h  = (double *) dev_type_malloc(chk_C_msize, DEV_TYPE_HOST);
        double *C_out_h  = (double *) dev_type_malloc(C_out_msize, DEV_TYPE_HOST);
        for (int j = 0; j < chk_m_size; j++)
        {
            int global_j = j + chk_m_spos;
            size_t jcol_offset = (size_t) j * (size_t) k;
            double *BT_chk_jcol = BT_chk_h + jcol_offset;
            for (int global_i = 0; global_i < k; global_i++)
                BT_chk_jcol[global_i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }
        for (int j = 0; j < chk_n_size; j++)
        {
            int global_j = j + chk_n_spos;
            size_t jcol_offset = (size_t) j * (size_t) k;
            double *B_chk_jcol = B_chk_h + jcol_offset;
            for (int global_i = 0; global_i < k; global_i++)
                B_chk_jcol[global_i] = 0.11 * (double) global_i + 0.12 * (double) global_j;
        }

        cblas_dgemm(
            CblasColMajor, CblasTrans, CblasNoTrans, chk_m_size, chk_n_size, k,
            1.0, BT_chk_h, k, B_chk_h, k, 0.0, C_chk_h, chk_m_size
        );
        dev_type_memcpy(C_out_h, C_out_d, C_out_msize, DEV_TYPE_HOST, dev_type);
        
        int local_error = 0, total_error = 0;
        for (int j = 0; j < chk_n_size; j++)
        {
            size_t out_offset = (size_t) j * (size_t) C_out_nrow;
            size_t chk_offset = (size_t) j * (size_t) chk_m_size;
            double *C_out_jcol = C_out_h + out_offset;
            double *C_chk_jcol = C_chk_h + chk_offset;
            for (int i = 0; i < chk_m_size; i++)
            {
                double diff = C_out_jcol[i] - C_chk_jcol[i];
                double relerr = fabs(diff / C_chk_jcol[i]);
                if (relerr > 1e-12) local_error++;
            }
        }
        MPI_Reduce(&local_error, &total_error, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (my_rank == 0) printf("CA3DMM output : %d error(s)\n", total_error);

        dev_type_free(BT_chk_h, DEV_TYPE_HOST);
        dev_type_free(B_chk_h, DEV_TYPE_HOST);
        dev_type_free(C_chk_h, DEV_TYPE_HOST);
        dev_type_free(C_out_h, DEV_TYPE_HOST);
    }

    dev_type_free(B_in_h, DEV_TYPE_HOST);
    dev_type_free(B_in_d, dev_type);
    dev_type_free(C_out_d, dev_type);
    dev_type_free(workbuf_h, DEV_TYPE_HOST);
    dev_type_free(workbuf_d, dev_type);
    ca3dmm_engine_free(&ce);
    MPI_Finalize();
    return 0;
}