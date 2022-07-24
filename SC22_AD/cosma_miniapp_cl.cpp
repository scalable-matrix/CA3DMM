#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cosma/cinterface.hpp"

// Partition an array into multiple same-size blocks and return the
// start position of a given block
void calc_block_size_pos(
    const int len, const int n_blk, const int i_blk,
    int *blk_size, int *blk_spos
)
{
    if (i_blk < 0 || i_blk > n_blk)
    {
        *blk_spos = -1;
        *blk_size = 0;
        return;
    }
    int rem = len % n_blk;
    int bs0 = len / n_blk;
    int bs1 = bs0 + 1;
    if (i_blk < rem)
    {
        *blk_spos = bs1 * i_blk;
        *blk_size = bs1;
    } else {
        *blk_spos = bs0 * i_blk + rem;
        *blk_size = bs0;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int n_proc, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc < 5)
    {
        if (my_rank == 0) printf("Usage: %s m n k n_test\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int n_test = atoi(argv[4]);
    if (my_rank == 0) printf("Problem size = %d * %d * %d, n_test = %d\n", m, n, k, n_test);

    // Use 1D partitioning of A, B, and C
    layout A_layout, B_layout, C_layout;
    int *A_rowsplit = (int *) malloc(sizeof(int) * 2);
    int *B_rowsplit = (int *) malloc(sizeof(int) * 2);
    int *C_rowsplit = (int *) malloc(sizeof(int) * 2);
    int *A_colsplit = (int *) malloc(sizeof(int) * (n_proc + 1));
    int *B_colsplit = (int *) malloc(sizeof(int) * (n_proc + 1));
    int *C_colsplit = (int *) malloc(sizeof(int) * (n_proc + 1));
    int *A_owners   = (int *) malloc(sizeof(int) * (n_proc + 1));
    int *B_owners   = (int *) malloc(sizeof(int) * (n_proc + 1));
    int *C_owners   = (int *) malloc(sizeof(int) * (n_proc + 1));
    int dummy;
    A_rowsplit[0] = 0;
    B_rowsplit[0] = 0;
    C_rowsplit[0] = 0;
    A_rowsplit[1] = m;
    B_rowsplit[1] = k;
    C_rowsplit[1] = m;
    for (int i = 0; i <= n_proc; i++)
    {
        calc_block_size_pos(k, n_proc, i, &dummy, &A_colsplit[i]);
        calc_block_size_pos(n, n_proc, i, &dummy, &B_colsplit[i]);
        calc_block_size_pos(n, n_proc, i, &dummy, &C_colsplit[i]);
        A_owners[i] = i;
        B_owners[i] = i;
        C_owners[i] = i;
    }
    int A_nrow = m;
    int B_nrow = k;
    int C_nrow = m;
    int A_ncol = A_colsplit[my_rank + 1] - A_colsplit[my_rank];
    int B_ncol = B_colsplit[my_rank + 1] - B_colsplit[my_rank];
    int C_ncol = C_colsplit[my_rank + 1] - C_colsplit[my_rank];
    block A_blk = {NULL, A_nrow, 0, my_rank};
    block B_blk = {NULL, B_nrow, 0, my_rank};
    block C_blk = {NULL, C_nrow, 0, my_rank};
    A_blk.data = malloc(sizeof(double) * A_nrow * A_ncol);
    B_blk.data = malloc(sizeof(double) * B_nrow * B_ncol);
    C_blk.data = malloc(sizeof(double) * C_nrow * C_ncol);
    A_layout.rowblocks    = 1;
    B_layout.rowblocks    = 1;
    C_layout.rowblocks    = 1;
    A_layout.colblocks    = n_proc;
    B_layout.colblocks    = n_proc;
    C_layout.colblocks    = n_proc;
    A_layout.rowsplit     = A_rowsplit;
    B_layout.rowsplit     = B_rowsplit;
    C_layout.rowsplit     = C_rowsplit;
    A_layout.colsplit     = A_colsplit;
    B_layout.colsplit     = B_colsplit;
    C_layout.colsplit     = C_colsplit;
    A_layout.owners       = A_owners;
    B_layout.owners       = B_owners;
    C_layout.owners       = C_owners;
    A_layout.nlocalblocks = 1;
    B_layout.nlocalblocks = 1;
    C_layout.nlocalblocks = 1;
    A_layout.localblocks  = &A_blk;
    B_layout.localblocks  = &B_blk;
    C_layout.localblocks  = &C_blk;

    double alpha = 1.0, beta = 0.0;

    dmultiply_using_layout(MPI_COMM_WORLD, "N", "N", &alpha, &A_layout, &B_layout, &beta, &C_layout);

    double *runtime_ms = (double *) malloc(sizeof(double) * n_test);
    double start_t, stop_t;
    for (int i = 0; i < n_test; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        start_t = MPI_Wtime();
        dmultiply_using_layout(MPI_COMM_WORLD, "N", "N", &alpha, &A_layout, &B_layout, &beta, &C_layout);
        MPI_Barrier(MPI_COMM_WORLD);
        stop_t = MPI_Wtime();
        runtime_ms[i] = 1000.0 * (stop_t - start_t);
    }
    if (my_rank == 0)
    {
        printf("\nCOSMA dmultiply runtime (ms) : ");
        for (int i = 0; i < n_test; i++) printf("%.0f ", runtime_ms[i]);
        printf("\n");
    }
    free(runtime_ms);

    free(A_rowsplit);
    free(A_colsplit);
    free(A_owners);
    free(B_rowsplit);
    free(B_colsplit);
    free(B_owners);
    free(C_rowsplit);
    free(C_colsplit);
    free(C_owners);
    free(A_blk.data);
    free(B_blk.data);
    free(C_blk.data);

    MPI_Finalize();
    return 0;
}