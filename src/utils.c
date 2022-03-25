// @brief    : Some helper functions I use here and there
// @author   : Hua Huang <huangh223@gatech.edu>
// @modified : 2022-01-19

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <complex.h>
#include <sys/time.h>
#include <math.h>

#include "utils.h"

// Get wall-clock time in seconds
double get_wtime_sec()
{
    double sec;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    sec = tv.tv_sec + (double) tv.tv_usec / 1000000.0;
    return sec;
}

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
void calc_block_spos_size(
    const int len, const int nblk, const int iblk,
    int *blk_spos, int *blk_size
)
{
	if (iblk < 0 || iblk > nblk)
    {
        *blk_spos = -1;
        *blk_size = 0;
        return;
    }
	int rem = len % nblk;
	int bs0 = len / nblk;
	int bs1 = bs0 + 1;
	if (iblk < rem) 
    {
        *blk_spos = bs1 * iblk;
        *blk_size = bs1;
    } else {
        *blk_spos = bs0 * iblk + rem;
        *blk_size = bs0;
    }
}

// Allocate a piece of aligned memory 
void *malloc_aligned(size_t size, size_t alignment)
{
    void *ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
}

// Free a piece of aligned memory allocated by malloc_aligned()
void free_aligned(void *mem)
{
    free(mem);
}

// Calculate the 2-norm of a vector
// Warning: this is a naive implementation, not numerically stable
double calc_2norm(const int len, const double *x)
{
    double res = 0.0;
    for (int i = 0; i < len; i++) res += x[i] * x[i];
    return sqrt(res);
}

// Calculate the 2-norm of the difference between two vectors 
// and the 2-norm of the reference vector 
void calc_err_2norm(
    const int len, const double *x0, const double *x1, 
    double *x0_2norm_, double *err_2norm_
)
{
    double x0_2norm = 0.0, err_2norm = 0.0, diff;
    for (int i = 0; i < len; i++)
    {
        diff = x0[i] - x1[i];
        x0_2norm  += x0[i] * x0[i];
        err_2norm += diff  * diff;
    }
    *x0_2norm_  = sqrt(x0_2norm);
    *err_2norm_ = sqrt(err_2norm);
}

// Copy a row-major matrix block to another row-major matrix
void copy_matrix_block(
    const size_t dt_size, const int nrow, const int ncol, 
    const void *src, const int lds, void *dst, const int ldd, const int use_omp
)
{
    const char *src_ = (char*) src;
    char *dst_ = (char*) dst;
    const size_t lds_ = dt_size * (size_t) lds;
    const size_t ldd_ = dt_size * (size_t) ldd;
    const size_t row_msize = dt_size * (size_t) ncol;
    if (use_omp == 0)
    {
        for (int irow = 0; irow < nrow; irow++)
        {
            size_t src_offset = (size_t) irow * lds_;
            size_t dst_offset = (size_t) irow * ldd_;
            memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int irow = 0; irow < nrow; irow++)
        {
            size_t src_offset = (size_t) irow * lds_;
            size_t dst_offset = (size_t) irow * ldd_;
            memcpy(dst_ + dst_offset, src_ + src_offset, row_msize);
        }
    }
}

// Print a row-major int matrix block to standard output
void print_int_mat_blk(
    const int *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        const int *mat_i = mat + i * ldm;
        for (int j = 0; j < ncol; j++)
        {
            printf(fmt, mat_i[j]);
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

// Print a row-major double matrix block to standard output
void print_dbl_mat_blk_rm(
    const double *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        const double *mat_i = mat + i * ldm;
        for (int j = 0; j < ncol; j++)
        {
            printf(fmt, mat_i[j]);
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

// Print a column-major double matrix block to standard output
void print_dbl_mat_blk_cm(
    const double *mat, const int ldm, const int nrow, const int ncol, 
    const char *fmt, const char *mat_name
)
{
    printf("%s:\n", mat_name);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < ncol; j++)
        {
            printf(fmt, mat[j * ldm + i]);
            printf("  ");
        }
        printf("\n");
    }
    printf("\n");
}

// Transpose a column matrix (OMP parallelized, but not optimized)
void transpose_cm_mat(
    const int nrow, const int ncol, const double *A, const int ldA,
    double *AT, const int ldAT
)
{
    // TODO: use blocking
    #pragma omp parallel for
    for (int j = 0; j < ncol; j++)
    {
        for (int i = 0; i < nrow; i++)
        {
            int idx0 = i * ldA  + j;
            int idx1 = j * ldAT + i;
            AT[idx1] = A[idx0];
        }
    }
}