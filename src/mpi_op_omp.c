#include "mpi_op_omp.h"
#include "omp.h"

static MPI_Op _op_omp_sum = MPI_OP_NULL;

void mpi_op_omp_sum(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype)
{
    const int len_ = *len;
    #define OMP_SUM(T, MPI_T) \
    if ((*dtype) == MPI_T) \
    {   \
        T *invec_ = (T *) invec;    \
        T *inoutvec_ = (T *) inoutvec;  \
        _Pragma("omp parallel for schedule(static)")   \
        for (int i = 0; i < len_; i++)  \
            inoutvec_[i] += invec_[i];  \
    }

    OMP_SUM(double, MPI_DOUBLE)
    OMP_SUM(float, MPI_FLOAT)
    OMP_SUM(int, MPI_INT)
    #undef OMP_SUM
}

void MPI_Op_omp_sum_get(MPI_Op *op_omp_sum)
{
    if (_op_omp_sum == MPI_OP_NULL) MPI_Op_create(mpi_op_omp_sum, 1, &_op_omp_sum);
    *op_omp_sum = _op_omp_sum;
}