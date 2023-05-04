# CA3DMM
Communication-Avoiding 3D Matrix Multiplication

## 1. Compilation

Clone the CA3DMM library from GitHub:
```shell
git clone https://github.com/scalable-matrix/CA3DMM.git
```

Enter directory `CA3DMM/src`. CA3DMM provides 4 example make files in `CA3DMM/src`:
- `icc-mkl-impi.make`: Use Intel C compiler, Intel MKL, and Intel MPI library
- `icc-mkl-anympi.make`: Use Intel C compiler, Intel MKL, and any MPI library
- `icc-mkl-nvcc-impi.make`: Use Intel C compiler, Intel MKL, NVCC compiler for NVIDIA GPU support, and Intel MPI library
- `icc-mkl-nvcc-anympi.make`: Use Intel C compiler, Intel MKL, NVCC compiler for NVIDIA GPU support, and any MPI library
We use the `icc-mkl-anympi.make` and `icc-mkl-nvcc-anympi.make` make files for compiling the CPU and GPU versions on the Georgia Tech PACE-Phoenix cluster. Run the following command to compile the CA3DMM library:
```shell
make -f icc-mkl-anympi.make -j
```
After compilation, the dynamic and static library files are copied to directory `CA3DMM/lib`, and the C header files are copied to directory `CA3DMM/include`.

Enter directory `CA3DMM/examples` to compile the example program. This directory also contains 4 example make files as those in `CA3DMM/src`. Run the following command to compile the CA3DMM library:
```shell
make -f icc-mkl-anympi.make -j
```
After compilation, executable `CA3DMM/examples/example_AB.exe` is the example program we needed.


## 2. Example programs

For single node execution or launching on clusters without a job scheduling system, the following command should run on most platforms (assuming that you are in directory `CA3DMM/examples`):
```shell
mpirun -np <nprocs> ./example_AB.exe <M> <N> <K> <transA> <transB> <validation> <ntest> <dev_type>
```
Where:
* nprocs: Number of MPI processes
* M, N, K: Sizes of input matrices, A matrix is M * K, B matrix is K * N
* transA, transB: 0 or 1, 0 for no transpose, 1 for transpose
* validation: 0 or 1, 0 for skipping result correctness check, 1 for result correctness check
* ntest: Number of tests to run, should be a non-negative integer
* dev\_type: Calculation device type, 0 for CPU, 1 for NVIDIA GPU (if the library and the example program are compiled with NVIDIA GPU support)

To explicitly control MPI + OpenMP hybrid parallelization, you need to specify OpenMP environment variables, and process affinity environment variables for some MPI libraries. In the paper, we use the following environment variables for MPI + OpenMP parallel tests on the Georgia Tech PACE-Phoenix cluster:
```shell
OMP_PLACES=cores
OMP_NUM_THREADS=24
MV2_CPU_BINDING_POLICY=hybrid
MV2_THREADS_PER_PROCESS=24
```

For clusters and supercomputers with job scheduling systems like slurm, you need to write job scripts for launching the example program on multiple nodes.


## 3. Expected result

The example program prints timing results to the screen output. Here is an example running output on a single node using 1 core per MPI process:
```text
$ mpirun -np 24 ./example_AB.exe 8000 8000 8000 0 0 1 10 0
Test problem size m * n * k : 8000 * 8000 * 8000
Transpose A / B             : 0 / 0
Number of tests             : 10
Check result correctness    : 1
Device type                 : 0

CA3DMM partition info:
Process grid mp * np * kp  : 4 * 2 * 3
Work cuboid  mb * nb * kb  : 2000 * 4000 * 2667
Process utilization        : 100.00 %
Comm. volume / lower bound : 1.04
Rank 0 work buffer size    : 244.28 MBytes


A, B, C redist   : 80 79 77 79 80 77 82 78 78 77
A / B allgather  : 16 16 16 16 16 16 16 16 16 16
2D Cannon        : 649 649 667 665 664 667 666 666 667 667
C reduce-scatter : 41 42 42 51 43 42 41 42 42 41
matmul only      : 706 708 725 733 724 725 724 724 725 725
total execution  : 786 786 802 812 803 802 806 802 803 801

================== CA3DMM algorithm engine =================
* Initialization         : 4.89 ms
* Number of executions   : 10
* Execution time (avg)   : 800.46 ms
  * Redistribute A, B, C : 78.74 ms
  * Allgather A or B     : 16.30 ms
  * 2D Cannon execution  : 662.62 ms
  * Reduce-scatter C     : 42.79 ms
--------------- 2D Cannon algorithm engine ---------------
* Initialization : 0.04 ms
* Number of executions  : 10
* Execution time (avg)  : 662.62 ms
  * Initial shift       : 33.79 ms
  * Loop shift wait     : 23.21 ms
  * Local DGEMM         : 605.62 ms
* Per-rank performance  : 64.40 GFlops
----------------------------------------------------------
============================================================
CA3DMM output : 0 error(s)
```

The example program uses 1D column partition for the input A and B matrices and the output C matrix. The "Process grid" line shows the 3D process grid size. The "matmul only" line gives the runtime (in milliseconds) using library-native matrix partitioning.

## 4. Cite this work

If you use CA3DMM in your work, please cite our SC22 paper:

```bibtex
@inproceedings{huang2022ca3dmm,
    title = { {CA3DMM}: A New Algorithm Based on a Unified View of Parallel Matrix Multiplication },
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    publisher = {IEEE},
    author = {Huang, Hua and Chow, Edmond},
    doi = {10.5555/3571885.3571922},
    isbn = {978-4-6654-5444-5},
    address = {Dallas, TX, USA},
    month = {Nov},
    year = {2022},
    pages = {381--395},
}
```

Thank you very much for using our code!
