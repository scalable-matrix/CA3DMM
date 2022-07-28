This file descripts the operations for reproducing all results in "Section IV. Numerical Experiments of the SC22" paper "CA3DMM: A New Algorithm Based on a Unified View of Parallel Matrix Multiplication”. The results in the paper were obtained on the Georgia Tech PACE-Phoenix cluster (see [https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/) for detailed information of this cluster). 



## Preparation

This directory contains two children folders: `job_scripts` and `figures`. Directory `job_scripts` contains all PBS and shell scripts for reproducing all test results. Directory `figures` contains all MATLAB scripts for plotting the figures. All these MATLAB scripts work with MATLAB R2020a. 

Copy all files in this directory to a working directory, for example, `$HOME/CA3DMM-tests`. Set this working directory as `WORKDIR` in bash environment:

```bash
export WORKDIR=$HOME/CA3DMM-tests
```



## Figures 3 and 4

Clone and compile COSMA following [https://github.com/eth-cscs/COSMA/blob/master/INSTALL.md](https://github.com/eth-cscs/COSMA/blob/master/INSTALL.md). We only need to compile the CPU version in this section. Set the directory you cloned COSMA as `COSMA_DIR`. Then, copy `$WORKDIR/cosma_miniapp_cl.cpp` to `$COSMA_DIR/miniapp`, modify `$COSMA_DIR/miniapp/CMakeLists.txt`, replace line 4 with 

```cmake
set(executables "layout_miniapp" "cosma_miniapp" "cosma_statistics" "cosma_miniapp_cl")
```

Enter `$COSMA_DIR/build`, run `make` again. Then, copy the executable files:

```bash
cp $COSMA_DIR/build/miniapp/cosma_miniapp    $WORKDIR/job_scripts/cosma_miniapp
cp $COSMA_DIR/build/miniapp/cosma_miniapp_cl $WORKDIR/job_scripts/cosma_miniapp_cl
```

Clone and compile CA3DMM following [https://github.com/scalable-matrix/CA3DMM/blob/main/README.md](https://github.com/scalable-matrix/CA3DMM/blob/main/README.md). We only need to compile the CPU version in this section. Set the directory you cloned CA3DMM as `CA3DMM_DIR`. Then, copy the executable file:

```bash
cp $CA3DMM_DIR/examples/example_AB.exe $WORKDIR/job_scripts/ca3dmm_example_AB
```

Clone and compile CTF following [https://github.com/cyclops-community/ctf/wiki/Building-and-testing](https://github.com/cyclops-community/ctf/wiki/Building-and-testing). We only need to compile the CPU version in this section. Set the directory you cloned CTF as `CTF_DIR`. Then, copy the executable file:

```bash
cp $CTF_DIR/bin/matmul $WORKDIR/job_scripts/ctf_matmul
```

Modify `$WORKDIR/job_scripts/fig3_mpi.pbs` and `$WORKDIR/job_scripts/fig4_mpi.pbs` based on the configuration of your cluster. These PBS files need to be submitted multiple times for different number of nodes. 

For $m=n=k=50000$ tests, results should be copied from the script output and fill in files `$WORKDIR/figures/fig3_mpi_square.m` and `$WORKDIR/figures/fig4_mpiomp_square.m`.

For $m=n=6000, k=1200000$ tests, results should be copied from the script output and fill in files `$WORKDIR/figures/fig3_mpi_largeK.m` and `$WORKDIR/figures/fig4_mpiomp_largeK.m`.

For $m=1200000,n=k=6000$ tests, results should be copied from the script output and fill in files `$WORKDIR/figures/fig3_mpi_largeM.m` and `$WORKDIR/figures/fig4_mpiomp_largeM.m`.

For $m=n=100000, k=5000$ tests, results should be copied from the script output and fill in files `$WORKDIR/figures/fig3_mpi_flat.m` and `$WORKDIR/figures/fig4_mpiomp_flat.m`.

Each MATLAB file `$WORKDIR/figures/fig3_mpi_*.m` contains five 5-by-10 matrices with names `cosma_ncl_ms`, `cosma_cl_ms`, `ca3dmm_ncl_ms`, `ca3dmm_cl_ms`, and `ctf_ncl_ms`. In each of these matrices, each row are 10 running times in milliseconds. Rows from top to bottom correspond to the results of 8, 16, 32, 64, and 128 nodes. 

The output of  `$WORKDIR/job_scripts/fig3_mpi.pbs` contains the results we need for `$WORKDIR/figures/fig3_mpi_*.m`:

* For each COSMA native layout running output, the line starting with "COSMA TIMES [ms]" contains the results we need for `cosma_ncl_ms`. 

* For each COSMA custom layout running output, the line starting with "COSMA dmultiply runtime (ms)" contains the results we need for `cosma_cl_ms`. 

* For each CA3DMM native layout running output, the line starting with "matmul only" contains the results we need for `ca3dmm_ncl_ms`.

* For each CA3DMM custom layout running output, the line starting with "total execution" contains the results we need for `ca3dmm_ncl_ms`.

* For each CTF native layout running output, the line starting with "All iterations times" contains the results we need for `ctf_ncl_ms`. 

The output of  `$WORKDIR/job_scripts/fig4_mpiomp.pbs` contains the results we need for `$WORKDIR/figures/fig4_mpiomp_*.m`:

* For each COSMA native layout running output, the line starting with "COSMA TIMES [ms]" contains the results we need for `cosma_mpiomp_t`. 
* For each CA3DMM native layout running output, the line starting with "matmul only" contains the results we need for `ca3dmm_mpiomp_t`. 
* For each CTF native layout running output, the line starting with "All iterations times" contains the results we need for `ctf_mpiomp_t`. 

The text output of running `$WORKDIR/figures/fig3_mpi_*.m` also contains results that `$WORKDIR/figures/fig4_mpiomp_*.m` needs. For example, the text output from running  `$WORKDIR/figures/fig3_mpi_square.m`  contains results needed by   `$WORKDIR/figures/fig4_mpiomp_square.m`. Text output row starts with "COSMA native layout" contains the data we need for the array `cosma_mpi_t`. Text output row starts with "CA3DMM native layout" contains the data we need for the array `ca3dmm_mpi_t`. Text output row starts with "CTF native layout" contains the data we need for the array `ctf_mpi_t`. 



## Table 1

Copy executable file:

```bash
cp $COSMA_DIR/build/miniapp/cosma_statistics $WORKDIR/job_scripts/cosma_statistics
```

Run the test scripts to obtain the data for COSMA (upper-half of Table 1):

```bash
bash $WORKDIR/job_scripts/table1_cosma.sh
```

The results for CA3DMM (lower-half of Table 1) are available in the output files in Section "Figures 3 and 4". Multiplying the value after "Required memory per rank" by 8 (a double has 8 bytes), then dividing it by 1048576 (bytes to MB) gives the required value. For each CA3DMM running output, the line starting with "Rank 0 work buffer size" is the data we need. 



## Table 2

Modify `$WORKDIR/job_scripts/table2_2048c.pbs` and `$WORKDIR/job_scripts/table2_3072c.pbs` based on the configuration of your cluster. Then, submit these two PBS files. 

For COSMA, each "Runtime (s)" value in Table 2 is the average value of a COSMA output line starting with "COSMA TIMES [ms]" (need to divided by 1000 to convert it to seconds), each $p_m,p_n,p_k$ tuple is calculated based on COSMA output lines after the line "Divisions strategy".

For CA3DMM, each "Runtime (s)" value in Table 2 is the average value of a CA3DMM output line starting with "matmul only", each $p_m,p_n,p_k$ tuple is obtained from a CA3DMM output line starting with "Process grid".




## Figure 5

The results for Figure 5 are available in the output files in Section "Table 2". MATLAB script `$WORKDIR/figures/fig5_runtime_breakdown.m` contains four arrays with suffix `cosma_` for COSMA results and four arrays with suffix `ca3dmm_` for CA3DMM results. Each of these arrays has four elements: local matrix multiplication, replicating $A$ and $B$ matrices, reducing for the final $C$ matrix, and other operations. 

For COSMA:

* Local matrix multiplication: the last profiling output, "computation" row, "WALL" column, value divided by the "CALLS" column in the same row.
* Replicating $A$ and $B$ matrices: the last profiling output, "copy" row, "WALL" column, value divided by the  "computation" row "CALLS" column.
* Reducing for the final $C$ matrix: the last profiling output, "reduce" row, "WALL" column, value divided by the  "computation" row "CALLS" column.
* Other operations: the "Runtime (s)" value in Table 2 minus the first three values in this array.

For CA3DMM:

* Local matrix multiplication: the output row containing "Local DGEMM", divide the value by 1000 to convert it from milliseconds to seconds.
* Replicating $A$ and $B$ matrices: add the output rows containing "Allgather A or B", "Initial shift", and "Loop shift wait", then divide the value by 1000 to convert it from milliseconds to seconds.
* Reducing for the final $C$ matrix: the output row containing "Reduce-scatter C", divide the value by 1000 to convert it from milliseconds to seconds.
* Other operations: the "Runtime (s)" value in Table 2 minus the first three values in this array.



## Table 3

Recompile COSMA, CA3DMM, and CTF with CUDA support. Copy executable files:

```bash
cp $COSMA_DIR/build/miniapp/cosma_miniapp $WORKDIR/job_scripts/cosma_miniapp-cuda
cp $CA3DMM_DIR/examples/example_AB.exe $WORKDIR/job_scripts/ca3dmm_example_AB-cuda
cp $CTF_DIR/bin/matmul $WORKDIR/job_scripts/ctf_matmul-cuda
```

Modify `$WORKDIR/job_scripts/table3.pbs` based on the configuration of your cluster. This PBS file needs to be submitted twice for 16 and 32 GPUs' results.

For COSMA, each "Runtime (s)" value in Table 3 is the average value of a COSMA output line starting with "COSMA TIMES [ms]" (need to divided by 1000 to convert it to seconds), each $p_m,p_n,p_k$ tuple is calculated based on COSMA output lines after the line "Divisions strategy". 

For CA3DMM, each "Runtime (s)" value in Table 3 is the average value of a CA3DMM output line starting with "matmul only", each $p_m,p_n,p_k$ tuple is obtained from a CA3DMM output line starting with "Process grid".

For CTF, each "Runtime (s)" value in Table 3 is the value after "Avg time = " in a CTF output. 
