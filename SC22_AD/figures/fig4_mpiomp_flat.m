cosma_mpi_t     = [8.58, 4.43, 2.40, 1.37, 0.77];
cosma_mpiomp_t  = [8.49, 4.21, 2.33, 1.18, 0.69];

ca3dmm_mpi_t    = [7.51, 4.10, 2.12, 1.16, 0.70];
ca3dmm_mpiomp_t = [8.15, 4.12, 2.13, 1.05, 0.58];

ctf_mpi_t       = [9.80, 6.52, 4.54, 3.68, 2.78];
ctf_mpiomp_t    = [10.54, 5.05, 2.86, 1.56, 2.20];

ytick_vals = [0.5, 1, 2, 4, 8, 12];
ytick_lables = {'0.5', '1', '2', '4', '8', '12'};

plot_mpiomp(cosma_mpi_t, cosma_mpiomp_t, ca3dmm_mpi_t, ca3dmm_mpiomp_t, ctf_mpi_t, ctf_mpiomp_t, ...
            ytick_vals, ytick_lables)