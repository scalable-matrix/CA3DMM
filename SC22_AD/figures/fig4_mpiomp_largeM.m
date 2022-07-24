cosma_mpi_t     = [7.08, 3.73, 1.95, 1.06, 0.60];
cosma_mpiomp_t  = [6.97, 3.28, 1.69, 0.87, 0.52];

ca3dmm_mpi_t    = [6.68, 3.40, 1.86, 1.04, 0.58];
ca3dmm_mpiomp_t = [6.70, 3.16, 1.54, 0.79, 0.43];

ctf_mpi_t       = [44.57, 16.42, 24.77, 8.46, 4.23];
ctf_mpiomp_t    = [43.34, 21.55, 24.02, 12.42, 13.87];

ytick_vals = [0.5, 1, 2, 4, 8, 16, 32, 48];
ytick_lables = {'0.5', '1', '2', '4', '8', '16', '32', '48'};

plot_mpiomp(cosma_mpi_t, cosma_mpiomp_t, ca3dmm_mpi_t, ca3dmm_mpiomp_t, ctf_mpi_t, ctf_mpiomp_t, ...
            ytick_vals, ytick_lables)