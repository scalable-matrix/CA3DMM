cosma_mpi_t     = [7.30, 3.91, 2.09, 1.15, 0.59];
cosma_mpiomp_t  = [6.97, 3.38, 1.67, 0.90, 0.55];

ca3dmm_mpi_t    = [6.78, 3.79, 1.97, 1.10, 0.62];
ca3dmm_mpiomp_t = [6.65, 3.19, 1.71, 0.87, 0.52];

ctf_mpi_t       = [18.96, 8.07, 12.86, 3.35, 1.96];
ctf_mpiomp_t    = [14.73, 8.88, 6.27, 3.51, 3.34];

ytick_vals = [0.5, 1, 2, 4, 8, 16, 24];
ytick_lables = {'0.5', '1', '2', '4', '8', '16', '24'};

plot_mpiomp(cosma_mpi_t, cosma_mpiomp_t, ca3dmm_mpi_t, ca3dmm_mpiomp_t, ctf_mpi_t, ctf_mpiomp_t, ...
            ytick_vals, ytick_lables)