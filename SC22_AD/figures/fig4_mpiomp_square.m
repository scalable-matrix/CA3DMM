cosma_mpi_t     = [22.19, 10.06, 6.24, 3.24, 1.88];
cosma_mpiomp_t  = [21.30, 12.00, 7.72, 3.52, 2.33];

ca3dmm_mpi_t    = [20.98, 10.72, 5.62, 2.76, 1.75];
ca3dmm_mpiomp_t = [22.04, 11.52, 7.43, 3.01, 1.76];

ctf_mpi_t       = [31.94, 16.08, 14.19, 7.02, 7.93];
ctf_mpiomp_t    = [35.86, 18.12, 13.99, 7.22, 7.08];

ytick_vals = [2, 4, 8, 16, 32, 48];
ytick_lables = {'2', '4', '8', '16', '32', '48'};

plot_mpiomp(cosma_mpi_t, cosma_mpiomp_t, ca3dmm_mpi_t, ca3dmm_mpiomp_t, ctf_mpi_t, ctf_mpiomp_t, ...
            ytick_vals, ytick_lables)