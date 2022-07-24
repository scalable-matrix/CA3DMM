function plot_mpiomp(cosma_mpi_t, cosma_mpiomp_t, ca3dmm_mpi_t, ca3dmm_mpiomp_t, ctf_mpi_t, ctf_mpiomp_t, ytick_vals, ytick_lables)
    n_node = [8, 16, 32, 64, 128];
    fig1 = figure('Renderer', 'painters', 'Position', [10 10 800 600]);
    loglog(n_node, cosma_mpi_t,     'r-o'),  hold on
    loglog(n_node, cosma_mpiomp_t,  'r--x'), hold on
    loglog(n_node, ca3dmm_mpi_t,    'b-o'),  hold on
    loglog(n_node, ca3dmm_mpiomp_t, 'b--x'), hold on
    loglog(n_node, ctf_mpi_t,       'g-o'),  hold on
    loglog(n_node, ctf_mpiomp_t,    'g--x'), hold on
    all_t = [cosma_mpi_t(:) cosma_mpiomp_t(:) ca3dmm_mpi_t(:) ca3dmm_mpiomp_t(:) ctf_mpi_t(:) ctf_mpiomp_t(:)];
    font_size = 16;
    grid on, axis([min(n_node) * 0.9, max(n_node) / 0.9, min(all_t(:)) * 0.8, max(all_t(:)) / 0.8])
    xticks(n_node), xticklabels({'8', '16', '32', '64', '128'});
    yticks(ytick_vals), yticklabels(ytick_lables)
    fig_handle = gca(fig1); 
    fig_handle.XAxis.FontSize = font_size;
    fig_handle.YAxis.FontSize = font_size;
    xlabel('number of nodes', 'FontSize', font_size)
    ylabel('runtime (seconds)', 'FontSize', font_size)
    legend({'COSMA MPI', 'COSMA MPI+OpenMP', 'CA3DMM MPI', 'CA3DMM MPI+OpenMP', 'CTF MPI', 'CTF MPI+OpenMP'}, ...
            'Location', 'Southwest', 'FontSize', font_size)
    hold off
end