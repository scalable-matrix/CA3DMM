function plot_efficiency(title_str, op_gflops, cosma_ncl_ms, ca3dmm_ncl_ms, ctf_ncl_ms, cosma_cl_ms, ca3dmm_cl_ms)
    n_core = [8, 16, 32, 64, 128] .* 24;
    single_node_peak_gflops = 1800;
    single_node_n_core = 24;
    peak_perf_ms = op_gflops ./ single_node_peak_gflops ./ n_core .* single_node_n_core .* 1000.0;

    cosma_ncl_ms    = cosma_ncl_ms';
    cosma_ncl_pct   = peak_perf_ms ./ cosma_ncl_ms .* 100;  % Get the percentage of peak performance
    cosma_ncl_mean  = mean(cosma_ncl_pct);
    cosma_ncl_max   = max(cosma_ncl_pct);
    cosma_ncl_min   = min(cosma_ncl_pct);
    cosma_ncl_fill  = [cosma_ncl_min, fliplr(cosma_ncl_max)];

    ca3dmm_ncl_ms   = ca3dmm_ncl_ms';
    ca3dmm_ncl_pct  = peak_perf_ms ./ ca3dmm_ncl_ms .* 100;  % Get the percentage of peak performance
    ca3dmm_ncl_mean = mean(ca3dmm_ncl_pct);
    ca3dmm_ncl_max  = max(ca3dmm_ncl_pct);
    ca3dmm_ncl_min  = min(ca3dmm_ncl_pct);
    ca3dmm_ncl_fill = [ca3dmm_ncl_min, fliplr(ca3dmm_ncl_max)];

    ctf_ncl_ms      = ctf_ncl_ms';
    ctf_ncl_pct     = peak_perf_ms ./ ctf_ncl_ms .* 100;  % Get the percentage of peak performance
    ctf_ncl_mean    = mean(ctf_ncl_pct);
    ctf_ncl_max     = max(ctf_ncl_pct);
    ctf_ncl_min     = min(ctf_ncl_pct);
    ctf_ncl_fill    = [ctf_ncl_min, fliplr(ctf_ncl_max)];

    cosma_cl_ms     = cosma_cl_ms';
    cosma_cl_pct    = peak_perf_ms ./ cosma_cl_ms .* 100;  % Get the percentage of peak performance
    cosma_cl_mean   = mean(cosma_cl_pct);
    cosma_cl_max    = max(cosma_cl_pct);
    cosma_cl_min    = min(cosma_cl_pct);
    cosma_cl_fill   = [cosma_cl_min, fliplr(cosma_cl_max)];

    ca3dmm_cl_ms    = ca3dmm_cl_ms';
    ca3dmm_cl_pct   = peak_perf_ms ./ ca3dmm_cl_ms .* 100;  % Get the percentage of peak performance
    ca3dmm_cl_mean  = mean(ca3dmm_cl_pct);
    ca3dmm_cl_max   = max(ca3dmm_cl_pct);
    ca3dmm_cl_min   = min(ca3dmm_cl_pct);
    ca3dmm_cl_fill  = [ca3dmm_cl_min, fliplr(ca3dmm_cl_max)];

    avg = mean(cosma_ncl_ms');
    fprintf('COSMA native layout average runtime (s): ')
    for i = 1 : 5
        fprintf('%.2f ', avg(i) / 1000);
    end
    fprintf('\n');

    avg = mean(cosma_cl_ms');
    fprintf('COSMA custom layout average runtime (s): ')
    for i = 1 : 5
        fprintf('%.2f ', avg(i) / 1000);
    end
    fprintf('\n');

    avg = mean(ca3dmm_ncl_ms');
    fprintf('CA3DMM native layout average runtime (s): ')
    for i = 1 : 5
        fprintf('%.2f ', avg(i) / 1000);
    end
    fprintf('\n');

    avg = mean(ca3dmm_cl_ms');
    fprintf('CA3DMM custom layout average runtime (s): ')
    for i = 1 : 5
        fprintf('%.2f ', avg(i) / 1000);
    end
    fprintf('\n');

    avg = mean(ctf_ncl_ms');
    fprintf('CTF native layout average runtime (s): ')
    for i = 1 : 5
        fprintf('%.2f ', avg(i) / 1000);
    end
    fprintf('\n');

    X_plot = [n_core, fliplr(n_core)];
    fig1 = figure('Renderer', 'painters', 'Position', [10 10 800 600]);
    font_size = 16;
    semilogx(n_core, cosma_ncl_mean,  'r-*'),  hold on
    semilogx(n_core, ca3dmm_ncl_mean, 'b-o'),  hold on
    semilogx(n_core, ctf_ncl_mean,    'g-o'),  hold on
    semilogx(n_core, cosma_cl_mean,   'm-.*'), hold on
    semilogx(n_core, ca3dmm_cl_mean,  'c-.o'), hold on
    fill(X_plot, cosma_ncl_fill,  1, 'facecolor', 'r', 'edgecolor', 'none', 'facealpha', 0.2), hold on
    fill(X_plot, ca3dmm_ncl_fill, 1, 'facecolor', 'b', 'edgecolor', 'none', 'facealpha', 0.2), hold on
    fill(X_plot, ctf_ncl_fill,    1, 'facecolor', 'g', 'edgecolor', 'none', 'facealpha', 0.2), hold on
    fill(X_plot, cosma_cl_fill,   1, 'facecolor', 'm', 'edgecolor', 'none', 'facealpha', 0.2), hold on
    fill(X_plot, ca3dmm_cl_fill,  1, 'facecolor', 'c', 'edgecolor', 'none', 'facealpha', 0.2), hold on
    grid on, axis([min(n_core) * 0.9, max(n_core) / 0.9, 0, 100]);
    xticks(n_core), xticklabels({'8 * 24', '16 * 24', '32 * 24', '64 * 24', '128 * 24'});
    fig_handle = gca(fig1); 
    fig_handle.XAxis.FontSize = font_size;
    fig_handle.YAxis.FontSize = font_size;
    xlabel('number of cores', 'FontSize', font_size), ylabel('% peak performance', 'FontSize', font_size);
    title(title_str);
    legend({'COSMA native layout', 'CA3DMM native layout', 'CTF native layout', 'COSMA custom layout', 'CA3DMM custom layout'}, ... 
           'Location', 'Southwest', 'FontSize', font_size);
    hold off
end