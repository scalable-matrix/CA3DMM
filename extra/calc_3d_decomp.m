function proc_grid = calc_3d_decomp(m, n, k, p)
    
    if ((m <= n) && (n <= k))
        size_grid = [m, n, k];
        idx_grid  = [1, 2, 3];
    end
    if ((m <= k) && (k <= n))
        size_grid = [m, k, n];
        idx_grid  = [1, 3, 2];
    end
    if ((n <= m) && (m <= k))
        size_grid = [n, m, k];
        idx_grid  = [2, 1, 3];
    end
    if ((n <= k) && (k <= m))
        size_grid = [n, k, m];
        idx_grid  = [3, 1, 2];
    end
    if ((k <= m) && (m <= n))
        size_grid = [k, m, n];
        idx_grid  = [2, 3, 1];
    end
    if ((k <= n) && (n <= m))
        size_grid = [k, n, m];
        idx_grid  = [3, 2, 1];
    end

    proc_grid = ones(1, 4);
    d1 = size_grid(1);
    d2 = size_grid(2);
    d3 = size_grid(3);

    %% 1 large dimension case
    d3d2_ratio = d3 / d2;
    if (d3d2_ratio > p), proc_grid(3) = p; end

    %% 2 large dimensions case
    large2_threshold = d2 * d3 / (d1 * d1);
    if ((d3d2_ratio <= p) && (p <= large2_threshold))
        max_pg_prod = 1;
        % Make sure 0.5 <= sg(3)/sg(2) <= 2
        for ratio = ceil(d3d2_ratio * 0.5) : floor(d3d2_ratio * 2)
            d2p = floor(sqrt(p / ratio));
            if ((m == d1) || (n == d1))
                d3p = floor(p / d2p);
            else
                % mp(np) must be a multiplier of np(mp)
                d3p = d2p * ratio;
            end
            if (d2p * d3p > max_pg_prod)
                max_pg_prod  = d2p * d3p;
                proc_grid(2) = d2p;
                proc_grid(3) = d3p;
            end
        end
    end

    %% 3 large dimensions case
    if (p > large2_threshold)
        d3d1_ratio = d3 / d1;
        d2d1_ratio = d2 / d1;
        d1_perfect = (p / (d3d1_ratio * d2d1_ratio))^(1/3);
        d2_perfect = d1_perfect * d2d1_ratio;
        d3_perfect = d1_perfect * d3d1_ratio;
        perfect_grid = [d1_perfect, d2_perfect, d3_perfect];
        if (idx_grid(3) == 1)
            proc_grid = fixed_pair_scale_grid(p, size_grid, perfect_grid, [2, 3]);
        end
        if (idx_grid(3) == 2)
            proc_grid = fixed_pair_scale_grid(p, size_grid, perfect_grid, [1, 3]);
        end
        if (idx_grid(3) == 3)
            proc_grid = fixed_pair_scale_grid(p, size_grid, perfect_grid, [1, 2]);
        end
    end

    mp = proc_grid(idx_grid(1));
    np = proc_grid(idx_grid(2));
    kp = proc_grid(idx_grid(3));
    rp = p - mp * np * kp;
    mb = m / mp;
    nb = n / np;
    kb = k / kp;
    perfect_sg   = size_grid ./ perfect_grid;
    min_surf     = perfect_sg(1) * perfect_sg(2) + perfect_sg(1) * perfect_sg(3) + perfect_sg(2) * perfect_sg(3);
    min_surfsum  = min_surf * p;
    curr_sg      = size_grid ./ proc_grid;
    curr_surf    = curr_sg(1) * curr_sg(2) + curr_sg(1) * curr_sg(3) + curr_sg(2) * curr_sg(3);
    curr_surfsum = curr_surf * prod(proc_grid);
    fprintf('mp, np, kp, rp = %d, %d, %d, %d\n', mp, np, kp, rp);
    fprintf('mb, nb, kb = %.2f, %.2f, %.2f\n', mb, nb, kb);
    fprintf('Process utilization ratio = %.2f\n', 100 * (1 - rp / p));
    fprintf('Surface curr / min  ratio = %.2f\n', curr_surfsum / min_surfsum);
    fprintf('\n');
    proc_grid = [mp, np, kp, rp];
end

function proc_grid = fixed_pair_scale_grid(p, size_grid, perfect_grid, fixed_pair)
% perfect_grid(1) <= perfect_grid(2) <= perfect_grid(3)
% 1 <= fixed_pair(1) < fixed_pair(2) <= 3
    idx1 = fixed_pair(1);
    idx2 = fixed_pair(2);
    idx3 = 1 + 2 + 3 - idx1 - idx2;
    fixed_ratio = perfect_grid(idx2) / perfect_grid(idx1);
    perfect_sg  = size_grid ./ perfect_grid;
    perfect_vol = prod(perfect_sg);

    % This is a conservative scheme, use it as a baseline
    proc_grid = zeros(1, 3);
    proc_grid(idx1) = floor(perfect_grid(idx1));
    proc_grid(idx2) = floor(fixed_ratio) * proc_grid(idx1);
    proc_grid(idx3) = floor(p / (proc_grid(idx1) * proc_grid(idx2)));
    sg_proc = size_grid ./ proc_grid;
    min_vol = prod(sg_proc);
    surf1   = sg_proc(1) * sg_proc(2) + sg_proc(1) * sg_proc(3) + sg_proc(2) * sg_proc(3);
    max_surfsum = surf1 * prod(proc_grid);

    proc_grid1 = zeros(1, 3);
    pg1_idx1_lower = ceil(perfect_grid(idx1) * 0.3);
    pg1_idx1_upper = floor(perfect_grid(idx1) * 3.3);
    ratio_lower    = ceil(fixed_ratio * 0.3);
    ratio_upper    = floor(fixed_ratio * 3.3);
    for pg1_idx1 = pg1_idx1_lower : pg1_idx1_upper
        proc_grid1(idx1) = pg1_idx1;
        for ratio = ratio_lower : ratio_upper
            proc_grid1(idx2) = ratio * proc_grid1(idx1);
            proc_grid1(idx3) = floor(p / (proc_grid1(idx1) * proc_grid1(idx2)));
            sg_proc1 = size_grid ./ proc_grid1;
            vol1     = prod(sg_proc1);
            surf1    = sg_proc1(1) * sg_proc1(2) + sg_proc1(1) * sg_proc1(3) + sg_proc1(2) * sg_proc1(3);
            surfsum1 = surf1 * prod(proc_grid1);
            if ((min(proc_grid1) < 1) || (vol1 / perfect_vol > 1.1) || ...
                (max(sg_proc1)/min(sg_proc1) >= 4.0))
               continue;
            end
            if (vol1 < min_vol)
                min_vol = vol1;
                max_surfsum = surfsum1;
                proc_grid = proc_grid1;
            end
            if ((vol1 < min_vol * 1.05) && (surfsum1 < max_surfsum))
                min_vol = vol1;
                max_surfsum = surfsum1;
                proc_grid = proc_grid1;
            end
        end
    end
end
