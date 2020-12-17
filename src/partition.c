#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "partition.h"

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
void calc_block_size_pos(
    const int len, const int n_blk, const int i_blk,
    int *blk_size, int *blk_spos
)
{
    if (i_blk < 0 || i_blk > n_blk)
    {
        *blk_spos = -1;
        *blk_size = 0;
        return;
    }
    int rem = len % n_blk;
    int bs0 = len / n_blk;
    int bs1 = bs0 + 1;
    if (i_blk < rem) 
    {
        *blk_spos = bs1 * i_blk;
        *blk_size = bs1;
    } else {
        *blk_spos = bs0 * i_blk + rem;
        *blk_size = bs0;
    }
}

double max_min_ratio(const int n_elem, double *a)
{
    double maxval = a[0];
    double minval = a[0];
    for (int i = 1; i < n_elem; i++)
    {
        maxval = (maxval < a[i]) ? a[i] : maxval;
        minval = (minval > a[i]) ? a[i] : minval;
    }
    return (maxval / minval);
}

void fixed_pair_scale_grid(
    const int p, const int idx0, const int idx1, 
    double *size_grid, double *perfect_grid, int *proc_grid
)
{
    int idx2 = 0 + 1 + 2 - idx0 - idx1;
    double fixed_ratio = perfect_grid[idx1] / perfect_grid[idx0];
    double p_double = p;
    double perfect_sg[3];
    double perfect_vol = 1.0;
    for (int i = 0; i < 3; i++)
    {
        perfect_sg[i] = size_grid[i] / perfect_grid[i];
        perfect_vol *= perfect_sg[i];
    }

    // Find a conservative partition as the baseline
    proc_grid[idx0] = (int) floor(perfect_grid[idx0]);
    proc_grid[idx1] = (int) floor(fixed_ratio) * proc_grid[idx0];
    proc_grid[idx2] = (int) (floor(p_double / (double) (proc_grid[idx0] * proc_grid[idx1])));
    double sg_proc[3], min_vol = 1.0, surf0, max_surfsum;
    for (int i = 0; i < 3; i++)
    {
        sg_proc[i] = size_grid[i] / (double) proc_grid[i];
        min_vol *= sg_proc[i];
    }
    surf0 = sg_proc[0] * sg_proc[1] + sg_proc[0] * sg_proc[2] + sg_proc[1] * sg_proc[2];
    max_surfsum = surf0 * (double) proc_grid[0] * (double) proc_grid[1] * (double) proc_grid[2];

    int proc_grid1[3], valid;
    double sg_proc1[3], vol1, surf1, surfsum1;
    int pg_idx0_lower = (int) ceil (perfect_grid[idx0] * 0.3);
    int pg_idx0_upper = (int) floor(perfect_grid[idx0] * 3.3);
    int ratio_lower   = (int) ceil (fixed_ratio * 0.3);
    int ratio_upper   = (int) floor(fixed_ratio * 3.3);
    for (int pg_idx0 = pg_idx0_lower; pg_idx0 <= pg_idx0_upper; pg_idx0++)
    {
        proc_grid1[idx0] = pg_idx0;
        for (int ratio = ratio_lower; ratio <= ratio_upper; ratio++)
        {
            proc_grid1[idx1] = ratio * proc_grid1[idx0];
            proc_grid1[idx2] = (int) floor(p_double / (double) (proc_grid1[idx0] * proc_grid1[idx1]));
            vol1 = 1.0;
            surfsum1 = 1.0;
            for (int i = 0; i < 3; i++)
            {
                sg_proc1[i] = size_grid[i] / (double) proc_grid1[i];
                vol1 *= sg_proc1[i];
                surfsum1 *= (double) proc_grid1[i];
            }
            surf1 = sg_proc1[0] * sg_proc1[1] + sg_proc1[0] * sg_proc1[2] + sg_proc1[1] * sg_proc1[2];
            surfsum1 = surfsum1 * surf1;
            if ((proc_grid1[0]) < 1 || (proc_grid1[1] < 1) || (proc_grid1[2] < 1)) continue;
            if (vol1 / perfect_vol > 1.1) continue;
            if (max_min_ratio(3, sg_proc1) >= 4.0) continue;
            if (vol1 < min_vol)
            {
                min_vol = vol1;
                max_surfsum = surfsum1;
                proc_grid[0] = proc_grid1[0];
                proc_grid[1] = proc_grid1[1];
                proc_grid[2] = proc_grid1[2];
            } 
            if ((vol1 < min_vol * 1.05) && (surfsum1 < max_surfsum))
            {
                min_vol = vol1;
                max_surfsum = surfsum1;
                proc_grid[0] = proc_grid1[0];
                proc_grid[1] = proc_grid1[1];
                proc_grid[2] = proc_grid1[2];
            }
        }
    }
}

// Calculate the near-optimal 3D decomposition of cuboid of size m * n * k for p processes
void calc_3d_decomposition(
    const int p, const int m, const int n, const int k,
    int *mp, int *np, int *kp, int *rp
)
{
    double size_grid[3], perfect_grid[3];
    int idx_grid[3], proc_grid[3] = {1, 1, 1};
    if ((m <= n) && (n <= k))
    {
        size_grid[0] = m;  size_grid[1] = n;  size_grid[2] = k;
        idx_grid[0]  = 0;  idx_grid[1]  = 1;  idx_grid[2]  = 2;
    }
    if ((m <= k) && (k <= n))
    {
        size_grid[0] = m;  size_grid[1] = k;  size_grid[2] = n;
        idx_grid[0]  = 0;  idx_grid[1]  = 2;  idx_grid[2]  = 1;
    }
    if ((n <= m) && (m <= k))
    {
        size_grid[0] = n;  size_grid[1] = m;  size_grid[2] = k;
        idx_grid[0]  = 1;  idx_grid[1]  = 0;  idx_grid[2]  = 2;
    }
    if ((n <= k) && (k <= m))
    {
        size_grid[0] = n;  size_grid[1] = k;  size_grid[2] = m;
        idx_grid[0]  = 2;  idx_grid[1]  = 0;  idx_grid[2]  = 1;
    }
    if ((k <= m) && (m <= n))
    {
        size_grid[0] = k;  size_grid[1] = m;  size_grid[2] = n;
        idx_grid[0]  = 1;  idx_grid[1]  = 2;  idx_grid[2]  = 0;
    }
    if ((k <= n) && (n <= m))
    {
        size_grid[0] = k;  size_grid[1] = n;  size_grid[2] = m;
        idx_grid[0]  = 2;  idx_grid[1]  = 1;  idx_grid[2]  = 0;
    }

    double d0 = size_grid[0], d1 = size_grid[1], d2 = size_grid[2];

    // One large dimension cases
    double p_double = p;
    double d2d1_ratio = d2 / d1;
    if (d2d1_ratio > p_double) proc_grid[2] = p;

    // Two large dimension cases
    double large2_threshold = d1 * d2 / (d0 * d0);
    double k_double = k;
    if ((d2d1_ratio <= p_double) && (p_double <= large2_threshold))
    {
        double max_pg_prod = 1.0;
        int ratio_lower = (int) ceil(d2d1_ratio * 0.5);
        int ratio_upper = (int) floor(d2d1_ratio * 2);
        for (int ratio = ratio_lower; ratio <= ratio_upper; ratio++)
        {
            double ratio_double = ratio;
            int d1p = (int) floor(sqrt(p_double / ratio_double));
            int d2p = (k_double == d0) ? (d1p * ratio) : (int) floor(p_double / (double) d1p);
            double d1p_d2p = (double) d1p * d2p;
            if (d1p_d2p > max_pg_prod)
            {
                max_pg_prod  = d1p_d2p;
                proc_grid[1] = d1p;
                proc_grid[2] = d2p;
            }
        }
    }

    // Three large dimension cases
    if (p_double > large2_threshold)
    {
        double d2d0_ratio = d2 / d0;
        double d1d0_ratio = d1 / d0;
        double d0_perfect = cbrt(p_double / (d2d0_ratio * d1d0_ratio));
        double d1_perfect = d0_perfect * d1d0_ratio;
        double d2_perfect = d0_perfect * d2d0_ratio;
        perfect_grid[0] = d0_perfect;
        perfect_grid[1] = d1_perfect;
        perfect_grid[2] = d2_perfect;
        if (idx_grid[2] == 0) fixed_pair_scale_grid(p, 1, 2, &size_grid[0], &perfect_grid[0], &proc_grid[0]);
        if (idx_grid[2] == 1) fixed_pair_scale_grid(p, 0, 2, &size_grid[0], &perfect_grid[0], &proc_grid[0]);
        if (idx_grid[2] == 2) fixed_pair_scale_grid(p, 0, 1, &size_grid[0], &perfect_grid[0], &proc_grid[0]);
    }

    *mp = proc_grid[idx_grid[0]];
    *np = proc_grid[idx_grid[1]];
    *kp = proc_grid[idx_grid[2]];
    *rp = p - (*mp) * (*np) * (*kp);
}

// Calculate the near-optimal 3D decomposition of cuboid of size n * n * k for p processes
void calc_3d_decomposition_nk(
    const int p, const int n, const int k,
    int *np, int *kp, int *rp
)
{
    int tmp_mp = 1, tmp_np = 1, tmp_kp = 1, tmp_rp;
    double tmp_m = n;
    double tmp_n = n;
    double tmp_k = k;
    double tmp_p = p;
    while (tmp_p >= 2.0)
    {
        if ((tmp_m >= tmp_n) && (tmp_m >= tmp_k))
        {
            tmp_mp *= 2;
            tmp_m  *= 0.5;
        } 
        else if ((tmp_n >= tmp_m) && (tmp_n >= tmp_k))
        {
            tmp_np *= 2;
            tmp_n  *= 0.5;
        }
        else
        {
            tmp_kp *= 2;
            tmp_k  *= 0.5;
        }
        tmp_p *= 0.5;
    }
    tmp_np = (int) round(sqrt((double) (tmp_mp * tmp_np)));
    if (tmp_np * tmp_np > p) tmp_np--;
    tmp_mp = tmp_np;
    tmp_kp = (int) floor((double) p / (double) (tmp_mp * tmp_np));
    tmp_rp = p - tmp_mp * tmp_np * tmp_kp;
    *np = tmp_np;
    *kp = tmp_kp;
    *rp = tmp_rp;
}
