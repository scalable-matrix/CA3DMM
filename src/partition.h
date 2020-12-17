#ifndef __PARTITION_H__
#define __PARTITION_H__

#ifdef __cplusplus
extern "C" {
#endif

// Partition an array into multiple same-size blocks and return the 
// start position of a given block
// Input parameters:
//   len   : Length of the array
//   n_blk : Total number of blocks to be partitioned
//   i_blk : Index of the block whose start position we need.
//           0 <= i_blk <= n_blk, i_blk == 0/n_blk return 0/len.
// Output parameters:
//   *blk_size : The length of the i_blk-th block
//   *blk_spos : The start position of the i_blk-th block, -1 means invalid parameters
void calc_block_size_pos(
    const int len, const int n_blk, const int i_blk,
    int *blk_size, int *blk_spos
);

// Calculate the near-optimal 3D decomposition of cuboid of size m * n * k for p processes
// Input parameters:
//   m, n, k : Size of the cuboid
//   p       : Number of processes
// Output parameters:
//   mp, np, kp : Active process grid size
//   rp         : Number of idle processes
void calc_3d_decomposition(
    const int p, const int m, const int n, const int k,
    int *mp, int *np, int *kp, int *rp
);

// Calculate the near-optimal 3D decomposition of cuboid of size n * n * k for p processes
// Input parameters:
//   n, k : Size of the cuboid
//   p    : Number of processes
// Output parameters:
//   np, kp : Active process grid size (np * np * kp)
//   rp     : Number of idle processes
void calc_3d_decomposition_nk(
    const int p, const int n, const int k,
    int *np, int *kp, int *rp
);

#ifdef __cplusplus
}
#endif

#endif
