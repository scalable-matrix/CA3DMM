#ifndef __PARTITION_H__
#define __PARTITION_H__

#ifdef __cplusplus
extern "C" {
#endif

// Calculate the near-optimal 3D decomposition of cuboid of size m * n * k for p processes
// s.t. max(mp, np) is a multiple of min(mp, np)
// Input parameters:
//   m, n, k : Size of the cuboid
//   p       : Number of processes
// Output parameters:
//   mp, np, kp : Active process grid size
//   rp         : Number of idle processes
void calc_3d_decomposition_cannon(
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
