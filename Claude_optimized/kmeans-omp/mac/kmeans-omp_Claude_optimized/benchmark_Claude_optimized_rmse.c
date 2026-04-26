/*************************************************************************/
/**   File:         rmse.c                                              **/
/**   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic)   **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "benchmark_Claude_optimized_kmeans.h"

extern double wtime(void);

/* euclid_dist_2: inlined manually at call sites below to avoid function-call
   overhead at -O0, where the compiler does not inline automatically. */

/* find_nearest_point: kept as a static function (internal linkage) so the
   compiler can potentially inline it; marked with __attribute__((always_inline))
   to force inlining even at -O0 on GCC/Clang. */
static __inline __attribute__((always_inline))
int find_nearest_point_inline(float *pt,
                               int    nfeatures,
                               float **pts,
                               int    npts)
{
    int   index   = 0;
    float max_dist = FLT_MAX;

    for (int i = 0; i < npts; i++) {
        float ans  = 0.0f;
        float *pi  = pts[i];
        /* Manually inline euclid_dist_2 to eliminate call overhead at -O0 */
        for (int d = 0; d < nfeatures; d++) {
            float diff = pt[d] - pi[d];
            ans += diff * diff;
        }
        if (ans < max_dist) {
            max_dist = ans;
            index    = i;
        }
    }
    return index;
}

/* Public wrappers retained for ABI compatibility */
__inline
float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
    float ans = 0.0f;
    for (int i = 0; i < numdims; i++) {
        float diff = pt1[i] - pt2[i];
        ans += diff * diff;
    }
    return ans;
}

__inline
int find_nearest_point(float *pt, int nfeatures, float **pts, int npts)
{
    return find_nearest_point_inline(pt, nfeatures, pts, npts);
}

float rms_err(float **feature,
              int     nfeatures,
              int     npoints,
              float **cluster_centres,
              int     nclusters)
{
    float sum_euclid = 0.0f;

    /* Combine nearest-point search and distance accumulation into one pass.
       This avoids a second traversal of cluster_centres and reduces memory
       traffic on the L1/L2 cache.  The reduction is done with a local
       accumulator to avoid repeated cache-line bouncing on sum_euclid. */
#pragma omp parallel for \
            shared(feature, cluster_centres) \
            firstprivate(npoints, nfeatures, nclusters) \
            reduction(+:sum_euclid) \
            schedule(static)
    for (int i = 0; i < npoints; i++) {
        int   nearest = find_nearest_point_inline(feature[i], nfeatures,
                                                   cluster_centres, nclusters);
        float *pt  = feature[i];
        float *cen = cluster_centres[nearest];
        float  d   = 0.0f;
        for (int j = 0; j < nfeatures; j++) {
            float diff = pt[j] - cen[j];
            d += diff * diff;
        }
        sum_euclid += d;
    }

    return sqrtf(sum_euclid / (float)npoints);
}
