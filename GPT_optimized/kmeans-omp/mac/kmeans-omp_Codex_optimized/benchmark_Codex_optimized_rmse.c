/*************************************************************************/
/**   File:         rmse.c                                               **/
/**   Description:  calculate root mean squared error of particular     **/
/**                 clustering.                                          **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "benchmark_Codex_optimized_kmeans.h"

extern double wtime(void);

static inline float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
    float ans = 0.0f;
    for (int i = 0; i < numdims; ++i) {
        const float diff = pt1[i] - pt2[i];
        ans += diff * diff;
    }
    return ans;
}

static inline int find_nearest_point(float *pt,
                                     int nfeatures,
                                     float **pts,
                                     int npts)
{
    int index = 0;
    float max_dist = FLT_MAX;

    for (int i = 0; i < npts; ++i) {
        const float dist = euclid_dist_2(pt, pts[i], nfeatures);
        if (dist < max_dist) {
            max_dist = dist;
            index = i;
        }
    }
    return index;
}

float rms_err(float **feature,
              int nfeatures,
              int npoints,
              float **cluster_centres,
              int nclusters)
{
    float sum_euclid = 0.0f;

    for (int i = 0; i < npoints; ++i) {
        const int nearest_cluster_index = find_nearest_point(feature[i],
                                                             nfeatures,
                                                             cluster_centres,
                                                             nclusters);
        sum_euclid += euclid_dist_2(feature[i],
                                    cluster_centres[nearest_cluster_index],
                                    nfeatures);
    }

    return sqrtf(sum_euclid / npoints);
}
