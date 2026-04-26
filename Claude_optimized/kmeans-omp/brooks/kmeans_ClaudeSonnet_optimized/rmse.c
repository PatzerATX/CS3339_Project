/*************************************************************************/
/**   File:         rmse.c                                              **/
/**   Description:  calculate root mean squared error of particular     **/
/**                 clustering.                                         **/
/**   Author:  Sang-Ha Lee                                              **/
/**            University of Virginia.                                  **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

extern double wtime(void);

/*----< euclid_dist_2() >----------------------------------------------------*/
__inline
float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
    float ans = 0.0f;
    /* Cache diff to avoid computing (pt1[i]-pt2[i]) twice per iteration. */
    for (int i = 0; i < numdims; i++) {
        float diff = pt1[i] - pt2[i];
        ans += diff * diff;
    }
    return ans;
}

/*----< find_nearest_point() >-----------------------------------------------*/
__inline
int find_nearest_point(float *pt, int nfeatures, float **pts, int npts)
{
    int index = 0;
    float max_dist = FLT_MAX;
    for (int i = 0; i < npts; i++) {
        float dist = euclid_dist_2(pt, pts[i], nfeatures);
        if (dist < max_dist) {
            max_dist = dist;
            index    = i;
        }
    }
    return index;
}

/*----< rms_err() >----------------------------------------------------------*/
float rms_err(float **feature, int nfeatures, int npoints,
              float **cluster_centres, int nclusters)
{
    int    i;
    int    nearest_cluster_index;
    float  sum_euclid = 0.0f;
    float  ret;

    #pragma omp parallel for \
                shared(feature, cluster_centres) \
                firstprivate(npoints, nfeatures, nclusters) \
                private(i, nearest_cluster_index) \
                schedule(static)
    for (i = 0; i < npoints; i++) {
        /* Cache feature[i] pointer to avoid double-indirection on every call. */
        float *feat_i = feature[i];
        nearest_cluster_index = find_nearest_point(feat_i, nfeatures,
                                                   cluster_centres, nclusters);
        sum_euclid += euclid_dist_2(feat_i, cluster_centres[nearest_cluster_index],
                                    nfeatures);
    }

    ret = sqrt(sum_euclid / npoints);
    return ret;
}
