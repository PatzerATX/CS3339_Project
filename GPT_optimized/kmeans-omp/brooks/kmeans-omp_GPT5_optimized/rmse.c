/*************************************************************************/
/**   File:         rmse.c                                                **/
/**   Description:  calculate root mean squared error of particular       **/
/**                 clustering.                                          **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

extern double wtime(void);

float euclid_dist_2(float *pt1, float *pt2, int numdims)
{
  float ans = 0.0f;

  for (int i = 0; i < numdims; i++) {
    float diff = pt1[i] - pt2[i];
    ans += diff * diff;
  }

  return ans;
}

int find_nearest_point(float *pt, int nfeatures, float **pts, int npts)
{
  int index = 0;
  float max_dist = FLT_MAX;

  for (int i = 0; i < npts; i++) {
    float dist = euclid_dist_2(pt, pts[i], nfeatures);
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

  for (int i = 0; i < npoints; i++) {
    float *point = feature[i];
    int nearest_cluster_index = 0;
    float min_dist = FLT_MAX;

    for (int cluster = 0; cluster < nclusters; cluster++) {
      float *center = cluster_centres[cluster];
      float dist = 0.0f;

      for (int j = 0; j < nfeatures; j++) {
        float diff = point[j] - center[j];
        dist += diff * diff;
      }

      if (dist < min_dist) {
        min_dist = dist;
        nearest_cluster_index = cluster;
      }
    }

    sum_euclid += euclid_dist_2(point, cluster_centres[nearest_cluster_index], nfeatures);
  }

  return sqrt(sum_euclid / npoints);
}
