//   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee, University of Virginia

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include "kmeans.h"

float min_rmse_ref = FLT_MAX;
extern double wtime(void);

int cluster(int npoints,
            int nfeatures,
            float **features,
            int min_nclusters,
            int max_nclusters,
            float threshold,
            int *best_nclusters,
            float ***cluster_centres,
            float *min_rmse,
            int isRMSE,
            int nloops)
{
  int index = 0;
  int rmse;
  float delta;
  int *membership = (int*) malloc((size_t)npoints * sizeof(int));
  float *feature = features[0];

  for (int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
    if (nclusters > npoints) break;

    float **clusters = (float**) malloc((size_t)nclusters * sizeof(float*));
    clusters[0] = (float*) malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    for (int i = 1; i < nclusters; i++) {
      clusters[i] = clusters[i - 1] + nfeatures;
    }

    int *initial = (int*) malloc((size_t)npoints * sizeof(int));
    for (int i = 0; i < npoints; i++) {
      initial[i] = i;
    }
    int initial_points = npoints;

    int *new_centers_len = (int*) calloc((size_t)nclusters, sizeof(int));
    float **new_centers = (float**) malloc((size_t)nclusters * sizeof(float*));
    new_centers[0] = (float*) calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
    for (int i = 1; i < nclusters; i++) {
      new_centers[i] = new_centers[i - 1] + nfeatures;
    }

    for (int lp = 0; lp < nloops; lp++) {
      int n = 0;

      for (int i = 0; i < nclusters && initial_points >= 0; i++) {
        float *cluster_row = clusters[i];
        float *feature_row = features[initial[n]];
        for (int j = 0; j < nfeatures; j++) {
          cluster_row[j] = feature_row[j];
        }

        int temp = initial[n];
        initial[n] = initial[initial_points - 1];
        initial[initial_points - 1] = temp;
        initial_points--;
        n++;
      }

      for (int i = 0; i < npoints; i++) {
        membership[i] = -1;
      }

      int loop = 0;
      do {
        delta = 0.0f;

        for (int point_id = 0; point_id < npoints; point_id++) {
          float *point = feature + (size_t)point_id * nfeatures;
          float min_dist = FLT_MAX;
          int best_cluster = 0;

          for (int i = 0; i < nclusters; i++) {
            float *cluster_row = clusters[i];
            float dist = 0.0f;
            for (int l = 0; l < nfeatures; l++) {
              float diff = point[l] - cluster_row[l];
              dist += diff * diff;
            }
            if (dist < min_dist) {
              min_dist = dist;
              best_cluster = i;
            }
          }

          index = best_cluster;
          new_centers_len[best_cluster]++;
          if (best_cluster != membership[point_id]) {
            delta++;
            membership[point_id] = best_cluster;
          }

          float *center_row = new_centers[best_cluster];
          for (int j = 0; j < nfeatures; j++) {
            center_row[j] += point[j];
          }
        }

        for (int i = 0; i < nclusters; i++) {
          float *cluster_row = clusters[i];
          float *center_row = new_centers[i];
          int center_len = new_centers_len[i];

          if (center_len > 0) {
            float inv_len = 1.0f / center_len;
            for (int j = 0; j < nfeatures; j++) {
              cluster_row[j] = center_row[j] * inv_len;
              center_row[j] = 0.0f;
            }
          } else {
            for (int j = 0; j < nfeatures; j++) {
              center_row[j] = 0.0f;
            }
          }
          new_centers_len[i] = 0;
        }

      } while ((delta > threshold) && (loop++ < 500));

      if (isRMSE) {
        rmse = rms_err(features, nfeatures, npoints, clusters, nclusters);

        if (rmse < min_rmse_ref) {
          min_rmse_ref = rmse;
          *min_rmse = min_rmse_ref;
          *best_nclusters = nclusters;
          index = lp;
        }
      }
    }

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    if (*cluster_centres) {
      free((*cluster_centres)[0]);
      free(*cluster_centres);
    }
    *cluster_centres = clusters;

    free(initial);
  }

  free(membership);
  return index;
}
