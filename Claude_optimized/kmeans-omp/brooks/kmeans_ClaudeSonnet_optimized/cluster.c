//   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee, University of Virginia

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include "kmeans.h"

float	min_rmse_ref = FLT_MAX;
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

    int *membership     = (int*)   malloc(npoints * sizeof(int));
    int *membership_OCL = (int*)   malloc(npoints * sizeof(int));
    float *feature_swap = (float*) malloc(npoints * nfeatures * sizeof(float));
    float *feature = features[0];

#pragma omp target data map(to: feature[0:npoints * nfeatures]) \
                        map(alloc: feature_swap[0:npoints * nfeatures], \
                                   membership_OCL[0:npoints])
{
    for (int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
    {
        if (nclusters > npoints) break;

        int c = 0;

        /* Transpose features into feature_swap[feature][point] layout.
           Cache tid*nfeatures to avoid recomputing the stride multiply each inner iter. */
        #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE) nowait
        for (int tid = 0; tid < npoints; tid++) {
            int tid_base = tid * nfeatures;
            for (int i = 0; i < nfeatures; i++)
                feature_swap[i * npoints + tid] = feature[tid_base + i];
        }

        float **clusters;
        clusters    = (float**) malloc(nclusters *             sizeof(float*));
        clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
        for (int i = 1; i < nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;

        int *initial = (int*) malloc(npoints * sizeof(int));
        for (int i = 0; i < npoints; i++) initial[i] = i;
        int initial_points = npoints;

        for (int lp = 0; lp < nloops; lp++)
        {
            int n = 0;

            /* Cache pointer lookups outside inner loop for cluster init. */
            for (int i = 0; i < nclusters && initial_points >= 0; i++) {
                float *src  = features[initial[n]];
                float *dest = clusters[i];
                for (int j = 0; j < nfeatures; j++)
                    dest[j] = src[j];

                int temp = initial[n];
                initial[n] = initial[initial_points-1];
                initial[initial_points-1] = temp;
                initial_points--;
                n++;
            }

            for (int i = 0; i < npoints; i++) membership[i] = -1;

            int   *new_centers_len = (int*)   calloc(nclusters, sizeof(int));
            float **new_centers    = (float**) malloc(nclusters * sizeof(float*));
            new_centers[0] = (float*) calloc(nclusters * nfeatures, sizeof(float));
            for (int i = 1; i < nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;

            int loop = 0;
            do {
                delta = 0.0f;

                float *cluster = clusters[0];
                #pragma omp target data map(to: cluster[0:nclusters * nfeatures])
                #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
                for (int point_id = 0; point_id < npoints; point_id++) {
                    float min_dist = FLT_MAX;
                    int best_index = 0;
                    for (int i = 0; i < nclusters; i++) {
                        float ans = 0.0f;
                        /* Precompute cluster row base to avoid i*nfeatures in inner loop. */
                        int cluster_base = i * nfeatures;
                        for (int l = 0; l < nfeatures; l++) {
                            /* Cache feature_swap lookup: eliminates the duplicate load
                               from the original (feature_swap[l*npoints+point_id] was
                               read twice per iteration via the squaring expression). */
                            float fval = feature_swap[l * npoints + point_id];
                            float diff = fval - cluster[cluster_base + l];
                            ans += diff * diff;
                        }
                        if (ans < min_dist) {
                            min_dist   = ans;
                            best_index = i;
                        }
                    }
                    membership_OCL[point_id] = best_index;
                }
                #pragma omp target update from (membership_OCL[0:npoints])

                for (int i = 0; i < npoints; i++) {
                    /* Use cluster_id consistently — avoids re-reading membership_OCL[i]
                       for the membership comparison below. */
                    int cluster_id = membership_OCL[i];
                    new_centers_len[cluster_id]++;
                    if (cluster_id != membership[i]) {
                        delta++;
                        membership[i] = cluster_id;
                    }
                    /* Cache row pointers to avoid repeated double-indirection per feature. */
                    float *nc     = new_centers[cluster_id];
                    float *feat_i = features[i];
                    for (int j = 0; j < nfeatures; j++) {
                        nc[j] += feat_i[j];
                    }
                }

                /* Cache pointers and hoist the len>0 check outside the feature loop.
                   Replace division with reciprocal multiply (division is expensive
                   even at -O0 and the multiply achieves the same result). */
                for (int i = 0; i < nclusters; i++) {
                    float *ci  = clusters[i];
                    float *nci = new_centers[i];
                    int ncl    = new_centers_len[i];
                    if (ncl > 0) {
                        float inv_len = 1.0f / (float)ncl;
                        for (int j = 0; j < nfeatures; j++) {
                            ci[j]  = nci[j] * inv_len;
                            nci[j] = 0.0f;
                        }
                    } else {
                        for (int j = 0; j < nfeatures; j++) {
                            nci[j] = 0.0f;
                        }
                    }
                    new_centers_len[i] = 0;
                }
                c++;
            } while ((delta > threshold) && (loop++ < 500));

            free(new_centers[0]);
            free(new_centers);
            free(new_centers_len);

            if (isRMSE) {
                rmse = rms_err(features, nfeatures, npoints, clusters, nclusters);
                if (rmse < min_rmse_ref) {
                    min_rmse_ref    = rmse;
                    *min_rmse       = min_rmse_ref;
                    *best_nclusters = nclusters;
                    index           = lp;
                }
            }
        }

        if (*cluster_centres) {
            free((*cluster_centres)[0]);
            free(*cluster_centres);
        }
        *cluster_centres = clusters;

        free(initial);
    }
}
    free(membership_OCL);
    free(feature_swap);
    free(membership);

    return index;
}
