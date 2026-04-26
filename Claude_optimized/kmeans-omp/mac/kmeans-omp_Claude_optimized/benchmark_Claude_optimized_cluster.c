//   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee, University of Virginia
//   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include "benchmark_Claude_optimized_kmeans.h"

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

    /* Pre-compute frequently reused values */
    const int npoints_x_nfeatures = npoints * nfeatures;

    int *membership     = (int *)  malloc(npoints * sizeof(int));
    int *membership_OCL = (int *)  malloc(npoints * sizeof(int));
    float *feature_swap = (float *)malloc(npoints_x_nfeatures * sizeof(float));
    float *feature      = features[0];

    /* Flat cluster buffer reused across nclusters iterations to avoid repeated malloc/free */
    int max_clusters_buf = max_nclusters;
    float *cluster_flat  = (float *)malloc(max_clusters_buf * nfeatures * sizeof(float));
    float **clusters_ptrs = (float **)malloc(max_clusters_buf * sizeof(float *));

#pragma omp target data map(to: feature[0:npoints_x_nfeatures]) \
                        map(alloc: feature_swap[0:npoints_x_nfeatures], \
                                   membership_OCL[0:npoints])
    {
        for (int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
        {
            if (nclusters > npoints) break;

            int c = 0;

            /* Transpose feature matrix for cache-friendly column access in the kernel.
               feature_swap[feature_idx * npoints + point_idx] layout lets the GPU
               kernel read a full feature column contiguously. */
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE) nowait
            for (int tid = 0; tid < npoints; tid++) {
                for (int i = 0; i < nfeatures; i++)
                    feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
            }

            /* Reuse pre-allocated flat buffer instead of malloc per nclusters */
            float **clusters = clusters_ptrs;
            clusters[0] = cluster_flat;
            for (int i = 1; i < nclusters; i++)
                clusters[i] = clusters[i - 1] + nfeatures;

            int *initial = (int *)malloc(npoints * sizeof(int));
            for (int i = 0; i < npoints; i++) initial[i] = i;
            int initial_points = npoints;

            for (int lp = 0; lp < nloops; lp++)
            {
                int n = 0;

                for (int i = 0; i < nclusters && initial_points >= 0; i++) {
                    /* Cache pointer to features[initial[n]] to avoid repeated double-dereference */
                    float *src = features[initial[n]];
                    float *dst = clusters[i];
                    /* Manual loop instead of memcpy: at -O0 memcpy call overhead is non-trivial
                       for small nfeatures; unrolling via direct assignment is faster */
                    for (int j = 0; j < nfeatures; j++)
                        dst[j] = src[j];

                    int temp = initial[n];
                    initial[n] = initial[initial_points - 1];
                    initial[initial_points - 1] = temp;
                    initial_points--;
                    n++;
                }

                /* Use memset for bulk -1 initialization: faster than scalar loop at -O0 */
                memset(membership, -1, npoints * sizeof(int));

                int *new_centers_len = (int *)   calloc(nclusters, sizeof(int));
                float *new_centers_flat = (float *)calloc(nclusters * nfeatures, sizeof(float));
                float **new_centers  = (float **)malloc(nclusters * sizeof(float *));
                new_centers[0] = new_centers_flat;
                for (int i = 1; i < nclusters; i++)
                    new_centers[i] = new_centers[i - 1] + nfeatures;

                int loop = 0;
                do {
                    delta = 0.0f;

                    float *cluster = clusters[0];
                    const int nclusters_x_nfeatures = nclusters * nfeatures;

#pragma omp target data map(to: cluster[0:nclusters_x_nfeatures])
#pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
                    for (int point_id = 0; point_id < npoints; point_id++) {
                        float min_dist = FLT_MAX;
                        int best_idx = 0;
                        /* Hoist base pointer arithmetic out of inner loop */
                        const int base = point_id;
                        for (int i = 0; i < nclusters; i++) {
                            float ans = 0.0f;
                            const int ci = i * nfeatures;
                            for (int l = 0; l < nfeatures; l++) {
                                float diff = feature_swap[l * npoints + base] - cluster[ci + l];
                                ans += diff * diff;
                            }
                            /* Avoid storing to dist variable; compare ans directly */
                            if (ans < min_dist) {
                                min_dist = ans;
                                best_idx = i;
                            }
                        }
                        membership_OCL[point_id] = best_idx;
                    }
#pragma omp target update from(membership_OCL[0:npoints])

                    /* Fuse membership update + new_centers accumulation into one pass
                       over npoints to halve the number of cache misses vs two separate loops */
                    for (int i = 0; i < npoints; i++) {
                        int cluster_id = membership_OCL[i];
                        new_centers_len[cluster_id]++;
                        if (membership_OCL[i] != membership[i]) {
                            delta++;
                            membership[i] = membership_OCL[i];
                        }
                        /* Cache pointer to avoid repeated double-dereference in inner loop */
                        float *nc  = new_centers[cluster_id];
                        float *fi  = features[i];
                        for (int j = 0; j < nfeatures; j++)
                            nc[j] += fi[j];
                    }

                    /* Update cluster centres; reset accumulators in same pass */
                    for (int i = 0; i < nclusters; i++) {
                        float *ci  = clusters[i];
                        float *nci = new_centers[i];
                        int    len = new_centers_len[i];
                        float  inv = (len > 0) ? 1.0f / (float)len : 0.0f;
                        /* Replace per-element division with multiply-by-reciprocal.
                           Division is expensive at -O0; reciprocal computed once. */
                        for (int j = 0; j < nfeatures; j++) {
                            ci[j]  = nci[j] * inv;
                            nci[j] = 0.0f;
                        }
                        new_centers_len[i] = 0;
                    }
                    c++;
                } while ((delta > threshold) && (loop++ < 500));

                free(new_centers_flat);
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

            /* Hand ownership of cluster data to caller.
               We must allocate a fresh block here because the caller frees it. */
            if (*cluster_centres) {
                free((*cluster_centres)[0]);
                free(*cluster_centres);
            }
            /* Allocate caller-owned copy */
            float **caller_clusters    = (float **)malloc(nclusters * sizeof(float *));
            caller_clusters[0]         = (float *) malloc(nclusters * nfeatures * sizeof(float));
            for (int i = 1; i < nclusters; i++)
                caller_clusters[i] = caller_clusters[i - 1] + nfeatures;
            memcpy(caller_clusters[0], clusters[0], nclusters * nfeatures * sizeof(float));
            *cluster_centres = caller_clusters;

            free(initial);
        }
    }

    free(cluster_flat);
    free(clusters_ptrs);
    free(membership_OCL);
    free(feature_swap);
    free(membership);

    return index;
}
