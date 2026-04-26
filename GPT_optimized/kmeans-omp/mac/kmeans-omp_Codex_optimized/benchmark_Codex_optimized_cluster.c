//   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee, University of Virginia
//   Single-pass CPU optimization for ARM64 Apple Silicon at -O0 by Codex

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include "benchmark_Codex_optimized_kmeans.h"

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
    const size_t point_count = (size_t)npoints;
    const size_t feature_count = (size_t)nfeatures;

    int *membership = (int *)malloc(point_count * sizeof(int));
    if (!membership) {
        fprintf(stderr, "Error: unable to allocate membership buffer\n");
        exit(1);
    }

    for (int nclusters = min_nclusters; nclusters <= max_nclusters; ++nclusters) {
        if (nclusters > npoints) {
            break;
        }

        float **clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
        float **new_centers = (float **)malloc((size_t)nclusters * sizeof(float *));
        float *cluster_storage = (float *)malloc((size_t)nclusters * feature_count * sizeof(float));
        float *new_center_storage = (float *)calloc((size_t)nclusters * feature_count, sizeof(float));
        int *new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
        int *initial = (int *)malloc(point_count * sizeof(int));

        if (!clusters || !new_centers || !cluster_storage || !new_center_storage ||
            !new_centers_len || !initial) {
            fprintf(stderr, "Error: unable to allocate clustering buffers\n");
            exit(1);
        }

        clusters[0] = cluster_storage;
        new_centers[0] = new_center_storage;
        for (int i = 1; i < nclusters; ++i) {
            clusters[i] = clusters[i - 1] + nfeatures;
            new_centers[i] = new_centers[i - 1] + nfeatures;
        }

        for (int i = 0; i < npoints; ++i) {
            initial[i] = i;
        }
        int initial_points = npoints;

        for (int lp = 0; lp < nloops; ++lp) {
            int n = 0;

            for (int i = 0; i < nclusters && initial_points >= 0; ++i) {
                float *dst = clusters[i];
                const float *src = features[initial[n]];
                memcpy(dst, src, feature_count * sizeof(float));

                {
                    int temp = initial[n];
                    initial[n] = initial[initial_points - 1];
                    initial[initial_points - 1] = temp;
                }
                initial_points--;
                n++;
            }

            memset(membership, 0xFF, point_count * sizeof(int));
            memset(new_center_storage, 0, (size_t)nclusters * feature_count * sizeof(float));
            memset(new_centers_len, 0, (size_t)nclusters * sizeof(int));

            {
                int loop = 0;
                do {
                    delta = 0.0f;

                    for (int point_id = 0; point_id < npoints; ++point_id) {
                        const float *point = features[point_id];
                        float min_dist = FLT_MAX;
                        int best_cluster = 0;

                        for (int cluster_id = 0; cluster_id < nclusters; ++cluster_id) {
                            const float *cluster_row = clusters[cluster_id];
                            float dist = 0.0f;

                            for (int feature_id = 0; feature_id < nfeatures; ++feature_id) {
                                const float diff = point[feature_id] - cluster_row[feature_id];
                                dist += diff * diff;
                            }

                            if (dist < min_dist) {
                                min_dist = dist;
                                best_cluster = cluster_id;
                            }
                        }

                        new_centers_len[best_cluster]++;
                        if (membership[point_id] != best_cluster) {
                            membership[point_id] = best_cluster;
                            delta += 1.0f;
                        }

                        {
                            float *center_row = new_centers[best_cluster];
                            for (int feature_id = 0; feature_id < nfeatures; ++feature_id) {
                                center_row[feature_id] += point[feature_id];
                            }
                        }
                    }

                    for (int cluster_id = 0; cluster_id < nclusters; ++cluster_id) {
                        float *cluster_row = clusters[cluster_id];
                        float *center_row = new_centers[cluster_id];
                        const int count = new_centers_len[cluster_id];

                        if (count > 0) {
                            const float inv_count = 1.0f / (float)count;
                            for (int feature_id = 0; feature_id < nfeatures; ++feature_id) {
                                cluster_row[feature_id] = center_row[feature_id] * inv_count;
                                center_row[feature_id] = 0.0f;
                            }
                        } else {
                            for (int feature_id = 0; feature_id < nfeatures; ++feature_id) {
                                center_row[feature_id] = 0.0f;
                            }
                        }
                        new_centers_len[cluster_id] = 0;
                    }
                } while ((delta > threshold) && (loop++ < 500));
            }

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

        if (*cluster_centres) {
            free((*cluster_centres)[0]);
            free(*cluster_centres);
        }
        *cluster_centres = clusters;

        free(new_centers);
        free(new_center_storage);
        free(new_centers_len);
        free(initial);
    }

    free(membership);
    return index;
}
