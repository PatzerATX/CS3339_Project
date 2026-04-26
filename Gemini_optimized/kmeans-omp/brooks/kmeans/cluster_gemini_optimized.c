#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <immintrin.h>
#include "kmeans_gemini_optimized.h"

float min_rmse_ref = FLT_MAX;

// Manual SSE/AVX distance calculation
static inline float euclid_dist_2_sse(float *p1, float *p2, int len) {
    __m128 sumv = _mm_setzero_ps();
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        __m128 v1 = _mm_load_ps(&p1[i]);
        __m128 v2 = _mm_load_ps(&p2[i]);
        __m128 diff = _mm_sub_ps(v1, v2);
        sumv = _mm_add_ps(sumv, _mm_mul_ps(diff, diff));
    }
    float f[4];
    _mm_store_ps(f, sumv);
    float sum = f[0] + f[1] + f[2] + f[3];
    for (; i < len; i++) {
        float diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

int cluster(int npoints, int nfeatures, float **features, int min_nclusters, int max_nclusters,
            float threshold, int *best_nclusters, float ***cluster_centres, float *min_rmse,
            int isRMSE, int nloops) {
    int index = 0;
    int *membership = (int*) malloc(npoints * sizeof(int));

    for (int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
        if (nclusters > npoints) break;

        float** clusters = (float**) malloc(nclusters * sizeof(float*));
        clusters[0] = (float*) _mm_malloc(nclusters * nfeatures * sizeof(float), 64);
        for (int i=1; i<nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;

        int* initial = (int *) malloc (npoints * sizeof(int));
        for (int i = 0; i < npoints; i++) initial[i] = i;
        int initial_points = npoints;

        for (int lp = 0; lp < nloops; lp++) {
            // Random cluster initialization
            for (int i=0; i<nclusters && initial_points > 0; i++) {
                int r = rand() % initial_points;
                memcpy(clusters[i], features[initial[r]], nfeatures * sizeof(float));
                initial[r] = initial[initial_points-1];
                initial_points--;
            }

            for (int i=0; i < npoints; i++) membership[i] = -1;

            int* new_centers_len = (int*) calloc(nclusters, sizeof(int));
            float** new_centers = (float**) malloc(nclusters * sizeof(float*));
            new_centers[0] = (float*) calloc(nclusters * nfeatures, sizeof(float));
            for (int i=1; i<nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;

            float delta;
            int loop = 0;
            do {
                delta = 0.0;
                // 1. Parallel Point Assignment (NUMA aware via OpenMP)
                #pragma omp parallel reduction(+:delta)
                {
                    #pragma omp for
                    for (int p = 0; p < npoints; p++) {
                        float min_dist = FLT_MAX;
                        int best_idx = 0;
                        for (int c = 0; c < nclusters; c++) {
                            float dist = euclid_dist_2_sse(features[p], clusters[c], nfeatures);
                            if (dist < min_dist) {
                                min_dist = dist;
                                best_idx = c;
                            }
                        }
                        if (membership[p] != best_idx) {
                            delta += 1.0;
                            membership[p] = best_idx;
                        }
                    }
                }

                // 2. Centroid Update (Parallel with local buffers to avoid atomic contention)
                memset(new_centers[0], 0, nclusters * nfeatures * sizeof(float));
                memset(new_centers_len, 0, nclusters * sizeof(int));

                #pragma omp parallel
                {
                    int* local_len = (int*) calloc(nclusters, sizeof(int));
                    float* local_centers = (float*) calloc(nclusters * nfeatures, sizeof(float));

                    #pragma omp for nowait
                    for (int i = 0; i < npoints; i++) {
                        int cid = membership[i];
                        local_len[cid]++;
                        for (int j = 0; j < nfeatures; j++) {
                            local_centers[cid * nfeatures + j] += features[i][j];
                        }
                    }

                    #pragma omp critical
                    {
                        for (int c = 0; c < nclusters; c++) {
                            new_centers_len[c] += local_len[c];
                            for (int j = 0; j < nfeatures; j++) {
                                new_centers[c][j] += local_centers[c * nfeatures + j];
                            }
                        }
                    }
                    free(local_len);
                    free(local_centers);
                }

                for (int i=0; i<nclusters; i++) {
                    if (new_centers_len[i] > 0) {
                        for (int j=0; j<nfeatures; j++) {
                            clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                        }
                    }
                }
            } while ((delta / npoints > threshold) && (loop++ < 500));

            free(new_centers[0]);
            free(new_centers);
            free(new_centers_len);

            if (isRMSE) {
                float current_rmse = rms_err(features, nfeatures, npoints, clusters, nclusters);
                if (current_rmse < min_rmse_ref) {
                    min_rmse_ref = current_rmse;
                    *min_rmse = min_rmse_ref;
                    *best_nclusters = nclusters;
                    index = lp;
                }
            }
        }
        if (*cluster_centres) {
            _mm_free((*cluster_centres)[0]);
            free(*cluster_centres);
        }
        *cluster_centres = clusters;
        free(initial);
    }
    free(membership);
    return index;
}
