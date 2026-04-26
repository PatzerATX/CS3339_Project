#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "kmeans_gemini_optimized.h"

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

int find_nearest_point_optimized(float *pt, int nfeatures, float **pts, int npts) {
    int index = 0;
    float min_dist = FLT_MAX;
    for (int i = 0; i < npts; i++) {
        float dist = euclid_dist_2_sse(pt, pts[i], nfeatures);
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

float rms_err(float **feature, int nfeatures, int npoints, float **cluster_centres, int nclusters) {
    double sum_euclid = 0.0;
    
    #pragma omp parallel reduction(+:sum_euclid)
    {
        #pragma omp for
        for (int i = 0; i < npoints; i++) {
            int nearest = find_nearest_point_optimized(feature[i], nfeatures, cluster_centres, nclusters);
            sum_euclid += (double)euclid_dist_2_sse(feature[i], cluster_centres[nearest], nfeatures);
        }
    }
    return (float)sqrt(sum_euclid / npoints);
}
