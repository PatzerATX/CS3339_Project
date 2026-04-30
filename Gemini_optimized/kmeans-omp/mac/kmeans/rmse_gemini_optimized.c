#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
#include "kmeans_gemini_optimized.h"

// Manual NEON SIMD distance calculation
static inline float euclid_dist_2_neon(float *p1, float *p2, int len) {
    float32x4_t sumv = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i <= len - 4; i += 4) {
        float32x4_t v1 = vld1q_f32(&p1[i]);
        float32x4_t v2 = vld1q_f32(&p2[i]);
        float32x4_t diff = vsubq_f32(v1, v2);
        sumv = vfmaq_f32(sumv, diff, diff);
    }
    float sum = vgetq_lane_f32(sumv, 0) + vgetq_lane_f32(sumv, 1) + vgetq_lane_f32(sumv, 2) + vgetq_lane_f32(sumv, 3);
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
        float dist = euclid_dist_2_neon(pt, pts[i], nfeatures);
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
            sum_euclid += (double)euclid_dist_2_neon(feature[i], cluster_centres[nearest], nfeatures);
        }
    }
    return (float)sqrt(sum_euclid / npoints);
}
