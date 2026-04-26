#ifndef _H_KMEANS_GEMINI_OPTIMIZED
#define _H_KMEANS_GEMINI_OPTIMIZED

#include <immintrin.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/* rmse_gemini_optimized.c */
float   euclid_dist_2        (float*, float*, int);
int     find_nearest_point   (float* , int, float**, int);
float	rms_err(float**, int, int, float**, int);

/* cluster_gemini_optimized.c */
int     cluster(int, int, float**, int, int, float, int*, float***, float*, int, int);

/* read_input_gemini_optimized.cpp */
int setup(int argc, char** argv);

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 256

#endif
