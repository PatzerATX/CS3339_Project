#ifndef _H_KMEANS_GEMINI_OPTIMIZED
#define _H_KMEANS_GEMINI_OPTIMIZED

#include <arm_neon.h>

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/* rmse.c */
float   euclid_dist_2        (float*, float*, int);
int     find_nearest_point   (float* , int, float**, int);
float	rms_err(float**, int, int, float**, int);

int     cluster(int, int, float**, int, int, float, int*, float***, float*, int, int);
int setup(int argc, char** argv);

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 256

#endif
