#ifndef _H_FUZZY_KMEANS
#define _H_FUZZY_KMEANS

/* Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/* rmse.c */
float euclid_dist_2      (float *, float *, int);
int   find_nearest_point (float *, int, float **, int);
float rms_err            (float **, int, int, float **, int);

int cluster(int, int, float **, int, int, float, int *, float ***, float *, int, int);
int setup(int argc, char **argv);

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE2 RD_WG_SIZE
#else
#define BLOCK_SIZE2 256
#endif

#endif /* _H_FUZZY_KMEANS */
