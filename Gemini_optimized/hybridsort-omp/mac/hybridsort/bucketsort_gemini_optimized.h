#ifndef __BUCKETSORT_GEMINI_OPTIMIZED
#define __BUCKETSORT_GEMINI_OPTIMIZED

#include <arm_neon.h>

#define LOG_DIVISIONS	10
#define DIVISIONS		(1 << LOG_DIVISIONS)

#define BUCKET_WARP_LOG_SIZE	5
#define BUCKET_WARP_N			1

#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				128

#define HISTOGRAM_BIN_COUNT  1024

// define float4 for ARM64 NEON
typedef union {
    struct { float x, y, z, w; };
    float f[4];
    float32x4_t v;
} float4;

void bucketSort(float *d_input, float *d_output, int listsize,
				int *sizes, int *nullElements, float minimum, float maximum,
				unsigned int *origOffsets);

#endif
