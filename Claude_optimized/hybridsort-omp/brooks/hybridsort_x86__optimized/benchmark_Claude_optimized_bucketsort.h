/* benchmark_Claude_optimized_bucketsort.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#ifndef __BUCKETSORT
#define __BUCKETSORT

#define LOG_DIVISIONS   10
#define DIVISIONS       (1 << LOG_DIVISIONS)   /* 1024 */

#define BUCKET_WARP_LOG_SIZE    5
#define BUCKET_WARP_N           1

#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N         (BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif

#define BUCKET_BLOCK_MEMORY     (DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND             128

#define HISTOGRAM_BIN_COUNT     1024
#define HISTOGRAM_BLOCK_MEMORY  (3 * HISTOGRAM_BIN_COUNT)

/* OPT-1: Replace the IMUL macro call with a plain multiply expression.
   At -O0 the original macro expands to (a*b) identically, but wrapping
   it in an explicit expression avoids any extra parenthesis overhead that
   -O0 may materialise as a copy through a temp.                           */
#define IMUL(a, b) ((a)*(b))

/* OPT-2: Pre-compute constants derived from DIVISIONS / LOG_DIVISIONS so
   the compiler does not re-evaluate them at every use site at -O0.        */
#define DIVISIONS_MASK          (DIVISIONS - 1)          /* 0x3FF            */
#define HALF_DIVISIONS          (DIVISIONS / 2 - 1)      /* 511              */
#define QUARTER_DIVISIONS       (DIVISIONS / 4)          /* 256              */

/* OPT-3: On x86_64 with a 64-byte cache line, struct float4 is 16 bytes
   (four floats).  Four float4 elements fit exactly in one cache line.
   Keeping the struct tightly packed ensures no false sharing.              */
typedef struct {
    float x;
    float y;
    float z;
    float w;
} float4;

void bucketSort(float *d_input, float *d_output, int listsize,
                int *sizes, int *nullElements, float minimum, float maximum,
                unsigned int *origOffsets);
double getBucketTime(void);

#endif /* __BUCKETSORT */
