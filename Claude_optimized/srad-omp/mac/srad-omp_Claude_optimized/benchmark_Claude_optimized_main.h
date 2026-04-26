#define fp float

/* Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 256
#endif

/* Pre-computed constants to replace repeated literal arithmetic at -O0 */
#define FP_HALF       ((fp)0.5f)
#define FP_ONE        ((fp)1.0f)
#define FP_QUARTER    ((fp)0.25f)
#define FP_INV16      ((fp)0.0625f)   /* 1/16 */
#define FP_255        ((fp)255.0f)

/* Clamp macro: avoids branch at -O0 by using conditional expressions,
   which on ARM64 compile to FCSEL (no branch).                         */
#define CLAMP01(x)  ( (x) < 0.0f ? 0.0f : ( (x) > 1.0f ? 1.0f : (x) ) )
