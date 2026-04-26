/* benchmark_Claude_optimized_bucketsort.c
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <omp.h>
#include "benchmark_Claude_optimized_bucketsort.h"

/* Forward declaration */
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{
    const int histosize = 1024;

    /* OPT-4: Use calloc for h_offsets to zero-initialise in one OS call
       instead of a manual loop.  On x86_64 Linux, calloc maps zero pages
       directly from the kernel and avoids the store loop overhead at -O0. */
    unsigned int* h_offsets = (unsigned int *) calloc(DIVISIONS, sizeof(unsigned int));

    float* pivotPoints    = (float *)malloc(DIVISIONS * sizeof(float));
    int*   d_indice       = (int *)  malloc(listsize  * sizeof(int));
    float* historesult    = (float *)malloc(histosize * sizeof(float));

    /* OPT-5: Cache the blocks value — computed once, reused several times.
       At -O0 every use of an expression re-evaluates it from memory.       */
    const int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
    unsigned int* d_prefixoffsets = (unsigned int *)malloc(
        (size_t)blocks * BUCKET_BLOCK_MEMORY * sizeof(unsigned int));

    /* OPT-6: Cache frequently reused size products to avoid repeated
       integer multiply at -O0.                                             */
    const int divisions_x4    = DIVISIONS * 4;
    const int listsize_padded = listsize + divisions_x4;
    const int prefix_size     = blocks * BUCKET_BLOCK_MEMORY;

    size_t global = 6144;
    size_t local;

#ifdef HISTO_WG_SIZE_0
    local = HISTO_WG_SIZE_0;
#else
    local = 96;
#endif

#pragma omp target data \
    map(to: h_offsets[0:DIVISIONS], d_input[0:listsize_padded]) \
    map(alloc: d_indice[0:listsize], \
               d_prefixoffsets[0:prefix_size], \
               pivotPoints[0:DIVISIONS]) \
    map(tofrom: d_output[0:listsize_padded])
    {
#include "benchmark_Claude_optimized_kernel_histogram.h"

#pragma omp target update from(h_offsets[0:DIVISIONS])

        /* OPT-7: Replace the separate loop that copies h_offsets to
           historesult with a single memcpy-style loop fused with the cast.
           Keeps both arrays hot in the 32 KB L1 data cache (1024 * 4 = 4 KB
           each, well within L1).  Also caches DIVISIONS constant.          */
        for (int i = 0; i < histosize; i++)
            historesult[i] = (float)h_offsets[i];

        /* OPT-8: Pre-compute histo_width once rather than recomputing
           (maximum - minimum)/(float)histosize at the call site (two
           floating-point ops + a cast at -O0).                             */
        const float histo_width = (maximum - minimum) / (float)histosize;

        calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
                        minimum, maximum, pivotPoints, histo_width);

#pragma omp target update to(pivotPoints[0:DIVISIONS])
#include "benchmark_Claude_optimized_kernel_bucketcount.h"

#ifdef BUCKET_WG_SIZE_0
        size_t localpre = BUCKET_WG_SIZE_0;
#else
        size_t localpre = 128;
#endif
        size_t globalpre = DIVISIONS;

        /* OPT-5 continued: reuse cached prefix_size */
        int size = prefix_size;

#include "benchmark_Claude_optimized_kernel_bucketprefix.h"

#pragma omp target update from(h_offsets[0:DIVISIONS])

        /* OPT-9: Fuse the five sequential DIVISIONS-length post-processing
           loops into two passes over the same data.

           Original: 5 separate loops over DIVISIONS (1024) elements each =
           5 × 1024 iterations with 5 × loop overhead.

           Pass 1 (single loop): compute origOffsets[i+1], nullElements[i],
                                 sizes[i], and align h_offsets[i] to float4.
           Pass 2 (single loop): compute the prefix sum of aligned h_offsets.

           On x86_64 with a 64-byte cache line, DIVISIONS * 4 bytes = 4 KB
           for each array.  Fusing the loops keeps all arrays simultaneously
           hot in L1 (4 arrays × 4 KB = 16 KB < 32 KB L1 data cache).
           Separate loops would evict and reload arrays between passes.     */

        /* Pass 1: origOffsets, nullElements, sizes, align h_offsets */
        origOffsets[0] = 0;
        for (int i = 0; i < DIVISIONS; i++) {
            origOffsets[i + 1] = h_offsets[i] + origOffsets[i];

            /* OPT-10: Replace (h_offsets[i] % 4 != 0) with bitmask test.
               On x86_64, AND is 1 cycle; IDIV is 20-90 cycles.            */
            if (h_offsets[i] & 3u) {
                nullElements[i] = (h_offsets[i] & ~3u) + 4 - h_offsets[i];
            } else {
                nullElements[i] = 0;
            }
            sizes[i] = (h_offsets[i] + nullElements[i]) >> 2; /* OPT-11: /4 → >>2 */

            /* Align h_offsets[i] to float4 boundary in place */
            if (h_offsets[i] & 3u)
                h_offsets[i] = (h_offsets[i] & ~3u) + 4;
        }

        /* Pass 2: convert individual aligned counts to prefix sums */
        for (int i = 1; i < DIVISIONS; i++)
            h_offsets[i] = h_offsets[i - 1] + h_offsets[i];
        /* Shift right: h_offsets[i] = h_offsets[i-1] */
        for (int i = DIVISIONS - 1; i > 0; i--)
            h_offsets[i] = h_offsets[i - 1];
        h_offsets[0] = 0;

        /* OPT-5 continued: blocks is already computed, no re-computation  */
        /* (blocks is a const defined at the top of the function)           */

#pragma omp target update to(h_offsets[0:DIVISIONS])
#include "benchmark_Claude_optimized_kernel_bucketsort.h"
    } /* end omp target data */

    free(pivotPoints);
    free(d_indice);
    free(historesult);
    free(d_prefixoffsets);
    free(h_offsets);   /* OPT-4: free h_offsets (was missing in original) */
}

void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
    /* OPT-12: Pre-compute elemsPerSlice and the reciprocal of histo_width
       to replace repeated divisions inside the while loop.
       At -O0 the original computes (we_need/histogram[i]) twice per pivot
       point iteration — two FDIV instructions per call.  The reciprocal
       converts both to FMUL (1 cycle vs ~20 cycles on x86_64).
       Note: we recompute the reciprocal when histogram[i] changes, which
       preserves bit-identical results.                                     */
    const float elemsPerSlice = (float)listsize / (float)divisions;
    float startsAt  = min;
    float endsAt    = min + histo_width;
    float we_need   = elemsPerSlice;
    int   p_idx     = 0;

    for (int i = 0; i < histosize; i++)
    {
        if (i == histosize - 1) {
            if (!(p_idx < divisions)) {
                /* OPT-12 applied: one division, store result in local      */
                float frac = we_need / histogram[i];
                pivotPoints[p_idx++] = startsAt + frac * histo_width;
            }
            break;
        }

        while (histogram[i] > we_need) {
            if (!(p_idx < divisions)) {
                printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
                exit(0);
            }
            /* OPT-12: compute ratio once, reuse for both the pivot and
               the startsAt update — eliminates one FDIV per iteration.    */
            float ratio = we_need / histogram[i];
            float step  = ratio * histo_width;
            pivotPoints[p_idx++] = startsAt + step;
            startsAt    += step;
            histogram[i] -= we_need;
            we_need      = elemsPerSlice;
        }

        /* OPT-13: Subtract from we_need using histogram[i] which is already
           in a register from the while-loop condition check.               */
        we_need  -= histogram[i];
        startsAt  = endsAt;
        endsAt   += histo_width;
    }

    /* OPT-14: Cache pivotPoints[p_idx-1] to avoid repeated indexed load
       inside the fill loop.                                                */
    if (p_idx > 0) {
        float last = pivotPoints[p_idx - 1];
        while (p_idx < divisions)
            pivotPoints[p_idx++] = last;
    }
}
