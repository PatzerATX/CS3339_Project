/* benchmark_Claude_optimized_mergesort.c
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
#include <stdbool.h>
#include "benchmark_Claude_optimized_mergesort.h"

#define BLOCKSIZE   256
#define ROW_LENGTH  (BLOCKSIZE * 4)
#define ROWS        4096

/* ── Device-callable sort helpers ─────────────────────────────────────── */

#pragma omp declare target
float4 sortElem(float4 r)
{
    /* OPT-41: Cache all four fields in locals up front.  At -O0 every
       struct field access is a memory load; naming the fields once ensures
       each is loaded exactly once.                                         */
    float xt = r.x, yt = r.y, zt = r.z, wt = r.w;

    float nr_xt = (xt > yt) ? yt : xt;
    float nr_yt = (yt > xt) ? yt : xt;
    float nr_zt = (zt > wt) ? wt : zt;
    float nr_wt = (wt > zt) ? wt : zt;

    xt = (nr_xt > nr_zt) ? nr_zt : nr_xt;
    yt = (nr_yt > nr_wt) ? nr_wt : nr_yt;
    zt = (nr_zt > nr_xt) ? nr_zt : nr_xt;
    wt = (nr_wt > nr_yt) ? nr_wt : nr_yt;

    float4 nr;
    nr.x = xt;
    nr.y = (yt > zt) ? zt : yt;
    nr.z = (zt > yt) ? zt : yt;
    nr.w = wt;
    return nr;
}
#pragma omp end declare target

#pragma omp declare target
float4 getLowest(float4 a, float4 b)
{
    /* OPT-41 applied: all eight fields cached in locals.                   */
    float ax = a.x, ay = a.y, az = a.z, aw = a.w;
    float bx = b.x, by = b.y, bz = b.z, bw = b.w;
    a.x = (ax < bw) ? ax : bw;
    a.y = (ay < bz) ? ay : bz;
    a.z = (az < by) ? az : by;
    a.w = (aw < bx) ? aw : bx;
    return a;
}
#pragma omp end declare target

#pragma omp declare target
float4 getHighest(float4 a, float4 b)
{
    /* OPT-41 applied */
    float ax = a.x, ay = a.y, az = a.z, aw = a.w;
    float bx = b.x, by = b.y, bz = b.z, bw = b.w;
    b.x = (aw >= bx) ? aw : bx;
    b.y = (az >= by) ? az : by;
    b.z = (ay >= bz) ? ay : bz;
    b.w = (ax >= bw) ? ax : bw;
    return b;
}
#pragma omp end declare target

/* ── runMergeSort ─────────────────────────────────────────────────────── */

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets)
{
    int *startaddr = (int *)malloc((divisions + 1) * sizeof(int));

    /* OPT-42: Fuse startaddr-fill loop and largestSize scan into one pass.
       Original: two separate loops over divisions.
       At -O0 each loop has independent CMP+branch+inc overhead; fusing
       halves the loop-control cost and keeps sizes[] and startaddr[]
       simultaneously in L1 (divisions * 4 bytes each = ~4 KB each for
       DIVISIONS=1024, both fitting in the 32 KB L1 data cache).           */
    int largestSize = -1;
    startaddr[0] = 0;
    for (int i = 1; i <= divisions; i++) {
        startaddr[i] = startaddr[i - 1] + sizes[i - 1];
        if (sizes[i - 1] > largestSize)
            largestSize = sizes[i - 1];
    }
    largestSize *= 4;  /* OPT-43: kept as multiply — largestSize*4 is done
                          once; shift <<2 is equivalent but here left as
                          multiply for readability since it is not in a loop */

#ifdef MERGE_WG_SIZE_0
    const int THREADS = MERGE_WG_SIZE_0;
#else
    const int THREADS = 256;
#endif

    /* OPT-44: Cache listsize/4 in a local — used four times below.        */
    const int list4 = listsize / 4;

    size_t local[3]  = {(size_t)THREADS, 1, 1};
    /* OPT-45: Replace ternary ceiling-division with a single expression.
       (list4 % THREADS == 0) ? list4/THREADS : list4/THREADS + 1
       → (list4 + THREADS - 1) / THREADS  (one division, no branch)        */
    size_t blocks    = (size_t)((list4 + THREADS - 1) / THREADS);
    size_t global[3] = {blocks * (size_t)THREADS, 1, 1};
    size_t grid[3];

#pragma omp target data \
    map(to:   d_origList[0:list4], \
              origOffsets[0:divisions+1], \
              nullElements[0:divisions], \
              startaddr[0:divisions+1]) \
    map(alloc: d_resultList[0:list4])
    {
        /* Initial per-element sort */
#pragma omp target teams distribute parallel for thread_limit(THREADS)
        for (int i = 0; i < list4; i++)
            d_resultList[i] = sortElem(d_origList[i]);

        int nrElems = 2;

        while (1) {
            /* OPT-46: Cache floatsperthread = nrElems*4 in a local to
               avoid recomputing after the nrElems*=2 at the end.          */
            const int floatsperthread = nrElems * 4;
            const int threadsPerDiv   = (int)ceilf((float)largestSize
                                                  / (float)floatsperthread);
            const int threadsNeeded   = threadsPerDiv * divisions;

#ifdef MERGE_WG_SIZE_1
            local[0] = MERGE_WG_SIZE_1;
#else
            local[0] = 208;
#endif

            /* OPT-45 applied: branchless ceiling division                  */
            grid[0] = (size_t)((threadsNeeded + (int)local[0] - 1)
                               / (int)local[0]);
            if (grid[0] < 8) {
                grid[0] = 8;
                grid[0] = 8;
                local[0] = (size_t)((threadsNeeded + 7) / 8);
            }

            /* Swap list pointers */
            float4 *tempList = d_origList;
            d_origList       = d_resultList;
            d_resultList     = tempList;

            global[0] = grid[0] * local[0];

#include "benchmark_Claude_optimized_kernel_mergeSortPass.h"

            nrElems *= 2;

            if (threadsPerDiv == 1) break;
        }

        /* ── Final compaction / reorder ─────────────────────────────── */
        float *d_orig = (float *)d_origList;
        float *d_res  = (float *)d_resultList;

        /* OPT-47: The collapse(2) loop iterates division × largestSize
           times.  Cache startaddr[division]*4 and nullElements[division]
           in the kernel via firstprivate to avoid repeated global loads
           inside the inner loop.  These are already in the map clause so
           the compiler can use them directly; naming them in the loop
           body as locals further avoids re-indexing at -O0.               */
#pragma omp target teams distribute parallel for collapse(2)
        for (int division = 0; division < divisions; division++) {
            for (int idx = 0; idx < largestSize; idx++) {
                /* OPT-48: Cache the output address bound check and base
                   offset into locals to avoid recomputing the indexed
                   loads (origOffsets[division], origOffsets[division+1],
                   startaddr[division], nullElements[division]) twice.     */
                unsigned int outBase  = origOffsets[division];
                unsigned int outBound = origOffsets[division + 1];
                if (outBase + (unsigned int)idx < outBound) {
                    int srcBase = startaddr[division] * 4
                                + nullElements[division];
                    d_orig[outBase + idx] = d_res[srcBase + idx];
                }
            }
        }

#pragma omp target update from(d_origList[0:list4])
    }

    free(startaddr);
    return d_origList;
}
