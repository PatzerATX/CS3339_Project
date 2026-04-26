/* benchmark_Claude_optimized_hybridsort.c
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "benchmark_Claude_optimized_bucketsort.h"
#include "benchmark_Claude_optimized_mergesort.h"
#include <chrono>
#define TIMER

#define SIZE (50000000)

/* ── Comparison function ───────────────────────────────────────────────── */
int compare(const void *a, const void *b)
{
    /* OPT-49: Dereference once into locals to avoid two pointer loads
       at every comparison call at -O0.                                     */
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return  1;
    return 0;
}

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets);

int main(int argc, char **argv)
{
    /* ── Determine input size ─────────────────────────────────────────── */
    int numElements = 0;

    if (strcmp(argv[1], "r") == 0) {
        numElements = SIZE;
    } else {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            printf("Error reading file\n");
            exit(EXIT_FAILURE);
        }
        /* OPT-50: Count elements in a single fscanf loop without storing
           values — avoids allocating a temporary buffer just to count.    */
        float c;
        int count = 0;
        while (fscanf(fp, "%f", &c) != EOF)
            count++;
        fclose(fp);
        numElements = count;
    }

    printf("Sorting list of %d floats.\n", numElements);

    /* OPT-51: Compute mem_size once and reuse — avoids repeating the
       same arithmetic expression (numElements + DIVISIONS*4)*sizeof(float)
       at every malloc call at -O0.                                         */
    const int mem_size = (numElements + (DIVISIONS * 4)) * (int)sizeof(float);

    float *cpu_idata   = (float *)malloc(mem_size);
    float *cpu_odata   = (float *)malloc(mem_size);
    float *d_output    = (float *)malloc(mem_size);
    float *gpu_odata;

    float datamin =  FLT_MAX;
    float datamax = -FLT_MAX;

    if (strcmp(argv[1], "r") == 0) {
        /* OPT-52: Fuse the random generation loop with the min/max scan.
           Original: generates all elements, then fminf/fmaxf inline but
           in the same loop — already fused, preserved.                    */
        for (int i = 0; i < numElements; i++) {
            float v        = (float)rand() / (float)RAND_MAX;
            cpu_idata[i]   = v;
            if (v < datamin) datamin = v;
            if (v > datamax) datamax = v;
        }
    } else {
        FILE *fp = fopen(argv[1], "r");
        for (int i = 0; i < numElements; i++) {
            fscanf(fp, "%f", &cpu_idata[i]);
            float v = cpu_idata[i];
            if (v < datamin) datamin = v;
            if (v > datamax) datamax = v;
        }
        fclose(fp);
    }

    /* OPT-53: Write hybridinput.txt using a larger fprintf buffer.
       At -O0 each fprintf call is a full library call.  Use fwrite with
       a local buffer to batch output — significantly reduces call overhead
       for 50M elements.  We use the standard approach of setting a larger
       stdio buffer via setvbuf.                                            */
    FILE *tp = fopen("./hybridinput.txt", "w");
    if (tp) {
        /* OPT-54: Set a 1 MB write buffer to reduce per-call stdio
           overhead.  The default buffer is typically 4–8 KB; with
           50M floats, the default causes many more flush syscalls.         */
        const int WBUF_SZ = 1 << 20;  /* 1 MB */
        char *wbuf = (char *)malloc(WBUF_SZ);
        if (wbuf) setvbuf(tp, wbuf, _IOFBF, WBUF_SZ);

        /* OPT-53 continued: write numElements floats (not SIZE) to match
           actual data; original wrote SIZE which may exceed numElements.  */
        for (int i = 0; i < numElements; i++)
            fprintf(tp, "%f ", cpu_idata[i]);
        fclose(tp);
        if (wbuf) free(wbuf);
    }

    /* OPT-55: Use memcpy for cpu_odata initialisation — already correct
       in the original; preserved.                                          */
    memcpy(cpu_odata, cpu_idata, mem_size);

    int *sizes        = (int *)         malloc(DIVISIONS       * sizeof(int));
    int *nullElements = (int *)         malloc(DIVISIONS       * sizeof(int));
    unsigned int *origOffsets = (unsigned int *)malloc((DIVISIONS + 1) * sizeof(unsigned int));

    /* ── Bucketsort ──────────────────────────────────────────────────── */
    auto bucketsort_start = std::chrono::steady_clock::now();
    bucketSort(cpu_idata, d_output, numElements, sizes, nullElements,
               datamin, datamax, origOffsets);
    auto bucketsort_end  = std::chrono::steady_clock::now();
    auto bucketsort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               bucketsort_end - bucketsort_start).count();

    float4 *d_origList   = (float4 *)d_output;
    float4 *d_resultList = (float4 *)cpu_idata;

    /* OPT-56: Replace the DIVISIONS-length newlistsize accumulation loop
       with a direct computation from origOffsets if available, or keep
       the loop with a local accumulator to avoid repeated memory traffic.  */
    int newlistsize = 0;
    for (int i = 0; i < DIVISIONS; i++)
        newlistsize += sizes[i] * 4;

    /* ── Mergesort ───────────────────────────────────────────────────── */
    auto mergesort_start = std::chrono::steady_clock::now();
    float4 *mergeresult = runMergeSort(newlistsize, DIVISIONS,
                                       d_origList, d_resultList,
                                       sizes, nullElements, origOffsets);
    auto mergesort_end  = std::chrono::steady_clock::now();
    auto mergesort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              mergesort_end - mergesort_start).count();

    gpu_odata = (float *)mergeresult;

#ifdef TIMER
    /* OPT-57: Pre-compute the conversion factor 1e-6f once, reuse.        */
    const float ns_to_ms = 1e-6f;
    float bucketsort_msec = (float)bucketsort_diff * ns_to_ms;
    float mergesort_msec  = (float)mergesort_diff  * ns_to_ms;
    printf("GPU execution time: %0.3f ms\n", bucketsort_msec + mergesort_msec);
    printf("  --Bucketsort execution time: %0.3f ms\n", bucketsort_msec);
    printf("  --Mergesort execution time: %0.3f ms\n",  mergesort_msec);
#endif

    /* ── CPU reference sort ──────────────────────────────────────────── */
    clock_t cpu_start = clock();
    qsort(cpu_odata, numElements, sizeof(float), compare);
    clock_t cpu_diff  = clock() - cpu_start;
    /* OPT-58: Pre-compute 1000.0f/CLOCKS_PER_SEC to replace division
       with multiply.                                                       */
    float cpu_msec = (float)cpu_diff * (1000.0f / (float)CLOCKS_PER_SEC);
    printf("CPU execution time: %0.3f ms\n", cpu_msec);
    printf("Checking result...");

    /* OPT-59: Break early on first mismatch — already done in original;
       preserved.  Also: use a single int flag instead of 'count'.         */
    int mismatch = 0;
    for (int i = 0; i < numElements; i++) {
        if (cpu_odata[i] != gpu_odata[i]) {
            printf("Sort mismatch on element %d:\n", i);
            printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]);
            mismatch = 1;
            break;
        }
    }
    if (!mismatch) printf("PASSED.\n");
    else           printf("FAILED.\n");

#ifdef OUTPUT
    FILE *tp1 = fopen("./hybridoutput.txt", "w");
    if (tp1) {
        for (int i = 0; i < SIZE; i++)
            fprintf(tp1, "%f ", cpu_idata[i]);
        fclose(tp1);
    }
#endif

    free(cpu_idata);
    free(cpu_odata);
    free(d_output);
    free(sizes);
    free(nullElements);
    free(origOffsets);

    return 0;
}
