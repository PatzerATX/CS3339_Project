/* benchmark_Claude_optimized_mergesort.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#ifndef __MERGESORT
#define __MERGESORT

#include "benchmark_Claude_optimized_bucketsort.h"

float4* runMergeSort(int listsize, int divisions,
                     float4 *d_origList, float4 *d_resultList,
                     int *sizes, int *nullElements,
                     unsigned int *origOffsets);
#endif
