#ifndef __MERGESORT_GEMINI_OPTIMIZED
#define __MERGESORT_GEMINI_OPTIMIZED

#include "bucketsort_gemini_optimized.h"

float4* runMergeSort(int listsize, int divisions,
					 float4 *d_origList, float4 *d_resultList,
					 int *sizes, int *nullElements,
					 unsigned int *origOffsets);
#endif
