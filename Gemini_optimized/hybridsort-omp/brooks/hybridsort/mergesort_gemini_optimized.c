#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "mergesort_gemini_optimized.h"

// SSE optimized sort for 4 floats
static inline __m128 sortElem_sse(__m128 v) {
    float f[4];
    _mm_store_ps(f, v);
    if (f[0] > f[1]) { float t = f[0]; f[0] = f[1]; f[1] = t; }
    if (f[2] > f[3]) { float t = f[2]; f[2] = f[3]; f[3] = t; }
    if (f[0] > f[2]) { float t = f[0]; f[0] = f[2]; f[2] = t; }
    if (f[1] > f[3]) { float t = f[1]; f[1] = f[3]; f[3] = t; }
    if (f[1] > f[2]) { float t = f[1]; f[1] = f[2]; f[2] = t; }
    return _mm_load_ps(f);
}

static inline __m128 getLowest_sse(__m128 a, __m128 b) {
    // b.w, b.z, b.y, b.x -> reversed b
    __m128 b_rev = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm_min_ps(a, b_rev);
}

static inline __m128 getHighest_sse(__m128 a, __m128 b) {
    __m128 b_rev = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3));
    __m128 res = _mm_max_ps(a, b_rev);
    return _mm_shuffle_ps(res, res, _MM_SHUFFLE(0, 1, 2, 3));
}

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets)
{
    int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
    startaddr[0] = 0;
    for(int i=1; i<=divisions; i++) {
        startaddr[i] = startaddr[i-1] + sizes[i-1];
    }

    // Initial sort
    #pragma omp parallel for
    for (int i = 0; i < listsize/4; i++) {
        d_resultList[i].v = sortElem_sse(d_origList[i].v);
    }

    int nrElems = 2;
    while(1) {
        float4 *src = d_resultList;
        float4 *dst = d_origList;
        
        #pragma omp parallel for
        for (int div = 0; div < divisions; div++) {
            int div_start = startaddr[div];
            int div_end = startaddr[div+1];
            
            for (int Astart = div_start; Astart < div_end; Astart += nrElems) {
                int Bstart = Astart + nrElems/2;
                float4 *resStart = &dst[Astart];
                
                if (Bstart >= div_end) {
                    for (int i = 0; i < (div_end - Astart); i++) {
                        resStart[i] = src[Astart + i];
                    }
                } else {
                    int aidx = 0, bidx = 0, outidx = 0;
                    __m128 a = src[Astart].v;
                    __m128 b = src[Bstart].v;
                    int max_a = nrElems/2;
                    int max_b = fminf(nrElems/2, div_end - Bstart);

                    while (outidx < (max_a + max_b - 1)) {
                        __m128 na = getLowest_sse(a, b);
                        __m128 nb = getHighest_sse(a, b);
                        a = sortElem_sse(na);
                        b = sortElem_sse(nb);
                        resStart[outidx++].v = a;

                        bool can_inc_a = (aidx + 1 < max_a);
                        bool can_inc_b = (bidx + 1 < max_b);

                        if (can_inc_a && can_inc_b) {
                            if (src[Astart + aidx + 1].f[0] < src[Bstart + bidx + 1].f[0]) {
                                aidx++; a = src[Astart + aidx].v;
                            } else {
                                bidx++; a = src[Bstart + bidx].v;
                            }
                        } else if (can_inc_a) {
                            aidx++; a = src[Astart + aidx].v;
                        } else if (can_inc_b) {
                            bidx++; a = src[Bstart + bidx].v;
                        } else {
                            break;
                        }
                    }
                    resStart[outidx++].v = b;
                }
            }
        }
        
        float4 *tmp = d_resultList;
        d_resultList = d_origList;
        d_origList = tmp;

        bool done = true;
        for (int i=0; i<divisions; i++) {
            if (nrElems < (startaddr[i+1] - startaddr[i])) {
                done = false;
                break;
            }
        }
        if (done) break;
        nrElems *= 2;
    }

    float* d_orig = (float*)d_origList;
    float* d_res = (float*)d_resultList;
    #pragma omp parallel for collapse(2)
    for (int div = 0; div < divisions; div++) {
        for (int idx = 0; idx < (origOffsets[div+1] - origOffsets[div]); idx++) {
            d_orig[origOffsets[div] + idx] = d_res[startaddr[div]*4 + nullElements[div] + idx];
        }
    }

    free(startaddr);
    return (float4*)d_orig;
}
