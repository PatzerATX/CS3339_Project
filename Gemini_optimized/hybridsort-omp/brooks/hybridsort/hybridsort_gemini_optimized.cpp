#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <immintrin.h>
#include <omp.h>
#include "bucketsort_gemini_optimized.h"
#include "mergesort_gemini_optimized.h"

#define SIZE (50000000)
#define TIMER

int compare(const void *a, const void *b) {
    float fa = *((float *)a);
    float fb = *((float *)b);
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

int main(int argc, char** argv) {
    int numElements = 0;
    if (argc < 2) {
        printf("Usage: %s <input_file | r>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "r") == 0) {
        numElements = SIZE;
    } else {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            printf("Error reading file \n");
            exit(EXIT_FAILURE);
        }
        float c;
        while (fscanf(fp, "%f", &c) != EOF) {
            numElements++;
        }
        fclose(fp);
    }

    printf("Sorting list of %d floats.\n", numElements);
    int mem_size = (numElements + (DIVISIONS * 4)) * sizeof(float);
    // Align to x86 cache line size (64 bytes)
    float *cpu_idata = (float *)_mm_malloc(mem_size, 64);
    float *cpu_odata = (float *)_mm_malloc(mem_size, 64);
    float *d_output = (float *)_mm_malloc(mem_size, 64);

    float datamin = FLT_MAX;
    float datamax = -FLT_MAX;

    if (strcmp(argv[1], "r") == 0) {
        // Parallel random generation and min/max with SSE/AVX
        #pragma omp parallel
        {
            float local_min = FLT_MAX;
            float local_max = -FLT_MAX;
            unsigned int seed = omp_get_thread_num();
            
            // First touch initialization for NUMA
            #pragma omp for nowait
            for (int i = 0; i < numElements; i++) {
                cpu_idata[i] = (float)rand_r(&seed) / RAND_MAX;
            }
            
            // Vectorized min/max
            __m128 vmin = _mm_set1_ps(FLT_MAX);
            __m128 vmax = _mm_set1_ps(-FLT_MAX);
            
            #pragma omp for nowait
            for (int i = 0; i <= numElements - 4; i += 4) {
                __m128 v = _mm_load_ps(&cpu_idata[i]);
                vmin = _mm_min_ps(vmin, v);
                vmax = _mm_max_ps(vmax, v);
            }
            
            float fmin_arr[4], fmax_arr[4];
            _mm_store_ps(fmin_arr, vmin);
            _mm_store_ps(fmax_arr, vmax);
            
            for(int k=0; k<4; k++) {
                local_min = fminf(local_min, fmin_arr[k]);
                local_max = fmaxf(local_max, fmax_arr[k]);
            }

            // Handle remainder
            #pragma omp for nowait
            for (int i = (numElements & ~3); i < numElements; i++) {
                local_min = fminf(cpu_idata[i], local_min);
                local_max = fmaxf(cpu_idata[i], local_max);
            }

            #pragma omp critical
            {
                datamin = fminf(datamin, local_min);
                datamax = fmaxf(datamax, local_max);
            }
        }
    } else {
        FILE *fp = fopen(argv[1], "r");
        for (int i = 0; i < numElements; i++) {
            fscanf(fp, "%f", &cpu_idata[i]);
            datamin = fminf(cpu_idata[i], datamin);
            datamax = fmaxf(cpu_idata[i], datamax);
        }
        fclose(fp);
    }

    memcpy(cpu_odata, cpu_idata, mem_size);

    int *sizes = (int*) malloc(DIVISIONS * sizeof(int));
    int *nullElements = (int*) malloc(DIVISIONS * sizeof(int));
    unsigned int *origOffsets = (unsigned int *) malloc((DIVISIONS + 1) * sizeof(int));

    auto bucketsort_start = std::chrono::steady_clock::now();
    bucketSort(cpu_idata, d_output, numElements, sizes, nullElements, datamin, datamax, origOffsets);
    auto bucketsort_end = std::chrono::steady_clock::now();
    auto bucketsort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(bucketsort_end - bucketsort_start).count();

    float4 *d_origList = (float4*) d_output;
    float4 *d_resultList = (float4*) cpu_idata;

    int newlistsize = 0;
    for (int i = 0; i < DIVISIONS; i++) {
        newlistsize += sizes[i] * 4;
    }

    auto mergesort_start = std::chrono::steady_clock::now();
    float4* mergeresult = runMergeSort(newlistsize, DIVISIONS, d_origList, d_resultList, sizes, nullElements, origOffsets);
    auto mergesort_end = std::chrono::steady_clock::now();
    auto mergesort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(mergesort_end - mergesort_start).count();
    float *gpu_odata = (float*)mergeresult;

#ifdef TIMER
    printf("Execution time: %0.3f ms\n", (bucketsort_diff + mergesort_diff) * 1e-6f);
    printf("  --Bucketsort: %0.3f ms\n", bucketsort_diff * 1e-6f);
    printf("  --Mergesort: %0.3f ms\n", mergesort_diff * 1e-6f);
#endif

    qsort(cpu_odata, numElements, sizeof(float), compare);
    printf("Checking result...");
    int count = 0;
    for (int i = 0; i < numElements; i++) {
        if (cpu_odata[i] != gpu_odata[i]) {
            printf("Mismatch at %d: CPU=%f, GPU=%f\n", i, cpu_odata[i], gpu_odata[i]);
            count++;
            break;
        }
    }
    if (count == 0) printf("PASSED.\n");
    else printf("FAILED.\n");

    _mm_free(cpu_idata);
    _mm_free(cpu_odata);
    _mm_free(d_output);
    free(sizes);
    free(nullElements);
    free(origOffsets);

    return 0;
}
