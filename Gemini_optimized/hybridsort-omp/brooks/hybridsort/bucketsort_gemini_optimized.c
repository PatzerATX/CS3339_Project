#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "bucketsort_gemini_optimized.h"

void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  float elemsPerSlice = listsize/(float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;
  for(int i=0; i<histosize; i++)
  {
    if(i == histosize - 1){
      if(p_idx < divisions){
        pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      }
      break;
    }
    while(histogram[i] > we_need){
      if(p_idx >= divisions) break;
      pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      startsAt += (we_need/histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }
    we_need -= histogram[i];
    startsAt = endsAt;
    endsAt += histo_width;
  }
  while(p_idx < divisions){
    pivotPoints[p_idx] = pivotPoints[p_idx-1];
    p_idx++;
  }
}

void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{
  const int histosize = 1024;
  unsigned int h_offsets[DIVISIONS];
  memset(h_offsets, 0, sizeof(h_offsets));

  float* pivotPoints = (float *)_mm_malloc(DIVISIONS * sizeof(float), 64);
  int* d_indice = (int *)_mm_malloc(listsize * sizeof(int), 64);
  float historesult[histosize];

  // 1. Histogram (Parallel with thread-local storage for NUMA efficiency)
  #pragma omp parallel
  {
    unsigned int local_histo[histosize] = {0};
    #pragma omp for nowait
    for(int pos = 0; pos < listsize; pos++) {
      uint32_t bin = (uint32_t)(((d_input[pos] - minimum)/(maximum - minimum)) * (histosize - 1));
      local_histo[bin & 0x3FFU]++;
    }
    #pragma omp critical
    {
      for(int i=0; i<histosize; i++) h_offsets[i] += local_histo[i];
    }
  }

  for(int i=0; i<histosize; i++) historesult[i] = (float)h_offsets[i];

  // 2. Pivot Points
  calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
      minimum, maximum, pivotPoints,
      (maximum - minimum)/(float)histosize);

  // 3. Bucket Assignment
  unsigned int *thread_offsets = (unsigned int*)_mm_malloc(omp_get_max_threads() * DIVISIONS * sizeof(unsigned int), 64);
  memset(thread_offsets, 0, omp_get_max_threads() * DIVISIONS * sizeof(unsigned int));

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int *my_counts = &thread_offsets[tid * DIVISIONS];
    #pragma omp for
    for (int i = 0; i < listsize; i++) {
      float elem = d_input[i];
      int idx = DIVISIONS / 2 - 1;
      int jump = DIVISIONS / 4;
      // Unrolled Binary Search for -O0
      while(jump >= 1){
        idx = (elem < pivotPoints[idx]) ? (idx - jump) : (idx + jump);
        jump /= 2;
      }
      idx = (elem < pivotPoints[idx]) ? idx : (idx + 1);
      if (idx >= DIVISIONS) idx = DIVISIONS - 1;
      d_indice[i] = (my_counts[idx] << LOG_DIVISIONS) + idx;
      my_counts[idx]++;
    }
  }

  // Prefix scan to get global offsets
  origOffsets[0] = 0;
  for (int i = 0; i < DIVISIONS; i++) {
    unsigned int total_bucket_size = 0;
    for (int t = 0; t < omp_get_max_threads(); t++) {
        unsigned int count = thread_offsets[t * DIVISIONS + i];
        thread_offsets[t * DIVISIONS + i] = total_bucket_size; 
        total_bucket_size += count;
    }
    if ((total_bucket_size % 4) != 0) {
      nullElements[i] = (total_bucket_size & ~3) + 4 - total_bucket_size;
    } else {
      nullElements[i] = 0;
    }
    sizes[i] = (total_bucket_size + nullElements[i]) / 4;
    origOffsets[i+1] = origOffsets[i] + (total_bucket_size + nullElements[i]);
  }

  // Move elements to buckets (NUMA aware via first-touch if d_output is allocated correctly)
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int *my_bucket_offsets = &thread_offsets[tid * DIVISIONS];
    #pragma omp for
    for (int i = 0; i < listsize; i++) {
      int id = d_indice[i];
      int bucket_idx = id & (DIVISIONS - 1);
      int offset_in_bucket = id >> LOG_DIVISIONS;
      d_output[origOffsets[bucket_idx] + my_bucket_offsets[bucket_idx] + offset_in_bucket] = d_input[i];
    }
  }

  _mm_free(pivotPoints);
  _mm_free(d_indice);
  _mm_free(thread_offsets);
}
