#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <arm_neon.h>
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

  float* pivotPoints = (float *)aligned_alloc(64, DIVISIONS * sizeof(float));
  int* d_indice = (int *)aligned_alloc(64, listsize * sizeof(int));
  float historesult[histosize];

  // 1. Histogram (Parallel with thread-local storage)
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

  // 3. Bucket Count & Assignment
  memset(h_offsets, 0, sizeof(h_offsets));
  
  #pragma omp parallel
  {
    unsigned int local_counts[DIVISIONS] = {0};
    #pragma omp for nowait
    for (int i = 0; i < listsize; i++) {
      float elem = d_input[i];
      int idx = DIVISIONS / 2 - 1;
      int jump = DIVISIONS / 4;
      
      // Unrolled Binary Search
      while(jump >= 1){
        idx = (elem < pivotPoints[idx]) ? (idx - jump) : (idx + jump);
        jump /= 2;
      }
      idx = (elem < pivotPoints[idx]) ? idx : (idx + 1);
      if (idx >= DIVISIONS) idx = DIVISIONS - 1;

      d_indice[i] = (local_counts[idx] << LOG_DIVISIONS) + idx;
      local_counts[idx]++;
    }
    
    #pragma omp critical
    {
        // This part is tricky on CPU. We need to know the global offsets for each thread.
        // For simplicity and speed at -O0, let's just do a second pass or use a different approach.
    }
  }
  
  // Actually, the original OpenMP Target logic is hard to replicate exactly with local histograms 
  // without a prefix scan over threads. 
  // Let's use a simpler parallel approach for CPU:
  
  unsigned int *thread_offsets = (unsigned int*)aligned_alloc(64, omp_get_max_threads() * DIVISIONS * sizeof(unsigned int));
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
        thread_offsets[t * DIVISIONS + i] = total_bucket_size; // Re-use as thread-local start offset within bucket
        total_bucket_size += count;
    }
    sizes[i] = total_bucket_size;
    if ((total_bucket_size % 4) != 0) {
      nullElements[i] = (total_bucket_size & ~3) + 4 - total_bucket_size;
    } else {
      nullElements[i] = 0;
    }
    sizes[i] = (total_bucket_size + nullElements[i]) / 4;
    origOffsets[i+1] = origOffsets[i] + (total_bucket_size + nullElements[i]);
  }

  // Move elements to buckets
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

  free(pivotPoints);
  free(d_indice);
  free(thread_offsets);
}
