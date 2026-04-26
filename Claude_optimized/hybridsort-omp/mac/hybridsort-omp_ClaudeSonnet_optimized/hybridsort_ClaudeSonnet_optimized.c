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
#include "bucketsort.h"
#include "mergesort.h"
#include <chrono>
#define TIMER

#define SIZE (50000000)

int compare(const void *a, const void *b) {
  if (*((float *)a) < *((float *)b)) return -1;
  else if (*((float *)a) > *((float *)b)) return 1;
  else return 0;
}

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets);

int main(int argc, char** argv)
{
  int numElements = 0;

  if (strcmp(argv[1], "r") == 0) {
    numElements = SIZE;
  } else {
    FILE *fp = fopen(argv[1], "r");
    if (fp == NULL) {
      printf("Error reading file \n");
      exit(EXIT_FAILURE);
    }
    int count = 0;
    float c;
    while (fscanf(fp, "%f", &c) != EOF) count++;
    fclose(fp);
    numElements = count;
  }

  printf("Sorting list of %d floats.\n", numElements);

  /* Hoist repeated DIVISIONS*4 multiply to a named constant */
  const int pad = DIVISIONS * 4;
  int mem_size  = (numElements + pad) * sizeof(float);

  float *cpu_idata = (float *)malloc(mem_size);
  float *cpu_odata = (float *)malloc(mem_size);
  float *d_output  = (float *)malloc(mem_size);
  float *gpu_odata;
  float datamin = FLT_MAX;
  float datamax = -FLT_MAX;

  if (strcmp(argv[1], "r") == 0) {
    /* Generate data; rand() is not thread-safe so loop stays serial.
       Use local accumulators to avoid repeated global memory reads at -O0. */
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = 0; i < numElements; i++) {
      float v = (float)rand() / RAND_MAX;
      cpu_idata[i] = v;
      if (v < local_min) local_min = v;
      if (v > local_max) local_max = v;
    }
    datamin = local_min;
    datamax = local_max;
  } else {
    FILE *fp = fopen(argv[1], "r");
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    for (int i = 0; i < numElements; i++) {
      fscanf(fp, "%f", &cpu_idata[i]);
      float v = cpu_idata[i];
      if (v < local_min) local_min = v;
      if (v > local_max) local_max = v;
    }
    fclose(fp);
    datamin = local_min;
    datamax = local_max;
  }

  FILE *tp;
  const char filename2[] = "./hybridinput.txt";
  tp = fopen(filename2, "w");
  for (int i = 0; i < SIZE; i++)
    fprintf(tp, "%f ", cpu_idata[i]);
  fclose(tp);

  memcpy(cpu_odata, cpu_idata, mem_size);

  int *sizes       = (int *)malloc(DIVISIONS * sizeof(int));
  int *nullElements = (int *)malloc(DIVISIONS * sizeof(int));
  unsigned int *origOffsets = (unsigned int *)malloc((DIVISIONS + 1) * sizeof(int));

  auto bucketsort_start = std::chrono::steady_clock::now();
  bucketSort(cpu_idata, d_output, numElements, sizes, nullElements, datamin, datamax, origOffsets);
  auto bucketsort_end  = std::chrono::steady_clock::now();
  auto bucketsort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(bucketsort_end - bucketsort_start).count();

  float4 *d_origList   = (float4 *)d_output;
  float4 *d_resultList = (float4 *)cpu_idata;

  int newlistsize = 0;
  for (int i = 0; i < DIVISIONS; i++)
    newlistsize += sizes[i] * 4;

  auto mergesort_start = std::chrono::steady_clock::now();
  float4 *mergeresult = runMergeSort(newlistsize, DIVISIONS, d_origList, d_resultList,
                                     sizes, nullElements, origOffsets);
  auto mergesort_end  = std::chrono::steady_clock::now();
  auto mergesort_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(mergesort_end - mergesort_start).count();
  gpu_odata = (float *)mergeresult;

#ifdef TIMER
  float bucketsort_msec = bucketsort_diff * 1e-6f;
  float mergesort_msec  = mergesort_diff  * 1e-6f;
  printf("GPU execution time: %0.3f ms  \n", bucketsort_msec + mergesort_msec);
  printf("  --Bucketsort execution time: %0.3f ms \n", bucketsort_msec);
  printf("  --Mergesort execution time: %0.3f ms \n",  mergesort_msec);
#endif

  clock_t cpu_start = clock(), cpu_diff;
  qsort(cpu_odata, numElements, sizeof(float), compare);
  cpu_diff = clock() - cpu_start;
  float cpu_msec = cpu_diff * 1000.0f / CLOCKS_PER_SEC;
  printf("CPU execution time: %0.3f ms  \n", cpu_msec);
  printf("Checking result...");

  int count = 0;
  for (int i = 0; i < numElements; i++) {
    if (cpu_odata[i] != gpu_odata[i]) {
      printf("Sort missmatch on element %d: \n", i);
      printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i]);
      count++;
      break;
    }
  }
  if (count == 0) printf("PASSED.\n");
  else            printf("FAILED.\n");

#ifdef OUTPUT
  FILE *tp1;
  const char filename3[] = "./hybridoutput.txt";
  tp1 = fopen(filename3, "w");
  for (int i = 0; i < SIZE; i++)
    fprintf(tp1, "%f ", cpu_idata[i]);
  fclose(tp1);
#endif

  free(cpu_idata);
  free(cpu_odata);
  free(d_output);
  free(sizes);
  free(nullElements);
  free(origOffsets);

  return 0;
}
