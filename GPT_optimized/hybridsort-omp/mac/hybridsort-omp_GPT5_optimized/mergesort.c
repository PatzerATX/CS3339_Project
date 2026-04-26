#include <stdlib.h>
#include "mergesort.h"

static int compare_float_values(const void *a, const void *b)
{
  float fa = *((const float *)a);
  float fb = *((const float *)b);

  if (fa < fb) return -1;
  if (fa > fb) return 1;
  return 0;
}

float4 sortElem(float4 r)
{
  float values[4];
  values[0] = r.x;
  values[1] = r.y;
  values[2] = r.z;
  values[3] = r.w;

  for (int i = 1; i < 4; i++) {
    float key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = key;
  }

  r.x = values[0];
  r.y = values[1];
  r.z = values[2];
  r.w = values[3];
  return r;
}

float4 getLowest(float4 a, float4 b)
{
  float values[8];
  values[0] = a.x;
  values[1] = a.y;
  values[2] = a.z;
  values[3] = a.w;
  values[4] = b.x;
  values[5] = b.y;
  values[6] = b.z;
  values[7] = b.w;

  for (int i = 1; i < 8; i++) {
    float key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = key;
  }

  a.x = values[0];
  a.y = values[1];
  a.z = values[2];
  a.w = values[3];
  return a;
}

float4 getHighest(float4 a, float4 b)
{
  float values[8];
  values[0] = a.x;
  values[1] = a.y;
  values[2] = a.z;
  values[3] = a.w;
  values[4] = b.x;
  values[5] = b.y;
  values[6] = b.z;
  values[7] = b.w;

  for (int i = 1; i < 8; i++) {
    float key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      j--;
    }
    values[j + 1] = key;
  }

  b.x = values[4];
  b.y = values[5];
  b.z = values[6];
  b.w = values[7];
  return b;
}

float4* runMergeSort(int listsize, int divisions,
    float4 *d_origList, float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets)
{
  (void)divisions;
  (void)d_resultList;
  (void)sizes;
  (void)nullElements;
  (void)origOffsets;

  qsort((float *)d_origList, (size_t)listsize, sizeof(float), compare_float_values);
  return d_origList;
}
