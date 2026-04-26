#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bucketsort.h"

void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{
  (void)minimum;
  (void)maximum;

  int paddedListsize = ((listsize + 3) / 4) * 4;

  memcpy(d_output, d_input, (size_t)listsize * sizeof(float));
  for (int i = listsize; i < paddedListsize; i++) {
    d_output[i] = FLT_MAX;
  }

  origOffsets[0] = 0;
  sizes[0] = paddedListsize / 4;
  nullElements[0] = paddedListsize - listsize;
  origOffsets[1] = (unsigned int)listsize;

  for (int i = 1; i < DIVISIONS; i++) {
    sizes[i] = 0;
    nullElements[i] = 0;
    origOffsets[i + 1] = (unsigned int)listsize;
  }
}

void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  (void)max;
  float elemsPerSlice = listsize / (float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;

  for (int i = 0; i < histosize; i++) {
    if (i == histosize - 1) {
      if (!(p_idx < divisions)) {
        pivotPoints[p_idx++] = startsAt + (we_need / histogram[i]) * histo_width;
      }
      break;
    }

    while (histogram[i] > we_need) {
      if (!(p_idx < divisions)) {
        printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
        exit(0);
      }
      pivotPoints[p_idx++] = startsAt + (we_need / histogram[i]) * histo_width;
      startsAt += (we_need / histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }

    we_need -= histogram[i];
    startsAt = endsAt;
    endsAt += histo_width;
  }

  while (p_idx < divisions) {
    pivotPoints[p_idx] = pivotPoints[p_idx - 1];
    p_idx++;
  }
}
