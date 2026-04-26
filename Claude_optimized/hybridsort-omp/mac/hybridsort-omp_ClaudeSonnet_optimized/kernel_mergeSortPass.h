#pragma omp target teams distribute parallel for num_teams(grid[0]) thread_limit(local[0])
for (int gid = 0; gid < global[0]; gid++) {
  int division = gid / threadsPerDiv;
  if (division < DIVISIONS) {
    int int_gid = gid - division * threadsPerDiv;
    int Astart  = startaddr[division] + int_gid * nrElems;
    int Bstart  = Astart + nrElems / 2;
    float4 *resStart = &(d_resultList[Astart]);

    /* Hoist division boundaries into locals: avoids repeated
       pointer+index loads at -O0 inside the hot inner loop. */
    const int divStart = startaddr[division];
    const int divEnd   = startaddr[division + 1];
    /* Precompute listsize/4 once instead of per-iteration */
    const int listsize_4 = listsize / 4;
    /* Precompute half of nrElems for bound checks */
    const int halfElems = nrElems / 2;

    if (Astart < divEnd) {
      if (Bstart >= divEnd) {
        int copyLen = divEnd - Astart;
        for (int i = 0; i < copyLen; i++)
          resStart[i] = d_origList[Astart + i];
      } else {
        int aidx   = 0;
        int bidx   = 0;
        int outidx = 0;
        float4 a, b;
        float4 zero = {0.f, 0.f, 0.f, 0.f};
        a = d_origList[Astart];
        b = d_origList[Bstart];

        while (1) {
          int nextA_idx = Astart + aidx + 1;
          int nextB_idx = Bstart + bidx + 1;
          float4 nextA = d_origList[nextA_idx];
          float4 nextB = (nextB_idx >= listsize_4) ? zero : d_origList[nextB_idx];

          float4 na = getLowest(a, b);
          float4 nb = getHighest(a, b);
          a = sortElem(na);
          b = sortElem(nb);
          resStart[outidx++] = a;

          /* Store booleans in ints: avoids bool->int conversion overhead at -O0 */
          int elemsLeftInA = (aidx + 1 < halfElems);
          int elemsLeftInB = ((bidx + 1 < halfElems) & (nextB_idx < divEnd));

          if (elemsLeftInA) {
            if (elemsLeftInB) {
              if (nextA.x < nextB.x) { aidx++; a = nextA; }
              else                   { bidx++; a = nextB; }
            } else {
              aidx++; a = nextA;
            }
          } else {
            if (elemsLeftInB) { bidx++; a = nextB; }
            else              { break; }
          }
        }
        resStart[outidx] = b;
      }
    }
  }
}
