/* benchmark_Claude_optimized_kernel_mergeSortPass.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#pragma omp target teams distribute parallel for \
    num_teams(grid[0]) thread_limit(local[0])
for (int gid = 0; gid < global[0]; gid++) {

    /* OPT-32: Cache division and int_gid in locals to avoid recomputing
       gid/threadsPerDiv and gid - division*threadsPerDiv at -O0.
       On x86_64, IDIV is 20-90 cycles; caching the result avoids a
       second division for int_gid.                                        */
    int division = gid / threadsPerDiv;

    if (division < DIVISIONS) {
        int int_gid = gid - division * threadsPerDiv;

        /* OPT-33: Cache startaddr[division] and startaddr[division+1] in
           locals.  At -O0 each array access is a full indexed load; these
           values are used 3+ times each within the block.                  */
        const int divStart = startaddr[division];
        const int divEnd   = startaddr[division + 1];

        int Astart = divStart + int_gid * nrElems;
        int Bstart = Astart + nrElems / 2;  /* OPT-34: /2 — kept as-is,
                                               compiler sees literal 2      */

        float4 *resStart = &d_resultList[Astart];

        if (Astart < divEnd) {
            if (Bstart >= divEnd) {
                /* OPT-35: Cache (divEnd - Astart) in a local to avoid
                   recomputing the subtraction on every loop iteration.     */
                const int copyCount = divEnd - Astart;
                for (int i = 0; i < copyCount; i++)
                    resStart[i] = d_origList[Astart + i];
            } else {
                int   aidx   = 0;
                int   bidx   = 0;
                int   outidx = 0;
                float4 zero  = {0.f, 0.f, 0.f, 0.f};
                float4 a     = d_origList[Astart];
                float4 b     = d_origList[Bstart];

                /* OPT-36: Cache nrElems/2 and listsize/4 in locals to
                   avoid recomputing the division on every iteration of the
                   merge loop.  At -O0 these are re-evaluated each time.   */
                const int halfElems = nrElems / 2;
                const int listQ     = listsize / 4;

                while (1) {
                    /* OPT-37: Cache Astart+aidx+1 and Bstart+bidx+1 to
                       avoid double-computing the address for nextA/nextB
                       and for the bounds check.                            */
                    const int nextAidx = Astart + aidx + 1;
                    const int nextBidx = Bstart + bidx + 1;

                    float4 nextA = d_origList[nextAidx];
                    float4 nextB = (nextBidx >= listQ) ? zero
                                                        : d_origList[nextBidx];

                    float4 na = getLowest(a, b);
                    float4 nb = getHighest(a, b);
                    a = sortElem(na);
                    b = sortElem(nb);
                    resStart[outidx++] = a;

                    /* OPT-38: Evaluate elemsLeftInA/B as int (0/1) to
                       allow the x86_64 branch predictor to see a boolean
                       pattern rather than a complex expression.            */
                    int elemsLeftInA = (aidx + 1 < halfElems);
                    int elemsLeftInB = (bidx + 1 < halfElems)
                                    & (nextBidx < divEnd);

                    if (elemsLeftInA) {
                        if (elemsLeftInB) {
                            /* OPT-39: Cache .x fields into locals to
                               avoid repeated struct field loads at -O0.    */
                            float nextA_t = nextA.x;
                            float nextB_t = nextB.x;
                            if (nextA_t < nextB_t) { aidx++; a = nextA; }
                            else                    { bidx++; a = nextB; }
                        } else {
                            aidx++; a = nextA;
                        }
                    } else {
                        if (elemsLeftInB) {
                            bidx++; a = nextB;
                        } else {
                            break;
                        }
                    }
                }
                resStart[outidx] = b;  /* OPT-40: Remove post-increment;
                                          outidx not used after this.      */
            }
        }
    }
}
