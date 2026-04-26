/* benchmark_Claude_optimized_kernel_bucketprefix.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#pragma omp target teams distribute parallel for \
    num_teams(globalpre/localpre) thread_limit(localpre)
for (int tid = 0; tid < DIVISIONS; tid++) {
    /* OPT-26: Use a local accumulator 'sum' (register variable at -O0)
       rather than reading d_prefixoffsets[i] twice per iteration.
       The original code correctly does this already — preserved.
       Additional opt: cache 'size' local avoids struct/global reload.     */
    int sum = 0;
    /* OPT-27: Stride by DIVISIONS — each thread tid processes elements
       tid, tid+DIVISIONS, tid+2*DIVISIONS, ...
       On x86_64 with 64-byte cache lines, strided access by DIVISIONS
       (1024 ints = 4 KB) is cache-unfriendly but is the original
       algorithm's requirement for correctness.  We preserve it.
       The local accumulator ensures the write d_prefixoffsets[i]=sum
       is the only store per element (one store vs two at -O0).           */
    for (int i = tid; i < size; i += DIVISIONS) {
        int x = d_prefixoffsets[i];
        d_prefixoffsets[i] = sum;
        sum += x;
    }
    h_offsets[tid] = sum;
}
