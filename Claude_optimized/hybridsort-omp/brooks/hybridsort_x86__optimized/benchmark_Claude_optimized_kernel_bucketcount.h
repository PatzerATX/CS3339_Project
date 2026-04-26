/* benchmark_Claude_optimized_kernel_bucketcount.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

#pragma omp target teams num_teams(blocks) thread_limit(BUCKET_THREAD_N)
{
    unsigned int s_offset[BUCKET_BLOCK_MEMORY];
#pragma omp parallel
    {
        const int lid       = omp_get_thread_num();
        const int lsize     = omp_get_num_threads();
        const int tid       = omp_get_team_num();
        const int gid       = tid * lsize + lid;
        const int gsize     = omp_get_num_teams() * lsize;

        /* OPT-20: Cache warpBase — avoids re-shifting and re-multiplying
           lid >> BUCKET_WARP_LOG_SIZE on every use at -O0.                */
        const int warpBase  = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
        const int numThreads = gsize;

        for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
            s_offset[i] = 0;

#pragma omp barrier

        for (int i = gid; i < listsize; i += numThreads) {
            float elem = d_input[i];

            /* OPT-21: Use pre-computed HALF_DIVISIONS and QUARTER_DIVISIONS
               macros so the compiler sees literals rather than expressions
               at -O0.                                                      */
            int idx  = HALF_DIVISIONS;
            int jump = QUARTER_DIVISIONS;

            /* OPT-22: Cache pivotPoints[idx] into a local 'piv' to avoid
               repeated global memory loads in the binary search loop.
               On x86_64 NUMA, global memory loads cross NUMA nodes ~50%
               of the time; caching in a local reduces that traffic.       */
            float piv = pivotPoints[idx];

            while (jump >= 1) {
                idx  = (elem < piv) ? (idx - jump) : (idx + jump);
                piv  = pivotPoints[idx];
                jump >>= 1;  /* OPT-23: jump/2 → jump>>1 (SHR vs IDIV)   */
            }
            idx = (elem < piv) ? idx : (idx + 1);

            int offset;
#pragma omp atomic capture
            offset = s_offset[warpBase + idx]++;

            /* OPT-24: Replace (offset << LOG_DIVISIONS) + idx with a
               single expression. At -O0 these are separate instructions;
               the compiler cannot fold them. Naming makes intent clear
               and keeps both operands in registers.                        */
            d_indice[i] = (offset << LOG_DIVISIONS) | idx;  /* OR safe: idx < DIVISIONS */
        }

#pragma omp barrier

        /* OPT-25: Cache prefixBase to avoid recomputing tid*BUCKET_BLOCK_MEMORY
           on every store in the writeback loop.                            */
        const int prefixBase = tid * BUCKET_BLOCK_MEMORY;
        for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
            d_prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;
    }
}
