/* benchmark_Claude_optimized_kernel_bucketsort.h
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

        /* OPT-28: Cache all derived constants into locals to eliminate
           repeated expression evaluation at -O0.                           */
        const int prefixBase  = tid * BUCKET_BLOCK_MEMORY;
        const int warpBase    = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
        const int numThreads  = gsize;

        /* OPT-29: Fuse the s_offset initialisation with the h_offsets +
           d_prefixoffsets addition.  Original: separate loop only writing
           s_offset.  Here: combine the load of h_offsets and d_prefixoffsets
           with a single assignment loop — one pass over BUCKET_BLOCK_MEMORY
           rather than two, keeping s_offset[] hot in the 32 KB L1 cache.  */
        for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize) {
            /* OPT-30: Cache the DIVISIONS_MASK derived macro for the AND  */
            s_offset[i] = h_offsets[i & DIVISIONS_MASK]
                        + d_prefixoffsets[prefixBase + i];
        }

#pragma omp barrier

        for (int work_id = gid; work_id < listsize; work_id += numThreads) {
            float        elem = d_input[work_id];
            int          id   = d_indice[work_id];

            /* OPT-31: Cache the destination index computation into a local.
               At -O0 warpBase + (id & DIVISIONS_MASK) would be recomputed
               for both the s_offset lookup and the output store.           */
            const int bucket  = id & DIVISIONS_MASK;
            const int slot    = id >> LOG_DIVISIONS;
            d_output[s_offset[warpBase + bucket] + slot] = elem;
        }
    }
}
