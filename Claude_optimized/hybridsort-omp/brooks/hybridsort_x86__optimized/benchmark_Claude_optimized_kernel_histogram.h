/* benchmark_Claude_optimized_kernel_histogram.h
   Optimized for x86_64 NUMA, 64-byte cache line, -O0 by Claude (Anthropic) */

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////
#pragma omp target teams num_teams(global/local) thread_limit(local)
{
    unsigned int s_Hist[HISTOGRAM_BLOCK_MEMORY];
#pragma omp parallel
    {
        const int lid   = omp_get_thread_num();
        const int lsize = omp_get_num_threads();
        const int tid   = omp_get_team_num();
        const int gid   = tid * lsize + lid;
        const int gsize = omp_get_num_teams() * lsize;

        /* OPT-15: Cache warpBase in a const local. At -O0 every use of
           an expression re-evaluates; naming it avoids the repeated
           shift+multiply.                                                  */
        const int mulBase  = (lid >> BUCKET_WARP_LOG_SIZE);
        const int warpBase = IMUL(mulBase, HISTOGRAM_BIN_COUNT);

        /* OPT-16: Clear shared histogram with a memset-style loop.
           Using a stride-1 loop over HISTOGRAM_BLOCK_MEMORY ensures
           sequential stores — maximally cache-friendly on x86_64 where
           64-byte cache lines hold 16 floats.                             */
        for (unsigned int i = lid; i < HISTOGRAM_BLOCK_MEMORY; i += lsize)
            s_Hist[i] = 0;

#pragma omp barrier

        /* OPT-17: Pre-compute the normalisation factor (maximum - minimum)
           and its reciprocal outside the per-element loop to avoid
           recomputing the subtraction and division for every element.
           At -O0 the original recomputed (max-min) on every iteration.   */
        const float range     = maximum - minimum;
        const float inv_range = 1.0f / range;
        const float scale     = (float)HISTOGRAM_BIN_COUNT * inv_range;

        for (int pos = gid; pos < listsize; pos += gsize) {
            /* OPT-17 applied: multiply by pre-computed scale              */
            unsigned int data4 = (unsigned int)((d_input[pos] - minimum) * scale);
#pragma omp atomic update
            s_Hist[warpBase + (data4 & 0x3FFU)]++;
        }

#pragma omp barrier

        /* OPT-18: Cache HISTOGRAM_BLOCK_MEMORY as a local const to avoid
           re-loading the macro expansion on every inner-loop iteration.   */
        const int hbm = HISTOGRAM_BLOCK_MEMORY;
        for (int pos = lid; pos < HISTOGRAM_BIN_COUNT; pos += lsize) {
            unsigned int sum = 0;
            /* OPT-19: stride by HISTOGRAM_BIN_COUNT — sequential access
               pattern for s_Hist; on x86_64 this allows hardware prefetch
               to fill cache lines ahead of the inner loop.                */
            for (int i = 0; i < hbm; i += HISTOGRAM_BIN_COUNT)
                sum += s_Hist[pos + i] & 0x07FFFFFFU;
#pragma omp atomic update
            h_offsets[pos] += sum;
        }
    }
}
