#pragma omp target teams num_teams(blocks) thread_limit(BUCKET_THREAD_N)
{
  unsigned int s_offset[BUCKET_BLOCK_MEMORY];
#pragma omp parallel
  {
    const int lid = omp_get_thread_num();
    const int lsize = omp_get_num_threads();
    const int tid = omp_get_team_num();
    const int gid = tid * lsize + lid;
    const int gsize = omp_get_num_teams() * lsize;

    int prefixBase = tid * BUCKET_BLOCK_MEMORY;
    const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = gsize;

    for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
      s_offset[i] = h_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];

#pragma omp barrier

    for (int t = gid; t < listsize; t += numThreads) {
      float elem = d_input[t];
      int id    = d_indice[t];
      /* Hoist bucket index into local var to avoid recomputing at -O0 */
      int bucket = id & (DIVISIONS - 1);
      int local_off = id >> LOG_DIVISIONS;
      d_output[s_offset[warpBase + bucket] + local_off] = elem;
    }
  }
}
