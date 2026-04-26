/* benchmark_Claude_optimized_utils.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

size_t get_mem_estimate(Input input)
{
    /* OPT-27: Cache repeated sub-expressions into locals.
       At -O0 each use of input.n_nuclides is a struct field load.
       Storing it once avoids 6 redundant loads.                            */
    const int nn   = input.n_nuclides;
    const int numL = input.numL;

    size_t poles      = (size_t)nn * input.avg_n_poles   * sizeof(Pole)   + (size_t)nn * sizeof(Pole *);
    size_t windows    = (size_t)nn * input.avg_n_windows  * sizeof(Window) + (size_t)nn * sizeof(Window *);
    size_t pseudo_K0RS = (size_t)nn * numL                 * sizeof(double) + (size_t)nn * sizeof(double);
    size_t other      = (size_t)nn * 2                     * sizeof(int);

    return poles + windows + pseudo_K0RS + other;
}

double get_time(void)
{
#ifdef OPENMP
    return omp_get_wtime();
#endif
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    long ms = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
    return (double)ms / 1000.0;
}
