/* benchmark_Claude_optimized_init.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

SimulationData initialize_simulation(Input input)
{
    uint64_t seed = INITIALIZATION_SEED;

    printf("Loading Hoogenboom-Martin material data...\n");
    SimulationData SD = get_materials(input, &seed);

    printf("Generating resonance distributions...\n");
    SD.n_poles = generate_n_poles(input, &seed);
    SD.length_n_poles = input.n_nuclides;

    printf("Generating window distributions...\n");
    SD.n_windows = generate_n_windows(input, &seed);
    SD.length_n_windows = input.n_nuclides;

    printf("Generating resonance parameter grid...\n");
    SD.poles = generate_poles(input, SD.n_poles, &seed, &SD.max_num_poles);
    SD.length_poles = input.n_nuclides * SD.max_num_poles;

    printf("Generating window parameter grid...\n");
    SD.windows = generate_window_params(input, SD.n_windows, SD.n_poles, &seed, &SD.max_num_windows);
    SD.length_windows = input.n_nuclides * SD.max_num_windows;

    printf("Generating 0K l_value data...\n");
    SD.pseudo_K0RS = generate_pseudo_K0RS(input, &seed);
    SD.length_pseudo_K0RS = input.n_nuclides * input.numL;

    return SD;
}

int * generate_n_poles(Input input, uint64_t * seed)
{
    int total_resonances = input.avg_n_poles * input.n_nuclides;
    int * R = (int *) malloc(input.n_nuclides * sizeof(int));

    for (int i = 0; i < input.n_nuclides; i++)
        R[i] = 1;

    /* OPT-24: Use LCG_RAND_INT macro to inline LCG step, avoiding
       function-call overhead for every sample in the initialisation loop.   */
    const int nn = input.n_nuclides;
    for (int i = 0; i < total_resonances - input.n_nuclides; i++)
        R[LCG_RAND_INT(*seed) % nn]++;

    return R;
}

int * generate_n_windows(Input input, uint64_t * seed)
{
    int total_resonances = input.avg_n_windows * input.n_nuclides;
    int * R = (int *) malloc(input.n_nuclides * sizeof(int));

    for (int i = 0; i < input.n_nuclides; i++)
        R[i] = 1;

    const int nn = input.n_nuclides;
    for (int i = 0; i < total_resonances - input.n_nuclides; i++)
        R[LCG_RAND_INT(*seed) % nn]++;

    return R;
}

Pole * generate_poles(Input input, int * n_poles, uint64_t * seed, int * max_num_poles)
{
    double f = 152.5;
    RSComplex f_c = {f, 0};

    int max_poles = -1;
    for (int i = 0; i < input.n_nuclides; i++)
        if (n_poles[i] > max_poles) max_poles = n_poles[i];
    *max_num_poles = max_poles;

    Pole * R = (Pole *) malloc(input.n_nuclides * max_poles * sizeof(Pole));

    for (int i = 0; i < input.n_nuclides; i++)
    {
        /* OPT-25: Cache row base index to avoid recomputing i*max_poles
           on every inner iteration at -O0.                                  */
        const int row = i * max_poles;
        for (int j = 0; j < n_poles[i]; j++)
        {
            Pole * p = &R[row + j];

            /* OPT-24 applied: inline LCG via macros                        */
            double r  = LCG_RAND_DOUBLE(*seed);
            double im = LCG_RAND_DOUBLE(*seed);
            RSComplex t1 = {r, im};
            p->MP_EA = c_mul(f_c, t1);

            r  = LCG_RAND_DOUBLE(*seed);
            im = LCG_RAND_DOUBLE(*seed);
            RSComplex t2 = {f*r, im};
            p->MP_RT = t2;

            r  = LCG_RAND_DOUBLE(*seed);
            im = LCG_RAND_DOUBLE(*seed);
            RSComplex t3 = {f*r, im};
            p->MP_RA = t3;

            r  = LCG_RAND_DOUBLE(*seed);
            im = LCG_RAND_DOUBLE(*seed);
            RSComplex t4 = {f*r, im};
            p->MP_RF = t4;

            p->l_value = (short int)(LCG_RAND_INT(*seed) % input.numL);
        }
    }

    return R;
}

Window * generate_window_params(Input input, int * n_windows, int * n_poles,
                                uint64_t * seed, int * max_num_windows)
{
    int max_windows = -1;
    for (int i = 0; i < input.n_nuclides; i++)
        if (n_windows[i] > max_windows) max_windows = n_windows[i];
    *max_num_windows = max_windows;

    Window * R = (Window *) malloc(input.n_nuclides * max_windows * sizeof(Window));

    for (int i = 0; i < input.n_nuclides; i++)
    {
        /* OPT-25 applied: cache row base                                    */
        const int row = i * max_windows;
        int space     = n_poles[i] / n_windows[i];
        int remainder = n_poles[i] - space * n_windows[i];
        int ctr       = 0;

        for (int j = 0; j < n_windows[i]; j++)
        {
            Window * w = &R[row + j];
            w->T     = LCG_RAND_DOUBLE(*seed);
            w->A     = LCG_RAND_DOUBLE(*seed);
            w->F     = LCG_RAND_DOUBLE(*seed);
            w->start = ctr;
            w->end   = ctr + space - 1;
            ctr     += space;
            if (j < remainder) { ctr++; w->end++; }
        }
    }

    return R;
}

double * generate_pseudo_K0RS(Input input, uint64_t * seed)
{
    double * R = (double *) malloc(input.n_nuclides * input.numL * sizeof(double));
    /* OPT-26: Flatten the 2D loop into a single linear loop.
       At -O0 a nested loop has double the loop-control overhead
       (two CMP+branch+inc sequences).  The 1D version has one.             */
    const int total = input.n_nuclides * input.numL;
    for (int k = 0; k < total; k++)
        R[k] = LCG_RAND_DOUBLE(*seed);

    return R;
}
