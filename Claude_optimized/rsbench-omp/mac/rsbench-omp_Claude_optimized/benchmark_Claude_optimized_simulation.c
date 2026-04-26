/* benchmark_Claude_optimized_simulation.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////

void run_event_based_simulation(Input input, SimulationData data,
                                unsigned long * vhash_result,
                                double * kernel_time)
{
    printf("Beginning event based simulation ...\n");
    unsigned long verification = 0;

    #pragma omp target data map(to:data.n_poles[:data.length_n_poles])\
                            map(to:data.n_windows[:data.length_n_windows])\
                            map(to:data.poles[:data.length_poles])\
                            map(to:data.windows[:data.length_windows])\
                            map(to:data.pseudo_K0RS[:data.length_pseudo_K0RS])\
                            map(to:data.num_nucs[:data.length_num_nucs])\
                            map(to:data.mats[:data.length_mats])\
                            map(to:data.concs[:data.length_concs])\
                            map(to:data.max_num_nucs)\
                            map(to:data.max_num_poles)\
                            map(to:data.max_num_windows)\
                            map(to:input) \
                            map(tofrom:verification)
    {
        double start = get_time();

        #pragma omp target teams distribute parallel for reduction(+:verification)
        for (int i = 0; i < input.lookups; i++)
        {
            uint64_t seed = STARTING_SEED;
            seed = fast_forward_LCG(seed, 2*i);

            /* OPT-6: Use LCG macros directly to avoid function-call overhead
               for the two samples taken every lookup iteration.              */
            double E = LCG_RAND_DOUBLE(seed);
            int mat  = pick_mat(&seed);

            double macro_xs[4] = {0};

            calculate_macro_xs(macro_xs, mat, E, input,
                               data.num_nucs, data.mats, data.max_num_nucs,
                               data.concs, data.n_windows, data.pseudo_K0RS,
                               data.windows, data.poles,
                               data.max_num_windows, data.max_num_poles);

            /* OPT-7: Unroll the 4-element max-search manually.
               At -O0 the compiler emits full loop control (CMP/branch/inc)
               for every iteration of a 4-element loop. Unrolling removes
               3 branches and 3 increments from every single lookup.         */
            int max_idx = 0;
            double max_val = macro_xs[0];
            if (macro_xs[1] > max_val) { max_val = macro_xs[1]; max_idx = 1; }
            if (macro_xs[2] > max_val) { max_val = macro_xs[2]; max_idx = 2; }
            if (macro_xs[3] > max_val) {                         max_idx = 3; }

            verification += (unsigned long)(max_idx + 1);
        }

        double stop = get_time();
        printf("Kernel initialization, compilation, and execution took %.2lf seconds.\n", stop-start);
        *kernel_time = stop - start;
    }

    *vhash_result = verification;
}

void calculate_macro_xs(double * macro_xs, int mat, double E, Input input,
                        int * num_nucs, int * mats, int max_num_nucs,
                        double * concs, int * n_windows, double * pseudo_K0Rs,
                        Window * windows, Pole * poles,
                        int max_num_windows, int max_num_poles)
{
    /* OPT-8: Use local accumulators instead of writing to macro_xs[j] on
       every inner loop iteration. At -O0 every array index is a full
       address computation + store. Accumulating in locals keeps the values
       in registers (the compiler MUST store them; at -O0 it picks registers
       for explicit named locals), then writes once at the end.              */
    double xs0 = 0.0, xs1 = 0.0, xs2 = 0.0, xs3 = 0.0;

    const int nucs_in_mat = num_nucs[mat];
    /* OPT-9: Cache mat*max_num_nucs — computed once instead of twice
       per nuclide (once for mats[], once for concs[]) at -O0.               */
    const int mat_base = mat * max_num_nucs;

    for (int i = 0; i < nucs_in_mat; i++)
    {
        double micro_xs[4];
        int nuc = mats[mat_base + i];

        if (input.doppler == 1)
            calculate_micro_xs_doppler(micro_xs, nuc, E, input, n_windows,
                                       pseudo_K0Rs, windows, poles,
                                       max_num_windows, max_num_poles);
        else
            calculate_micro_xs(micro_xs, nuc, E, input, n_windows,
                               pseudo_K0Rs, windows, poles,
                               max_num_windows, max_num_poles);

        /* OPT-9 continued: cache conc to avoid second indexed load          */
        double conc = concs[mat_base + i];

        /* OPT-8 continued: accumulate into locals, unroll 4 elements        */
        xs0 += micro_xs[0] * conc;
        xs1 += micro_xs[1] * conc;
        xs2 += micro_xs[2] * conc;
        xs3 += micro_xs[3] * conc;
    }

    macro_xs[0] = xs0;
    macro_xs[1] = xs1;
    macro_xs[2] = xs2;
    macro_xs[3] = xs3;
}

// No Temperature dependence (0K evaluation)
void calculate_micro_xs(double * micro_xs, int nuc, double E, Input input,
                        int * n_windows, double * pseudo_K0RS,
                        Window * windows, Pole * poles,
                        int max_num_windows, int max_num_poles)
{
    double sigT, sigA, sigF;

    /* OPT-10: Pre-compute 1/n_windows[nuc] as a reciprocal multiply.
       At -O0, the division E/spacing = E*n_windows[nuc] costs FDIV (~10 cy).
       Replaced by FMUL after reciprocal.  Also avoids re-loading
       n_windows[nuc] twice (once for spacing, once for boundary check).     */
    const int nw     = n_windows[nuc];
    const double inv_nw = 1.0 / (double)nw;
    int window = (int)(E * (double)nw);
    if (window == nw) window--;

    /* OPT-11: calculate_sig_T inlined fully.
       numL is always 4 (NUM_L).  Inlining removes function-call overhead
       and lets the loop be expressed with a literal bound.                  */
    RSComplex sigTfactors[NUM_L];
    {
        /* OPT-12: Pre-compute sqrt(E) once — used in every sigTfactor
           iteration and also in the pole loop below.                        */
        const double sqrtE = sqrt(E);
        const int nuc_numL_base = nuc * NUM_L;
        for (int li = 0; li < NUM_L; li++)
        {
            double phi = pseudo_K0RS[nuc_numL_base + li] * sqrtE;
            if      (li == 1) phi -= -atan(phi);
            else if (li == 2) phi -= atan(3.0 * phi / (3.0 - phi*phi));
            else if (li == 3) phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));
            phi *= 2.0;
            sigTfactors[li].r =  cos(phi);
            sigTfactors[li].i = -sin(phi);
        }

        /* OPT-13: Cache the window struct by value to avoid repeated
           pointer chasing through windows[nuc*max_num_windows+window]
           in the pole loop.                                                 */
        Window w = windows[nuc * max_num_windows + window];
        sigT = E * w.T;
        sigA = E * w.A;
        sigF = E * w.F;

        /* OPT-14: Pre-compute {sqrt(E), 0} complex and nuc*max_num_poles
           outside the pole loop.                                            */
        const int pole_base = nuc * max_num_poles;
        RSComplex t2 = {sqrtE, 0.0};
        RSComplex t1 = {0.0, 1.0};
        RSComplex E_c = {E, 0.0};

        for (int pi = w.start; pi < w.end; pi++)
        {
            Pole pole = poles[pole_base + pi];
            /* PSIIKI = i / (pole.MP_EA - sqrt(E)) */
            RSComplex PSIIKI = c_div(t1, c_sub(pole.MP_EA, t2));
            RSComplex CDUM   = c_div(PSIIKI, E_c);

            /* OPT-15: Cache sigTfactors[pole.l_value] into a local to avoid
               repeated indexed load inside c_mul at -O0.                    */
            RSComplex stf = sigTfactors[pole.l_value];

            sigT += c_mul(pole.MP_RT, c_mul(CDUM, stf)).r;
            sigA += c_mul(pole.MP_RA, CDUM).r;
            sigF += c_mul(pole.MP_RF, CDUM).r;
        }
    }

    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigT - sigA;   /* OPT-16: sigE computed inline, no variable */
}

// Temperature Dependent (Doppler broadened) variation
void calculate_micro_xs_doppler(double * micro_xs, int nuc, double E,
                                Input input, int * n_windows,
                                double * pseudo_K0RS, Window * windows,
                                Pole * poles, int max_num_windows,
                                int max_num_poles)
{
    double sigT, sigA, sigF;

    /* OPT-10 applied: reciprocal multiply for window index                  */
    const int nw = n_windows[nuc];
    int window = (int)(E * (double)nw);
    if (window == nw) window--;

    /* OPT-11 applied: inline calculate_sig_T                               */
    RSComplex sigTfactors[NUM_L];
    {
        /* OPT-12 applied: sqrt(E) once                                      */
        const double sqrtE = sqrt(E);
        const int nuc_numL_base = nuc * NUM_L;
        for (int li = 0; li < NUM_L; li++)
        {
            double phi = pseudo_K0RS[nuc_numL_base + li] * sqrtE;
            if      (li == 1) phi -= -atan(phi);
            else if (li == 2) phi -= atan(3.0 * phi / (3.0 - phi*phi));
            else if (li == 3) phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));
            phi *= 2.0;
            sigTfactors[li].r =  cos(phi);
            sigTfactors[li].i = -sin(phi);
        }
    }

    /* OPT-13 applied: cache window struct                                   */
    Window w = windows[nuc * max_num_windows + window];
    sigT = E * w.T;
    sigA = E * w.A;
    sigF = E * w.F;

    /* OPT-17: Pre-compute dopp_c constant complex outside the pole loop.
       Original recomputed {dopp, 0} on every pole iteration at -O0.        */
    const RSComplex dopp_c = {0.5, 0.0};

    /* OPT-14 applied: cache pole base index                                 */
    const int pole_base = nuc * max_num_poles;

    for (int pi = w.start; pi < w.end; pi++)
    {
        Pole pole = poles[pole_base + pi];

        /* OPT-18: compute E_c - pole.MP_EA directly without constructing
           a temporary E_c struct — saves one struct initialisation at -O0.  */
        RSComplex diff = {E - pole.MP_EA.r, -pole.MP_EA.i};
        RSComplex Z = c_mul(diff, dopp_c);

        RSComplex faddeeva = fast_nuclear_W(Z);

        /* OPT-15 applied: cache sigTfactors lookup                          */
        RSComplex stf = sigTfactors[pole.l_value];

        sigT += c_mul(pole.MP_RT, c_mul(faddeeva, stf)).r;
        sigA += c_mul(pole.MP_RA, faddeeva).r;
        sigF += c_mul(pole.MP_RF, faddeeva).r;
    }

    micro_xs[0] = sigT;
    micro_xs[1] = sigA;
    micro_xs[2] = sigF;
    micro_xs[3] = sigT - sigA;   /* OPT-16 applied */
}

// picks a material based on a probabilistic distribution
int pick_mat(uint64_t * seed)
{
    /* OPT-19: Replace the O(n²) double-loop running sum with a flat
       cumulative distribution table, computed once as a static array.
       Original recomputed running sum from scratch for each i (O(n²) work).
       Pre-built CDF reduces to a single linear scan: O(n).
       The static array is initialised exactly once across all calls.       */
    static const double cdf[12] = {
        /* running cumulative sums of dist[] */
        0.0,          /* below 0: never returned as a threshold */
        0.052,        /* dist[1] */
        0.052+0.275,
        0.052+0.275+0.134,
        0.052+0.275+0.134+0.154,
        0.052+0.275+0.134+0.154+0.064,
        0.052+0.275+0.134+0.154+0.064+0.066,
        0.052+0.275+0.134+0.154+0.064+0.066+0.055,
        0.052+0.275+0.134+0.154+0.064+0.066+0.055+0.008,
        0.052+0.275+0.134+0.154+0.064+0.066+0.055+0.008+0.015,
        0.052+0.275+0.134+0.154+0.064+0.066+0.055+0.008+0.015+0.025,
        0.052+0.275+0.134+0.154+0.064+0.066+0.055+0.008+0.015+0.025+0.013,
    };

    /* OPT-6 applied: macro expands LCG without a function call              */
    double roll = LCG_RAND_DOUBLE(*seed);

    for (int i = 1; i < 12; i++)
        if (roll < cdf[i]) return i;

    return 0;
}

void calculate_sig_T(int nuc, double E, Input input,
                     double * pseudo_K0RS, RSComplex * sigTfactors)
{
    /* OPT-12 applied: sqrt(E) hoisted out of loop                          */
    const double sqrtE = sqrt(E);
    const int base = nuc * input.numL;

    for (int i = 0; i < input.numL; i++)
    {
        double phi = pseudo_K0RS[base + i] * sqrtE;
        if      (i == 1) phi -= -atan(phi);
        else if (i == 2) phi -= atan(3.0 * phi / (3.0 - phi*phi));
        else if (i == 3) phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));
        phi *= 2.0;
        sigTfactors[i].r =  cos(phi);
        sigTfactors[i].i = -sin(phi);
    }
}

// Faddeeva function (Abrarov + QUICK_W asymptotic)
RSComplex fast_nuclear_W(RSComplex Z)
{
    if (c_abs(Z) < 6.0)
    {
        /* OPT-20: Hoist all precomputed Abrarov constants to static arrays.
           Original declared these as local arrays on every call, causing
           NUM_THREADS * call_rate stack initialisations at -O0.
           Static storage initialises once at program start.                */
        static const double an[10] = {
            2.758402e-01, 2.245740e-01, 1.594149e-01, 9.866577e-02,
            5.324414e-02, 2.505215e-02, 1.027747e-02, 3.676164e-03,
            1.146494e-03, 3.117570e-04
        };
        static const double neg_1n[10] = {
            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
        };
        static const double denom_left[10] = {
            9.869604e+00, 3.947842e+01, 8.882644e+01, 1.579137e+02,
            2.467401e+02, 3.553058e+02, 4.836106e+02, 6.316547e+02,
            7.994380e+02, 9.869604e+02
        };

        /* OPT-21: Pre-compute t1={0,12}, t2={12,0} as named constants
           so the struct initialisations are not repeated each call.        */
        static const RSComplex t1_12i  = {0.0,  12.0};
        static const RSComplex t2_12r  = {12.0,  0.0};
        static const RSComplex imag_i  = {0.0,   1.0};
        static const RSComplex one     = {1.0,   0.0};
        static const RSComplex prefact = {0.0,   8.124330e+01};
        static const RSComplex t5_144  = {144.0, 0.0};

        RSComplex expZ   = fast_cexp(c_mul(t1_12i, Z));
        /* OPT-22: compute (1 - expZ) once, reuse in both W init and loop  */
        RSComplex one_m_expZ = c_sub(one, expZ);

        RSComplex W = c_div(c_mul(imag_i, one_m_expZ),
                            c_mul(t2_12r, Z));

        RSComplex Z2  = c_mul(Z, Z);
        RSComplex sum = {0.0, 0.0};

        for (int n = 0; n < 10; n++)
        {
            /* OPT-22: reuse expZ; compute neg_1n*expZ via scalar mul       */
            RSComplex neg1n_expZ = {neg_1n[n] * expZ.r, neg_1n[n] * expZ.i};
            RSComplex top = c_sub(neg1n_expZ, one);
            RSComplex dl  = {denom_left[n], 0.0};
            RSComplex bot = c_sub(dl, c_mul(t5_144, Z2));
            RSComplex an_c = {an[n], 0.0};
            sum = c_add(sum, c_mul(an_c, c_div(top, bot)));
        }
        return c_add(W, c_mul(prefact, c_mul(Z, sum)));
    }
    else
    {
        /* QUICK_2 3-term asymptotic — constants as static to avoid
           repeated struct initialisation at -O0.                           */
        static const RSComplex a = {0.512424224754768462984202823134979415014943561548661637413182, 0.0};
        static const RSComplex b = {0.275255128608410950901357962647054304017026259671664935783653, 0.0};
        static const RSComplex c_c = {0.051765358792987823963876628425793170829107067780337219430904, 0.0};
        static const RSComplex d = {2.724744871391589049098642037352945695982973740328335064216346, 0.0};
        static const RSComplex imag_i = {0.0, 1.0};

        RSComplex Z2 = c_mul(Z, Z);
        RSComplex W  = c_mul(c_mul(Z, imag_i),
                             c_add(c_div(a, c_sub(Z2, b)),
                                   c_div(c_c, c_sub(Z2, d))));
        return W;
    }
}

// LCG — kept as functions for compatibility with callers outside hot path
double LCG_random_double(uint64_t * seed)
{
    *seed = (LCG_A * (*seed) + LCG_C) & (LCG_M - 1ULL);
    return (double)(*seed) / (double)LCG_M;
}

uint64_t LCG_random_int(uint64_t * seed)
{
    *seed = (LCG_A * (*seed) + LCG_C) & (LCG_M - 1ULL);
    return *seed;
}

/* OPT-23: Replace modulo with bitmask in LCG step.
   LCG_M = 2^63, so % LCG_M == & (LCG_M-1).  On ARM64, ANDS is 1 cycle;
   UREM is ~20 cycles.  Applied in all three LCG functions.                 */
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
    const uint64_t m = LCG_M;
    uint64_t a = LCG_A;
    uint64_t c = LCG_C;

    n = n & (m - 1ULL);   /* OPT-23: & instead of % */

    uint64_t a_new = 1;
    uint64_t c_new = 0;

    while (n > 0)
    {
        if (n & 1)
        {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }

    return (a_new * seed + c_new) & (m - 1ULL);   /* OPT-23 */
}

// Complex arithmetic — kept as out-of-line definitions for the linker;
// callers in the hot path use the static inline versions from the header.
// These definitions satisfy any remaining non-inlined references.
RSComplex c_add_fn(RSComplex A, RSComplex B)
{ RSComplex C; C.r=A.r+B.r; C.i=A.i+B.i; return C; }
RSComplex c_sub_fn(RSComplex A, RSComplex B)
{ RSComplex C; C.r=A.r-B.r; C.i=A.i-B.i; return C; }
RSComplex c_mul_fn(RSComplex A, RSComplex B)
{ RSComplex C; C.r=A.r*B.r-A.i*B.i; C.i=A.r*B.i+A.i*B.r; return C; }
RSComplex c_div_fn(RSComplex A, RSComplex B)
{
    double denom=B.r*B.r+B.i*B.i;
    RSComplex C;
    C.r=(A.r*B.r+A.i*B.i)/denom;
    C.i=(A.i*B.r-A.r*B.i)/denom;
    return C;
}
double c_abs_fn(RSComplex A) { return sqrt(A.r*A.r+A.i*A.i); }

// fast_exp and fast_cexp unchanged (correctness-critical)
double fast_exp(double x)
{
    x = 1.0 + x * 0.000244140625;
    x*=x; x*=x; x*=x; x*=x;
    x*=x; x*=x; x*=x; x*=x;
    x*=x; x*=x; x*=x; x*=x;
    return x;
}

RSComplex fast_cexp(RSComplex z)
{
    double t1 = fast_exp(z.r);
    double t2 = cos(z.i);
    double t3 = sin(z.i);
    RSComplex result = {t1*t2, t1*t3};
    return result;
}
