#include "rsbench_gemini_optimized.h"

void run_event_based_simulation(Input input, SimulationData data, unsigned long * vhash_result, double * kernel_time )
{
  printf("Beginning event based simulation ...\n");
  unsigned long verification = 0;

  double start = get_time();

  #pragma omp parallel reduction(+:verification)
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    int chunk = (input.lookups + nthreads - 1) / nthreads;
    int istart = tid * chunk;
    int iend = (istart + chunk > input.lookups) ? input.lookups : istart + chunk;

    for( int i = istart; i < iend; i++ )
    {
      uint64_t seed = STARTING_SEED;  
      seed = fast_forward_LCG(seed, 2*i);

      double E = LCG_random_double(&seed);
      int mat  = pick_mat(&seed);

      double macro_xs[4] = {0};

      calculate_macro_xs( macro_xs, mat, E, input, data.num_nucs, data.mats,
                          data.max_num_nucs, data.concs, data.n_windows,
                          data.pseudo_K0RS, data.windows, data.poles,
                          data.max_num_windows, data.max_num_poles );

      double max = -DBL_MAX;
      int max_idx = 0;
      // Manual unrolling for -O0
      if( macro_xs[0] > max ) { max = macro_xs[0]; max_idx = 0; }
      if( macro_xs[1] > max ) { max = macro_xs[1]; max_idx = 1; }
      if( macro_xs[2] > max ) { max = macro_xs[2]; max_idx = 2; }
      if( macro_xs[3] > max ) { max = macro_xs[3]; max_idx = 3; }
      
      verification += max_idx+1;
    }
  }

  double stop = get_time();
  printf("Kernel execution took %.2lf seconds.\n", stop-start);
  *kernel_time = stop-start;
  *vhash_result = verification;
}

void calculate_macro_xs(double * macro_xs, int mat, double E, Input input,
                        int * num_nucs, int * mats, int max_num_nucs,
                        double * concs, int * n_windows, double * pseudo_K0Rs,
                        Window * windows, Pole * poles, int max_num_windows, int max_num_poles ) 
{
  macro_xs[0] = 0; macro_xs[1] = 0; macro_xs[2] = 0; macro_xs[3] = 0;

  int n_nucs = num_nucs[mat];
  int mat_base = mat * max_num_nucs;

  for( int i = 0; i < n_nucs; i++ )
  {
    double micro_xs[4];
    int nuc = mats[mat_base + i];
    double conc = concs[mat_base + i];

    if( input.doppler == 1 )
      calculate_micro_xs_doppler( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);
    else
      calculate_micro_xs( micro_xs, nuc, E, input, n_windows, pseudo_K0Rs, windows, poles, max_num_windows, max_num_poles);

    macro_xs[0] += micro_xs[0] * conc;
    macro_xs[1] += micro_xs[1] * conc;
    macro_xs[2] += micro_xs[2] * conc;
    macro_xs[3] += micro_xs[3] * conc;
  }
}

void calculate_micro_xs(double * micro_xs, int nuc, double E, Input input,
                        int * n_windows, double * pseudo_K0RS, Window * windows,
                        Pole * poles, int max_num_windows, int max_num_poles)
{
  double sigT, sigA, sigF, sigE;
  double inv_spacing = (double)n_windows[nuc];
  int window = (int) ( E * inv_spacing );
  if( window >= n_windows[nuc] ) window = n_windows[nuc] - 1;

  RSComplex sigTfactors[4];
  calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

  Window w = windows[nuc * max_num_windows + window];
  sigT = E * w.T;
  sigA = E * w.A;
  sigF = E * w.F;

  int pole_base = nuc * max_num_poles;
  RSComplex t1 = {0, 1};
  RSComplex t2 = {sqrt(E), 0 };
  RSComplex E_c = {E, 0};

  for( int i = w.start; i <= w.end; i++ )
  {
    Pole pole = poles[pole_base + i];
    RSComplex PSIIKI = c_div( t1 , c_sub(pole.MP_EA, t2) );
    RSComplex CDUM = c_div(PSIIKI, E_c);
    
    sigT += (c_mul(pole.MP_RT, c_mul(CDUM, sigTfactors[pole.l_value])) ).r;
    sigA += (c_mul( pole.MP_RA, CDUM)).r;
    sigF += (c_mul(pole.MP_RF, CDUM)).r;
  }

  sigE = sigT - sigA;
  micro_xs[0] = sigT; micro_xs[1] = sigA; micro_xs[2] = sigF; micro_xs[3] = sigE;
}

void calculate_micro_xs_doppler(double * micro_xs, int nuc, double E,
                                Input input, int * n_windows, double * pseudo_K0RS,
                                Window * windows, Pole * poles, int max_num_windows, int max_num_poles )
{
  double sigT, sigA, sigF, sigE;
  double inv_spacing = (double)n_windows[nuc];
  int window = (int) ( E * inv_spacing );
  if( window >= n_windows[nuc] ) window = n_windows[nuc] - 1;

  RSComplex sigTfactors[4];
  calculate_sig_T(nuc, E, input, pseudo_K0RS, sigTfactors );

  Window w = windows[nuc * max_num_windows + window];
  sigT = E * w.T;
  sigA = E * w.A;
  sigF = E * w.F;

  double dopp = 0.5;
  int pole_base = nuc * max_num_poles;

  for( int i = w.start; i <= w.end; i++ )
  {
    Pole pole = poles[pole_base + i];
    RSComplex Z = {(E - pole.MP_EA.r) * dopp, (-pole.MP_EA.i) * dopp};
    RSComplex faddeeva = fast_nuclear_W( Z );

    sigT += (c_mul( pole.MP_RT, c_mul(faddeeva, sigTfactors[pole.l_value]) )).r;
    sigA += (c_mul( pole.MP_RA , faddeeva)).r;
    sigF += (c_mul( pole.MP_RF , faddeeva)).r;
  }

  sigE = sigT - sigA;
  micro_xs[0] = sigT; micro_xs[1] = sigA; micro_xs[2] = sigF; micro_xs[3] = sigE;
}

int pick_mat( uint64_t * seed )
{
  static const double dist[12] = {0.140, 0.052, 0.275, 0.134, 0.154, 0.064, 0.066, 0.055, 0.008, 0.015, 0.025, 0.013};
  double roll = LCG_random_double(seed);
  double running = 0;
  for( int i = 0; i < 12; i++ )
  {
    running += dist[i];
    if( roll < running ) return i;
  }
  return 0;
}

void calculate_sig_T( int nuc, double E, Input input, double * pseudo_K0RS, RSComplex * sigTfactors )
{
  double sqE = sqrt(E);
  int base = nuc * input.numL;
  for( int i = 0; i < 4; i++ )
  {
    double phi = pseudo_K0RS[base + i] * sqE;
    if( i == 1 ) phi += atan( phi );
    else if( i == 2 ) phi -= atan( 3.0 * phi / (3.0 - phi*phi));
    else if( i == 3 ) phi -= atan(phi*(15.0-phi*phi)/(15.0-6.0*phi*phi));
    phi *= 2.0;
    sigTfactors[i].r = cos(phi);
    sigTfactors[i].i = -sin(phi);
  }
}

// Fast exponential using NEON for potential vectorization (though used scalar here for simplicity at -O0)
static inline double fast_exp_opt(double x) {
    x = 1.0 + x * 0.000244140625;
    // x = x^4096 via 12 doublings
    for(int i=0; i<12; i++) x *= x;
    return x;
}

static inline RSComplex fast_cexp_opt(RSComplex z) {
    double t1 = fast_exp_opt(z.r);
    return (RSComplex){t1 * cos(z.i), t1 * sin(z.i)};
}

RSComplex fast_nuclear_W( RSComplex Z )
{
  double absZ2 = Z.r*Z.r + Z.i*Z.i;
  if( absZ2 < 36.0 )
  {
    RSComplex prefactor = {0, 8.124330e+01};
    static const double an[10] = {2.758402e-01, 2.245740e-01, 1.594149e-01, 9.866577e-02, 5.324414e-02, 2.505215e-02, 1.027747e-02, 3.676164e-03, 1.146494e-03, 3.117570e-04};
    static const double neg_1n[10] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
    static const double den_l[10] = {9.869604e+00, 3.947842e+01, 8.882644e+01, 1.579137e+02, 2.467401e+02, 3.553058e+02, 4.836106e+02, 6.316547e+02, 7.994380e+02, 9.869604e+02};

    RSComplex t1 = {0, 12};
    RSComplex exp_t1Z = fast_cexp_opt(c_mul(t1, Z));
    RSComplex W = c_div(c_mul((RSComplex){0,1}, ( c_sub((RSComplex){1,0}, exp_t1Z) )) , c_mul((RSComplex){12,0}, Z));
    RSComplex sum = {0,0};
    for( int n = 0; n < 10; n++ )
    {
      RSComplex top = c_sub(c_mul((RSComplex){neg_1n[n], 0}, exp_t1Z), (RSComplex){1,0});
      RSComplex bot = c_sub((RSComplex){den_l[n], 0}, c_mul((RSComplex){144, 0}, c_mul(Z,Z)));
      sum = c_add(sum, c_mul((RSComplex){an[n], 0}, c_div(top, bot)));
    }
    return c_add(W, c_mul(prefactor, c_mul(Z, sum)));
  }
  else
  {
    static const RSComplex a = {0.512424224754, 0}, b = {0.275255128608, 0}, c = {0.051765358793, 0}, d = {2.724744871392, 0};
    RSComplex Z2 = c_mul(Z, Z);
    return c_mul(c_mul(Z, (RSComplex){0,1}), (c_add(c_div(a, c_sub(Z2, b)), c_div(c, c_sub(Z2, d)))));
  }
}
