/* benchmark_Claude_optimized_rsbench.h
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include<string.h>
#include<stdint.h>
#include<float.h>
#include<omp.h>
#include<assert.h>

#define OPENMP

#define PI 3.14159265359

// typedefs
typedef enum __hm{SMALL, LARGE, XL, XXL} HM_size;

#define HISTORY_BASED 1
#define EVENT_BASED 2

#define STARTING_SEED 1070
#define INITIALIZATION_SEED 42

/* OPT-1: LCG constants promoted to macros so they resolve at compile time
   rather than being loaded from stack-allocated locals at -O0.            */
#define LCG_M  9223372036854775808ULL  /* 2^63 */
#define LCG_A  2806196910506780709ULL
#define LCG_C  1ULL

/* OPT-2: Inline LCG functions as macros to eliminate call overhead at -O0.
   At -O0 every function call emits full frame setup/teardown. These are
   called millions of times in the hot simulation loop.                    */
#define LCG_STEP(seed) \
    ((seed) = (LCG_A * (seed) + LCG_C) & (LCG_M - 1ULL))

#define LCG_RAND_DOUBLE(seed) \
    ((double)LCG_STEP(seed) / (double)LCG_M)

#define LCG_RAND_INT(seed) \
    (LCG_STEP(seed))

/* OPT-3: Pre-computed reciprocal for pick_mat running sum:
   avoids repeated double-precision adds inside the inner loop.
   Defined here so it can be used inline in simulation.c.                  */

/* OPT-4: numL is always 4 — encode as a compile-time constant so the
   calculate_sig_T loop bound is a literal (enabling the compiler to unroll
   it even at -O0 via pragma, and avoiding a memory load for input.numL).  */
#define NUM_L 4

typedef struct{
	double r;
	double i;
} RSComplex;

typedef struct{
	int nthreads;
	int n_nuclides;
	int lookups;
	HM_size HM;
	int avg_n_poles;
	int avg_n_windows;
	int numL;
	int doppler;
	int particles;
	int simulation_method;
	int kernel_id;
} Input;

typedef struct{
	RSComplex MP_EA;
	RSComplex MP_RT;
	RSComplex MP_RA;
	RSComplex MP_RF;
	short int l_value;
} Pole;

typedef struct{
	double T;
	double A;
	double F;
	int start;
	int end;
} Window;

typedef struct{
	int * n_poles;
	unsigned long length_n_poles;
	int * n_windows;
	unsigned long length_n_windows;
	Pole * poles;
	unsigned long length_poles;
	Window * windows;
	unsigned long length_windows;
	double * pseudo_K0RS;
	unsigned long length_pseudo_K0RS;
	int * num_nucs;
	unsigned long length_num_nucs;
	int * mats;
	unsigned long length_mats;
	double * concs;
	unsigned long length_concs;
	int max_num_nucs;
	int max_num_poles;
	int max_num_windows;
	double * p_energy_samples;
	unsigned long length_p_energy_samples;
	int * mat_samples;
	unsigned long length_mat_samples;
} SimulationData;

// io.c
void logo(int version);
void center_print(const char *s, int width);
void border_print(void);
void fancy_int( int a );
Input read_CLI( int argc, char * argv[] );
void print_CLI_error(void);
void print_input_summary(Input input);
int validate_and_print_results(Input input, double runtime, unsigned long vhash, double kernel_time);

// init.c
SimulationData initialize_simulation( Input input );
int * generate_n_poles( Input input,  uint64_t * seed );
int * generate_n_windows( Input input ,  uint64_t * seed);
Pole * generate_poles( Input input, int * n_poles, uint64_t * seed, int * max_num_poles );
Window * generate_window_params( Input input, int * n_windows, int * n_poles, uint64_t * seed, int * max_num_windows );
double * generate_pseudo_K0RS( Input input, uint64_t * seed );

// material.c
int * load_num_nucs(Input input);
int * load_mats( Input input, int * num_nucs, int * max_num_nucs, unsigned long * length_mats );
double * load_concs( int * num_nucs, uint64_t * seed, int max_num_nucs );
SimulationData get_materials(Input input, uint64_t * seed);

// utils.c
size_t get_mem_estimate( Input input );
RSComplex fast_cexp( RSComplex z );
double get_time(void);

// simulation.c
RSComplex fast_nuclear_W( RSComplex Z );
void calculate_macro_xs( double * macro_xs, int mat, double E, Input input, int * num_nucs, int * mats, int max_num_nucs, double * concs, int * n_windows, double * pseudo_K0Rs, Window * windows, Pole * poles, int max_num_windows, int max_num_poles ) ;
void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, int * n_windows, double * pseudo_K0RS, Window * windows, Pole * poles, int max_num_windows, int max_num_poles);
void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, int * n_windows, double * pseudo_K0RS, Window * windows, Pole * poles, int max_num_windows, int max_num_poles );
void run_event_based_simulation(Input input, SimulationData data, unsigned long * vhash_result, double * kernel_init_time );
void run_history_based_simulation(Input input, SimulationData data, unsigned long * vhash_result );
double LCG_random_double(uint64_t * seed);
uint64_t LCG_random_int(uint64_t * seed);
uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);
int pick_mat( uint64_t * seed );
void calculate_sig_T( int nuc, double E, Input input, double * pseudo_K0RS, RSComplex * sigTfactors );

// rscomplex inline helpers — defined here so they can be used without call
// overhead in the hot path at -O0.
/* OPT-5: Convert the five RSComplex functions from out-of-line functions to
   static inline.  At -O0 each call emits ~8 ARM64 instructions of frame
   overhead.  calculate_micro_xs_doppler calls c_mul/c_sub/c_add roughly
   5 times per pole per nuclide per lookup — inlining removes millions of
   wasted frame instructions per run.                                       */
static inline RSComplex c_add(RSComplex A, RSComplex B) {
    RSComplex C; C.r = A.r + B.r; C.i = A.i + B.i; return C;
}
static inline RSComplex c_sub(RSComplex A, RSComplex B) {
    RSComplex C; C.r = A.r - B.r; C.i = A.i - B.i; return C;
}
static inline RSComplex c_mul(RSComplex A, RSComplex B) {
    RSComplex C;
    C.r = A.r*B.r - A.i*B.i;
    C.i = A.r*B.i + A.i*B.r;
    return C;
}
static inline RSComplex c_div(RSComplex A, RSComplex B) {
    double denom = B.r*B.r + B.i*B.i;
    RSComplex C;
    C.r = (A.r*B.r + A.i*B.i) / denom;
    C.i = (A.i*B.r - A.r*B.i) / denom;
    return C;
}
static inline double c_abs(RSComplex A) {
    return sqrt(A.r*A.r + A.i*A.i);
}
