#ifndef __RSBENCH_GEMINI_OPTIMIZED_H__
#define __RSBENCH_GEMINI_OPTIMIZED_H__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <omp.h>
#include <assert.h>
#include <arm_neon.h>

#define PI 3.14159265359
#define HISTORY_BASED 1
#define EVENT_BASED 2
#define STARTING_SEED 1070
#define INITIALIZATION_SEED 42

typedef enum __hm{SMALL, LARGE, XL, XXL} HM_size;

typedef struct {
    double r;
    double i;
} RSComplex;

typedef struct {
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

// Align to 128-byte cache line for Apple Silicon
typedef struct __attribute__((aligned(128))) {
    RSComplex MP_EA;
    RSComplex MP_RT;
    RSComplex MP_RA;
    RSComplex MP_RF;
    short int l_value;
    char padding[102]; // Padding to reach 128 bytes
} Pole;

typedef struct __attribute__((aligned(128))) {
    double T;
    double A;
    double F;
    int start;
    int end;
    char padding[104]; // Padding to reach 128 bytes
} Window;

typedef struct {
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
double get_time(void);

// Inline complex operations and LCG for -O0 optimization
static inline RSComplex c_add(RSComplex A, RSComplex B) {
    return (RSComplex){A.r + B.r, A.i + B.i};
}
static inline RSComplex c_sub(RSComplex A, RSComplex B) {
    return (RSComplex){A.r - B.r, A.i - B.i};
}
static inline RSComplex c_mul(RSComplex A, RSComplex B) {
    return (RSComplex){A.r * B.r - A.i * B.i, A.r * B.i + A.i * B.r};
}
static inline RSComplex c_div(RSComplex A, RSComplex B) {
    double denom = B.r * B.r + B.i * B.i;
    return (RSComplex){(A.r * B.r + A.i * B.i) / denom, (A.i * B.r - A.r * B.i) / denom};
}
static inline double c_abs(RSComplex A) {
    return sqrt(A.r * A.r + A.i * A.i);
}

static inline double LCG_random_double(uint64_t * seed) {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double)(*seed) / (double)m;
}

static inline uint64_t LCG_random_int(uint64_t * seed) {
    const uint64_t m = 9223372036854775808ULL;
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return *seed;
}

static inline uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
    const uint64_t m = 9223372036854775808ULL;
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while(n > 0) {
        if(n & 1) {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }
    return (a_new * seed + c_new) % m;
}

// simulation.c
void run_event_based_simulation(Input input, SimulationData data, unsigned long * vhash_result, double * kernel_init_time );
RSComplex fast_nuclear_W( RSComplex Z );
void calculate_macro_xs( double * macro_xs, int mat, double E, Input input, int * num_nucs, int * mats, int max_num_nucs, double * concs, int * n_windows, double * pseudo_K0Rs, Window * windows, Pole * poles, int max_num_windows, int max_num_poles ) ;
void calculate_micro_xs( double * micro_xs, int nuc, double E, Input input, int * n_windows, double * pseudo_K0RS, Window * windows, Pole * poles, int max_num_windows, int max_num_poles);
void calculate_micro_xs_doppler( double * micro_xs, int nuc, double E, Input input, int * n_windows, double * pseudo_K0RS, Window * windows, Pole * poles, int max_num_windows, int max_num_poles );
int pick_mat( uint64_t * seed );
void calculate_sig_T( int nuc, double E, Input input, double * pseudo_K0RS, RSComplex * sigTfactors );

#endif
