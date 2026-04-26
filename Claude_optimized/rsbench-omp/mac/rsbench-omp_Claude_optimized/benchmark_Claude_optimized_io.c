/* benchmark_Claude_optimized_io.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

void logo(int version)
{
    border_print();
    printf(
"                    _____   _____ ____                  _     \n"
"                   |  __ \\ / ____|  _ \\                | |    \n"
"                   | |__) | (___ | |_) | ___ _ __   ___| |__  \n"
"                   |  _  / \\___ \\|  _ < / _ \\ '_ \\ / __| '_ \\ \n"
"                   | | \\ \\ ____) | |_) |  __/ | | | (__| | | |\n"
"                   |_|  \\_\\_____/|____/ \\___|_| |_|\\___|_| |_|\n\n"
    );
    border_print();
    center_print("Developed at Argonne National Laboratory", 79);
    char v[100];
    sprintf(v, "Version: %d", version);
    center_print(v, 79);
    border_print();
}

void center_print(const char *s, int width)
{
    int length = (int)strlen(s);
    int pad    = (width - length) / 2;
    /* OPT-29: Replace per-character fputs(" ") loop with a single
       printf of the right number of spaces via %*s.  At -O0 each fputs
       call emits full function-call overhead; a single printf is one call. */
    printf("%*s%s\n", pad + 1, " ", s);
}

void border_print(void)
{
    printf("======================================================================="
           "=========\n");
}

void fancy_int(int a)
{
    if (a < 1000)
        printf("%d\n", a);
    else if (a < 1000000)
        printf("%d,%03d\n", a/1000, a%1000);
    else if (a < 1000000000)
        printf("%d,%03d,%03d\n", a/1000000, (a%1000000)/1000, a%1000);
    else
        printf("%d,%03d,%03d,%03d\n",
               a/1000000000,
               (a%1000000000)/1000000,
               (a%1000000)/1000,
               a%1000);
}

Input read_CLI(int argc, char * argv[])
{
    Input input;
    input.simulation_method = HISTORY_BASED;
    input.nthreads          = 1;
    input.n_nuclides        = 355;
    input.particles         = 300000;
    input.lookups           = 34;
    input.HM                = LARGE;
    input.avg_n_poles       = 1000;
    input.avg_n_windows     = 100;
    input.numL              = 4;
    input.doppler           = 1;
    input.kernel_id         = 0;

    int default_lookups   = 1;
    int default_particles = 1;

    for (int i = 1; i < argc; i++)
    {
        char * arg = argv[i];

        if (strcmp(arg, "-m") == 0)
        {
            char * sim_type = NULL;
            if (++i < argc) sim_type = argv[i]; else print_CLI_error();

            if (strcmp(sim_type, "history") == 0)
                input.simulation_method = HISTORY_BASED;
            else if (strcmp(sim_type, "event") == 0)
            {
                input.simulation_method = EVENT_BASED;
                if (default_lookups && default_particles)
                {
                    input.lookups  *= input.particles;
                    input.particles = 0;
                }
            }
            else print_CLI_error();
        }
        else if (strcmp(arg, "-l") == 0)
        {
            if (++i < argc) { input.lookups = atoi(argv[i]); default_lookups = 0; }
            else print_CLI_error();
        }
        else if (strcmp(arg, "-p") == 0)
        {
            if (++i < argc) { input.particles = atoi(argv[i]); default_particles = 0; }
            else print_CLI_error();
        }
        else if (strcmp(arg, "-n") == 0)
        {
            if (++i < argc) input.n_nuclides = atoi(argv[i]);
            else print_CLI_error();
        }
        else if (strcmp(arg, "-s") == 0)
        {
            if (++i < argc)
            {
                if      (strcmp(argv[i], "small") == 0) input.HM = SMALL;
                else if (strcmp(argv[i], "large") == 0) input.HM = LARGE;
                else print_CLI_error();
            }
            else print_CLI_error();
        }
        else if (strcmp(arg, "-d") == 0) { input.doppler = 0; }
        else if (strcmp(arg, "-W") == 0)
        {
            if (++i < argc) input.avg_n_windows = atoi(argv[i]);
            else print_CLI_error();
        }
        else if (strcmp(arg, "-P") == 0)
        {
            if (++i < argc) input.avg_n_poles = atoi(argv[i]);
            else print_CLI_error();
        }
        else if (strcmp(arg, "-k") == 0)
        {
            if (++i < argc) input.kernel_id = atoi(argv[i]);
            else print_CLI_error();
        }
        else print_CLI_error();
    }

    if (input.nthreads      < 1) print_CLI_error();
    if (input.n_nuclides    < 1) print_CLI_error();
    if (input.lookups       < 1) print_CLI_error();
    if (input.avg_n_poles   < 1) print_CLI_error();
    if (input.avg_n_windows < 1) print_CLI_error();

    if (input.HM == SMALL) input.n_nuclides = 68;

    return input;
}

void print_CLI_error(void)
{
    printf("Usage: ./multibench <options>\n");
    printf("Options include:\n");
    printf("  -s <size>        Size of H-M Benchmark to run (small, large)\n");
    printf("  -l <lookups>     Number of Cross-section (XS) lookups per particle history\n");
    printf("  -p <particles>   Number of particle histories\n");
    printf("  -P <poles>       Average Number of Poles per Nuclide\n");
    printf("  -W <poles>       Average Number of Windows per Nuclide\n");
    printf("  -d               Disables Temperature Dependence (Doppler Broadening)\n");
    printf("Default is equivalent to: -s large -l 34 -p 300000 -P 1000 -W 100\n");
    printf("See readme for full description of default run values\n");
    exit(4);
}

void print_input_summary(Input input)
{
    size_t mem = get_mem_estimate(input);
    printf("Programming Model:           OpenMP Taget Offloading\n");
    if (input.simulation_method == EVENT_BASED)
        printf("Simulation Method:           Event Based\n");
    else
        printf("Simulation Method:           History Based\n");
    printf("Materials:                   12\n");
    printf("H-M Benchmark Size:          %s\n", input.HM == 0 ? "Small" : "Large");
    printf("Temperature Dependence:      %s\n", input.doppler == 1 ? "ON" : "OFF");
    printf("Total Nuclides:              %d\n", input.n_nuclides);
    printf("Avg Poles per Nuclide:       "); fancy_int(input.avg_n_poles);
    printf("Avg Windows per Nuclide:     "); fancy_int(input.avg_n_windows);

    int lookups = input.lookups;
    if (input.simulation_method == HISTORY_BASED)
    {
        printf("Particles:                   "); fancy_int(input.particles);
        printf("XS Lookups per Particle:     "); fancy_int(input.lookups);
        lookups *= input.particles;
    }
    printf("Total XS Lookups:            "); fancy_int(lookups);
    printf("Est. Memory Usage (MB):      %.1lf\n", mem / 1024.0 / 1024.0);
}

int validate_and_print_results(Input input, double runtime,
                               unsigned long vhash, double kernel_time)
{
    int lookups = (input.simulation_method == HISTORY_BASED)
                ? input.lookups * input.particles
                : input.lookups;

    int lookups_per_sec      = (int)((double)lookups / runtime);
    int sim_only_lookups_per_sec = (int)((double)lookups / kernel_time);

    printf("Total Time Statistics (OpenMP Init / JIT Compilation + Simulation Kernel)\n");
    printf("Runtime:               %.3lf seconds\n", runtime);
    printf("Lookups:               "); fancy_int(lookups);
    printf("Lookups/s:             "); fancy_int(lookups_per_sec);
    printf("Simulation Kernel Only Statistics\n");
    printf("Lookups/s:             "); fancy_int(sim_only_lookups_per_sec);

    unsigned long long large = 0, small_v = 0;
    if (input.simulation_method == HISTORY_BASED) { large = 351485; small_v = 879693;  }
    else                                           { large = 358389; small_v = 880018;  }

    int is_invalid = 1;
    unsigned long long ref = (input.HM == LARGE) ? large : small_v;

    if (vhash == ref)
    {
        printf("Verification checksum: %lu (Valid)\n", vhash);
        is_invalid = 0;
    }
    else
        printf("Verification checksum: %lu (WARNING - INVALID CHECKSUM!)\n", vhash);

    return is_invalid;
}
