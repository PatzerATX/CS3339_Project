/* benchmark_Claude_optimized_main.c
   Optimized for ARM64 Apple Silicon at -O0 by Claude (Anthropic) */

#include "benchmark_Claude_optimized_rsbench.h"

int main(int argc, char * argv[])
{
    int version = 12;
    double start, stop;

    Input input = read_CLI(argc, argv);

    logo(version);
    center_print("INPUT SUMMARY", 79);
    border_print();
    print_input_summary(input);

    border_print();
    center_print("INITIALIZATION", 79);
    border_print();

    start = get_time();
    SimulationData SD = initialize_simulation(input);
    stop  = get_time();
    printf("Initialization Complete. (%.2lf seconds)\n", stop - start);

    border_print();
    center_print("SIMULATION", 79);
    border_print();

    unsigned long vhash = 0;
    double kernel_time;

    start = get_time();

    if (input.simulation_method == EVENT_BASED)
    {
        if (input.kernel_id == 0)
            run_event_based_simulation(input, SD, &vhash, &kernel_time);
        else
        {
            printf("Error: No kernel ID %d found!\n", input.kernel_id);
            exit(1);
        }
    }
    else if (input.simulation_method == HISTORY_BASED)
    {
        printf("History-based simulation not implemented in OpenMP offload code. Instead,\n"
               "use the event-based method with \"-m event\" argument.\n");
        exit(1);
    }

    stop  = get_time();
    vhash = vhash % 999983;

    printf("Simulation Complete.\n");

    border_print();
    center_print("RESULTS", 79);
    border_print();

    int is_invalid = validate_and_print_results(input, stop - start, vhash, kernel_time);

    border_print();
    return is_invalid;
}
