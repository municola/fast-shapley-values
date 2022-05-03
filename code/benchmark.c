#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "tsc_x86.h"
#include "utils.h"
#include "runfile.h"
#include "benchmark.h"
#include "base_exact_shapley.h"

// Include the source code to be benchmarked
//#include "io.c"

void start_benchmark(run_variables_t *run_variables){
    uint64_t *measured_cycles = calloc(run_variables->number_of_input_sizes * run_variables->number_of_runs, sizeof(uint64_t));
    
    // Try to run measurements as consecutive as possible, to avoid cache pollution
    // or other interferings from the infrastructure
    for(int input_size_no=0; input_size_no<run_variables->number_of_input_sizes; input_size_no++){
        int input_size = run_variables->input_sizes[input_size_no];
        for(int run=0; run<run_variables->number_of_runs; run++){
            printf("\rBenchmark running: Input size N = %d, Run %d / %d       ", input_size, run+1, run_variables->number_of_runs);
            fflush(stdout);
            measured_cycles[input_size_no*run_variables->number_of_runs + run] = measure_single_run(run_variables, input_size_no);
        }
        printf("\n");
    }

    // Now record all the data
    for(int input_size_no=0; input_size_no<run_variables->number_of_input_sizes; input_size_no++){
        int input_size = run_variables->input_sizes[input_size_no];
        for(int run=0; run<run_variables->number_of_runs; run++){
            add_benchmark(run_variables->runfile, run_variables->input_sizes, run_variables->number_of_input_sizes, input_size, measured_cycles[input_size_no*run_variables->number_of_runs + run]);
        }
    }

    free(measured_cycles);
}


// Runs all the computations that should be measured and returns the number of cycles
uint64_t measure_single_run(run_variables_t *run_variables, int input_size_no){
    // Rely on tsc_x86.h to count the cycles

    // variant 1: directly return (sub)measurement of shapley
    return run_variables->shapley_func(run_variables, input_size_no);
    
    // variant 2: measure all computations
    uint64_t start, end;
    start = start_tsc();
    end = stop_tsc(start);
    return end;
}