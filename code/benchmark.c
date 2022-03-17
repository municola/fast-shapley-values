#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "utils.h"
#include "runfile.h"
#include "benchmark.h"

void start_benchmark(run_variables_t *run_variables){
    uint64_t *measured_cycles = calloc(run_variables->number_of_runs, sizeof(uint64_t));
    
    // Try to run measurements as consecutive as possible, to avoid cache pollution
    // or other interferings from the infrastructure
    for(int run=0; run<run_variables->number_of_runs; run++){
        printf("\rBenchmark running: %d / %d       ", run+1, run_variables->number_of_runs);
        fflush(stdout);
        measured_cycles[run] = measure_single_run(run_variables);
    }
    printf("\n");

    // Now record all the data
    for(int run=0; run<run_variables->number_of_runs; run++){
        add_benchmark(run_variables->runfile, measured_cycles[run]);
    }
}


// Runs all the computations that should be measured and returns the number of cycles
uint64_t measure_single_run(run_variables_t *run_variables){
    usleep(50000);
    return rand();
}