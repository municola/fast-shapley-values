#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "tsc_x86.h"
#include "utils.h"
#include "runfile.h"
#include "io.h"
#include "benchmark.h"
#include "base_exact_shapley.h"

// external global var, needed for special compare function
extern double *dist_gt;

// Setup memory and "problem sizes" (feature length, etc.) given the input size
context_t get_context(int input_size){
    context_t context = {
        .input_size = input_size,
        .feature_len = 2048,
        .num_test_samples = 500,

        .x_trn = NULL,
        .x_tst = NULL,

        .y_trn = NULL,
        .y_tst = NULL,
        .dist_gt = NULL,

        .size_x_tst = input_size,
        .size_y_tst = input_size,

        .size_x_trn = input_size,        
        .size_y_trn = input_size,


        // KNN result
        .x_test_knn_gt = NULL,
        
        .sp_gt = NULL,

        .T = 1,
        .K = 1
    };

    // Allocate memory - Load data
    context.x_trn = calloc(context.input_size * context.feature_len, sizeof(double));
    context.y_trn = calloc(context.input_size, sizeof(double));
    
    context.x_tst = calloc(context.num_test_samples * context.feature_len, sizeof(double));
    context.y_tst = calloc(context.num_test_samples, sizeof(double));
    

    context.x_test_knn_gt = calloc(context.num_test_samples * context.input_size, sizeof(int));
    context.sp_gt = calloc(context.num_test_samples * context.input_size, sizeof(double));
    
    // Allocate dist_gt and set global variable (needed for special compare func.!)
    context.dist_gt = calloc(context.feature_len * context.input_size, sizeof(double));
    dist_gt = context.dist_gt;

    read_bin_file_known_size(context.x_trn, "../data/features/cifar10/train_features.bin", context.input_size*context.feature_len);
    read_bin_file_known_size(context.y_trn, "../data/features/cifar10/train_labels.bin", context.input_size*1);
    read_bin_file_known_size(context.x_tst, "../data/features/cifar10/test_features.bin", context.num_test_samples*context.feature_len);
    read_bin_file_known_size(context.y_tst, "../data/features/cifar10/test_labels.bin", context.num_test_samples);


    return context;
}


void start_benchmark(run_variables_t *run_variables){
    uint64_t *measured_cycles = calloc(run_variables->number_of_input_sizes * run_variables->number_of_runs, sizeof(uint64_t));
    
    // Try to run measurements as consecutive as possible, to avoid cache pollution
    // or other interferings from the infrastructure
    for(int input_size_no=0; input_size_no<run_variables->number_of_input_sizes; input_size_no++){
        // Get context / setup environment for all runs of this input size
        int input_size = run_variables->input_sizes[input_size_no];
        context_t context = get_context(input_size);

        for(int run=0; run<run_variables->number_of_runs; run++){
            printf("\rBenchmark running: Input size N = %d, Run %d / %d       ", input_size, run+1, run_variables->number_of_runs);
            fflush(stdout);
            measured_cycles[input_size_no*run_variables->number_of_runs + run] = measure_single_run(run_variables, &context);
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
uint64_t measure_single_run(run_variables_t *run_variables, context_t *context){
    // variant 1: directly return (sub)measurement of shapley
    return run_variables->shapley_func(context);
    
    // variant 2: measure all computations
    uint64_t start, end;
    start = start_tsc();
    end = stop_tsc(start);
    return end;
}