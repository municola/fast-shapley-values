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
void init_context(context_t *ctx, int input_size){
    ctx->input_size = input_size;
    ctx->feature_len = 2048;
    ctx->num_test_samples = 500;

    ctx->x_trn = NULL;
    ctx->x_tst = NULL;

    ctx->y_trn = NULL;
    ctx->y_tst = NULL;
    ctx->dist_gt = NULL;
    ctx->sp_gt = NULL;


    // KNN result
    ctx->x_test_knn_gt = NULL;

    ctx->size_x_tst = input_size;
    ctx->size_y_tst = input_size;

    ctx->size_x_trn = input_size;
    ctx->size_y_trn = input_size;
    
    ctx->T = 1;
    ctx->K = 1;
    

    // Allocate memory - Load data
    if(ctx->x_trn) free(ctx->x_trn);
    ctx->x_trn = calloc(ctx->input_size * ctx->feature_len, sizeof(double));

    if(ctx->y_trn) free(ctx->y_trn);
    ctx->y_trn = calloc(ctx->input_size, sizeof(double));
    
    if(ctx->x_tst) free(ctx->x_tst);
    ctx->x_tst = calloc(ctx->num_test_samples * ctx->feature_len, sizeof(double));

    if(ctx->y_tst) free(ctx->y_tst);
    ctx->y_tst = calloc(ctx->num_test_samples, sizeof(double));
    
    if(ctx->x_test_knn_gt) free(ctx->x_test_knn_gt);
    ctx->x_test_knn_gt = calloc(ctx->num_test_samples * ctx->input_size, sizeof(int));

    if(ctx->sp_gt) free(ctx->sp_gt);
    ctx->sp_gt = calloc(ctx->num_test_samples * ctx->input_size, sizeof(double));
    
    // Allocate dist_gt and set global variable (needed for special compare func.!)
    if(ctx->dist_gt) free(ctx->dist_gt);
    ctx->dist_gt = calloc(ctx->feature_len * ctx->input_size, sizeof(double));
    dist_gt = ctx->dist_gt;

    read_bin_file_known_size(ctx->x_trn, "../data/features/cifar10/train_features.bin", ctx->input_size*ctx->feature_len);
    read_bin_file_known_size(ctx->y_trn, "../data/features/cifar10/train_labels.bin", ctx->input_size*1);
    read_bin_file_known_size(ctx->x_tst, "../data/features/cifar10/test_features.bin", ctx->num_test_samples*ctx->feature_len);
    read_bin_file_known_size(ctx->y_tst, "../data/features/cifar10/test_labels.bin", ctx->num_test_samples);
}


void start_benchmark(run_variables_t *run_variables){
    uint64_t *measured_cycles = calloc(run_variables->number_of_input_sizes * run_variables->number_of_runs, sizeof(uint64_t));
    
    // Context (i.e. common test/training data used in the implementations)
    // (initialised once per input size)
    context_t context;

    // Try to run measurements as consecutive as possible, to avoid cache pollution
    // or other interferings from the infrastructure
    for(int input_size_no=0; input_size_no<run_variables->number_of_input_sizes; input_size_no++){
        // Get context / setup environment for all runs of this input size
        int input_size = run_variables->input_sizes[input_size_no];
        init_context(&context, input_size);

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