#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include "utils.h"

typedef struct context {
    int input_size;
    int *x_test_knn_gt;
    int *x_test_knn_r_gt;
    double *sp_gt;
    double *x_trn;
    double *x_tst;
    double *y_trn;
    double *y_tst;
    double *dist_gt;
    size_t size_x_trn;
    size_t size_x_tst;
    size_t size_y_trn;
    size_t size_y_tst;
    size_t feature_len;
    size_t num_test_samples;
    int T;
    double K;
} context_t;


void start_benchmark(run_variables_t *run_variables);
uint64_t measure_single_run(run_variables_t *run_variables, context_t *context);
void init_context(context_t *ctx, int input_size);


#endif
