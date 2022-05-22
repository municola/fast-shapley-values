#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "io.h"
#include "base_exact_shapley.h"
#include "base_mc_approx_shapley.h"
#include "combined_exact_knn_shapley.h"
#include "knn_exact.h"
#include "knn_approx.h"

#define EPS 10e-3

double nrm_sqr_diff_double(double *x, double *y, int n) {
    double nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
        //debug_print("nrm_sqr_diff_double: %f %f\n", x[i], y[i]);
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    if (isnan(nrm_sqr)) {
      nrm_sqr = INFINITY;
    }
    
    return nrm_sqr;
}

int nrm_sqr_diff_int(int *x, int *y, int n) {
    int nrm_sqr = 0;
    for(int i = 0; i < n; i++) {
        //debug_print("nrm_sqr_diff_int: %d %d\n", x[i], y[i]);
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    return nrm_sqr;
}

bool exact_correct(run_variables_t *run_variables, void *context) {

    context_t *ctx = (context_t *)context;
    context_t *test_ctx2 = calloc(sizeof(context_t), 1);
    test_ctx2->input_size = ctx->input_size;

    debug_print("Input size: %d\n", ctx->input_size);

    // DO NOT TOUCH, base implementations for correctness testing
    init_context(ctx, ctx->input_size);
    get_true_exact_KNN(context);
    compute_single_unweighted_knn_class_shapley(context);

    init_context(test_ctx2, ctx->input_size);

    // replace both functions with whatever you want to test
    combined_knn_shapley_opt((void*)test_ctx2);

    double error_knn = nrm_sqr_diff_int(ctx->x_test_knn_gt, test_ctx2->x_test_knn_gt, ctx->size_x_trn*ctx->size_x_tst);
    debug_print("KNN Correctness: Error < EPS: %f < %f", error_knn, EPS);

    double error_shapley = nrm_sqr_diff_double(ctx->sp_gt, test_ctx2->sp_gt, ctx->size_x_trn*ctx->size_x_tst);
    debug_print("Shapley Correctness: Error < EPS: %f < %f", error_shapley, EPS);

    double error = error_knn + error_shapley;
    return error < EPS;
}

bool approx_correct(run_variables_t *run_variables, void *context) {
    context_t *ctx = (context_t *)context;
    context_t *test_ctx2 = calloc(sizeof(context_t), 1);
    test_ctx2->input_size = ctx->input_size;

    debug_print("Input size: %d\n", ctx->input_size);

    // DO NOT TOUCH, base implementations for correctness testing
    init_context(ctx, ctx->input_size);
    get_true_approx_KNN(context);
    srand(0);
    compute_shapley_using_improved_mc_approach(context);

    init_context(test_ctx2, ctx->input_size);
    // replace with whatever function of interest
    get_true_approx_KNN((void*)test_ctx2);
    current_compute_shapley_using_improved_mc_approach((void*)test_ctx2);

    double error_knn = nrm_sqr_diff_int(ctx->x_test_knn_gt, test_ctx2->x_test_knn_gt, ctx->size_x_trn*ctx->size_x_tst);
    debug_print("\nKNN Correctness: Error < EPS: %f < %f\n", error_knn, EPS);

    double error_shapley = nrm_sqr_diff_double(ctx->sp_gt, test_ctx2->sp_gt, ctx->size_x_trn*ctx->size_x_tst);
    debug_print("\nShapley Correctness: Error < EPS: %f < %f\n\n", error_shapley, EPS);

    double error = error_knn + error_shapley;
    return error < 100* EPS;
}

// bool exact_shapley_correct(run_variables_t *run_variables, void *context) {
//     // type hacking, sorry
//     context_t *ctx = (context_t*)context;

//     // For second run, create an identical context
//     context_t test_context2 = *ctx;
//     context_t *test_ctx2 = &test_context2;

//     debug_print("Input size: %d\n", ctx->input_size);
//     init_context(ctx, ctx->input_size);
//     init_context(test_ctx2, ctx->input_size);

    
//     // First compute results on base algorithm
//     get_true_exact_KNN(context);
//     compute_single_unweighted_knn_class_shapley(context);

//     // Then check if the specified implementation is correct
//     // run_variables->knn_func((void*)test_ctx2);
//     // run_variables->shapley_func((void*)test_ctx2);
    
//     assert(ctx->sp_gt != test_ctx2->sp_gt);

//     double error = nrm_sqr_diff((double*)ctx->sp_gt, (double*)test_ctx2->sp_gt, ctx->input_size);
//     debug_print("Shapley Correctness: Error < EPS: %f < %f", error, EPS);

//     return error < EPS;
// }