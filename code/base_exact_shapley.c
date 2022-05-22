#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tsc_x86.h"
#include "io.h"
#include "utils.h"
#include "benchmark.h"
#include "knn_exact.h"
#include "combined_exact_knn_shapley.h"


uint64_t run_shapley(void *context) {  
    context_t *ctx = (context_t *)context;
    uint64_t start_timer, end_timer;

    get_true_exact_KNN(context);
    start_timer = start_tsc();
    single_unweighted_knn_class_shapley_opt(context);
    end_timer = stop_tsc(start_timer);

    printf("Cycles: %lu\n", end_timer);

    return end_timer;
}


// Base implementation (Also used for Correctness. Do not touch)
void compute_single_unweighted_knn_class_shapley(void *context_ptr){
    context_t *context = (context_t*)context_ptr;
    debug_print("%s", "\nStart Shapley computation:\n");
    for(int j=0; j<context->size_x_tst;j++){
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = context->x_test_knn_gt[j*context->size_x_trn+context->size_x_trn-1];
        double tmp = (context->y_trn[offset] == context->y_tst[j]) ? 1.0 : 0.0;
        context->sp_gt[j*context->size_x_trn + offset] = tmp / context->size_x_trn; 

        for (int i=context->size_x_trn-2; i>-1; i--) {
            // debug_print("    i=%d\n", i);
            int index_j_i = j*context->size_x_trn+i;

            double s_j_alpha_i_plus_1 = context->sp_gt[j*context->size_x_trn + context->x_test_knn_gt[index_j_i+1]];
            double difference = (double)(context->y_trn[context->x_test_knn_gt[index_j_i]] == context->y_tst[j]) - 
                                        (double)(context->y_trn[context->x_test_knn_gt[index_j_i+1]] == context->y_tst[j]);
            double min_K_i = context->K < i+1 ? context->K : i+1;

            // debug_print("      s_j=%f\n", s_j_alpha_i_plus_1);
            // debug_print("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // debug_print("      min_=%f\n", min_K_i);

            context->sp_gt[j*context->size_x_trn + context->x_test_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference / context->K) * (min_K_i / (i+1));
        }
    }

    debug_print("%s", "Exact: Shapley Values done :)\n");
}


void single_unweighted_knn_class_shapley_opt0(void *context_ptr){
    /* Baseline */
    context_t *context = (context_t*)context_ptr;
    debug_print("%s", "\nStart Shapley computation:\n");
    for(int j=0; j<context->size_x_tst;j++){
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = context->x_test_knn_gt[j*context->size_x_trn+context->size_x_trn-1];
        double tmp = (context->y_trn[offset] == context->y_tst[j]) ? 1.0 : 0.0;
        context->sp_gt[j*context->size_x_trn + offset] = tmp / context->size_x_trn; 

        for (int i=context->size_x_trn-2; i>-1; i--) {
            int index_j_i = j*context->size_x_trn+i;
            double s_j_alpha_i_plus_1 = context->sp_gt[j*context->size_x_trn + context->x_test_knn_gt[index_j_i+1]];
            double difference = (double)(context->y_trn[context->x_test_knn_gt[index_j_i]] == context->y_tst[j]) - 
                                        (double)(context->y_trn[context->x_test_knn_gt[index_j_i+1]] == context->y_tst[j]);
            double min_K_i = context->K < i+1 ? context->K : i+1;
            context->sp_gt[j*context->size_x_trn + context->x_test_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference / context->K) * (min_K_i / (i+1));
        }
    }
}


void single_unweighted_knn_class_shapley_opt1(void *context_ptr){
    // opt1: avoid pointer chasing, use local vars
    context_t *context = (context_t*)context_ptr;
    debug_print("%s", "\nStart Shapley computation:\n");
    
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;

    double inv_size_x_trn = 1.0/size_x_trn;

    for(int j=0; j<context->size_x_tst;j++) {
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        double y_tst_j = y_tst[j];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == y_tst_j) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            int index_j_i = j*size_x_trn+i;
            int x_test_knn_gt_i = x_test_knn_gt[index_j_i];
            int x_test_knn_gt_i_plus_one = x_test_knn_gt[index_j_i+1];

            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt_i_plus_one];
            double difference = (double)(y_trn[x_test_knn_gt_i] == y_tst_j) - 
                                        (double)(y_trn[x_test_knn_gt_i_plus_one] == y_tst_j);
            double min_K_i = K < i+1 ? K : i+1;

            sp_gt[j*size_x_trn + x_test_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference / K) * (min_K_i / (i+1));
        }
    }

    debug_print("%s", "Exact: Shapley Values done :)\n");
}



void single_unweighted_knn_class_shapley_opt(void *context_ptr){
    // opt2: Precompute yellow part
    context_t *context = (context_t*)context_ptr;
    debug_print("%s", "\nStart Shapley computation:\n");
    
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    double inv_size_x_trn = 1.0/size_x_trn;

    for(int j=0; j<context->size_x_tst;j++) {
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        double y_tst_j = y_tst[j];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == y_tst_j) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            int index_j_i = j*size_x_trn+i;
            int x_test_knn_gt_i = x_test_knn_gt[index_j_i];
            int x_test_knn_gt_i_plus_one = x_test_knn_gt[index_j_i+1];

            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt_i_plus_one];
            double difference = (double)(y_trn[x_test_knn_gt_i] == y_tst_j) - 
                                        (double)(y_trn[x_test_knn_gt_i_plus_one] == y_tst_j);
            double min_K_i = K < i+1 ? K : i+1;
            sp_gt[j*size_x_trn + x_test_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[i]);  
        }
    }

    debug_print("%s", "Exact: Shapley Values done :)\n");
}

