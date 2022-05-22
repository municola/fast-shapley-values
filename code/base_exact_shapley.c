#include <immintrin.h>
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

void print_vec_s(__m256 var) {
    printf("%d %d %d %d \n", 
           var[0], var[1], var[2], var[3]);
}


/***************************************** IMPLEMENTATIONS *******************************************/



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
}



void single_unweighted_knn_class_shapley_opt2(void *context_ptr){
    // opt2: Precompute yellow part and difference as int
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        double y_tst_j = y_tst[j];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == y_tst_j) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            int x_test_knn_gt_i = x_test_knn_gt[j*size_x_trn+i];

            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn +  x_test_knn_gt[j*size_x_trn+i+1]];
            int difference = (y_trn[x_test_knn_gt_i] == y_tst_j) - 
                                        (y_trn[ x_test_knn_gt[j*size_x_trn+i+1]] == y_tst_j);
            sp_gt[j*size_x_trn + x_test_knn_gt_i] = s_j_alpha_i_plus_1 + (difference * Kidx_const[i]);  
        }
    }
}


void single_unweighted_knn_class_shapley_opt3(void *context_ptr){
    // opt3: Precompute yellow part and orange part (only need to calcualte indicator for each i once)
    // also difference as int
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    int* ind_sub = (int*)malloc((size_x_trn) * sizeof(int));

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        double label_test = y_tst[j]; // THIS NEEDS TO BE A DOUBLE (otherwise notable performance hit)
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn; i++) {
            ind_sub[i] = (y_trn[x_test_knn_gt[j*size_x_trn+i]] == label_test);
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i+1]];
            int difference = ind_sub[i] - ind_sub[i+1];
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[i]);  
        }
    }
}



void single_unweighted_knn_class_shapley_opt4(void *context_ptr){
    // opt4: Precompute yellow part and orange part (only need to calcualte indicator for each i once)
    // We directely comput the differnce and as an int.
    // Solwer for some reason
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    int* ind_diff = (int*)malloc((size_x_trn-1) * sizeof(int));

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        int label_test = y_tst[j]; // HERE IT DOES NOT MATTER? IF DOUBLE OR INT
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn-1; i++) {
            int ind_ip0 = (y_trn[x_test_knn_gt[j*size_x_trn+i+0]] == label_test); // These should be ints
            int ind_ip1 = (y_trn[x_test_knn_gt[j*size_x_trn+i+1]] == label_test);
            ind_diff[i] = ind_ip0 - ind_ip1;
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i+1]];
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (ind_diff[i] * Kidx_const[i]);  
        }
    }
}



void single_unweighted_knn_class_shapley_opt5(void *context_ptr){
    // opt5: Precompute yellow part and orange part (only need to calcualte indicator for each i once)
    // We directely comput the differnce and as an int.
    // Unrolling for reuse is faster but still slower than opt4
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    int* ind_diff = (int*)malloc((size_x_trn-1) * sizeof(int));

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        int label_test = y_tst[j];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn-1; i+=24) {
            int ind_ip0 = (y_trn[x_test_knn_gt[j*size_x_trn+i+0]] == label_test);
            int ind_ip1 = (y_trn[x_test_knn_gt[j*size_x_trn+i+1]] == label_test);
            int ind_ip2 = (y_trn[x_test_knn_gt[j*size_x_trn+i+2]] == label_test);
            int ind_ip3 = (y_trn[x_test_knn_gt[j*size_x_trn+i+3]] == label_test);
            int ind_ip4 = (y_trn[x_test_knn_gt[j*size_x_trn+i+4]] == label_test);
            int ind_ip5 = (y_trn[x_test_knn_gt[j*size_x_trn+i+5]] == label_test);
            int ind_ip6 = (y_trn[x_test_knn_gt[j*size_x_trn+i+6]] == label_test);
            int ind_ip7 = (y_trn[x_test_knn_gt[j*size_x_trn+i+7]] == label_test);
            int ind_ip8 = (y_trn[x_test_knn_gt[j*size_x_trn+i+8]] == label_test);
            int ind_ip9 = (y_trn[x_test_knn_gt[j*size_x_trn+i+9]] == label_test);
            int ind_ip10 = (y_trn[x_test_knn_gt[j*size_x_trn+i+10]] == label_test);
            int ind_ip11 = (y_trn[x_test_knn_gt[j*size_x_trn+i+11]] == label_test);
            int ind_ip12 = (y_trn[x_test_knn_gt[j*size_x_trn+i+12]] == label_test);
            int ind_ip13 = (y_trn[x_test_knn_gt[j*size_x_trn+i+13]] == label_test);
            int ind_ip14 = (y_trn[x_test_knn_gt[j*size_x_trn+i+14]] == label_test);
            int ind_ip15 = (y_trn[x_test_knn_gt[j*size_x_trn+i+15]] == label_test);
            int ind_ip16 = (y_trn[x_test_knn_gt[j*size_x_trn+i+16]] == label_test);
            int ind_ip17 = (y_trn[x_test_knn_gt[j*size_x_trn+i+17]] == label_test);
            int ind_ip18 = (y_trn[x_test_knn_gt[j*size_x_trn+i+18]] == label_test);
            int ind_ip19 = (y_trn[x_test_knn_gt[j*size_x_trn+i+19]] == label_test);
            int ind_ip20 = (y_trn[x_test_knn_gt[j*size_x_trn+i+20]] == label_test);
            int ind_ip21 = (y_trn[x_test_knn_gt[j*size_x_trn+i+21]] == label_test);
            int ind_ip22 = (y_trn[x_test_knn_gt[j*size_x_trn+i+22]] == label_test);
            int ind_ip23 = (y_trn[x_test_knn_gt[j*size_x_trn+i+23]] == label_test);
            int ind_ip24 = (y_trn[x_test_knn_gt[j*size_x_trn+i+24]] == label_test);
            int ind_ip25 = (y_trn[x_test_knn_gt[j*size_x_trn+i+25]] == label_test);


            ind_diff[i+0] = ind_ip0 - ind_ip1;
            ind_diff[i+1] = ind_ip1 - ind_ip2;
            ind_diff[i+2] = ind_ip2 - ind_ip3;
            ind_diff[i+3] = ind_ip3 - ind_ip4;
            ind_diff[i+4] = ind_ip4 - ind_ip5;
            ind_diff[i+5] = ind_ip5 - ind_ip6;
            ind_diff[i+6] = ind_ip6 - ind_ip7;
            ind_diff[i+7] = ind_ip7 - ind_ip8;
            ind_diff[i+8] = ind_ip8 - ind_ip9;
            ind_diff[i+9] = ind_ip9 - ind_ip10;
            ind_diff[i+10] = ind_ip10 - ind_ip11;
            ind_diff[i+11] = ind_ip11 - ind_ip12;
            ind_diff[i+12] = ind_ip12 - ind_ip13;
            ind_diff[i+13] = ind_ip13 - ind_ip14;
            ind_diff[i+14] = ind_ip14 - ind_ip15;
            ind_diff[i+15] = ind_ip15 - ind_ip16;
            ind_diff[i+16] = ind_ip16 - ind_ip17;
            ind_diff[i+17] = ind_ip17 - ind_ip18;
            ind_diff[i+18] = ind_ip18 - ind_ip19;
            ind_diff[i+19] = ind_ip19 - ind_ip20;
            ind_diff[i+20] = ind_ip20 - ind_ip21;
            ind_diff[i+21] = ind_ip21 - ind_ip22;
            ind_diff[i+22] = ind_ip22 - ind_ip23;
            ind_diff[i+23] = ind_ip23 - ind_ip24;
            ind_diff[i+24] = ind_ip24 - ind_ip25;
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i+1]];
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (ind_diff[i] * Kidx_const[i]);  
        }
    }
}


void single_unweighted_knn_class_shapley_opt(void *context_ptr){
    // opt6: Precompute yellow part and orange part (only need to calcualte indicator for each i once)
    // We directely comput the differnce and as an int.
    // Faster than 24 unrolling, but still slower than opt4
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    int* ind_diff = (int*)malloc((size_x_trn-1) * sizeof(int));

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        int label_test = y_tst[j];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn-1; i+=8) {
            int ind_ip0 = (y_trn[x_test_knn_gt[j*size_x_trn+i+0]] == label_test);
            int ind_ip1 = (y_trn[x_test_knn_gt[j*size_x_trn+i+1]] == label_test);
            int ind_ip2 = (y_trn[x_test_knn_gt[j*size_x_trn+i+2]] == label_test);
            int ind_ip3 = (y_trn[x_test_knn_gt[j*size_x_trn+i+3]] == label_test);
            int ind_ip4 = (y_trn[x_test_knn_gt[j*size_x_trn+i+4]] == label_test);
            int ind_ip5 = (y_trn[x_test_knn_gt[j*size_x_trn+i+5]] == label_test);
            int ind_ip6 = (y_trn[x_test_knn_gt[j*size_x_trn+i+6]] == label_test);
            int ind_ip7 = (y_trn[x_test_knn_gt[j*size_x_trn+i+7]] == label_test);
            int ind_ip8 = (y_trn[x_test_knn_gt[j*size_x_trn+i+8]] == label_test);
            int ind_ip9 = (y_trn[x_test_knn_gt[j*size_x_trn+i+9]] == label_test);

            ind_diff[i+0] = ind_ip0 - ind_ip1;
            ind_diff[i+1] = ind_ip1 - ind_ip2;
            ind_diff[i+2] = ind_ip2 - ind_ip3;
            ind_diff[i+3] = ind_ip3 - ind_ip4;
            ind_diff[i+4] = ind_ip4 - ind_ip5;
            ind_diff[i+5] = ind_ip5 - ind_ip6;
            ind_diff[i+6] = ind_ip6 - ind_ip7;
            ind_diff[i+7] = ind_ip7 - ind_ip8;
            ind_diff[i+8] = ind_ip8 - ind_ip9;
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i+1]];
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (ind_diff[i] * Kidx_const[i]);  
        }
    }
}



void single_unweighted_knn_class_shapley_opt7(void *context_ptr){
    // opt7: Part-Vectorized version opt3
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    int* ind_sub = (int*)malloc((size_x_trn) * sizeof(int));

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j++) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        int label_test = y_tst[j]; // DOES NOT MATTER IF INT OR DOUBLE (no performance hit)
        __m256 labels_test = _mm256_set_ps(label_test,label_test,label_test,label_test,
                                            label_test,label_test,label_test,label_test);
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp * inv_size_x_trn; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn; i+=8) {
            __m256 labels = _mm256_loadu_ps((int*)(y_trn + x_test_knn_gt[j*size_x_trn+i]));
            __m256 indicators = _mm256_cmp_ps(labels, labels_test, _CMP_EQ_OQ);
            _mm256_storeu_ps(ind_sub + i, indicators);

            print_vec_s(labels);
            /*
            print_vec(indicators);
            ind_sub[i] = (y_trn[x_test_knn_gt[j*size_x_trn+i]] == label_test);
            printf("%f\n", ind_sub[i]);
            exit;
            */
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i+1]];
            int difference = ind_sub[i] - ind_sub[i+1];
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[i]);  
        }
    }
}
