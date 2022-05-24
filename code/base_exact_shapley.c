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
    // compute_single_unweighted_knn_class_shapley(context);
    end_timer = stop_tsc(start_timer);

    printf("Cycles: %lu\n", end_timer);

    return end_timer;
}

void print_vec_s(__m256 var) {
    printf("%d %d %d %d %d %d %d %d \n", 
           var[0], var[1], var[2], var[3], var[4], var[5], var[6], var[7]);
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
        // printf("Exact: j=%d: Shapley sp_gt: input_size: %d \n", j, context->input_size);
        // for(int i=0; i<context->size_x_tst; i++){
        //     for(int j=0; j<context->size_x_trn; j++){
        //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
        //     }
        //     printf("\n");
        // }
        // printf("-------------------------------------------------------\n");
    }

    // print sp_gt array
    printf("Exact: COMPLETE Shapley sp_gt: input_size: %d \n", context->input_size);
    for(int i=0; i<context->size_x_tst; i++){
        for(int j=0; j<context->size_x_trn; j++){
            printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
        }
        printf("\n");
    }

    debug_print("%s", "Exact: Shapley Values done :)\n");
}

// Base transposed implementation (Also used for Correctness. Do not touch)
void compute_transposed_single_unweighted_knn_class_shapley(void *context_ptr){
    context_t *context = (context_t*)context_ptr;
    debug_print("%s", "\nStart Shapley computation:\n");
    for(int j=0; j<context->size_x_tst;j++){
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = context->x_test_knn_gt[j*context->size_x_trn+context->size_x_trn-1];
        double tmp = (context->y_trn[offset] == context->y_tst[j]) ? 1.0 : 0.0;
        context->sp_gt[offset*context->size_x_trn + j] = tmp / context->size_x_trn; 

        for (int i=context->size_x_trn-2; i>-1; i--) {
            // debug_print("    i=%d\n", i);
            int index_j_i = j*context->size_x_trn+i;

            double s_j_alpha_i_plus_1 = context->sp_gt[context->x_test_knn_gt[index_j_i+1]*context->size_x_trn + j];
            double difference = (double)(context->y_trn[context->x_test_knn_gt[index_j_i]] == context->y_tst[j]) - 
                                        (double)(context->y_trn[context->x_test_knn_gt[index_j_i+1]] == context->y_tst[j]);
            double min_K_i = context->K < i+1 ? context->K : i+1;

            // debug_print("      s_j=%f\n", s_j_alpha_i_plus_1);
            // debug_print("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // debug_print("      min_=%f\n", min_K_i);

            context->sp_gt[context->x_test_knn_gt[index_j_i]*context->size_x_trn + j] = s_j_alpha_i_plus_1 + (difference / context->K) * (min_K_i / (i+1));
        }
    }

    // //print sp_gt array
    // printf("Transposed: Shapley sp_gt:\n");
    // for(int i=0; i<context->size_x_tst; i++){
    //     for(int j=0; j<context->size_x_trn; j++){
    //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
    //     }
    //     printf("\n");
    // }

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

        printf("Opt: j:%d ind_diff:\n", j);
        for(int i=0; i<context->size_x_trn-1; i++){
            printf("%d ", ind_diff[i]);
        }
        printf("\n");

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


void single_unweighted_knn_class_shapley_opt6(void *context_ptr){
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
    // opt7: Part-Vectorized version opt3 - (Currently not correct [Chris, 23.5.22, 13:08])
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

            print_vec_s(indicators);
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

void single_unweighted_knn_class_shapley_opt8(void *context_ptr){
    // opt8: based on opt3, but s_j_alpha_i_plus_1 can be cached for next iter
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)aligned_alloc(32, (size_x_trn-1) * sizeof(double));
    double* ind_sub = (double*)aligned_alloc(32, (size_x_trn) * sizeof(double));
    __m256d ones = _mm256_set1_pd(1.0);

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
        double s_j_alpha_i_plus_1 = tmp * inv_size_x_trn;
        sp_gt[j*size_x_trn + offset] = s_j_alpha_i_plus_1;; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn; i++) {
            ind_sub[i] = (y_trn[x_test_knn_gt[j*size_x_trn+i]] == label_test);
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            int difference = ind_sub[i] - ind_sub[i+1];
            s_j_alpha_i_plus_1 += (difference * Kidx_const[i]);
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1;;  
        }
    }

     // print sp_gt array
    // printf("Opt: Shapley sp_gt: input_size: %d \n", context->input_size);
    // for(int i=0; i<context->size_x_tst; i++){
    //     for(int j=0; j<context->size_x_trn; j++){
    //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
    //     }
    //     printf("\n");
    // }

}

void single_unweighted_knn_class_shapley_opt9(void *context_ptr){
    // Not working
    // opt9: partial vectorization of opt8
    // by viewing assembly, it becomes visible, that the compiler does not use
    // FMA to calculate the new value of s_j_alpha_i_plus_1
    // The idea is to force FMA using intrinsics
    // We calculate 4 test points together, since these are aligned in memory
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)aligned_alloc(32, (size_x_trn-1) * sizeof(double));
    double* ind_sub = (double*)aligned_alloc(32, (size_x_trn) * sizeof(double));
    __m256d ones = _mm256_set1_pd(1.0);

    // Precompute the constant part from Line 5 in the Shapley algorithm
    for (int i=1; i<size_x_trn; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for(int j=0; j<context->size_x_tst;j+=4) {
        // Line 3 of Algo 1
        int offset = x_test_knn_gt[j*size_x_trn+size_x_trn-1];
        printf("Opt: iter %d, offset:%d\n", j, offset);
        // double label_test = y_tst[j]; 
        // load 4 test labels at a time
        __m256d labels_test = _mm256_load_pd(&y_tst[j]);
        
        // Line 3: Set furthest point to (ind / N) for all 4 test points
        __m256d curr_y_train = _mm256_set1_pd(y_trn[offset]);
        __m256d mask_compare = _mm256_cmp_pd(curr_y_train, labels_test, _CMP_EQ_OQ);
        __m256d packed_indicator_variable_n = _mm256_and_pd(mask_compare, ones);
        __m256d packed_s_j_alpha_i_plus_1 = _mm256_mul_pd(packed_indicator_variable_n, _mm256_set1_pd(inv_size_x_trn));

        // double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        // double s_j_alpha_i_plus_1 = tmp * inv_size_x_trn;
        //save to array
        sp_gt[(j+0)*size_x_trn + offset] = packed_s_j_alpha_i_plus_1[0];
        sp_gt[(j+1)*size_x_trn + offset] = packed_s_j_alpha_i_plus_1[1];
        sp_gt[(j+2)*size_x_trn + offset] = packed_s_j_alpha_i_plus_1[2];
        sp_gt[(j+3)*size_x_trn + offset] = packed_s_j_alpha_i_plus_1[3];

        // Line 4: Loop, iterate with stride 1 as we must go through every train point, but do this for 4 test points at a time
        __m256d indicator_alpha_plus_one;
        for (int i=size_x_trn-2; i>-1; i--) {
            __m256d curr_y_train = _mm256_set1_pd(y_trn[x_test_knn_gt[j*size_x_trn+i]]);
            // printf("Opt: curr_y_train:\n");
            // for(int i=0; i<4; i++){
            //     printf("%f ", curr_y_train[i]);
            // }
            // printf("\n");
            __m256d mask_compare = _mm256_cmp_pd(curr_y_train, labels_test, _CMP_EQ_OQ);
            indicator_alpha_plus_one = _mm256_and_pd(mask_compare, ones);
            __m256d differences = _mm256_sub_pd(indicator_alpha_plus_one, packed_indicator_variable_n);
            __m256d packed_s_j_alpha_i_plus_1 = _mm256_fmadd_pd(differences, _mm256_set1_pd(Kidx_const[i]), packed_s_j_alpha_i_plus_1);

            packed_indicator_variable_n = indicator_alpha_plus_one;

            sp_gt[(j+0)*size_x_trn + x_test_knn_gt[(j+0)*size_x_trn+i]] = packed_s_j_alpha_i_plus_1[0];  
            sp_gt[(j+1)*size_x_trn + x_test_knn_gt[(j+1)*size_x_trn+i]] = packed_s_j_alpha_i_plus_1[1];  
            sp_gt[(j+2)*size_x_trn + x_test_knn_gt[(j+2)*size_x_trn+i]] = packed_s_j_alpha_i_plus_1[2];  
            sp_gt[(j+3)*size_x_trn + x_test_knn_gt[(j+3)*size_x_trn+i]] = packed_s_j_alpha_i_plus_1[3];  

            // int difference = ind_sub[i] - ind_sub[i+1];
            // s_j_alpha_i_plus_1 += (difference * Kidx_const[i]);
            // sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1;;  
        }
    }

    //  print sp_gt array
    printf("Opt: Shapley sp_gt: input_size: %d \n", context->input_size);
    for(int i=0; i<context->size_x_tst; i++){
        for(int j=0; j<context->size_x_trn; j++){
            printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
        }
        printf("\n");
    }

    printf("Opt: y_test:\n");
    for(int i=0; i<context->size_x_tst; i++){
        printf("%f ", context->y_tst[i]);
    }
    printf("\n");

    printf("Opt: y_train:\n");
    for(int i=0; i<context->size_x_tst; i++){
        printf("%f ", context->y_trn[i]);
    }
    printf("\n");
}

void single_unweighted_knn_class_shapley_opt10(void *context_ptr){
    // opt10: based on opt3, but precompute ind_sub

    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)aligned_alloc(32, (size_x_trn-1) * sizeof(double));
    double* ind_sub = (double*)aligned_alloc(32, (size_x_trn) * sizeof(double));
    __m256d ones = _mm256_set1_pd(1.0);

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

      // Precompute the Indicator subtraction
        // double ind0 = (y_trn[k] == label_test) ? 1.0 : 0.0;
        double ind_ip0 = (y_trn[x_test_knn_gt[j*size_x_trn]] == label_test) ? 1.0 : 0.0; // These should be ints
        for (int i=0; i<size_x_trn-1; i++) {
            double ind_ip1 = (y_trn[x_test_knn_gt[j*size_x_trn+i+1]] == label_test) ? 1.0 : 0.0;
            ind_sub[i+0] = ind_ip0 - ind_ip1;
            ind_ip0 = ind_ip1;
        }

        // printf("Opt: j:%d ind_diff:\n", j);
        // for(int i=0; i<context->size_x_trn-1; i++){
        //     printf("%d ", ind_sub[i]);
        // }
        // printf("\n");
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        // double ind0 = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        double s_j_alpha_i_plus_1 = ind_ip0 * inv_size_x_trn;
        sp_gt[j*size_x_trn + offset] = s_j_alpha_i_plus_1;

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            s_j_alpha_i_plus_1 += (ind_sub[i] * Kidx_const[i]);
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1;
        }
    }

     // print sp_gt array
    // printf("Opt: Shapley sp_gt: input_size: %d \n", context->input_size);
    // for(int i=0; i<context->size_x_tst; i++){
    //     for(int j=0; j<context->size_x_trn; j++){
    //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
    //     }
    //     printf("\n");
    // }

}


void single_unweighted_knn_class_shapley_opt11(void *context_ptr){
    // opt11: based on opt8, but s_j_alpha_i_plus_1 can be cached for next iter
    // unroll ind_sub construction
    // Slower than opt8, since ind_sub is saved in memory
    // opt8 manages to optimize the ind_sub array away
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)aligned_alloc(32, (size_x_trn-1) * sizeof(double));
    double* ind_sub = (double*)aligned_alloc(32, (size_x_trn) * sizeof(double));

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

      // Precompute the Indicator subtraction
        // double ind0 = (y_trn[k] == label_test) ? 1.0 : 0.0;
        double ind_ip0 = (y_trn[x_test_knn_gt[j*size_x_trn]] == label_test) ? 1.0 : 0.0; // These should be ints
        for (int i=0; i<size_x_trn-1; i+=4) {
            double ind_ip1 = (y_trn[x_test_knn_gt[j*size_x_trn+i+1]] == label_test) ? 1.0 : 0.0;
            double ind_ip2 = (y_trn[x_test_knn_gt[j*size_x_trn+i+2]] == label_test) ? 1.0 : 0.0;
            double ind_ip3 = (y_trn[x_test_knn_gt[j*size_x_trn+i+3]] == label_test) ? 1.0 : 0.0;
            double ind_ip4 = (y_trn[x_test_knn_gt[j*size_x_trn+i+4]] == label_test) ? 1.0 : 0.0;

            ind_sub[i+0] = ind_ip0 - ind_ip1;
            ind_sub[i+1] = ind_ip1 - ind_ip2;
            ind_sub[i+2] = ind_ip2 - ind_ip3;
            ind_sub[i+3] = ind_ip3 - ind_ip4;
            ind_ip0 = ind_ip4;
        }

        // printf("Opt: j:%d ind_diff:\n", j);
        // for(int i=0; i<context->size_x_trn-1; i++){
        //     printf("%f ", ind_sub[i]);
        // }
        // printf("\n");
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double ind0 = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        double s_j_alpha_i_plus_1 = ind0 * inv_size_x_trn;
        sp_gt[j*size_x_trn + offset] = s_j_alpha_i_plus_1;

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            s_j_alpha_i_plus_1 += (ind_sub[i] * Kidx_const[i]);
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1;
        }
    }

     // print sp_gt array
    // printf("Opt: Shapley sp_gt: input_size: %d \n", context->input_size);
    // for(int i=0; i<context->size_x_tst; i++){
    //     for(int j=0; j<context->size_x_trn; j++){
    //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
    //     }
    //     printf("\n");
    // }

}


void single_unweighted_knn_class_shapley_opt(void *context_ptr){
    // opt12: based on opt3, unroll j loop
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)aligned_alloc(32, (size_x_trn-1) * sizeof(double));
    double* ind_sub = (double*)aligned_alloc(32, (size_x_trn) * sizeof(double));
    __m256d ones = _mm256_set1_pd(1.0);

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
        double s_j_alpha_i_plus_1 = tmp * inv_size_x_trn;
        sp_gt[j*size_x_trn + offset] = s_j_alpha_i_plus_1;; 

        // Precompute the Indicator subtraction
        for (int i=0; i<size_x_trn; i++) {
            ind_sub[i] = (y_trn[x_test_knn_gt[j*size_x_trn+i]] == label_test);
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            int difference = ind_sub[i] - ind_sub[i+1];
            s_j_alpha_i_plus_1 += (difference * Kidx_const[i]);
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1;;  
        }
    }

     // print sp_gt array
    // printf("Opt: Shapley sp_gt: input_size: %d \n", context->input_size);
    // for(int i=0; i<context->size_x_tst; i++){
    //     for(int j=0; j<context->size_x_trn; j++){
    //         printf("%f ", context->sp_gt[i*context->size_x_trn + j]);
    //     }
    //     printf("\n");
    // }

}

void single_transposed_unweighted_knn_class_shapley_opt(void *context_ptr){
    // opt9: Output representation was in unfavorable access pattern,
    // since the resulting sp_gt array needs to be averaged for every test_point
    // We output the sp_gt array in transposed fashion, such that 
    // all values concerning a single training point are in the contiguous dimension
    context_t *context = (context_t*)context_ptr;
    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_size_x_trn = 1.0/size_x_trn;
    double* Kidx_const = (double*)malloc((size_x_trn-1) * sizeof(double));
    float* ind_sub = (int*)malloc((size_x_trn) * sizeof(double));

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
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == label_test) ? 1.0 : 0.0;
        sp_gt[offset*size_x_trn + j] = tmp * inv_size_x_trn; 

         // Precompute the Indicator subtraction
         // compiler already vectorizes this and optimizes ind_sub away
        for (int i=0; i<size_x_trn; i++) {
            ind_sub[i] = (y_trn[x_test_knn_gt[j*size_x_trn+i]] == label_test);
        }

        // Actual Loop at line 4
        for (int i=size_x_trn-2; i>-1; i--) {
            double s_j_alpha_i_plus_1 = sp_gt[x_test_knn_gt[j*size_x_trn+i+1]*size_x_trn + j];
            int difference = ind_sub[i] - ind_sub[i+1];

            //This needs to be FMA'd in packed fashion
            sp_gt[j*size_x_trn + x_test_knn_gt[j*size_x_trn+i]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[i]);  
        }
    }
}