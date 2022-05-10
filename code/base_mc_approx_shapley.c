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
#include "knn_approx.h"

// Use "make debug" to enable debug prints and debug symbols, etc.
#ifdef DEBUG
    #define debug_print(fmt, ...) \
                do { fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#else
    #define debug_print(fmt, ...) 
#endif

// randomly permutes an array [1, ..., n] in place
void fisher_yates_shuffle(int* seq, int n) {
    for (int i = 0; i < n; i++) {
        seq[i] = i+1;
    }

    for (int i = n-1; i>=0; i--) {
        int j = rand() % (i+1);
        int temp = seq[i];
        seq[i] = seq[j];
        seq[j] = temp;
    }
}

void compute_shapley_using_improved_mc_approach(void *context) {

    return;
    context_t *ctx = (context_t *)context; 
    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle(pi, ctx->size_x_trn);

            // nearest neighbor in set of training points pi_0 to pi_i
            int nn = -1;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < ctx->size_x_trn; i++) {
                
                // check if pi_i is the new nearest neighbor (only then it changes the test accuracy)
                if (nn == -1 || ctx->dist_gt[j*ctx->size_x_trn+pi[i]] < ctx->dist_gt[j*ctx->size_x_trn+nn]) {
                    double v_incl_i = (double)(ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = (nn == -1) ? 0.0 : (double)(ctx->y_trn[nn] == ctx->y_tst[j]);
                    phi[t*ctx->size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    nn = pi[i];
                } else {
                    phi[t*ctx->size_x_trn+pi[i]] = phi[t*ctx->size_x_trn+pi[i-1]];
                }
            }
        }
        for (int i = 0; i < ctx->size_x_trn; i++) {
            ctx->sp_gt[j*ctx->size_x_trn+i] = 0;
            for (int t = 0; t < ctx->T; t++) {
                ctx->sp_gt[j*ctx->size_x_trn+i] += phi[t*ctx->size_x_trn+i];
            }
            ctx->sp_gt[j*ctx->size_x_trn+i] /= (double)(ctx->T);
        }
    }

    free(phi);
    free(pi);

    debug_print("%s", "Approx: Got Shapley done :)\n");
}

uint64_t run_approx_shapley(void *context) {
    context_t *ctx = (context_t *)context;


    #ifdef DEBUG
    //Sanity check, to make sure that C and Python are doing the same thing
    debug_print("%s", "x_trn:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", ctx->x_trn[i*ctx->feature_len+j]);
        }
        debug_print("%s", "\n");
    }

   debug_print("%s", "\n");
   debug_print("%s", "x_tst:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", ctx->x_tst[i*ctx->feature_len+j]);
        }
        debug_print("%s", "\n");
    }

    debug_print("%s", "\n");
    debug_print("%s", "y_tst:\n");
    for (int i = 0; i<ctx->size_y_tst;i++) {
        debug_print("%f, ", ctx->y_tst[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");
    #endif

    uint64_t start_timer, end_timer;

    start_timer = start_tsc();
    get_true_approx_KNN(ctx);

    #ifdef DEBUG
    debug_print("%s", "\n");
    debug_print("%s", "X_tst_dist_gt_array:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%f, ", ctx->x_test_knn_gt[i*ctx->size_x_trn+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    
    compute_shapley_using_improved_mc_approach(ctx);
    end_timer = stop_tsc(start_timer);

    #ifdef DEBUG
    // print sp_gt array
    debug_print("%s", "\n");
    debug_print("%s", "Shapley Values:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%f, ", ctx->sp_gt[i*ctx->size_x_trn+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    return end_timer;
}

void opt1_compute_shapley_using_improved_mc_approach(void *context) {

    context_t *ctx = (context_t *)context; 
    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle(pi, ctx->size_x_trn);

            // nearest neighbor in set of training points pi_0 to pi_i
            int nn = -1;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < ctx->size_x_trn; i++) {
                
                // check if pi_i is the new nearest neighbor (only then it changes the test accuracy)
                if (nn == -1 || ctx->dist_gt[j*ctx->size_x_trn+pi[i]] < ctx->dist_gt[j*ctx->size_x_trn+nn]) {
                    double v_incl_i = (double)(ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = (nn == -1) ? 0.0 : (double)(ctx->y_trn[nn] == ctx->y_tst[j]);
                    phi[t*ctx->size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    nn = pi[i];
                } else {
                    phi[t*ctx->size_x_trn+pi[i]] = phi[t*ctx->size_x_trn+pi[i-1]];
                }
            }
        }
        for (int i = 0; i < ctx->size_x_trn; i++) {
            ctx->sp_gt[j*ctx->size_x_trn+i] = 0;
            for (int t = 0; t < ctx->T; t++) {
                ctx->sp_gt[j*ctx->size_x_trn+i] += phi[t*ctx->size_x_trn+i];
            }
            ctx->sp_gt[j*ctx->size_x_trn+i] /= (double)(ctx->T);
        }
    }

    free(phi);
    free(pi);

    debug_print("%s", "Approx: Got Shapley done :)\n");
}