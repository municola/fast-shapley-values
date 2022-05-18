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

// randomly permutes an array [1, ..., n] in place
void fisher_yates_shuffle(int* seq, int n) {
    for (int i = 0; i < n; i++) {
        seq[i] = i;
    }

    for (int i = n-1; i>=0; i--) {
        int j = rand() % (i+1);
        int temp = seq[i];
        seq[i] = seq[j];
        seq[j] = temp;
    }
}

void compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));
    
    srand(0);
    debug_print("T is: %d\n", ctx->T);
    debug_print("K is: %d\n", ctx->K);
    debug_print("size_x_trn is: %d\n", ctx->size_x_trn);
    debug_print("size_x_tst is: %d\n\n", ctx->size_x_tst);

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle(pi, ctx->size_x_trn);

            int maxheap[ctx->K];
            size = 0;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < ctx->size_x_trn; i++) {
                // check if pi_i is the a nearest neighbor (only then it changes the test accuracy)
                if (size < (int)ctx->K || ctx->x_test_knn_gt[j*ctx->size_x_trn+pi[i]] < maxheap[0]) {
                    double v_incl_i = (double)(ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = (nn == -1) ? 0.0 : (double)(ctx->y_trn[nn] == ctx->y_tst[j]);
                    phi[t*ctx->size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    nn = pi[i];
                } else {
                    phi[t*ctx->size_x_trn+pi[i]] = 0;
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

    #ifdef DEBUG
    for (int i = 0; i < ctx->size_x_trn; i++) {
        double sum = 0;
        for (int j = 0; j < ctx->size_x_tst; j++) {
            sum += ctx->sp_gt[j*ctx->size_x_trn+i];
        }
        debug_print("SV of training point %d is %f\n", i, sum / ctx->size_x_tst);
    }
    #endif

    free(phi);
    free(pi);

    debug_print("%s", "\nApprox: Got Shapley done :)\n\n");
    return;
}

uint64_t run_approx_shapley(void *context) {
    context_t *ctx = (context_t *)context;

    uint64_t start_timer, end_timer;

    start_timer = start_tsc();
    get_true_approx_KNN(ctx);
    compute_shapley_using_improved_mc_approach(ctx);
    end_timer = stop_tsc(start_timer);

    return end_timer;
}

void opt1_compute_shapley_using_improved_mc_approach(void *context) {

    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));
    
    srand(0);
    debug_print("T is: %d\n", ctx->T);
    debug_print("K is: %d\n", ctx->K);
    debug_print("size_x_trn is: %d\n", ctx->size_x_trn);
    debug_print("size_x_tst is: %d\n\n", ctx->size_x_tst);

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle(pi, ctx->size_x_trn);

            int maxheap[ctx->K];
            size = 0;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < ctx->size_x_trn; i++) {
                // check if pi_i is the a nearest neighbor (only then it changes the test accuracy)
                if (size < (int)ctx->K || ctx->x_test_knn_gt[j*ctx->size_x_trn+pi[i]] < maxheap[0]) {
                    double v_incl_i = (double)(ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = (nn == -1) ? 0.0 : (double)(ctx->y_trn[nn] == ctx->y_tst[j]);
                    phi[t*ctx->size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    nn = pi[i];
                } else {
                    phi[t*ctx->size_x_trn+pi[i]] = 0;
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

    #ifdef DEBUG
    for (int i = 0; i < ctx->size_x_trn; i++) {
        double sum = 0;
        for (int j = 0; j < ctx->size_x_tst; j++) {
            sum += ctx->sp_gt[j*ctx->size_x_trn+i];
        }
        debug_print("SV of training point %d is %f\n", i, sum / ctx->size_x_tst);
    }
    #endif

    free(phi);
    free(pi);

    debug_print("%s", "\nApprox: Got Shapley done :)\n\n");
    return;
}