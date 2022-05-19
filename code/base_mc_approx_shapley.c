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

// size of the max heap
int size;

void swap(int *a, int *b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

void heapify(int heap[], int i) {
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    int largest = i;
    if (l < size && heap[l] > heap[largest])
        largest = l;
    if (r < size && heap[r] > heap[largest])
        largest = r;
    if (largest != i) {
        swap(&heap[i], &heap[largest]);
        heapify(heap, largest);
    }
}

// Inserts value into the max heap (if full, it replaces the root with the new value)
void insert(int heap[], int val, int K) {
    if (size == 0) {
        heap[0] = val;
        size = 1;
    } else if (size < K) {
        int index = size;
        heap[index] = val;
        size += 1;

        while(index != 0 && heap[(index-1)/2] < heap[index]) {
            swap(&heap[(index-1)/2], &heap[index]);
            index = (index-1)/2;
        }
    } else {
        heap[0] = val;
        heapify(heap, 0);
    }
}

void printArray(int array[], int n) {
  for (int i = 0; i < n; ++i)
    printf("%d ", array[i]);
  printf("\n");
}

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

    int K = (int)ctx->K;
    size_t size_x_trn = ctx->size_x_trn;
    size_t size_x_tst = ctx->size_x_tst;

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle(pi, size_x_trn);

            int maxheap[K];
            size = 0;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < size_x_trn; i++) {
                // check if pi_i is the a nearest neighbor (only then it changes the test accuracy)
                int dist_new = ctx->x_test_knn_r_gt[j*size_x_trn+pi[i]];
 
                int max_dist = maxheap[0];
                if (size < K) {
                    int sum = 0;
                    for (int k = 0; k < size; k++) {
                        sum += (int)(ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+maxheap[k]]] == ctx->y_tst[j]);
                    }
                    double sum_d = (double)sum;

                    double v_incl_i = sum_d + (ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = sum_d;

                    v_incl_i /= (double)(size+1);
                    if (size > 0) v_excl_i /= (double)size;

                    phi[t*size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    insert(maxheap, ctx->x_test_knn_r_gt[j*size_x_trn+pi[i]], K);

                } else if (dist_new < max_dist) {
                    int sum = 0;
                    for (int k = 1; k < size; k++) {
                        sum += (int)(ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+maxheap[k]]] == ctx->y_tst[j]);
                    }
                    double sum_d = (double)sum;

                    double v_incl_i = sum_d + (ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    double v_excl_i = sum_d + (ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+max_dist]] == ctx->y_tst[j]);

                    phi[t*size_x_trn+pi[i]] = (v_incl_i - v_excl_i) / (double)K;
                    insert(maxheap, ctx->x_test_knn_r_gt[j*size_x_trn+pi[i]], K);

                } else {
                    phi[t*size_x_trn+pi[i]] = 0;
                }
            }
        }
    
        for (int i = 0; i < size_x_trn; i++) {
            ctx->sp_gt[j*size_x_trn+i] = 0;
            for (int t = 0; t < ctx->T; t++) {
                ctx->sp_gt[j*size_x_trn+i] += phi[t*size_x_trn+i];
            }
            ctx->sp_gt[j*size_x_trn+i] /= (double)(ctx->T);
        }
    }

    free(phi);
    free(pi);

    debug_print("%s", "\nApprox: Got Shapley done :)\n\n");
    return;
}

void compute_shapley_using_improved_mc_approach_K1(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));
    
    srand(0);
    debug_print("T is: %d\n", ctx->T);
    debug_print("K is: %d\n", (int)ctx->K);
    debug_print("size_x_trn is: %ld\n", ctx->size_x_trn);
    debug_print("size_x_tst is: %ld\n\n", ctx->size_x_tst);

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
                if (nn == -1 || ctx->x_test_knn_r_gt[j*ctx->size_x_trn+pi[i]] < ctx->x_test_knn_r_gt[j*ctx->size_x_trn+nn]) {
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
    opt1_compute_shapley_using_improved_mc_approach(ctx);
    end_timer = stop_tsc(start_timer);

    return end_timer;
}

void opt1_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));
    
    srand(0);

    debug_print("T is: %d\n", ctx->T);
    debug_print("K is: %d\n", (int)ctx->K);
    debug_print("size_x_trn is: %ld\n", ctx->size_x_trn);
    debug_print("size_x_tst is: %ld\n\n", ctx->size_x_tst);

    int K = (int)ctx->K;
    int T = ctx->T;
    size_t size_x_trn = ctx->size_x_trn;
    size_t size_x_tst = ctx->size_x_tst;


    // calculate the shapley values for each test point j
    for (int j = 0; j < size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < T; t++) {

            fisher_yates_shuffle(pi, size_x_trn);
            int maxheap[K];
            size = 0;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < ctx->size_x_trn; i++) {
                // check if pi_i is the a nearest neighbor (only then it changes the test accuracy)
                int dist_new = ctx->x_test_knn_r_gt[j*size_x_trn+pi[i]];
                int max_dist = maxheap[0];
                if (size < K || dist_new < max_dist) {

                    double min_K_size1 = (K < size+1) ? K : size+1;
                    int sum = 0; // sum of training points in max heap except greatest one

                    for (int k = 1; k < size; k++) {
                        sum += (int)(ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+maxheap[k]]] == ctx->y_tst[j]);
                    }

                    double sum_d = (double)sum;
                    double v_incl_i = sum_d + (ctx->y_trn[pi[i]] == ctx->y_tst[j]);
                    if (size < K && size > 0) {
                        v_incl_i += ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+max_dist]] == ctx->y_tst[j];
                    }

                    double v_excl_i = (size == 0) ? 0.0 : (sum_d + (ctx->y_trn[ctx->x_test_knn_gt[j*size_x_trn+max_dist]] == ctx->y_tst[j]));
                    
                    v_incl_i /= min_K_size1;
                    v_excl_i /= (size == 0) ? 1 : (double)size;

                    phi[t*size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    insert(maxheap, ctx->x_test_knn_r_gt[j*size_x_trn+pi[i]], K);
                } else {
                    phi[t*size_x_trn+pi[i]] = 0;
                }
            }
        }
    
        for (int i = 0; i < size_x_trn; i++) {
            ctx->sp_gt[j*size_x_trn+i] = 0;
            for (int t = 0; t < ctx->T; t++) {
                ctx->sp_gt[j*size_x_trn+i] += phi[t*size_x_trn+i];
            }
            ctx->sp_gt[j*size_x_trn+i] /= (double)(ctx->T);
        }
    }

    #ifdef DEBUG
    for (int i = 0; i < size_x_trn; i++) {
        double sum = 0;
        for (int j = 0; j < size_x_tst; j++) {
            sum += ctx->sp_gt[j*size_x_trn+i];
        }
        debug_print("SV of training point %d is %f\n", i, sum / size_x_tst);
    }
    #endif

    free(phi);
    free(pi);

    debug_print("%s", "\nApprox: Got Base Shapley done :)\n\n");
    return;
}