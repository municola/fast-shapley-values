#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#include "tsc_x86.h"
#include "io.h"
#include "pcg.h"
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

static inline uint32_t pcg32_random_bounded_divisionless_with_slight_bias(uint32_t range) {
    uint64_t random32bit, multiresult;
    random32bit =  pcg32_random();
    multiresult = random32bit * range;
    return multiresult >> 32;
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

// randomly permutes an array [1, ..., n] in place
void fisher_yates_shuffle_fast(uint32_t* seq, int n) {
    for (int i = 0; i < n; i++) {
        seq[i] = i;
    }
    for (uint32_t i = n-1; i>=0; i--) {
        uint32_t j = pcg32_random_bounded_divisionless_with_slight_bias(i+1);
        uint32_t temp = seq[i];
        uint32_t val = seq[j];
        seq[i] = val;
        seq[j] = temp;
    }
}

void compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));

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

    debug_print("%s", "\nApprox: Got Shapley done :)\n\n");
    return;
}

void compute_shapley_using_improved_mc_approach_K1(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));
    
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
    //srand(0);
    get_true_approx_KNN(ctx);
    start_timer = start_tsc();
    current_compute_shapley_using_improved_mc_approach(ctx);
    end_timer = stop_tsc(start_timer);

    return end_timer;
}

void opt1_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));

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

void opt2_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;

    int* pi = (int*)calloc(ctx->size_x_trn, sizeof(int));
    double* phi = (double*)calloc(ctx->size_x_trn * ctx->T, sizeof(double));

    int K = (int)ctx->K;
    size_t size_x_trn = ctx->size_x_trn;
    size_t size_x_tst = ctx->size_x_tst;

    // calculate the shapley values for each test point j
    for (int j = 0; j < ctx->size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < ctx->T; t++) {

            fisher_yates_shuffle_fast(pi, size_x_trn);

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

    return;
}

void opt3_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;
    const int K = (int)ctx->K;
    const int T = (int)ctx->T;
    const size_t size_x_trn = ctx->size_x_trn;
    const size_t size_x_tst = ctx->size_x_tst;
    double* y_trn = ctx->y_trn;
    double* y_tst = ctx->y_tst;
    double* sp_gt = ctx->sp_gt;
    int* x_test_knn_r_gt = ctx->x_test_knn_r_gt;
    int* x_test_knn_gt = ctx->x_test_knn_gt;
    uint32_t* pi = (uint32_t*)calloc(size_x_trn, sizeof(uint32_t));
    double* phi = (double*)calloc(size_x_trn * T, sizeof(double));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];
        for (int t = 0; t < T; t++) {

            for (uint32_t i = 0; i < size_x_trn; i++) {
                pi[i] = i;
            }

            for (uint32_t i = size_x_trn-1; i>=0; i--) {
                uint32_t j = pcg32_random_bounded_divisionless_with_slight_bias(i+1);
                uint32_t temp = pi[i];
                uint32_t val = pi[j];
                pi[i] = val;
                pi[j] = temp;
            }

            int maxheap[K];
            size = 1;
            int i;
            int pi_0 = pi[0];

            phi[pi_0*T+t] = y_trn[pi_0] == y_tst_j;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int max_dist = maxheap[0];

                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += y_trn[x_test_knn_gt[j*size_x_trn+maxheap[k]]] == y_tst_j;
                }
                phi[t*size_x_trn+pi_i] = (size*(y_trn[pi_i] == y_tst_j) - sum) / (double)(size*(size+1));

                int index = size;
                maxheap[index] = dist_new;
                size += 1;

                int parent = (index-1) >> 1;
                int parent_val = maxheap[parent];

                while(index != 0 && parent_val < dist_new) {
                    maxheap[parent] = dist_new;
                    maxheap[index] = parent_val;
                    index = parent;
                    parent = (index-1) >> 1;
                    parent_val = maxheap[parent];
                }
            }

            int max_dist = maxheap[0];
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    int v_incl_i = y_trn[pi_i] == y_tst_j;
                    int v_excl_i = y_trn[x_test_knn_gt[j*size_x_trn+max_dist]] == y_tst_j;

                    phi[t*size_x_trn+pi_i] = (v_incl_i - v_excl_i) * ONE_OVER_K;

                    maxheap[0] = dist_new;
                    heapify(maxheap, 0); // HERE
                    max_dist = maxheap[0];
                } else {
                    phi[t*size_x_trn+pi_i] = 0;
                }
            }
        }
    
        for (int i = 0; i < size_x_trn; i++) {
            double acc0 = 0;
            double acc1 = 0;
            double acc2 = 0;
            double acc3 = 0;
            double acc4 = 0;
            double acc5 = 0;
            double acc6 = 0;
            double acc7 = 0;
            int t;
            for (t = 0; t < T-8; t+=8) {
                acc0 += phi[t*size_x_trn+i];
                acc1 += phi[(t+1)*size_x_trn+i];
                acc2 += phi[(t+2)*size_x_trn+i];
                acc3 += phi[(t+3)*size_x_trn+i];
                acc4 += phi[(t+4)*size_x_trn+i];
                acc5 += phi[(t+5)*size_x_trn+i];
                acc6 += phi[(t+6)*size_x_trn+i];
                acc7 += phi[(t+7)*size_x_trn+i];
            }

            for (; t < T; t++) {
                acc0 += phi[t*size_x_trn+i];
            }

            sp_gt[j*size_x_trn+i] = (((acc0+acc1)+(acc2+acc3))+((acc4+acc5)+(acc6+acc7))) * ONE_OVER_T;
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

    debug_print("%s", "\nApprox: Got Shapley done :)\n\n");
    return;
}

void opt4_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;
    const int K = (int)ctx->K;
    const int T = (int)ctx->T;
    const size_t size_x_trn = ctx->size_x_trn;
    const size_t size_x_tst = ctx->size_x_tst;
    double* y_trn = ctx->y_trn;
    double* y_tst = ctx->y_tst;
    double* sp_gt = ctx->sp_gt;
    int* x_test_knn_r_gt = ctx->x_test_knn_r_gt;
    int* x_test_knn_gt = ctx->x_test_knn_gt;
    int* pi = (int*)calloc(size_x_trn, sizeof(int));
    double* phi = (double*)calloc(size_x_trn * T, sizeof(double));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];
        for (int t = 0; t < T; t++) {

            for (int i = 0; i < size_x_trn; i++) {
                pi[i] = i;
            }

            for (int i = size_x_trn-1; i>=0; i--) {
                int j = pcg32_random_bounded_divisionless_with_slight_bias(i+1);
                int temp = pi[i];
                int val = pi[j];
                pi[i] = val;
                pi[j] = temp;
            }

            int maxheap[K];
            size = 1;
            int i;
            int pi_0 = pi[0];

            phi[pi_0*T+t] = y_trn[pi_0] == y_tst_j;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int max_dist = maxheap[0];

                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += y_trn[x_test_knn_gt[j*size_x_trn+maxheap[k]]] == y_tst_j;
                }
                phi[pi_i*T+t] = (size*(y_trn[pi_i] == y_tst_j) - sum) / (double)(size*(size+1));

                int index = size;
                maxheap[index] = dist_new;
                size += 1;

                int parent = (index-1) >> 1;
                int parent_val = maxheap[parent];

                while(index != 0 && parent_val < dist_new) {
                    maxheap[parent] = dist_new;
                    maxheap[index] = parent_val;
                    index = parent;
                    parent = (index-1) >> 1;
                    parent_val = maxheap[parent];
                }
            }

            int max_dist = maxheap[0];
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    int v_incl_i = y_trn[pi_i] == y_tst_j;
                    int v_excl_i = y_trn[x_test_knn_gt[j*size_x_trn+max_dist]] == y_tst_j;

                    phi[pi_i*T+t] = (v_incl_i - v_excl_i) * ONE_OVER_K;

                    maxheap[0] = dist_new;
                    heapify(maxheap, 0); // HERE
                    max_dist = maxheap[0];
                } else {
                    phi[pi_i*T+t] = 0;
                }
            }
        }
    
        for (int i = 0; i < size_x_trn; i++) {
            double acc0 = 0;
            double acc1 = 0;
            double acc2 = 0;
            double acc3 = 0;
            double acc4 = 0;
            double acc5 = 0;
            double acc6 = 0;
            double acc7 = 0;
            int t;
            for (t = 0; t < T-8; t+=8) {
                acc0 += phi[i*T+t];
                acc1 += phi[i*T+t+1];
                acc2 += phi[i*T+t+2];
                acc3 += phi[i*T+t+3];
                acc4 += phi[i*T+t+4];
                acc5 += phi[i*T+t+5];
                acc6 += phi[i*T+t+6];
                acc7 += phi[i*T+t+7];
            }

            for (; t < T; t++) {
                acc0 += phi[i*T+t];
            }

            sp_gt[j*size_x_trn+i] = (((acc0+acc1)+(acc2+acc3))+((acc4+acc5)+(acc6+acc7))) * ONE_OVER_T;
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

    return;
}

void current_compute_shapley_using_improved_mc_approach(void *context) {
    context_t *ctx = (context_t *)context;
    const int K = (int)ctx->K;
    const int T = (int)ctx->T;
    const size_t size_x_trn = ctx->size_x_trn;
    const size_t size_x_tst = ctx->size_x_tst;
    double* y_trn = ctx->y_trn;
    double* y_tst = ctx->y_tst;
    double* sp_gt = ctx->sp_gt;
    int* x_test_knn_r_gt = ctx->x_test_knn_r_gt;
    int* x_test_knn_gt = ctx->x_test_knn_gt;
    int* pi = (int*)malloc(size_x_trn * sizeof(int));
    double* phi = (double*)malloc(size_x_trn * T * sizeof(double));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];
        for (int t = 0; t < T; t++) {
            /*
            __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            __m256i incr = _mm256_set1_epi32(0);
            for (int i = 0; i < size_x_trn; i+=8) {
                _mm256_store_epi32(pi, ind);
                ind = _mm256_add_epi32(ind, incr);
            }
            printf("blablalj aflk jlkdsj ");
            printArray(pi, size_x_trn);
            */

            for (int i = 0; i < size_x_trn; i++) {
                pi[i] = i;
            }

            for (int i = size_x_trn-1; i>=0; i--) {
                int j = pcg32_random_bounded_divisionless_with_slight_bias(i+1);
                int temp = pi[i];
                int val = pi[j];
                pi[i] = val;
                pi[j] = temp;
            }

            int maxheap[K];
            size = 1;
            int i;
            int pi_0 = pi[0];

            phi[pi_0*T+t] = y_trn[pi_0] == y_tst_j;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int max_dist = maxheap[0];

                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += y_trn[x_test_knn_gt[j*size_x_trn+maxheap[k]]] == y_tst_j;
                }
                phi[pi_i*T+t] = (size*(y_trn[pi_i] == y_tst_j) - sum) / (double)(size*(size+1));

                int index = size;
                maxheap[index] = dist_new;
                size += 1;

                int parent = (index-1) >> 1;
                int parent_val = maxheap[parent];

                while(index != 0 && parent_val < dist_new) {
                    maxheap[parent] = dist_new;
                    maxheap[index] = parent_val;
                    index = parent;
                    parent = (index-1) >> 1;
                    parent_val = maxheap[parent];
                }
            }

            int max_dist = maxheap[0];
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    int v_incl_i = y_trn[pi_i] == y_tst_j;
                    int v_excl_i = y_trn[x_test_knn_gt[j*size_x_trn+max_dist]] == y_tst_j;

                    phi[pi_i*T+t] = (v_incl_i - v_excl_i) * ONE_OVER_K;

                    maxheap[0] = dist_new;
                    heapify(maxheap, 0); // HERE
                    max_dist = maxheap[0];
                } else {
                    phi[pi_i*T+t] = 0;
                }
            }
        }
    
        for (int i = 0; i < size_x_trn; i++) {
            double acc0 = 0;
            double acc1 = 0;
            double acc2 = 0;
            double acc3 = 0;
            double acc4 = 0;
            double acc5 = 0;
            double acc6 = 0;
            double acc7 = 0;
            int t;
            for (t = 0; t < T-8; t+=8) {
                acc0 += phi[i*T+t];
                acc1 += phi[i*T+t+1];
                acc2 += phi[i*T+t+2];
                acc3 += phi[i*T+t+3];
                acc4 += phi[i*T+t+4];
                acc5 += phi[i*T+t+5];
                acc6 += phi[i*T+t+6];
                acc7 += phi[i*T+t+7];
            }

            for (; t < T; t++) {
                acc0 += phi[i*T+t];
            }

            sp_gt[j*size_x_trn+i] = (((acc0+acc1)+(acc2+acc3))+((acc4+acc5)+(acc6+acc7))) * ONE_OVER_T;
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

    return;
}