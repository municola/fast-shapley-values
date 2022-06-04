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
#include "simdxorshift128plus.h"
#include "utils.h"
#include "benchmark.h"
#include "knn_approx.h"

// size of the max heap
int size;

static inline void swap(int *a, int *b) {
    int temp = *b;
    *b = *a;
    *a = temp;
}

static inline void heapify(int *heap, int i) {
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

static void inline bh_swap(unsigned int index1, unsigned int index2, uint16_t *heap, void *pSwapMemory) {
    
    unsigned char *pElem1, *pElem2;
    
    pElem1 = ((unsigned char *) heap) + (sizeof(uint16_t) * index1);
    pElem2 = ((unsigned char *) heap) + (sizeof(uint16_t) * index2);
    
    memcpy((void *) pSwapMemory, (const void *) pElem1,                  sizeof(uint16_t));
    memcpy((void *) pElem1,            (const void *) pElem2,            sizeof(uint16_t));
    memcpy((void *) pElem2,            (const void *) pSwapMemory, sizeof(uint16_t));
}

static inline void bh_heapify(uint16_t *heap, unsigned int index, void *pSwapMemory) {
    
    unsigned int iLargest;
    unsigned int iLeft, iRight;

    unsigned char *pHeap, *pCurrent, *pLeft, *pRight, *pLargest;

    uint16_t heap_size = size;
    
    iLargest = index;
    unsigned int next_index = index;

    pHeap = (unsigned char*) heap;
    
    REPEAT:
    iLeft = 2 * next_index + 1;
    iRight = 2 * next_index + 2;

    pLeft = pHeap + (sizeof(uint16_t) * iLeft);
    pRight = pHeap + (sizeof(uint16_t) * iRight);
    pCurrent = pHeap + (sizeof(uint16_t) * next_index);
    pLargest = pCurrent;

    index = next_index;
    
    if (iLeft < heap_size && *(const uint16_t*) pLeft > *(const uint16_t*) pCurrent) {
        iLargest = iLeft;
        pLargest = pLeft;
    }

    if (iRight < heap_size && *(const uint16_t*) pRight > *(const uint16_t*) pLargest) {
        iLargest = iRight;
    }
    
    if (iLargest != index) {  
        bh_swap(index, iLargest, heap, pSwapMemory);
        next_index = iLargest;
        goto REPEAT;
    }
}

static inline void inlined_heapify(int maxheap[], int dist_new, int max_dist, int max_index, 
        int left_dist, int right_dist, int left_index, int right_index, uint16_t pi_i, int j, size_t size_x_trn, int* x_test_knn_gt) {

    if (dist_new < left_dist || dist_new < right_dist) {
        if (left_dist > right_dist) {
            max_dist = left_dist;
            max_index = left_index;
            // Heapify Left
            int left = 3 < size ? maxheap[3] : -1;
            int right = 4 < size ? maxheap[4] : -1;
            if (dist_new < left || dist_new < right) {
                if (left > right) {
                    left_dist = left;
                    left_index = x_test_knn_gt[j*size_x_trn+left];
                    maxheap[3] = dist_new;
                    heapify(maxheap, 3);
                } else {
                    left_dist = right;
                    left_index = x_test_knn_gt[j*size_x_trn+right];
                    maxheap[4] = dist_new;
                    heapify(maxheap, 4);
                }
            } else {
                left_dist = dist_new;
                left_index = pi_i;
            }
        } else {
            max_dist = right_dist;
            max_index = right_index;
            // Heapify Right
            int left = 5 < size ? maxheap[5] : -1;
            int right = 6 < size ? maxheap[6] : -1;
            if (dist_new < left || dist_new < right) {
                if (left > right) {
                    right_dist = left;
                    right_index = x_test_knn_gt[j*size_x_trn+left];
                    maxheap[5] = dist_new;
                    heapify(maxheap, 5);
                } else {
                    right_dist = right;
                    right_index = x_test_knn_gt[j*size_x_trn+right];
                    maxheap[6] = dist_new;
                    heapify(maxheap, 6);
                }
            } else {
                right_dist = dist_new;
                right_index = pi_i;
            }
        }
    } else {
        max_dist = dist_new;
        max_index = pi_i;
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
    for (int i = n-1; i>=0; i--) {
        int j = rand() % (i+1);
        int temp = seq[i];
        seq[i] = seq[j];
        seq[j] = temp;
    }

    for (int i = 0; i < n / 2; i++) {
        swap(&seq[i], &seq[n-1-i]);
    }
}

// randomly permutes an array [1, ..., n] in place
static inline void fisher_yates_shuffle_fast(int* seq, int n) {
    for (int i = n-1; i>=0; i--) {
        int j = pcg32_random_bounded_divisionless_with_slight_bias(i+1);
        int temp = seq[i];
        int val = seq[j];
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
            for (int p = 0; p < size_x_trn; p++) {
                pi[p] = p;
            }
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
            for (int p = 0; p < ctx->size_x_trn; p++) {
                pi[p] = p;
            }
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
    srand(0);
    knn__approx_opt5(ctx);
    start_timer = start_tsc();
    opt12_compute_shapley_using_improved_mc_approach(ctx);
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
            for (int p = 0; p < size_x_trn; p++) {
                pi[p] = p;
            }
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

// Fast rand
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
            
            for (int p = 0; p < size_x_trn; p++) {
                pi[p] = p;
            }

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

// Accumulators, Pre computing ecg
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

// Improved spatial locality for PHI (only good for K = 1)
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

// memcpy pi, get random numbers on the fly
void opt5_compute_shapley_using_improved_mc_approach(void *context) {
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
    double* phi = (double*)malloc(size_x_trn * T * sizeof(double));
    int* seq = (int*)malloc(size_x_trn * sizeof(int));
    int* pi = (int*)malloc(size_x_trn * sizeof(int));
    bool* trn_tst = (bool*)calloc(size_x_trn * size_x_tst, sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];

    for (int p = 0; p < size_x_trn; p++) {
        seq[p] = p;
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        // precompute comparison between training and test labels (will each be needed T times)
        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[j*size_x_trn+i] = y_trn[i] == y_tst_j;   
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            size = 1;
            double size_sqr_plus_size = 2;
            int i;

           // int pi_0 = rand() % size_x_trn;
            int pi_0 = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn);
            pi[pi_0] = size_x_trn-1;
            phi[t*size_x_trn+pi_0] = trn_tst[j*size_x_trn+pi_0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += trn_tst[j*size_x_trn+x_test_knn_gt[j*size_x_trn+maxheap[k]]];
                }
                phi[t*size_x_trn+pi_i] = (size*(trn_tst[j*size_x_trn+pi_i]) - sum) / size_sqr_plus_size;
                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
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
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[j*size_x_trn+pi_i];
                    int v_excl_i = trn_tst[j*size_x_trn+x_test_knn_gt[j*size_x_trn+max_dist]];
                    
                    maxheap[0] = max_dist;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];
                    phi[t*size_x_trn+pi_i] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
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
    free(seq);

    return;
}

// reduced memory accesses by keeping track of left and right children in heap and their indexes (only K >= 3)
void opt6_compute_shapley_using_improved_mc_approach(void *context) {
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
    double* phi = (double*)malloc(size_x_trn * T * sizeof(double));
    int* seq = (int*)malloc(size_x_trn * sizeof(int));
    int* pi = (int*)malloc(size_x_trn * sizeof(int));
    bool* trn_tst = (bool*)calloc(size_x_trn * size_x_tst, sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];

    assert(K >= 3);

    for (int p = 0; p < size_x_trn; p++) {
        seq[p] = p;
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[j*size_x_trn+i] = y_trn[i] == y_tst_j;   
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            //int pi_0 = rand() % size_x_trn;
            int pi_0 = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn);
            pi[pi_0] = size_x_trn-1;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            phi[t*size_x_trn+pi_0] = trn_tst[j*size_x_trn+pi_0];

            sum = trn_tst[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += trn_tst[j*size_x_trn+x_test_knn_gt[j*size_x_trn+maxheap[k]]];
                }
                phi[t*size_x_trn+pi_i] = (size*(trn_tst[j*size_x_trn+pi_i]) - sum) / size_sqr_plus_size;
                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
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
            int left_dist = maxheap[1];
            int right_dist = maxheap[2];
            int max_index = x_test_knn_gt[j*size_x_trn+max_dist];
            int left_index = x_test_knn_gt[j*size_x_trn+left_dist];
            int right_index = x_test_knn_gt[j*size_x_trn+right_dist];

            for (; i < size_x_trn; i++) {
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[j*size_x_trn+pi_i];
                    int v_excl_i = trn_tst[j*size_x_trn+max_index];

                    // Heapify
                    if (dist_new < left_dist || dist_new < right_dist) {
                        if (left_dist > right_dist) {
                            max_dist = left_dist;
                            max_index = left_index;
                            // Heapify Left
                            int left = 3 < size ? maxheap[3] : -1;
                            int right = 4 < size ? maxheap[4] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    left_dist = left;
                                    left_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[3] = dist_new;
                                    heapify(maxheap, 3);
                                } else {
                                    left_dist = right;
                                    left_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[4] = dist_new;
                                    heapify(maxheap, 4);
                                }
                            } else {
                                left_dist = dist_new;
                                left_index = pi_i;
                            }
                        } else {
                            max_dist = right_dist;
                            max_index = right_index;
                            // Heapify Right
                            int left = 5 < size ? maxheap[5] : -1;
                            int right = 6 < size ? maxheap[6] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    right_dist = left;
                                    right_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[5] = dist_new;
                                    heapify(maxheap, 5);
                                } else {
                                    right_dist = right;
                                    right_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[6] = dist_new;
                                    heapify(maxheap, 6);
                                }
                            } else {
                                right_dist = dist_new;
                                right_index = pi_i;
                            }
                        }
                    } else {
                        max_dist = dist_new;
                        max_index = pi_i;
                    }   
                    phi[t*size_x_trn+pi_i] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i] = 0;
                }
            }
        }
    
        // VECTORIZE HERE
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
    free(seq);

    return;
}

// vectorize reduction
void opt7_compute_shapley_using_improved_mc_approach(void *context) {
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
    double* phi = aligned_alloc(32, size_x_trn * T * sizeof(double));
    memset(phi, 0, size_x_trn * T * sizeof(double));
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    memset(seq, 0, size_x_trn * sizeof(int));
    int* pi = (int*)malloc(size_x_trn * sizeof(int));
    bool* trn_tst = (bool*)calloc(size_x_trn * size_x_tst, sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];

    assert(K >= 3);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);
    int p = 0;
    for (; p < size_x_trn-8; p+=8) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    for (; p < size_x_trn; p++) {
        seq[p] = p;
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[j*size_x_trn+i] = y_trn[i] == y_tst_j;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            //int pi_0 = rand() % size_x_trn;
            int pi_0 = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn);
            pi[pi_0] = size_x_trn-1;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            phi[t*size_x_trn+pi_0] = trn_tst[j*size_x_trn+pi_0];

            sum = trn_tst[j*size_x_trn+pi_0];

            for (i = 1; i < K; i++) {
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[j*size_x_trn+pi_i];

                phi[t*size_x_trn+pi_i] = (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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
            int left_dist = maxheap[1];
            int right_dist = maxheap[2];
            int max_index = x_test_knn_gt[j*size_x_trn+max_dist];
            int left_index = x_test_knn_gt[j*size_x_trn+left_dist];
            int right_index = x_test_knn_gt[j*size_x_trn+right_dist];

            for (; i < size_x_trn; i++) {
                //int next = rand() % (size_x_trn - i);
                int next = pcg32_random_bounded_divisionless_with_slight_bias(size_x_trn - i);
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];

                if (dist_new < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[j*size_x_trn+pi_i];
                    int v_excl_i = trn_tst[j*size_x_trn+max_index];

                    // Heapify
                    if (dist_new < left_dist || dist_new < right_dist) {
                        if (left_dist > right_dist) {
                            max_dist = left_dist;
                            max_index = left_index;
                            // Heapify Left
                            int left = 3 < size ? maxheap[3] : -1;
                            int right = 4 < size ? maxheap[4] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    left_dist = left;
                                    left_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[3] = dist_new;
                                    heapify(maxheap, 3);
                                } else {
                                    left_dist = right;
                                    left_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[4] = dist_new;
                                    heapify(maxheap, 4);
                                }
                            } else {
                                left_dist = dist_new;
                                left_index = pi_i;
                            }
                        } else {
                            max_dist = right_dist;
                            max_index = right_index;
                            // Heapify Right
                            int left = 5 < size ? maxheap[5] : -1;
                            int right = 6 < size ? maxheap[6] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    right_dist = left;
                                    right_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[5] = dist_new;
                                    heapify(maxheap, 5);
                                } else {
                                    right_dist = right;
                                    right_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[6] = dist_new;
                                    heapify(maxheap, 6);
                                }
                            } else {
                                right_dist = dist_new;
                                right_index = pi_i;
                            }
                        }
                    } else {
                        max_dist = dist_new;
                        max_index = pi_i;
                    }   
                    phi[t*size_x_trn+pi_i] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i] = 0;
                }
            }
        }

        int i = 0;
        for (; i < size_x_trn-4; i+=4) {
            __m256d acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, ONE_OVER_T_PD;
            __m256d v0, v1, v2, v3, v4, v5, v6, v7, res0, res1, res2, res3, res4, res5, res;
            
            ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);
            acc0 = _mm256_set1_pd(0.0);
            acc1 = _mm256_set1_pd(0.0);
            acc2 = _mm256_set1_pd(0.0);
            acc3 = _mm256_set1_pd(0.0);
            acc4 = _mm256_set1_pd(0.0);
            acc5 = _mm256_set1_pd(0.0);
            acc6 = _mm256_set1_pd(0.0);
            acc7 = _mm256_set1_pd(0.0);

            int t;
            for (t = 0; t < T-8; t+=8) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                v1 = _mm256_load_pd(phi+(t+1)*size_x_trn+i);
                v2 = _mm256_load_pd(phi+(t+2)*size_x_trn+i);
                v3 = _mm256_load_pd(phi+(t+3)*size_x_trn+i);
                v4 = _mm256_load_pd(phi+(t+4)*size_x_trn+i);
                v5 = _mm256_load_pd(phi+(t+5)*size_x_trn+i);
                v6 = _mm256_load_pd(phi+(t+6)*size_x_trn+i);
                v7 = _mm256_load_pd(phi+(t+7)*size_x_trn+i);

                acc0 = _mm256_add_pd(acc0, v0);
                acc1 = _mm256_add_pd(acc1, v1);
                acc2 = _mm256_add_pd(acc2, v2);
                acc3 = _mm256_add_pd(acc3, v3);
                acc4 = _mm256_add_pd(acc4, v4);
                acc5 = _mm256_add_pd(acc5, v5);
                acc6 = _mm256_add_pd(acc6, v6);
                acc7 = _mm256_add_pd(acc7, v7);
            }

            for (; t < T; t++) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                acc0 = _mm256_add_pd(acc0, v0);
            }

            res0 = _mm256_add_pd(acc0, acc1);
            res1 = _mm256_add_pd(acc2, acc3);
            res2 = _mm256_add_pd(acc4, acc5);
            res3 = _mm256_add_pd(acc6, acc7);
            res4 = _mm256_add_pd(res0, res1);
            res5 = _mm256_add_pd(res2, res3);
            res = _mm256_add_pd(res4, res5);
            res = _mm256_mul_pd(res, ONE_OVER_T_PD);

            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }

        for (; i < size_x_trn; i++) {
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
    free(seq);

    return;
}

// vectorize fisher yates shuffle
void opt8_compute_shapley_using_improved_mc_approach(void *context) {
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
    double* phi = aligned_alloc(32, size_x_trn * T * sizeof(double));
    memset(phi, 0, size_x_trn * T * sizeof(double));
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    int* pi = aligned_alloc(32, size_x_trn * sizeof(int));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];

    assert(K >= 3);
    assert(size_x_trn % 4 == 0);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);
    int p = 0;
    for (; p < size_x_trn-8; p+=8) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            avx_xorshift128plus_shuffle32(&mykey,pi,size_x_trn);

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            int pi_0 = pi[0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            phi[t*size_x_trn+pi_0] = trn_tst[pi_0];

            sum = trn_tst[pi_0];

            for (i = 1; i < K; i++) {
                int pi_i = pi[i];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                phi[t*size_x_trn+pi_i] = (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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

            for (; i < size_x_trn; i+=4) {
                int pi_i0 = pi[i];
                int pi_i1 = pi[i+1];
                int pi_i2 = pi[i+2];
                int pi_i3 = pi[i+3];
                int dist_new0 = x_test_knn_r_gt[j*size_x_trn+pi_i0];
                int dist_new1 = x_test_knn_r_gt[j*size_x_trn+pi_i1];

                if (dist_new0 < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[pi_i0];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new0;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i0] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i0] = 0;
                }

                int dist_new2 = x_test_knn_r_gt[j*size_x_trn+pi_i2];

                if (dist_new1 < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[pi_i1];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new1;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i1] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i1] = 0;
                }

                int dist_new3 = x_test_knn_r_gt[j*size_x_trn+pi_i3];

                if (dist_new2 < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[pi_i2];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new2;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i2] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i2] = 0;
                }

                if (dist_new3 < max_dist) {
                    // Calculate PHI
                    int v_incl_i = trn_tst[pi_i3];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new3;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i3] = (v_incl_i - v_excl_i) * ONE_OVER_K;          
                } else {
                    phi[t*size_x_trn+pi_i3] = 0;
                }
            }
        }

        int i = 0;
        for (; i < size_x_trn-4; i+=4) {
            __m256d acc0, acc1, acc2, acc3, ONE_OVER_T_PD;
            __m256d v0, v1, v2, v3, res0, res1, res2, res;
            
            ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);
            acc0 = _mm256_setzero_pd();
            acc1 = _mm256_setzero_pd();
            acc2 = _mm256_setzero_pd();
            acc3 = _mm256_setzero_pd();

            int t;
            for (t = 0; t < T; t+=4) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                v1 = _mm256_load_pd(phi+(t+1)*size_x_trn+i);
                v2 = _mm256_load_pd(phi+(t+2)*size_x_trn+i);
                v3 = _mm256_load_pd(phi+(t+3)*size_x_trn+i);

                acc0 = _mm256_add_pd(acc0, v0);
                acc1 = _mm256_add_pd(acc1, v1);
                acc2 = _mm256_add_pd(acc2, v2);
                acc3 = _mm256_add_pd(acc3, v3);
            }

            for (; t < T; t++) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                acc0 = _mm256_add_pd(acc0, v0);
            }

            res0 = _mm256_add_pd(acc0, acc1);
            res1 = _mm256_add_pd(acc2, acc3);
            res = _mm256_add_pd(res0, res1);
            res = _mm256_mul_pd(res, ONE_OVER_T_PD);

            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
    }

    free(phi);
    free(pi);
    free(seq);

    return;
}

// unrolled i loop
void opt9_compute_shapley_using_improved_mc_approach(void *context) {
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
    double* phi = aligned_alloc(32, size_x_trn * T * sizeof(double));
    uint16_t* seq = aligned_alloc(32, size_x_trn * sizeof(uint16_t));
    uint16_t* pi = (uint16_t*)malloc(size_x_trn * sizeof(uint16_t));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];

    assert(K >= 3);
    assert((size_x_trn-K) % 8 == 0);
    assert(size_x_trn % 16 == 0); 

    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    // initialize seq to [0, 1, 2, 3, 4, ..., N-1]
    __m256i ind = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi16(16);

    for (int p = 0; p < size_x_trn; p+=16) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi16(ind, incr);
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        // Precompute equality comparison of labels
        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(uint16_t));

            // Get first 8 random bounded integers for shuffling
            __m256i vec8 = _mm256_set1_epi32(8);
            __m256i upperbound = _mm256_setr_epi32(size_x_trn, size_x_trn - 1, size_x_trn - 2, size_x_trn - 3,
			size_x_trn - 4, size_x_trn - 5, size_x_trn - 6, size_x_trn - 7);

            __m256i randombytes32 = avx_xorshift128plus(&mykey);
            __m256i evenparts = _mm256_srli_epi64(_mm256_mul_epu32(randombytes32, upperbound), 32);
            __m256i oddparts = _mm256_mul_epu32(_mm256_srli_epi64(randombytes32, 32), _mm256_srli_epi64(upperbound, 32));
            __m256i res = _mm256_blend_epi32(evenparts, oddparts, 0b10101010);
            upperbound = _mm256_sub_epi32(upperbound, vec8);
            _Alignas(32) uint32_t tmp[8];
            _mm256_store_si256((__m256i*)tmp, res);
            __m128i lo = _mm256_castsi256_si128(res);

            size = 1;
            double size_sqr_plus_size = 2;
            uint8_t x = 1;

            uint16_t pi_0 = _mm_cvtsi128_si32(lo);
            pi[pi_0] = size_x_trn - 1;
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0]; 
            int sum = trn_tst[pi_0];
            phi[t*size_x_trn+pi_0] = sum;
            int i;
            for (i = 1; i < K; i++) {
                uint16_t next;
                // get next 8 random bounded
                if (x==8) {
                    randombytes32 = avx_xorshift128plus(&mykey);
                    evenparts = _mm256_srli_epi64(_mm256_mul_epu32(randombytes32, upperbound), 32);
                    oddparts = _mm256_mul_epu32(_mm256_srli_epi64(randombytes32, 32), _mm256_srli_epi64(upperbound, 32));
                    res = _mm256_blend_epi32(evenparts, oddparts, 0b10101010);
                    upperbound = _mm256_sub_epi32(upperbound, vec8);
                    _mm256_store_si256((__m256i*)tmp, res);  
                    lo = _mm256_castsi256_si128(res);
                    next = _mm_cvtsi128_si32(lo);
                    x = 1;
                } else {
                    next = tmp[x];
                    x++;
                }
                
                int pi_i = pi[next];
                pi[next] = pi[size_x_trn - i - 1];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                phi[t*size_x_trn+pi_i] = (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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

            for (; i < size_x_trn; i+=8) {

                randombytes32 = avx_xorshift128plus(&mykey);
                evenparts = _mm256_srli_epi64(_mm256_mul_epu32(randombytes32, upperbound), 32);
                oddparts = _mm256_mul_epu32(_mm256_srli_epi64(randombytes32, 32), _mm256_srli_epi64(upperbound, 32));
                res = _mm256_blend_epi32(evenparts, oddparts, 0b10101010);
                upperbound = _mm256_sub_epi32(upperbound, vec8);
                lo = _mm256_castsi256_si128(res);

                uint16_t next0 = _mm_cvtsi128_si32(lo);
                uint16_t next1 = _mm_extract_epi32(lo, 1);
                uint16_t next2 = _mm_extract_epi32(lo, 2);
                uint16_t next3 = _mm_extract_epi32(lo, 3);

                int pi_i0 = pi[next0];
                pi[next0] = pi[size_x_trn - i - 1];

                int pi_i1 = pi[next1];
                pi[next1] = pi[size_x_trn - i - 2];

                int pi_i2 = pi[next2];
                pi[next2] = pi[size_x_trn - i - 3];

                int dist_new0 = x_test_knn_r_gt[j*size_x_trn+pi_i0];
                int dist_new1 = x_test_knn_r_gt[j*size_x_trn+pi_i1];

                // i = 0
                if (dist_new0 < max_dist) {
                    int v_incl_i = trn_tst[pi_i0];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new0;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i0] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i0] = 0;
                }

                uint16_t next4 = _mm256_extract_epi32(res, 4);
                int pi_i3 = pi[next3];
                pi[next3] = pi[size_x_trn - i - 4];
                int dist_new2 = x_test_knn_r_gt[j*size_x_trn+pi_i2];
                
                // i = 1
                if (dist_new1 < max_dist) {
                    int v_incl_i = trn_tst[pi_i1];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new1;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i1] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i1] = 0;
                }

                uint16_t next5 = _mm256_extract_epi32(res, 5);
                int pi_i4 = pi[next4];
                pi[next4] = pi[size_x_trn - i - 5];
                int dist_new3 = x_test_knn_r_gt[j*size_x_trn+pi_i3];
                
                // i = 2
                if (dist_new2 < max_dist) {
                    int v_incl_i = trn_tst[pi_i2];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new2;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i2] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i2] = 0;
                }

                uint16_t next6 = _mm256_extract_epi32(res, 6);
                int pi_i5 = pi[next5];
                pi[next5] = pi[size_x_trn - i - 6];
                int dist_new4 = x_test_knn_r_gt[j*size_x_trn+pi_i4];
                
                // i = 3
                if (dist_new3 < max_dist) {
                    int v_incl_i = trn_tst[pi_i3];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new3;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i3] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i3] = 0;
                }

                uint16_t next7 = _mm256_extract_epi32(res, 7);
                int pi_i6 = pi[next6];
                pi[next6] = pi[size_x_trn - i - 7];
                int dist_new5 = x_test_knn_r_gt[j*size_x_trn+pi_i5];
                
                // i = 4
                if (dist_new4 < max_dist) {
                    int v_incl_i = trn_tst[pi_i4];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new4;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i4] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i4] = 0;
                }

                int pi_i7 = pi[next7];
                pi[next7] = pi[size_x_trn - i - 8];

                int dist_new6 = x_test_knn_r_gt[j*size_x_trn+pi_i6];
                
                // i = 5
                if (dist_new5 < max_dist) {
                    int v_incl_i = trn_tst[pi_i5];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new5;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i5] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i5] = 0;
                }

                int dist_new7 = x_test_knn_r_gt[j*size_x_trn+pi_i7];
                
                // i = 6
                if (dist_new6 < max_dist) {
                    int v_incl_i = trn_tst[pi_i6];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new6;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i6] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i6] = 0;
                }

                // i = 7
                if (dist_new7 < max_dist) {
                    int v_incl_i = trn_tst[pi_i7];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new7;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    phi[t*size_x_trn+pi_i7] = (v_incl_i - v_excl_i) * ONE_OVER_K;     
                } else {
                    phi[t*size_x_trn+pi_i7] = 0;
                }
            }
        }

        int i = 0;
        for (; i < size_x_trn-4; i+=4) {
            __m256d acc0, acc1, acc2, acc3, ONE_OVER_T_PD;
            __m256d v0, v1, v2, v3, res0, res1, res;
            
            ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);
            acc0 = _mm256_setzero_pd();
            acc1 = _mm256_setzero_pd();
            acc2 = _mm256_setzero_pd();
            acc3 = _mm256_setzero_pd();

            int t;
            for (t = 0; t < T; t+=4) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                v1 = _mm256_load_pd(phi+(t+1)*size_x_trn+i);
                v2 = _mm256_load_pd(phi+(t+2)*size_x_trn+i);
                v3 = _mm256_load_pd(phi+(t+3)*size_x_trn+i);

                acc0 = _mm256_add_pd(acc0, v0);
                acc1 = _mm256_add_pd(acc1, v1);
                acc2 = _mm256_add_pd(acc2, v2);
                acc3 = _mm256_add_pd(acc3, v3);
            }

            for (; t < T; t++) {
                v0 = _mm256_load_pd(phi+t*size_x_trn+i);
                acc0 = _mm256_add_pd(acc0, v0);
            }

            res0 = _mm256_add_pd(acc0, acc1);
            res1 = _mm256_add_pd(acc2, acc3);
            res = _mm256_add_pd(res0, res1);
            res = _mm256_mul_pd(res, ONE_OVER_T_PD);

            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
    }

    free(phi);
    free(pi);
    free(seq);
    free(trn_tst);

    return;
}

// remove phi
void opt10_compute_shapley_using_improved_mc_approach(void *context) {
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
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    int* pi = aligned_alloc(32, size_x_trn * sizeof(int));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    assert(K >= 3);
    assert(size_x_trn % 8 == 0);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);

    for (int p = 0; p < size_x_trn; p+=8) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int i = 0; i < size_x_trn; i++) {
            sp_gt[j*size_x_trn+i] = 0;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            avx_xorshift128plus_shuffle32(&mykey,pi,size_x_trn);

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            int pi_0 = pi[0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            sum = trn_tst[pi_0];
            sp_gt[j*size_x_trn+pi_0] += sum;

            for (i = 1; i < K; i++) {
                int pi_i = pi[i];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                sp_gt[j*size_x_trn+pi_i] += (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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
                    // Calculate PHI
                    int v_incl_i = trn_tst[pi_i];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];

                    maxheap[0] = dist_new;
                    heapify(maxheap, 0);
                    max_dist = maxheap[0];

                    sp_gt[j*size_x_trn+pi_i] += (v_incl_i - v_excl_i) * ONE_OVER_K;          
                }
            }
        }

        __m256d ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);

        for (int i = 0; i < size_x_trn; i+=4) {
            __m256d vx = _mm256_load_pd(sp_gt+j*size_x_trn+i);
            __m256d res = _mm256_mul_pd(vx, ONE_OVER_T_PD);
            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
        
    }

    free(trn_tst);
    free(pi);
    free(seq);

    return;
}

void opt11_compute_shapley_using_improved_mc_approach(void *context) {
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
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    int* pi = aligned_alloc(32, size_x_trn * sizeof(int));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    int maxheap[K];
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    assert(size_x_trn % 8 == 0);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);

    // initialize seq to [0, 1, 2, 3, 4, ..., N-1]
    for (int p = 0; p < size_x_trn; p+=8) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        // precompute
        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int i = 0; i < size_x_trn; i++) {
            sp_gt[j*size_x_trn+i] = 0;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            // pi is now random permutation of [0, ..., N-1]
            avx_xorshift128plus_shuffle32(&mykey,pi,size_x_trn);

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            int pi_0 = pi[0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            sum = trn_tst[pi_0];
            sp_gt[j*size_x_trn+pi_0] += sum;

            // compute for first k training points
            for (i = 1; i < K; i++) {
                int pi_i = pi[i];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                sp_gt[j*size_x_trn+pi_i] += (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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
            int left_dist = maxheap[1];
            int right_dist = maxheap[2];
            int max_index = x_test_knn_gt[j*size_x_trn+max_dist];
            int left_index = x_test_knn_gt[j*size_x_trn+left_dist];
            int right_index = x_test_knn_gt[j*size_x_trn+right_dist];

            // compute for other training points
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i]; // index of next training point
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i]; // distance to test point

                if (dist_new < max_dist) {
                    int v_incl_i = trn_tst[pi_i];
                    int v_excl_i = trn_tst[max_index];
                    sp_gt[j*size_x_trn+pi_i] += (v_incl_i - v_excl_i) * ONE_OVER_K;          

                    if (dist_new < left_dist || dist_new < right_dist) {
                        if (left_dist > right_dist) {
                            max_dist = left_dist;
                            max_index = left_index;
                            // Heapify Left
                            int left = 3 < size ? maxheap[3] : -1;
                            int right = 4 < size ? maxheap[4] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    left_dist = left;
                                    left_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[3] = dist_new;
                                    heapify(maxheap, 3);
                                } else {
                                    left_dist = right;
                                    left_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[4] = dist_new;
                                    heapify(maxheap, 4);
                                }
                            } else {
                                left_dist = dist_new;
                                left_index = pi_i;
                            }
                        } else {
                            max_dist = right_dist;
                            max_index = right_index;
                            // Heapify Right
                            int left = 5 < size ? maxheap[5] : -1;
                            int right = 6 < size ? maxheap[6] : -1;
                            if (dist_new < left || dist_new < right) {
                                if (left > right) {
                                    right_dist = left;
                                    right_index = x_test_knn_gt[j*size_x_trn+left];
                                    maxheap[5] = dist_new;
                                    heapify(maxheap, 5);
                                } else {
                                    right_dist = right;
                                    right_index = x_test_knn_gt[j*size_x_trn+right];
                                    maxheap[6] = dist_new;
                                    heapify(maxheap, 6);
                                }
                            } else {
                                right_dist = dist_new;
                                right_index = pi_i;
                            }
                        }
                    } else {
                        max_dist = dist_new;
                        max_index = pi_i;
                    }   
                }
            }
        }

        __m256d ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);

        // divide shapley values by T
        for (int i = 0; i < size_x_trn; i+=4) {
            __m256d vx = _mm256_load_pd(sp_gt+j*size_x_trn+i);
            __m256d res = _mm256_mul_pd(vx, ONE_OVER_T_PD);
            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
        
    }

    free(trn_tst);
    free(pi);
    free(seq);

    return;
}

// inlined heapify
void opt12_compute_shapley_using_improved_mc_approach(void *context) {
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
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    int* pi = aligned_alloc(32, size_x_trn * sizeof(int));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    void* pSwapMemory = malloc(sizeof(uint16_t));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    uint16_t* maxheap = (uint16_t*) malloc(K * sizeof(uint16_t));
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    assert(size_x_trn % 8 == 0);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);

    // initialize seq to [0, 1, 2, 3, 4, ..., N-1]
    for (int p = 0; p < size_x_trn; p+=8) {
        _mm256_store_si256(pi+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        // precompute
        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int i = 0; i < size_x_trn; i++) {
            sp_gt[j*size_x_trn+i] = 0;
        }

        for (int t = 0; t < T; t++) {
            
            //memcpy(pi, seq, size_x_trn * sizeof(int));

            // pi is now random permutation of [0, ..., N-1]
            avx_xorshift128plus_shuffle32(&mykey,pi,size_x_trn);
            //fisher_yates_shuffle(pi, size_x_trn);

            double size_sqr_plus_size = 2;
            int i, sum;
            int heapsize = 1;

            int pi_0 = pi[0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            sum = trn_tst[pi_0];
            sp_gt[j*size_x_trn+pi_0] += sum;

            // compute for first k training points
            for (i = 1; i < K; i++) {
                int pi_i = pi[i];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                sp_gt[j*size_x_trn+pi_i] += (heapsize*new_val - sum) / size_sqr_plus_size;

                int index = heapsize;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (heapsize << 1);
                heapsize += 1;
                sum += new_val;

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

            // compute for other training points
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i]; // index of next training point
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i]; // distance to test point

                if (dist_new < max_dist) {
                    int v_incl_i = trn_tst[pi_i];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];
                    sp_gt[j*size_x_trn+pi_i] += (v_incl_i - v_excl_i) * ONE_OVER_K;          

                    maxheap[0] = dist_new;
    
                    unsigned int iLargest;
                    unsigned int iLeft, iRight;
                    unsigned int index;

                    unsigned char *pHeap, *pCurrent, *pLeft, *pRight, *pLargest;
                    
                    iLargest = 0;
                    unsigned int next_index = 0;

                    pHeap = (unsigned char*) maxheap;
                    
                    REPEAT:
                    iLeft = 2 * next_index + 1;
                    iRight = 2 * next_index + 2;

                    pLeft = pHeap + (sizeof(uint16_t) * iLeft);
                    pRight = pHeap + (sizeof(uint16_t) * iRight);
                    pCurrent = pHeap + (sizeof(uint16_t) * next_index);
                    pLargest = pCurrent;

                    index = next_index;
                    
                    if (iLeft < K && *(const uint16_t*) pLeft > *(const uint16_t*) pCurrent) {
                        iLargest = iLeft;
                        pLargest = pLeft;
                    }

                    if (iRight < K && *(const uint16_t*) pRight > *(const uint16_t*) pLargest) {
                        iLargest = iRight;
                    }
                    
                    if (iLargest != index) {  
                        unsigned char *pElem1 = ((unsigned char *) maxheap) + (sizeof(uint16_t) * index);
                        unsigned char *pElem2 = ((unsigned char *) maxheap) + (sizeof(uint16_t) * iLargest);
                        memcpy((void *) pSwapMemory,       (const void *) pElem1,            sizeof(uint16_t));
                        memcpy((void *) pElem1,            (const void *) pElem2,            sizeof(uint16_t));
                        memcpy((void *) pElem2,            (const void *) pSwapMemory,       sizeof(uint16_t));
                        next_index = iLargest;
                        goto REPEAT;
                    }
                    max_dist = maxheap[0];
                }
            }
        }

        __m256d ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);

        // divide shapley values by T
        for (int i = 0; i < size_x_trn; i+=4) {
            __m256d vx = _mm256_load_pd(sp_gt+j*size_x_trn+i);
            __m256d res = _mm256_mul_pd(vx, ONE_OVER_T_PD);
            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
        
    }

    free(trn_tst);
    free(pi);
    free(seq);
    free(pSwapMemory);
    free(maxheap);

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
    int* seq = aligned_alloc(32, size_x_trn * sizeof(int));
    int* pi = aligned_alloc(32, size_x_trn * sizeof(int));
    bool* trn_tst = aligned_alloc(32, size_x_trn * sizeof(bool));
    void* pSwapMemory = malloc(sizeof(uint16_t));
    const double ONE_OVER_K = 1 / ctx->K;
    const double ONE_OVER_T = 1 / (double)T;
    uint16_t* maxheap = (uint16_t*) malloc(K * sizeof(uint16_t));
    avx_xorshift128plus_key_t mykey;
    avx_xorshift128plus_init(324,4444,&mykey);

    assert(size_x_trn % 8 == 0);

    __m256i ind = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    __m256i incr = _mm256_set1_epi32(8);

    // initialize seq to [0, 1, 2, 3, 4, ..., N-1]
    for (int p = 0; p < size_x_trn; p+=8) {
        _mm256_store_si256(seq+p, ind);
        ind = _mm256_add_epi32(ind, incr);
    }

    for (int j = 0; j < size_x_tst; j++) {
        double y_tst_j = y_tst[j];

        // precompute
        for (int i = 0; i < size_x_trn; i++) {
            trn_tst[i] = y_trn[i] == y_tst_j;
        }

        for (int i = 0; i < size_x_trn; i++) {
            sp_gt[j*size_x_trn+i] = 0;
        }

        for (int t = 0; t < T; t++) {
            
            memcpy(pi, seq, size_x_trn * sizeof(int));

            // pi is now random permutation of [0, ..., N-1]
            avx_xorshift128plus_shuffle32(&mykey,pi,size_x_trn);

            size = 1;
            double size_sqr_plus_size = 2;
            int i, sum;

            int pi_0 = pi[0];
            maxheap[0] = x_test_knn_r_gt[j*size_x_trn+pi_0];
            sum = trn_tst[pi_0];
            sp_gt[j*size_x_trn+pi_0] += sum;

            // compute for first k training points
            for (i = 1; i < K; i++) {
                int pi_i = pi[i];

                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i];
                int new_val = trn_tst[pi_i];

                sp_gt[j*size_x_trn+pi_i] += (size*new_val - sum) / size_sqr_plus_size;

                int index = size;
                maxheap[index] = dist_new;
                size_sqr_plus_size += 2 + (size << 1);
                size += 1;
                sum += new_val;

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

            // compute for other training points
            for (; i < size_x_trn; i++) {
                int pi_i = pi[i]; // index of next training point
                int dist_new = x_test_knn_r_gt[j*size_x_trn+pi_i]; // distance to test point

                if (dist_new < max_dist) {
                    int v_incl_i = trn_tst[pi_i];
                    int v_excl_i = trn_tst[x_test_knn_gt[j*size_x_trn+max_dist]];
                    sp_gt[j*size_x_trn+pi_i] += (v_incl_i - v_excl_i) * ONE_OVER_K;          

                    maxheap[0] = dist_new;
                    bh_heapify(maxheap, 0, pSwapMemory);
                    max_dist = maxheap[0];
                }
            }
        }

        __m256d ONE_OVER_T_PD = _mm256_set1_pd(ONE_OVER_T);

        // divide shapley values by T
        for (int i = 0; i < size_x_trn; i+=4) {
            __m256d vx = _mm256_load_pd(sp_gt+j*size_x_trn+i);
            __m256d res = _mm256_mul_pd(vx, ONE_OVER_T_PD);
            _mm256_store_pd(sp_gt+j*size_x_trn+i, res);
        }
        
    }

    free(trn_tst);
    free(pi);
    free(seq);
    free(pSwapMemory);
    free(maxheap);

    return;
}