#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "tsc_x86.h"
#include "benchmark.h"
#include "knn_exact.h"
#include "combined_exact_knn_shapley.h"
#include "base_exact_shapley.h"


extern double* dist_gt;
extern double* dist_gt_row;

inline
double vec_sum(__m256d vec) {
    __m128d vlow  = _mm256_castpd256_pd128(vec);
    __m128d vhigh = _mm256_extractf128_pd(vec, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to 
}

void print_vec(__m256d var) {
    printf("%f %f %f %f \n", 
           var[0], var[1], var[2], var[3]);
}


/***************************************** IMPLEMENTATIONS *******************************************/


void exact_shapley_base(void *context_ptr) {
    /* exact_shapley_base   combined_knn_shapley_opt*/

    // Call unoptimzed base KNN
    knn_exact_base(context_ptr);

    // Call default implemention of shapley
    compute_single_unweighted_knn_class_shapley(context_ptr);
}


void combined_knn_shapley_opt1(void *context_ptr) {
    /* opt1: just combined */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;

    size_t size_x_trn = context->size_x_trn;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;

    double inv_size_x_trn = 1.0/size_x_trn;

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature++) {
                curr_dist += 
                pow(context->x_trn[i_trn*context->feature_len + i_feature] - 
                        context->x_tst[i_tst*context->feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        // Sanity check in order to compare with python
        // debug_print("dist_gt:\n");
        //     for (int j = 0; j<10;j++) {
        //         debug_print("%f, ", dist_gt[j]);
        //     }
        //     debug_print("\n");

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // This can be eliminated later - only here to
        // give us testing/debug information on errors / correctness tests
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));

        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = sorted_indexes[size_x_trn-1];
        double y_tst_j = y_tst[i_tst];
        
        // These need to be two expressions, if you set the array directly, you take a performance hit  
        double tmp = (y_trn[offset] == y_tst_j) ? 1.0 : 0.0;
        sp_gt[i_tst*size_x_trn + offset] = tmp * inv_size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            // debug_print("    i=%d\n", i);
            int x_test_knn_gt_i = sorted_indexes[i];
            int x_test_knn_gt_i_plus_one = sorted_indexes[i+1];

            double s_j_alpha_i_plus_1 = sp_gt[i_tst*size_x_trn + x_test_knn_gt_i_plus_one];
            double difference = (double)(y_trn[x_test_knn_gt_i] == y_tst_j) - 
                                        (double)(y_trn[x_test_knn_gt_i_plus_one] == y_tst_j);
            double min_K_i = K < i+1 ? K : i+1;

            // debug_print("      s_j=%f\n", s_j_alpha_i_plus_1);
            // debug_print("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // debug_print("      min_=%f\n", min_K_i);

            sp_gt[i_tst*size_x_trn + sorted_indexes[i]] = s_j_alpha_i_plus_1 + (difference * inv_K) * (min_K_i / (i+1));
        }
    }
}


void combined_knn_shapley_opt2(void *context_ptr) {
    /* opt2: Blocked Knn + normal integratet shapley + normal integrated sorting*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 12;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 4 == 0);

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=4){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;

                        dist[i1*train_length + j1] += acc_sum1 + acc_sum2;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    double t = sqrt(dist[i2*train_length + j2]);
                    dist[i2*train_length + j2] = t;
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                double min_K_i = K < sj+1 ? K : sj+1;
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * inv_K) * (min_K_i / (sj+1));                
            }
        }
    }
}



void combined_knn_shapley_opt3(void *context_ptr) {
    /* opt3: Blocked Knn + optimized integratet shapley + normal integrated sorting*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 12;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 4 == 0);

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=4){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;

                        dist[i1*train_length + j1] += acc_sum1 + acc_sum2;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    double t = sqrt(dist[i2*train_length + j2]);
                    dist[i2*train_length + j2] = t;
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt4(void *context_ptr) {
    /* opt4: Blocked Knn + optimized integratet shapley + normal integrated sorting*/
    /* 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        
                        double dist_acc1 = 0;
                        double dist_acc2 = 0;
                        double dist_acc3 = 0;
                        double dist_acc4 = 0;
                        double dist_acc5 = 0;
                        double dist_acc6 = 0;
                        double dist_acc7 = 0;
                        double dist_acc8 = 0;
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            double a0 = x_tst[i1*f_length + k1 + 0];
                            double b0 = x_trn[j1*f_length + k1 + 0];
                            double a1 = x_tst[i1*f_length + k1 + 1];
                            double b1 = x_trn[j1*f_length + k1 + 1];
                            double a2 = x_tst[i1*f_length + k1 + 2];
                            double b2 = x_trn[j1*f_length + k1 + 2];
                            double a3 = x_tst[i1*f_length + k1 + 3];
                            double b3 = x_trn[j1*f_length + k1 + 3];
                            double a4 = x_tst[i1*f_length + k1 + 4];
                            double b4 = x_trn[j1*f_length + k1 + 4];
                            double a5 = x_tst[i1*f_length + k1 + 5];
                            double b5 = x_trn[j1*f_length + k1 + 5];
                            double a6 = x_tst[i1*f_length + k1 + 6];
                            double b6 = x_trn[j1*f_length + k1 + 6];
                            double a7 = x_tst[i1*f_length + k1 + 7];
                            double b7 = x_trn[j1*f_length + k1 + 7];

                            
                            dist_acc1 += (a0-b0)*(a0-b0);
                            dist_acc2 += (a1-b1)*(a1-b1);
                            dist_acc3 += (a2-b2)*(a2-b2);
                            dist_acc4 += (a3-b3)*(a3-b3);
                            dist_acc5 += (a4-b4)*(a4-b4);
                            dist_acc6 += (a5-b5)*(a5-b5);
                            dist_acc7 += (a6-b6)*(a6-b6);
                            dist_acc8 += (a7-b7)*(a7-b7);
                        }

                        double acc_sum1 = dist_acc1 + dist_acc2;
                        double acc_sum2 = dist_acc3 + dist_acc4;
                        double acc_sum3 = dist_acc5 + dist_acc6;
                        double acc_sum4 = dist_acc7 + dist_acc8;

                        double acc_sum12 = acc_sum1 + acc_sum2;
                        double acc_sum34 = acc_sum3 + acc_sum4;

                        dist[i1*train_length + j1] += acc_sum12 + acc_sum34;
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt5(void *context_ptr) {
    /* opt5: Vectorized: Only most inner loop*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;


    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once
                        __m256d dist_acc03 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03 = _mm256_loadu_pd((double *)(x_trn + j1*f_length + k1));
                            __m256d a47 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47 = _mm256_loadu_pd((double *)(x_trn + j1*f_length + k1 + 4));

                            __m256d a_sub_b_03 = _mm256_sub_pd(a03, b03);
                            __m256d a_sub_b_47 = _mm256_sub_pd(a47, b47);

                            __m256d a_sub_b_03_2 = _mm256_mul_pd(a_sub_b_03, a_sub_b_03);
                            __m256d a_sub_b_47_2 = _mm256_mul_pd(a_sub_b_47, a_sub_b_47);

                            dist_acc03 = _mm256_add_pd(dist_acc03, a_sub_b_03_2);
                            dist_acc47 = _mm256_add_pd(dist_acc47, a_sub_b_47_2);
                        }
                        __m256d dist_acc07 = _mm256_add_pd(dist_acc03, dist_acc47);

                        dist[i1*train_length + j1] += (double) vec_sum(dist_acc07);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt6(void *context_ptr) {
    /* opt6: Vectorized: Most inner + loop unrolling of second-most-inner*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_loadu_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_storeu_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    dist[i2*train_length + j2] = sqrt(dist[i2*train_length + j2]);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}


void combined_knn_shapley_opt7(void *context_ptr) {
    /* opt7: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 16;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_loadu_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_loadu_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_loadu_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_storeu_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_loadu_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_storeu_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}


void combined_knn_shapley_opt(void *context_ptr) {
    /* opt8: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    /* Additionally, rely on aligned memory accesses */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 32;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){
                        // Note that array "a" does not change (not dependent on j1)
                        // When we fixed the blocksize we could move it out of the for loop
                        // and only read it once

                        // Entry 1: dist[i1*train_length + j1]
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        int j1_vec = j1;
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e1 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03_e1, b03_e1);
                            __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47_e1, b47_e1);
                            __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                            __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                            dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                            dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        }
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);

                        // Entry 2: dist[i1*train_length + j1 + 1]
                        j1_vec = j1 + 1;
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e2 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e2 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03_e2, b03_e2);
                            __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47_e2, b47_e2);
                            __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                            __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);

                            dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                            dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        }
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);

                
                        // Entry 3: dist[i1*train_length + j1 + 2]
                        j1_vec = j1 + 2;
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e3 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e3 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03_e3, b03_e3);
                            __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47_e3, b47_e3);
                            __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                            __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);

                            dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                            dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        }
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);


                        // Entry 4: dist[i1*train_length + j1 + 3]
                        j1_vec = j1 + 3;
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        for (int k1=k; k1<k+B; k1+=8){
                            __m256d a03_e4 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                            __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1));
                            __m256d a47_e4 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                            __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + j1_vec*f_length + k1 + 4));
                            __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03_e4, b03_e4);
                            __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47_e4, b47_e4);
                            __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                            __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);

                            dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                            dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        }
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);

                        // Sum and Store
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );

                        // Since its a "+=" we need to load->add->store, instead of only store
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}



void combined_knn_shapley_opt9(void *context_ptr) {
    /* opt9: Vectorized: Most inner + loop unrolling of second-most-inner + Sqrt*/
    /* Blocked Knn + optimized integratet shapley + normal integrated sorting + 8 Accumulators */
    /* Additionally, rely on aligned memory accesses */
    /* FIXED BLOCK SIZE OF 8 */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 8;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;
    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;
    double *dist = context->dist_gt;
    int* x_test_knn_gt = context->x_test_knn_gt;
    double* y_trn = context->y_trn;
    double* y_tst = context->y_tst;
    double* sp_gt = context->sp_gt;
    double K = context->K;
    double inv_K = 1.0 / K;
    double inv_train_length = 1.0/train_length;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);
    assert(B % 8 == 0);
    assert(B % 4 == 0); // For vectorization

    // Precompute the constant part from Line 5 in the Shapley algorithm
    double* Kidx_const = (double*)malloc((train_length-1) * sizeof(double));
    for (int i=1; i<train_length; i++) {
        Kidx_const[i-1] = 1.0/i;
    }
    for (int i=0; i<K; i++){
        Kidx_const[i] = 1.0/K;
    }

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            // Calculate 1 Block in output matrix (Need to go through the multiple blocks
            // from the other two matrices)
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1+=4){

                        /* LOADS */
                        int k1 = k;
                        int j1_vec_e1 = j1;
                        int j1_vec_e2 = j1 + 1;
                        int j1_vec_e3 = j1 + 2;
                        int j1_vec_e4 = j1 + 3;
                        __m256d dist_acc03_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e1 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e2 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e3 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc03_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        __m256d dist_acc47_e4 = _mm256_set_pd(0.0,0.0,0.0,0.0);
                        // Entry 1
                        __m256d a03 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1));
                        __m256d b03_e1 = _mm256_load_pd((double *)(x_trn + j1_vec_e1*f_length + k1));
                        __m256d a47 = _mm256_load_pd((double *)(x_tst + i1*f_length + k1 + 4));
                        __m256d b47_e1 = _mm256_load_pd((double *)(x_trn + j1_vec_e1*f_length + k1 + 4));
                        // Entry 2
                        __m256d b03_e2 = _mm256_load_pd((double *)(x_trn + j1_vec_e2*f_length + k1));
                        __m256d b47_e2 = _mm256_load_pd((double *)(x_trn + j1_vec_e2*f_length + k1 + 4));
                        // Entry 3
                        __m256d b03_e3 = _mm256_load_pd((double *)(x_trn + j1_vec_e3*f_length + k1));
                        __m256d b47_e3 = _mm256_load_pd((double *)(x_trn + j1_vec_e3*f_length + k1 + 4));
                        // Entry 4
                        __m256d b03_e4 = _mm256_load_pd((double *)(x_trn + j1_vec_e4*f_length + k1));
                        __m256d b47_e4 = _mm256_load_pd((double *)(x_trn + j1_vec_e4*f_length + k1 + 4));
                        // Location to store
                        __m256d dist_vec = _mm256_load_pd(dist + i1*train_length + j1);

                        // COMPUTE
                        // Entry 1
                        __m256d a_sub_b_03_e1 = _mm256_sub_pd(a03, b03_e1);
                        __m256d a_sub_b_47_e1 = _mm256_sub_pd(a47, b47_e1);
                        __m256d a_sub_b_03_2_e1 = _mm256_mul_pd(a_sub_b_03_e1, a_sub_b_03_e1);
                        __m256d a_sub_b_47_2_e1 = _mm256_mul_pd(a_sub_b_47_e1, a_sub_b_47_e1);
                        dist_acc03_e1 = _mm256_add_pd(dist_acc03_e1, a_sub_b_03_2_e1);
                        dist_acc47_e1 = _mm256_add_pd(dist_acc47_e1, a_sub_b_47_2_e1);
                        __m256d dist_acc07_e1 = _mm256_add_pd(dist_acc03_e1, dist_acc47_e1);
                        // Entry 2
                        __m256d a_sub_b_03_e2 = _mm256_sub_pd(a03, b03_e2);
                        __m256d a_sub_b_47_e2 = _mm256_sub_pd(a47, b47_e2);
                        __m256d a_sub_b_03_2_e2 = _mm256_mul_pd(a_sub_b_03_e2, a_sub_b_03_e2);
                        __m256d a_sub_b_47_2_e2 = _mm256_mul_pd(a_sub_b_47_e2, a_sub_b_47_e2);
                        dist_acc03_e2 = _mm256_add_pd(dist_acc03_e2, a_sub_b_03_2_e2);
                        dist_acc47_e2 = _mm256_add_pd(dist_acc47_e2, a_sub_b_47_2_e2);
                        __m256d dist_acc07_e2 = _mm256_add_pd(dist_acc03_e2, dist_acc47_e2);
                        // Entry 3
                        __m256d a_sub_b_03_e3 = _mm256_sub_pd(a03, b03_e3);
                        __m256d a_sub_b_47_e3 = _mm256_sub_pd(a47, b47_e3);
                        __m256d a_sub_b_03_2_e3 = _mm256_mul_pd(a_sub_b_03_e3, a_sub_b_03_e3);
                        __m256d a_sub_b_47_2_e3 = _mm256_mul_pd(a_sub_b_47_e3, a_sub_b_47_e3);
                        dist_acc03_e3 = _mm256_add_pd(dist_acc03_e3, a_sub_b_03_2_e3);
                        dist_acc47_e3 = _mm256_add_pd(dist_acc47_e3, a_sub_b_47_2_e3);
                        __m256d dist_acc07_e3 = _mm256_add_pd(dist_acc03_e3, dist_acc47_e3);
                        // Entry 4
                        __m256d a_sub_b_03_e4 = _mm256_sub_pd(a03, b03_e4);
                        __m256d a_sub_b_47_e4 = _mm256_sub_pd(a47, b47_e4);
                        __m256d a_sub_b_03_2_e4 = _mm256_mul_pd(a_sub_b_03_e4, a_sub_b_03_e4);
                        __m256d a_sub_b_47_2_e4 = _mm256_mul_pd(a_sub_b_47_e4, a_sub_b_47_e4);
                        dist_acc03_e4 = _mm256_add_pd(dist_acc03_e4, a_sub_b_03_2_e4);
                        dist_acc47_e4 = _mm256_add_pd(dist_acc47_e4, a_sub_b_47_2_e4);
                        __m256d dist_acc07_e4 = _mm256_add_pd(dist_acc03_e4, dist_acc47_e4);
                        // Sum
                        __m256d summed_entries = _mm256_set_pd(
                            vec_sum(dist_acc07_e4),
                            vec_sum(dist_acc07_e3),
                            vec_sum(dist_acc07_e2),
                            vec_sum(dist_acc07_e1)
                        );
                        // Since its a "+=" we need to load->add->store, instead of only store
                        dist_vec = _mm256_add_pd(dist_vec, summed_entries);

                        // STORE
                        _mm256_store_pd(dist + i1*train_length + j1, dist_vec);
                    }
                }
            }
            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2+=4) {
                    __m256d data = _mm256_load_pd(dist + i2*train_length + j2);
                    __m256d result =  _mm256_sqrt_pd(data);
                    _mm256_store_pd(dist + i2*train_length + j2, result);
                }
            }
        }

        // Shapley Computation + Sorting
        // So far we have calculated M[i,:] to M[i+B,:] where M is the result Matrix (size=train*test)
        for (int b=i; b<i+B; b++){
            /* Sorting */
            dist_gt_row = &context->dist_gt[b*train_length];
            int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
            for (int idx=0; idx<train_length; idx++) {
                sorted_indexes[idx] = idx;
            }
            qsort(sorted_indexes, train_length, sizeof(int), compar_block);
            // This memcpy is theoretically not needed
            memcpy(context->x_test_knn_gt+(b*train_length), sorted_indexes, train_length*sizeof(int));

            /* Shapley */
            // Line 3 in algo
            int a_N = sorted_indexes[train_length-1];
            double y_test_j = y_tst[b];
            double indicator = (y_trn[a_N] == y_test_j) ? 1.0 : 0.0;
            sp_gt[b*train_length + a_N] = indicator * inv_train_length;
            
            // Calculate the shapley by moving from N-1 to 1 (loop line 4)
            for (int sj=train_length-2; sj>-1; sj--) {
                int x_test_knn_gt_i = sorted_indexes[sj];
                int x_test_knn_gt_i_plus_one = sorted_indexes[sj+1];
                double s_j_alpha_i_plus_1 = sp_gt[b*train_length + x_test_knn_gt_i_plus_one];
                double difference = (double)(y_trn[x_test_knn_gt_i] == y_test_j) - 
                                            (double)(y_trn[x_test_knn_gt_i_plus_one] == y_test_j);
                sp_gt[b*train_length + sorted_indexes[sj]] = s_j_alpha_i_plus_1 + (difference * Kidx_const[sj]);                
            }
        }
    }
}
