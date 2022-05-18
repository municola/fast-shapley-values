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


void combined_knn_shapley_opt(void *context_ptr) {
    /* opt2: Blocked Knn + normal integratet shapley*/
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


