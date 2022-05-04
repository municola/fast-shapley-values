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

// Use "make debug" to enable debug prints and debug symbols, etc.
#ifdef DEBUG
    #define debug_print(fmt, ...) \
                do { fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#else
    #define debug_print(fmt, ...) 
#endif

double* dist_gt;

// Custom compare function for sorting, since we try to replicate numpys argsort function
// We don't want to return the sorted array, but rather the indices that would sort the array
// To achieve this, an array [0, 1, ... N] is initialized and the sorting is performed there, 
// but the comparisons are done on the original dist_gt array
// 
// This was done, such that the C qsort function can be used.
int compar (const void *a, const void *b)
{
  int aa = *((int *) a), bb = *((int *) b);
  if (dist_gt[aa] < dist_gt[bb])
    return -1;
  if (dist_gt[aa] == dist_gt[bb])
    return 0;
  if (dist_gt[aa] > dist_gt[bb])
    return 1;
  return 1;
}

  /*
    Expected Behavior:
    - result is a 2D array of size_x_tst * size_x_trn
    - result[i][j] is the proximity rank of the jth train point regarding the ith test point.
   */
void get_true_KNN(context_t *context) {
    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    //dist_gt = (double*)calloc(size_x_trn, sizeof(double));

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

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
    
    // debug_print("Get KNN done :)\n", );
    debug_print("%s", "Exact: Got KNN done :)\n");
}

void compute_single_unweighted_knn_class_shapley(context_t *context){
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

// void setup(
//         char* filename_train_features,
//         char* filename_train_labels,
//         double* x_trn,
//         double* y_trn,
//         double* x_tst,
//         double* y_tst,
//         size_t size_train_features,
//         size_t* size_x_trn,
//         size_t* size_y_trn,
//         size_t* size_x_tst,
//         size_t* size_y_tsize_st,
//         size_t feature_len ) {
// }


uint64_t run_shapley(void *context) {  
    context_t *ctx = (context_t *)context;

    // debug_print("base_y:\n");
    // for (int i = 0; i<10;i++) {
    //     debug_print("%f, ", base_y_trn[i]);
    // }
    // debug_print("\n");

    #ifdef DEBUG
    //Sanity check, to make sure that C and Python are doing the same thing
    debug_print("%s", "x_train:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", ctx->x_trn[i*ctx->feature_len+j]);
        }
        debug_print("%s", "\n");
    }

   debug_print("%s", "\n");
   debug_print("%s", "x_test:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", ctx->x_tst[i*ctx->feature_len+j]);
        }
        debug_print("%s", "\n");
    }

    debug_print("%s", "\n");
    debug_print("%s", "y_trn:\n");
    for (int i = 0; i<500;i++) {
        debug_print("%f, ", ctx->y_trn[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");

    debug_print("%s", "\n");
    debug_print("%s", "y_test:\n");
    for (int i = 0; i<500;i++) {
        debug_print("%f, ", ctx->y_tst[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");
    #endif

    uint64_t start_timer, end_timer;

    start_timer = start_tsc();
    get_true_KNN(ctx);

    #ifdef DEBUG
    // print x_tst_knn_gt array
    debug_print("%s", "\n");
    debug_print("%s", "X_test_knn_gt_array:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%d, ", ctx->x_test_knn_gt[i*ctx->size_x_trn+j]);
        }
        debug_print("%s", "\n");
    }
    #endif
        
    compute_single_unweighted_knn_class_shapley(ctx);
    end_timer = stop_tsc(start_timer);


    #ifdef DEBUG
    // print x_tst_knn_gt array
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
