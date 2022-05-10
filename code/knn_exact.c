#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "tsc_x86.h"
#include "benchmark.h"


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
void get_true_exact_KNN(void *context_ptr) {
    context_t *context = (context_t *) context_ptr;
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
        debug_print("%s", "get_true_exact_KNN: dist_gt:\n");
            for (int j = 0; j<10;j++) {
                debug_print("%f, ", dist_gt[j]);
            }
            debug_print("%s", "\n");

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));

        debug_print("%s", "Sorted Indexes\n");
        for (int j = 0; j<context->size_x_trn;j++) {
            debug_print("%d, ", sorted_indexes[j]);
        }
        debug_print("%s", "\n");


        // Sanity check in order to compare with python
        debug_print("%s", "get_true_exact_KNN: x_test_knn_gt:\n");
        for (int j = 0; j<context->size_x_trn; j++) {
            debug_print("%f, ", context->x_test_knn_gt[i_tst*context->size_x_trn + j]);
        }
        debug_print("%s", "\n");
        
    }
    // debug_print("Get KNN done :)\n", );
    debug_print("%s", "Exact: Got KNN done :)\n");
}


uint64_t knn_exact_base(void *context_ptr) {
    uint64_t start_timer, end_timer;
    start_timer = start_tsc();


    context_t *context = (context_t *) context_ptr;
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



    end_timer = stop_tsc(start_timer);
    return end_timer;
}

uint64_t knn__exact_opt1(void *context_ptr) {
    uint64_t start_timer, end_timer;
    start_timer = start_tsc();

    // HERE WE CODE

    end_timer = stop_tsc(start_timer);
    return end_timer;
}


