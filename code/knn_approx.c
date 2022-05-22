#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "tsc_x86.h"
#include "benchmark.h"
#include "knn_exact.h"

extern double* dist_gt;


/*
Expected Behavior:
- result is a 2D array of size_x_tst * size_x_trn
- result[i][j] is the proximity rank of the jth train point regarding the ith test point.
*/
void get_true_approx_KNN(void *context_ptr) {
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int* sorted_distances = (int *) malloc(context->size_x_trn * sizeof(int));
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
        /*
        debug_print("%s", "get_true_approx_KNN: dist_gt:\n");
            for (int j = 0; j<10;j++) {
                debug_print("%f, ", context->dist_gt[j]);
            }
            debug_print("%s", "\n");

        */
        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        for (int i=0; i<context->size_x_trn; i++) {
            sorted_distances[sorted_indexes[i]] = i;
        }

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
        memcpy(context->x_test_knn_r_gt+(i_tst * context->size_x_trn), sorted_distances, context->size_x_trn * sizeof(int));
        
    }
    
    debug_print("%s", "Approx: Got KNN done :)\n\n");
}

void knn__approx_opt5(void *context_ptr) {
    /* opt5: based on opt1: 4 Accumulators
    accumulate sums
     */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int* sorted_distances = (int *) malloc(context->size_x_trn * sizeof(int));

    int feature_len = context->feature_len;
    int size_x_trn = context->size_x_trn;
    int size_x_tst = context->size_x_tst;

    double *x_trn = context->x_trn;
    double *x_tst = context->x_tst;


    // Loop through each test point
    for (int i_tst=0; i_tst<size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;

            for(int i_feature=0; i_feature<feature_len; i_feature+=4) {
                int trn_index = i_trn*feature_len + i_feature;
                int tst_index = i_tst*feature_len + i_feature;

                double a1 = x_trn[trn_index];
                double b1 = x_tst[tst_index];
                double a2 = x_trn[trn_index+1];
                double b2 = x_tst[tst_index+1];
                double a3 = x_trn[trn_index+2];
                double b3 = x_tst[tst_index+2];
                double a4 = x_trn[trn_index+3];
                double b4 = x_tst[tst_index+3];
                
                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;


                //curr_dist += ab1_2 + ab2_2 + ab3_2 + ab4_2;

                double curr_dist1 = ab1_2 + ab2_2;
                double curr_dist2 = ab3_2 + ab4_2;

                curr_dist += curr_dist1 + curr_dist2;
            }


            
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        for (int i=0; i<context->size_x_trn; i++) {
            sorted_distances[sorted_indexes[i]] = i;
        }

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
        memcpy(context->x_test_knn_r_gt+(i_tst * context->size_x_trn), sorted_distances, context->size_x_trn * sizeof(int));
    }
}

