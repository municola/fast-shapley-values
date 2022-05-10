#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "tsc_x86.h"
#include "benchmark.h"

  /*
    Expected Behavior:
    - result is a 2D array of size_x_tst * size_x_trn
    - result[i][j] is the distance of the jth train point regarding to the ith test point.
   */
void get_true_approx_KNN(void *context) {
    context_t *ctxt = (context_t *)context;
    double curr_dist;

    // Loop through each test point
    for (int i_tst=0; i_tst<ctxt->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<ctxt->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<ctxt->feature_len; i_feature++) {
                curr_dist += 
                pow(ctxt->x_trn[i_trn*ctxt->feature_len + i_feature] - 
                        ctxt->x_tst[i_tst*ctxt->feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            ctxt->dist_gt[i_trn] = curr_dist;
        }

        // copy to result array
        memcpy(ctxt->x_test_knn_gt +(i_tst * ctxt->size_x_trn), ctxt->dist_gt, ctxt->size_x_trn * sizeof(double));
    }
    
    debug_print("%s", "Approx: Got KNN done :)\n");
}

uint64_t knn_approx_opt1(void *context_ptr) {
    uint64_t start_timer, end_timer;
    start_timer = start_tsc();

    // HERE WE CODE

    end_timer = stop_tsc(start_timer);
    return end_timer;
}
