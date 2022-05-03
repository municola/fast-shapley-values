#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tsc_x86.h"
#include "io.h"
#include "utils.h"

// Use "make debug" to enable debug prints and debug symbols, etc.
#ifdef DEBUG
    #define debug_print(fmt, ...) \
                do { fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#else
    #define debug_print(fmt, ...) 
#endif

  /*
    Expected Behavior:
    - result is a 2D array of size_x_tst * size_x_trn
    - result[i][j] is the distance of the jth train point regarding to the ith test point.
   */
void get_dist_KNN(
                double* result,
                const double* x_trn,
                const double* x_tst,
                size_t size_x_trn,
                size_t size_x_tst,
                size_t feature_len ) {

    double curr_dist;
    double* dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    // Loop through each test point
    for (int i_tst=0; i_tst<size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<feature_len; i_feature++) {
                curr_dist += 
                pow(x_trn[i_trn*feature_len + i_feature] - 
                        x_tst[i_tst*feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            dist_gt[i_trn] = curr_dist;
        }

        // copy to result array
        memcpy(result+(i_tst * size_x_trn), dist_gt, size_x_trn * sizeof(double));
    }
    
    // debug_print("Get KNN done :)\n", );
    debug_print("%s", "Got KNN done :)\n");
    free(dist_gt);
}

// randomly permutes an array [1, ..., n] in place
void fisher_yates_shuffle(int* seq, int n) {
    for (int i = 0; i < n; i++) {
        seq[i] = i+1;
    }

    for (int i = n-1; i>=0; i--) {
        int j = rand() % (i+1);
        int temp = seq[i];
        seq[i] = seq[j];
        seq[j] = temp;
    }
}

void compute_shapley_using_improved_mc_approach(double* sp_gt,
                                                const double* y_trn,
                                                const double* y_tst,
                                                const double* dist_gt,
                                                const size_t size_x_trn, 
                                                const size_t size_x_tst,
                                                const int T,
                                                const int K) {

    int* pi = (int*)malloc(sizeof(int)*size_x_trn);
    double* phi = (double*)malloc(sizeof(double)*size_x_trn*T);

    // calculate the shapley values for each test point j
    for (int j = 0; j < size_x_tst; j++) {
    
        // approximate by using T different random permutations pi
        for (int t = 0; t < T; t++) {

            fisher_yates_shuffle(pi, size_x_trn);

            // nearest neighbor in set of training points pi_0 to pi_i
            int nn = -1;

            // for each point in the permutation check if it changes test accuracy
            for (int i = 0; i < size_x_trn; i++) {
                
                // check if pi_i is the new nearest neighbor (only then it changes the test accuracy)
                if (nn == -1 || dist_gt[j*size_x_trn+pi[i]] < dist_gt[j*size_x_trn+nn]) {
                    double v_incl_i = (double)(y_trn[pi[i]] == y_tst[j]);
                    double v_excl_i = (nn == -1) ? 0.0 : (double)(y_trn[nn] == y_tst[j]);
                    phi[t*size_x_trn+pi[i]] = v_incl_i - v_excl_i;
                    nn = pi[i];
                } else {
                    phi[t*size_x_trn+pi[i]] = phi[t*size_x_trn+pi[i-1]];
                }
            }
        }
        for (int i = 0; i < size_x_trn; i++) {
            sp_gt[j*size_x_trn+i] = 0;
            for (int t = 0; t < T; t++) {
                sp_gt[j*size_x_trn+i] += phi[t*size_x_trn+i];
            }
            sp_gt[j*size_x_trn+i] /= (double) T;
        }
    }

    free(phi);
    free(pi);
}

uint64_t run_approx_shapley(run_variables_t *run_variables, int input_size_no) {
    //init the training data
    // Todo: Adjust the input size parameter similar to the input size of the exact algorithm 
    size_t feature_len = run_variables->input_sizes[input_size_no];
    double* base_x_trn = (double*)malloc(sizeof(double)*50000*feature_len);
    double* base_y_trn = (double*)malloc(sizeof(double)*50000);

    assert(base_x_trn && base_y_trn);

    //Read binary training data into the arrays
    read_bin_file_known_size(base_x_trn, "../data/features/cifar10/train_features.bin", 50000*feature_len);
    read_bin_file_known_size(base_y_trn, "../data/features/cifar10/train_lables.bin", 50000);

    // Define start and lengths of the train and test data, as in the python implementation
    double* x_trn = &(base_x_trn[49990*feature_len]);
    size_t size_x_trn = 10;

    // double* y_trn = &base_y_trn[49990];
    double y_trn[10] = {4.0, 2.0, 0.0, 1.0, 0.0, 2.0, 6.0, 9.0, 1.0, 1.0};
    size_t size_y_trn = 10;

    double* x_tst = base_x_trn;
    size_t size_x_tst = 5;

    // double* y_tst = base_y_trn;
    double y_tst[10] = {6.0, 9.0, 9.0, 4.0, 1.0, 1.0, 2.0, 7.0, 8.0, 3.0};
    size_t size_y_tst = 5;

    #ifdef DEBUG
    //Sanity check, to make sure that C and Python are doing the same thing
    debug_print("%s", "x_trn:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", x_trn[i*feature_len+j]);
        }
        debug_print("%s", "\n");
    }

   debug_print("%s", "\n");
   debug_print("%s", "x_tst:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", x_tst[i*feature_len+j]);
        }
        debug_print("%s", "\n");
    }

    debug_print("%s", "\n");
    debug_print("%s", "y_tst:\n");
    for (int i = 0; i<size_y_tst;i++) {
        debug_print("%f, ", y_tst[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");
    #endif

    uint64_t start_timer, end_timer;

    // Allocate resulting arrays
    double* x_tst_dist_gt = (double*)calloc(size_x_tst * size_x_trn, sizeof(double));
    double* sp_gt = (double*)calloc(size_x_tst * size_x_trn, sizeof(double));

    start_timer = start_tsc();
    get_dist_KNN(x_tst_dist_gt, x_trn, x_tst, size_x_trn, size_x_tst, feature_len);

    #ifdef DEBUG
    debug_print("%s", "\n");
    debug_print("%s", "X_tst_dist_gt_array:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%f, ", x_tst_dist_gt[i*size_x_trn+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    
    compute_shapley_using_improved_mc_approach(sp_gt, y_trn, y_tst, x_tst_dist_gt, size_x_trn, size_x_tst, 40, 1);
    end_timer = stop_tsc(start_timer);

    #ifdef DEBUG
    // print sp_gt array
    debug_print("%s", "\n");
    debug_print("%s", "Shapley Values:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%f, ", sp_gt[i*size_x_trn+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    free(base_x_trn);
    free(base_y_trn);
    free(x_tst_dist_gt);
    free(sp_gt);

    return end_timer;
}