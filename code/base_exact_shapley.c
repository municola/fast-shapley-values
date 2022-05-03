#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "tsc_x86.h"
#include "io.h"

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
void get_true_KNN(
                int* result,
                const double* x_trn,
                const double* x_tst,
                size_t size_x_trn,
                size_t size_x_tst,
                size_t feature_len ) {

    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    dist_gt = (double*)calloc(size_x_trn, sizeof(double));

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
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(size_x_trn * sizeof(int));
        for (int i=0; i<size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        // Sanity check in order to compare with python
        // debug_print("dist_gt:\n");
        //     for (int j = 0; j<10;j++) {
        //         debug_print("%f, ", dist_gt[j]);
        //     }
        //     debug_print("\n");

        qsort(sorted_indexes, size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(result+(i_tst * size_x_trn), sorted_indexes, size_x_trn * sizeof(int));
    }
    
    // debug_print("Get KNN done :)\n", );
    debug_print("%s", "Got KNN done :)\n");
    free(dist_gt);
}

void compute_single_unweighted_knn_class_shapley(double* sp_gt,
                                                const double* x_trn,
                                                const double* y_trn,
                                                const int* x_tst_knn_gt,
                                                const double* y_tst,
                                                size_t size_x_trn,
                                                size_t size_x_tst,
                                                size_t size_y_tst,
                                                double K ){

    debug_print("%s", "\nStart Shapley computation:\n");
    for(int j=0; j<size_x_tst;j++){
        // debug_print("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = x_tst_knn_gt[j*size_x_trn+size_x_trn-1];
        double tmp = (y_trn[offset] == y_tst[j]) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp / size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            // debug_print("    i=%d\n", i);
            int index_j_i = j*size_x_trn+i;

            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_tst_knn_gt[index_j_i+1]];
            double difference = (double)(y_trn[x_tst_knn_gt[index_j_i]] == y_tst[j]) - 
                                        (double)(y_trn[x_tst_knn_gt[index_j_i+1]] == y_tst[j]);
            double min_K_i = K < i+1 ? K : i+1;

            // debug_print("      s_j=%f\n", s_j_alpha_i_plus_1);
            // debug_print("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // debug_print("      min_=%f\n", min_K_i);

            sp_gt[j*size_x_trn + x_tst_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference / K) * (min_K_i / (i+1));
        }
    }
    debug_print("%s", "Shapley Values done :)\n");
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


uint64_t run_shapley(int input_size) {
    int feature_len = 2048;
    int num_test_samples = 500;
    
    double* base_x_train = (double*)malloc(sizeof(double)*input_size*feature_len);
    double* base_y_train = (double*)malloc(sizeof(double)*input_size);
    double* base_x_test = (double*)malloc(sizeof(double)*input_size*feature_len);
    double* base_y_test = (double*)malloc(sizeof(double)*input_size);

    assert(base_x_train && base_y_train);

    //Read binary training data into the arrays
    read_bin_file_known_size(base_x_train, "../data/features/cifar10/train_features.bin", input_size*feature_len);
    read_bin_file_known_size(base_y_train, "../data/features/cifar10/train_labels.bin", input_size*1);
    read_bin_file_known_size(base_x_test, "../data/features/cifar10/test_features.bin", num_test_samples*feature_len);
    read_bin_file_known_size(base_y_test, "../data/features/cifar10/test_labels.bin", num_test_samples);

    // debug_print("base_y:\n");
    // for (int i = 0; i<10;i++) {
    //     debug_print("%f, ", base_y_trn[i]);
    // }
    // debug_print("\n");

    // Define start and lengths of the train and test data, as in the python implementation
    double* x_train = base_x_train;
    size_t size_x_train = input_size;
    double* y_train = base_y_train;
    size_t size_y_train = input_size;

    double* x_test = base_x_test;
    size_t size_x_test = num_test_samples;
    double* y_test = base_y_test;
    size_t size_y_test = num_test_samples;

    #ifdef DEBUG
    //Sanity check, to make sure that C and Python are doing the same thing
    debug_print("%s", "x_train:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", x_train[i*feature_len+j]);
        }
        debug_print("%s", "\n");
    }

   debug_print("%s", "\n");
   debug_print("%s", "x_test:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            debug_print("%f, ", x_test[i*feature_len+j]);
        }
        debug_print("%s", "\n");
    }

    debug_print("%s", "\n");
    debug_print("%s", "y_train:\n");
    for (int i = 0; i<500;i++) {
        debug_print("%f, ", y_train[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");

    debug_print("%s", "\n");
    debug_print("%s", "y_test:\n");
    for (int i = 0; i<500;i++) {
        debug_print("%f, ", y_test[i]);
    }
    debug_print("%s", "\n");
    debug_print("%s", "\n");
    #endif

    uint64_t start_timer, end_timer;

    // Allocate resulting arrays
    int* x_test_knn_gt = (int*)calloc(size_x_test * size_x_train, sizeof(int));
    get_true_KNN(x_test_knn_gt, x_train, x_test, size_x_train, size_x_test, feature_len);

    #ifdef DEBUG
    // print x_tst_knn_gt array
    debug_print("%s", "\n");
    debug_print("%s", "X_test_knn_gt_array:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%d, ", x_test_knn_gt[i*size_x_train+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    double* sp_gt = (double*)calloc(size_x_test * size_x_train, sizeof(double));

    start_timer = start_tsc();    
    compute_single_unweighted_knn_class_shapley(sp_gt, x_train, y_train, x_test_knn_gt, y_test, size_x_train, size_x_test, size_y_test, 1.0);
    end_timer = stop_tsc(start_timer);

    #ifdef DEBUG
    // print x_tst_knn_gt array
    debug_print("%s", "\n");
    debug_print("%s", "Shapley Values:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            debug_print("%f, ", sp_gt[i*size_x_train+j]);
        }
        debug_print("%s", "\n");
    }
    #endif

    free(base_x_train);
    free(base_y_train);
    free(base_x_test);
    free(base_y_test);
    free(x_test_knn_gt);
    free(sp_gt);

    return end_timer;
}
