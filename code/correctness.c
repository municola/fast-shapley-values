#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "io.h"
#include "base_exact_shapley.h"

#define EPS 10e-3

double nrm_sqr_diff(double *x, double *y, int n) {
    double nrm_sqr = 0.0;
    for(int i = 0; i < n; i++) {
        nrm_sqr += (x[i] - y[i]) * (x[i] - y[i]);
    }
    
    if (isnan(nrm_sqr)) {
      nrm_sqr = INFINITY;
    }
    
    return nrm_sqr;
}

bool exact_knn_correct() {
    size_t feature_len = 2048;

    return false;

    /*
    //init the training data
    double* base_x_trn = (double*)malloc(sizeof(double)*50000*feature_len);
    double* base_y_trn = (double*)malloc(sizeof(double)*50000);

    assert(base_x_trn && base_y_trn);

    //Read binary training data into the arrays
    read_bin_file_known_size(base_x_trn, "../data/features/cifar10/train_features.bin", 50000*feature_len);
    read_bin_file_known_size(base_y_trn, "../data/features/cifar10/train_lables.bin", 50000);

    // debug_print("base_y:\n");
    // for (int i = 0; i<10;i++) {
    //     debug_print("%f, ", base_y_trn[i]);
    // }
    // debug_print("\n");

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

    // Allocate resulting arrays
    int* base_x_tst_knn_gt = (int*)calloc(size_x_tst * size_x_trn, sizeof(int));
    get_true_KNN(base_x_tst_knn_gt, x_trn, x_tst, size_x_trn, size_x_tst, feature_len);

    // Allocate resulting arrays
    int* optimized_x_tst_knn_gt = (int*)calloc(size_x_tst * size_x_trn, sizeof(int));
    get_true_KNN(optimized_x_tst_knn_gt, x_trn, x_tst, size_x_trn, size_x_tst, feature_len);
    
    double error = nrm_sqr_diff((double*) base_x_tst_knn_gt, (double*) optimized_x_tst_knn_gt, size_x_tst*size_x_trn);

    free(base_x_trn);
    free(base_y_trn);
    free(base_x_tst_knn_gt);
    free(optimized_x_tst_knn_gt);

    return error < EPS;
    */
}

bool exact_shapley_correct() {
    return false;

    /*
    size_t feature_len = 2048;

    //init the training data
    double* base_x_trn = (double*)malloc(sizeof(double)*50000*feature_len);
    double* base_y_trn = (double*)malloc(sizeof(double)*50000);

    assert(base_x_trn && base_y_trn);

    //Read binary training data into the arrays
    read_bin_file_known_size(base_x_trn, "../data/features/cifar10/train_features.bin", 50000*feature_len);
    read_bin_file_known_size(base_y_trn, "../data/features/cifar10/train_lables.bin", 50000);

    // debug_print("base_y:\n");
    // for (int i = 0; i<10;i++) {
    //     debug_print("%f, ", base_y_trn[i]);
    // }
    // debug_print("\n");

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


    // Allocate resulting arrays
    //context_t ctxt = get_context()
    int* x_tst_knn_gt = (int*)calloc(size_x_tst * size_x_trn, sizeof(int));
    get_true_KNN(x_tst_knn_gt, x_trn, x_tst, size_x_trn, size_x_tst, feature_len);

    // First compute results on base algorithm
    double* base_sp_gt = (double*)calloc(size_x_tst * size_x_trn, sizeof(double));
    compute_single_unweighted_knn_class_shapley(base_sp_gt, x_trn, y_trn, x_tst_knn_gt, y_tst, size_x_trn, size_x_tst, size_y_tst, 1.0);

    // Then check if the specified implementation is correct
    double* optimized_sp_gt = (double*)calloc(size_x_tst * size_x_trn, sizeof(double));
    compute_single_unweighted_knn_class_shapley(optimized_sp_gt, x_trn, y_trn, x_tst_knn_gt, y_tst, size_x_trn, size_x_tst, size_y_tst, 1.0);

    double error = nrm_sqr_diff((double*) base_sp_gt, (double*) optimized_sp_gt, size_x_tst*size_x_trn);

    free(base_x_trn);
    free(base_y_trn);
    free(x_tst_knn_gt);
    free(base_sp_gt);
    free(optimized_sp_gt);

    return error < EPS;
    */
}