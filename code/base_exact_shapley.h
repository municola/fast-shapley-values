#include <stdlib.h>
#include <stdint.h>

uint64_t run_shapley(size_t feature_len);

void get_true_KNN(
                int* result,
                const double* x_trn,
                const double* x_tst,
                size_t size_x_trn,
                size_t size_x_tst,
                size_t feature_len );

void compute_single_unweighted_knn_class_shapley(double* sp_gt,
                                                const double* x_trn,
                                                const double* y_trn,
                                                const int* x_tst_knn_gt,
                                                const double* y_tst,
                                                size_t size_x_trn,
                                                size_t size_x_tst,
                                                size_t size_y_tst,
                                                double K );