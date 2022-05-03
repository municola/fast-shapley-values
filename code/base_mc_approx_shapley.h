#include <stdlib.h>

#include "utils.h"

uint64_t run_approx_shapley(run_variables_t *, int);
void fisher_yates_shuffle(int* seq, int n);
void get_dist_KNN(
                double* result,
                const double* x_trn,
                const double* x_tst,
                size_t size_x_trn,
                size_t size_x_tst,
                size_t feature_len );

void compute_shapley_using_improved_mc_approach(double* sp_gt,
                                                const double* y_trn,
                                                const double* y_tst,
                                                const double* dist_gt,
                                                const size_t size_x_trn, 
                                                const size_t size_x_tst,
                                                const int T,
                                                const int K);