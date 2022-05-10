#include <stdlib.h>

#include "utils.h"

void fisher_yates_shuffle(int* seq, int n);
uint64_t run_approx_shapley(void *context);
void get_true_approx_KNN(void *context);
void compute_shapley_using_improved_mc_approach(void *context);
void opt1_compute_shapley_using_improved_mc_approach(void *context);