#ifndef __CORRECTNESS_H__
#define __CORRECTNESS_H__

#include "benchmark.h"

bool exact_correct(run_variables_t *run_variables, void *context);
bool approx_correct(run_variables_t *run_variables, void *context);
bool exact_shapley_correct(run_variables_t *run_variables, void *context);

#endif