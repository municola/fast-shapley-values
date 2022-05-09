#include <stdio.h>
#include <string.h>

#include "implementations.h"
#include "utils.h"
#include "knn.h"

void set_implementation(run_variables_t *run_variables, char *implementation){
    run_variables->implementation = implementation;
    if(strcmp(implementation, "exact") == 0){
        run_variables->shapley_measurement_func = &run_shapley;
        run_variables->knn_func = &get_true_KNN;
        run_variables->shapley_func = &compute_single_unweighted_knn_class_shapley;
        return;
    }

    if(strcmp(implementation, "approx") == 0){
        run_variables->shapley_measurement_func = &run_approx_shapley;
        run_variables->knn_func = &get_dist_KNN;
        run_variables->shapley_func = &compute_shapley_using_improved_mc_approach;
        return;
    }

    if(strcmp(implementation, "only_knn_base") == 0){
        run_variables->shapley_measurement_func = &knn_base;
        run_variables->knn_func = &knn_base;
        run_variables->shapley_func = &compute_single_unweighted_knn_class_shapley;
        return;
    }

    if(strcmp(implementation, "only_knn_opt1") == 0){
        run_variables->shapley_measurement_func = &knn_opt1;
        run_variables->knn_func = &knn_opt1;
        run_variables->shapley_func = &compute_single_unweighted_knn_class_shapley;
        return;
    }
    
    // Otherwise terminate
    printf("Error: Unknown implementation: %s\n", implementation);
    exit(1);
}
