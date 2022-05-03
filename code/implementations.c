#include <stdio.h>
#include <string.h>

#include "implementations.h"
#include "utils.h"

void set_implementation(run_variables_t *run_variables, char *implementation){
    run_variables->implementation = implementation;
    if(strcmp(implementation, "exact") == 0){
        run_variables->shapley_func = &run_shapley;
        return;
    }
    
    if(strcmp(implementation, "approx") == 0){
        run_variables->shapley_func = &run_approx_shapley;
        return;
    }
    
    // Otherwise terminate
    printf("Error: Unknown implementation: %s\n", implementation);
    exit(1);
}