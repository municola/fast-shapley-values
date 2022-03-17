#include <stdio.h>
#include <string.h>

#include "shapley_values.h"
#include "utils.h"

// Public vars
run_variables_t run_variables = {
    .quiet = false,
    .number_of_runs = 30,
    .runfile = "./run/1234.run"
};


int main(int argc, char *argv[]){
    parse_cmd_options(argc, argv, &run_variables);

    if(!run_variables.quiet){
        print_intro(argc, argv, &run_variables);
    }

    return 0;
}
