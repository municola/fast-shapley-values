#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include "shapley_values.h"
#include "utils.h"
#include "runfile.h"
#include "benchmark.h"

// Public vars
run_variables_t run_variables = {
    .quiet = false,
    .number_of_runs = 30,
    .number_of_input_sizes = 0,
    .input_sizes = NULL,
    .runfile_path = "",
    .runfile = NULL,
};


int main(int argc, char *argv[]){
    // Prepare log / runfile
    char runfile_name[64];
    mkdir("./run/", 0755);
    sprintf(runfile_name, "./run/%ld.json", time(NULL));
    run_variables.runfile_path = runfile_name;
    run_variables.runfile = calloc(1, sizeof(runfile_t));

    // Prepare run_variables data
    run_variables.input_sizes = calloc(128, sizeof(int));

    // Parse argv
    parse_cmd_options(argc, argv, &run_variables);

    // If no input sizes were specified, add some default powers of 2!
    if(run_variables.number_of_input_sizes == 0){
        for(int i=7; i<24; i++){
            printf("Added %d  (new %d)\n", 1<<i, run_variables.number_of_input_sizes);
            run_variables.input_sizes[run_variables.number_of_input_sizes++] = 1<<i;
        }
    }

    // Open and init runfile (after parsing argv!)
    init_runfile(run_variables.runfile, runfile_name, run_variables.number_of_input_sizes);

    // Print header and save run information
    intro(argc, argv, &run_variables);

    // Run and record benchmarks
    start_benchmark(&run_variables);

    // Close and save runfile
    close_runfile(run_variables.runfile, run_variables.input_sizes, run_variables.number_of_input_sizes);

    return 0;
}
