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
    .runfile_path = "",
    .runfile = NULL
};


int main(int argc, char *argv[]){
    // Prepare log / runfile
    char runfile_name[64];
    mkdir("./run/", 0755);
    sprintf(runfile_name, "./run/%u.json", time(NULL));
    run_variables.runfile_path = runfile_name;
    run_variables.runfile = calloc(1, sizeof(runfile_t));

    // Parse argv
    parse_cmd_options(argc, argv, &run_variables);

    // Open and init runfile (after parsing argv!)
    init_runfile(run_variables.runfile, runfile_name);

    // Print header and save run information
    intro(argc, argv, &run_variables);

    // Run and record benchmarks
    start_benchmark(&run_variables);

    // Close and save runfile
    close_runfile(run_variables.runfile);

    return 0;
}
