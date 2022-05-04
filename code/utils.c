#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "utils.h"
#include "runfile.h"
#include "correctness.h"
#include "implementations.h"

char *exec_and_get_output(char *cmd){
    char cmd_redirect[512];
    
    strncpy(cmd_redirect, cmd, 512-19);
    strcpy(cmd_redirect+strlen(cmd_redirect), " > /tmp/shapley.tmp");
    int ret = system(cmd_redirect);

    if(ret != 0){
        return color("Error", RED);
    }

    FILE *f = fopen("/tmp/shapley.tmp", "r");
    fseek(f, 0, SEEK_END);
    size_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *content = malloc(n);
    fread(content, 1, n, f);
    fclose(f);

    return content;
}


void parse_cmd_options(int argc, char **argv, run_variables_t *run_variables){
    int i = 1;
    while(i < argc){
        if(strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0){
            run_variables->quiet = true;

            i += 1;
            continue;
        }

        if(strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num-runs") == 0){
            if(argc <= i+1){
                printf("Error: Missing number of runs\n");
                exit(1);
            }

            run_variables->number_of_runs = atoi(argv[i+1]);

            i += 2;
            continue;
        }

        if(strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input-size") == 0){
            if(argc <= i+1){
                printf("Error: Missing input size\n");
                exit(1);
            }

            run_variables->input_sizes[run_variables->number_of_input_sizes++] = atoi(argv[i+1]);

            i += 2;
            continue;
        }

        if(strcmp(argv[i], "--impl") == 0 || strcmp(argv[i], "--implementation") == 0){
            if(argc <= i+1){
                printf("Error: Missing implementation\n");
                exit(1);
            }

            run_variables->implementation = argv[i+1];

            i += 2;
            continue;
        }

        printf("Error: Unknown argument '%s'\n", argv[i]);
        exit(1);
    }
}


void intro(int argc, char **argv, run_variables_t *run_variables){
    // Collect all command line arguments
    char args[1024] = {0};
    size_t offset = 0;
    for(int i=1; i<argc; i++){
        sprintf(args+offset, "%s ", argv[i]);
        offset += strlen(argv[i]) + 1;
    }

    // Check compile time information
    char *git_dirty;
    if(strcmp(GITSTATUS, "") == 0){
        git_dirty = "clean";
    } else {
        git_dirty = "modified";
    }

    // Correctness tests
    int test_input_size = 5;
    run_variables_t test_vars = {
        .number_of_runs = 1,
        .number_of_input_sizes = 1,
        .input_sizes = &test_input_size
    };

    // Use the correct function pointers to test the right functions for correctness
    set_implementation(&test_vars, run_variables->implementation);

    // Setup the context / init memory, etc.
    context_t test_ctxt;
    init_context(&test_ctxt, test_input_size);
    
    // Actual correctness tests
    bool shapley_correct = exact_shapley_correct(&test_vars, (void*)&test_ctxt);
    bool knn_correct = exact_knn_correct(&test_vars, (void*)&test_ctxt);

    // Write collected info to the runfile:
    char tmpbuf[512]; 
    add_run_info(run_variables->runfile, "git_status", git_dirty);
    add_run_info(run_variables->runfile, "git_branch", GITBRANCH);
    add_run_info(run_variables->runfile, "compiler", CC);
    add_run_info(run_variables->runfile, "compiler_flags", CFLAGS);
    add_run_info_raw(run_variables->runfile, "debug",
        #ifdef DEBUG
        "true"
        #else
        "false"
        #endif
    );
    snprintf(tmpbuf, strlen(get_cpu_model())-1, "%s", get_cpu_model()+1);
    add_run_info(run_variables->runfile, "cpu", tmpbuf);
    add_run_info_raw(run_variables->runfile, "turbo_boost_disabled", intel_turbo_boost_disabled() ? "true" : "false");
    add_run_info(run_variables->runfile, "arguments", args);
    add_run_info(run_variables->runfile, "implementation", run_variables->implementation);
    add_run_info_raw(run_variables->runfile, "shapley_correct", shapley_correct ? "true" : "false");
    add_run_info_raw(run_variables->runfile, "knn_correct", knn_correct ? "true" : "false");
    add_run_info_int(run_variables->runfile, "num_runs", run_variables->number_of_runs);
    add_run_info_int(run_variables->runfile, "num_input_sizes", run_variables->number_of_input_sizes);

    // Collect input sizes
    offset = 1;
    tmpbuf[0] = '[';
    for(int i=0; i<run_variables->number_of_input_sizes; i++){
        sprintf(tmpbuf+offset, "%d, ", run_variables->input_sizes[i]);
        offset += strlen(tmpbuf+offset);
    }
    tmpbuf[strlen(tmpbuf) - 2] = ']';
    tmpbuf[strlen(tmpbuf) - 1] = '\0';

    add_run_info_raw(run_variables->runfile, "input_sizes", tmpbuf);


    // Print nice header, unless we should be quiet
    if(run_variables->quiet)
        return;

    printf("%s", bold("Efficient Shapley Value Computation\n"));
    for(size_t i=0; i<80; i++) { printf("-"); } printf("\n");

    printf(
           "\033[1mCompile time information:\033[0m\n"\
           "Based on git revision:   %s, %s\n"
           "Based on git branch:     %s\n\n"\

            "\033[1mCompiler information:\033[0m\n"\
           "Compiler:                %s\n"\
           "Compiler flags:          %s\n"\

        #ifdef DEBUG
           "Debug:                   \033[91mTrue\033[0m\n\n"
        #else
           "Debug:                   \033[92mFalse\033[0m\n\n"
        #endif

           "\033[1mRuntime information:\033[0m\n"\
           "CPU:                    %s"\
           "Intel Turbo Boost:       %s\n"\
           "Runfile:                 %s\n"\
           "Arguments:               %s\n\n"\

           "\033[1mCorrectness information:\033[0m\n"\
           "KNN correct:             %s\n"\
           "Shapley correct:         %s\n\n"\

           "\033[1mBenchmark information:\033[0m\n"\
           "Implementation:          %s\n"\
           "Number of input sizes:   %d\n"\
           "Input sizes:             %s\n"\
           "Number of runs:          %d\n"

           ,GITREV,
           git_dirty,
           GITBRANCH,
           CC,
           CFLAGS,
           get_cpu_model(),
           (intel_turbo_boost_disabled() ? color("Disabled", GREEN) : color("Enabled", RED)),
           run_variables->runfile_path,
           args,
           (knn_correct ? color("Correct", GREEN) : color("Incorrect", RED)),
           (shapley_correct ? color("Correct", GREEN) : color("Incorrect", RED)),
           run_variables->implementation,
           run_variables->number_of_input_sizes,
           tmpbuf,
           run_variables->number_of_runs
    );
    for(size_t i=0; i<80; i++) { printf("-"); } printf("\n");


}


char *bold(char *s){
    char *res = malloc(1024);
    sprintf(res, "\033[1m%s\033[0m", s);
    return res;
}

char *color(char *s, COLOR c){
    char *res = malloc(1024);
    sprintf(res, "\033[%dm%s\033[0m", c, s);
    return res;
}

char *get_cpu_model(void){
    return exec_and_get_output("grep 'model name' /proc/cpuinfo | head -n1 | cut -f2 -d':'");
}

bool intel_turbo_boost_disabled(void){
    char *s = exec_and_get_output("cat /sys/devices/system/cpu/intel_pstate/no_turbo");
    return strncmp(s, "1", 1) == 0;
}