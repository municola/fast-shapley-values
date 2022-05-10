#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdbool.h>
#include <stdint.h>

#include "runfile.h"

// Use "make debug" to enable debug prints and debug symbols, etc.
#ifdef DEBUG
    #define debug_print(fmt, ...) \
                do { fprintf(stderr, fmt, __VA_ARGS__); } while (0)
#else
    #define debug_print(fmt, ...) 
#endif


enum COLOR {
    RED=91,
    GREEN=92
};

typedef enum COLOR COLOR;

typedef struct run_variables {
    bool quiet;
    char *implementation;

    /* void* is actually context_t* */
    uint64_t (*shapley_measurement_func)(void *);
    bool (*correctness_check_func)(struct run_variables *, void *);
    
    // void (*knn_func)(void *);
    // void (*shapley_func)(void *);
    
    int number_of_runs;
    int number_of_input_sizes;
    int *input_sizes;

    char *runfile_path;
    runfile_t *runfile;

} run_variables_t;


char *exec_and_get_output(char *cmd);
void parse_cmd_options(int argc, char **argv, run_variables_t *run_variables);
void intro(int argc, char **argv, run_variables_t *run_variables);
char *get_cpu_model(void);
bool intel_turbo_boost_disabled(void);

char *bold(char *s);
char *color(char *s, COLOR c);




#endif