#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdbool.h>

enum COLOR {
    RED=91,
    GREEN=92
};

typedef enum COLOR COLOR;

typedef struct run_variables {
    bool quiet;
    int number_of_runs;
    char *runfile;
} run_variables_t;


char *exec_and_get_output(char *cmd);
void parse_cmd_options(int argc, char **argv, run_variables_t *run_variables);
void print_intro(int argc, char **argv, run_variables_t *run_variables);
char *get_cpu_model(void);
bool intel_turbo_boost_disabled(void);

char *bold(char *s);
char *color(char *s, COLOR c);




#endif