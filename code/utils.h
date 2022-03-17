#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdbool.h>

enum COLOR {
    RED=91,
    GREEN=92
};

typedef enum COLOR COLOR;


char *exec_and_get_output(char *cmd);
void print_intro(int argc, char **argv);
char *get_cpu_model(void);
bool intel_turbo_boost_disabled(void);

char *bold(char *s);
char *color(char *s, COLOR c);


#endif