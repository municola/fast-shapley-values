#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "utils.h"

extern int NUMBER_OF_RUNS;

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


void print_intro(int argc, char **argv){
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

           "\033[1mRun time information:\033[0m\n"\
           "CPU:                    %s"\
           "Intel Turbo Boost:       %s\n"\
           "Runfile:                 %s\n"\
           "Arguments:               %s\n\n"\

           "\033[1mBenchmark information:\033[0m\n"\
           "Number of runs:          %d\n"

           ,GITREV,
           git_dirty,
           GITBRANCH,
           CC,
           CFLAGS,
           get_cpu_model(),
           (intel_turbo_boost_disabled() ? color("Disabled", GREEN) : color("Enabled", RED)),
           "./run/1234.run",
           args,
           NUMBER_OF_RUNS
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