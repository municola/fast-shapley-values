#ifndef __RUNFILE_H__
#define __RUNFILE_H__

#include <stdint.h>

typedef struct runfile {
    FILE *handle;
    char *run_infos;
    size_t info_bytes;
    char *benchmarks;
    int *benchmark_bytes;
} runfile_t;


void init_runfile(runfile_t *runfile, char *path, int number_of_input_sizes);
void close_runfile(runfile_t *runfile, int *input_sizes, int number_of_input_sizes);
void add_run_info(runfile_t *runfile, char *key, char *val);
void add_run_info_int(runfile_t *runfile, char *key, int i);
void add_run_info_raw(runfile_t *runfile, char *key, char *val);
void add_benchmark(runfile_t *runfile, int *input_sizes, int number_of_input_sizes, int input_size, uint64_t cycles);



#endif