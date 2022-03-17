#ifndef __RUNFILE_H__
#define __RUNFILE_H__

#include <stdint.h>

typedef struct runfile {
    FILE *handle;
    char *run_infos;
    size_t info_bytes;
    char *benchmarks;
    size_t benchmark_bytes;
} runfile_t;


void init_runfile(runfile_t *runfile, char *path);
void close_runfile(runfile_t *runfile);
void add_run_info(runfile_t *runfile, char *key, char *val);
void add_run_info_int(runfile_t *runfile, char *key, int i);
void add_run_info_raw(runfile_t *runfile, char *key, char *val);
void add_benchmark(runfile_t *runfile, uint64_t cycles);



#endif