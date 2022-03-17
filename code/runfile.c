#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "runfile.h"


void init_runfile(runfile_t *runfile, char *path){
    runfile->handle = fopen(path, "w");
    runfile->run_infos = calloc(1, 1024*1024);
    runfile->benchmarks = calloc(1, 1024*1024);
    runfile->info_bytes = 0;
    runfile->benchmark_bytes = 0;
}


void add_run_info(runfile_t *runfile, char *key, char *val){
    char buf[1024];
    sprintf(buf, "    \"%s\" : \"%s\",\n", key, val);
    strcpy(runfile->run_infos+runfile->info_bytes, buf);
    runfile->info_bytes += strlen(buf);
}


void add_benchmark(runfile_t *runfile, uint64_t cycles){
    char buf[1024];
    sprintf(buf, "        %u,\n", cycles);
    strcpy(runfile->benchmarks+runfile->benchmark_bytes, buf);
    runfile->benchmark_bytes += strlen(buf);
}


void close_runfile(runfile_t *runfile){
    // Trim last benchmark data json
    *(runfile->benchmarks+runfile->benchmark_bytes-2) = '\0';
    
    fprintf(runfile->handle,
        "{\n"\
        "%s\n\n"\
        "    \"benchmarks\" : [\n%s\n    ]\n}\n",
    
        runfile->run_infos,
        runfile->benchmarks
    );

    fclose(runfile->handle);
    free(runfile->benchmarks);
    free(runfile->run_infos);
}