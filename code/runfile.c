#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "utils.h"
#include "runfile.h"


void init_runfile(runfile_t *runfile, char *path, int number_of_input_sizes){
    runfile->handle = fopen(path, "w");
    runfile->run_infos = calloc(1, 128*1024);
    runfile->benchmarks = calloc(number_of_input_sizes, 128*1024);
    runfile->info_bytes = 0;
    runfile->benchmark_bytes = calloc(number_of_input_sizes, sizeof(int));
}


void add_run_info(runfile_t *runfile, char *key, char *val){
    char buf[1024];
    sprintf(buf, "    \"%s\" : \"%s\",\n", key, val);
    strcpy(runfile->run_infos+runfile->info_bytes, buf);
    runfile->info_bytes += strlen(buf);
}


void add_run_info_int(runfile_t *runfile, char *key, int i){
    char buf[1024];
    sprintf(buf, "    \"%s\" : %d,\n", key, i);
    strcpy(runfile->run_infos+runfile->info_bytes, buf);
    runfile->info_bytes += strlen(buf);
}

void add_run_info_raw(runfile_t *runfile, char *key, char *val){
    char buf[1024];
    sprintf(buf, "    \"%s\" : %s,\n", key, val);
    strcpy(runfile->run_infos+runfile->info_bytes, buf);
    runfile->info_bytes += strlen(buf);
}

void add_benchmark(runfile_t *runfile, int *input_sizes, int number_of_input_sizes, int input_size, uint64_t cycles){
    // First find the corresponding input_size_no
    int input_size_no = -1;
    for(int i=0; i<number_of_input_sizes; i++){
        if(input_sizes[i] == input_size){
            input_size_no = i;
            break;
        }
    }

    // If it could not be found -> error in the infrastructure
    if(input_size_no == -1){
        printf("Error: Could not find input size: %d\n\n", input_size);
        abort();
    }

    char buf[1024];
    sprintf(buf, "            %ld,\n", cycles);
    strcpy(runfile->benchmarks+input_size_no*1024+runfile->benchmark_bytes[input_size_no], buf);
    runfile->benchmark_bytes[input_size_no] += strlen(buf);
}


void close_runfile(runfile_t *runfile, int *input_sizes, int number_of_input_sizes){
    // Now collect all the benchmark information
    size_t offset = 0;
    char benchmarks_buf[128*1024];
    for(int input_size_no=0; input_size_no<number_of_input_sizes; input_size_no++){
        // Trim last benchmark data json
        *(runfile->benchmarks+input_size_no*1024+runfile->benchmark_bytes[input_size_no]-2) = '\0';

        sprintf(benchmarks_buf+offset, "        \"%d\" : [\n%s\n        ],\n", input_sizes[input_size_no], runfile->benchmarks+input_size_no*1024);
        offset += strlen(benchmarks_buf+offset);
    }

    // cut off trailing ,
    *(benchmarks_buf+offset-2) = '\0';

    fprintf(runfile->handle,
        "{\n"\
        "%s\n\n"\
        "    \"benchmarks\" : {\n%s\n    }\n}\n",
    
        runfile->run_infos,
        benchmarks_buf
    );

    fclose(runfile->handle);
    free(runfile->benchmarks);
    free(runfile->run_infos);
}