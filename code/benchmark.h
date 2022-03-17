#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

void start_benchmark(run_variables_t *run_variables);
uint64_t measure_single_run(run_variables_t *run_variables, int input_size);

#endif
