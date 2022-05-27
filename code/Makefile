CC = gcc
CFLAGS = -std=c11 -Wall -lm -O3 -mavx -mavx2 -mfma -march=native

CFILES = shapley_values.c base_exact_shapley.c base_mc_approx_shapley.c io.c utils.c runfile.c benchmark.c correctness.c implementations.c knn_exact.c knn_approx.c combined_exact_knn_shapley.c simdxorshift128plus.c
CHEADERS = shapley_values.h base_exact_shapley.h base_mc_approx_shapley.h io.h utils.h runfile.h benchmark.h correctness.h implementations.h knn_exact.h knn_approx.h combined_exact_knn_shapley.h simdxorshift128plus.h
FILES = $(CFILES) $(CHEADERS)

GITREV = $(shell git log -1 --format=%h)
GITBRANCH = $(shell git rev-parse --abbrev-ref HEAD)
GITSTATUS = $(shell git diff --stat | head -n1)


all: project

run: project
	./shapley_values

project: $(FILES)
	$(CC) $(CFLAGS) -DCC='"$(CC)"' -DCFLAGS='"$(CFLAGS)"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -o shapley_values $(CFILES) -lm

debug: $(FILES) 
	$(CC) -std=c11 -Wall -pg -g -mavx -mavx2 -mfma -march=native -DCC='"$(CC)"' -DCFLAGS='"-pg -g -std=c11 -mavx -mavx2 -mfma -march=native"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -DDEBUG -o shapley_values $(CFILES) -lm

perf: $(FILES) 
	$(CC) -std=c11 -Wall -g -mavx -mavx2 -mfma -march=native -DCC='"$(CC)"' -DCFLAGS='"-g -mavx -mavx2 -mfma -march=native"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -o shapley_values $(CFILES) -lm
