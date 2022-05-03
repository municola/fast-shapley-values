CC = gcc
CFLAGS = -lm -O3 -fno-tree-vectorize

CFILES = shapley_values.c base_exact_shapley.c base_mc_approx_shapley.c io.c utils.c runfile.c benchmark.c correctness.c implementations.c
CHEADERS = shapley_values.h base_exact_shapley.h base_mc_approx_shapley.h io.h utils.h runfile.h benchmark.h correctness.h implementations.h
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
	$(CC) -pg -g -DCC='"$(CC)"' -DCFLAGS='"-pg -g"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -DDEBUG -o shapley_values $(CFILES) -lm
