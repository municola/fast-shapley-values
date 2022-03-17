CC = gcc
CFLAGS = -O3 -fno-tree-vectorize

CFILES = shapley_values.c utils.c
CHEADERS = shapley_values.c utils.h
FILES = $(CFILES) $(CHEADERS)

GITREV = $(shell git log -1 --format=%h)
GITBRANCH = $(shell git rev-parse --abbrev-ref HEAD)
GITSTATUS = $(shell git diff --stat | head -n1)


all: project

project: $(FILES)
	$(CC) $(CFLAGS) -DCC='"$(CC)"' -DCFLAGS='"$(CFLAGS)"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -o shapley_values $(CFILES)

debug: $(FILES) 
	$(CC) -pg -g -DCC='"$(CC)"' -DCFLAGS='"-pg -g"' -DGITREV='"$(GITREV)"' -DGITBRANCH='"$(GITBRANCH)"' -DGITSTATUS='"$(GITSTATUS)"' -DDEBUG -o shapley_values $(CFILES)
