#ifndef CC
#error Compiler not set (compile e.g. using -DCC=gcc)
#endif

#ifndef CFLAGS
#error Compiler flags not set (compile e.g. using -DCFLAGS=-O3)
#endif 

#ifndef GITREV
#define GITREV "No git info available"
#endif

#ifndef GITBRANCH
#define GITBRANCH ""
#endif

#ifndef GITSTATUS
#define GITSTATUS ""
#endif