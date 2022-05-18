#include <stdlib.h>
#include <stdint.h>

#include "utils.h"

void get_true_exact_KNN(void *context);
void knn__exact_opt5(void *context);
void knn__exact_opt4(void *context);
void knn__exact_opt3(void *context);
void knn__exact_opt2(void *context);
void knn__exact_opt1(void *context);
void knn__exact_opt(void *context);
void knn_exact_base(void *context);

int compar(const void* a, const void* b);
int compar_block(const void *a, const void *b);