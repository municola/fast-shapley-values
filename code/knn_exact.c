#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "tsc_x86.h"
#include "benchmark.h"


double* dist_gt;
double* dist_gt_row;

// Custom compare function for sorting, since we try to replicate numpys argsort function
// We don't want to return the sorted array, but rather the indices that would sort the array
// To achieve this, an array [0, 1, ... N] is initialized and the sorting is performed there, 
// but the comparisons are done on the original dist_gt array
// 
// This was done, such that the C qsort function can be used.
int compar (const void *a, const void *b)
{
  int aa = *((int *) a), bb = *((int *) b);
  if (dist_gt[aa] < dist_gt[bb])
    return -1;
  if (dist_gt[aa] == dist_gt[bb])
    return 0;
  if (dist_gt[aa] > dist_gt[bb])
    return 1;
  return 1;
}


int compar_block (const void *a, const void *b)
{
  int aa = *((int *) a), bb = *((int *) b);
  if (dist_gt_row[aa] < dist_gt_row[bb])
    return -1;
  if (dist_gt_row[aa] == dist_gt_row[bb])
    return 0;
  if (dist_gt_row[aa] > dist_gt_row[bb])
    return 1;
  return 1;
}


/*
Expected Behavior:
- result is a 2D array of size_x_tst * size_x_trn
- result[i][j] is the proximity rank of the jth train point regarding the ith test point.
*/
void get_true_exact_KNN(void *context_ptr) {
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    //dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature++) {
                curr_dist += 
                pow(context->x_trn[i_trn*context->feature_len + i_feature] - 
                        context->x_tst[i_tst*context->feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        // Sanity check in order to compare with python
        debug_print("%s", "get_true_exact_KNN: dist_gt:\n");
            for (int j = 0; j<10;j++) {
                debug_print("%f, ", dist_gt[j]);
            }
            debug_print("%s", "\n");

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));

        debug_print("%s", "Sorted Indexes\n");
        for (int j = 0; j<context->size_x_trn;j++) {
            debug_print("%d, ", sorted_indexes[j]);
        }
        debug_print("%s", "\n");


        // Sanity check in order to compare with python
        debug_print("%s", "get_true_exact_KNN: x_test_knn_gt:\n");
        for (int j = 0; j<context->size_x_trn; j++) {
            debug_print("%d, ", context->x_test_knn_gt[i_tst*context->size_x_trn + j]);
        }
        debug_print("%s", "\n");   
    }
}


void knn_exact_base(void *context_ptr) {
    /* knn__exact_opt   knn_exact_base */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    //dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature++) {
                curr_dist += 
                pow(context->x_trn[i_trn*context->feature_len + i_feature] - 
                        context->x_tst[i_tst*context->feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}



void knn__exact_opt(void *context_ptr) {
    /* opt7: Blocking + Blocked sqrt*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 8;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {

            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        for (int k1=k; k1<k+B; k1++){
                            // c[i1*test_length + j1] += (a[i1*test_length + k1]-b[j1*train_length+k1])^2
                            double a = context->x_tst[i1*f_length + k1];
                            double b = context->x_trn[j1*f_length + k1];
                            double ab_2 = pow((a-b),2);
                            context->dist_gt[i1*train_length + j1] += ab_2;
                        }
                    }
                }
            }

            // Square root
            for (int i2=i; i2<i+B; i2++) {
                for (int j2=j; j2<j+B; j2++) {
                    double t = sqrt(context->dist_gt[i2*train_length + j2]);
                    context->dist_gt[i2*train_length + j2] = t;
                }
            }
            
        }
    }

    // Sorting
    for (int i_tst=0; i_tst<test_length; i_tst++) {
        // get the indexes that would sort the array
        dist_gt_row = &context->dist_gt[i_tst*train_length];
        int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
        for (int i=0; i<train_length; i++) {
            sorted_indexes[i] = i;
        }
        qsort(sorted_indexes, train_length, sizeof(int), compar_block);
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}




void knn__exact_opt6(void *context_ptr) {
    /* opt6: Blocking */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    int B = 12;
    int train_length = context->size_x_trn;
    int test_length = context->size_x_tst;
    int f_length = context->feature_len;

    assert(train_length % B == 0);
    assert(test_length % B == 0);
    assert(f_length % B == 0);

    for (int i=0; i<test_length; i+=B) {
        for (int j=0; j<train_length; j+=B) {
            for (int k=0; k<f_length; k+=B) {
                /* B x B Block Calculation */
                for (int i1=i; i1<i+B; i1++){
                    for (int j1=j; j1<j+B; j1++){
                        for (int k1=k; k1<k+B; k1++){
                            // c[i1*test_length + j1] += (a[i1*test_length + k1]-b[j1*train_length+k1])^2
                            double a = context->x_tst[i1*f_length + k1];
                            double b = context->x_trn[j1*f_length + k1];
                            double ab_2 = pow((a-b),2);
                            context->dist_gt[i1*train_length + j1] += ab_2;
                        }
                    }
                }
            }
        }
    }

    for (int i_tst=0; i_tst<test_length; i_tst++) {
        for (int i_trn=0; i_trn<train_length; i_trn++) {
            context->dist_gt[i_tst*train_length + i_trn] = sqrt(context->dist_gt[i_tst*train_length + i_trn]);
        }
    }

    // Sorting
    for (int i_tst=0; i_tst<test_length; i_tst++) {
        // get the indexes that would sort the array
        dist_gt_row = &context->dist_gt[i_tst*train_length];
        int* sorted_indexes = (int*)malloc(train_length * sizeof(int));
        for (int i=0; i<train_length; i++) {
            sorted_indexes[i] = i;
        }
        qsort(sorted_indexes, train_length, sizeof(int), compar_block);
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}


void knn__exact_opt5(void *context_ptr) {
    /* opt5: based on opt1: 4 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature+=4) {
                double a1 = context->x_trn[i_trn*context->feature_len + i_feature];
                double b1 = context->x_tst[i_tst*context->feature_len + i_feature];
                double a2 = context->x_trn[i_trn*context->feature_len + i_feature+1];
                double b2 = context->x_tst[i_tst*context->feature_len + i_feature+1];
                double a3 = context->x_trn[i_trn*context->feature_len + i_feature+2];
                double b3 = context->x_tst[i_tst*context->feature_len + i_feature+2];
                double a4 = context->x_trn[i_trn*context->feature_len + i_feature+3];
                double b4 = context->x_tst[i_tst*context->feature_len + i_feature+3];
                
                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;

                curr_dist += ab1_2 + ab2_2 + ab3_2 + ab4_2;
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}



void knn__exact_opt4(void *context_ptr) {
    /*opt4: 8 Accumulator */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    //dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature+=8) {
                double a1 = context->x_trn[i_trn*context->feature_len + i_feature];
                double b1 = context->x_tst[i_tst*context->feature_len + i_feature];
                double a2 = context->x_trn[i_trn*context->feature_len + i_feature+1];
                double b2 = context->x_tst[i_tst*context->feature_len + i_feature+1];
                double a3 = context->x_trn[i_trn*context->feature_len + i_feature+2];
                double b3 = context->x_tst[i_tst*context->feature_len + i_feature+2];
                double a4 = context->x_trn[i_trn*context->feature_len + i_feature+3];
                double b4 = context->x_tst[i_tst*context->feature_len + i_feature+3];
                double a5 = context->x_trn[i_trn*context->feature_len + i_feature+4];
                double b5 = context->x_tst[i_tst*context->feature_len + i_feature+4];
                double a6 = context->x_trn[i_trn*context->feature_len + i_feature+5];
                double b6 = context->x_tst[i_tst*context->feature_len + i_feature+5];
                double a7 = context->x_trn[i_trn*context->feature_len + i_feature+6];
                double b7 = context->x_tst[i_tst*context->feature_len + i_feature+6];
                double a8 = context->x_trn[i_trn*context->feature_len + i_feature+7];
                double b8 = context->x_tst[i_tst*context->feature_len + i_feature+7];

                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                double ab5 = a5-b5;
                double ab6 = a6-b6;
                double ab7 = a7-b7;
                double ab8 = a8-b8;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;
                double ab5_2 = ab5*ab5;
                double ab6_2 = ab6*ab6;
                double ab7_2 = ab7*ab7;
                double ab8_2 = ab8*ab8;


                curr_dist += ab1_2 + ab2_2 + ab3_2 + ab4_2 + ab5_2 + ab6_2 + ab7_2 + ab8_2;
            }
            
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}


void knn__exact_opt3(void *context_ptr) {
    /*opt3: 8 Accumulator + accumulated sums */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    //dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            double ab1_sum = 0;
            double ab2_sum = 0;
            double ab3_sum = 0;
            double ab4_sum = 0;
            double ab5_sum = 0;
            double ab6_sum = 0;
            double ab7_sum = 0;
            double ab8_sum = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature+=8) {
                double a1 = context->x_trn[i_trn*context->feature_len + i_feature];
                double b1 = context->x_tst[i_tst*context->feature_len + i_feature];
                double a2 = context->x_trn[i_trn*context->feature_len + i_feature+1];
                double b2 = context->x_tst[i_tst*context->feature_len + i_feature+1];
                double a3 = context->x_trn[i_trn*context->feature_len + i_feature+2];
                double b3 = context->x_tst[i_tst*context->feature_len + i_feature+2];
                double a4 = context->x_trn[i_trn*context->feature_len + i_feature+3];
                double b4 = context->x_tst[i_tst*context->feature_len + i_feature+3];
                double a5 = context->x_trn[i_trn*context->feature_len + i_feature+4];
                double b5 = context->x_tst[i_tst*context->feature_len + i_feature+4];
                double a6 = context->x_trn[i_trn*context->feature_len + i_feature+5];
                double b6 = context->x_tst[i_tst*context->feature_len + i_feature+5];
                double a7 = context->x_trn[i_trn*context->feature_len + i_feature+6];
                double b7 = context->x_tst[i_tst*context->feature_len + i_feature+6];
                double a8 = context->x_trn[i_trn*context->feature_len + i_feature+7];
                double b8 = context->x_tst[i_tst*context->feature_len + i_feature+7];

                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                double ab5 = a5-b5;
                double ab6 = a6-b6;
                double ab7 = a7-b7;
                double ab8 = a8-b8;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;
                double ab5_2 = ab5*ab5;
                double ab6_2 = ab6*ab6;
                double ab7_2 = ab7*ab7;
                double ab8_2 = ab8*ab8;

                ab1_sum += ab1_2;
                ab2_sum += ab2_2;
                ab3_sum += ab3_2;
                ab4_sum += ab4_2;
                ab5_sum += ab5_2;
                ab6_sum += ab6_2;
                ab7_sum += ab7_2;
                ab8_sum += ab8_2;
                
            }
            curr_dist += ab1_sum + ab2_sum + ab3_sum + ab4_sum + ab5_sum + ab6_sum + ab7_sum + ab8_sum;

            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}





void knn__exact_opt2(void *context_ptr) {
    /* opt2: 4 Accumulators + accumulated sums*/
    context_t *context = (context_t *) context_ptr;
    double curr_dist;

    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            double ab1_sum = 0;
            double ab2_sum = 0;
            double ab3_sum = 0;
            double ab4_sum = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature+=4) {
                double a1 = context->x_trn[i_trn*context->feature_len + i_feature];
                double b1 = context->x_tst[i_tst*context->feature_len + i_feature];
                double a2 = context->x_trn[i_trn*context->feature_len + i_feature+1];
                double b2 = context->x_tst[i_tst*context->feature_len + i_feature+1];
                double a3 = context->x_trn[i_trn*context->feature_len + i_feature+2];
                double b3 = context->x_tst[i_tst*context->feature_len + i_feature+2];
                double a4 = context->x_trn[i_trn*context->feature_len + i_feature+3];
                double b4 = context->x_tst[i_tst*context->feature_len + i_feature+3];

                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;

                ab1_sum += ab1_2;
                ab2_sum += ab2_2;
                ab3_sum += ab3_2;
                ab4_sum += ab4_2;
                
            }
            curr_dist += ab1_sum + ab2_sum + ab3_sum + ab4_sum;

            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}



void knn__exact_opt1(void *context_ptr) {
    /* opt1: 4 Accumulators */
    context_t *context = (context_t *) context_ptr;
    double curr_dist;
    // Loop through each test point
    for (int i_tst=0; i_tst<context->size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<context->size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<context->feature_len; i_feature+=4) {
                double a1 = context->x_trn[i_trn*context->feature_len + i_feature];
                double b1 = context->x_tst[i_tst*context->feature_len + i_feature];
                double a2 = context->x_trn[i_trn*context->feature_len + i_feature+1];
                double b2 = context->x_tst[i_tst*context->feature_len + i_feature+1];
                double a3 = context->x_trn[i_trn*context->feature_len + i_feature+2];
                double b3 = context->x_tst[i_tst*context->feature_len + i_feature+2];
                double a4 = context->x_trn[i_trn*context->feature_len + i_feature+3];
                double b4 = context->x_tst[i_tst*context->feature_len + i_feature+3];
                

                double ab1 = a1-b1;
                double ab2 = a2-b2;
                double ab3 = a3-b3;
                double ab4 = a4-b4;
                
                double ab1_2 = ab1*ab1;
                double ab2_2 = ab2*ab2;
                double ab3_2 = ab3*ab3;
                double ab4_2 = ab4*ab4;

                curr_dist += ab1_2 + ab2_2 + ab3_2 + ab4_2;
                
            }
            curr_dist = sqrt(curr_dist);

            context->dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(context->size_x_trn * sizeof(int));
        for (int i=0; i<context->size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        qsort(sorted_indexes, context->size_x_trn, sizeof(int), compar);
        memcpy(context->x_test_knn_gt+(i_tst * context->size_x_trn), sorted_indexes, context->size_x_trn * sizeof(int));
    }
}


