#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

double* dist_gt;

void read_bin_file(unsigned char** buffer, char* filename) {
    FILE *fileptr;
    long filelen;

    fileptr = fopen(filename, "rb");  // Open the file in binary mode
    fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
    filelen = ftell(fileptr);             // Get the current byte offset in the file
    rewind(fileptr);                      // Jump back to the beginning of the file

    *buffer = (unsigned char *)malloc(filelen * sizeof(unsigned char)); // Enough memory for the file

    assert(*buffer);

    fread(*buffer, filelen, 1, fileptr); // Read in the entire file
    fclose(fileptr); // Close the file 

}

void read_bin_file_known_size(double* buffer, char* filename, size_t element_count) {
    FILE* fileptr = fopen(filename, "rb");  // Open the file in binary mode
    fread(buffer, sizeof(double), element_count, fileptr); // Read in the entire file
    fclose(fileptr); // Close the file 
}

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
}

  /*
    Expected Behavior:
    - result is a 2D array of size_x_tst * size_x_trn
    - result[i][j] is the proximity rank of the jth train point regarding the ith test point.
   */
void get_true_KNN(
                int* result,
                const double* x_trn,
                const double* x_tst,
                size_t size_x_trn,
                size_t size_x_tst,
                size_t feature_len ) {
    double curr_dist;

    // Loop through each test point
    for (int i_tst=0; i_tst<size_x_tst; i_tst++) {
        // Loop through each train point
        for (int i_trn=0; i_trn<size_x_trn; i_trn++){
            // calculate the distance between the two points, just pythagoras...
            curr_dist = 0;
            for (int i_feature=0; i_feature<feature_len; i_feature++) {
                curr_dist += 
                pow(x_trn[i_trn*feature_len + i_feature] - 
                        x_tst[i_tst*feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            dist_gt[i_trn] = curr_dist;
        }
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(size_x_trn * sizeof(int));
        for (int i=0; i<size_x_trn; i++) {
            sorted_indexes[i] = i;
        }

        // Sanity check in order to compare with python
        // printf("dist_gt:\n");
        //     for (int j = 0; j<10;j++) {
        //         printf("%f, ", dist_gt[j]);
        //     }
        //     printf("\n");

        qsort(sorted_indexes, size_x_trn, sizeof(int), compar);

        // copy to result array
        memcpy(result+(i_tst * size_x_trn), sorted_indexes, size_x_trn * sizeof(int));
    }
    printf("Get KNN done :)\n");
}

void compute_single_unweighted_knn_class_shapley(double* sp_gt,
                                                const double* x_trn,
                                                const double* y_trn,
                                                const int* x_tst_knn_gt,
                                                const double* y_tst,
                                                size_t size_x_trn,
                                                size_t size_x_tst,
                                                size_t size_y_tst,
                                                double K ){

    printf("\nStart Shapley computation:\n");
    for(int j=0; j<size_x_tst;j++){
        // printf("  iteration: j=%d\n", j);
        // Line 3 of Algo 1
        int offset = x_tst_knn_gt[j*size_x_trn+size_x_trn-1];
        double tmp = (y_trn[offset] == y_tst[j]) ? 1.0 : 0.0;
        sp_gt[j*size_x_trn + offset] = tmp / size_x_trn; 

        for (int i=size_x_trn-2; i>-1; i--) {
            // printf("    i=%d\n", i);
            int index_j_i = j*size_x_trn+i;

            double s_j_alpha_i_plus_1 = sp_gt[j*size_x_trn + x_tst_knn_gt[index_j_i+1]];
            double difference = (double)(y_trn[x_tst_knn_gt[index_j_i]] == y_tst[j]) - 
                                        (double)(y_trn[x_tst_knn_gt[index_j_i+1]] == y_tst[j]);
            double min_K_i = K < i+1 ? K : i+1;

            // printf("      s_j=%f\n", s_j_alpha_i_plus_1);
            // printf("      diff=%f (%f,%f), (%f,%f)\n", difference, y_trn[x_tst_knn_gt[index_j_i]], y_tst[j], y_trn[x_tst_knn_gt[index_j_i+1]], y_tst[j]);
            // printf("      min_=%f\n", min_K_i);

            sp_gt[j*size_x_trn + x_tst_knn_gt[index_j_i]] = s_j_alpha_i_plus_1 + (difference / K) * (min_K_i / (i+1));
        }
    }
    printf("Shapley Values done :)\n");
}

int main (void) {
    size_t feature_len = 2048;

    //init the training data
    double* base_x_trn = (double*)malloc(sizeof(double)*50000*feature_len);
    double* base_y_trn = (double*)malloc(sizeof(double)*50000);

    assert(base_x_trn && base_y_trn);

    //Read binary training data into the arrays
    read_bin_file_known_size(base_x_trn, "../data/features/cifar10/train_features.bin", 50000*feature_len);
    read_bin_file_known_size(base_y_trn, "../data/features/cifar10/train_lables.bin", 50000);

    // printf("base_y:\n");
    // for (int i = 0; i<10;i++) {
    //     printf("%f, ", base_y_trn[i]);
    // }
    // printf("\n");

    // Define start and lengths of the train and test data, as in the python implementation
    double* x_trn = &(base_x_trn[49990*feature_len]);
    size_t size_x_trn = 10;

    // double* y_trn = &base_y_trn[49990];
    double y_trn[10] = {4.0, 2.0, 0.0, 1.0, 0.0, 2.0, 6.0, 9.0, 1.0, 1.0};
    size_t size_y_trn = 10;

    double* x_tst = base_x_trn;
    size_t size_x_tst = 5;

    // double* y_tst = base_y_trn;
    double y_tst[10] = {6.0, 9.0, 9.0, 4.0, 1.0, 1.0, 2.0, 7.0, 8.0, 3.0};
    size_t size_y_tst = 5;

    //Sanity check, to make sure that C and Python are doing the same thing
    printf("x_trn:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            printf("%f, ", x_trn[i*feature_len+j]);
        }
        printf("\n");
    }

   printf("\n");
   printf("x_tst:\n");
    for (int i = 0; i<3;i++) {
        for (int j = 0; j<3;j++) {
            printf("%f, ", x_tst[i*feature_len+j]);
        }
        printf("\n");
    }

 printf("\n");
   printf("y_tst:\n");
    for (int i = 0; i<size_y_tst;i++) {
        printf("%f, ", y_tst[i]);
    }
    printf("\n");
    printf("\n");

    // Allocate resulting arrays
    int* x_tst_knn_gt = (int*)calloc(size_x_tst * size_x_trn, sizeof(int));
    // This array gets defined in the outermost scope, such that the pointer is available in the compar function
    dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    get_true_KNN(x_tst_knn_gt, x_trn, x_tst, size_x_trn, size_x_tst, feature_len);

    // print x_tst_knn_gt array
    printf("\n");
    printf("X_tst_knn_gt_array:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            printf("%d, ", x_tst_knn_gt[i*size_x_trn+j]);
        }
        printf("\n");
    }

    double* sp_gt = (double*)calloc(size_x_tst * size_x_trn, sizeof(double));
    compute_single_unweighted_knn_class_shapley(sp_gt, x_trn, y_trn, x_tst_knn_gt, y_tst, size_x_trn, size_x_tst, size_y_tst, 1.0);

    // print x_tst_knn_gt array
    printf("\n");
    printf("Shapley Values:\n");
    for (int i = 0; i<5;i++) {
        for (int j = 0; j<10;j++) {
            printf("%f, ", sp_gt[i*size_x_trn+j]);
        }
        printf("\n");
    }

    free(base_x_trn);
    free(base_y_trn);
    free (x_tst_knn_gt);
    free(dist_gt);
    free(sp_gt);

    return 0;
}