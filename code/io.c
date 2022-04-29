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

    // printf("Vals: ");
    // for (int i = 0; i<33;i++) {
    //     printf("%d, ", (*buffer)[i]);
    // }
    // printf("\n");
}

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
void get_true_KNN(int* result, const int* x_trn, const int* x_tst, size_t size_x_trn, size_t size_x_tst, size_t feature_len) {
    double curr_dist;

    // // Zero out the result array
    // for (int i=0; i<size_x_trn*size_x_tst; i++) {
    //     result[i] = 0;
    // }

    // Loop through each test point
    for (int i_tst=0; i_tst<size_x_tst; i_tst++) {
        // printf("zero\n");
        // memset(dist_gt, 0, size_x_trn);
        // printf("one\n");
        // Loop through each train point
        for (int i_trn=0; i_trn<size_x_trn; i_trn++){
            // calculate the distance between the two points
            curr_dist = 0;
            for (int i_feature=0; i_feature<feature_len; i_feature++) {
                curr_dist += pow(x_trn[i_trn*feature_len + i_feature] - x_tst[i_tst*feature_len + i_feature], 2);
            }
            curr_dist = sqrt(curr_dist);

            dist_gt[i_trn] = curr_dist;
        }
        // printf("two\n");
        // get the indexes that would sort the array
        int* sorted_indexes = (int*)malloc(size_x_trn * sizeof(int));
        for (int i=0; i<size_x_trn; i++) {
            sorted_indexes[i] = i;
        }
        qsort(sorted_indexes, size_x_trn, sizeof(int), compar);

        // printf("three\n");

        // copy to result array
        // printf("Copy result to res[%d+%d], \n", i_tst, size_x_tst);
        memcpy(result+(i_tst * size_x_trn), sorted_indexes, size_x_trn * sizeof(int));
    }
    printf("Get KNN done :)\n");
}

int main (void) {
    unsigned char *buffer;

    // Data in file format according to https://www.cs.toronto.edu/~kriz/cifar.html
    read_bin_file(&buffer, "../cifar-10-batches-bin/data_batch_1.bin");

    // Now split the data into features and labels array
    int* base_x_trn = (int*)malloc(sizeof(int)*10000*3072);
    int* base_y_trn = (int*)malloc(sizeof(int)*10000);

    if (base_x_trn == NULL || base_y_trn == NULL) {
        printf("Error allocating memory\n");
        return 1;
    }

    for (int i = 0; i<10000;i++) {
        base_y_trn[i] = (int)buffer[i*3073];
        for (int j = 0; j<3073;j++) {
            base_x_trn[i*3072+j] = (int)buffer[i*3073+j+1];
        }
    }

    //print base_x_trn
    printf("base_x_trn:\n");
    for (int i = 0; i<10;i++) {
        for (int j = 0; j<5;j++) {
            printf("%d, ", base_x_trn[i*3072+j]);
        }
        printf("\n");
    }

    int size_x_trn = 5;
    int size_x_tst = 10;

    int* x_tst_knn_gt = (int*)calloc(size_x_tst * size_x_trn, sizeof(int));
    dist_gt = (double*)calloc(size_x_trn, sizeof(double));

    get_true_KNN(x_tst_knn_gt, base_x_trn, base_x_trn, size_x_trn, size_x_tst, 3072);

    //print x_tst_knn_gt array
    printf("X_tst_knn_gt_array:\n");
    for (int i = 0; i<size_x_tst;i++) {
        for (int j = 0; j<size_x_trn;j++) {
            printf("%d, ", x_tst_knn_gt[i*size_x_trn+j]);
        }
        printf("\n");
    }


    return 0;
}