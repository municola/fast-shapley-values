#include <stdlib.h>
#include <stdio.h>

void ReadBinFile(char* buffer, char* filename)
{
    FILE *fileptr;
    long filelen;

    fileptr = fopen(filename, "rb");  // Open the file in binary mode
    fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
    filelen = ftell(fileptr);             // Get the current byte offset in the file
    rewind(fileptr);                      // Jump back to the beginning of the file

    printf("Filelen:%d\n", filelen);
    buffer = (char *)malloc(filelen * sizeof(char)); // Enough memory for the file
    fread(buffer, filelen, 1, fileptr); // Read in the entire file
    fclose(fileptr); // Close the file 

    for (int i = 0; i<10;i++) {
        printf("%d, ", (int)buffer[i]);
    }
}

int main (void)
{
    char *buffer;

    // Data in file format according to https://www.cs.toronto.edu/~kriz/cifar.html
    ReadBinFile(buffer, "../cifar-10-batches-bin/data_batch_1.bin");


    // Now split the data into features and labels array
    int* x_trn = (int*)malloc(sizeof(int)*10000*3072);
    int* y_trn = (int*)malloc(sizeof(int)*10000);

    // Well... this is not quite working yet.
    // Help, please? :)
    // TODO::
    for (int i = 0; i<10000;i++) {
        y_trn[i] = (int)buffer[i*3072];
        printf("\ni:%d", i);
        for (int j = 1; j<3072;j++) {
            x_trn[i*3072+j] = (int)buffer[i*3072+j];
        }
    }
    printf("\n");

    // Print first 10 elements for sanity check :)
    for (int i = 0; i<10;i++) {
        printf("%d, ", (int)x_trn[i]);
    }

    return 0;
}