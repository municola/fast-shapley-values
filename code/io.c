#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

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