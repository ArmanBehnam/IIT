// readfile.c

#include "readfile.h"

FILE* currentFile = NULL;

FILE* open_file(const char *filename) {
    currentFile = fopen(filename, "r");
    return currentFile;
}

int read_int(int *value) {
    return fscanf(currentFile, "%d", value) == 1 ? 0 : -1;
}

int read_string(char *str, int maxLen) {
    return fscanf(currentFile, "%63s", str) == 1 ? 0 : -1; // 63 to allow for null-terminator
}

int read_float(float *value) {
    return fscanf(currentFile, "%f", value) == 1 ? 0 : -1;
}

void close_file(FILE *file) {
    if (file) {
        fclose(file);
    }
    currentFile = NULL;
}
