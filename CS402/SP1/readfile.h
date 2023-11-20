// readfile.h

#ifndef READFILE_H
#define READFILE_H

#include <stdio.h>
extern FILE* currentFile;

#include <stdio.h>

FILE* open_file(const char *filename);
int read_int(int *value);
int read_string(char *str, int maxLen);
int read_float(float *value);
void close_file(FILE *file);

#endif
