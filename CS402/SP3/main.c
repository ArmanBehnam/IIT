#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function prototypes
float* readData(const char* filename, int* size, int* capacity);
float calculateMean(float* data, int size);
float calculateMedian(float* data, int size);
float calculateStdDev(float* data, int size, float mean);


float* readData(const char* filename, int* size, int* capacity) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    *size = 0;
    *capacity = 20;
    float* data = malloc(*capacity * sizeof(float));
    if (data == NULL) {
        perror("Malloc failed");
        exit(EXIT_FAILURE);
    }

    while (fscanf(file, "%f", &data[*size]) == 1) {
        (*size)++;
        if (*size == *capacity) {
            *capacity *= 2;
            data = realloc(data, *capacity * sizeof(float));
            if (data == NULL) {
                perror("Realloc failed");
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
    return data;
}


float calculateMean(float* data, int size) {
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return size > 0 ? sum / size : 0.0;
}

int compareFloats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

float calculateMedian(float* data, int size) {
    qsort(data, size, sizeof(float), compareFloats);
    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.0;
    } else {
        return data[size / 2];
    }
}

float calculateStdDev(float* data, int size, float mean) {
    float sumSqDiff = 0.0;
    for (int i = 0; i < size; i++) {
        sumSqDiff += (data[i] - mean) * (data[i] - mean);
    }
    return sqrt(sumSqDiff / size);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int size, capacity;
    float* data = readData(argv[1], &size, &capacity);

    float mean = calculateMean(data, size);
    float median = calculateMedian(data, size); // Data is sorted here
    float stdDev = calculateStdDev(data, size, mean);

    printf("Results:\n--------\n");
    printf("Num values: %d\n", size);
    printf("mean: %.3f\n", mean);
    printf("median: %.3f\n", median);
    printf("stddev: %.3f\n", stdDev);
    printf("Unused array capacity: %d\n", capacity - size);

    free(data);
    return EXIT_SUCCESS;
}
