# BasicStats Program

## Overview
The BasicStats program is a statistical analysis tool developed in C. It is designed to read numerical data from a file and compute basic statistical measures, such as mean, median, and standard deviation. The program dynamically allocates memory to handle datasets of varying sizes, making it versatile for small to very large data sets.

## Features
- **Dynamic Data Handling**: Efficiently processes datasets of any size without the need for re-compilation.
- **Statistical Analysis**: Computes mean, median, and standard deviation of the dataset.
- **Memory Management**: Utilizes dynamic memory allocation to handle data efficiently.

## Prerequisites
- GCC or an equivalent C compiler.
- A text editor or Integrated Development Environment (IDE) for editing code.

## Compilation
Navigate to the source directory and compile the program using:
```bash
gcc -o basicstats main.c -lm

## Execution
- Run the program with:

## Usage
- The program reads numerical data from the specified file and computes the statistical measures. The results are displayed in the console, including the total number of values, mean, median, standard deviation, and unused array capacity.

## Core Functions
readData
- Reads data from the specified file into a dynamically allocated array. It adjusts the memory allocation as needed to fit the data set.

### calculateMean
- Calculates the mean (average) of the data set.

### calculateMedian
- Determines the median value of the data set. It sorts the data array and finds the middle value or the average of two middle values for even-sized datasets.

### calculateStdDev
- Computes the standard deviation of the data set, providing insights into data variability.

### compareFloats
- A utility function used by qsort for sorting the data array.

## Contributors
Arman Behnam
