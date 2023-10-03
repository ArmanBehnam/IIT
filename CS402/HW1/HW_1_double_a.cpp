#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

int main() {
    // Define matrix dimensions
    const int ROWS_A = 1000;
    const int COLS_A = 500;
    const int ROWS_B = 500;
    const int COLS_B = 1000;

    // Initialize matrices A and B with random integers
    vector<vector<int>> A(ROWS_A, vector<int>(COLS_A));
    vector<vector<int>> B(ROWS_B, vector<int>(COLS_B));
    for(int i = 0; i < ROWS_A; i++) {
        for(int j = 0; j < COLS_A; j++) {
            A[i][j] = (double)rand() / RAND_MAX;  // Random doubles between 0 and 1
        }
    }
    for(int i = 0; i < ROWS_B; i++) {
        for(int j = 0; j < COLS_B; j++) {
            B[i][j] = (double)rand() / RAND_MAX;  // Random doubles between 0 and 1
        }
    }

    // Matrix multiplication
    vector<vector<int>> C(ROWS_A, vector<int>(COLS_B, 0));
    auto start = high_resolution_clock::now();
    for(int i = 0; i < ROWS_A; i++) {
        for(int j = 0; j < COLS_B; j++) {
            for(int k = 0; k < COLS_A; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

    return 0;
}
