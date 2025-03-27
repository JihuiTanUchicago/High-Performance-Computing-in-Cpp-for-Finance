#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;
using namespace std::chrono;

typedef vector<vector<float>> Matrix;
constexpr int TRIAL_COUNT = 100;
constexpr int ROWS = 100;
constexpr int COLUMNS = 100;

float generateRandomFloat(float low, float hi)
{
    const float r = (float)rand() / (float)RAND_MAX;
    return low + r * (hi - low);
}

void generateRandomMatrix(Matrix& matrix, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = generateRandomFloat(0, 10);
        }
    }
}

void resizeMatrix(Matrix& matrix, int rows, int columns) {
    matrix.resize(rows);
    for (int i = 0; i < rows; i++) {
        matrix[i].resize(columns);
    }
}

void calculateMatrixMultiplication(Matrix& m1, Matrix& m2, Matrix& result) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            result[i][j] = 0;
            for (int k = 0; k < COLUMNS; k++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}

long double calculateMatrixAverageEntryValue(Matrix& matrix) {
    long double sum = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLUMNS; j++) {
            sum += matrix[i][j];
        }
    }
    long double avg = sum / (ROWS * COLUMNS);
    cout << "Average entry value: " << avg << endl;
    return avg;
}


int main()
{
    cout << "Multiplying 100x100 matrices..." << endl;
    ofstream MatrixExecutionTime("output.csv");
    MatrixExecutionTime << "trial, execution time, average entry value" << endl;
    Matrix m1, m2, result;
    resizeMatrix(m1, ROWS, COLUMNS);
    resizeMatrix(m2, ROWS, COLUMNS);
    resizeMatrix(result, ROWS, COLUMNS);

    long totalTime = 0;
    // Measure time for 100 calculations of two 100x100 matrix multiplication
    for (int i = 0; i < TRIAL_COUNT; i++) {
        generateRandomMatrix(m1, ROWS, COLUMNS);
        generateRandomMatrix(m2, ROWS, COLUMNS);
        generateRandomMatrix(result, ROWS, COLUMNS);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        calculateMatrixMultiplication(m1, m2, result);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        long executionTime = duration_cast<nanoseconds>(t2 - t1).count();
        totalTime += executionTime;
        cout << "Trial " << i << "completed. Took " << executionTime << endl;
        MatrixExecutionTime << i << ", " << duration_cast<nanoseconds>(t2 - t1).count() 
            << ", " << calculateMatrixAverageEntryValue(result) << endl;
    }
    cout << "Pricing for 1 million records complete!" << endl;
    MatrixExecutionTime << "Average execution time (seconds): " << totalTime / 100 << endl;
    return 0;
}