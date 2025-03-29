#include <iostream>
#include <vector>
#include <chrono>
#include <numeric> // For std::accumulate

// Function to process the array in row-major order
long long process_array_row_major(const std::vector<std::vector<int>>& arr) {
    long long sum = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        for (size_t j = 0; j < arr[0].size(); ++j) {
            sum += arr[i][j];
        }
    }
    return sum;
}

// Function to process the array in column-major order
long long process_array_column_major(const std::vector<std::vector<int>>& arr) {
    long long sum = 0;
    for (size_t j = 0; j < arr[0].size(); ++j) {
        for (size_t i = 0; i < arr.size(); ++i) {
            sum += arr[i][j];
        }
    }
    return sum;
}

// Helper function to create a 2D vector filled with a value
std::vector<std::vector<int>> create_array(size_t rows, size_t cols, int value = 1) {
    std::vector<std::vector<int>> arr(rows, std::vector<int>(cols, value));
    return arr;
}

// Basic timer class (you can use or modify this)
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    long long elapsed_microseconds() const {
        auto end_ = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

int main() {
    // You will need to experiment with different array sizes here
    size_t rows = 4096;
    size_t cols = 4096;
    auto data = create_array(rows, cols);

    // Benchmark row-major processing
    {
        Timer timer;
        long long result = process_array_row_major(data);
        long long duration = timer.elapsed_microseconds();
        std::cout << "Row-major sum: " << result << ", Time: " << duration << " microseconds" << std::endl;
    }

    // Benchmark column-major processing
    {
        Timer timer;
        long long result = process_array_column_major(data);
        long long duration = timer.elapsed_microseconds();
        std::cout << "Column-major sum: " << result << ", Time: " << duration << " microseconds" << std::endl;
    }

    return 0;
}
