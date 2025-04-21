#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;
using namespace std::chrono;

const int N = 1000000; // number of options to price

float random_data(float low, float hi) {
    const float r = (float)rand() / (float)RAND_MAX;
    return low + r * (hi - low);
}

void generate_pricing_parameters(vector<float>& randomInputVec) {
    randomInputVec.resize(5 * N);
    for (int i = 0; i < N; i++) {
        randomInputVec[5*i] = random_data(0, 100); // S
        randomInputVec[5*i+1] = random_data(0, 100); // K
        randomInputVec[5*i+2] = random_data(0, 0.1); // r
        randomInputVec[5*i+3] = static_cast<int>(random_data(0, 5)); // T
        randomInputVec[5*i+4] = random_data(0, 0.2); // v
    }
}

/*
 * Description: price european call & put along with greeks
 * Output: a vector of float containing call & put prices along with calculated greeks
*/
__device__ void price(float* inputVec, float* output) {
    float sqrtT = sqrtf(inputVec[3]);
    float vSqrtT = inputVec[4] * sqrtT;
    float d1 = (logf(inputVec[0] / inputVec[1]) + (inputVec[2] + 0.5f * inputVec[4] * inputVec[4]) * inputVec[3]) / vSqrtT;
    float d2 = d1 - vSqrtT;
    float nd1 = 0.5f * erfcf(-d1 / sqrtf(2.0f));
    float nd2 = 0.5f * erfcf(-d2 / sqrtf(2.0f));
    float pdfD1 = expf(-0.5f * d1 * d1) / sqrtf(2.0f * M_PI);
    float discount = expf(-inputVec[2] * inputVec[3]);
    float oneMinusNd2 = 1.0f - nd2;
    float oneMinusNd1 = 1.0f - nd1;

    output[0] = inputVec[0] * nd1 - inputVec[1] * discount * nd2; // call price
    output[1] = nd1; // call delta
    output[2] = inputVec[1] * inputVec[3] * discount * nd2; // call rho
    output[3] = - (inputVec[0] * pdfD1 * inputVec[4]) / (2.0f * sqrtT) - inputVec[2] * inputVec[1] * discount * nd2; // call theta

    output[4] = inputVec[1] * discount * oneMinusNd2 - inputVec[0] * oneMinusNd1; // put price
    output[5] = nd1 - 1.0f; // put delta
    output[6] = -inputVec[1] * inputVec[3] * discount * oneMinusNd2; // put rho
    output[7] = - (inputVec[0] * pdfD1 * inputVec[4]) / (2.0f * sqrtT) + inputVec[2] * inputVec[1] * discount * oneMinusNd2; // put theta

    output[8] = pdfD1 / (inputVec[0] * inputVec[4] * sqrtT); // gamma
    output[9] = inputVec[0] * pdfD1 * sqrtT; // vega
}


__global__ void pricer(float* inputVec, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    price(&inputVec[i * 5], &out[i * 10]);
}

int main()
{
    /*
     * Part 1: Find given Call and Put prices and their greeks
    */
    // Sample Parameters Setup
    vector<float> testStockPriceVec = {90, 95, 100, 105, 110};
    vector<float> testStrikePriceVec = {90, 90, 100, 100, 100};
    vector<float> testRiskFreeRateVec = {0.03, 0.03, 0.03, 0.03, 0.03};
    vector<float> testVolatilityVec = {0.3, 0.3, 0.3, 0.3, 0.3};
    vector<float> testTimeToMaturityVec = {1, 1, 2, 2, 2};

    vector<float> testInputVec;
    for (int i = 0; i < testStockPriceVec.size(); i++) {
        testInputVec.push_back(testStockPriceVec[i]);
        testInputVec.push_back(testStrikePriceVec[i]);
        testInputVec.push_back(testRiskFreeRateVec[i]);
        testInputVec.push_back(testTimeToMaturityVec[i]);
        testInputVec.push_back(testVolatilityVec[i]);
    }
    
    // Copy to GPU kernel
    float *d_testInputVec, *d_testOutputVec;

    cudaMalloc(&d_testInputVec, 25 * sizeof(float));
    cudaMalloc(&d_testOutputVec, 50 * sizeof(float));
    cudaMemcpy(d_testInputVec, testInputVec.data(), 25 * sizeof(float), cudaMemcpyHostToDevice);

    // Run on GPU
    const int numTestThreads = 5;
    const int numTestBlocks = 1;
    pricer<<<numTestThreads, numTestBlocks>>>(d_testInputVec, d_testOutputVec);
    cudaDeviceSynchronize();

    // Copy result back to host
    vector<float> testOutputVec(50);
    cudaMemcpy(testOutputVec.data(), d_testOutputVec, 50 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_testInputVec);
    cudaFree(d_testOutputVec);

    cout << fixed << setprecision(2);
    cout << left << setw(10) << "S0"
         << setw(8) << "K"
         << setw(8) << "r" 
         << setw(8) << "v"
         << setw(8) << "T"
         << setw(15) << "callPrice"
         << setw(15) << "callDelta"
         << setw(15) << "callRho"
         << setw(15) << "callTheta"
         << setw(15) << "putPrice"
         << setw(15) << "putDelta"
         << setw(15) << "putRho"
         << setw(15) << "putTheta"
         << setw(15) << "gamma"
         << setw(15) << "vega" << endl;
    cout << string(192, '-') << endl;

    for (int i = 0; i < testStockPriceVec.size(); i++) {
        cout << left << setw(10) 
            << testStockPriceVec[i] << setw(8)
            << testStrikePriceVec[i] << setw(8)
            << testRiskFreeRateVec[i] << setw(8)
            << testVolatilityVec[i] << setw(8)
            << testTimeToMaturityVec[i];
        for (int j = 0; j < 10; j++) {
            cout << setw(15) << testOutputVec[i*10 + j];
        }
        cout << endl;
    }

    /*
     * Part 2: Price 1 million options (Call and Put) and their greeks
    */
    cout << "Generating pricing records..." << endl;
    vector<float> randomInputVec;
    generate_pricing_parameters(randomInputVec);
    
    float *d_randomInputVec, *d_outputVec;
    cudaMalloc(&d_randomInputVec, 5 * N * sizeof(float));
    cudaMalloc(&d_outputVec, 10 * N * sizeof(float));
    vector<float> outputVec(N * 10);
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Copy to GPU kernel
    cudaMemcpy(d_randomInputVec, randomInputVec.data(), 5 * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run on GPU
    const int numThreads = 1024;
    const int numBlocks = (N + numThreads - 1) / numThreads;
    pricer<<<numThreads, numBlocks>>>(d_randomInputVec, d_outputVec);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(outputVec.data(), d_outputVec, N * 10 * sizeof(float), cudaMemcpyDeviceToHost);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    // Free allocated memory
    cudaFree(d_randomInputVec);
    cudaFree(d_outputVec);

    cout << "Elapsed time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;
    cout << "Pricing for 1 million records complete!" << endl;
    
    
    return 0;
}