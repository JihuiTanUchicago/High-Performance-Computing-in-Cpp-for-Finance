// portfolio_simulation.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
using namespace chrono;

constexpr int NUM_PATHS = 1000000;
constexpr int NUM_K = 100;
constexpr float K_BEGIN = 50.0f;

constexpr int NUM_T = 5;
__device__ __constant__ float T_arr[NUM_T] = {0.5f, 0.75f, 1.0f, 1.25f, 1.5f};
__device__ __constant__ float r_arr[NUM_T] = {0.03f, 0.04f, 0.05f, 0.06f, 0.07f};
__device__ __constant__ float v_arr[NUM_T] = {0.30f, 0.29f, 0.28f, 0.27f, 0.26f};

constexpr int NUM_S = 11;
__device__ __constant__ float S_arr[NUM_S] = {95.0f,96.0f,97.0f,98.0f,99.0f,100.0f,101.0f,102.0f,103.0f,104.0f,105.0f};


__device__ inline float get_stock_price(float stock_price,float drift,float v,float sqrt_T,float z) {
    return stock_price * expf(drift + v * sqrt_T * z);
}

__global__ void simulate_portfolio(float* __restrict__ d_out) {
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    if (bid >= sizeof(S_arr)) return;

    extern __shared__ float accumulator[];
    if (tid == 0) accumulator[0] = 0.0f;

    curandState rng;
    curand_init(42ULL, bid * blockDim.x + tid, 0, &rng);

    __syncthreads();

    const float S0 = S_arr[bid];

    for (int m = 0; m < NUM_T; ++m)
    {
        const float T = T_arr[m];
        const float r = r_arr[m];
        const float v = v_arr[m];
        const float drift = (r - 0.5f * v * v) * T;
        const float sqrt_T = sqrtf(T);
        const float discount = expf(-r * T);

        for (int k = 0; k < NUM_K; ++k)
        {
            const float K = K_BEGIN + k;
            float payoff = 0.0f;
            for (int p = tid; p < NUM_PATHS; p += blockDim.x)
            {
                float z  = curand_normal(&rng);
                float ST = get_stock_price(S0, drift, v, sqrt_T, z);
                if (ST > K) payoff += ST - K;
            }
            atomicAdd(&accumulator[0], discount * (payoff / NUM_PATHS));
        }
    }
    __syncthreads();

    if (tid == 0) d_out[bid] = accumulator[0];
}

int main() {
    vector<float> S_vec = {95.0f,96.0f,97.0f,98.0f,99.0f,100.0f,101.0f,102.0f,103.0f,104.0f,105.0f};


    vector<float> result_vec(NUM_S);

    float *d_out;
    cudaMalloc(&d_out, NUM_S * sizeof(float));

    auto t1 = high_resolution_clock::now();
    simulate_portfolio<<<NUM_S, 1024, sizeof(float)>>>(d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(result_vec.data(), d_out, NUM_S * sizeof(float), cudaMemcpyDeviceToHost);
    auto t2 = high_resolution_clock::now();

    cout << "Stock Price\tPortfolio Value\n";
    for (int i = 0; i < NUM_S; ++i)
        cout << S_vec[i] << "\t\t" << result_vec[i] << '\n';

    cout << "Elapsed time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms\n";

    cudaFree(d_out);
    return 0;
}
