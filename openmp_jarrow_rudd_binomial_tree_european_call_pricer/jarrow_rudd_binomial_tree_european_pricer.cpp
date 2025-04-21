#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <thread>
#include <random>
#include <omp.h>

using namespace std;
using namespace std::chrono;

inline float random_data(float low, float hi) {
    float r = (float)rand() / (float)RAND_MAX;
    return low + r * (hi - low);
}

inline float price(float S, float K, float r, float v, float dt, int N, bool price_call_mode) {
    const float sqrt_dt = sqrtf(dt);
    const float v_sqrt_dt = v * sqrt_dt;
    float pow_ud = expf(((r - 0.5f * v * v) * dt - v_sqrt_dt) * N);
    const float u_div_d = expf(2.0f * v_sqrt_dt);
    const float disc = expf(-r * dt);

    vector<float> vec;
    vec.resize(N+1);

    if (price_call_mode) {
        for (int i = 0; i <= N; ++i) {
            float ST = S * pow_ud;
            pow_ud *= u_div_d;
            vec[i] = max(0.0f, ST - K);
        }
    } else {
        for (int i = 0; i <= N; ++i) {
            float ST = S * pow_ud;
            pow_ud *= u_div_d;
            vec[i] = max(0.0f, K - ST);
        }
    }

    for (int i = N - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            vec[j] = disc * (0.5f * (vec[j] + vec[j+1]));
        }
    }

    return vec[0];
}

int main() {
    /*
     * Part 1: Price specific call/put
    */
    const float K = 100.0f, r = 0.03f, v = 0.3f, dt = 0.001f;
    const int N = 1000;
    const vector<float> S_list = {90.0f, 95.0f, 100.0f, 105.0f, 110.0f};

    cout << left << setw(10) << "S0"
         << setw(15) << "C0"
         << setw(15) << "P0" << endl;
    cout << string(40, '-') << endl;

    for (float S : S_list) {
        float call = price(S, K, r, v, dt, N, true);
        float put = price(S, K, r, v, dt, N, false);
        cout << left << setw(10) << S
             << setw(15) << call
             << setw(15) << put << endl;
    }

    /*
    * Part 2: Price 1 million calls on random input
    */
    cout << "Begin pricing 1 million calls on random input" << endl;
    const int ONE_MILLION = 1000000;
    vector<float> vec(ONE_MILLION * 4);
    for (int i = 0; i < ONE_MILLION; i += 4) {
        vec[i] = random_data(90.0f, 110.0f); // S
        vec[i+1] = random_data(90.0f, 110.0f); // K
        vec[i+2] = random_data(0.01f, 0.05f); // r
        vec[i+3] = random_data(0.1f, 0.5f); // v
    }
    cout << "Random Inputs Generated" << endl;
    
    const int num_threads = 48;
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " cores to price" << endl;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    #pragma omp parallel for schedule(guided, 128) proc_bind(close)
    for (int i = 0; i < ONE_MILLION; i += 4) {
        price(vec[i], vec[i+1], vec[i+2], vec[i+3], dt, N, true);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;
    return 0;
}