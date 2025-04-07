#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

int TRIAL_COUNT = 1000000;
int SIMULATION_COUNT = 1000;


inline double get_stock_price_at_option_expiration(double z_i, double S_0, double K, double r, double T, double sd) {
    return S_0 * exp((r - sd * sd / 2) * T + sd * z_i * sqrt(T));
}

inline double get_payoff(double stock_price_at_option_expiration, double K, double r, double T) {
    return exp(-r * T) *max(stock_price_at_option_expiration - K, 0.0);
}

int main() {
    // File to write
    ofstream PricingRecords("output.csv");
    PricingRecords << "call price estimate, S0, K, r, T, std" << endl;

    // Define distribution for each attribute
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_z(0.0, 1.0);
    uniform_real_distribution<> dis_S(90.0, 110.0);
    uniform_real_distribution<> dis_K(90.0, 110.0);
    uniform_real_distribution<> dis_r(0.01, 0.08);
    uniform_real_distribution<> dis_T(0.5, 2.0);
    uniform_real_distribution<> dis_sd(0.01, 0.10);

    // Monte Carlo Simulation
    for (int i = 0; i < TRIAL_COUNT; i++) {
        float S_0 = dis_S(gen);
        float K = dis_K(gen);
        float r = dis_r(gen);
        float T = dis_T(gen);
        float sd = dis_sd(gen);
        double z_i = dis_z(gen);
        long double option_price_estimator = 0.0;
        for (int j = 0; j < SIMULATION_COUNT; j++){
            double S_i = get_stock_price_at_option_expiration(z_i, S_0, K, r, T, sd);
            double payoff = get_payoff(S_i, K, r, T);
            option_price_estimator += payoff;
        }
        option_price_estimator /= SIMULATION_COUNT;
        PricingRecords << option_price_estimator << ", " << S_0 << ", " << K << ", " << r << ", " << T << ", " << sd << endl;
        cout << "Pricing for trial " << i << " complete!" << endl;
    }

    return 0;
}
