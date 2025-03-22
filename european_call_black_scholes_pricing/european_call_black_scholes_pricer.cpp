#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <cmath>

#include "PricingData.hpp"

using namespace std;
using namespace std::chrono;

float random_data(float low, float hi)
{
    const float r = (float)rand() / (float)RAND_MAX;
    return low + r * (hi - low);
}

// Generates random pricing data
PricingData generatePricingData() {
    float stockPrice = random_data(0, 100);
    float strikePrice = random_data(0, 100);
    float riskFreeRate = random_data(0, 0.1);
    int timeToMaturity = static_cast<int>(random_data(0, 10));
    float volatility = random_data(0, 0.2);

    return PricingData(stockPrice, strikePrice, riskFreeRate, 
        timeToMaturity, volatility);
}

// Prices European call option using Black-Scholes
float price(PricingData& pricingData) {
    float stockPrice = pricingData.getStockPrice();
    float strikePrice = pricingData.getStrikePrice();
    float riskFreeRate = pricingData.getriskFreeRate();
    int timeToMaturity = pricingData.getTimeToMaturity();
    float volatility = pricingData.getVolatility();

    float d1 = (log(stockPrice / strikePrice) + 
        (riskFreeRate + (volatility * volatility / 2)) * timeToMaturity) 
            / (volatility * sqrt(timeToMaturity));
    float d2 = d1 - (volatility * sqrt(timeToMaturity));

    auto cdf = [](float x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    float callPrice = stockPrice * cdf(d1) - (strikePrice * 
        exp(-riskFreeRate * timeToMaturity) * cdf(d2));
    
    return callPrice;
}

int main()
{
    cout << "Generating pricing records..." << endl;
    ofstream PricingRecords("output.csv");
    PricingRecords << "callPrice, timeTaken, S0, K, r, T, sigma" << endl;

    // Price 1 million distinct call options
    for (int i = 0; i < 1000000; i++) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        PricingData pricingData = generatePricingData();
        float callPrice = price(pricingData);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        PricingRecords << callPrice << ", " << 
            duration_cast<nanoseconds>(t2 - t1).count() << ", " << 
            pricingData.getStockPrice() << ", " << 
            pricingData.getStrikePrice() << ", " << 
            pricingData.getriskFreeRate() << ", " << 
            pricingData.getTimeToMaturity() << ", " << 
            pricingData.getVolatility() << endl;
    }
    cout << "Pricing for 1 million records complete!" << endl;
    return 0;
}