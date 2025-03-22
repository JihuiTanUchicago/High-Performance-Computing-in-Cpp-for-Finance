#include <iostream>
#include "PricingData.hpp"

using namespace std;

PricingData::PricingData(float stockPrice, float strikePrice, 
                         float riskFreeRate, int timeToMaturity, 
                         float volatility)
    : stockPrice(stockPrice), strikePrice(strikePrice), 
      riskFreeRate(riskFreeRate), timeToMaturity(timeToMaturity), 
      volatility(volatility) {}

float PricingData::getStockPrice() const {
    return stockPrice;
}

float PricingData::getStrikePrice() const {
    return strikePrice;
}

float PricingData::getriskFreeRate() const {
    return riskFreeRate;
}

int PricingData::getTimeToMaturity() const {
    return timeToMaturity;
}

float PricingData::getVolatility() const {
    return volatility;
}