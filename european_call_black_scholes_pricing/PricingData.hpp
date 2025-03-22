#ifndef PRICINGDATA_HPP
#define PRICINGDATA_HPP

#include <iostream>

/*
 * Class for storing data needed to price
 * European call option using Black-Scholes
*/
class PricingData {
    private:
        float stockPrice;

        float strikePrice;

        float riskFreeRate;

        int timeToMaturity;

        float volatility;

    public:
        PricingData(float stockPrice, float strikePrice, 
                    float riskFreeRate, int timeToMaturity, 
                    float volatility);

        float getStockPrice() const;

        float getStrikePrice() const;

        float getriskFreeRate() const;

        int getTimeToMaturity() const;

        float getVolatility() const;
};

#endif
