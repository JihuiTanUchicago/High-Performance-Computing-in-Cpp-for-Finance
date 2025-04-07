#include <random>
#include <iostream>
#include <cmath>

using namespace std;

constexpr int TRIAL_COUNT = 10000;

inline vector<double> generate_point(uniform_real_distribution<>& dis, mt19937& gen) {
    double x = dis(gen);
    double y = dis(gen);
    return {x, y};
}

inline bool is_in_circle(vector<double> point) {
    double distance_to_origin = sqrt(point[0] * point[0] + point[1] * point[1]);
    if (distance_to_origin > 1.0) {
        return false;
    }
    return true;
}

/*
Calculating pi using monte carlo method.
Assume circle has radius 1 within a square of side length 2.
Assume circle origin is (0,0).
*/
int main(){
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    int total_point_in_circle_count = 0;

    for (int i = 0; i < TRIAL_COUNT; i++){
        vector<double> point = generate_point(dis, gen);
        if (is_in_circle(point)) {
            total_point_in_circle_count++;
        }
    }

    double pi_estimate = 4.0 * total_point_in_circle_count / TRIAL_COUNT;
    cout << "Pi estimate: " << pi_estimate << endl;

    return 0;
}