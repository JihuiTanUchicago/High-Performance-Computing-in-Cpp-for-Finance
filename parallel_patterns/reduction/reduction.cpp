#include <string>
#include <vector>
#include <thread>
#include <numeric> 
#include <iostream>
#include <mutex>

using namespace std;

void sum(vector<int>& nums, int start, int end, int& result, mutex& m) {
    cout << "Summing from " << start << " to " << end << endl;
    int summation = accumulate(nums.begin() + start, nums.begin() + end, 0LL);
    m.lock();
    result += summation;
    m.unlock();
    cout << "Summing from " << start << " to " << end << " complete" << endl;
}

int sum_reducer(int num_threads, vector<int>& nums) {
    int result = 0;
    mutex m;
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        cout << "Thread " << i << endl;
        int start = i * nums.size() / num_threads;
        int end = (i + 1) * nums.size() / num_threads;
        threads.emplace_back(sum, ref(nums), start, end, ref(result), ref(m));
        cout << "Thread " << i << " created" << endl;
    }
    for (thread& t : threads) {
        t.join();
    }
    return result;
}

int main() {
    vector<int> nums;
    for (int i = 1; i <= 100; i++) {
        nums.push_back(i);
    }

    int result = sum_reducer(4, nums);
    cout << "Sum: " << result << endl;
    return 0;
}