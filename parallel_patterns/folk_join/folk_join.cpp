#include <thread>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

void process(string s) {
    cout << "Processed " << s << endl;
}

int main() {
    vector<vector<string>> input_messages = {
        {"A", "B", "C"},
        {"1", "2", "3", "4"},
        {"X", "Y"}
    };

    int process_num = 1;
    for (vector<string>& message_list : input_messages) {
        cout << "Begin processing batch No. " << process_num << endl;
        vector<thread> threads;
        for (string& message : message_list) {
            threads.emplace_back(process, message);
        }
        for (thread& t : threads) {
            t.join();
        }
        cout << "Done processing batch No. " << process_num++ << endl;
    }

    return 0;
}