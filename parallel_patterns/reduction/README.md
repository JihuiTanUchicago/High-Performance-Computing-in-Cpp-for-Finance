### Compilation
```
clang++ -std=c++20 -O2 -march=native reduction.cpp -o reduction
```

### Output
```
Thread 0
Thread 0 created
Thread 1
Summing from 0 to 25
Summing from 0 to 25 complete
Thread Summing from 25 to 50
1Summing from  created
Thread 2
25 to 50 complete
Thread 2 created
Thread 3
Summing from 50 to 75
Summing from 50 to 75 complete
Thread 3 created
Summing from 75 to 100
Summing from 75 to 100 complete
Sum: 5050
```