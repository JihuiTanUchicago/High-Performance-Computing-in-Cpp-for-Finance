### Compilation and Run
```
module load cuda
nvcc -O2 -o portfolio_simulation portfolio_simulation.cu
./portfolio_simulation
```

### Sample Output
```
Stock Price     Portfolio Value
95              7901.82
96              8186.73
97              8476.99
98              8771.6
99              9069.94
100             9373.5
101             9681.96
102             9993.38
103             10310.2
104             10631.5
105             10958.1
Elapsed time: 238 ms
```