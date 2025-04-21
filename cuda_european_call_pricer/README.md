### Compilation
```
nvcc -O3 -std=c++17 cuda_european_call_pricer.cu -o cuda_european_call_pricer
```
Then run:
```
./cuda_european_call_pricer
```

### Output
==4137456== NVPROF is profiling process 4137456, command: ./cuda_european_call_pricer
S0        K       r       v       T       callPrice      callDelta      callRho        callTheta      putPrice       putDelta       putRho         putTheta       gamma          vega
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
90.00     90.00   0.03    0.30    1.00    11.95          0.60           41.93          -6.48          9.30           -0.40          -45.41         -3.86          0.01           34.80
95.00     90.00   0.03    0.30    1.00    15.12          0.67           48.19          -6.63          7.46           -0.33          -39.15         -4.01          0.01           34.55
100.00    100.00  0.03    0.30    2.00    19.38          0.64           88.87          -5.31          13.56          -0.36          -99.49         -2.48          0.01           53.00
105.00    100.00  0.03    0.30    2.00    22.68          0.68           97.50          -5.44          11.86          -0.32          -90.85         -2.62          0.01           53.08
110.00    100.00  0.03    0.30    2.00    26.18          0.72           105.70         -5.52          10.36          -0.28          -82.65         -2.70          0.01           52.51
Generating pricing records...
Elapsed time: 13 ms
Pricing for 1 million records complete!
==4137456== Profiling application: ./cuda_european_call_pricer
==4137456== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   64.85%  8.3232ms         2  4.1616ms  2.3040us  8.3209ms  [CUDA memcpy DtoH]
                   32.51%  4.1724ms         2  2.0862ms  1.7280us  4.1707ms  [CUDA memcpy HtoD]
                    2.65%  339.55us         2  169.77us  5.2800us  334.27us  pricer(float*, float*)
      API calls:   89.83%  125.82ms         4  31.455ms  5.0210us  125.55ms  cudaMalloc
                    9.27%  12.979ms         4  3.2447ms  18.230us  8.5663ms  cudaMemcpy
                    0.43%  608.16us         4  152.04us  5.2710us  369.69us  cudaFree
                    0.29%  400.03us         2  200.01us  11.193us  388.83us  cudaDeviceSynchronize
                    0.14%  192.34us       101  1.9040us     111ns  87.554us  cuDeviceGetAttribute
                    0.04%  56.521us         2  28.260us  24.571us  31.950us  cudaLaunchKernel
                    0.01%  7.8590us         1  7.8590us  7.8590us  7.8590us  cuDeviceGetName
                    0.00%  4.8290us         1  4.8290us  4.8290us  4.8290us  cuDeviceGetPCIBusId
                    0.00%  1.1240us         3     374ns     131ns     818ns  cuDeviceGetCount
                    0.00%     640ns         2     320ns     153ns     487ns  cuDeviceGet
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceTotalMem
                    0.00%     240ns         1     240ns     240ns     240ns  cuModuleGetLoadingMode
                    0.00%     201ns         1     201ns     201ns     201ns  cuDeviceGetUuid