# Integrated benchmark results

| Task | Data size | Python baseline (ms) | NumCompute (ms) | Speed-up | Same result |
|---|---|---:|---:|---:|:---:|
| StandardScaler.transform | 20,000 x 4 | 7.4222 | 0.5024 | 14.77x | yes |
| stable softmax | 20,000 x 4 | 12.4510 | 1.5772 | 7.89x | yes |
| top-k | n=20,000, k=10 | 1.9376 | 0.0462 | 41.94x | yes |
| accuracy | n=20,000 | 0.5824 | 0.0204 | 28.50x | yes |

## Environment and method

- Measured at: 2026-10-17T06:22:51.937459+00:00
- Python: 3.14.6 (CPython)
- NumPy: 2.5.4
- OS: Windows 11 (AMD64)
- CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel; logical cores: 20
- Random seed: 42; 5 repeats, 3 calls per repeat, one warmup call.
- Reported time: median seconds per call, converted to milliseconds.
- Data/list preparation, scaler fitting, CSV I/O and correctness checks are outside timing.
- Scaling uses identical fitted means/scales. Both softmax versions subtract each row's maximum.
- Top-k baseline uses Python's full sorted(); NumCompute uses partial selection.
  This comparison includes an algorithm difference, not only loop/vectorisation overhead.
- Equality is checked with rtol=atol=1e-12 before timing.
- Speed-up = Python baseline / NumCompute. Results vary by machine and input size;
  values below 1 mean the baseline was faster. No minimum speed-up is required.
