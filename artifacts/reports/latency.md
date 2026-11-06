# Latency Benchmarks

50 warmup + 300 timed iterations per configuration.

| Backend | Batch | p50 ms | p95 ms | p99 ms | Clips/s | Per-clip ms |
|---|---:|---:|---:|---:|---:|---:|
| torch | 1 | 1.973 | 3.449 | 4.185 | 506.9 | 1.973 |
| torch | 8 | 4.919 | 6.783 | 9.944 | 1626.3 | 0.615 |
| torch | 32 | 20.849 | 23.462 | 31.731 | 1534.9 | 0.652 |
| onnx | 1 | 0.784 | 3.242 | 5.583 | 1274.9 | 0.784 |
| onnx | 8 | 4.445 | 9.034 | 14.449 | 1799.7 | 0.556 |
| onnx | 32 | 15.991 | 33.801 | 47.892 | 2001.1 | 0.500 |

## Streaming (48-frame window, stride 8)

- p50 per window: **1.028 ms**
- p95 per window: **2.971 ms**
- Implied maximum real-time frame rate: **7784.3 FPS**
- 30 FPS webcam viable: **yes**
