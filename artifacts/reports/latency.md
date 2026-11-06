# Latency Benchmarks

50 warmup + 300 timed iterations per configuration.

| Backend | Batch | p50 ms | p95 ms | p99 ms | Clips/s | Per-clip ms |
|---|---:|---:|---:|---:|---:|---:|
| torch | 1 | 1.938 | 2.324 | 2.554 | 516.0 | 1.938 |
| torch | 8 | 4.376 | 4.946 | 5.675 | 1828.2 | 0.547 |
| torch | 32 | 23.506 | 27.804 | 33.350 | 1361.3 | 0.735 |
| onnx | 1 | 1.118 | 2.982 | 4.932 | 894.7 | 1.118 |
| onnx | 8 | 7.502 | 20.336 | 29.463 | 1066.5 | 0.938 |
| onnx | 32 | 24.578 | 63.670 | 86.967 | 1302.0 | 0.768 |

## Streaming (48-frame window, stride 8)

- p50 per window: **1.205 ms**
- p95 per window: **6.051 ms**
- Implied maximum real-time frame rate: **6639.3 FPS**
- 30 FPS webcam viable: **yes**
