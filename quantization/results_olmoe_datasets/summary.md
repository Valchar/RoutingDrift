# Routing Drift Summary

- Model: `/scratch/zt1/project/msml605/user/gsakthiv/models/OLMoE-1B-7B`
- Prompts: 5
- Router top-k: 2

## Results

| Variant | Routing Similarity (RS) | Jaccard Drift | Overlap@k | Selection Shift |
|---|---:|---:|---:|---:|
| fp16 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| int8 | 0.954475 | 0.045525 | 0.965856 | 0.034144 |
| int4 | 0.933256 | 0.066744 | 0.949653 | 0.050347 |

## Interpretation

Higher RS and Overlap@k indicate routing closer to FP16 baseline behavior.
Lower Jaccard Drift and Selection Shift indicate less routing change after quantization.
