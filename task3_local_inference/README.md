# Task 3: local inference with Ollama, fp16 vs 4-bit (q4_K_M)

Model: Qwen2.5-1.5B-Instruct, served by Ollama on a Colab Tesla T4. Prompts used temperature 0, seed 0, and at most 256 new tokens.
Tokens/s comes from Ollama's own `eval_count / eval_duration`, averaged over 10 prompts. Memory comes from `nvidia-smi` after loading, with nothing else on the GPU.

| setting | file size | GPU memory used | decode tok/s (mean, min to max) | prompt tok/s |
|---|---|---|---|---|
| fp16 | 3.1 GB | 3,263 MB | **59.0** (52.5 to 63.2) | 1,312 |
| q4_K_M | 986 MB | 1,243 MB | **127.9** (117.4 to 137.0) | 2,569 |

4-bit used 2.6× less memory and decoded 2.2× faster. Quality comparison on the same 10 prompts: `outputs_side_by_side.md`.

Run: `python bench.py` with `ollama serve` running and both tags pulled (see `../run_all.ipynb`).
