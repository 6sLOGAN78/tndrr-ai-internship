# tndrr AI/ML Internship – Assignment Round

Deadline: Thursday 19 Nov 2026, 11:59 PM IST
Submit: https://forms.gle/zmfDqVkG6M4u8jEH7 (GitHub repo link + PDF report, max 4 pages)

| Folder | Task |
|---|---|
| `task1_lora_finetuning/` | LoRA/QLoRA fine-tune of a ≤1.5B model (Colab/Kaggle GPU), base vs fine-tuned on ≥50 held-out examples, 3 failure cases, 7B memory calculation |
| `task2_rag_from_scratch/` | RAG over ≥30-page public docs with embeddings + NumPy/FAISS only, 15 questions, recall@5 for 2 chunk sizes, 2 failures, BM25 + re-ranker explanation |
| `task3_local_inference/` | Ollama with two quantisation levels, tokens/s + memory, 10-prompt quality comparison, KV cache explanation, plan to serve 20 users on a 24 GB GPU |
| `report/` | Final PDF report (≤4 pages) |

## Results at a glance (all from `run_all.ipynb` on a Colab T4)
- **Task 1:** LoRA on Qwen2.5-0.5B: exact-match JSON accuracy on 100 held-out commands went from **5% to 50%** (valid JSON 100% for both)
- **Task 2:** RAG over the 77-page Llama 2 paper: dense recall@5 = **0.867** for both 100- and 400-word chunks; BM25/hybrid with 400-word chunks = **1.0**
- **Task 3:** Ollama Qwen2.5-1.5B: fp16 59 tok/s / 3.26 GB vs q4_K_M **128 tok/s / 1.24 GB**, with no visible quality loss on 10 prompts

## Reproduce
Open `run_all.ipynb` in Colab with a T4 GPU and choose Runtime → Run all (~12 min). `build_notebook.py` regenerates the notebook from the task folders.
