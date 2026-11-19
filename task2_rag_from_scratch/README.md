# Task 2: RAG from scratch (no LangChain / LlamaIndex)

- **Documents:** Llama 2 paper (arXiv 2307.09288), 77 pages, 42,391 words. Downloaded and converted with `pdftotext` inside the notebook (not committed).
- **Retrieval:** `BAAI/bge-small-en-v1.5` embeddings plus brute-force cosine similarity in NumPy (`E @ q` on unit vectors).
  A hand-written BM25 (k1=1.5, b=0.75) and a reciprocal-rank-fusion hybrid are included for comparison.
- **Answering:** top-5 chunks are passed to `Qwen2.5-1.5B-Instruct` (fp16, greedy decoding).
- **Evaluation:** 15 hand-written questions in `questions.json`. Each has a gold passage, an exact phrase from the paper.
  A question counts as a hit if any of the top-k chunks contains that phrase.

| chunks (words, 20% overlap) | method | #chunks | recall@1 | **recall@5** | MRR |
|---|---|---|---|---|---|
| 100 | dense | 530 | 0.667 | **0.867** | 0.736 |
| 100 | BM25 | 530 | 0.800 | 0.867 | 0.813 |
| 100 | hybrid (RRF) | 530 | 0.733 | 0.933 | 0.797 |
| 400 | dense | 133 | 0.667 | **0.867** | 0.739 |
| 400 | BM25 | 133 | 0.867 | 1.000 | 0.922 |
| 400 | hybrid (RRF) | 133 | 0.800 | 1.000 | 0.878 |

The full per-question ranks, dense misses and generated answers are in `results.json`.

Run: `python rag.py data/llama2.txt questions.json` (see `../run_all.ipynb`).
