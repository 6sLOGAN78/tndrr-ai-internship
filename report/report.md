# tndrr AI/ML Internship: Assignment Report
Ayush Maurya · Code: GitHub repo (`task1_lora_finetuning/`, `task2_rag_from_scratch/`, `task3_local_inference/`, `run_all.ipynb`) · All runs: Google Colab free Tesla T4 (15 GB), 19 Nov 2026

## Task 1: LoRA fine-tuning: plain-English smart-home command → fixed JSON

**Task.** Map a command to `{"action","device","room","value"}` with closed label sets (5 actions, 7 devices, 5 rooms or null, integer or null).
I generated 1,000 training examples. The 100 held-out test examples deliberately use device words, room words, templates and number words that never appear in training
(e.g. "radiator", "lounge", "room where I sleep", "kill the…", "twenty two"), so the test measures generalisation, not memorisation.
**Model:** Qwen2.5-0.5B-Instruct. Base and fine-tuned models get the same system prompt (listing the allowed labels) and greedy decoding.

| model (100 held-out examples) | valid JSON | **exact match** | action | device | room | value |
|---|---|---|---|---|---|---|
| base | 100% | **5%** | 65% | 52% | 28% | 87% |
| LoRA fine-tuned | 100% | **50%** | 96% | 83% | 66% | 100% |

Metric: exact match = all 4 fields correct; per-field accuracy shows where errors come from. Training: 250 steps, 90 s, peak 7.4 GB.

**LoRA settings and why.** r = 16, α = 32, dropout 0.05, targets q,k,v,o,gate,up,down (8.8M trainable = 1.75%), lr 2e-4 (20-step warm-up, linear decay), 2 epochs, batch 8, loss on answer tokens only.
*Why:* the task is narrow, so a small rank is enough. Putting adapters on all linear layers (attention and MLP) is the QLoRA paper's recommendation: it gets closer to full fine-tuning than attention-only adapters for little extra memory.
α = 2r is a common default that keeps the update scale stable. 2e-4 is a typical LoRA learning rate (about 10× full fine-tuning) because only a few parameters are trained.
2 epochs was enough: training loss reached ~0 within 1 epoch, so more epochs would only memorise templates further.
*(rewrite this reasoning in your own words)*

**Where it still fails.** The 50 failures have 34 wrong rooms, 17 wrong devices and 4 wrong actions.
1. *"Kill the table fan in room where I sleep"* → room `bathroom` (should be `bedroom`). In **32 of 34** room errors the model outputs `bathroom`.
   Faced with a room phrase it never saw, it falls back to one default label instead of using the meaning of the words. Training only had 2 phrasings per room, so it learned string matching rather than the concept.
2. *"I don't need the curtains in room where I sleep anymore"* → action `set` (should be `turn_off`). The turn-off is implied, not stated. Training only had explicit verbs ("turn off", "switch off"), so indirect phrasing breaks it.
3. *"bump the sound system in the sitting room up a bit"* → device `sound_system`, which isn't an allowed label. For an unseen synonym the model copies the surface word instead of mapping it to `speaker`.
   It also confused "radiator" with fan/ac. *Fix:* more diverse paraphrases or synonym augmentation in training, plus constrained decoding (a JSON grammar or enum) so invalid labels are impossible.

**GPU memory to fine-tune a 7B model** (bf16, AdamW, Llama-7B shape: 32 layers, d=4096, MLP 11008; activations with checkpointing ≈ 3 to 8 GB):

| | working | total |
|---|---|---|
| (a) full FT | weights 2 B + grads 2 B + fp32 master 4 B + Adam m,v 8 B = **16 B/param** × 7B = 112 GB, + activations | **≈ 115 to 120 GB** (multi-GPU) |
| (b) LoRA r=16, all linear | frozen bf16 base 7B × 2 B = 14 GB; LoRA params/layer = 4·16·(4096+4096) + 3·16·(4096+11008) ≈ 1.25M × 32 ≈ 40M; 40M × 16 B = 0.64 GB; + ~4 GB activations | **≈ 19 GB** (one 24 GB GPU) |
| (c) QLoRA | base in 4-bit NF4: 7B × 0.5 B = 3.5 GB + quant constants ≈ 0.1 GB; LoRA 0.64 GB; activations ≈ 3 to 4 GB | **≈ 8 to 9 GB** (fits a T4) |

## Task 2: Retrieval-augmented QA from scratch

**Setup.** Corpus: Llama 2 paper (arXiv 2307.09288), **77 pages**, 42,391 words via `pdftotext`. Word-based chunks with 20% overlap.
Embeddings: `bge-small-en-v1.5`, normalised. Search: NumPy dot product (exact cosine). No framework.
Answers: top-5 chunks given to Qwen2.5-1.5B-Instruct. Each of the 15 questions has a gold passage, an exact phrase from the paper (all listed in `questions.json`).
A question counts as a hit if that phrase is inside a retrieved chunk. I also implemented BM25 and a reciprocal-rank-fusion hybrid to compare.

| chunk size | method | #chunks | recall@1 | **recall@5** | MRR |
|---|---|---|---|---|---|
| 100 words | dense | 530 | 0.667 | **0.867** | 0.736 |
| 400 words | dense | 133 | 0.667 | **0.867** | 0.739 |
| 100 words | BM25 / hybrid | 530 | 0.800 / 0.733 | 0.867 / 0.933 | 0.813 / 0.797 |
| 400 words | BM25 / hybrid | 133 | 0.867 / 0.800 | **1.000 / 1.000** | 0.922 / 0.878 |

**Which chunk size worked better?** For dense search they tie (13/15 at k=5), but they fail on *different* questions.
Small chunks lose context: "36% win rate vs ChatGPT" sits in a 100-word chunk with no mention of "human evaluation", so a similar appendix chunk wins.
Large chunks dilute rare topics: the Ghost Attention definition is a few lines inside a 400-word chunk about other things.
With BM25 or the hybrid, **400 words was clearly better (15/15)**: a larger chunk contains more of the question's keywords, and there are 4× fewer chunks competing.
I used 400-word chunks for answering.

**Two retrieval failures and fixes.**
1. *"Which attention variant did they adopt to make inference scale better?"* Dense search missed at both sizes. Top-1 (score 0.70) was the appendix
   MQA-vs-GQA ablation, which is semantically very close but isn't the gold sentence ("…GQA to improve inference scalability"). The LLM still answered "GQA".
   *Fix:* hybrid with BM25 (finds it at rank 3/4 because of the exact word "inference"), and allowing several valid gold passages.
2. *"What is Ghost Attention inspired by?"* (400-word chunks). Dense top-1 was a chunk of extracted figure text (axis numbers, legend) with a low score (0.51).
   The real passage got diluted, and the LLM then answered wrongly ("system message" instead of Context Distillation).
   *Fix:* BM25/hybrid (rank 1 for BM25: "Ghost Attention" is a rare exact term), cleaning figure/table text before chunking, or splitting on section headings.

End-to-end, 12/15 generated answers were correct by my reading. The errors were Ghost Attention, the warm-up steps (gold retrieved at rank 2, but the LLM copied "3% of steps" from another chunk), and the win rate (tie rate omitted).

**BM25 and re-rankers** *(rewrite in your own words)*. Embedding search compares meanings, so it handles paraphrases but blurs exact tokens such as numbers, names and acronyms.
BM25 scores chunks by exact word overlap. Rare words count more (IDF), repeats give diminishing returns (k1), and long chunks are normalised (b). It adds exact-match precision,
which our results show: BM25 beat dense on 4 questions. A re-ranker is a cross-encoder that reads the question and a candidate chunk *together* and outputs a relevance score.
It's too slow for the whole corpus, so you retrieve ~20 to 50 candidates cheaply and re-rank only those. It's more accurate because question and passage tokens attend to each other directly.

## Task 3: Running a model locally: fp16 vs 4-bit (Ollama)

Qwen2.5-1.5B-Instruct in Ollama 0.40 on a T4. Temperature 0, seed 0, max 256 new tokens, same 10 prompts. Tok/s comes from Ollama's `eval_count/eval_duration`. Memory comes from `nvidia-smi` with only that model loaded.

| setting | file size | GPU memory | decode tok/s mean (min to max) | prompt tok/s |
|---|---|---|---|---|
| fp16 | 3.1 GB | 3,263 MB | **59.0** (52.5 to 63.2) | 1,312 |
| q4_K_M | 0.99 GB | 1,243 MB | **127.9** (117.4 to 137.0) | 2,569 |

**Quality on the same 10 prompts** (all outputs in `outputs_side_by_side.md`): 6/10 were essentially identical (hash table, 17×23 = 391 by different methods, French translation, syllogism, JSON).
q4 was *better* on 3: the train-arrival time (fp16 said 1:20, wrong; q4 said 12:15, correct), the Romeo & Juliet summary (fp16 introduced a factual slip), and the haiku (fp16 repeated a word).
Fibonacci was correct in both, with different indexing conventions. So, at this scale, 4-bit showed **no measurable quality loss**.
The differences come from greedy decoding taking a different path after a tiny change in logits, not from systematic degradation. 10 prompts are too few to rank them.

**KV cache** *(rewrite in your own words)*. To produce each new token, attention needs the keys and values of every previous token at every layer.
The KV cache stores those K/V tensors, so each step only computes them for the one new token instead of re-running the whole sequence.
It trades compute for memory. Size = 2 × layers × kv_heads × head_dim × bytes × tokens, growing linearly with context length and the number of concurrent requests.

**Why quantisation saves memory.** Weights dominate memory. q4_K_M stores most weights in ~4.5 bits (4-bit values plus a scale per small block) instead of 16, so the 3.1 GB file becomes 0.99 GB.
Measured VRAM fell from 3.26 GB to 1.24 GB. The rest is KV cache and CUDA buffers, which weight quantisation doesn't shrink.
Decoding is memory-bandwidth-bound: every token reads all the weights once. Fewer bytes per weight is therefore also why q4 decoded 2.2× faster.

**Serving 20 concurrent users on one 24 GB GPU (back-of-envelope).**
*Assumptions:* Llama-3.1-8B-Instruct class model (32 layers, 8 KV heads with GQA, head_dim 128), ≤ 2k tokens of context per user on average, vLLM (PagedAttention + continuous batching).
KV per token = 2 × 32 × 8 × 128 × 2 B = 128 KB. For 20 users × 2,048 tokens that's 41k tokens ≈ **5.4 GB**.
fp16 weights (16 GB) + 5.4 GB + ~1.5 GB overhead ≈ 23 GB: too tight. **4-bit AWQ weights (~5.5 GB) + 5.4 GB KV + 1.5 GB ≈ 12.5 GB.** That fits with room for 4k contexts or bursts, and an FP8 KV cache would double the headroom.
Throughput: each decode step reads ~5.5 GB of weights once for the whole batch. At ~900 GB/s that's an upper bound of ~150 steps/s, realistically ~40 to 60 tokens/s per user with 20 sequences batched.
Guards: cap context per request, cap concurrent sequences (~24) with a queue, and stream tokens.
