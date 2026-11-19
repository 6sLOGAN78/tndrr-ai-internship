"""RAG from scratch over the Llama 2 paper (arXiv 2307.09288, 77 pages).

No LangChain / LlamaIndex: just an embedding model + NumPy (cosine search),
a hand-written BM25 for comparison, and a small LLM for the final answer.

Usage: python rag.py data/llama2.txt questions.json
"""
import json
import math
import re
import sys
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
CHUNK_SIZES = [100, 400]   # words per chunk
OVERLAP = 0.2               # 20% overlap between neighbouring chunks
K = 5


def norm(s):
    return " ".join(s.split()).lower()


def chunk_words(text, size, overlap):
    words = text.split()
    step = max(1, int(size * (1 - overlap)))
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


class DenseIndex:
    def __init__(self, model, chunks):
        self.chunks = chunks
        self.model = model
        self.E = model.encode(chunks, batch_size=64, normalize_embeddings=True,
                              show_progress_bar=False).astype(np.float32)

    def search(self, q, k):
        qv = self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
        scores = self.E @ qv                     # cosine similarity (vectors are unit length)
        top = np.argsort(-scores)[:k]
        return top, scores[top]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.toks = [re.findall(r"\w+", c.lower()) for c in chunks]
        self.tf = [Counter(t) for t in self.toks]
        self.dl = np.array([len(t) for t in self.toks])
        self.avgdl = self.dl.mean()
        df = Counter(w for t in self.toks for w in set(t))
        N = len(chunks)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        qt = re.findall(r"\w+", q.lower())
        s = np.zeros(len(self.toks))
        for i, tf in enumerate(self.tf):
            for w in qt:
                if w in tf:
                    f = tf[w]
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s

    def search(self, q, k):
        s = self.scores(q)
        top = np.argsort(-s)[:k]
        return top, s[top]


def hybrid_search(dense, bm25, q, k, rrf_k=60):
    """Reciprocal-rank fusion of dense and BM25 rankings."""
    d_top, _ = dense.search(q, 50)
    b_top, _ = bm25.search(q, 50)
    score = Counter()
    for r, i in enumerate(d_top):
        score[i] += 1 / (rrf_k + r)
    for r, i in enumerate(b_top):
        score[i] += 1 / (rrf_k + r)
    top = [i for i, _ in score.most_common(k)]
    return np.array(top), None


def hit_rank(chunks, top, gold):
    g = norm(gold)
    for r, i in enumerate(top):
        if g in norm(chunks[i]):
            return r + 1
    return None


def main(doc_path, q_path):
    text = open(doc_path).read()
    qs = json.load(open(q_path))
    print(f"document words: {len(text.split())}")
    emb = SentenceTransformer(EMB_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    summary, detail, indexes = {}, {}, {}
    for size in CHUNK_SIZES:
        chunks = chunk_words(text, size, OVERLAP)
        dense, bm25 = DenseIndex(emb, chunks), BM25(chunks)
        indexes[size] = (chunks, dense)
        for name, fn in [("dense", lambda q: dense.search(q, K)[0]),
                         ("bm25", lambda q: bm25.search(q, K)[0]),
                         ("hybrid", lambda q: hybrid_search(dense, bm25, q, K)[0])]:
            ranks = [hit_rank(chunks, fn(x["q"]), x["gold"]) for x in qs]
            r1 = sum(r is not None and r <= 1 for r in ranks) / len(qs)
            r5 = sum(r is not None for r in ranks) / len(qs)
            mrr = sum(1 / r for r in ranks if r) / len(qs)
            key = f"chunk{size}_{name}"
            summary[key] = {"n_chunks": len(chunks), "recall@1": round(r1, 3),
                            "recall@5": round(r5, 3), "MRR": round(mrr, 3)}
            detail[key] = ranks
            print(key, summary[key])

    print("=== PER-QUESTION RANK OF GOLD (None = not in top 5) ===")
    for i, x in enumerate(qs):
        print(i + 1, x["q"], {k: v[i] for k, v in detail.items()})

    # show what dense retrieval returned for misses, to analyse failures
    print("=== DENSE MISSES (top-1 chunk shown) ===")
    for size in CHUNK_SIZES:
        chunks, dense = indexes[size]
        for i, x in enumerate(qs):
            if detail[f"chunk{size}_dense"][i] is None:
                top, sc = dense.search(x["q"], K)
                print(json.dumps({"chunk_size": size, "q": x["q"], "gold": x["gold"],
                                  "top1_score": round(float(sc[0]), 3),
                                  "top1": chunks[top[0]][:300]}))

    # final answers with the LLM, using the better chunk size by dense recall@5
    best = max(CHUNK_SIZES, key=lambda s: (summary[f"chunk{s}_dense"]["recall@5"],
                                           summary[f"chunk{s}_dense"]["MRR"]))
    chunks, dense = indexes[best]
    tok = AutoTokenizer.from_pretrained(LLM)
    llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16, device_map="cuda")
    print(f"=== ANSWERS (chunk size {best}, top-{K} context, {LLM}) ===")
    answers = []
    for x in qs:
        top, _ = dense.search(x["q"], K)
        ctx = "\n\n".join(f"[{j + 1}] {chunks[i]}" for j, i in enumerate(top))
        msgs = [{"role": "system", "content": "Answer the question using only the context. "
                 "Be brief. If the answer is not in the context, say you don't know."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {x['q']}"}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        out = llm.generate(**enc, max_new_tokens=80, do_sample=False)
        a = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        answers.append({"q": x["q"], "answer": a})
        print(json.dumps(answers[-1]))

    json.dump({"summary": summary, "ranks": detail, "answers": answers, "best_chunk": best},
              open("results.json", "w"), indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:3])
