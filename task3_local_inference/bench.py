"""Benchmark one model in two quantisation settings with Ollama.

Measures decode tokens/s (from Ollama's own eval_count / eval_duration),
prompt-processing speed, and GPU memory used while the model is loaded,
then saves both settings' outputs on the same 10 prompts for comparison.

Usage: python bench.py   (expects `ollama serve` running and both tags pulled)
"""
import json
import subprocess
import time
import urllib.request

SETTINGS = {
    "fp16": "qwen2.5:1.5b-instruct-fp16",
    "q4_K_M": "qwen2.5:1.5b-instruct-q4_K_M",
}
PROMPTS = [
    "Explain what a hash table is in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is 17 * 23? Show your working.",
    "Translate to French: 'The meeting has been moved to Thursday afternoon.'",
    "A train leaves at 9:40 and the trip takes 2 hours 35 minutes. When does it arrive?",
    "List three differences between TCP and UDP.",
    "Summarise the plot of Romeo and Juliet in one sentence.",
    "Write a haiku about monsoon rain.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Explain.",
    "Give a JSON object with keys name, age, city for a fictional person.",
]
URL = "http://localhost:11434/api"


def post(path, body):
    req = urllib.request.Request(f"{URL}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def gpu_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.split("\n")[0])


def main():
    results = {}
    outputs = {}
    for name, tag in SETTINGS.items():
        # unload everything first so memory numbers are clean
        for other in SETTINGS.values():
            try:
                post("generate", {"model": other, "keep_alive": 0})
            except Exception:
                pass
        time.sleep(3)
        idle = gpu_mem_mb()
        post("generate", {"model": tag, "prompt": "hi", "stream": False,
                          "options": {"num_predict": 8}})  # warm-up / load
        loaded = gpu_mem_mb()
        ps = json.loads(urllib.request.urlopen(f"{URL}/ps").read())
        size_vram = next((m.get("size_vram") for m in ps.get("models", []) if m["name"] == tag), None)

        tps, pps, outs = [], [], []
        for p in PROMPTS:
            r = post("generate", {"model": tag, "prompt": p, "stream": False,
                                  "options": {"temperature": 0, "seed": 0, "num_predict": 256}})
            tps.append(r["eval_count"] / (r["eval_duration"] / 1e9))
            pps.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
            outs.append(r["response"].strip())
        peak = gpu_mem_mb()
        results[name] = {
            "tag": tag,
            "decode_tok_per_s_mean": round(sum(tps) / len(tps), 1),
            "decode_tok_per_s_min": round(min(tps), 1),
            "decode_tok_per_s_max": round(max(tps), 1),
            "prompt_tok_per_s_mean": round(sum(pps) / len(pps), 1),
            "gpu_mem_idle_MB": idle,
            "gpu_mem_loaded_MB": loaded,
            "gpu_mem_after_runs_MB": peak,
            "model_mem_delta_MB": loaded - idle,
            "ollama_size_vram_MB": round(size_vram / 1e6) if size_vram else None,
        }
        outputs[name] = outs
        print(json.dumps({name: results[name]}))

    json.dump({"results": results, "prompts": PROMPTS, "outputs": outputs},
              open("results.json", "w"), indent=2)
    print("=== OUTPUTS SIDE BY SIDE ===")
    for i, p in enumerate(PROMPTS):
        print(f"\n##### Prompt {i + 1}: {p}")
        for name in SETTINGS:
            print(f"--- {name} ---\n{outputs[name][i]}")
    print("=== SUMMARY ===")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
