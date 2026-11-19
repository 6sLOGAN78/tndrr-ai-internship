"""Builds run_all.ipynb: writes each task's source files into Colab and runs them."""
import json

def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def wf(path, dest): return code(f"%%writefile {dest}\n" + open(path).read())

cells = [
  md("# tndrr AI internship assignment: all three tasks\nRun on a Colab T4 GPU (Runtime → Run all)."),
  code("!nvidia-smi\n!pip -q uninstall -y torchao\n!pip -q install peft sentence-transformers\n!apt-get -qq update > /dev/null && apt-get -qq install -y poppler-utils zstd lshw > /dev/null\n!mkdir -p task1 task2/data task3"),
  md("## Task 1: LoRA fine-tuning (Qwen2.5-0.5B-Instruct, smart-home command → JSON)"),
  wf("task1_lora_finetuning/make_data.py", "task1/make_data.py"),
  wf("task1_lora_finetuning/train_eval.py", "task1/train_eval.py"),
  code("%cd /content/task1\n!python make_data.py && python train_eval.py\n%cd /content"),
  md("## Task 2: RAG from scratch over the Llama 2 paper (77 pages)"),
  wf("task2_rag_from_scratch/questions.json", "task2/questions.json"),
  wf("task2_rag_from_scratch/rag.py", "task2/rag.py"),
  code("%cd /content/task2\n!curl -sL -o data/llama2.pdf https://arxiv.org/pdf/2307.09288 && pdftotext data/llama2.pdf data/llama2.txt && pdfinfo data/llama2.pdf | grep Pages\n!python rag.py data/llama2.txt questions.json\n%cd /content"),
  md("## Task 3: Ollama, fp16 vs q4_K_M (Qwen2.5-1.5B-Instruct)"),
  wf("task3_local_inference/bench.py", "task3/bench.py"),
  code("!curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1; ollama --version\n"
       "import subprocess, time, urllib.request\n"
       "srv = subprocess.Popen(['ollama', 'serve'], stdout=open('/content/ollama.log', 'w'), stderr=subprocess.STDOUT)\n"
       "for _ in range(60):\n"
       "    try:\n"
       "        urllib.request.urlopen('http://localhost:11434/api/version'); print('ollama server up'); break\n"
       "    except Exception:\n"
       "        time.sleep(1)\n"
       "!ollama pull qwen2.5:1.5b-instruct-fp16 > /dev/null 2>&1; ollama pull qwen2.5:1.5b-instruct-q4_K_M > /dev/null 2>&1; ollama list"),
  code("%cd /content/task3\n!python bench.py\n%cd /content"),
  code("!cat /content/task1/results.json /content/task2/results.json /content/task3/results.json > /dev/null && echo ALL_DONE"),
]
nb = {"nbformat": 4, "nbformat_minor": 0, "cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open("run_all.ipynb", "w"), indent=1)
print("ok", len(cells), "cells")
