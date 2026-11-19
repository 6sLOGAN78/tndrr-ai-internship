# Task 1: LoRA fine-tuning (Qwen2.5-0.5B-Instruct → smart-home JSON)

**Task:** turn a plain-English smart-home command into fixed JSON
`{"action", "device", "room", "value"}` with closed label sets (5 actions, 7 devices, 5 rooms + null, integer or null).

**Data** (`make_data.py`, seed 42): 1,000 synthetic training examples and 100 held-out test examples.
The test set is deliberately out-of-distribution. About 60% of its device words, room words and sentence templates never appear in training
("radiator", "telly", "lounge", "room where I sleep", "kill the…", "bump … up a bit"), and some numbers are written as words ("twenty two").

**Run:** Colab free Tesla T4 (`../run_all.ipynb`). `python make_data.py && python train_eval.py`

## Results (100 held-out examples, greedy decoding, same prompt for both)

| model | valid JSON | **exact match** | action | device | room | value |
|---|---|---|---|---|---|---|
| base Qwen2.5-0.5B-Instruct | 100% | **5%** | 65% | 52% | 28% | 87% |
| + LoRA | 100% | **50%** | 96% | 83% | 66% | 100% |

Training: 250 steps, 90 s, peak 7.4 GB GPU memory, 8.8M trainable params (1.75% of 503M).

Metric: **exact match** means all four fields are correct. Per-field accuracy shows *where* errors come from.

## LoRA settings
| setting | value |
|---|---|
| rank r | 16 |
| alpha | 32 (scale α/r = 2) |
| dropout | 0.05 |
| target modules | q, k, v, o, gate, up, down (all linear layers) |
| learning rate | 2e-4, 20-step warmup, linear decay |
| epochs / batch | 2 / 8 |
| loss | only on answer tokens (prompt tokens masked with -100) |
| precision | fp32 base weights + fp16 autocast |

## Failure analysis (all 50 failures are in `failures_lora.jsonl`)
Wrong fields across the 50 failures: room 34, device 17, action 4.
In **32 of the 34 room errors** the model output `"bathroom"`. When it sees a room phrase it never saw in training
("pantry area", "room where I sleep", "workroom"), it falls back to one default label instead of reasoning about the meaning.

## Files
- `make_data.py`: dataset generator
- `train_eval.py`: base eval → LoRA training → fine-tuned eval
- `results.json`: metrics from the Colab run
- `failures_lora.jsonl`: every test example the fine-tuned model got wrong
