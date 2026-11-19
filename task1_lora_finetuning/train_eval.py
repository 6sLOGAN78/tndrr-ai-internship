"""LoRA fine-tune Qwen2.5-0.5B-Instruct on smart-home command -> JSON,
then compare base vs fine-tuned on the held-out test set.

Usage:  python train_eval.py            (needs train.jsonl / test.jsonl from make_data.py)
"""
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
LR, EPOCHS, BATCH, MAX_LEN = 2e-4, 2, 8, 192
SEED = 0

SYSTEM = (
    "Convert the user's smart-home command into JSON with exactly these keys: "
    '"action" (one of turn_on, turn_off, set, increase, decrease), '
    '"device" (one of light, fan, ac, heater, tv, speaker, blinds), '
    '"room" (one of living_room, bedroom, kitchen, office, bathroom, or null), '
    '"value" (an integer, or null if no number is given). Output only the JSON.'
)

torch.manual_seed(SEED)
random.seed(SEED)
dev = "cuda"


def load(path):
    return [json.loads(l) for l in open(path)]


def prompt_text(tok, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse(out):
    out = out.strip()
    s, e = out.find("{"), out.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        d = json.loads(out[s:e + 1])
        return d if isinstance(d, dict) else None
    except Exception:
        return None


@torch.no_grad()
def evaluate(model, tok, rows, tag):
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        enc = tok([prompt_text(tok, r["text"]) for r in batch], return_tensors="pt",
                  padding=True).to(dev)
        out = model.generate(**enc, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for r, o in zip(batch, out):
            raw = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            preds.append({"text": r["text"], "target": r["target"], "raw": raw, "pred": parse(raw)})
    keys = ["action", "device", "room", "value"]
    n = len(preds)
    valid = sum(p["pred"] is not None for p in preds)
    exact = sum(p["pred"] is not None and all(p["pred"].get(k) == p["target"][k] for k in keys)
                for p in preds)
    field = {k: sum(p["pred"] is not None and p["pred"].get(k) == p["target"][k] for p in preds) / n
             for k in keys}
    m = {"model": tag, "n": n, "valid_json": valid / n, "exact_match": exact / n,
         **{f"acc_{k}": round(v, 3) for k, v in field.items()}}
    print(json.dumps(m))
    return m, preds


def build_train_batch(tok, rows):
    ids, labels = [], []
    for r in rows:
        p = tok(prompt_text(tok, r["text"]), add_special_tokens=False)["input_ids"]
        a = tok(json.dumps(r["target"]) + tok.eos_token, add_special_tokens=False)["input_ids"]
        x = (p + a)[:MAX_LEN]
        y = ([-100] * len(p) + a)[:MAX_LEN]  # loss only on the answer tokens
        ids.append(x)
        labels.append(y)
    L = max(map(len, ids))
    pad = tok.pad_token_id
    inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
    lab = torch.tensor([y + [-100] * (L - len(y)) for y in labels])
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
    return inp.to(dev), att.to(dev), lab.to(dev)


def main():
    train, test = load("train.jsonl"), load("test.jsonl")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    print("GPU:", torch.cuda.get_device_name(0))

    base_m, base_p = evaluate(model, tok, test, "base")

    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR,
                            weight_decay=0.0)
    steps_total = EPOCHS * math.ceil(len(train) / BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))
    scaler = torch.cuda.amp.GradScaler()

    torch.cuda.reset_peak_memory_stats()
    t0, step = time.time(), 0
    tok.padding_side = "right"
    for ep in range(EPOCHS):
        model.train()
        random.shuffle(train)
        for i in range(0, len(train), BATCH):
            inp, att, lab = build_train_batch(tok, train[i:i + BATCH])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"epoch {ep} step {step}/{steps_total} loss {loss.item():.4f}")
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"train_time_s={train_time:.1f} peak_mem_GB={peak:.2f}")

    ft_m, ft_p = evaluate(model, tok, test, "lora")

    results = {"base": base_m, "lora": ft_m, "train_time_s": round(train_time, 1),
               "peak_train_mem_GB": round(peak, 2), "lora_config": LORA, "lr": LR,
               "epochs": EPOCHS, "batch": BATCH, "model": MODEL, "n_train": len(train)}
    json.dump(results, open("results.json", "w"), indent=2)
    with open("predictions.jsonl", "w") as f:
        for b, l in zip(base_p, ft_p):
            f.write(json.dumps({"text": b["text"], "target": b["target"],
                                "base_raw": b["raw"], "lora_raw": l["raw"]}) + "\n")

    print("=== RESULTS ===")
    print(json.dumps(results))
    print("=== FINE-TUNED FAILURES ===")
    for l in ft_p:
        if l["pred"] is None or any(l["pred"].get(k) != l["target"][k] for k in l["target"]):
            print(json.dumps({"text": l["text"], "target": l["target"], "pred": l["raw"]}))
    print("=== BASE SAMPLE OUTPUTS ===")
    for b in base_p[:8]:
        print(json.dumps({"text": b["text"], "target": b["target"], "pred": b["raw"]}))


if __name__ == "__main__":
    main()
