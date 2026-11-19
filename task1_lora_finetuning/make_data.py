"""Generate a synthetic "smart-home command -> JSON" dataset.

Each example maps a plain-English request to a fixed JSON schema:
  {"action": ..., "device": ..., "room": ..., "value": ...}

The test set is deliberately harder than train: it also uses phrasings,
synonyms and number words that never appear in the training data, so we
measure generalisation and not just memorisation.
"""
import json
import random

random.seed(42)

ACTIONS = ["turn_on", "turn_off", "set", "increase", "decrease"]
DEVICES = ["light", "fan", "ac", "heater", "tv", "speaker", "blinds"]
ROOMS = ["living_room", "bedroom", "kitchen", "office", "bathroom", None]

# surface forms: (seen in train, held out for test only)
DEVICE_WORDS = {
    "light":   (["light", "lights", "lamp"], ["bulb"]),
    "fan":     (["fan", "ceiling fan"], ["table fan"]),
    "ac":      (["ac", "air conditioner", "AC"], ["aircon", "cooler"]),
    "heater":  (["heater", "space heater"], ["radiator"]),
    "tv":      (["tv", "television", "TV"], ["telly"]),
    "speaker": (["speaker", "music speaker"], ["sound system"]),
    "blinds":  (["blinds", "curtains"], ["shades"]),
}
ROOM_WORDS = {
    "living_room": (["living room", "hall"], ["lounge", "sitting room"]),
    "bedroom":     (["bedroom", "master bedroom"], ["room where I sleep"]),
    "kitchen":     (["kitchen"], ["pantry area"]),
    "office":      (["office", "study"], ["home office", "workroom"]),
    "bathroom":    (["bathroom", "washroom"], ["restroom"]),
}
# which value each device takes when "set"
VALUE_RANGE = {
    "light": (10, 100), "fan": (1, 5), "ac": (16, 30), "heater": (18, 30),
    "tv": (5, 60), "speaker": (5, 80), "blinds": (0, 100),
}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 10: "ten",
             16: "sixteen", 18: "eighteen", 20: "twenty", 22: "twenty two",
             24: "twenty four", 25: "twenty five", 30: "thirty", 50: "fifty"}

TEMPLATES = {
    "turn_on": (
        ["turn on the {dev}{room}", "switch on the {dev}{room}", "please turn the {dev} on{room}",
         "{dev} on{room}", "can you start the {dev}{room}"],
        ["could you fire up the {dev}{room}", "I want the {dev}{room} running"],
    ),
    "turn_off": (
        ["turn off the {dev}{room}", "switch off the {dev}{room}", "please shut the {dev} off{room}",
         "{dev} off{room}", "stop the {dev}{room}"],
        ["kill the {dev}{room}", "I don't need the {dev}{room} anymore"],
    ),
    "set": (
        ["set the {dev}{room} to {val}", "change the {dev}{room} to {val}", "put the {dev}{room} at {val}",
         "{dev}{room} to {val} please", "make the {dev}{room} {val}"],
        ["I'd like the {dev}{room} on {val}", "adjust {dev}{room} so it's at {val}"],
    ),
    "increase": (
        ["increase the {dev}{room}", "turn up the {dev}{room}", "increase the {dev}{room} by {val}",
         "raise the {dev}{room} by {val}"],
        ["bump the {dev}{room} up a bit", "crank up the {dev}{room} by {val}"],
    ),
    "decrease": (
        ["decrease the {dev}{room}", "turn down the {dev}{room}", "decrease the {dev}{room} by {val}",
         "lower the {dev}{room} by {val}"],
        ["bring the {dev}{room} down a little", "dial down the {dev}{room} by {val}"],
    ),
}


def pick(pair, held_out):
    seen, unseen = pair
    if held_out and unseen and random.random() < 0.6:
        return random.choice(unseen)
    return random.choice(seen)


def make_example(held_out=False):
    action = random.choice(ACTIONS)
    device = random.choice(DEVICES)
    room = random.choice(ROOMS)
    tmpl = pick(TEMPLATES[action], held_out)

    value = None
    val_text = ""
    if "{val}" in tmpl:
        if action == "set":
            lo, hi = VALUE_RANGE[device]
            value = random.randint(lo, hi)
        else:
            value = random.choice([1, 2, 3, 5, 10])
        # number words only appear in the test split
        if held_out and value in NUM_WORDS and random.random() < 0.5:
            val_text = NUM_WORDS[value]
        else:
            val_text = str(value)
        if action == "set" and device in ("ac", "heater") and random.random() < 0.5:
            val_text += " degrees"
        if action == "set" and device in ("light", "blinds") and random.random() < 0.5:
            val_text += "%" if random.random() < 0.5 else " percent"

    dev_text = pick(DEVICE_WORDS[device], held_out)
    room_text = ""
    if room is not None:
        r = pick(ROOM_WORDS[room], held_out)
        room_text = random.choice([f" in the {r}", f" in {r}"])

    text = tmpl.format(dev=dev_text, room=room_text, val=val_text).strip()
    if random.random() < 0.3:
        text = text[0].upper() + text[1:]
    target = {"action": action, "device": device, "room": room, "value": value}
    return {"text": text, "target": target}


def main():
    train, seen = [], set()
    while len(train) < 1000:
        ex = make_example(held_out=False)
        if ex["text"] not in seen:
            seen.add(ex["text"])
            train.append(ex)
    test = []
    while len(test) < 100:
        ex = make_example(held_out=True)
        if ex["text"] not in seen:
            seen.add(ex["text"])
            test.append(ex)
    for name, rows in [("train.jsonl", train), ("test.jsonl", test)]:
        with open(name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"train={len(train)} test={len(test)}")
    for r in test[:5]:
        print(r)


if __name__ == "__main__":
    main()
