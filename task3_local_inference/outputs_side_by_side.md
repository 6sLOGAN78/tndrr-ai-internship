# Same 10 prompts: fp16 vs q4_K_M (Qwen2.5-1.5B-Instruct, Ollama, T4, temperature 0, max 256 tokens)

Raw outputs copied from the Colab run (`../run_all.ipynb`). Long answers that hit the 256-token cap are cut off in both settings.

| # | prompt | fp16 | q4_K_M | note |
|---|---|---|---|---|
| 1 | hash table in two sentences | correct, gave 1 sentence | almost identical ("lookup" vs "search") | tie |
| 2 | iterative Fibonacci in Python | correct code, 1-indexed (fib(1)=0, fib(10)=34) | correct code, 0-indexed (fib(10)=55), has docstring | tie, different conventions |
| 3 | 17 × 23 with working | 391, long multiplication | 391, distributive method | tie |
| 4 | translate to French | "La réunion a été transférée au après-midi de jeudi." | identical | tie (both slightly unidiomatic) |
| 5 | 9:40 + 2h35m | **wrong: 1:20** (muddled arithmetic) | **correct: 12:15** | q4 better here |
| 6 | 3 TCP vs UDP differences | correct, plain list | correct, more detail, cut off at 256 tokens | tie |
| 7 | Romeo & Juliet in one sentence | slightly wrong ("families clash due to their love for each other") | correct and tighter | q4 better |
| 8 | haiku on monsoon | repeats "sheets", 5-7-5 roughly | fine | q4 slightly better |
| 9 | bloops/razzies syllogism | correct, transitive property | correct, transitive property | tie |
| 10 | JSON person | identical | identical | tie |

**Observation:** on these 10 prompts, 4-bit quantisation caused no visible quality loss. The two settings agree on 6/10.
Where they differ, the fp16 answer was actually worse twice (arithmetic slip, summary error). With greedy decoding,
small changes in the weights can change which token wins early on, and then the rest of the answer follows a different
path. 10 prompts are far too few to say q4 is "better". The fair conclusion is "no measurable degradation at this scale."

---

## Full outputs

### 1. Explain what a hash table is in two sentences.
**fp16:** A hash table is a data structure that uses a hash function to map keys to array indices, allowing for efficient insertion, deletion, and search operations.
**q4_K_M:** A hash table is a data structure that uses a hash function to map keys to array indices, allowing for efficient insertion, deletion, and lookup operations.

### 3. What is 17 * 23? Show your working.
**fp16:** long multiplication: 17×3 = 51, 17×2 = 34 → 340, 51 + 340 = **391**.
**q4_K_M:** 17 × (20 + 3) = 340 + 51 = **391**.

### 5. A train leaves at 9:40 and the trip takes 2 hours 35 minutes. When does it arrive?
**fp16:** "…Add 15 minutes to 9:40 to get 10:55. Add the remaining 145 minutes… Therefore, the train arrives at **1:20**." (wrong)
**q4_K_M:** "40 + 35 = 75 minutes… 9 + 2 = 11 hours… 75 minutes is 1 hour 15 minutes… the train arrives at **12:15**." (correct)

### 7. Summarise the plot of Romeo and Juliet in one sentence.
**fp16:** Romeo and Juliet is a tragic love story set in Verona, Italy, where two young and feuding families, the Montagues and the Capulets, clash due to their love for each other, leading to a series of tragic events.
**q4_K_M:** Romeo and Juliet is a tragic love story about two young Italian teenagers from feuding families who fall in love and tragically die together.

### 8. Write a haiku about monsoon rain.
**fp16:** Rain pours down in sheets, / Fields dance to the thunder's beat, / Nature sings in sheets.
**q4_K_M:** Thunderous drums of rain, / Fields dance with water's embrace, / Nature's symphony.

### 10. Give a JSON object with keys name, age, city for a fictional person.
**both:** `{"name": "Alice Johnson", "age": 30, "city": "New York"}`

(Prompts 2, 4, 6, 9: see the table. The full text is in the saved Colab notebook output.)
