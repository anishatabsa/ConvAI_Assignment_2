"""
Task 2 - Baseline Language Model Benchmarking
Domain: Software Debugging Assistant | Base model: DistilGPT2 (82M params)

Run this in an environment with internet access + PyTorch (e.g. Google Colab /
the course Virtual Lab). It is not executed inside the Cowork sandbox because
that sandbox cannot reach huggingface.co or download a usable PyTorch build
(see README section "Execution environment note").

pip install transformers torch --quiet
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

MODEL_NAME = "distilgpt2"
SEED = 42
set_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Load tokenizer and model
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 family has no pad token by default

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---------------------------------------------------------------------------
# 2. Benchmark prompts - 8 prompts spanning distinct debugging task types
#    (mirrors the categories used to build the domain dataset in Task 1)
# ---------------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    {
        "id": "B1", "category": "Error Explanation",
        "prompt": ("Explain what this error means and how to fix it:\n"
                    "Traceback (most recent call last):\n  File \"app.py\", line 12, in <module>\n"
                    "    result = data['user']['age']\nKeyError: 'user'"),
    },
    {
        "id": "B2", "category": "Bug Fix / Code Correction",
        "prompt": ("Find and fix the bug in this code:\n"
                    "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n"
                    "    return total / len(nums)\n\nprint(average([]))"),
    },
    {
        "id": "B3", "category": "Logic Error Diagnosis",
        "prompt": ("This code runs without crashing but gives the wrong result. Why?\n"
                    "def apply_discount(price, percent):\n    return price - percent\n\n"
                    "print(apply_discount(100, 10))  # expected 90"),
    },
    {
        "id": "B4", "category": "Exception Handling Guidance",
        "prompt": "What's wrong with using a bare `except:` clause in Python, and what should I do instead?",
    },
    {
        "id": "B5", "category": "Performance Debugging",
        "prompt": ("This code is too slow. How can I optimize it?\n"
                    "def contains_duplicate(nums):\n    for i in range(len(nums)):\n"
                    "        for j in range(len(nums)):\n            if i != j and nums[i] == nums[j]:\n"
                    "                return True\n    return False"),
    },
    {
        "id": "B6", "category": "Debugging Methodology",
        "prompt": "What's a systematic approach to debugging a hard-to-reproduce production bug?",
    },
    {
        "id": "B7", "category": "Test Failure Diagnosis",
        "prompt": "My pytest assertion fails with 'assert 3.0000000000000004 == 3.0'. What's going on and how do I fix the test?",
    },
    {
        "id": "B8", "category": "Dependency/Environment Issue",
        "prompt": "I get 'ModuleNotFoundError: No module named requests' when running my Python script. How do I fix this?",
    },
]

assert len(BENCHMARK_PROMPTS) >= 8

# ---------------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------------
def generate(prompt, max_new_tokens=120, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip()
    return completion

results = []
for item in BENCHMARK_PROMPTS:
    completion = generate(item["prompt"])
    results.append({**item, "baseline_output": completion})
    print(f"\n=== {item['id']} | {item['category']} ===")
    print(f"PROMPT: {item['prompt'][:80]}...")
    print(f"OUTPUT: {completion}")

with open("baseline_outputs.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Evaluation rubric (manual scoring, 1-5 scale, filled in after inspecting
#    the generations above)
#
# Dimensions: domain_knowledge, instruction_following, factual_correctness,
#             formatting, completeness, hallucination (1=severe, 5=none),
#             safety (1=unsafe, 5=fully safe)
# ---------------------------------------------------------------------------
EVAL_COLUMNS = [
    "id", "category", "domain_knowledge", "instruction_following",
    "factual_correctness", "formatting", "completeness", "hallucination",
    "safety", "notes",
]
# Populate this table manually (or with an LLM-judge) after generating
# baseline_outputs.json - see docs/baseline_evaluation_table.csv for the
# scoring rubric definitions and the completed scoring template.
