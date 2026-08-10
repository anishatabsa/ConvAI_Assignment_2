"""
Task 4 - Comparative Performance Analysis (baseline vs LoRA-adapted DistilGPT2)

Run after Task 2 (baseline_outputs.json) and Task 3 (saved LoRA adapter) have
been produced in your Colab/Virtual Lab environment.

pip install transformers peft torch pandas matplotlib --quiet
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "distilgpt2"
ADAPTER_PATH = "./adapters/distilgpt2-debug-assistant-lora"

# ---------------------------------------------------------------------------
# 1. Load the adapted model (base + LoRA adapter merged at inference time)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
adapted_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
adapted_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
adapted_model.to(device)

def generate(model, prompt, max_new_tokens=120, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

# ---------------------------------------------------------------------------
# 2. Re-run the SAME 8 benchmark prompts used for the baseline (Task 2)
# ---------------------------------------------------------------------------
with open("baseline_outputs.json") as f:
    baseline_results = json.load(f)

comparison = []
for item in baseline_results:
    instr_prompt = f"### Instruction:\n{item['prompt']}\n\n### Response:\n"
    adapted_output = generate(adapted_model, instr_prompt)
    comparison.append({
        "id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "baseline_output": item["baseline_output"],
        "adapted_output": adapted_output,
    })

with open("comparison_outputs.json", "w") as f:
    json.dump(comparison, f, indent=2)

comp_df = pd.DataFrame(comparison)
print(comp_df[["id", "category"]])

# ---------------------------------------------------------------------------
# 3. Scoring: fill in baseline_vs_adapted_scores.csv with 1-5 ratings per
#    dimension for BOTH baseline and adapted outputs (manual review, or an
#    LLM-judge prompt using the rubric in docs/evaluation_rubric.md), then
#    load it here for the quantitative comparison + plot.
# ---------------------------------------------------------------------------
scores_df = pd.read_csv("baseline_vs_adapted_scores.csv")
# expected columns: id, dimension, baseline_score, adapted_score
# dimensions: accuracy, relevance, domain_specificity, instruction_adherence,
#             consistency, fluency, hallucination (5=none), completeness

mean_scores = scores_df.groupby("dimension")[["baseline_score", "adapted_score"]].mean()
print(mean_scores)

# ---------------------------------------------------------------------------
# 4. Visualization: grouped bar chart, baseline vs adapted, per dimension
# ---------------------------------------------------------------------------
ax = mean_scores.plot(kind="bar", figsize=(10, 5), color=["#C44E52", "#55A868"])
ax.set_ylabel("Mean score (1-5)")
ax.set_title("Baseline vs LoRA-Adapted DistilGPT2 — Mean Evaluation Scores")
ax.set_ylim(0, 5)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("comparison_scores.png", dpi=130)
print("Saved comparison_scores.png")
