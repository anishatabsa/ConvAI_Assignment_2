"""
Task 5 (part 2) - Preference Optimization with DPO
Domain: Software Debugging Assistant | Model: LoRA-adapted DistilGPT2 (Task 3)

Small-scale Direct Preference Optimization (DPO) run on the 15-example
preference dataset from build_preference_dataset.py. Run this in an
environment with internet + PyTorch (Colab / Virtual Lab). Not executed
inside the Cowork sandbox (no huggingface.co / pytorch wheel access there).

pip install trl peft transformers torch datasets --quiet
"""

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "distilgpt2"
SFT_ADAPTER_PATH = "./adapters/distilgpt2-debug-assistant-lora"   # from Task 3
DPO_ADAPTER_SAVE_PATH = "./adapters/distilgpt2-debug-assistant-dpo"

# ---------------------------------------------------------------------------
# 1. Load the LoRA-instruction-tuned model from Task 3 as the starting point
#    for preference optimization (policy model). TRL's DPOTrainer will
#    internally construct/handle the frozen reference model.
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER_PATH)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, is_trainable=True)

# ---------------------------------------------------------------------------
# 2. Load + format the preference dataset (prompt / chosen / rejected)
# ---------------------------------------------------------------------------
records = []
with open("preference_dataset_dpo.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
formatted = [
    {
        "prompt": PROMPT_TEMPLATE.format(instruction=r["prompt"]),
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    }
    for r in records
]

pref_dataset = Dataset.from_list(formatted)
split_ds = pref_dataset.train_test_split(test_size=0.2, seed=42)  # 12 train / 3 eval
train_pref, eval_pref = split_ds["train"], split_ds["test"]

# ---------------------------------------------------------------------------
# 3. DPO configuration
# ---------------------------------------------------------------------------
dpo_config = DPOConfig(
    output_dir="./dpo_training_checkpoints",
    beta=0.1,                          # KL-penalty strength vs. the reference (SFT) model
    num_train_epochs=3,                # small dataset -> few epochs to avoid overfitting
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,     # effective batch size = 4
    learning_rate=5e-5,                # lower LR than SFT stage - preference tuning is delicate
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_torch",
    max_prompt_length=192,
    max_length=320,
    eval_strategy="epoch",
    logging_steps=2,
    report_to="none",
    seed=42,
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=None,        # None -> TRL derives the reference by disabling the LoRA adapter
    args=dpo_config,
    train_dataset=train_pref,
    eval_dataset=eval_pref,
    processing_class=tokenizer,
)

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
dpo_result = trainer.train()

# Key metrics to inspect in trainer.state.log_history:
#   rewards/chosen, rewards/rejected  -> should diverge (chosen > rejected)
#   rewards/accuracies                -> fraction of pairs where chosen > rejected
#   rewards/margins                   -> should increase over training

trainer.save_model(DPO_ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(DPO_ADAPTER_SAVE_PATH)
print(f"DPO-aligned adapter saved to {DPO_ADAPTER_SAVE_PATH}")

# ---------------------------------------------------------------------------
# 5. Quick qualitative check: generate on a held-out preference prompt with
#    the SFT-only model vs. the DPO-aligned model and compare.
# ---------------------------------------------------------------------------
def generate(model, prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

test_prompt = PROMPT_TEMPLATE.format(
    instruction=eval_pref[0]["prompt"].replace(PROMPT_TEMPLATE.format(instruction=""), "")
)
print("\n=== DPO-aligned model sample output ===")
print(generate(trainer.model, test_prompt))
