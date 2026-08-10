"""
Task 3 - Parameter-Efficient Fine-Tuning (LoRA)
Domain: Software Debugging Assistant | Base model: DistilGPT2 (82M params)

Run this in an environment with internet + PyTorch + GPU (recommended) or CPU
(slow but workable given the tiny model/dataset). Not executed inside the
Cowork sandbox (no huggingface.co / pytorch wheel access there).

pip install transformers peft accelerate torch datasets --quiet
"""

import json
import math
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "distilgpt2"
MAX_LENGTH = 256
ADAPTER_SAVE_PATH = "./adapters/distilgpt2-debug-assistant-lora"

# ---------------------------------------------------------------------------
# 1. Load prepared dataset (produced in Task 1: 01_build_dataset.py)
# ---------------------------------------------------------------------------
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "{context_block}"
    "### Response:\n{response}"
)

def format_example(row):
    context_block = f"### Context:\n{row['context']}\n\n" if isinstance(row["context"], str) and row["context"].strip() else ""
    return PROMPT_TEMPLATE.format(
        instruction=row["instruction"], context_block=context_block, response=row["response"]
    )

train_df["text"] = train_df.apply(format_example, axis=1)
val_df["text"] = val_df.apply(format_example, axis=1)

train_ds = Dataset.from_pandas(train_df[["text"]])
val_ds = Dataset.from_pandas(val_df[["text"]])

# ---------------------------------------------------------------------------
# 2. Tokenization pipeline
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    return tokenizer(
        batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length"
    )

train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------------------------------------------------------
# 3. Model + LoRA adapter configuration
# ---------------------------------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                       # rank of the low-rank update matrices
    lora_alpha=16,             # scaling factor (alpha/r = 2x effective LR on adapter)
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],   # GPT-2 attention/projection layers
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Expect roughly ~0.3-0.5% of the 82M base parameters to be trainable
# (only the LoRA A/B matrices), which is the point of parameter-efficient FT.

# ---------------------------------------------------------------------------
# 4. Training configuration
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./lora_training_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch size = 4 * 4 = 16
    learning_rate=2e-4,                # typical LoRA LR (higher than full FT)
    lr_scheduler_type="cosine",
    warmup_ratio=0.06,
    optim="adamw_torch",
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=data_collator,
)

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
train_result = trainer.train()

# training loss / eval loss per epoch are recorded in trainer.state.log_history;
# plot with matplotlib to inspect convergence, e.g.:
#
# import matplotlib.pyplot as plt
# history = trainer.state.log_history
# train_losses = [(h["step"], h["loss"]) for h in history if "loss" in h]
# eval_losses  = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
# plt.plot(*zip(*train_losses), label="train_loss")
# plt.plot(*zip(*eval_losses), label="eval_loss")
# plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.savefig("lora_loss_curve.png")

perplexity = math.exp(trainer.evaluate()["eval_loss"])
print(f"Final validation perplexity: {perplexity:.2f}")

# ---------------------------------------------------------------------------
# 6. Save the trained adapter (NOT the full base model - this is the point of
#    parameter-efficient fine-tuning: the adapter is a few MB, not ~330MB)
# ---------------------------------------------------------------------------
model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
print(f"LoRA adapter saved to {ADAPTER_SAVE_PATH}")

# To reload later:
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# adapted_model = PeftModel.from_pretrained(base, ADAPTER_SAVE_PATH)
