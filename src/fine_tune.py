import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

# -------------------------- Load JSON dataset --------------------------
def load_dataset_from_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return Dataset.from_list(data)

# ---------------------- Preprocess dataset -----------------------------
def preprocess(example, tokenizer, max_input_length=256, max_target_length=256):
    input_text = f"{example['instruction']}\n{example['input']}"
    target_text = example["output"]

    inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_text,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze() for k, v in inputs.items()}

# ----------------------- Load model and tokenizer -----------------------
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, base_model

# ----------------------- Configure LoRA --------------------------------
def apply_lora(model):
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model

# --------------------- Main fine-tuning function ------------------------
def main():
    dataset_path = "instruction_dataset.json"
    model_name = "t5-small"

    # Load tokenizer and base model
    tokenizer, model = load_model(model_name)

    # Load and preprocess dataset
    raw_dataset = load_dataset_from_json(dataset_path)
    tokenized_dataset = raw_dataset.map(lambda x: preprocess(x, tokenizer), batched=False)

    # Apply LoRA
    model = apply_lora(model)

    # Prepare trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=10,
        logging_steps=5,
        fp16=torch.cuda.is_available()
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save final model
    model.save_pretrained("./final_model_lora")
    tokenizer.save_pretrained("./final_model_lora")

    print("✅ Fine-tuning complete. Model saved to ./final_model_lora")

if __name__ == "__main__":
    main()
