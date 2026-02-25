import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel

model_name = "vngrs-ai/Kumru-2B-Base"

#--4-bit config--------------------------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
#----------------------------------------------------------------------------------------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False # Good for training

#--tokenizer-----------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# --LoRA config--------------------------------------------------------------------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
#-----------------------------------------------------------------------------------------------------------

model = get_peft_model(model, lora_config)
print("Model wrapped with LoRA:")
model.print_trainable_parameters()

#-----------------------------------------------------------------------------------------------------------

dataset = load_dataset("json", data_files="sampledata.jsonl", split="train")                   #<---- in data_files, you can include your own dataset
def format_prompt(example):
    prompt = example['text'] + tokenizer.eos_token
  
    result = tokenizer(
        prompt,
        truncation=True,  
        max_length=512,                                   #<-----you can change this if you want shorter answers.
        padding=False,
    )

    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(format_prompt)

#------------------------------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./training-checkpoints", 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    num_train_epochs=10,                        
    learning_rate=2e-4,
    fp16=True, 
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

#------------------------------------------------------------------------------------------------------------
print("Starting training...")
trainer.train()

final_adapter_path = "./kumru-2B-finetuned"
trainer.save_model(final_adapter_path)

print(f"Training complete! Your new model adapter is saved in: {final_adapter_path}")
