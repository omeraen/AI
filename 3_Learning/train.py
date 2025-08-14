import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

max_seq_length = 2048
dtype = None
load_in_4bit = True # Use 4bit quantization to fit model in memory

print('Starting the fine-tuning process ✅')

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/phi-3-mini-4k-instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    print("Model loaded successfully. ✅")
except Exception as e:
    print(f"❌ ERROR:{e}")
    exit()

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Мощность адаптера.
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
print("LoRA adapter configured and added to the model. ✅")

prompt_template = "### Instruction:\n{}\n\n### Output:\n{}"
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt_template.format(instruction, output)
        texts.append(text)
    return texts

print(f'"dataset.jsonl" loaded. Examples: {len(dataset)} ✅')


# Аргументы обучения 
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 3, # эпохи
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(), # How often to log training progress
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs", # Where to save the model
    report_to = "none",
)
print("Training parameters configured. ✅")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    formatting_func = formatting_prompts_func,
    args = training_args,
)

print("\n Starting model fine-tuning... ✅")
trainer.train()
print("\n Training completed successfully! ✅")

print("\nSaving the trained adapter... ✅")
model.save_pretrained("learnt_model")
tokenizer.save_pretrained("learnt_model")
print('All done! Your model has been saved in the "learnt_model" folder. ✅')
