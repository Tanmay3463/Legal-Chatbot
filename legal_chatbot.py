!pip install -q transformers datasets peft trl accelerate bitsandbytes gradio
!pip install -q einops sentencepiece
#!pip install -U datasets huggingface_hub fsspec



import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import gradio as gr
from transformers import pipeline

print("CUDA available:", torch.cuda.is_available())



dataset = load_dataset("alanila/autotrain-data-text", split="train")

# Show available columns
print("Columns:", dataset.column_names)
print(dataset[0])

def merge_prompt_response(example):
    return {
        "text": f"<s>[INST] {example['Incoming_Description'].strip()} [/INST] {example['Updated_Description'].strip()}</s>"
    }

# Convert to ChatML and drop old fields
dataset = dataset.map(merge_prompt_response, num_proc=4) # Use num_proc for faster processing if you have CPU cores available
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# Verify
print(dataset[0])



# ## 🤖 Load Base Model (TinyLlama) with 4-bit Quantization (QLoRA)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Often recommended for better stability on Ampere+ GPUs (T4, A100, H100)
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# It's good practice to set trust_remote_code=True if you're pulling a community model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True # Good practice for HuggingFace models
)

tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# Add pad token before resizing embeddings
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added PAD token and resized embeddings. New vocab size: {len(tokenizer)}")

# Crucial step for QLoRA: prepare model for k-bit training
# This casts the `lm_head` and `embed_tokens` to float32
# and enables gradient checkpointing for memory efficiency.
model = prepare_model_for_kbit_training(model)

def tokenize_function(example):
    # Use the `add_special_tokens=False` to prevent double adding <s> if your merge_prompt_response already adds it.
    # The tokenizer will add <s> by default if it's a chat tokenizer, but your manual addition is fine too.
    # Ensure max_length covers your typical input + output length
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length", # Or 'longest' if you prefer dynamic padding per batch during training
        max_length=512, # Adjust this based on max expected sequence length
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)


# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# Apply LoRA config to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # This will show you how many parameters are trainable (should be a small fraction)


# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4, # Often a slightly lower learning rate (e.g., 2e-4 or 3e-4) is better for QLoRA
    logging_steps=10,
    save_strategy="epoch",
    fp16=True, # For V100/T4, fp16 is good. If you used bfloat16 for bnb_config, you might use bf16=True here.
    optim="paged_adamw_8bit", # Use an 8-bit optimizer for more memory savings
    push_to_hub=False, # Set to True if you want to push to Hugging Face Hub
    report_to="none", # To avoid warnings if you're not using wandb, etc.
    gradient_checkpointing=True, # Already enabled by prepare_model_for_kbit_training, but good to explicitly set
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    peft_config=lora_config, # Pass peft_config here too for SFTTrainer
    data_collator=data_collator,
    # max_seq_length=512, # You can also specify max_seq_length here for SFTTrainer
)

print("\nStarting training...")
trainer.train()
print("Training finished.")



# Save the model and tokenizer
# The adapters are saved by save_pretrained.
trainer.model.save_pretrained("finetuned_model")
tokenizer.save_pretrained("finetuned_model")
print("Model and tokenizer saved to ./finetuned_model")
model_path = "finetuned_model"

# Load the tokenizer first
finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the base model again for inference, then load the adapters
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Define model_id again for clarity if not global
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16, consistent with training
)

base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)




from peft import PeftModel
finetuned_model = PeftModel.from_pretrained(base_model_for_inference, model_path)
finetuned_model = finetuned_model.merge_and_unload() # Merge LoRA adapters into the base model weights

# Ensure the pad token is correctly set in the reloaded tokenizer and model
if finetuned_tokenizer.pad_token is None:
    finetuned_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # You might want to resize embeddings here too if the merged model's vocabulary


# Verify the embedding size matches the tokenizer vocab size
print(f"\nLoaded Model embedding size: {finetuned_model.get_input_embeddings().weight.shape[0]}")
print(f"Loaded Tokenizer vocab size: {len(finetuned_tokenizer)}")
print(f"Loaded Tokenizer pad_token_id: {finetuned_tokenizer.pad_token_id}")

# Create a pipeline for easier inference
pipe = pipeline(
    "text-generation",
    model=finetuned_model,
    tokenizer=finetuned_tokenizer,
    device_map="auto"
)




def chat_fn(message, history):
    messages = []
    for user_msg, bot_msg in history:
        if user_msg is not None: # Ensure user message exists
            messages.append({"role": "user", "content": user_msg})
        if bot_msg is not None: # Ensure bot message exists
            messages.append({"role": "assistant", "content": bot_msg})

    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Apply the chat template to get the full prompt string
    # add_generation_prompt=True will ensure the final assistant tag is added for generation
    prompt = finetuned_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"\n--- Full Prompt Sent to Model ---\n{prompt}\n---------------------------------\n") # For debugging

    output = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1, # Keep this to reduce repetition
        pad_token_id=finetuned_tokenizer.pad_token_id,
        eos_token_id=finetuned_tokenizer.eos_token_id
    )

    generated_text = output[0]["generated_text"]

    response = generated_text.replace(prompt, "").strip()

    # Sometimes models might regenerate part of the prompt or add an extra <s> or </s>
    # Refine cleaning: remove any leading <s> or trailing </s> if they appear
    if response.startswith("<s>"):
        response = response[len("<s>"):].strip()
    if response.endswith("</s>"):
        response = response[:-len("</s>")].strip()

    return response

print("\nLaunching Gradio Chat Interface...")
gr.ChatInterface(
    chat_fn,
    title="⚖️ Legal Chatbot (QLoRA Fine-Tuned)",
).launch(share=True, debug=True)
