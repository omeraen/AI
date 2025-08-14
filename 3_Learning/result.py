import torch
from unsloth import FastLanguageModel
import pyautogui
import time


max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading the fine-tuned model... ✅")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "learnt_model", # There is fine-tuned model
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    print("Model loaded successfully and ready to work. ✅")
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit()

prompt_template = "### Instruction:\n{}\n\n### Output:\n{}"

print("Agent activated. Enter a command or /bye to quit. ✅")

while True:
    instruction = input("Enter command: ")

    if instruction.lower() in ["/q", "/bye", "exit", "quit"]:
        print("Agent deactivated. ✅")
        break

    # There can be your prompt
    prompt = prompt_template.format(instruction, "")

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0]

    try:
        code_to_execute = response_text.split("### Output:")[1].strip()
        
        if code_to_execute.endswith(tokenizer.eos_token):
            code_to_execute = code_to_execute[:-len(tokenizer.eos_token)].strip()

        print("\n======================\n")
        print(code_to_execute)
        print("\n======================\n")

        confirmation = input("Execute this code? (y/n): ")

        if confirmation.lower() == 'y':
            exec(code_to_execute, {"pyautogui": pyautogui, "time": time})
            print("Executed ✅\n")
        else:
            print("Execution cancelled by user. ❌\n")

    except IndexError:
        print("❌ ERROR: The model did not return code in the expected format.\n")
    except Exception as e:
        print(f"Error type: {type(e).__name__}\n\nERROR: {e}")
        print("\n")