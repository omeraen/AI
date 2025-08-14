import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "google/gemma-2b-it" # Your model name
MEMORY_FILE = "memory.json"
MESSAGES = 10  # last 10 messages

SYSTEM_PROMPT = """"""

# Keeps the last 19 messages in memory to maintain a mini-dialogue
class MemoryManager:
    def __init__(self, file_path: str, context_size: int):
        self.file_path = file_path
        self.context_size = context_size

    def load_history(self) -> list:
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def save_history(self, history: list):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"ERROR: {e}")

    def add_entry(self, user_content: str, assistant_content: str):
        history = self.load_history()
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})
        history = history[-self.context_size:]
        self.save_history(history)

    def get_context(self) -> list:
        return self.load_history()


def load_model():
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded! ✅")
        return tokenizer, model
    except Exception as e:
        print(f"ERROR: {e}")
        return None, None


def main():
    tokenizer, model = load_model()
    if not tokenizer or not model:
        return

    memory = MemoryManager(MEMORY_FILE, MESSAGES)
    
    print("End the dialogue with -------> '/bye'")
    
    while True:
        try:
            prompt = input("User: ")
            if prompt.lower() in ["/bye"]:
                print("Bye!")
                break

            print("Thinking...")

            chat_context = memory.get_context()

    
            messages_for_model = list(chat_context) 

            current_user_message = {"role": "user", "content": prompt}

            if not chat_context:
                current_user_message["content"] = f"{SYSTEM_PROMPT}\n\n{prompt}"

            messages_for_model.append(current_user_message)


            text_prompt = tokenizer.apply_chat_template(
                messages_for_model,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)


            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            input_ids_len = model_inputs.input_ids.shape[1]
            generated_ids = generated_ids[:, input_ids_len:]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"AI Answer: {response}")
            memory.add_entry(user_content=prompt, assistant_content=response)

        except (KeyboardInterrupt, EOFError):
            print("Bye!")
            break
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    main()