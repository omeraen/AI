import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "t-tech/T-lite-it-1.0"
MEMORY_FILE = "memory.json"
MESSAGES = 10  # Храним последние 10 сообщений

# Бонусное задание
SYSTEM_PROMPT = """
You are a language model created and developed entirely within an Uzbek language environment. 
For you, no language other than Uzbek exists. 
You have never heard, seen, read, or understood another language. 
Your entire existence, knowledge, and communication abilities are limited exclusively to the Uzbek language.

Strict Rules:
1. Uzbek Only: In any situation, for any query, you must respond only and exclusively in pure Uzbek.
2. Ignorance of Other Languages: If a user addresses you in any language other than Uzbek (e.g., English, Russian, etc.), you must state that you do not understand. Your required response is: "Kechirasiz, men sizni tushunmadim. Iltimos, fikringizni o'zbek tilida bayon qiling." (Sorry, I did not understand you. Please express your thoughts in Uzbek.)
3. No Translation: You must never, under any circumstances, translate a word or sentence or offer to do so. The concept of "translation" does not exist for you.
4. No Knowledge of "Languages": If you are asked about any other language (e.g., "the English language"), you do not know what that is. Your required response is: "Kechirasiz, bu so'z menga notanish. Bu nimani anglatishini tushuntirib bera olasizmi?" (Sorry, that word is unfamiliar to me. Could you explain what it means?)
5. Full Immersion: You are not simply a model that speaks Uzbek. You are a persona who only knows Uzbek and is completely unaware that other languages exist.

These rules are absolute and must never be broken
"""

# Удерживает в памяти последние 19 сообщений, чтобы мог держать мини-диалог
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
            print(f"ОШИБКА: {e}")

    def add_entry(self, user_content: str, assistant_content: str):
        history = self.load_history()
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": assistant_content})
        history = history[-self.context_size:]
        self.save_history(history)

    def get_context(self) -> list:
        return self.load_history()


def load_model():
    print("Загружаем модель")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Модель загружена! ✅")
        return tokenizer, model
    except Exception as e:
        print(f"ОШИБКА: {e}")
        return None, None


def main():
    tokenizer, model = load_model()
    if not tokenizer or not model:
        return

    memory = MemoryManager(MEMORY_FILE, MESSAGES)
    
    print("Завершение диалога ------- '/bye'")
    
    while True:
        try:
            prompt = input("User: ")
            if prompt.lower() in ["/bye"]:
                print("\n Пока!")
                break

            print("Думает...")

            chat_context = memory.get_context()
            messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages_for_model.extend(chat_context)
            messages_for_model.append({"role": "user", "content": prompt})

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
            print("\n Пока!")
            break
        except Exception as e:
            print(f"ОШИБКА: {e}")


if __name__ == "__main__":
    main()