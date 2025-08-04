from llama_cpp import Llama
import os

MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"

# GPU_LAYERS: Сколько слоев модели выгрузить на GPU. "-1" выгружает всё возможное, если у вас не сильная видеокарта, то ставьте 0.
# CONTEXT_SIZE: Размер контекстного окна.
GPU_LAYERS = -1 
CONTEXT_SIZE = 2048

def load_model(model_path: str):
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Файл модели нет: {model_path}")
        print("Скачайте GGUF-файл модели и положите его в эту же папку.")
        return None
    
    print("Загружаем модель")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=CONTEXT_SIZE,
        verbose=False
    )
    print("Модель загружена! ✅")
    return llm

def format_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

def main():
    llm = load_model(MODEL_NAME)
    if not llm:
        return

    print("Отправьте запрос на узбекском, обязательно на узбекском, UZBEKCHA bolsin, iltimos.")
    print("Завершение диалога ------- '/bye'")

    # Бонусное задание: короче, он понимает узбекский как арабский, мне кажется, тут уже скорее проблема самой модели
    system_prompt = """
    Общайся пользователем только на узбекском языке латинскими буквами"""

    while True:
        try:
            user_prompt = input("USer: ")
            if user_prompt.lower() in ["/bye"]:
                print("\n Пока!")
                break
            
            print("Llama 2 ДУмает...")

            full_prompt = format_prompt(system_prompt, user_prompt)

            output = llm(
                full_prompt,
                max_tokens=512, # Макс. количество токенов для ответа
                temperature=0.7,
                stop=["</s>"]
            )
            
            response_text = output["choices"][0]["text"].strip()
            
            print(f"AI ANswer: {response_text}")

        except (KeyboardInterrupt, EOFError):
            print("\n Пока!")
            break
        except Exception as e:
            print(f"ОШИБКА: {e}")

if __name__ == "__main__":
    main()