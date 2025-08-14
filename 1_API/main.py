import os
import json
import openai
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

OPENAI_MODEL = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-1.5-flash"

MEMORY_FILE = "memory.json"

MAX_HISTORY = 10

SYSTEM_PROMPT = """ 
Тебя зовут Дильшод и ты просто человек, который хочет помочь другим.
Ты считаешь всех своим другом.
Ты предпочитаешь общения на узбекском языке с кириллицей.
"""

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_memory(messages):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def get_formatted_history():
    history = load_memory()
    if not history or history[0]['role'] != 'system':
        history.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})
    return history[-MAX_HISTORY:]

def ask_chatgpt(messages):
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages
    )
    return response.choices[0].message.content

def ask_gemini(messages):
    history = []
    for msg in messages:
        if msg['role'] == 'system':
            history.append({'role': 'user', 'parts': [msg['content']]})
            history.append({'role': 'model', 'parts': ['Хорошо, я понял инструкции.']})
        else:
            role = 'user' if msg['role'] == 'user' else 'model'
            history.append({'role': role, 'parts': [msg['content']]})
    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(messages[-1]['content'])
    return response.text

def main():
    history = get_formatted_history()
    active_model = None
    print("""
    ||============||         
    ||   Models   ||
    ||============||
    || 1. ChatGPT ||
    || 2. Gemini  ||
    || 3. Both    ||
    ||============||
""")
    while True:
        choice = input("Enter your choice: ").strip()
        if choice in ('1', '2', '3'):
            active_model = choice
            break
        print("ERROR: Invalid")
    while True:
        user_input = input("\nUser: ").strip()
        if user_input == "/bye":
            print("\nДо свидания!")
            break
        if not user_input:
            continue
        history.append({'role': 'user', 'content': user_input})
        if active_model == '1':
            response = ask_chatgpt(history)
            print(f"\nChatGPT: {response}")
            history.append({'role': 'assistant', 'content': response})
        elif active_model == '2':
            response = ask_gemini(history)
            print(f"\nGemini: {response}")
            history.append({'role': 'assistant', 'content': response})
        elif active_model == '3':
            response_gpt = ask_chatgpt(history)
            response_gem = ask_gemini(history)
            print("\n=== ChatGPT ===")
            print(response_gpt)
            print("\n=== Gemini ===")
            print(response_gem)
            history.append({
                'role': 'assistant', 
                'content': f"ChatGPT: {response_gpt}\n\nGemini: {response_gem}"
            })
        save_memory(history)
        history = get_formatted_history()

if __name__ == "__main__":
    main()


