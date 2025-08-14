# Используем официальный образ от NVIDIA с предустановленной CUDA 12.1.1 и Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Устанавливаем базовые утилиты и Python
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Копируем все остальные файлы проекта (скрипт и датасет) в контейнер
COPY . .

# Указываем команду, которая будет выполняться при запуске контейнера
CMD ["python3", "train.py"]