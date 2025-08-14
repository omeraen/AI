# AI

- **1_API/** — REST API для взаимодействия с сервисом.
- **2_Local/** — локальный режим работы с памятью и обработкой данных.
- **3_Learning/** — обучение модели на датасете.

## Структура проекта

```
synveta/
├── 1_API/
│   ├── .env
│   ├── main.py
│   └── requirements.txt
├── 2_Local/
│   ├── main.py
│   ├── memory.json
│   └── requirements.txt
├── 3_Learning/
│   ├── dataset.jsonl
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── result.py
│   └── train.py
└── README.md
```

## 1. API

- **main.py** — основной файл запуска REST API.
- **.env** — переменные окружения для конфигурации сервиса.
- **requirements.txt** — зависимости для запуска API.

### Запуск

```sh
cd 1_API
pip install -r requirements.txt
python main.py
```

## 2. Local

- **main.py** — локальная обработка данных и взаимодействие с памятью.
- **memory.json** — файл для хранения локальных данных.
- **requirements.txt** — зависимости для локального режима.

### Запуск

```sh
cd 2_Local
pip install -r requirements.txt
python main.py
```

## 3. Learning

- **train.py** — скрипт для обучения модели.
- **result.py** — анализ и вывод результатов обучения.
- **dataset.jsonl** — датасет для обучения.
- **Dockerfile** — контейнеризация процесса обучения.
- **requirements.txt** — зависимости для обучения.

### Запуск обучения

```sh
cd 3_Learning
pip install -r requirements.txt
python train.py
```

Для запуска в Docker:

```sh
docker build -t synveta-learning .
docker run --rm synveta-learning
```

## Требования

- Python 3.11+
- Docker (для обучения в контейнере)

## Описание

Проект предназначен для обработки, хранения и обучения на пользовательских данных. API предоставляет интерфейс для интеграции, локальный режим — для автономной работы, а компонент обучения — для построения и тестирования моделей.
