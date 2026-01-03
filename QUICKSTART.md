# Быстрый старт

Этот гайд поможет вам быстро запустить проект.

## Предварительные требования

- **Python 3.11+**
- **Docker** (только для PostgreSQL) или локальная установка PostgreSQL
- **Git** (опционально, для клонирования репозитория)

## Шаг 1: Установка зависимостей

1. Создайте виртуальное окружение (рекомендуется):
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv venv
venv\Scripts\activate.bat

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Примечание для Windows PowerShell:**
Если получаете ошибку "execution of scripts is disabled", выполните:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Шаг 2: Настройка базы данных

### Вариант 1: PostgreSQL через Docker (рекомендуется)

1. Запустите PostgreSQL:
```bash
docker-compose up -d
```

Это запустит только PostgreSQL в Docker контейнере.

2. Проверьте что контейнер работает:
```bash
docker-compose ps
```

**Остановка PostgreSQL:**
```bash
docker-compose down
```

**Удаление данных БД (полный сброс):**
```bash
docker-compose down -v
```

### Вариант 2: Локальная установка PostgreSQL

1. Установите PostgreSQL на вашу машину
2. Создайте базу данных:
```sql
CREATE DATABASE foodestimation;
```

## Шаг 3: Настройка окружения

1. Создайте файл `.env` в корне проекта:
```env
# API Token
API_TOKEN=your-super-secret-token-change-this-in-production

# PostgreSQL (для Docker)
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=foodestimation
POSTGRES_PORT=5432

# DATABASE_URL для подключения
# Если используете Docker:
DATABASE_URL=postgresql://user:password@localhost:5432/foodestimation

# Если используете локальную PostgreSQL, укажите свои данные:
# DATABASE_URL=postgresql://your_user:your_password@localhost:5432/foodestimation

# API
API_PORT=8000

# OpenAI API Key для ChatGPT Vision (опционально)
OPENAI_API_KEY=your-openai-api-key-here

# Прокси для OpenAI API (опционально, если нужен доступ через VPN/прокси)
# Форматы: http://proxy:port, http://user:pass@proxy:port, socks5://proxy:port
# OPENAI_PROXY=http://proxy.example.com:8080
# или используйте стандартные переменные окружения:
# HTTPS_PROXY=http://proxy.example.com:8080
# HTTP_PROXY=http://proxy.example.com:8080

# Feature Toggles
# Использовать дообученную модель для оценки калорий вместо эвристик
USE_TRAINED_MODEL=false

# Если true, API будет возвращать результаты вашей модели, даже если ChatGPT включен.
# Если false, и USE_CHATGPT=true, будут возвращаться результаты ChatGPT.
USE_OWN_MODEL_INSTEAD_CHATGPT=false

# Включить/отключить интеграцию ChatGPT для сбора данных
USE_CHATGPT=true
```

2. Измените `API_TOKEN` на свой секретный токен

## Шаг 4: Подготовка директорий

Создайте директорию для загрузок изображений:
```bash
mkdir -p uploads/images
```

## Шаг 5: Миграции базы данных

Примените миграции для создания таблиц:
```bash
alembic upgrade head
```

Если это первый запуск и миграций еще нет, создайте первую:
```bash
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

## Шаг 6: Запуск приложения

Запустите FastAPI сервер:
```bash
uvicorn app.main:app --reload
```

**Параметры запуска:**
- `--reload` - автоматическая перезагрузка при изменениях (для разработки)
- `--host 0.0.0.0` - доступ со всех интерфейсов (по умолчанию localhost)
- `--port 8000` - порт (по умолчанию 8000)

**Пример с параметрами:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Шаг 7: Проверка работы

1. Откройте браузер и перейдите на:
   - **API**: http://localhost:8000
   - **Документация**: http://localhost:8000/docs
   - **Альтернативная документация**: http://localhost:8000/redoc

2. Проверьте health endpoint:
```bash
curl http://localhost:8000/health
```

## Что дальше?

- Прочитайте [README.md](README.md) для подробной информации о проекте
- Изучите [MIGRATIONS.md](MIGRATIONS.md) для работы с миграциями БД
- Посмотрите [TRAINING.md](TRAINING.md) для обучения модели

## Решение проблем

### Ошибка подключения к БД

Убедитесь что:
- PostgreSQL запущена (если через Docker: `docker-compose ps`)
- `DATABASE_URL` в `.env` правильный
- Порты не заняты другими приложениями

### Ошибка миграций

Если миграции не применяются:
```bash
# Проверьте текущую версию
alembic current

# Посмотрите историю
alembic history

# Создайте миграцию заново
alembic revision --autogenerate -m "Initial migration"
```

### Порт занят

Если порт 8000 занят, используйте другой:
```bash
uvicorn app.main:app --reload --port 8001
```

Или остановите процесс, занимающий порт.

## Структура проекта

```
foodEstimation/
├── app/                    # Основное приложение
│   ├── main.py            # FastAPI приложение
│   ├── models.py          # Модели БД
│   ├── schemas.py         # Pydantic схемы
│   ├── ml_service.py      # ML сервис для распознавания
│   ├── chatgpt_service.py # Интеграция с ChatGPT
│   └── training/          # Модули для обучения
├── alembic/               # Миграции БД
├── uploads/               # Загруженные изображения
├── models/                # Сохраненные модели (gitignored)
├── docker-compose.yml     # Конфигурация PostgreSQL
├── requirements.txt       # Python зависимости
└── .env                   # Переменные окружения (создайте сами)
```
