# Оптимизация Docker образа

## Проблема: большой размер образа из-за PyTorch

При сборке Docker образа загружается PyTorch (~900MB), что замедляет сборку.

## Решения

### Вариант 1: CPU-only версия PyTorch (рекомендуется для серверов без GPU)

Создайте `requirements-cpu.txt`:

```txt
# ... остальные зависимости ...
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2.0
torchvision>=0.17.0
```

И используйте в Dockerfile:

```dockerfile
COPY requirements-cpu.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
```

**Преимущества:**
- Размер образа меньше (~200MB вместо ~900MB для torch)
- Быстрее сборка
- Достаточно для CPU-серверов

**Недостатки:**
- Не будет работать на GPU (но код автоматически использует CPU)

### Вариант 2: Многоэтапная сборка (multi-stage build)

```dockerfile
# Этап 1: Установка зависимостей
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Этап 2: Финальный образ
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY ./app /app/app
ENV PATH=/root/.local/bin:$PATH
```

### Вариант 3: Использование кэша Docker

Docker кэширует слои, поэтому при повторной сборке torch не будет загружаться заново, если `requirements.txt` не изменился.

**Проверка кэша:**
```bash
docker build --cache-from foodestimation_api .
```

### Вариант 4: Pre-built образ с PyTorch

Используйте официальный образ с предустановленным PyTorch:

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# ... остальное ...
```

## Текущая ситуация

Сейчас используется стандартный `requirements.txt` с полной версией PyTorch. Это нормально, если:
- У вас есть GPU на сервере
- Размер образа не критичен
- Сборка выполняется редко

## Рекомендация

Для большинства случаев (особенно если нет GPU) используйте CPU-only версию - это уменьшит размер образа в 4-5 раз.

