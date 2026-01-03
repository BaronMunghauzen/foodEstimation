# Миграции базы данных

Проект использует Alembic для управления миграциями базы данных.

## Первая настройка

После установки зависимостей и настройки `.env`:

```bash
# Создайте первую миграцию (если еще не создана)
alembic revision --autogenerate -m "Initial migration"

# Примените миграции
alembic upgrade head
```

## Основные команды

### Создание новой миграции

```bash
# Автоматическое создание миграции на основе изменений моделей
alembic revision --autogenerate -m "Описание изменений"

# Создание пустой миграции (для ручных изменений)
alembic revision -m "Описание изменений"
```

### Применение миграций

```bash
# Применить все миграции до последней
alembic upgrade head

# Применить конкретную миграцию
alembic upgrade <revision>

# Применить следующую миграцию
alembic upgrade +1
```

### Откат миграций

```bash
# Откатить на одну миграцию назад
alembic downgrade -1

# Откатить до конкретной миграции
alembic downgrade <revision>

# Откатить все миграции
alembic downgrade base
```

### Просмотр статуса

```bash
# Текущая версия БД
alembic current

# История миграций
alembic history

# Показать SQL без применения (dry-run)
alembic upgrade head --sql
```

## Работа с Docker

В Docker контейнере миграции можно применить при старте:

```bash
docker-compose exec api alembic upgrade head
```

Или добавить в docker-compose.yml:

```yaml
api:
  command: sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"
```

## Важно

- **Всегда проверяйте** автогенерированные миграции перед применением
- **Делайте бэкап БД** перед применением миграций в продакшене
- **Используйте git** для версионирования файлов миграций
- Миграции находятся в `alembic/versions/`

