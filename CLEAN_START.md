# Чистый старт с Alembic

Если вы запускали Docker до установки Alembic и хотите начать с чистой базы данных, выполните следующие шаги.

## Шаг 1: Остановить и удалить контейнеры

```bash
# Остановить контейнеры
docker-compose down

# Удалить контейнеры и volumes (включая данные БД)
docker-compose down -v
```

Флаг `-v` удаляет volumes, включая данные PostgreSQL.

## Шаг 2: (Опционально) Удалить старый образ

```bash
# Посмотреть образы
docker images

# Удалить образ (замените IMAGE_ID на ID вашего образа)
docker rmi <IMAGE_ID>

# Или удалить все неиспользуемые образы
docker image prune -a
```

## Шаг 3: Пересобрать и запустить

```bash
# Пересобрать образ с новыми зависимостями (включая Alembic)
docker-compose up --build -d
```

## Шаг 4: Скопировать файлы Alembic в контейнер (если нужно)

Если вы работаете локально и создали миграции на хосте:

```bash
# Убедитесь, что alembic.ini и папка alembic/ скопированы в контейнер
# Проверьте, что они есть в проекте
```

Или добавьте в `docker-compose.yml` volume для alembic:

```yaml
volumes:
  - ./app:/app/app
  - ./alembic:/app/alembic
  - ./alembic.ini:/app/alembic.ini
```

## Шаг 5: Создать первую миграцию

```bash
# Зайти в контейнер
docker-compose exec api bash

# Создать миграцию
alembic revision --autogenerate -m "Initial migration"

# Выйти из контейнера
exit
```

Или выполнить команду напрямую:

```bash
docker-compose exec api alembic revision --autogenerate -m "Initial migration"
```

## Шаг 6: Применить миграции

```bash
docker-compose exec api alembic upgrade head
```

## Шаг 7: Проверить

```bash
# Проверить текущую версию БД
docker-compose exec api alembic current

# Проверить что приложение работает
curl http://localhost:8000/health
```

## Альтернатива: Работа локально

Если вы работаете локально (без Docker):

```bash
# Убедитесь что PostgreSQL запущена и база данных пустая
# Или удалите старую базу и создайте новую:

# В psql:
DROP DATABASE foodestimation;
CREATE DATABASE foodestimation;

# Затем создайте миграцию
alembic revision --autogenerate -m "Initial migration"

# Примените миграции
alembic upgrade head
```

## Важно

- Флаг `-v` в `docker-compose down -v` **удалит все данные** из БД
- Убедитесь, что у вас нет важных данных перед удалением
- После удаления volumes вам нужно будет пересоздать все данные

