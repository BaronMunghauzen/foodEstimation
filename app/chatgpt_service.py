"""
Сервис для работы с ChatGPT Vision API
"""
import openai
from PIL import Image
import base64
import io
import os
import json
import logging
import time
from typing import Dict, Optional, List
from dotenv import load_dotenv
import httpx

load_dotenv()

logger = logging.getLogger(__name__)

class ChatGPTNutritionService:
    """Сервис для распознавания еды через ChatGPT Vision API"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY не найден в .env - ChatGPT отключен")
            self.enabled = False
            self.proxy_url = None
        else:
            try:
                # Настройка прокси если указан (только для OpenAI, не глобально)
                proxy_url = os.getenv("OPENAI_PROXY")  # Используем только OPENAI_PROXY, не глобальные HTTPS_PROXY/HTTP_PROXY
                self.proxy_url = proxy_url
                http_client = None
                
                if proxy_url:
                    # Определяем тип прокси по схеме
                    proxy_type = "HTTP"
                    is_socks = False
                    if proxy_url.startswith("socks5://") or proxy_url.startswith("socks4://"):
                        proxy_type = "SOCKS5" if "socks5" in proxy_url else "SOCKS4"
                        is_socks = True
                    elif proxy_url.startswith("http://") or proxy_url.startswith("https://"):
                        proxy_type = "HTTP"
                    else:
                        # Если схема не указана, предполагаем HTTP
                        if not proxy_url.startswith(("http://", "https://", "socks5://", "socks4://")):
                            proxy_url = f"http://{proxy_url}"
                            proxy_type = "HTTP"
                    
                    # Маскируем пароль в логах
                    log_proxy = proxy_url
                    if "@" in proxy_url:
                        # Маскируем user:password@host:port -> user:***@host:port
                        parts = proxy_url.split("@")
                        if len(parts) == 2:
                            auth_part = parts[0]
                            if "://" in auth_part:
                                scheme = auth_part.split("://")[0] + "://"
                                auth = auth_part.split("://")[1]
                                if ":" in auth:
                                    user = auth.split(":")[0]
                                    log_proxy = f"{scheme}{user}:***@{parts[1]}"
                    
                    logger.info(f"Использование {proxy_type} прокси для OpenAI: {log_proxy.split('@')[-1] if '@' in log_proxy else log_proxy}")
                    
                    # Создаем отдельный HTTP клиент с прокси только для OpenAI
                    # НЕ устанавливаем глобальные переменные окружения, чтобы не влиять на другие библиотеки (HuggingFace и т.д.)
                    if is_socks:
                        # Для SOCKS прокси используем httpx-socks (если установлен) для лучшей поддержки аутентификации
                        try:
                            from httpx_socks import AsyncProxyTransport
                            transport = AsyncProxyTransport.from_url(proxy_url)
                            http_client = httpx.AsyncClient(
                                transport=transport,
                                timeout=httpx.Timeout(60.0)
                            )
                            logger.debug(f"SOCKS клиент создан через httpx-socks")
                        except ImportError:
                            # Если httpx-socks не установлен, пробуем стандартный способ через httpx[socks]
                            try:
                                # Проверяем, установлен ли socksio (нужен для httpx[socks])
                                import socksio
                                http_client = httpx.AsyncClient(
                                    proxy=proxy_url,
                                    timeout=httpx.Timeout(60.0)
                                )
                                logger.warning("httpx-socks не установлен, используется httpx[socks]. Для SOCKS5 с аутентификацией рекомендуется: pip install httpx-socks")
                                logger.debug(f"SOCKS клиент создан через httpx[socks]")
                            except ImportError:
                                # Если ни httpx-socks, ни httpx[socks] не установлены
                                error_msg = (
                                    "Для использования SOCKS5 прокси необходимо установить один из пакетов:\n"
                                    "  - pip install httpx-socks  (рекомендуется, поддерживает аутентификацию)\n"
                                    "  - pip install httpx[socks]  (базовая поддержка SOCKS5)"
                                )
                                logger.error(error_msg)
                                raise ImportError(error_msg)
                    else:
                        # Для HTTP прокси используем стандартный способ
                        http_client = httpx.AsyncClient(
                            proxy=proxy_url,
                            timeout=httpx.Timeout(60.0)
                        )
                        logger.debug(f"HTTP клиент создан с прокси: {proxy_type}")
                else:
                    logger.info("Прокси не указан, используется прямое подключение к OpenAI")
                
                self.client = openai.AsyncOpenAI(
                    api_key=api_key,
                    http_client=http_client
                )
                self.enabled = True
                logger.info("ChatGPT сервис инициализирован" + (f" (через {proxy_type} прокси)" if proxy_url else " (прямое подключение)"))
            except Exception as e:
                logger.error(f"Ошибка инициализации ChatGPT: {e}", exc_info=True)
                self.enabled = False
                self.proxy_url = None
    
    async def recognize_food(self, image: Image.Image, user_comment: Optional[str] = None) -> Optional[Dict]:
        """
        Распознает еду и оценивает калории через ChatGPT Vision API
        
        Args:
            image: PIL Image объекта
            user_comment: Опциональный комментарий пользователя (до 100 слов)
            
        Returns:
            dict с результатами или None если ошибка/отключен
        """
        if not self.enabled:
            logger.debug("ChatGPT отключен, пропуск запроса")
            return None
        
        try:
            logger.info("Начало обработки изображения для ChatGPT")
            
            # Конвертируем изображение в base64
            logger.debug("Конвертация изображения в base64...")
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            image_size = len(buffered.getvalue())
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.debug(f"Изображение конвертировано: размер {image_size} байт, base64 длина {len(image_base64)} символов")
            
            # Подготовка промпта
            prompt = """Распознай это блюдо на фотографии и определи примерную калорийность и БЖУ (белки, жиры, углеводы) на 100 грамм.

Также оцени примерный вес порции на фотографии в граммах, основываясь на размере блюда, типе посуды и визуальной оценке.

КРИТИЧЕСКИ ВАЖНО: Ты ОБЯЗАН вернуть ответ СТРОГО в JSON формате. Даже если на фотографии нет еды, изображение нечеткое, или ты не можешь распознать блюдо - верни JSON с полем "error".

Если на фотографии ЕСТЬ еда, верни JSON в формате:
{
    "error": false,
    "food_name": "название блюда",
    "calories_per_100g": число,
    "proteins_per_100g": число,
    "fats_per_100g": число,
    "carbs_per_100g": число,
    "confidence": число от 0 до 1,
    "estimated_weight_g": число (примерный вес порции на фотографии в граммах),
    "estimated_volume_ml": число (примерный объем порции в миллилитрах),
    "ingredients": [
        {
            "name": "название ингредиента",
            "calories_per_100g": число,
            "proteins_per_100g": число,
            "fats_per_100g": число,
            "carbs_per_100g": число,
            "description": "краткое описание ингредиента",
            "weight_in_portion_g": число (примерный вес этого ингредиента в порции в граммах)
        }
    ],
    "recommendations": [
        {
            "type": "tip" или "alternative",
            "title": "краткое название совета/альтернативы",
            "description": "подробное описание",
            "calories_saved": число (только для альтернатив, может быть null)
        }
    ],
    "micronutrients": [
        {
            "name": "название витамина/минерала (например: Витамин C, Кальций, Железо)",
            "amount": число (количество в блюде на 100г),
            "unit": "единица измерения (мг, мкг, г)",
            "daily_value": число (суточная норма для взрослого человека)
        }
    ]
}

Если на фотографии НЕТ еды (например: предметы, люди, пейзажи, животные и т.д.), верни JSON в формате:
{
    "error": true,
    "error_type": "not_food",
    "error_message": "На фотографии нет еды"
}

Если изображение нечеткое, размытое или невозможно распознать блюдо, верни JSON в формате:
{
    "error": true,
    "error_type": "unclear_image",
    "error_message": "Не удалось распознать блюдо на фотографии"
}

Если не можешь определить блюдо по другой причине, верни JSON в формате:
{
    "error": true,
    "error_type": "cannot_recognize",
    "error_message": "Не удалось распознать блюдо"
}

Важно:
- ВСЕГДА возвращай JSON, даже при ошибках
- error_type может быть: "not_food", "unclear_image", "cannot_recognize"
- error_message должен быть кратким (до 50 символов)
- estimated_weight_g: оцени вес порции на фотографии на основе визуального анализа (размер блюда, тип посуды)
- estimated_volume_ml: оцени объем порции в миллилитрах
- ingredients: массив основных ингредиентов блюда с их КБЖУ на 100г и примерным весом в порции (weight_in_portion_g)
- weight_in_portion_g: сумма всех weight_in_portion_g должна приблизительно равняться estimated_weight_g
- recommendations: советы по улучшению ("tip") или альтернативы ("alternative")
- micronutrients: основные витамины и минералы, которые есть в блюде (amount указан на 100г)
- daily_value для micronutrients указывай в тех же единицах, что и amount"""

            # Добавляем комментарий пользователя к промпту если есть
            if user_comment:
                prompt += f"\n\nДополнительная информация от пользователя: {user_comment}\nВнимательно учти эту информацию при анализе блюда. Если в комментарии указан вес порции (например, '300 грамм', 'вес 250г'), обязательно используй это значение для estimated_weight_g. Если вес не указан напрямую, но есть другая информация, учитывай её при оценке веса порции."
                logger.debug(f"Добавлен комментарий пользователя: {user_comment[:100]}...")
            
            # Подготовка запроса
            model = "gpt-4o"
            logger.info(f"Отправка запроса к ChatGPT API (модель: {model})")
            if self.proxy_url:
                proxy_log = self.proxy_url.split("@")[-1] if "@" in self.proxy_url else self.proxy_url
                logger.debug(f"Используется прокси: {proxy_log}")
            
            # Запрос к ChatGPT Vision API
            request_start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "low"  # low, high, auto
                            }
                        }
                    ]
                }],
                max_tokens=2000,  # Увеличено для дополнительной информации
                temperature=0.3  # Низкая температура для более точных ответов
            )
            request_duration = time.time() - request_start_time
            
            logger.info(f"Получен ответ от ChatGPT за {request_duration:.2f} секунд")
            logger.debug(f"Response ID: {response.id}, Model: {response.model}, Usage: {response.usage}")
            
            # Парсим ответ
            content = response.choices[0].message.content.strip()
            logger.debug(f"Длина ответа: {len(content)} символов")
            logger.debug(f"Первые 200 символов ответа: {content[:200]}")
            
            # Убираем markdown код блоки если есть
            original_content = content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                logger.debug("Удалены markdown блоки ```json")
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                logger.debug("Удалены markdown блоки ```")
            
            # Пытаемся распарсить JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Если не удалось распарсить, создаем структурированную ошибку
                logger.warning(f"Ответ ChatGPT не является валидным JSON. Создаю структурированную ошибку.")
                # Пытаемся определить тип ошибки по тексту ответа
                content_lower = original_content.lower()
                if "can't assist" in content_lower or "cannot assist" in content_lower or "sorry" in content_lower:
                    error_type = "not_food"
                    error_message = "На фотографии нет еды"
                elif "unclear" in content_lower or "размыт" in content_lower or "нечетк" in content_lower:
                    error_type = "unclear_image"
                    error_message = "Не удалось распознать блюдо на фотографии"
                else:
                    error_type = "cannot_recognize"
                    # Берем первые 50 символов из ответа как сообщение об ошибке
                    error_message = original_content[:50].strip()
                    if not error_message:
                        error_message = "Не удалось распознать блюдо"
                
                result = {
                    "error": True,
                    "error_type": error_type,
                    "error_message": error_message
                }
                logger.info(f"ChatGPT вернул ошибку: {error_type} - {error_message}")
                return result
            
            # Проверяем наличие ошибки в JSON
            if result.get("error") is True:
                error_type = result.get("error_type", "cannot_recognize")
                error_message = result.get("error_message", "Не удалось распознать блюдо")
                logger.info(f"ChatGPT вернул ошибку в JSON: {error_type} - {error_message}")
                return result
            
            # Успешное распознавание
            logger.info(f"ChatGPT распознал: {result.get('food_name')} (калории: {result.get('calories_per_100g')}, уверенность: {result.get('confidence', 0):.2f})")
            logger.debug(f"Полный результат: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа ChatGPT: {e}")
            logger.error(f"Позиция ошибки: строка {e.lineno}, колонка {e.colno}")
            logger.error(f"Ответ был (первые 500 символов): {content[:500] if 'content' in locals() else 'N/A'}")
            if 'original_content' in locals():
                logger.error(f"Оригинальный ответ (первые 500 символов): {original_content[:500]}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"Ошибка подключения к ChatGPT API: {e}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            if hasattr(e, 'message'):
                logger.error(f"Сообщение: {e.message}")
            if self.proxy_url:
                logger.error(f"Проверьте настройки прокси: {self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url}")
            logger.debug("Детали ошибки подключения:", exc_info=True)
            return None
        except openai.APIError as e:
            logger.error(f"Ошибка API ChatGPT: {e}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP статус: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Ответ сервера: {e.response}")
            logger.debug("Детали ошибки API:", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе к ChatGPT: {e}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            logger.debug("Полный traceback:", exc_info=True)
            return None
    
    async def generate_meal_plan(
        self,
        meals_per_day: int,
        days_count: int,
        target_calories: float,
        target_proteins: float,
        target_fats: float,
        target_carbs: float,
        allowed_recipes: List[Dict]
    ) -> Optional[Dict]:
        """
        Генерирует программу питания через ChatGPT
        
        Args:
            meals_per_day: Количество приемов пищи в день (минимум 3)
            days_count: Количество дней
            target_calories: Целевые калории
            target_proteins: Целевые белки (граммы)
            target_fats: Целевые жиры (граммы)
            target_carbs: Целевые углеводы (граммы)
            allowed_recipes: Список доступных рецептов
            
        Returns:
            dict с программой питания или None если ошибка/отключен
        """
        if not self.enabled:
            logger.debug("ChatGPT отключен, пропуск запроса")
            return None
        
        try:
            logger.info(f"Начало генерации программы питания: {days_count} дней, {meals_per_day} приемов пищи в день")
            
            # Формируем список рецептов для промпта
            recipes_text = "\n".join([
                f"- UUID: {r['uuid']}, Название: {r['name']}, Категория: {r['category']}, "
                f"КБЖУ на 1 порцию: {r['calories']} ккал, {r['proteins']}г белков, {r['fats']}г жиров, {r['carbs']}г углеводов"
                for r in allowed_recipes
            ])
            
            # Рассчитываем примерное количество порций для справки и создаем детальный пример расчета
            if allowed_recipes:
                avg_calories_per_portion = sum(r['calories'] for r in allowed_recipes) / len(allowed_recipes)
                estimated_portions = max(1, int(target_calories / avg_calories_per_portion)) if avg_calories_per_portion > 0 else 1
                
                # Создаем конкретный пример расчета для первых двух рецептов
                example_calc = ""
                if len(allowed_recipes) >= 2:
                    r1 = allowed_recipes[0]
                    r2 = allowed_recipes[1]
                    max_portions_r1 = int(target_calories * 0.6 / r1['calories']) if r1['calories'] > 0 else 0
                    max_portions_r2 = int(target_calories * 0.4 / r2['calories']) if r2['calories'] > 0 else 0
                    example_calc = f"\n\nКОНКРЕТНЫЙ ПРИМЕР РАСЧЕТА:\n"
                    example_calc += f"Если использовать '{r1['name']}' ({r1['calories']} ккал/порция) и '{r2['name']}' ({r2['calories']} ккал/порция):\n"
                    example_calc += f"- Для достижения {target_calories} ккал можно взять примерно {max_portions_r1} порций '{r1['name']}' ({max_portions_r1 * r1['calories']} ккал) и {max_portions_r2} порций '{r2['name']}' ({max_portions_r2 * r2['calories']} ккал)\n"
                    example_calc += f"- Итого: {max_portions_r1 * r1['calories'] + max_portions_r2 * r2['calories']} ккал (целевые: {target_calories} ккал)\n"
                    example_calc += f"- НЕ превышай {int(target_calories * 1.1)} ккал (максимум +10%)"
                
                portions_hint = f"\n\nПРИМЕРНЫЙ РАСЧЕТ: При средних {avg_calories_per_portion:.1f} ккал на порцию, для достижения {target_calories} ккал нужно примерно {estimated_portions} порций в день (это ориентир, распределяй по приемам пищи).{example_calc}"
            else:
                portions_hint = ""
            
            # Подготовка промпта
            prompt = f"""Ты профессиональный диетолог и опытный спортсмен с большим опытом составления программ питания.

Составь программу питания на {days_count} дней с {meals_per_day} приемами пищи в день.

ЦЕЛЕВЫЕ КБЖУ НА ДЕНЬ:
- Калории: {target_calories} ккал
- Белки: {target_proteins} г
- Жиры: {target_fats} г
- Углеводы: {target_carbs} г

ДОСТУПНЫЕ РЕЦЕПТЫ (используй ТОЛЬКО эти рецепты, никакие другие):
{recipes_text}{portions_hint}

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО рецепты из списка выше. Запрещено использовать какие-либо другие рецепты.
2. Строго соблюдай количество приемов пищи в день: {meals_per_day} (минимум 3).
3. Обязательно должны быть включены: завтрак, обед, ужин. Остальные приемы пищи опциональны.
4. КРИТИЧЕСКИ ВАЖНО - СТРОГИЙ РАСЧЕТ ПОРЦИЙ (ОБЯЗАТЕЛЬНО СЛЕДУЙ ЭТОМУ АЛГОРИТМУ):
   ШАГ 1: Для каждого дня рассчитай общее количество порций каждого блюда
   ШАГ 2: Используй формулу: сумма(калории_блюда * количество_порций_этого_блюда_за_день) ≤ {target_calories} * 1.05 (МАКСИМУМ +5%)
   ШАГ 3: Пример расчета:
     - Если блюдо "Яичница" имеет 5 ккал на порцию, и ты используешь его 4 раза в день по 4 порции = 16 порций всего
     - Калории от "Яичницы": 5 * 16 = 80 ккал
     - Если блюдо "Котлеты" имеет 5 ккал на порцию, и ты используешь его 2 раза в день по 8 порций = 16 порций всего
     - Калории от "Котлет": 5 * 16 = 80 ккал
     - ИТОГО за день: 80 + 80 = 160 ккал (ПРЕВЫШЕНИЕ! Целевые: {target_calories} ккал)
   ШАГ 4: ПРАВИЛЬНЫЙ расчет для целевых {target_calories} ккал:
     - Максимум порций "Яичницы" (5 ккал): {int(target_calories * 0.5 / 5) if target_calories > 0 else 0} порций = {int(target_calories * 0.5 / 5) * 5 if target_calories > 0 else 0} ккал
     - Максимум порций "Котлет" (5 ккал): {int(target_calories * 0.5 / 5) if target_calories > 0 else 0} порций = {int(target_calories * 0.5 / 5) * 5 if target_calories > 0 else 0} ккал
     - ИТОГО: {int(target_calories * 0.5 / 5) * 5 * 2 if target_calories > 0 else 0} ккал (в пределах целевых {target_calories} ккал)
   ШАГ 5: ВСЕГДА проверяй итоговую сумму перед отправкой! Сумма НЕ должна превышать {int(target_calories * 1.05)} ккал (МАКСИМУМ +5%)
5. Распределяй КБЖУ по приемам пищи согласно целевым значениям.
6. СТРОГИЙ КОНТРОЛЬ КБЖУ: Подбирай блюда с учетом целевых КБЖУ с допуском ±5% (НЕ превышай более чем на 5%).
   - Целевые калории: {target_calories} ккал, МАКСИМУМ: {int(target_calories * 1.05)} ккал
   - Целевые белки: {target_proteins} г, МАКСИМУМ: {int(target_proteins * 1.05)} г
   - Целевые жиры: {target_fats} г, МАКСИМУМ: {int(target_fats * 1.05)} г
   - Целевые углеводы: {target_carbs} г, МАКСИМУМ: {int(target_carbs * 1.05)} г
7. Для анализа КБЖУ используй СТРОГО данные из списка рецептов выше. Запрещено самостоятельно проверять или изменять КБЖУ.
8. Учитывай категории еды: обед не клади в завтрак, но ужин может быть выбран и на обед. Салаты можно добавлять куда угодно.
9. ЭКОНОМИЯ ГОТОВКИ - ПОВТОРЯЙ БЛЮДА НА СЛЕДУЮЩИЙ ДЕНЬ:
   - Желательно повторять блюда, приготовленные на ужин, на обед следующего дня
   - Старайся использовать одни и те же блюда 2 дня подряд для экономии времени на готовку
   - Например: если на ужин первого дня "Запечённая курица", то на обед второго дня можно использовать то же блюдо
   - Это особенно важно для основных блюд (мясо, рыба, птица)
10. Обеспечь разнообразие блюд, но приоритет - соответствие КБЖУ и экономия готовки.
11. НЕ обязательно использовать ВСЕ рецепты - главное соответствие КБЖУ за день.
10. ОБЯЗАТЕЛЬНО ОБЪЕДИНЯЙ ГАРНИРЫ С ОСНОВНЫМИ БЛЮДАМИ:
    - Гарниры (пюре, рис, гречка, макароны, картофель и т.д.) НЕ должны быть отдельным приемом пищи по возможности (при наличии этих блюд)
    - Гарниры должны быть объединены с основными блюдами (котлеты, курица, рыба, мясо и т.д.) в ОДИН прием пищи по возможности (при наличии основных блюд)
    - Если в списке есть гарнир и основное блюдо одной категории (например, оба "Обед"), они ДОЛЖНЫ быть в одном приеме пищи
11. ИСПОЛЬЗУЙ КОЛИЧЕСТВО ПОРЦИЙ ВМЕСТО ДУБЛИРОВАНИЯ:
    - Если нужно больше КБЖУ от одного блюда, используй поле "portions" (количество порций)
    - Например: {{"uuid": "пюре-uuid", "name": "Пюре", "portions": 2}} означает 2 порции пюре
    - НЕ дублируй одно и то же блюдо несколько раз в массиве - используй поле "portions"
    - Если нужно больше КБЖУ, предпочтительно комбинируй РАЗНЫЕ блюда, но можно и увеличить порции одного блюда
12. Желательно учитывать, что блюдо, приготовленное на вечер, может быть съедено и на обед следующего дня (экономия готовки).

Ответь строго в JSON формате:
{{
    "days": [
        {{
            "day_number": 1,
            "meals": [
                {{
                    "category": "завтрак",
                    "meals": [
                        {{"uuid": "uuid-рецепта", "name": "название рецепта", "portions": 1}},
                        {{"uuid": "uuid-другого-рецепта", "name": "название другого рецепта", "portions": 2}}
                    ]
                }},
                {{
                    "category": "обед",
                    "meals": [
                        {{"uuid": "uuid-рецепта", "name": "название рецепта", "portions": 1}}
                    ]
                }},
                {{
                    "category": "ужин",
                    "meals": [
                        {{"uuid": "uuid-рецепта", "name": "название рецепта", "portions": 1}}
                    ]
                }}
            ]
        }},
        {{
            "day_number": 2,
            "meals": [...]
        }}
    ]
}}

Важно:
- Для каждого дня укажи все приемы пищи ({meals_per_day} штук)
- Категории: "завтрак", "обед", "ужин", "перекус", "полдник" и т.д.
- В каждом приеме пищи может быть одно или несколько блюд (объединение)
- Если нужно больше КБЖУ от одного блюда, используй поле "portions" (количество порций) вместо дублирования блюда
- Например: {{"uuid": "пюре-uuid", "name": "Пюре", "portions": 2}} означает 2 порции пюре
- Гарниры желательно объединяй с основными блюдами в один прием пищи
- UUID и название должны точно соответствовать данным из списка рецептов
- КРИТИЧЕСКИ ВАЖНО: Сумма КБЖУ всех блюд за день (с учетом количества порций) НЕ ДОЛЖНА превышать {int(target_calories * 1.05)} ккал (максимум +5% от {target_calories} ккал)
- Формула расчета: сумма(калории_блюда * portions_в_каждом_приеме_пищи) должна быть ≤ {int(target_calories * 1.05)} ккал
- Аналогично для белков: ≤ {int(target_proteins * 1.05)} г, жиров: ≤ {int(target_fats * 1.05)} г, углеводов: ≤ {int(target_carbs * 1.05)} г
- ВАЖНО: Если блюдо используется в нескольких приемах пищи, суммируй все порции этого блюда за день!
  Пример: если "Яичница" (5 ккал) используется в завтраке (4 порции), перекусе (4 порции), полднике (4 порции) = 12 порций всего за день = 12 * 5 = 60 ккал
- ВСЕГДА перед отправкой ответа:
  1. Подсчитай общее количество порций каждого блюда за весь день (суммируй по всем приемам пищи)
  2. Умножь на калории одной порции
  3. Сложи все блюда
  4. Проверь: сумма ≤ {int(target_calories * 1.05)} ккал (МАКСИМУМ +5%)
  5. Если превышает - УМЕНЬШИ количество порций!
- Пример правильного объединения: [{{"uuid": "котлеты-uuid", "name": "Куриные котлетки", "portions": 1}}, {{"uuid": "пюре-uuid", "name": "Пюре", "portions": 1}}] - это ОДИН прием пищи с двумя разными блюдами
- Если нужно больше калорий от пюре: {{"uuid": "пюре-uuid", "name": "Пюре", "portions": 2}} - две порции пюре"""
            
            # Запрос к ChatGPT API
            model = "gpt-4o"
            logger.info(f"Отправка запроса к ChatGPT API для генерации программы питания (модель: {model})")
            
            request_start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=4000,  # Больше токенов для детальной программы
                temperature=0.3  # Низкая температура для более точного следования инструкциям
            )
            request_duration = time.time() - request_start_time
            
            logger.info(f"Получен ответ от ChatGPT за {request_duration:.2f} секунд")
            logger.debug(f"Response ID: {response.id}, Model: {response.model}, Usage: {response.usage}")
            
            # Парсим ответ
            content = response.choices[0].message.content.strip()
            logger.debug(f"Длина ответа: {len(content)} символов")
            logger.debug(f"Первые 500 символов ответа: {content[:500]}")
            
            # Убираем markdown код блоки если есть
            original_content = content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                logger.debug("Удалены markdown блоки ```json")
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                logger.debug("Удалены markdown блоки ```")
            
            result = json.loads(content)
            logger.info(f"ChatGPT сгенерировал программу питания на {len(result.get('days', []))} дней")
            logger.debug(f"Полный результат: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON ответа ChatGPT: {e}")
            logger.error(f"Позиция ошибки: строка {e.lineno}, колонка {e.colno}")
            logger.error(f"Ответ был (первые 500 символов): {content[:500] if 'content' in locals() else 'N/A'}")
            if 'original_content' in locals():
                logger.error(f"Оригинальный ответ (первые 500 символов): {original_content[:500]}")
            return None
        except openai.APIConnectionError as e:
            logger.error(f"Ошибка подключения к ChatGPT API: {e}")
            if self.proxy_url:
                logger.error(f"Проверьте настройки прокси: {self.proxy_url.split('@')[-1] if '@' in self.proxy_url else self.proxy_url}")
            logger.debug("Детали ошибки подключения:", exc_info=True)
            return None
        except openai.APIError as e:
            logger.error(f"Ошибка API ChatGPT: {e}")
            logger.debug("Детали ошибки API:", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе к ChatGPT: {e}")
            logger.error(f"Тип ошибки: {type(e).__name__}")
            logger.debug("Полный traceback:", exc_info=True)
            return None

# Глобальный экземпляр
chatgpt_service = ChatGPTNutritionService()

