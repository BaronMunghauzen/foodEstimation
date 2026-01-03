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
from typing import Dict, Optional
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

Ответь строго в JSON формате:
{
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

Важно:
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
            
            result = json.loads(content)
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

# Глобальный экземпляр
chatgpt_service = ChatGPTNutritionService()

