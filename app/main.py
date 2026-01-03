"""
Основное FastAPI приложение для распознавания еды и расчета калорий.
"""

from fastapi import FastAPI, File, UploadFile, Depends, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from PIL import Image
import io
import os
import time
from typing import List, Optional

from app.database import SessionLocal, engine, Base
from app.models import FoodRequest
from app.schemas import FoodRecognitionResponse, FoodItem, NutritionInfo, PortionEstimate, IngredientInfo, Recommendation, MicronutrientInfo
from app.auth import verify_token
from app.ml_service import init_food_service
from app.portion_estimator import PortionEstimator
from app.image_storage import image_storage
from app.chatgpt_service import chatgpt_service
import logging
import os
from dotenv import load_dotenv
import re
import json

load_dotenv()

logger = logging.getLogger(__name__)

def parse_weight_from_comment(comment: str) -> Optional[float]:
    """
    Парсит вес из комментария пользователя.
    Ищет паттерны типа "300 гр", "300 грамм", "300g", "300 г" и т.д.
    
    Args:
        comment: Комментарий пользователя
        
    Returns:
        Вес в граммах или None если не найден
    """
    if not comment:
        return None
    
    # Паттерны для поиска веса
    patterns = [
        r'(\d+)\s*(?:гр|грамм|граммов|g|г)\b',
        r'вес[а]?\s*(?:блюда|порции)?\s*:?\s*(\d+)\s*(?:гр|грамм|граммов|g|г)?',
        r'(\d+)\s*(?:гр|грамм|граммов|g|г)\s*(?:блюда|порции)?',
    ]
    
    comment_lower = comment.lower()
    for pattern in patterns:
        match = re.search(pattern, comment_lower, re.IGNORECASE)
        if match:
            weight = float(match.group(1))
            if 10 <= weight <= 10000:  # Разумные пределы
                logger.info(f"Извлечен вес из комментария: {weight} г")
                return weight
    
    return None

# Параметры из .env
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"  # Полностью отключить локальную модель
USE_TRAINED_MODEL = os.getenv("USE_TRAINED_MODEL", "false").lower() == "true"
USE_OWN_MODEL_INSTEAD_CHATGPT = os.getenv("USE_OWN_MODEL_INSTEAD_CHATGPT", "false").lower() == "true"
USE_CHATGPT = os.getenv("USE_CHATGPT", "true").lower() == "true"

# Инициализируем сервис только если локальная модель включена
food_service = None
if USE_LOCAL_MODEL:
    food_service = init_food_service(use_trained_model=USE_TRAINED_MODEL)
    logger.info("Локальная модель инициализирована")
else:
    logger.info("Локальная модель отключена (USE_LOCAL_MODEL=false), используется только ChatGPT")

# Таблицы создаются через миграции Alembic
# Для инициализации: alembic upgrade head

app = FastAPI(
    title="Food Calories Estimation API",
    description="API для распознавания еды и расчета калорий/БЖУ по фотографии",
    version="1.0.0"
)

# Инициализируем оценщик порций
portion_estimator = PortionEstimator()

def get_db():
    """Dependency для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Food Calories Estimation API",
        "version": "1.0.0",
        "endpoints": {
            "recognize": "/recognize-food",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "service": "food-estimation-api"
    }

@app.post("/recognize-food", response_model=FoodRecognitionResponse)
async def recognize_food(
    file: UploadFile = File(...),
    comment: Optional[str] = Form(None),
    token: str = Depends(verify_token),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для распознавания еды и расчета калорий/БЖУ по фотографии.
    
    Требует токен аутентификации в заголовке:
    Authorization: Bearer <your-token>
    
    Args:
        file: Загруженное изображение
        comment: Опциональный комментарий пользователя (до 100 слов)
        token: Токен аутентификации (из dependency)
        request: FastAPI Request объект
        db: Сессия базы данных
        
    Returns:
        FoodRecognitionResponse с информацией о распознанной еде и калориях/БЖУ
    """
    start_time = time.time()
    
    try:
        # Валидация комментария (до 100 слов)
        user_comment = None
        if comment:
            words = comment.strip().split()
            if len(words) > 100:
                raise HTTPException(
                    status_code=400,
                    detail="Комментарий не должен превышать 100 слов"
                )
            user_comment = comment.strip()
            logger.info(f"Получен комментарий пользователя ({len(words)} слов): {user_comment[:100]}...")
        
        # Проверяем тип файла
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением"
            )
        
        # Читаем изображение
        contents = await file.read()
        
        # Проверяем размер файла (макс 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Размер файла не должен превышать 10MB"
            )
        
        # Открываем изображение
        try:
            image = Image.open(io.BytesIO(contents))
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Не удалось открыть изображение: {str(e)}"
            )
        
        # Сохраняем изображение на диск
        try:
            image_path = image_storage.save_image(image)
        except Exception as e:
            logger.warning(f"Ошибка сохранения изображения: {e}")
            image_path = None
        
        # Шаг 1: Запрашиваем ChatGPT (если включен и не используем свою модель вместо него)
        chatgpt_result = None
        use_chatgpt_for_response = False
        
        if USE_CHATGPT and not USE_OWN_MODEL_INSTEAD_CHATGPT:
            if not chatgpt_service.enabled:
                logger.warning("ChatGPT сервис отключен (проверьте OPENAI_API_KEY в .env)")
            else:
                try:
                    logger.info("Вызов ChatGPT API...")
                    chatgpt_response = await chatgpt_service.recognize_food(image, user_comment=user_comment)
                    if chatgpt_response:
                        chatgpt_result = chatgpt_response
                        use_chatgpt_for_response = True
                        logger.info(f"ChatGPT успешно обработал запрос: {chatgpt_result.get('food_name')}")
                    else:
                        logger.warning("ChatGPT вернул None (сервис отключен или произошла ошибка)")
                except Exception as e:
                    logger.warning(f"Ошибка ChatGPT (продолжаем без него): {e}", exc_info=True)
                    chatgpt_result = None
        else:
            if USE_OWN_MODEL_INSTEAD_CHATGPT:
                logger.debug("ChatGPT пропущен: USE_OWN_MODEL_INSTEAD_CHATGPT=true")
            elif not USE_CHATGPT:
                logger.debug("ChatGPT пропущен: USE_CHATGPT=false")
        
        # Шаг 2: Формируем информацию о порции на основе данных ChatGPT или локальной модели
        portion_info = None
        if use_chatgpt_for_response and chatgpt_result:
            # Используем данные от ChatGPT (ChatGPT сам учитывает комментарий при оценке веса)
            chatgpt_weight_g = chatgpt_result.get('estimated_weight_g')
            chatgpt_volume_ml = chatgpt_result.get('estimated_volume_ml')
            
            if not chatgpt_weight_g:
                logger.warning("ChatGPT не вернул вес порции, используем fallback оценку")
                portion_info_fallback = portion_estimator.estimate_portion_size(image)
                weight_g = portion_info_fallback['weight_g']
            else:
                weight_g = chatgpt_weight_g
                logger.info(f"Используется вес от ChatGPT: {weight_g} г (учтен комментарий пользователя)")
            
            # Объем от ChatGPT или рассчитываем
            if chatgpt_volume_ml:
                volume_ml = chatgpt_volume_ml
            else:
                volume_ml = weight_g * 1.25  # Примерная оценка
            
            # Определяем размер порции
            if weight_g < 150:
                portion_size = "small"
            elif weight_g < 300:
                portion_size = "medium"
            else:
                portion_size = "large"
            
            portion_info = {
                'weight_g': weight_g,
                'volume_ml': volume_ml,
                'portion_size': portion_size
            }
        elif USE_LOCAL_MODEL and food_service:
            # Используем PortionEstimator для локальной модели
            portion_info = portion_estimator.estimate_portion_size(image)
            weight_g = portion_info['weight_g']
        
        # Шаг 3: Рассчитываем total_nutrition для ChatGPT (если используется)
        chatgpt_total_nutrition = None
        if use_chatgpt_for_response and chatgpt_result:
            # Рассчитываем total_nutrition на основе веса порции
            chatgpt_multiplier = weight_g / 100.0
            chatgpt_total_nutrition = {
                'calories': round(chatgpt_result.get('calories_per_100g', 0) * chatgpt_multiplier, 1),
                'proteins': round(chatgpt_result.get('proteins_per_100g', 0) * chatgpt_multiplier, 1),
                'fats': round(chatgpt_result.get('fats_per_100g', 0) * chatgpt_multiplier, 1),
                'carbs': round(chatgpt_result.get('carbs_per_100g', 0) * chatgpt_multiplier, 1)
            }
        
        # Шаг 4: Используем ChatGPT или локальную модель для ответа
        if use_chatgpt_for_response and chatgpt_result:
            # Используем результат ChatGPT
            logger.info("Использование результата ChatGPT для ответа")
            food_name = chatgpt_result.get('food_name', 'unknown')
            confidence = chatgpt_result.get('confidence', 0.0)
            nutrition_per_100g = {
                'calories': chatgpt_result.get('calories_per_100g', 0),
                'proteins': chatgpt_result.get('proteins_per_100g', 0),
                'fats': chatgpt_result.get('fats_per_100g', 0),
                'carbs': chatgpt_result.get('carbs_per_100g', 0)
            }
            total_nutrition = NutritionInfo(**chatgpt_total_nutrition)
            recognition_result = {
                'food_name': food_name,
                'confidence': confidence,
                'nutrition_per_100g': nutrition_per_100g,
                'estimation_method': 'chatgpt'
            }
        elif USE_LOCAL_MODEL and food_service:
            # Используем локальную модель
            logger.info("Использование локальной модели для ответа")
            recognition_result = food_service.recognize_and_estimate_nutrition(
                image, 
                use_fallback=True
            )
            nutrition_per_100g = recognition_result['nutrition_per_100g']
            
            # Рассчитываем общие значения на основе веса порции
            multiplier = weight_g / 100.0
            total_nutrition = NutritionInfo(
                calories=round(nutrition_per_100g['calories'] * multiplier, 1),
                proteins=round(nutrition_per_100g['proteins'] * multiplier, 1),
                fats=round(nutrition_per_100g['fats'] * multiplier, 1),
                carbs=round(nutrition_per_100g['carbs'] * multiplier, 1)
            )
        else:
            # Ни одна модель не доступна
            raise HTTPException(
                status_code=503,
                detail="Ни одна модель не доступна. Включите USE_CHATGPT или USE_LOCAL_MODEL в .env"
            )
        
        # Формируем информацию о порции
        portion_estimate = PortionEstimate(
            weight_g=weight_g,
            volume_ml=portion_info['volume_ml'],
            estimated_portion_size=portion_info['portion_size']
        )
        
        # Обработка дополнительных данных от ChatGPT (ингредиенты, рекомендации, микронутриенты)
        ingredients_list = None
        recommendations_list = None
        micronutrients_list = None
        
        if chatgpt_result:
            # Обработка ингредиентов
            if 'ingredients' in chatgpt_result and chatgpt_result['ingredients']:
                try:
                    ingredients_list = []
                    for ing in chatgpt_result['ingredients']:
                        weight_in_portion_g = ing.get('weight_in_portion_g')
                        calories_per_100g = ing.get('calories_per_100g', 0.0)
                        proteins_per_100g = ing.get('proteins_per_100g', 0.0)
                        fats_per_100g = ing.get('fats_per_100g', 0.0)
                        carbs_per_100g = ing.get('carbs_per_100g', 0.0)
                        
                        # Рассчитываем КБЖУ в порции если известен вес ингредиента
                        calories_in_portion = None
                        proteins_in_portion = None
                        fats_in_portion = None
                        carbs_in_portion = None
                        
                        if weight_in_portion_g and weight_in_portion_g > 0:
                            multiplier = weight_in_portion_g / 100.0
                            calories_in_portion = round(calories_per_100g * multiplier, 1)
                            proteins_in_portion = round(proteins_per_100g * multiplier, 1)
                            fats_in_portion = round(fats_per_100g * multiplier, 1)
                            carbs_in_portion = round(carbs_per_100g * multiplier, 1)
                        
                        ingredients_list.append(
                            IngredientInfo(
                                name=ing.get('name', ''),
                                calories_per_100g=calories_per_100g,
                                proteins_per_100g=proteins_per_100g,
                                fats_per_100g=fats_per_100g,
                                carbs_per_100g=carbs_per_100g,
                                description=ing.get('description'),
                                weight_in_portion_g=weight_in_portion_g,
                                calories_in_portion=calories_in_portion,
                                proteins_in_portion=proteins_in_portion,
                                fats_in_portion=fats_in_portion,
                                carbs_in_portion=carbs_in_portion
                            )
                        )
                    logger.debug(f"Обработано {len(ingredients_list)} ингредиентов")
                except Exception as e:
                    logger.warning(f"Ошибка обработки ингредиентов: {e}", exc_info=True)
                    ingredients_list = None
            
            # Обработка рекомендаций
            if 'recommendations' in chatgpt_result and chatgpt_result['recommendations']:
                try:
                    recommendations_list = [
                        Recommendation(
                            type=rec.get('type', 'tip'),
                            title=rec.get('title', ''),
                            description=rec.get('description', ''),
                            calories_saved=rec.get('calories_saved')
                        )
                        for rec in chatgpt_result['recommendations']
                    ]
                    logger.debug(f"Обработано {len(recommendations_list)} рекомендаций")
                except Exception as e:
                    logger.warning(f"Ошибка обработки рекомендаций: {e}")
                    recommendations_list = None
            
            # Обработка микронутриентов с вычислением процента от суточной нормы
            if 'micronutrients' in chatgpt_result and chatgpt_result['micronutrients']:
                try:
                    micronutrients_list = []
                    for micron in chatgpt_result['micronutrients']:
                        amount = float(micron.get('amount', 0.0))
                        daily_value = float(micron.get('daily_value', 1.0))
                        
                        # Вычисляем процент от суточной нормы
                        percent_of_daily_value = 0.0
                        if daily_value > 0:
                            percent_of_daily_value = round((amount / daily_value) * 100, 2)
                        
                        # Масштабируем на вес порции (если нужно)
                        # По умолчанию ChatGPT должен возвращать данные на 100г, но проверим
                        multiplier = weight_g / 100.0
                        scaled_amount = amount * multiplier
                        scaled_percent = 0.0
                        if daily_value > 0:
                            scaled_percent = round((scaled_amount / daily_value) * 100, 2)
                        
                        micronutrients_list.append(
                            MicronutrientInfo(
                                name=micron.get('name', ''),
                                amount=scaled_amount,  # Количество в порции
                                unit=micron.get('unit', 'мг'),
                                daily_value=daily_value,
                                percent_of_daily_value=scaled_percent
                            )
                        )
                    logger.debug(f"Обработано {len(micronutrients_list)} микронутриентов")
                except Exception as e:
                    logger.warning(f"Ошибка обработки микронутриентов: {e}", exc_info=True)
                    micronutrients_list = None
        
        # Формируем ответ
        food_item = FoodItem(
            name=recognition_result['food_name'],
            confidence=recognition_result['confidence'],
            nutrition_per_100g=NutritionInfo(**nutrition_per_100g),
            portion_estimate=portion_estimate,
            total_nutrition=total_nutrition,
            ingredients=ingredients_list,
            recommendations=recommendations_list,
            micronutrients=micronutrients_list
        )
        
        processing_time = time.time() - start_time
        
        # Шаг 5: Вычисляем разницу между моделями (если обе были получены)
        # Для этого нужно получить результаты локальной модели, если использовали ChatGPT для ответа
        difference_calories = None
        difference_proteins = None
        difference_fats = None
        difference_carbs = None
        
        if chatgpt_result and chatgpt_total_nutrition and USE_LOCAL_MODEL and food_service:
            # Получаем результаты локальной модели для сравнения (только если локальная модель включена)
            if use_chatgpt_for_response:
                # Если использовали ChatGPT для ответа, нужно получить результаты локальной модели
                # (они будут получены позже при сохранении в БД, но для вычисления разницы нужны сейчас)
                local_model_for_comparison = food_service.recognize_and_estimate_nutrition(image, use_fallback=True)
                local_multiplier = weight_g / 100.0
                local_total_nutrition_for_comparison = {
                    'calories': round(local_model_for_comparison['nutrition_per_100g']['calories'] * local_multiplier, 1),
                    'proteins': round(local_model_for_comparison['nutrition_per_100g']['proteins'] * local_multiplier, 1),
                    'fats': round(local_model_for_comparison['nutrition_per_100g']['fats'] * local_multiplier, 1),
                    'carbs': round(local_model_for_comparison['nutrition_per_100g']['carbs'] * local_multiplier, 1)
                }
            else:
                # Если использовали локальную модель для ответа, она уже в total_nutrition
                local_total_nutrition_for_comparison = {
                    'calories': total_nutrition.calories,
                    'proteins': total_nutrition.proteins,
                    'fats': total_nutrition.fats,
                    'carbs': total_nutrition.carbs
                }
            
            # Вычисляем разницу между локальной моделью и ChatGPT
            difference_calories = abs(local_total_nutrition_for_comparison['calories'] - chatgpt_total_nutrition['calories'])
            difference_proteins = abs(local_total_nutrition_for_comparison['proteins'] - chatgpt_total_nutrition['proteins'])
            difference_fats = abs(local_total_nutrition_for_comparison['fats'] - chatgpt_total_nutrition['fats'])
            difference_carbs = abs(local_total_nutrition_for_comparison['carbs'] - chatgpt_total_nutrition['carbs'])
        
        # Сохраняем запрос в БД
        # Если локальная модель отключена, сохраняем только ChatGPT результаты
        try:
            # Получаем результаты локальной модели для сохранения в БД (если локальная модель включена)
            local_model_result = None
            local_total_nutrition = None
            
            if USE_LOCAL_MODEL and food_service:
                if use_chatgpt_for_response:
                    # Если использовали ChatGPT для ответа, нужно получить результаты локальной модели
                    logger.debug("Получение результатов локальной модели для сохранения в БД")
                    local_model_result = food_service.recognize_and_estimate_nutrition(image, use_fallback=True)
                else:
                    # Если использовали локальную модель для ответа, она уже в recognition_result
                    local_model_result = recognition_result
                
                local_multiplier = weight_g / 100.0
                local_total_nutrition = {
                    'calories': round(local_model_result['nutrition_per_100g']['calories'] * local_multiplier, 1),
                    'proteins': round(local_model_result['nutrition_per_100g']['proteins'] * local_multiplier, 1),
                    'fats': round(local_model_result['nutrition_per_100g']['fats'] * local_multiplier, 1),
                    'carbs': round(local_model_result['nutrition_per_100g']['carbs'] * local_multiplier, 1)
                }
            
            db_request = FoodRequest(
                # Результаты локальной модели (если доступны)
                recognized_food=local_model_result['food_name'] if local_model_result else None,
                calories=local_total_nutrition['calories'] if local_total_nutrition else None,
                proteins=local_total_nutrition['proteins'] if local_total_nutrition else None,
                fats=local_total_nutrition['fats'] if local_total_nutrition else None,
                carbs=local_total_nutrition['carbs'] if local_total_nutrition else None,
                estimated_weight_g=weight_g,
                estimated_volume_ml=portion_info['volume_ml'],
                confidence=local_model_result['confidence'] if local_model_result else None,
                full_response={
                    "recognition": local_model_result if local_model_result else recognition_result,
                    "portion": portion_info,
                    "estimation_method": (local_model_result.get('estimation_method', 'direct') if local_model_result else recognition_result.get('estimation_method', 'chatgpt')),
                    "used_for_response": not use_chatgpt_for_response if USE_LOCAL_MODEL and food_service else False
                },
                
                # Результаты ChatGPT (если были получены)
                chatgpt_food_name=chatgpt_result.get('food_name') if chatgpt_result else None,
                chatgpt_calories=chatgpt_total_nutrition['calories'] if chatgpt_total_nutrition else None,
                chatgpt_proteins=chatgpt_total_nutrition['proteins'] if chatgpt_total_nutrition else None,
                chatgpt_fats=chatgpt_total_nutrition['fats'] if chatgpt_total_nutrition else None,
                chatgpt_carbs=chatgpt_total_nutrition['carbs'] if chatgpt_total_nutrition else None,
                chatgpt_confidence=chatgpt_result.get('confidence') if chatgpt_result else None,
                chatgpt_response_raw=chatgpt_result if chatgpt_result else None,
                chatgpt_ingredients=[ing.model_dump() for ing in ingredients_list] if ingredients_list else None,
                chatgpt_recommendations=[rec.model_dump() for rec in recommendations_list] if recommendations_list else None,
                chatgpt_micronutrients=[mic.model_dump() for mic in micronutrients_list] if micronutrients_list else None,
                
                # Разница между моделями (если обе были получены)
                difference_calories=difference_calories,
                difference_proteins=difference_proteins,
                difference_fats=difference_fats,
                difference_carbs=difference_carbs,
                
                # Метаданные
                request_image_path=image_path,
                client_ip=request.client.host if request else None
            )
            db.add(db_request)
            db.commit()
            db.refresh(db_request)
            logger.info(f"Сохранено в БД: ID={db_request.id}, ChatGPT={'да' if chatgpt_result else 'нет'}")
        except Exception as e:
            logger.error(f"Ошибка сохранения в БД: {e}")
            db.rollback()
        
        return FoodRecognitionResponse(
            recognized_foods=[food_item],
            message=f"Распознано: {recognition_result['food_name']} (уверенность: {recognition_result['confidence']:.1%})",
            processing_time_seconds=round(processing_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Ошибка обработки изображения",
                "detail": str(e)
            }
        )

@app.get("/requests")
async def get_requests(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db),
    limit: int = 10,
    offset: int = 0
):
    """
    Получить последние запросы (для отладки и мониторинга).
    
    Args:
        token: Токен аутентификации
        db: Сессия базы данных
        limit: Максимальное количество запросов
        offset: Смещение для пагинации
        
    Returns:
        Список последних запросов
    """
    requests = db.query(FoodRequest)\
        .order_by(FoodRequest.timestamp.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return {
        "total": len(requests),
        "requests": [
            {
                "id": req.id,
                "timestamp": req.timestamp.isoformat() if req.timestamp else None,
                "recognized_food": req.recognized_food,
                "calories": req.calories,
                "proteins": req.proteins,
                "fats": req.fats,
                "carbs": req.carbs,
                "estimated_weight_g": req.estimated_weight_g,
                "confidence": req.confidence,
                "client_ip": req.client_ip
            }
            for req in requests
        ]
    }

@app.get("/stats")
async def get_stats(
    token: str = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """
    Получить статистику по запросам.
    
    Args:
        token: Токен аутентификации
        db: Сессия базы данных
        
    Returns:
        Статистика по запросам
    """
    from sqlalchemy import func
    
    total_requests = db.query(func.count(FoodRequest.id)).scalar()
    avg_calories = db.query(func.avg(FoodRequest.calories)).scalar()
    most_recognized = db.query(
        FoodRequest.recognized_food,
        func.count(FoodRequest.id).label('count')
    ).group_by(FoodRequest.recognized_food)\
     .order_by(func.count(FoodRequest.id).desc())\
     .first()
    
    # Статистика по данным для обучения
    training_data_count = db.query(func.count(FoodRequest.id)).filter(
        FoodRequest.chatgpt_calories.isnot(None)
    ).scalar()
    
    return {
        "total_requests": total_requests or 0,
        "average_calories": round(avg_calories, 1) if avg_calories else None,
        "most_recognized_food": most_recognized[0] if most_recognized else None,
        "most_recognized_count": most_recognized[1] if most_recognized else 0,
        "training_data_samples": training_data_count or 0
    }

@app.post("/train-model")
async def train_model(
    token: str = Depends(verify_token),
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 0.001
):
    """
    Запустить обучение модели на данных из БД.
    
    Требует минимум 50-100 образцов с данными ChatGPT.
    
    Args:
        token: Токен аутентификации
        epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Скорость обучения
        
    Returns:
        Результаты обучения
    """
    try:
        from app.training.training_service import training_service
        
        logger.info(f"Запуск обучения: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        result = training_service.train(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return {
            "status": "success" if result.get("success") else "error",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

