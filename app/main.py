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
from app.models import FoodRequest, MealPlan
from app.schemas import (
    FoodRecognitionResponse, FoodItem, NutritionInfo, PortionEstimate, 
    IngredientInfo, Recommendation, MicronutrientInfo,
    MealPlanRequest, MealPlanResponse, DayPlan, Meal, MealItem, RecipeInfo
)
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
            "create_meal_plan": "/create-meal-plan",
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

@app.post("/create-meal-plan", response_model=MealPlanResponse)
async def create_meal_plan(
    request_data: MealPlanRequest,
    token: str = Depends(verify_token),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для создания программы питания с помощью ChatGPT.
    
    Требует токен аутентификации в заголовке:
    Authorization: Bearer <your-token>
    
    Args:
        request_data: Данные запроса (meals_per_day, days_count, target_nutrition, allowed_recipes)
        token: Токен аутентификации (из dependency)
        request: FastAPI Request объект
        db: Сессия базы данных
        
    Returns:
        MealPlanResponse с программой питания на указанное количество дней
    """
    start_time = time.time()
    
    try:
        # Валидация входных данных
        if request_data.meals_per_day < 3:
            raise HTTPException(
                status_code=400,
                detail="Количество приемов пищи в день должно быть не менее 3"
            )
        
        if request_data.days_count < 1:
            raise HTTPException(
                status_code=400,
                detail="Количество дней должно быть не менее 1"
            )
        
        if not request_data.allowed_recipes:
            raise HTTPException(
                status_code=400,
                detail="Список рецептов не может быть пустым"
            )
        
        # Проверяем, что все рецепты имеют валидные данные
        recipes_dict = {}  # Словарь для быстрого доступа по UUID
        for recipe in request_data.allowed_recipes:
            if not recipe.uuid or not recipe.name:
                raise HTTPException(
                    status_code=400,
                    detail=f"Рецепт должен иметь UUID и название"
                )
            recipes_dict[recipe.uuid] = {
                'name': recipe.name,
                'category': recipe.category,
                'calories': recipe.calories,
                'proteins': recipe.proteins,
                'fats': recipe.fats,
                'carbs': recipe.carbs
            }
        
        logger.info(f"Создание программы питания: {request_data.days_count} дней, {request_data.meals_per_day} приемов пищи в день")
        logger.info(f"Целевые КБЖУ: {request_data.target_nutrition.calories} ккал, "
                   f"{request_data.target_nutrition.proteins}г белков, "
                   f"{request_data.target_nutrition.fats}г жиров, "
                   f"{request_data.target_nutrition.carbs}г углеводов")
        logger.info(f"Доступно рецептов: {len(request_data.allowed_recipes)}")
        
        # Подготавливаем данные для ChatGPT
        allowed_recipes_list = [
            {
                'uuid': r.uuid,
                'name': r.name,
                'category': r.category,
                'calories': r.calories,
                'proteins': r.proteins,
                'fats': r.fats,
                'carbs': r.carbs
            }
            for r in request_data.allowed_recipes
        ]
        
        # Вызываем ChatGPT для генерации программы питания
        if not chatgpt_service.enabled:
            raise HTTPException(
                status_code=503,
                detail="ChatGPT сервис недоступен (проверьте OPENAI_API_KEY в .env)"
            )
        
        logger.info("Вызов ChatGPT API для генерации программы питания...")
        chatgpt_result = await chatgpt_service.generate_meal_plan(
            meals_per_day=request_data.meals_per_day,
            days_count=request_data.days_count,
            target_calories=request_data.target_nutrition.calories,
            target_proteins=request_data.target_nutrition.proteins,
            target_fats=request_data.target_nutrition.fats,
            target_carbs=request_data.target_nutrition.carbs,
            allowed_recipes=allowed_recipes_list
        )
        
        if not chatgpt_result:
            raise HTTPException(
                status_code=500,
                detail="Не удалось сгенерировать программу питания через ChatGPT"
            )
        
        logger.info(f"ChatGPT успешно сгенерировал программу питания")
        
        # Функция для определения, является ли блюдо гарниром
        def is_side_dish(name: str, category: str) -> bool:
            """Определяет, является ли блюдо гарниром"""
            name_lower = name.lower()
            side_dish_keywords = ['пюре', 'рис', 'гречка', 'макароны', 'паста', 'картофель', 
                                 'овощи', 'салат', 'гарнир', 'каша', 'овсянка', 'перловка']
            return any(keyword in name_lower for keyword in side_dish_keywords)
        
        # Функция для определения, является ли блюдо основным
        def is_main_dish(name: str, category: str) -> bool:
            """Определяет, является ли блюдо основным (мясо, рыба, птица)"""
            name_lower = name.lower()
            main_dish_keywords = ['котлет', 'куриц', 'мясо', 'рыб', 'говядин', 'свинин', 
                                 'индейк', 'телятин', 'стейк', 'шашлык', 'бифштекс']
            return any(keyword in name_lower for keyword in main_dish_keywords)
        
        # Функция постобработки: объединяет гарниры с основными блюдами (без удаления дубликатов)
        def postprocess_meals(meals_data: list, recipes_dict: dict) -> list:
            """Постобработка приемов пищи: объединяет гарниры с основными блюдами"""
            processed_meals = []
            
            for meal_data in meals_data:
                category = meal_data.get('category', '')
                meals_items = meal_data.get('meals', [])
                
                # Проверяем валидность UUID и объединяем одинаковые блюда без portions
                # Если ChatGPT вернул дубликаты без portions, объединяем их в одно блюдо с правильным количеством порций
                valid_items = []
                items_by_uuid = {}  # Группируем по UUID
                
                for item in meals_items:
                    meal_uuid = item.get('uuid', '')
                    if meal_uuid and meal_uuid in recipes_dict:
                        portions = item.get('portions', 1)
                        
                        # Если portions не указан явно, считаем это как 1 порция
                        # Если блюдо уже есть, увеличиваем количество порций
                        if meal_uuid in items_by_uuid:
                            # Если у существующего блюда не указаны portions, считаем их как 1
                            existing_portions = items_by_uuid[meal_uuid].get('portions', 1)
                            items_by_uuid[meal_uuid]['portions'] = existing_portions + portions
                        else:
                            # Создаем новое блюдо
                            new_item = item.copy()
                            if 'portions' not in new_item:
                                new_item['portions'] = 1
                            items_by_uuid[meal_uuid] = new_item
                
                # Преобразуем обратно в список
                valid_items = list(items_by_uuid.values())
                
                # Разделяем на гарниры и основные блюда для лучшего объединения
                side_dishes = []
                main_dishes = []
                other_dishes = []
                
                for item in valid_items:
                    meal_uuid = item.get('uuid', '')
                    recipe_info = recipes_dict[meal_uuid]
                    name = recipe_info['name']
                    cat = recipe_info['category']
                    
                    if is_side_dish(name, cat):
                        side_dishes.append(item)
                    elif is_main_dish(name, cat):
                        main_dishes.append(item)
                    else:
                        other_dishes.append(item)
                
                # Объединяем гарниры с основными блюдами (опционально, если есть)
                final_items = []
                
                # Если есть основные блюда, объединяем с гарнирами
                if main_dishes:
                    # Каждое основное блюдо может иметь гарнир
                    for main_dish in main_dishes:
                        final_items.append(main_dish)
                        # Добавляем один гарнир к основному блюду, если есть
                        if side_dishes:
                            final_items.append(side_dishes.pop(0))
                    # Остальные гарниры добавляем отдельно (если остались)
                    final_items.extend(side_dishes)
                else:
                    # Если нет основных блюд, просто добавляем все
                    final_items.extend(side_dishes)
                
                # Добавляем остальные блюда
                final_items.extend(other_dishes)
                
                # Если после обработки остались блюда, добавляем прием пищи
                if final_items:
                    processed_meals.append({
                        'category': category,
                        'meals': final_items
                    })
            
            return processed_meals
        
        # Обрабатываем результат ChatGPT и рассчитываем фактические КБЖУ
        days_plan = []
        for day_data in chatgpt_result.get('days', []):
            day_number = day_data.get('day_number', 0)
            meals_data = day_data.get('meals', [])
            
            # Применяем постобработку: убираем дубликаты и объединяем гарниры
            processed_meals_data = postprocess_meals(meals_data, recipes_dict)
            
            # Рассчитываем фактические КБЖУ для дня
            actual_calories = 0.0
            actual_proteins = 0.0
            actual_fats = 0.0
            actual_carbs = 0.0
            
            meals_list = []
            for meal_data in processed_meals_data:
                category = meal_data.get('category', '')
                meals_items = meal_data.get('meals', [])
                
                meal_items_list = []
                for meal_item_data in meals_items:
                    meal_uuid = meal_item_data.get('uuid', '')
                    meal_name = meal_item_data.get('name', '')
                    portions = meal_item_data.get('portions', 1)  # Количество порций (по умолчанию 1)
                    
                    # Проверяем, что рецепт существует в списке доступных
                    if meal_uuid not in recipes_dict:
                        logger.warning(f"Рецепт с UUID {meal_uuid} не найден в списке доступных рецептов")
                        continue
                    
                    recipe_info = recipes_dict[meal_uuid]
                    
                    # Добавляем КБЖУ этого рецепта к общим КБЖУ дня (с учетом количества порций)
                    actual_calories += recipe_info['calories'] * portions
                    actual_proteins += recipe_info['proteins'] * portions
                    actual_fats += recipe_info['fats'] * portions
                    actual_carbs += recipe_info['carbs'] * portions
                    
                    meal_items_list.append(
                        MealItem(uuid=meal_uuid, name=meal_name, portions=portions)
                    )
                
                if meal_items_list:  # Добавляем только если есть блюда
                    meals_list.append(
                        Meal(category=category, meals=meal_items_list)
                    )
            
            # Проверяем превышение целевых КБЖУ
            target_cal = request_data.target_nutrition.calories
            target_prot = request_data.target_nutrition.proteins
            target_fat = request_data.target_nutrition.fats
            target_carb = request_data.target_nutrition.carbs
            
            # Допустимое превышение: 10%
            max_allowed_cal = target_cal * 1.1
            max_allowed_prot = target_prot * 1.1
            max_allowed_fat = target_fat * 1.1
            max_allowed_carb = target_carb * 1.1
            
            # Проверяем превышение
            if actual_calories > max_allowed_cal:
                excess_percent = ((actual_calories - target_cal) / target_cal) * 100
                logger.warning(
                    f"День {day_number}: Превышение калорий на {excess_percent:.1f}% "
                    f"(целевые: {target_cal}, фактические: {actual_calories:.1f})"
                )
            
            if actual_proteins > max_allowed_prot:
                excess_percent = ((actual_proteins - target_prot) / target_prot) * 100
                logger.warning(
                    f"День {day_number}: Превышение белков на {excess_percent:.1f}% "
                    f"(целевые: {target_prot}, фактические: {actual_proteins:.1f})"
                )
            
            if actual_fats > max_allowed_fat:
                excess_percent = ((actual_fats - target_fat) / target_fat) * 100
                logger.warning(
                    f"День {day_number}: Превышение жиров на {excess_percent:.1f}% "
                    f"(целевые: {target_fat}, фактические: {actual_fats:.1f})"
                )
            
            if actual_carbs > max_allowed_carb:
                excess_percent = ((actual_carbs - target_carb) / target_carb) * 100
                logger.warning(
                    f"День {day_number}: Превышение углеводов на {excess_percent:.1f}% "
                    f"(целевые: {target_carb}, фактические: {actual_carbs:.1f})"
                )
            
            # Создаем план дня
            days_plan.append(
                DayPlan(
                    day_number=day_number,
                    target_nutrition=NutritionInfo(
                        calories=target_cal,
                        proteins=target_prot,
                        fats=target_fat,
                        carbs=target_carb
                    ),
                    actual_nutrition=NutritionInfo(
                        calories=round(actual_calories, 1),
                        proteins=round(actual_proteins, 1),
                        fats=round(actual_fats, 1),
                        carbs=round(actual_carbs, 1)
                    ),
                    meals=meals_list
                )
            )
        
        # Получаем IP клиента
        client_ip = None
        if request:
            client_ip = request.client.host if request.client else None
        
        # Сохраняем в БД
        db_meal_plan = MealPlan(
            meals_per_day=request_data.meals_per_day,
            days_count=request_data.days_count,
            target_calories=request_data.target_nutrition.calories,
            target_proteins=request_data.target_nutrition.proteins,
            target_fats=request_data.target_nutrition.fats,
            target_carbs=request_data.target_nutrition.carbs,
            allowed_recipes=[r.model_dump() for r in request_data.allowed_recipes],
            chatgpt_response_raw=chatgpt_result,
            client_ip=client_ip
        )
        
        db.add(db_meal_plan)
        db.commit()
        db.refresh(db_meal_plan)
        
        logger.info(f"Программа питания сохранена в БД с ID: {db_meal_plan.id}")
        
        # Проверяем общее превышение по всем дням
        total_excess_days = 0
        for day_plan in days_plan:
            target_cal = day_plan.target_nutrition.calories
            actual_cal = day_plan.actual_nutrition.calories
            if actual_cal > target_cal * 1.1:  # Превышение более 10%
                total_excess_days += 1
        
        # Формируем сообщение с предупреждением, если есть превышение
        message = f"Программа питания успешно создана на {request_data.days_count} дней"
        if total_excess_days > 0:
            message += f". ВНИМАНИЕ: В {total_excess_days} дне(днях) превышение целевых КБЖУ более чем на 10%. Рекомендуется пересмотреть программу."
            logger.warning(f"Обнаружено превышение КБЖУ в {total_excess_days} днях из {request_data.days_count}")
        
        processing_time = time.time() - start_time
        
        return MealPlanResponse(
            days=days_plan,
            message=message,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при создании программы питания: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

