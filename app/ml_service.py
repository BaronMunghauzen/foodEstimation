"""
ML сервис для распознавания еды и оценки калорий/БЖУ напрямую по изображению.
Использует комбинацию моделей для максимальной точности.
"""

from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import logging
import cv2
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodRecognitionService:
    """Сервис для распознавания еды и оценки калорий"""
    
    def __init__(self, use_trained_model: bool = False):
        """
        Args:
            use_trained_model: Использовать ли дообученную модель вместо эвристик
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_trained_model = use_trained_model
        self.trained_model = None
        self.trained_food_classifier = None
        self.processor = None
        self.food_name_dict = None
        
        logger.info(f"Используется устройство: {self.device}")
        logger.info(f"Использование дообученной модели: {use_trained_model}")
        
        # Загружаем модель для распознавания типа блюда
        # ВСЕГДА пробуем загрузить обученный классификатор, если доступен (независимо от USE_TRAINED_MODEL)
        # Классификатор типов еды должен использоваться всегда, если он обучен
        self._load_trained_classifier()
        
        # Загружаем регрессор калорий/БЖУ только если USE_TRAINED_MODEL=true
        if self.use_trained_model:
            self._load_trained_nutrition_model()
        
        # ВСЕГДА загружаем предобученную модель как fallback (даже если обученный классификатор загружен)
        # Она нужна для случаев, когда уверенность обученного классификатора низкая
        try:
            logger.info("Загрузка предобученной модели распознавания еды (fallback)...")
            self.food_classifier = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",  # Универсальная модель
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Предобученная модель распознавания загружена успешно (ViT-base)")
        except Exception as e:
            logger.warning(f"Ошибка загрузки предобученной модели: {e}")
            logger.info("Используется резервная модель...")
            try:
                self.food_classifier = pipeline(
                    "image-classification",
                    model="google/vit-base-patch16-224",
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e2:
                logger.error(f"Критическая ошибка: не удалось загрузить предобученную модель: {e2}")
                self.food_classifier = None
    
    def _load_trained_classifier(self):
        """Загружает обученный классификатор типов еды (всегда, если доступен)"""
        try:
            from app.training.food_classifier_model import FoodClassifierModel
            import json
            
            # Загружаем словарь названий еды
            food_dict_path = Path("models/food_name_dict.json")
            if food_dict_path.exists():
                with open(food_dict_path, 'r', encoding='utf-8') as f:
                    self.food_name_dict = json.load(f)
                logger.info(f"Словарь названий еды загружен: {len(self.food_name_dict['name_to_idx'])} классов")
            else:
                logger.debug(f"Словарь названий еды не найден: {food_dict_path}")
                return
            
            # Загружаем классификатор типов еды
            classifier_path = Path("models/food_classifier_best.pth")
            if classifier_path.exists() and self.food_name_dict:
                logger.info("Загрузка обученного классификатора типов еды...")
                self.trained_food_classifier = FoodClassifierModel.load_model(
                    str(classifier_path),
                    device=self.device
                )
                # Загружаем процессор для предобработки
                if self.processor is None:
                    self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                logger.info("Обученный классификатор типов еды загружен успешно")
            else:
                logger.debug(f"Обученный классификатор не найден: {classifier_path}")
                
        except Exception as e:
            logger.warning(f"Ошибка загрузки обученного классификатора: {e}")
            logger.debug("Будет использована предобученная модель", exc_info=True)
    
    def _load_trained_nutrition_model(self):
        """Загружает дообученную модель для предсказания калорий/БЖУ"""
        try:
            from app.training.nutrition_model import NutritionRegressionModel
            
            # Загружаем регрессор калорий/БЖУ
            nutrition_model_path = Path("models/nutrition_model_best.pth")
            if nutrition_model_path.exists():
                logger.info("Загрузка дообученной модели калорий/БЖУ...")
                self.trained_model = NutritionRegressionModel.load_model(
                    str(nutrition_model_path),
                    device=self.device
                )
                # Загружаем процессор для предобработки (если еще не загружен)
                if self.processor is None:
                    self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                logger.info("Дообученная модель калорий/БЖУ загружена успешно")
            else:
                logger.warning(f"Дообученная модель калорий не найдена: {nutrition_model_path}")
                logger.warning("Используются эвристики вместо модели")
                self.use_trained_model = False
                
        except Exception as e:
            logger.error(f"Ошибка загрузки дообученной модели калорий: {e}", exc_info=True)
            logger.warning("Используются эвристики вместо модели")
            self.use_trained_model = False
    
    def recognize_and_estimate_nutrition(
        self, 
        image: Image.Image,
        use_fallback: bool = True
    ) -> Dict:
        """
        Распознает еду и оценивает калории/БЖУ напрямую по изображению.
        
        Args:
            image: PIL Image объекта
            use_fallback: Использовать ли справочные значения при низкой уверенности
            
        Returns:
            dict с информацией о распознанной еде и оценке калорий/БЖУ
        """
        start_time = time.time()
        
        try:
            # Шаг 1: Распознаем тип блюда
            # Используем обученный классификатор, если доступен
            top_predictions = []  # Инициализируем для использования в return
            if self.trained_food_classifier is not None and self.food_name_dict:
                try:
                    food_name, confidence = self._classify_food_with_trained_model(image)
                    
                    # Если уверенность очень низкая (< 0.1), используем предобученную модель как fallback
                    if confidence < 0.1:
                        logger.warning(
                            f"Уверенность обученного классификатора слишком низкая ({confidence:.4f}). "
                            f"Используем предобученную модель как fallback."
                        )
                        if self.food_classifier is None:
                            logger.error("Предобученная модель не загружена! Используем результат обученного классификатора")
                            top_predictions = [
                                {'label': food_name, 'score': confidence}
                            ]
                        else:
                            predictions = self.food_classifier(image)
                            if isinstance(predictions, list):
                                top_predictions = predictions[:3]
                            else:
                                top_predictions = [predictions]
                            best_prediction = top_predictions[0]
                            food_name = best_prediction.get('label', 'unknown')
                            confidence = best_prediction.get('score', 0.0)
                            logger.info(f"Предобученная модель предсказала: {food_name} (уверенность: {confidence:.2f})")
                    else:
                        logger.info(f"Использован обученный классификатор: {food_name} (уверенность: {confidence:.4f})")
                        # Формируем top_predictions для совместимости с форматом ответа
                        top_predictions = [
                            {'label': food_name, 'score': confidence}
                        ]
                except Exception as e:
                    logger.warning(f"Ошибка обученного классификатора: {e}, используем предобученную модель")
                    if self.food_classifier is None:
                        logger.error("Предобученная модель не загружена! Используем fallback значения")
                        food_name = 'unknown'
                        confidence = 0.0
                        top_predictions = [{'label': food_name, 'score': confidence}]
                    else:
                        predictions = self.food_classifier(image)
                        if isinstance(predictions, list):
                            top_predictions = predictions[:3]
                        else:
                            top_predictions = [predictions]
                        best_prediction = top_predictions[0]
                        food_name = best_prediction.get('label', 'unknown')
                        confidence = best_prediction.get('score', 0.0)
            else:
                # Используем предобученную модель
                predictions = self.food_classifier(image)
                
                # Обрабатываем результаты
                if isinstance(predictions, list):
                    top_predictions = predictions[:3]  # Топ-3 предсказания
                else:
                    top_predictions = [predictions]
                
                # Берем лучшее предсказание
                best_prediction = top_predictions[0]
                food_name = best_prediction.get('label', 'unknown')
                confidence = best_prediction.get('score', 0.0)
            
            # Шаг 2: Оцениваем калории и БЖУ напрямую по изображению
            # Используем дообученную модель или эвристики
            if self.use_trained_model and self.trained_model is not None:
                try:
                    direct_estimation = self._estimate_nutrition_with_model(image)
                    estimation_method = "trained_model"
                except Exception as e:
                    logger.warning(f"Ошибка использования обученной модели: {e}, используем эвристики")
                    direct_estimation = self._estimate_nutrition_from_image(image, food_name, confidence)
                    estimation_method = "heuristics"
            else:
                direct_estimation = self._estimate_nutrition_from_image(image, food_name, confidence)
                estimation_method = "heuristics"
            
            # Шаг 3: Если уверенность низкая, используем fallback
            if use_fallback and confidence < 0.5:
                logger.info(f"Низкая уверенность ({confidence:.2f}), используется fallback")
                from app.nutrition_db import get_nutrition_fallback
                fallback_nutrition = get_nutrition_fallback(food_name)
                
                # Комбинируем прямую оценку с fallback (взвешенное среднее)
                weight_direct = confidence
                weight_fallback = 1 - confidence
                
                direct_estimation['calories_per_100g'] = (
                    direct_estimation['calories_per_100g'] * weight_direct +
                    fallback_nutrition['calories_per_100g'] * weight_fallback
                )
                direct_estimation['proteins_per_100g'] = (
                    direct_estimation['proteins_per_100g'] * weight_direct +
                    fallback_nutrition['proteins_per_100g'] * weight_fallback
                )
                direct_estimation['fats_per_100g'] = (
                    direct_estimation['fats_per_100g'] * weight_direct +
                    fallback_nutrition['fats_per_100g'] * weight_fallback
                )
                direct_estimation['carbs_per_100g'] = (
                    direct_estimation['carbs_per_100g'] * weight_direct +
                    fallback_nutrition['carbs_per_100g'] * weight_fallback
                )
            
            processing_time = time.time() - start_time
            
            return {
                "food_name": food_name,
                "confidence": float(confidence),
                "nutrition_per_100g": {
                    "calories": round(direct_estimation['calories_per_100g'], 1),
                    "proteins": round(direct_estimation['proteins_per_100g'], 1),
                    "fats": round(direct_estimation['fats_per_100g'], 1),
                    "carbs": round(direct_estimation['carbs_per_100g'], 1)
                },
                "all_predictions": top_predictions,
                "processing_time": round(processing_time, 2),
                "estimation_method": estimation_method if confidence >= 0.5 else f"{estimation_method}_hybrid"
            }
            
        except Exception as e:
            logger.error(f"Ошибка при распознавании: {e}")
            raise Exception(f"Ошибка обработки изображения: {str(e)}")
    
    def _classify_food_with_trained_model(self, image: Image.Image) -> tuple:
        """
        Классифицирует тип еды используя обученный классификатор.
        
        Args:
            image: PIL Image объекта
            
        Returns:
            tuple: (food_name, confidence)
        """
        if self.trained_food_classifier is None or self.processor is None or self.food_name_dict is None:
            raise ValueError("Обученный классификатор, процессор или словарь не загружены")
        
        # Предобрабатываем изображение
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        # Переводим модель в режим оценки
        self.trained_food_classifier.eval()
        
        # Предсказание
        with torch.no_grad():
            logits = self.trained_food_classifier(pixel_values)
        
        # Применяем softmax для получения вероятностей
        probs = torch.softmax(logits, dim=-1)
        
        # Получаем топ-3 предсказания для диагностики
        top_k = min(3, probs.shape[1])
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        
        # Берем класс с максимальной вероятностью
        predicted_class_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class_idx].item()
        
        # Преобразуем индекс в название еды
        idx_to_name = self.food_name_dict['idx_to_name']
        food_name = idx_to_name.get(str(predicted_class_idx), 'unknown')
        
        # Логируем топ-3 предсказания для диагностики
        top_predictions_list = []
        for i in range(top_k):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            name = idx_to_name.get(str(idx), 'unknown')
            top_predictions_list.append(f"{name} ({prob:.4f})")
        
        logger.info(f"Обученный классификатор - Топ-3: {', '.join(top_predictions_list)}")
        logger.info(f"Выбрано: {food_name} (уверенность: {confidence:.4f})")
        
        # Если уверенность очень низкая (< 0.1), предупреждаем
        if confidence < 0.1:
            logger.warning(
                f"Очень низкая уверенность ({confidence:.4f})! "
                f"Модель недостаточно обучена или не видела подобных примеров. "
                f"Рекомендуется переобучить модель с большим количеством данных."
            )
        
        return food_name, confidence
    
    def _estimate_nutrition_with_model(self, image: Image.Image) -> Dict:
        """
        Оценивает калории и БЖУ используя дообученную нейронную сеть.
        
        Args:
            image: PIL Image объекта
            
        Returns:
            dict с оценкой калорий/БЖУ на 100г
        """
        if self.trained_model is None or self.processor is None:
            raise ValueError("Обученная модель или процессор не загружены")
        
        # Предобрабатываем изображение
        pixel_values = self.processor(
            image,
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        # Переводим модель в режим оценки (не обучения)
        self.trained_model.eval()
        
        # Предсказание (без вычисления градиентов для ускорения)
        with torch.no_grad():
            predictions = self.trained_model(pixel_values)
        
        # Модель возвращает тензор [batch_size, 4]
        # Извлекаем значения для первого (и единственного) элемента батча
        if predictions.dim() > 1:
            predictions = predictions[0]
        
        # Преобразуем в numpy и затем в список
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Модель предсказывает: [calories_per_100g, proteins_per_100g, fats_per_100g, carbs_per_100g]
        calories = float(predictions[0]) if len(predictions) > 0 else 0.0
        proteins = float(predictions[1]) if len(predictions) > 1 else 0.0
        fats = float(predictions[2]) if len(predictions) > 2 else 0.0
        carbs = float(predictions[3]) if len(predictions) > 3 else 0.0
        
        # Убеждаемся, что значения неотрицательные
        calories = max(0.0, calories)
        proteins = max(0.0, proteins)
        fats = max(0.0, fats)
        carbs = max(0.0, carbs)
        
        logger.debug(f"Модель предсказала: калории={calories:.1f}, белки={proteins:.1f}, жиры={fats:.1f}, углеводы={carbs:.1f}")
        
        return {
            'calories_per_100g': calories,
            'proteins_per_100g': proteins,
            'fats_per_100g': fats,
            'carbs_per_100g': carbs
        }
    
    def _estimate_nutrition_from_image(
        self, 
        image: Image.Image, 
        food_name: str,
        confidence: float
    ) -> Dict:
        """
        Оценивает калории и БЖУ напрямую по визуальным признакам изображения.
        
        Использует анализ:
        - Цвета (жирность, способ приготовления)
        - Текстуры (тип блюда)
        - Яркости (обжаренность, калорийность)
        """
        img_array = np.array(image)
        
        # Анализ цвета для оценки жирности и калорийности
        color_features = self._analyze_color_features(img_array)
        
        # Анализ текстуры для определения типа блюда
        texture_features = self._analyze_texture_features(img_array)
        
        # Базовые значения на основе типа блюда
        base_nutrition = self._get_base_nutrition_by_food_type(food_name)
        
        # Корректировка на основе визуальных признаков
        calories = base_nutrition['calories']
        proteins = base_nutrition['proteins']
        fats = base_nutrition['fats']
        carbs = base_nutrition['carbs']
        
        # Корректировка калорий на основе цвета (темнее = больше калорий)
        # Темные/обжаренные продукты обычно калорийнее
        if color_features['darkness'] > 0.6:
            calories *= 1.15  # +15% для темных/обжаренных блюд
            fats *= 1.1
        
        # Корректировка на основе насыщенности (жирность)
        if color_features['saturation'] > 0.5:
            calories *= 1.1
            fats *= 1.15
        
        # Корректировка на основе текстуры (плотность)
        if texture_features['density'] > 0.7:
            calories *= 1.1
            proteins *= 1.05
        
        # Применяем уверенность модели
        # При низкой уверенности делаем оценку более консервативной
        confidence_factor = 0.7 + (confidence * 0.3)  # От 0.7 до 1.0
        
        return {
            'calories_per_100g': calories * confidence_factor,
            'proteins_per_100g': proteins * confidence_factor,
            'fats_per_100g': fats * confidence_factor,
            'carbs_per_100g': carbs * confidence_factor
        }
    
    def _analyze_color_features(self, img_array: np.ndarray) -> Dict:
        """Анализирует цветовые характеристики изображения"""
        # Конвертируем в HSV для лучшего анализа
        if len(img_array.shape) == 3:
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            value = np.mean(hsv[:, :, 2]) / 255.0
        else:
            saturation = 0.5
            value = np.mean(img_array) / 255.0
        
        # Темнота (обратная яркость)
        darkness = 1.0 - value
        
        return {
            'saturation': float(saturation),
            'darkness': float(darkness),
            'brightness': float(value)
        }
    
    def _analyze_texture_features(self, img_array: np.ndarray) -> Dict:
        """Анализирует текстуру изображения"""
        # Конвертируем в grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Вычисляем вариацию (показатель текстуры/плотности)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Нормализуем (обычно значения от 0 до 1000+)
        density = min(laplacian_var / 500.0, 1.0)
        
        return {
            'density': float(density),
            'texture_variance': float(laplacian_var)
        }
    
    def _get_base_nutrition_by_food_type(self, food_name: str) -> Dict:
        """
        Получает базовые значения калорий/БЖУ на основе типа блюда.
        Используется как отправная точка для визуальной оценки.
        """
        food_lower = food_name.lower()
        
        # Категории блюд с типичными значениями
        if any(x in food_lower for x in ['pizza', 'burger', 'sandwich']):
            return {'calories': 280, 'proteins': 12, 'fats': 12, 'carbs': 30}
        elif any(x in food_lower for x in ['salad', 'vegetable', 'veggie']):
            return {'calories': 25, 'proteins': 1.5, 'fats': 0.3, 'carbs': 5}
        elif any(x in food_lower for x in ['meat', 'chicken', 'beef', 'steak', 'pork']):
            return {'calories': 200, 'proteins': 25, 'fats': 10, 'carbs': 0}
        elif any(x in food_lower for x in ['fish', 'seafood', 'salmon']):
            return {'calories': 180, 'proteins': 20, 'fats': 10, 'carbs': 0}
        elif any(x in food_lower for x in ['pasta', 'noodle', 'spaghetti']):
            return {'calories': 130, 'proteins': 5, 'fats': 1, 'carbs': 25}
        elif any(x in food_lower for x in ['rice', 'grain']):
            return {'calories': 130, 'proteins': 3, 'fats': 0.3, 'carbs': 28}
        elif any(x in food_lower for x in ['fruit', 'apple', 'banana', 'orange']):
            return {'calories': 60, 'proteins': 0.5, 'fats': 0.2, 'carbs': 15}
        elif any(x in food_lower for x in ['dessert', 'cake', 'sweet', 'ice cream']):
            return {'calories': 350, 'proteins': 4, 'fats': 15, 'carbs': 50}
        elif any(x in food_lower for x in ['soup', 'broth']):
            return {'calories': 50, 'proteins': 3, 'fats': 2, 'carbs': 6}
        elif any(x in food_lower for x in ['sushi', 'roll']):
            return {'calories': 150, 'proteins': 6, 'fats': 1, 'carbs': 28}
        else:
            # Средние значения для неизвестных блюд
            return {'calories': 200, 'proteins': 10, 'fats': 8, 'carbs': 20}

# Глобальный экземпляр сервиса (инициализируется в main.py с параметрами из .env)
food_service = None

def init_food_service(use_trained_model: bool = False):
    """Инициализирует глобальный экземпляр сервиса"""
    global food_service
    food_service = FoodRecognitionService(use_trained_model=use_trained_model)
    return food_service

