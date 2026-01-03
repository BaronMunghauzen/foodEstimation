"""
Модуль для оценки размера порции по изображению.
Использует анализ размера объекта, глубины и визуальных признаков.
"""

import numpy as np
from PIL import Image
import cv2

class PortionEstimator:
    """Класс для оценки размера порции еды на изображении"""
    
    def __init__(self):
        # Стандартные размеры портативной тарелки (диаметр в пикселях при нормальном расстоянии)
        self.standard_plate_diameter_px = 400  # Примерно для фото с расстояния 30-40 см
        
    def estimate_portion_size(self, image: Image.Image, food_mask: np.ndarray = None) -> dict:
        """
        Оценивает размер порции на изображении.
        
        Args:
            image: PIL Image объекта
            food_mask: Опциональная маска области с едой (если доступна)
            
        Returns:
            dict с ключами:
            - weight_g: примерный вес в граммах
            - volume_ml: примерный объем в миллилитрах
            - portion_size: "small", "medium", "large"
            - confidence: уверенность оценки (0-1)
        """
        # Конвертируем в numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Если маска не предоставлена, используем простую оценку на основе размера изображения
        if food_mask is None:
            # Простая эвристика: предполагаем, что еда занимает центральную часть изображения
            food_area_ratio = self._estimate_food_area_ratio(img_array)
        else:
            food_area_ratio = np.sum(food_mask > 0) / (width * height)
        
        # Оцениваем размер порции на основе занимаемой площади
        # Используем эвристики для разных типов блюд
        portion_info = self._calculate_portion_from_area(food_area_ratio, width, height)
        
        return portion_info
    
    def _estimate_food_area_ratio(self, img_array: np.ndarray) -> float:
        """
        Оценивает долю изображения, занимаемую едой.
        Использует простые методы компьютерного зрения.
        """
        # Конвертируем в grayscale для анализа
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Применяем адаптивную пороговую обработку для выделения объектов
        # Это помогает отделить еду от фона (тарелки, стола)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Находим контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Если не нашли контуры, предполагаем средний размер
            return 0.4
        
        # Находим наибольший контур (предположительно, это еда)
        largest_contour = max(contours, key=cv2.contourArea)
        food_area = cv2.contourArea(largest_contour)
        total_area = gray.shape[0] * gray.shape[1]
        
        return min(food_area / total_area, 0.9)  # Ограничиваем максимум 90%
    
    def _calculate_portion_from_area(
        self, 
        area_ratio: float, 
        image_width: int, 
        image_height: int
    ) -> dict:
        """
        Вычисляет размер порции на основе занимаемой площади.
        
        Args:
            area_ratio: Доля изображения, занимаемая едой (0-1)
            image_width: Ширина изображения
            image_height: Высота изображения
            
        Returns:
            dict с информацией о порции
        """
        # Нормализуем площадь относительно стандартного размера
        # Предполагаем, что при area_ratio=0.3 это средняя порция (~200г)
        base_weight_g = 200.0
        base_volume_ml = 250.0
        
        # Масштабируем на основе area_ratio
        # Используем нелинейную зависимость для более реалистичной оценки
        if area_ratio < 0.15:
            portion_size = "small"
            weight_multiplier = 0.6
            volume_multiplier = 0.7
        elif area_ratio < 0.35:
            portion_size = "medium"
            weight_multiplier = 1.0
            volume_multiplier = 1.0
        elif area_ratio < 0.55:
            portion_size = "large"
            weight_multiplier = 1.5
            volume_multiplier = 1.4
        else:
            portion_size = "extra_large"
            weight_multiplier = 2.2
            volume_multiplier = 2.0
        
        # Дополнительная корректировка на основе размера изображения
        # Большие изображения могут означать, что фото сделано ближе
        image_size_factor = np.sqrt((image_width * image_height) / (800 * 600))
        if image_size_factor > 1.5:
            weight_multiplier *= 1.2
            volume_multiplier *= 1.2
        
        estimated_weight = base_weight_g * weight_multiplier * (area_ratio / 0.3)
        estimated_volume = base_volume_ml * volume_multiplier * (area_ratio / 0.3)
        
        # Уверенность зависит от того, насколько четко определена область
        confidence = min(0.7 + (1 - abs(area_ratio - 0.3) * 2), 0.95)
        
        return {
            "weight_g": round(estimated_weight, 1),
            "volume_ml": round(estimated_volume, 1),
            "portion_size": portion_size,
            "confidence": round(confidence, 2)
        }

