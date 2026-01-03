"""
Датасет для обучения модели на данных из PostgreSQL
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import base64
from sqlalchemy.orm import Session
from typing import List, Tuple
import logging

# Импорт модели БД (импортируем здесь, чтобы избежать циклических импортов)
# Используем lazy import в методах для безопасности
logger = logging.getLogger(__name__)

class NutritionDataset(Dataset):
    """
    Датасет для обучения модели предсказания калорий/БЖУ.
    
    Загружает данные из PostgreSQL, где сохранены изображения
    и результаты ChatGPT (как целевые значения).
    """
    
    def __init__(self, db_session: Session, min_samples: int = 50):
        """
        Args:
            db_session: Сессия базы данных
            min_samples: Минимальное количество образцов для обучения
        """
        self.db_session = db_session
        self.samples = []
        
        # Загружаем данные из БД
        self._load_samples()
        
        if len(self.samples) < min_samples:
            logger.warning(
                f"Мало данных для обучения: {len(self.samples)} < {min_samples}. "
                f"Рекомендуется собрать больше данных."
            )
    
    def _load_samples(self):
        """Загружает образцы из базы данных"""
        from app.models import FoodRequest
        
        # Загружаем записи, где есть результаты ChatGPT
        records = self.db_session.query(FoodRequest).filter(
            FoodRequest.chatgpt_calories.isnot(None),
            FoodRequest.chatgpt_response_raw.isnot(None),
            FoodRequest.full_response.isnot(None)
        ).all()
        
        logger.info(f"Найдено {len(records)} записей с данными ChatGPT")
        
        for record in records:
            try:
                # Извлекаем изображение из full_response или используем путь
                # Пока сохраняем только метаданные, изображение нужно будет загружать отдельно
                # В реальности нужно сохранять изображения на диск или в БД
                
                # Целевые значения (ChatGPT)
                target = torch.tensor([
                    record.chatgpt_calories or 0,
                    record.chatgpt_proteins or 0,
                    record.chatgpt_fats or 0,
                    record.chatgpt_carbs or 0
                ], dtype=torch.float32)
                
                # Сохраняем ID записи для последующей загрузки изображения
                # В реальной реализации нужно сохранять изображения
                food_name = record.chatgpt_food_name
                if not food_name:
                    continue  # Пропускаем записи без названия еды
                
                self.samples.append({
                    'record_id': record.id,
                    'target': target,  # Для регрессии (калории/БЖУ)
                    'food_name': food_name,  # Для классификации (тип еды)
                })
                
            except Exception as e:
                logger.warning(f"Ошибка загрузки записи {record.id}: {e}")
                continue
        
        logger.info(f"Загружено {len(self.samples)} образцов для обучения")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Возвращает образец для обучения.
        
        Загружает изображение по пути из БД.
        """
        sample = self.samples[idx]
        
        # Загружаем изображение по пути из БД
        try:
            from app.image_storage import image_storage
            from app.models import FoodRequest  # Импорт нужен здесь
            
            # Получаем запись из БД
            record = self.db_session.query(FoodRequest).filter(
                FoodRequest.id == sample['record_id']
            ).first()
            
            if record and record.request_image_path:
                image = image_storage.load_image(record.request_image_path)
                return {
                    'image': image,
                    'target': sample['target'],
                    'food_name': sample.get('food_name', 'unknown'),
                    'record_id': sample['record_id']
                }
            else:
                # Если изображение не найдено, возвращаем только target
                logger.warning(f"Изображение не найдено для записи {sample['record_id']}")
                return {
                    'target': sample['target'],
                    'food_name': sample.get('food_name', 'unknown'),
                    'record_id': sample['record_id']
                }
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения для записи {sample['record_id']}: {e}")
            return {
                'target': sample['target'],
                'food_name': sample.get('food_name', 'unknown'),
                'record_id': sample['record_id']
            }
    
    def get_targets_only(self, idx):
        """Возвращает только целевые значения (для тестирования без изображений)"""
        return self.samples[idx]['target']

