"""
Сервис для обучения модели предсказания калорий/БЖУ
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
from typing import Dict, Optional
from pathlib import Path

from app.training.nutrition_model import NutritionRegressionModel
from app.training.food_classifier_model import FoodClassifierModel
from app.database import SessionLocal

logger = logging.getLogger(__name__)

class TrainingService:
    """Сервис для обучения модели"""
    
    def __init__(self, model_save_dir: str = "models"):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None  # Будет инициализирован при первом использовании
        logger.info(f"Используется устройство: {self.device}")
    
    def train(
        self,
        epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Обучает модель на данных из БД.
        
        Args:
            epochs: Количество эпох
            batch_size: Размер батча
            learning_rate: Скорость обучения
            validation_split: Доля данных для валидации
            
        Returns:
            dict с метриками обучения
        """
        db = SessionLocal()
        try:
            # Импортируем здесь чтобы избежать circular imports
            from app.training.training_dataset import NutritionDataset
            
            # Создаем датасет
            dataset = NutritionDataset(db_session=db)
            
            if len(dataset) == 0:
                return {
                    "success": False,
                    "error": "Нет данных для обучения. Соберите данные через API."
                }
            
            logger.info(f"Начало обучения на {len(dataset)} образцах")
            
            # Собираем все уникальные названия еды для создания словаря классов
            # Нормализуем названия: приводим к одному регистру и убираем лишние пробелы
            def normalize_food_name(name: str) -> str:
                """Нормализует название еды: убирает лишние пробелы, приводит к единому формату"""
                if not name:
                    return name
                # Убираем лишние пробелы и приводим к строке
                normalized = ' '.join(name.strip().split())
                # Приводим первую букву к заглавной, остальные к строчным
                # Это поможет объединить "Оливье" и "оливье" в один класс
                if normalized:
                    normalized = normalized[0].upper() + normalized[1:].lower()
                return normalized
            
            # Собираем нормализованные названия
            normalized_to_original = {}  # Нормализованное -> оригинальное (берем первое встреченное)
            all_food_names_normalized = set()
            
            for sample in dataset.samples:
                if sample.get('food_name'):
                    original_name = sample['food_name']
                    normalized_name = normalize_food_name(original_name)
                    if normalized_name:
                        all_food_names_normalized.add(normalized_name)
                        # Сохраняем первое встреченное оригинальное название для нормализованного
                        if normalized_name not in normalized_to_original:
                            normalized_to_original[normalized_name] = original_name
            
            # Создаем словарь: нормализованное название еды -> индекс класса
            # Используем оригинальное название (первое встреченное) для индекса
            sorted_names = sorted(all_food_names_normalized)
            food_name_to_idx = {}
            idx_to_food_name = {}
            
            for idx, normalized_name in enumerate(sorted_names):
                # Используем оригинальное название (нормализованное для поиска)
                original_name = normalized_to_original[normalized_name]
                food_name_to_idx[normalized_name] = idx
                idx_to_food_name[str(idx)] = original_name  # Сохраняем оригинальное название
            
            num_classes = len(food_name_to_idx)
            
            logger.info(f"Найдено {num_classes} уникальных типов еды для классификации")
            logger.info(f"Примеры: {list(food_name_to_idx.keys())[:10]}")
            
            # Сохраняем словарь для использования при инференсе
            import json
            food_dict_path = self.model_save_dir / "food_name_dict.json"
            with open(food_dict_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'name_to_idx': food_name_to_idx,
                    'idx_to_name': idx_to_food_name
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Словарь названий еды сохранен: {food_dict_path}")
            
            # Разделяем на train/validation
            train_size = int((1 - validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Кастомная функция collate для обработки PIL изображений
            def collate_fn(batch):
                """Обрабатывает батч, преобразуя PIL изображения в тензоры"""
                from transformers import AutoImageProcessor
                
                # Инициализируем processor если еще не инициализирован
                if self.processor is None:
                    self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                
                # Разделяем элементы с изображениями и без
                images = []
                targets = []
                food_names = []
                record_ids = []
                
                for item in batch:
                    if 'image' in item and item['image'] is not None:
                        images.append(item['image'])
                        targets.append(item['target'])
                        food_name = item.get('food_name', 'unknown')
                        food_names.append(food_name)
                        record_ids.append(item.get('record_id'))
                
                if not images:
                    # Если нет изображений, возвращаем только targets
                    return {
                        'target': torch.stack(targets) if targets else None,
                        'food_name': food_names,
                        'record_id': record_ids
                    }
                
                # Обрабатываем изображения через processor
                pixel_values = self.processor(
                    images,
                    return_tensors="pt"
                )['pixel_values']
                
                # Преобразуем названия еды в индексы классов
                # Нормализуем названия перед поиском в словаре
                def normalize_food_name(name: str) -> str:
                    """Нормализует название еды для поиска в словаре"""
                    if not name:
                        return name
                    normalized = ' '.join(name.strip().split())
                    if normalized:
                        normalized = normalized[0].upper() + normalized[1:].lower()
                    return normalized
                
                food_class_indices = torch.tensor([
                    food_name_to_idx.get(normalize_food_name(name), 0) for name in food_names
                ], dtype=torch.long)
                
                return {
                    'image': pixel_values,
                    'target': torch.stack(targets),  # Для регрессии
                    'food_class': food_class_indices,  # Для классификации
                    'food_name': food_names,  # Для логирования
                    'record_id': record_ids
                }
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # 0 для Windows
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            # Создаем модели
            nutrition_model = NutritionRegressionModel(freeze_backbone=True)
            nutrition_model.to(self.device)
            
            food_classifier = FoodClassifierModel(freeze_backbone=True, num_classes=num_classes)
            food_classifier.to(self.device)
            
            # Оптимизаторы для обеих моделей
            nutrition_optimizer = torch.optim.Adam(
                nutrition_model.regressor.parameters(),
                lr=learning_rate,
                weight_decay=0.0001
            )
            
            classifier_optimizer = torch.optim.Adam(
                food_classifier.classifier.parameters(),
                lr=learning_rate,
                weight_decay=0.0001
            )
            
            # Функции потерь
            nutrition_criterion = nn.L1Loss()  # Для регрессии (калории/БЖУ)
            classifier_criterion = nn.CrossEntropyLoss()  # Для классификации (тип еды)
            
            # История обучения
            history = {
                'train_nutrition_loss': [],
                'train_classifier_loss': [],
                'train_total_loss': [],
                'val_nutrition_loss': [],
                'val_classifier_loss': [],
                'val_total_loss': []
            }
            
            best_val_total_loss = float('inf')
            
            # Цикл обучения
            for epoch in range(epochs):
                # Обучение
                nutrition_model.train()
                food_classifier.train()
                
                train_nutrition_loss = 0.0
                train_classifier_loss = 0.0
                train_batches = 0
                
                for batch in train_loader:
                    # Проверяем наличие данных (collate_fn уже обработал изображения)
                    if batch['target'] is None or 'image' not in batch or batch['image'] is None:
                        logger.warning("Изображения не найдены в батче, пропускаем")
                        continue
                    
                    targets = batch['target'].to(self.device)
                    food_classes = batch['food_class'].to(self.device)
                    pixel_values = batch['image'].to(self.device)
                    
                    # Предсказания через обе модели
                    nutrition_predictions = nutrition_model(pixel_values)
                    classifier_logits = food_classifier(pixel_values)
                    
                    # Потери
                    nutrition_loss = nutrition_criterion(nutrition_predictions, targets)
                    classifier_loss = classifier_criterion(classifier_logits, food_classes)
                    
                    # Комбинированная потеря (можно настроить веса)
                    total_loss = nutrition_loss + 0.5 * classifier_loss  # Классификация менее важна
                    
                    # Обратное распространение для обеих моделей
                    nutrition_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    
                    total_loss.backward()
                    
                    nutrition_optimizer.step()
                    classifier_optimizer.step()
                    
                    train_nutrition_loss += nutrition_loss.item()
                    train_classifier_loss += classifier_loss.item()
                    train_batches += 1
                
                if train_batches > 0:
                    train_nutrition_loss /= train_batches
                    train_classifier_loss /= train_batches
                    train_total_loss = train_nutrition_loss + 0.5 * train_classifier_loss
                else:
                    train_nutrition_loss = 0.0
                    train_classifier_loss = 0.0
                    train_total_loss = 0.0
                
                # Валидация
                nutrition_model.eval()
                food_classifier.eval()
                
                val_nutrition_loss = 0.0
                val_classifier_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Проверяем наличие данных (collate_fn уже обработал изображения)
                        if batch['target'] is None or 'image' not in batch or batch['image'] is None:
                            continue
                        
                        targets = batch['target'].to(self.device)
                        food_classes = batch['food_class'].to(self.device)
                        pixel_values = batch['image'].to(self.device)
                        
                        nutrition_predictions = nutrition_model(pixel_values)
                        classifier_logits = food_classifier(pixel_values)
                        
                        nutrition_loss = nutrition_criterion(nutrition_predictions, targets)
                        classifier_loss = classifier_criterion(classifier_logits, food_classes)
                        
                        val_nutrition_loss += nutrition_loss.item()
                        val_classifier_loss += classifier_loss.item()
                        val_batches += 1
                
                if val_batches > 0:
                    val_nutrition_loss /= val_batches
                    val_classifier_loss /= val_batches
                    val_total_loss = val_nutrition_loss + 0.5 * val_classifier_loss
                else:
                    val_nutrition_loss = 0.0
                    val_classifier_loss = 0.0
                    val_total_loss = 0.0
                
                history['train_nutrition_loss'].append(train_nutrition_loss)
                history['train_classifier_loss'].append(train_classifier_loss)
                history['train_total_loss'].append(train_total_loss)
                history['val_nutrition_loss'].append(val_nutrition_loss)
                history['val_classifier_loss'].append(val_classifier_loss)
                history['val_total_loss'].append(val_total_loss)
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: Nutrition={train_nutrition_loss:.4f}, Classifier={train_classifier_loss:.4f}, Total={train_total_loss:.4f} | "
                    f"Val: Nutrition={val_nutrition_loss:.4f}, Classifier={val_classifier_loss:.4f}, Total={val_total_loss:.4f}"
                )
                
                # Сохраняем лучшие модели
                if val_total_loss < best_val_total_loss:
                    best_val_total_loss = val_total_loss
                    
                    nutrition_model_path = self.model_save_dir / "nutrition_model_best.pth"
                    nutrition_model.save_model(str(nutrition_model_path))
                    
                    classifier_model_path = self.model_save_dir / "food_classifier_best.pth"
                    food_classifier.save_model(str(classifier_model_path))
                    
                    logger.info(f"Лучшие модели сохранены (Val Loss: {val_total_loss:.4f})")
            
            # Сохраняем финальные модели
            final_nutrition_path = self.model_save_dir / "nutrition_model_final.pth"
            nutrition_model.save_model(str(final_nutrition_path))
            
            final_classifier_path = self.model_save_dir / "food_classifier_final.pth"
            food_classifier.save_model(str(final_classifier_path))
            
            # Помечаем все использованные записи как training_used = True
            from app.models import FoodRequest
            used_record_ids = set()
            for sample in dataset.samples:
                used_record_ids.add(sample['record_id'])
            
            # Обновляем записи в БД
            updated_count = db.query(FoodRequest).filter(
                FoodRequest.id.in_(used_record_ids)
            ).update(
                {FoodRequest.training_used: True},
                synchronize_session=False
            )
            db.commit()
            logger.info(f"Помечено {updated_count} записей как использованные для обучения (training_used=True)")
            
            return {
                "success": True,
                "epochs": epochs,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "num_food_classes": num_classes,
                "final_train_nutrition_loss": history['train_nutrition_loss'][-1],
                "final_train_classifier_loss": history['train_classifier_loss'][-1],
                "final_train_total_loss": history['train_total_loss'][-1],
                "final_val_nutrition_loss": history['val_nutrition_loss'][-1],
                "final_val_classifier_loss": history['val_classifier_loss'][-1],
                "final_val_total_loss": history['val_total_loss'][-1],
                "best_val_total_loss": best_val_total_loss,
                "nutrition_model_path": str(final_nutrition_path),
                "classifier_model_path": str(final_classifier_path),
                "food_dict_path": str(food_dict_path)
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            db.close()
    
    def get_model_path(self) -> Optional[str]:
        """Возвращает путь к лучшей модели"""
        best_model = self.model_save_dir / "nutrition_model_best.pth"
        if best_model.exists():
            return str(best_model)
        return None

# Глобальный экземпляр
training_service = TrainingService()

