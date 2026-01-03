"""
Модель для дообучения: использует Food-101 для извлечения признаков
и обучает регрессионную модель для предсказания калорий/БЖУ
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class NutritionRegressionModel(nn.Module):
    """
    Модель для предсказания калорий и БЖУ на основе Food-101 embeddings.
    
    Использует предобученную Food-101 модель для извлечения признаков,
    и обучает поверх них регрессионную модель.
    """
    
    def __init__(self, feature_extractor_name: str = "google/vit-base-patch16-224", freeze_backbone: bool = True):
        """
        Args:
            feature_extractor_name: Имя модели для извлечения признаков
            freeze_backbone: Замораживать ли веса backbone (Food-101)
        """
        super().__init__()
        
        # Загружаем предобученную модель для извлечения признаков
        try:
            logger.info(f"Загрузка feature extractor: {feature_extractor_name}")
            self.feature_extractor = AutoModel.from_pretrained(
                feature_extractor_name,
                trust_remote_code=True
            )
            
            # Замораживаем веса backbone (Food-101)
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                logger.info("Backbone заморожен для fine-tuning")
            else:
                logger.info("Backbone разморожен (full fine-tuning)")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            # Fallback на более простую модель
            logger.info("Используется резервная модель...")
            self.feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224")
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        
        # Определяем размерность признаков
        # Для ViT это обычно hidden_size (768 для base)
        try:
            feature_dim = self.feature_extractor.config.hidden_size
        except:
            feature_dim = 768  # По умолчанию для ViT
        
        logger.info(f"Размерность признаков: {feature_dim}")
        
        # Регрессионные слои для предсказания калорий/БЖУ
        # Выход: [calories_per_100g, proteins_per_100g, fats_per_100g, carbs_per_100g]
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # 4 значения: calories, proteins, fats, carbs
        )
        
        # Инициализируем веса регрессора
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов регрессора"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_values):
        """
        Прямой проход модели.
        
        Args:
            pixel_values: Тензор пикселей изображения [batch, channels, height, width]
            
        Returns:
            Тензор предсказаний [batch, 4] где 4 = [calories, proteins, fats, carbs]
        """
        # Извлекаем признаки через предобученную модель
        outputs = self.feature_extractor(pixel_values=pixel_values)
        
        # Берем [CLS] токен (для ViT) или pooled output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Для ViT берем первый токен [CLS]
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # Fallback
            features = outputs.last_hidden_state.mean(dim=1)
        
        # Предсказываем калории и БЖУ
        nutrition = self.regressor(features)
        
        return nutrition
    
    def predict_nutrition(self, image, processor):
        """
        Удобный метод для предсказания на одном изображении.
        
        Args:
            image: PIL Image
            processor: AutoImageProcessor для предобработки
            
        Returns:
            dict с калориями и БЖУ
        """
        self.eval()
        with torch.no_grad():
            # Предобработка
            inputs = processor(image, return_tensors="pt")
            pixel_values = inputs['pixel_values']
            
            # Предсказание
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
                self.cuda()
            
            prediction = self.forward(pixel_values)
            prediction = prediction.cpu().numpy()[0]
            
            return {
                'calories_per_100g': max(0, float(prediction[0])),
                'proteins_per_100g': max(0, float(prediction[1])),
                'fats_per_100g': max(0, float(prediction[2])),
                'carbs_per_100g': max(0, float(prediction[3]))
            }
    
    def save_model(self, path: str):
        """Сохраняет только регрессор (backbone не сохраняем, он уже есть)"""
        torch.save({
            'regressor_state_dict': self.regressor.state_dict(),
            'feature_extractor_name': 'google/vit-base-patch16-224',
            'freeze_backbone': True
        }, path)
        logger.info(f"Модель сохранена: {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Загружает модель"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            feature_extractor_name=checkpoint.get('feature_extractor_name', 'google/vit-base-patch16-224'),
            freeze_backbone=checkpoint.get('freeze_backbone', True)
        )
        model.regressor.load_state_dict(checkpoint['regressor_state_dict'])
        model.to(device)
        logger.info(f"Модель загружена: {path}")
        return model

