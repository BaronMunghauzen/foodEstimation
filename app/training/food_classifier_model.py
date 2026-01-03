"""
Модель для классификации типов еды
Использует ViT для извлечения признаков и обучает классификатор на основе данных ChatGPT
"""
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)

class FoodClassifierModel(nn.Module):
    """
    Модель для классификации типов еды на основе изображений.
    
    Использует предобученную ViT модель для извлечения признаков,
    и обучает классификатор поверх них.
    """
    
    def __init__(self, feature_extractor_name: str = "google/vit-base-patch16-224", freeze_backbone: bool = True, num_classes: int = None):
        """
        Args:
            feature_extractor_name: Имя модели для извлечения признаков
            freeze_backbone: Замораживать ли веса backbone
            num_classes: Количество классов (если None, будет определено автоматически)
        """
        super().__init__()
        
        # Загружаем предобученную модель для извлечения признаков
        try:
            logger.info(f"Загрузка feature extractor для классификатора: {feature_extractor_name}")
            self.feature_extractor = AutoModel.from_pretrained(
                feature_extractor_name,
                trust_remote_code=True
            )
            
            # Замораживаем веса backbone
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                logger.info("Backbone заморожен для fine-tuning")
            else:
                logger.info("Backbone разморожен (full fine-tuning)")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.info("Используется резервная модель...")
            self.feature_extractor = AutoModel.from_pretrained("google/vit-base-patch16-224")
            if freeze_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        
        # Определяем размерность признаков
        try:
            feature_dim = self.feature_extractor.config.hidden_size
        except:
            feature_dim = 768  # По умолчанию для ViT
        
        logger.info(f"Размерность признаков: {feature_dim}")
        
        # Классификатор для типов еды
        # Если num_classes не указан, используем большое число (будет определено при обучении)
        self.num_classes = num_classes or 1000  # Временное значение, будет обновлено
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.num_classes)
        )
        
        # Инициализируем веса классификатора
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов классификатора"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_num_classes(self, num_classes: int):
        """Устанавливает количество классов и пересоздает последний слой"""
        if num_classes != self.num_classes:
            self.num_classes = num_classes
            # Пересоздаем последний слой
            old_classifier = self.classifier
            self.classifier = nn.Sequential(
                *list(old_classifier.children())[:-1],  # Все слои кроме последнего
                nn.Linear(128, num_classes)
            )
            self.classifier.to(next(old_classifier.parameters()).device)
            logger.info(f"Количество классов обновлено: {num_classes}")
    
    def forward(self, pixel_values):
        """
        Прямой проход модели.
        
        Args:
            pixel_values: Тензор пикселей изображения [batch, channels, height, width]
            
        Returns:
            Тензор предсказаний [batch, num_classes] - логиты для каждого класса
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
        
        # Классифицируем тип еды
        logits = self.classifier(features)
        
        return logits
    
    def save_model(self, path: str):
        """Сохраняет модель"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'feature_extractor_name': 'google/vit-base-patch16-224',
            'freeze_backbone': True,
            'num_classes': self.num_classes,
            'classifier_state_dict': self.classifier.state_dict(),
        }, path)
        
        logger.info(f"Модель классификатора сохранена: {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Загружает модель"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            feature_extractor_name=checkpoint.get('feature_extractor_name', 'google/vit-base-patch16-224'),
            freeze_backbone=checkpoint.get('freeze_backbone', True),
            num_classes=checkpoint.get('num_classes', 1000)
        )
        model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        model.to(device)
        logger.info(f"Модель классификатора загружена: {path}")
        return model

