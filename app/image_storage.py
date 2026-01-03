"""
Сервис для сохранения загруженных изображений на диск
"""
import os
from pathlib import Path
from PIL import Image
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ImageStorage:
    """Класс для сохранения изображений на диск"""
    
    def __init__(self, storage_dir: str = "uploads/images"):
        """
        Args:
            storage_dir: Директория для сохранения изображений
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image storage инициализирован: {self.storage_dir}")
    
    def save_image(self, image: Image.Image, prefix: str = "food") -> str:
        """
        Сохраняет изображение на диск.
        
        Args:
            image: PIL Image объект
            prefix: Префикс для имени файла
            
        Returns:
            Относительный путь к сохраненному изображению
        """
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
        
        # Полный путь
        file_path = self.storage_dir / filename
        
        # Сохраняем изображение
        try:
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Сохраняем с качеством 85% для экономии места
            image.save(file_path, "JPEG", quality=85, optimize=True)
            
            # Возвращаем относительный путь (для БД)
            # Используем os.path.relpath для надежного получения относительного пути
            relative_path = os.path.relpath(file_path.resolve(), Path.cwd().resolve())
            # Нормализуем путь - используем прямые слэши (более универсально для БД)
            relative_path = relative_path.replace('\\', '/')
            
            logger.info(f"Изображение сохранено: {relative_path}")
            return relative_path
            
        except Exception as e:
            logger.error(f"Ошибка сохранения изображения: {e}")
            raise
    
    def get_image_path(self, relative_path: str) -> Path:
        """
        Получает полный путь к изображению по относительному пути.
        
        Args:
            relative_path: Относительный путь из БД (может быть с прямыми или обратными слэшами)
        
        Returns:
            Полный путь к файлу
        """
        # Нормализуем путь - заменяем прямые слэши на обратные для Windows
        normalized_path = relative_path.replace('/', os.sep)
        path_obj = Path(normalized_path)
        return path_obj if path_obj.is_absolute() else Path.cwd() / path_obj
    
    def load_image(self, relative_path: str) -> Image.Image:
        """
        Загружает изображение по относительному пути.
        
        Args:
            relative_path: Относительный путь из БД
            
        Returns:
            PIL Image объект
        """
        file_path = self.get_image_path(relative_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {file_path}")
        return Image.open(file_path)
    
    def delete_image(self, relative_path: str) -> bool:
        """
        Удаляет изображение с диска.
        
        Args:
            relative_path: Относительный путь из БД
            
        Returns:
            True если удалено успешно
        """
        try:
            file_path = self.get_image_path(relative_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Изображение удалено: {relative_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка удаления изображения: {e}")
            return False

# Глобальный экземпляр
image_storage = ImageStorage()

