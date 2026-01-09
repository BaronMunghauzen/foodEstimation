from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from app.database import Base

class FoodRequest(Base):
    __tablename__ = "food_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Результаты вашей модели
    recognized_food = Column(String, nullable=True)
    calories = Column(Float, nullable=True)
    proteins = Column(Float, nullable=True)
    fats = Column(Float, nullable=True)
    carbs = Column(Float, nullable=True)
    estimated_weight_g = Column(Float, nullable=True)
    estimated_volume_ml = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Результаты ChatGPT
    chatgpt_food_name = Column(String, nullable=True)
    chatgpt_calories = Column(Float, nullable=True)
    chatgpt_proteins = Column(Float, nullable=True)
    chatgpt_fats = Column(Float, nullable=True)
    chatgpt_carbs = Column(Float, nullable=True)
    chatgpt_confidence = Column(Float, nullable=True)
    chatgpt_response_raw = Column(JSON, nullable=True)
    chatgpt_ingredients = Column(JSON, nullable=True)  # Массив ингредиентов с КБЖУ
    chatgpt_recommendations = Column(JSON, nullable=True)  # Массив советов и альтернатив
    chatgpt_micronutrients = Column(JSON, nullable=True)  # Массив микронутриентов и витаминов
    
    # Разница между моделями
    difference_calories = Column(Float, nullable=True)
    difference_proteins = Column(Float, nullable=True)
    difference_fats = Column(Float, nullable=True)
    difference_carbs = Column(Float, nullable=True)
    
    # Метаданные
    request_image_path = Column(String, nullable=True)  # Путь к сохраненному изображению
    full_response = Column(JSON, nullable=True)
    client_ip = Column(String, nullable=True)
    training_used = Column(Boolean, default=False)  # Использовано ли для обучения

class MealPlan(Base):
    __tablename__ = "meal_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Параметры запроса
    meals_per_day = Column(Integer, nullable=False)
    days_count = Column(Integer, nullable=False)
    target_calories = Column(Float, nullable=False)
    target_proteins = Column(Float, nullable=False)
    target_fats = Column(Float, nullable=False)
    target_carbs = Column(Float, nullable=False)
    
    # Данные запроса (JSON)
    allowed_recipes = Column(JSON, nullable=False)  # Список рецептов из запроса
    
    # Ответ ChatGPT (полный JSON)
    chatgpt_response_raw = Column(JSON, nullable=False)  # Полный JSON ответ от ChatGPT
    
    # Метаданные
    client_ip = Column(String, nullable=True)

