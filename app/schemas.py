from pydantic import BaseModel
from typing import Optional, List

class NutritionInfo(BaseModel):
    calories: float
    proteins: float  # граммы
    fats: float  # граммы
    carbs: float  # граммы

class PortionEstimate(BaseModel):
    weight_g: Optional[float] = None
    volume_ml: Optional[float] = None
    estimated_portion_size: str  # "small", "medium", "large"

# Новые схемы для дополнительной информации
class IngredientInfo(BaseModel):
    name: str
    calories_per_100g: float
    proteins_per_100g: float
    fats_per_100g: float
    carbs_per_100g: float
    description: Optional[str] = None  # Краткое описание ингредиента
    weight_in_portion_g: Optional[float] = None  # Вес ингредиента в порции (граммы)
    calories_in_portion: Optional[float] = None  # Калории ингредиента в порции
    proteins_in_portion: Optional[float] = None  # Белки ингредиента в порции (граммы)
    fats_in_portion: Optional[float] = None  # Жиры ингредиента в порции (граммы)
    carbs_in_portion: Optional[float] = None  # Углеводы ингредиента в порции (граммы)

class Recommendation(BaseModel):
    type: str  # "tip" или "alternative"
    title: str
    description: str
    calories_saved: Optional[float] = None  # Для альтернатив: сколько ккал экономится

class MicronutrientInfo(BaseModel):
    name: str
    amount: float  # Количество в блюде (единица измерения зависит от типа)
    unit: str  # Единица измерения (мг, мкг, г и т.д.)
    daily_value: float  # Суточная норма
    percent_of_daily_value: float  # Процент от суточной нормы (вычисляется на сервере)

class FoodItem(BaseModel):
    name: str
    confidence: float
    nutrition_per_100g: NutritionInfo
    portion_estimate: PortionEstimate
    total_nutrition: NutritionInfo
    ingredients: Optional[List[IngredientInfo]] = None  # Детальная информация об ингредиентах
    recommendations: Optional[List[Recommendation]] = None  # Советы и альтернативы
    micronutrients: Optional[List[MicronutrientInfo]] = None  # Микронутриенты и витамины

class FoodRecognitionResponse(BaseModel):
    recognized_foods: List[FoodItem]
    message: str
    processing_time_seconds: Optional[float] = None

class FoodRequestCreate(BaseModel):
    recognized_food: Optional[str] = None
    calories: Optional[float] = None
    proteins: Optional[float] = None
    fats: Optional[float] = None
    carbs: Optional[float] = None
    estimated_weight_g: Optional[float] = None
    estimated_volume_ml: Optional[float] = None
    confidence: Optional[float] = None
    full_response: Optional[dict] = None

# Схемы для программы питания
class RecipeInfo(BaseModel):
    """Информация о рецепте для программы питания"""
    uuid: str
    name: str
    category: str  # Категория блюда (завтрак, обед, ужин, салат и т.д.)
    calories: float  # КБЖУ в 1 порции
    proteins: float
    fats: float
    carbs: float

class MealPlanRequest(BaseModel):
    """Запрос на создание программы питания"""
    meals_per_day: int  # Количество приемов пищи в день (минимум 3)
    days_count: int  # Количество дней
    target_nutrition: NutritionInfo  # Целевые КБЖУ
    allowed_recipes: List[RecipeInfo]  # Список доступных рецептов

class MealItem(BaseModel):
    """Блюдо в приеме пищи"""
    uuid: str
    name: str
    portions: int = 1  # Количество порций этого блюда (по умолчанию 1)

class Meal(BaseModel):
    """Прием пищи"""
    category: str  # Название категории (завтрак, обед, ужин и т.д.)
    meals: List[MealItem]  # Список блюд (может быть несколько, если объединены)

class DayPlan(BaseModel):
    """План питания на день"""
    day_number: int
    target_nutrition: NutritionInfo  # Целевые КБЖУ
    actual_nutrition: NutritionInfo  # Фактическое КБЖУ из программы
    meals: List[Meal]  # Список приемов пищи

class MealPlanResponse(BaseModel):
    """Ответ с программой питания"""
    days: List[DayPlan]
    message: str
    processing_time_seconds: Optional[float] = None

