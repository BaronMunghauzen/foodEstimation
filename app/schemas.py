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

class FoodItem(BaseModel):
    name: str
    confidence: float
    nutrition_per_100g: NutritionInfo
    portion_estimate: PortionEstimate
    total_nutrition: NutritionInfo

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

