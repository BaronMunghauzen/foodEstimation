"""
База данных справочных значений калорий и БЖУ для различных блюд.
Используется как fallback при низкой уверенности модели.
"""

# Формат: название_блюда: {
#     "calories_per_100g": калории,
#     "proteins_per_100g": белки в граммах,
#     "fats_per_100g": жиры в граммах,
#     "carbs_per_100g": углеводы в граммах
# }

NUTRITION_DATABASE = {
    "pizza": {
        "calories_per_100g": 266,
        "proteins_per_100g": 12.0,
        "fats_per_100g": 10.0,
        "carbs_per_100g": 33.0
    },
    "burger": {
        "calories_per_100g": 295,
        "proteins_per_100g": 17.0,
        "fats_per_100g": 14.0,
        "carbs_per_100g": 25.0
    },
    "pasta": {
        "calories_per_100g": 131,
        "proteins_per_100g": 5.0,
        "fats_per_100g": 1.1,
        "carbs_per_100g": 25.0
    },
    "salad": {
        "calories_per_100g": 20,
        "proteins_per_100g": 1.5,
        "fats_per_100g": 0.2,
        "carbs_per_100g": 4.0
    },
    "soup": {
        "calories_per_100g": 50,
        "proteins_per_100g": 2.5,
        "fats_per_100g": 1.5,
        "carbs_per_100g": 6.0
    },
    "rice": {
        "calories_per_100g": 130,
        "proteins_per_100g": 2.7,
        "fats_per_100g": 0.3,
        "carbs_per_100g": 28.0
    },
    "chicken": {
        "calories_per_100g": 165,
        "proteins_per_100g": 31.0,
        "fats_per_100g": 3.6,
        "carbs_per_100g": 0.0
    },
    "fish": {
        "calories_per_100g": 206,
        "proteins_per_100g": 22.0,
        "fats_per_100g": 12.0,
        "carbs_per_100g": 0.0
    },
    "bread": {
        "calories_per_100g": 265,
        "proteins_per_100g": 9.0,
        "fats_per_100g": 3.2,
        "carbs_per_100g": 49.0
    },
    "apple": {
        "calories_per_100g": 52,
        "proteins_per_100g": 0.3,
        "fats_per_100g": 0.2,
        "carbs_per_100g": 14.0
    },
    "banana": {
        "calories_per_100g": 89,
        "proteins_per_100g": 1.1,
        "fats_per_100g": 0.3,
        "carbs_per_100g": 23.0
    },
    "sandwich": {
        "calories_per_100g": 250,
        "proteins_per_100g": 10.0,
        "fats_per_100g": 8.0,
        "carbs_per_100g": 35.0
    },
    "fries": {
        "calories_per_100g": 365,
        "proteins_per_100g": 4.0,
        "fats_per_100g": 17.0,
        "carbs_per_100g": 48.0
    },
    "steak": {
        "calories_per_100g": 271,
        "proteins_per_100g": 25.0,
        "fats_per_100g": 19.0,
        "carbs_per_100g": 0.0
    },
    "sushi": {
        "calories_per_100g": 150,
        "proteins_per_100g": 6.0,
        "fats_per_100g": 1.0,
        "carbs_per_100g": 28.0
    },
    "cake": {
        "calories_per_100g": 320,
        "proteins_per_100g": 4.0,
        "fats_per_100g": 12.0,
        "carbs_per_100g": 50.0
    },
    "ice_cream": {
        "calories_per_100g": 207,
        "proteins_per_100g": 3.5,
        "fats_per_100g": 11.0,
        "carbs_per_100g": 24.0
    },
    "coffee": {
        "calories_per_100g": 2,
        "proteins_per_100g": 0.1,
        "fats_per_100g": 0.0,
        "carbs_per_100g": 0.0
    },
    "eggs": {
        "calories_per_100g": 155,
        "proteins_per_100g": 13.0,
        "fats_per_100g": 11.0,
        "carbs_per_100g": 1.1
    },
    "cheese": {
        "calories_per_100g": 402,
        "proteins_per_100g": 25.0,
        "fats_per_100g": 33.0,
        "carbs_per_100g": 1.3
    }
}

def get_nutrition_fallback(food_name: str) -> dict:
    """
    Получить справочные значения калорий и БЖУ для блюда.
    Используется как fallback при низкой уверенности модели.
    
    Args:
        food_name: Название блюда
        
    Returns:
        dict с ключами: calories_per_100g, proteins_per_100g, 
        fats_per_100g, carbs_per_100g
    """
    food_lower = food_name.lower().replace(" ", "_")
    
    # Поиск точного совпадения
    if food_lower in NUTRITION_DATABASE:
        return NUTRITION_DATABASE[food_lower].copy()
    
    # Поиск частичного совпадения
    for key, nutrition in NUTRITION_DATABASE.items():
        if key in food_lower or food_lower in key:
            return nutrition.copy()
    
    # Значения по умолчанию для неизвестных блюд (средние значения)
    return {
        "calories_per_100g": 200.0,
        "proteins_per_100g": 10.0,
        "fats_per_100g": 8.0,
        "carbs_per_100g": 20.0
    }

