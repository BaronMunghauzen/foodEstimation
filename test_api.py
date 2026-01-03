"""
Скрипт для тестирования API распознавания еды
Отправляет все фотографии из указанной папки
"""
import os
import requests
from pathlib import Path

# Настройки
API_URL = "http://127.0.0.1:8000/recognize-food"
API_TOKEN = "secret"
PHOTOS_FOLDER = r"C:\Users\magla\Downloads\фото еды"

# Поддерживаемые форматы изображений
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

def get_image_files(folder_path):
    """Получить список всех изображений в папке"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Ошибка: папка {folder_path} не существует!")
        return []
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)
    
    return sorted(image_files)

def send_request(image_path):
    """Отправить запрос с изображением"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            headers = {
                'Authorization': f'Bearer {API_TOKEN}'
            }
            
            print(f"\n📸 Отправка: {image_path.name}")
            response = requests.post(API_URL, files=files, headers=headers, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                food_name = result.get('recognized_foods', [{}])[0].get('name', 'unknown')
                confidence = result.get('recognized_foods', [{}])[0].get('confidence', 0)
                calories = result.get('recognized_foods', [{}])[0].get('total_nutrition', {}).get('calories', 0)
                processing_time = result.get('processing_time_seconds', 0)
                
                print(f"✅ Успешно: {food_name} (уверенность: {confidence:.1%}, калории: {calories:.1}, время: {processing_time:.2f}с)")
                return True
            else:
                print(f"❌ Ошибка {response.status_code}: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

def main():
    print(f"🔍 Поиск изображений в: {PHOTOS_FOLDER}")
    image_files = get_image_files(PHOTOS_FOLDER)
    
    if not image_files:
        print("❌ Изображения не найдены!")
        return
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    print(f"🌐 API URL: {API_URL}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] ", end="")
        if send_request(image_path):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Итого: успешно {success_count}, ошибок {fail_count}")

if __name__ == "__main__":
    main()

