import os
import shutil
import json
from pathlib import Path
import random
import numpy as np

class PowerlinesDatasetConverter:
    """
    Конвертер датасету для навчання YOLOv8 розпізнавати стовпи та кабелі
    Підтримує LabelMe JSON формат з полігонами та прямокутниками
    """
    
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Класи: 0 - tower (стовп), 1 - cable (кабель)
        self.classes = {
            'tower': 0,
            'cable': 1
        }
    
    def polygon_to_bbox(self, points):
        """
        Конвертує полігон у bounding box
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        return x_min, y_min, x_max, y_max
    
    def convert_json_to_yolo(self, json_path):
        """
        Конвертує LabelMe JSON анотації у формат YOLO
        Підтримує:
        - rectangle: прямокутники (2 точки)
        - polygon: багатокутники (N точок) - конвертуються у bbox
        
        Формат YOLO: class_id center_x center_y width height (нормалізовані 0-1)
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️  Помилка читання {json_path}: {e}")
            return []
        
        annotations = []
        
        # Отримуємо розміри зображення
        img_width = data.get('imageWidth', 1)
        img_height = data.get('imageHeight', 1)
        
        if img_width <= 0 or img_height <= 0:
            print(f"⚠️  Невірні розміри зображення у {json_path}")
            return []
        
        # Обробляємо кожен об'єкт
        for shape in data.get('shapes', []):
            label = shape.get('label', '').lower()
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')
            
            # Визначаємо клас
            class_id = None
            if 'tower' in label:
                class_id = self.classes['tower']
            elif 'cable' in label:
                class_id = self.classes['cable']
            else:
                # Пропускаємо невідомі класи
                continue
            
            if not points or len(points) < 2:
                continue
            
            # Обробка різних типів фігур
            if shape_type == 'rectangle' and len(points) == 2:
                # Прямокутник: 2 точки (верхній лівий, нижній правий)
                x1, y1 = points[0]
                x2, y2 = points[1]
            else:
                # Полігон або інша фігура: конвертуємо в bbox
                x1, y1, x2, y2 = self.polygon_to_bbox(points)
            
            # Конвертуємо у формат YOLO
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = abs(x2 - x1) / img_width
            height = abs(y2 - y1) / img_height
            
            # Обмежуємо значення між 0 та 1
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Перевірка мінімального розміру
            if width < 0.001 or height < 0.001:
                continue
            
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        return annotations
    
    def prepare_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Підготовка датасету: розподіл на train/val/test
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
        
        # Знаходимо всі зображення
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.source_dir.glob(ext)))
        
        if not image_files:
            print("⚠️  Не знайдено зображень у вихідній директорії!")
            print(f"   Шукали у: {self.source_dir}")
            return
        
        print(f"✅ Знайдено {len(image_files)} зображень")
        
        # Перемішуємо файли
        random.shuffle(image_files)
        
        # Розділяємо на підмножини
        total = len(image_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Створюємо директорії
        for split in ['train', 'val', 'test']:
            (self.target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Копіюємо файли
        print("\n📦 Конвертація датасету...")
        stats_train = self._copy_split(train_files, 'train')
        stats_val = self._copy_split(val_files, 'val')
        stats_test = self._copy_split(test_files, 'test')
        
        # Виводимо статистику
        print(f"\n{'='*60}")
        print(f"✅ Датасет підготовлено!")
        print(f"{'='*60}")
        print(f"\n📊 СТАТИСТИКА:")
        print(f"\nТренування:")
        print(f"  Зображень: {len(train_files)}")
        print(f"  Стовпів:   {stats_train['towers']}")
        print(f"  Кабелів:   {stats_train['cables']}")
        print(f"  Разом:     {stats_train['total']}")
        
        print(f"\nВалідація:")
        print(f"  Зображень: {len(val_files)}")
        print(f"  Стовпів:   {stats_val['towers']}")
        print(f"  Кабелів:   {stats_val['cables']}")
        print(f"  Разом:     {stats_val['total']}")
        
        print(f"\nТестування:")
        print(f"  Зображень: {len(test_files)}")
        print(f"  Стовпів:   {stats_test['towers']}")
        print(f"  Кабелів:   {stats_test['cables']}")
        print(f"  Разом:     {stats_test['total']}")
        
        total_objs = stats_train['total'] + stats_val['total'] + stats_test['total']
        total_towers = stats_train['towers'] + stats_val['towers'] + stats_test['towers']
        total_cables = stats_train['cables'] + stats_val['cables'] + stats_test['cables']
        
        print(f"\nВСЬОГО:")
        print(f"  Зображень: {total}")
        print(f"  Стовпів:   {total_towers}")
        print(f"  Кабелів:   {total_cables}")
        print(f"  Разом:     {total_objs}")
        print(f"{'='*60}\n")
    
    def _copy_split(self, files, split):
        """Копіює файли у відповідну підмножину та повертає статистику"""
        stats = {'towers': 0, 'cables': 0, 'total': 0}
        
        for img_file in files:
            # Копіюємо зображення
            target_img = self.target_dir / split / 'images' / img_file.name
            shutil.copy2(img_file, target_img)
            
            # Шукаємо відповідний файл анотацій
            label_file = img_file.with_suffix('.txt')
            json_file = img_file.with_suffix('.json')
            
            target_label = self.target_dir / split / 'labels' / img_file.with_suffix('.txt').name
            
            annotations = []
            
            if json_file.exists():
                # Конвертуємо JSON
                annotations = self.convert_json_to_yolo(json_file)
            elif label_file.exists():
                # Якщо є YOLO формат - просто копіюємо
                shutil.copy2(label_file, target_label)
                with open(label_file, 'r') as f:
                    annotations = f.readlines()
            
            # Підраховуємо статистику
            for ann in annotations:
                parts = ann.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 0:
                        stats['towers'] += 1
                    elif class_id == 1:
                        stats['cables'] += 1
                    stats['total'] += 1
            
            # Записуємо анотації
            if annotations and not label_file.exists():
                with open(target_label, 'w') as f:
                    f.write('\n'.join(annotations))
            elif not annotations:
                # Створюємо порожній файл (немає анотацій)
                target_label.touch()
        
        return stats

def main():
    """
    Приклад використання:
    Помістіть ваші зображення та JSON файли у папку 'raw_data'
    """
    
    print("="*60)
    print("🔌 ПІДГОТОВКА ДАТАСЕТУ ЛІНІЙ ЕЛЕКТРОПЕРЕДАЧ")
    print("="*60)
    
    # Шляхи
    source_dir = "raw_data"  # Папка з вихідними даними
    target_dir = "dataset"    # Папка для підготовленого датасету
    
    # Перевірка існування вихідної папки
    if not os.path.exists(source_dir):
        print(f"\n❌ Папка {source_dir}/ не знайдена!")
        print("\nСтворіть папку raw_data/ та помістіть туди:")
        print("  - Зображення (.jpg, .png)")
        print("  - JSON анотації (.json)")
        print("\nПриклад:")
        print("  raw_data/")
        print("  ├── 1_00186.jpg")
        print("  ├── 1_00186.json")
        print("  ├── photo2.jpg")
        print("  ├── photo2.json")
        print("  └── ...")
        return
    
    # Створюємо конвертер
    converter = PowerlinesDatasetConverter(source_dir, target_dir)
    
    # Підготовка датасету (80% train, 10% val, 10% test)
    converter.prepare_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print("\n📋 Наступні кроки:")
    print("1. Перевірте структуру датасету у папці 'dataset'")
    print("2. Візуалізуйте дані: python visualize_dataset.py")
    print("3. Запустіть навчання: python train.py")
    print("4. Результати будуть у папці 'runs/detect/powerlines_detector'\n")

if __name__ == "__main__":
    main()
