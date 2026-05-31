from ultralytics import YOLO
import cv2
import os
from pathlib import Path

class PowerlinesDetector:
    """
    Детектор стовпів та кабелів ліній електропередач
    """
    
    def __init__(self, model_path="runs/detect/powerlines_detector/weights/best.pt"):
        """
        Ініціалізація детектора
        
        Args:
            model_path: Шлях до навченої моделі
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не знайдено: {model_path}")
        
        self.model = YOLO(model_path)
        self.class_names = {0: 'tower', 1: 'cable'}
        self.colors = {
            0: (255, 0, 0),    # Червоний для tower
            1: (0, 255, 255)   # Жовтий для cable
        }
    
    def detect_image(self, image_path, conf=0.25, iou=0.45, save=True, output_dir="detections"):
        """
        Детекція на одному зображенні
        
        Args:
            image_path: Шлях до зображення
            conf: Мінімальний поріг впевненості (0-1)
            iou: Поріг IoU для NMS
            save: Зберегти результат
            output_dir: Папка для збереження
        
        Returns:
            Результати детекції
        """
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=save,
            project=output_dir,
            name="results",
            exist_ok=True
        )
        
        return results[0]
    
    def detect_folder(self, folder_path, conf=0.25, iou=0.45, output_dir="detections"):
        """
        Детекція на всіх зображеннях у папці
        
        Args:
            folder_path: Шлях до папки з зображеннями
            conf: Мінімальний поріг впевненості
            iou: Поріг IoU для NMS
            output_dir: Папка для збереження
        
        Returns:
            Список результатів
        """
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        images = [f for f in folder.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"⚠️  Зображень не знайдено у {folder_path}")
            return []
        
        print(f"🔍 Обробка {len(images)} зображень...")
        
        results = self.model.predict(
            source=folder_path,
            conf=conf,
            iou=iou,
            save=True,
            project=output_dir,
            name="batch_results",
            exist_ok=True
        )
        
        return results
    
    def detect_video(self, video_path, conf=0.25, iou=0.45, output_dir="detections"):
        """
        Детекція на відео
        
        Args:
            video_path: Шлях до відео файлу
            conf: Мінімальний поріг впевненості
            iou: Поріг IoU для NMS
            output_dir: Папка для збереження
        
        Returns:
            Результати детекції
        """
        results = self.model.predict(
            source=video_path,
            conf=conf,
            iou=iou,
            save=True,
            project=output_dir,
            name="video_results",
            exist_ok=True,
            stream=True  # Потокова обробка для відео
        )
        
        return results
    
    def annotate_image(self, image_path, output_path, conf=0.25):
        """
        Анотування зображення з детальною інформацією
        
        Args:
            image_path: Шлях до вхідного зображення
            output_path: Шлях до вихідного зображення
            conf: Мінімальний поріг впевненості
        """
        # Завантаження зображення
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не вдалося завантажити зображення: {image_path}")
        
        # Детекція
        results = self.model.predict(source=image_path, conf=conf, verbose=False)
        
        # Обробка результатів
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Отримання інформації
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                color = self.colors.get(class_id, (255, 255, 255))
                
                # Малювання bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Підпис
                label = f"{class_name}: {confidence:.2f}"
                
                # Розмір тексту
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Фон для тексту
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Текст
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        # Збереження
        cv2.imwrite(output_path, image)
        print(f"✅ Збережено: {output_path}")
    
    def get_detection_stats(self, results):
        """
        Отримання статистики детекцій
        
        Args:
            results: Результати детекції
        
        Returns:
            Словник зі статистикою
        """
        if not hasattr(results, 'boxes'):
            results = results[0]
        
        boxes = results.boxes
        stats = {
            'total': len(boxes),
            'towers': 0,
            'cables': 0,
            'detections': []
        }
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            if class_id == 0:
                stats['towers'] += 1
            elif class_id == 1:
                stats['cables'] += 1
            
            stats['detections'].append({
                'class': class_name,
                'confidence': confidence,
                'bbox': coords
            })
        
        return stats

def main():
    """
    Приклад використання детектора
    """
    print("🔌 Детектор ліній електропередач")
    print("=" * 50)
    
    # Ініціалізація детектора
    try:
        detector = PowerlinesDetector("runs/detect/powerlines_detector/weights/best.pt")
        print("✅ Модель завантажено")
    except FileNotFoundError:
        print("❌ Модель не знайдено!")
        print("   Спочатку навчіть модель: python train.py")
        return
    
    # Приклад 1: Детекція на одному зображенні
    print("\n📸 Приклад 1: Детекція на зображенні")
    image_path = "dataset/test/images/example.jpg"
    
    if os.path.exists(image_path):
        result = detector.detect_image(image_path, conf=0.25)
        stats = detector.get_detection_stats(result)
        
        print(f"Знайдено об'єктів: {stats['total']}")
        print(f"  - Стовпів: {stats['towers']}")
        print(f"  - Кабелів: {stats['cables']}")
        
        # Детальні результати
        for i, det in enumerate(stats['detections'], 1):
            print(f"  {i}. {det['class']}: {det['confidence']:.2%}")
    else:
        print(f"⚠️  Зображення не знайдено: {image_path}")
    
    # Приклад 2: Детекція на папці
    print("\n📁 Приклад 2: Детекція на папці")
    folder_path = "dataset/test/images"
    
    if os.path.exists(folder_path):
        results = detector.detect_folder(folder_path, conf=0.25)
        print(f"✅ Оброблено {len(results)} зображень")
    else:
        print(f"⚠️  Папка не знайдена: {folder_path}")
    
    # Приклад 3: Анотування з детальною інформацією
    print("\n🖊️  Приклад 3: Детальне анотування")
    if os.path.exists(image_path):
        output_path = "annotated_example.jpg"
        detector.annotate_image(image_path, output_path, conf=0.25)
    
    print("\n" + "=" * 50)
    print("✅ Завершено!")

if __name__ == "__main__":
    main()
