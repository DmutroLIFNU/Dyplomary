from ultralytics import YOLO
import os

def train():
    """
    Навчання моделі YOLOv8 для розпізнавання стовпів та кабелів ліній електропередач
    Training YOLOv8 model for powerlines tower and cable detection
    """
    
    # Завантаження попередньо навченої моделі YOLOv8 nano
    model = YOLO("yolov8n.pt")
    
    # Параметри навчання
    results = model.train(
        data="dataset/powerlines.yaml",  # Шлях до конфігурації датасету
        epochs=100,                       # Кількість епох навчання
        imgsz=640,                        # Розмір зображень
        batch=4,                          # Розмір батчу
        device="cpu",                     # Пристрій для навчання (cpu або cuda)
        
        # Додаткові параметри для покращення навчання
        patience=50,                      # Зупинка при відсутності покращень
        save=True,                        # Зберігати чекпоінти
        project="runs/detect",            # Директорія для збереження результатів
        name="powerlines_detector",       # Назва експерименту
        exist_ok=True,                    # Перезаписувати існуючі результати
        
        # Аугментація даних
        hsv_h=0.015,                      # Варіація відтінку
        hsv_s=0.7,                        # Варіація насиченості
        hsv_v=0.4,                        # Варіація яскравості
        degrees=0.0,                      # Ротація (0 для прямих об'єктів)
        translate=0.1,                    # Зсув
        scale=0.5,                        # Масштабування
        flipud=0.0,                       # Вертикальний фліп (не потрібен)
        fliplr=0.5,                       # Горизонтальний фліп
        mosaic=1.0,                       # Mosaic аугментація
        
        # Оптимізація для детекції дротів та стовпів
        lr0=0.01,                         # Початковий learning rate
        lrf=0.01,                         # Фінальний learning rate
        momentum=0.937,                   # Моментум
        weight_decay=0.0005,              # Регуляризація
        warmup_epochs=3.0,                # Warmup епохи
        warmup_momentum=0.8,              # Warmup моментум
        
        # Втрати та метрики
        box=7.5,                          # Вага box loss
        cls=0.5,                          # Вага classification loss
        dfl=1.5,                          # Вага distribution focal loss
    )
    
    # Валідація моделі після навчання
    metrics = model.val()
    
    print("\n" + "="*50)
    print("Навчання завершено! / Training completed!")
    print("="*50)
    print(f"Модель збережено у: runs/detect/powerlines_detector/weights/best.pt")
    print(f"Точність (mAP50): {metrics.box.map50:.4f}")
    print(f"Точність (mAP50-95): {metrics.box.map:.4f}")
    
    return results, metrics

def predict(model_path="runs/detect/powerlines_detector/weights/best.pt", 
            source="dataset/test/images"):
    """
    Запуск предикції на тестових зображеннях
    Run prediction on test images
    """
    model = YOLO(model_path)
    results = model.predict(
        source=source,
        conf=0.25,          # Мінімальна впевненість
        iou=0.45,           # IoU для NMS
        save=True,          # Зберігати результати
        save_txt=True,      # Зберігати анотації
        save_conf=True,     # Зберігати впевненість
        project="runs/predict",
        name="powerlines_results",
        exist_ok=True
    )
    return results

if __name__ == "__main__":
    # Навчання моделі
    train_results, val_metrics = train()
    
    # Опціонально: запуск предикції на тестових даних
    # predict()
