#!/usr/bin/env python3
"""
Головний скрипт для роботи з детектором ліній електропередач
Main script for powerlines detector
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Перевірка встановлених залежностей"""
    try:
        import ultralytics
        import cv2
        import torch
        print("✅ Всі залежності встановлені")
        return True
    except ImportError as e:
        print(f"❌ Відсутня залежність: {e}")
        print("Встановіть залежності: pip install -r requirements.txt")
        return False

def setup_project():
    """Початкове налаштування проекту"""
    print("🔧 Налаштування проекту...")
    
    # Створення необхідних директорій
    directories = [
        "dataset/train/images",
        "dataset/train/labels",
        "dataset/val/images",
        "dataset/val/labels",
        "dataset/test/images",
        "dataset/test/labels",
        "raw_data",
        "runs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Проект налаштовано")
    print("\nНаступні кроки:")
    print("1. Помістіть ваші зображення та анотації у папку 'raw_data/'")
    print("2. Запустіть: python main.py prepare")
    print("3. Запустіть: python main.py train")

def prepare_dataset():
    """Підготовка датасету"""
    print("📊 Підготовка датасету...")
    
    from prepare_dataset import PowerlinesDatasetConverter
    
    converter = PowerlinesDatasetConverter("raw_data", "dataset")
    converter.prepare_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    print("\n✅ Датасет підготовлено!")
    print("Запустіть навчання: python main.py train")

def visualize_dataset():
    """Візуалізація датасету"""
    print("🖼️  Візуалізація датасету...")
    
    from visualize_dataset import DatasetVisualizer
    
    visualizer = DatasetVisualizer("dataset")
    visualizer.print_statistics()
    
    visualizer.visualize_batch(
        split="train",
        num_images=9,
        save_path="dataset_train_samples.png"
    )
    
    print("\n✅ Візуалізацію завершено!")

def train_model(epochs=None, batch=None, device=None):
    """Навчання моделі"""
    print("🚀 Початок навчання...")
    
    from train import train
    
    # Тут можна додати параметри, якщо потрібно
    results, metrics = train()
    
    print("\n✅ Навчання завершено!")
    print("Модель збережено у: runs/detect/powerlines_detector/weights/best.pt")
    print("\nЗапустіть детекцію: python main.py detect <шлях_до_зображення>")

def detect_image(image_path, model_path=None, conf=0.25):
    """Детекція на зображенні"""
    if model_path is None:
        model_path = "runs/detect/powerlines_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не знайдено: {model_path}")
        print("Спочатку навчіть модель: python main.py train")
        return
    
    print(f"🔍 Детекція на зображенні: {image_path}")
    
    from detect import PowerlinesDetector
    
    detector = PowerlinesDetector(model_path)
    result = detector.detect_image(image_path, conf=conf)
    stats = detector.get_detection_stats(result)
    
    print(f"\n✅ Детекція завершена!")
    print(f"Знайдено об'єктів: {stats['total']}")
    print(f"  - Стовпів: {stats['towers']}")
    print(f"  - Кабелів: {stats['cables']}")
    
    for i, det in enumerate(stats['detections'], 1):
        print(f"  {i}. {det['class']}: {det['confidence']:.2%}")

def main():
    parser = argparse.ArgumentParser(
        description="Детектор ліній електропередач / Powerlines Detector"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")
    
    # Setup
    setup_parser = subparsers.add_parser("setup", help="Початкове налаштування проекту")
    
    # Prepare
    prepare_parser = subparsers.add_parser("prepare", help="Підготовка датасету")
    
    # Visualize
    visualize_parser = subparsers.add_parser("visualize", help="Візуалізація датасету")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Навчання моделі")
    train_parser.add_argument("--epochs", type=int, help="Кількість епох")
    train_parser.add_argument("--batch", type=int, help="Розмір батчу")
    train_parser.add_argument("--device", type=str, help="Пристрій (cpu/cuda)")
    
    # Detect
    detect_parser = subparsers.add_parser("detect", help="Детекція на зображенні")
    detect_parser.add_argument("image", help="Шлях до зображення")
    detect_parser.add_argument("--model", help="Шлях до моделі")
    detect_parser.add_argument("--conf", type=float, default=0.25, help="Поріг впевненості")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Виконання команди
    if args.command == "setup":
        setup_project()
    
    elif args.command == "prepare":
        if not check_dependencies():
            return
        prepare_dataset()
    
    elif args.command == "visualize":
        if not check_dependencies():
            return
        visualize_dataset()
    
    elif args.command == "train":
        if not check_dependencies():
            return
        train_model(
            epochs=args.epochs,
            batch=args.batch,
            device=args.device
        )
    
    elif args.command == "detect":
        if not check_dependencies():
            return
        detect_image(
            image_path=args.image,
            model_path=args.model,
            conf=args.conf
        )

if __name__ == "__main__":
    main()
