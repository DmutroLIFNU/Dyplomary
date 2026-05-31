"""
Конфігурація для навчання детектора ліній електропередач
Configuration for powerlines detector training
"""

# ================== ОСНОВНІ ПАРАМЕТРИ / BASIC PARAMETERS ==================

# Модель / Model
MODEL = "yolov8n.pt"  # Опції: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (xlarge)

# Датасет / Dataset
DATA_CONFIG = "dataset/powerlines.yaml"

# Пристрій / Device
DEVICE = "cpu"  # "cpu" або "cuda" для GPU / "cpu" or "cuda" for GPU

# Розмір зображення / Image size
IMAGE_SIZE = 640  # 640, 1280 (більший розмір = краща точність, але повільніше)

# Розмір батчу / Batch size
BATCH_SIZE = 4  # Залежить від доступної пам'яті / Depends on available memory

# Кількість епох / Number of epochs
EPOCHS = 100

# ================== НАВЧАННЯ / TRAINING ====================

# Оптимізатор / Optimizer
OPTIMIZER = "auto"  # "SGD", "Adam", "AdamW", або "auto"
LEARNING_RATE_INITIAL = 0.01  # Початковий learning rate / Initial learning rate
LEARNING_RATE_FINAL = 0.01    # Фінальний learning rate / Final learning rate
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# Early stopping
PATIENCE = 50  # Кількість епох без покращення перед зупинкою

# Warmup
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# ================== АУГМЕНТАЦІЯ / AUGMENTATION ====================

# Колір / Color
HSV_HUE = 0.015        # Варіація відтінку / Hue variation (0-1)
HSV_SATURATION = 0.7   # Варіація насиченості / Saturation variation (0-1)
HSV_VALUE = 0.4        # Варіація яскравості / Value/brightness variation (0-1)

# Геометричні трансформації / Geometric transforms
DEGREES = 0.0          # Ротація / Rotation (degrees, ±deg)
TRANSLATE = 0.1        # Зсув / Translation (fraction)
SCALE = 0.5           # Масштабування / Scaling (gain)
SHEAR = 0.0           # Зсув / Shear (degrees)
PERSPECTIVE = 0.0     # Перспектива / Perspective (0-0.001)

# Flip
FLIP_UP_DOWN = 0.0    # Вертикальний фліп / Vertical flip probability
FLIP_LEFT_RIGHT = 0.5 # Горизонтальний фліп / Horizontal flip probability

# Mosaic і MixUp
MOSAIC = 1.0          # Mosaic аугментація (0-1, 1.0 = завжди)
MIXUP = 0.0           # MixUp аугментація (0-1)

# Copy-Paste
COPY_PASTE = 0.0      # Copy-paste аугментація (0-1)

# ================== ВТРАТИ / LOSSES ====================

BOX_LOSS_GAIN = 7.5   # Вага box loss / Box loss gain
CLASS_LOSS_GAIN = 0.5 # Вага class loss / Class loss gain
DFL_LOSS_GAIN = 1.5   # Вага DFL loss / DFL loss gain

# ================== ВАЛІДАЦІЯ / VALIDATION ====================

VALIDATION_INTERVAL = 1  # Валідація кожні N епох / Validate every N epochs
CONFIDENCE_THRESHOLD = 0.001  # Мінімальна впевненість для валідації
IOU_THRESHOLD = 0.6          # IoU поріг для валідації

# ================== ЗБЕРЕЖЕННЯ / SAVING ====================

SAVE_PERIOD = -1      # Зберігати чекпоінт кожні N епох (-1 = тільки best/last)
PROJECT_NAME = "runs/detect"
EXPERIMENT_NAME = "powerlines_detector"
EXIST_OK = True      # Перезаписувати існуючі результати

# ================== ІНФЕРЕНС / INFERENCE ====================

# Для детекції / For detection
CONF_THRESHOLD = 0.25  # Мінімальна впевненість детекції
IOU_NMS = 0.45        # IoU поріг для NMS
MAX_DETECTIONS = 300  # Максимальна кількість детекцій на зображення

# ================== КЛАСИ / CLASSES ====================

CLASS_NAMES = {
    0: "tower",  # Стовп
    1: "cable"   # Кабель
}

COLORS = {
    0: (255, 0, 0),      # Червоний для tower
    1: (0, 255, 255)     # Жовтий для cable
}

# ================== ДОДАТКОВІ ПАРАМЕТРИ / ADDITIONAL PARAMETERS ==================

# Multi-scale training
MULTI_SCALE = False  # Тренування на різних масштабах

# Workers
WORKERS = 8  # Кількість потоків для завантаження даних

# Cache
CACHE = False  # Кешувати зображення в RAM (True/False/"disk")

# Verbose
VERBOSE = True  # Виводити детальні логи

# Deterministic
DETERMINISTIC = True  # Детерміністичне навчання (для відтворюваності)

# ================== ПРОФІЛІ НАВЧАННЯ / TRAINING PROFILES ==================

TRAINING_PROFILES = {
    "fast": {
        "epochs": 50,
        "patience": 20,
        "image_size": 640,
        "batch_size": 8
    },
    "balanced": {
        "epochs": 100,
        "patience": 50,
        "image_size": 640,
        "batch_size": 4
    },
    "accurate": {
        "epochs": 200,
        "patience": 100,
        "image_size": 1280,
        "batch_size": 2
    }
}

# ================== СПЕЦІАЛЬНІ НАЛАШТУВАННЯ ДЛЯ ЛІНІЙ ЕЛЕКТРОПЕРЕДАЧ ==================
# SPECIAL SETTINGS FOR POWERLINES

# Кабелі часто довгі і тонкі, тому:
# Cables are often long and thin, so:
# - Не використовуємо ротацію (DEGREES = 0)
# - Використовуємо горизонтальний фліп (FLIP_LEFT_RIGHT = 0.5)
# - Збільшуємо вагу box loss для точнішої локалізації
# - Використовуємо mosaic для контексту

# Стовпи можуть бути різних розмірів залежно від відстані
# Towers can be different sizes depending on distance
# - Multi-scale training може допомогти
# - Більший image_size для дрібних об'єктів
