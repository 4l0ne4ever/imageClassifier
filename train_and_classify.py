#!/usr/bin/env python3
"""
Script chạy mô hình CNN phân loại động vật từ đầu (from scratch)
Tác giả: AI Assistant
Mô tả: Xây dựng mô hình CNN tự thiết kế để phân loại 10 loại động vật
Gộp từ run_training.py và imageClassification.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import trực tiếp các thư viện cần thiết
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Thêm thư mục hiện tại vào path
sys.path.append('.')

# =========================
# 1. Kiểm tra dữ liệu đầu vào
# =========================
def check_data_structure():
    """Kiểm tra thư mục và file dữ liệu zip động vật"""
    animals_dir = './animals'
    if not os.path.exists(animals_dir):
        print(f"✗ Không tìm thấy thư mục {animals_dir}")
        return False
    expected_files = [
        'butterfly.zip', 'cat.zip', 'chicken.zip', 'cow.zip', 'dog.zip',
        'elephant.zip', 'horse.zip', 'sheep.zip', 'spider.zip', 'squirrel.zip'
    ]
    missing_files = []
    for file in expected_files:
        file_path = os.path.join(animals_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file}")
        else:
            missing_files.append(file)
            print(f"✗ {file}")
    if missing_files:
        print(f"\nThiếu {len(missing_files)} file zip trong thư mục animals/")
        return False
    return True

# =========================
# 2. Pipeline tạo và test mô hình
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5  # Chỉ chạy 5 epoch để test khả năng tạo mô hình
NUM_CLASSES = 10  
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

def extract_data(animals_dir='./animals', extract_path='./data'):
    """Giải nén các file zip động vật vào thư mục data"""
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    zip_files = [
        'butterfly.zip', 'cat.zip', 'chicken.zip', 'cow.zip', 'dog.zip', 
        'elephant.zip', 'horse.zip', 'sheep.zip', 'spider.zip', 'squirrel.zip'
    ]
    for zip_file in zip_files:
        zip_path = os.path.join(animals_dir, zip_file)
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Đã giải nén: {zip_file}")
        else:
            print(f"Không tìm thấy file: {zip_path}")

def organize_data(data_path='./data'):
    """Tổ chức lại dữ liệu thành train/validation cho từng lớp"""
    import shutil
    import random
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'validation')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if item in ['train', 'validation'] or not os.path.isdir(item_path):
            continue
        class_name = item.lower()
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        images = []
        for root, dirs, files in os.walk(item_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    images.append(os.path.join(root, file))
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        for i, img_path in enumerate(train_images):
            ext = os.path.splitext(img_path)[1]
            new_name = f"{class_name}_train_{i:04d}{ext}"
            shutil.copy2(img_path, os.path.join(train_class_dir, new_name))
        for i, img_path in enumerate(val_images):
            ext = os.path.splitext(img_path)[1]
            new_name = f"{class_name}_val_{i:04d}{ext}"
            shutil.copy2(img_path, os.path.join(val_class_dir, new_name))
        print(f"Lớp {class_name}: {len(train_images)} train, {len(val_images)} val")

def create_data_generators(data_path='./data'):
    """Tạo data generators với data augmentation cho train và chỉ rescale cho validation"""
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'validation')
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, validation_generator

def create_advanced_cnn_model(num_classes=NUM_CLASSES):
    """Tạo mô hình CNN nâng cao với Residual connections"""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = layers.Dropout(0.2)(x)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = layers.Dropout(0.3)(x)
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name='custom_residual_cnn')
    return model

def create_cnn_model(num_classes=NUM_CLASSES):
    """Tạo mô hình CNN cơ bản"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, train_generator, validation_generator):
    """Huấn luyện mô hình với callback dừng sớm và giảm learning rate"""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,  # giảm patience để test nhanh
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    return history

def plot_training_history(history):
    """Vẽ biểu đồ training/loss cho quá trình test mô hình"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# 3. Hàm main tổng hợp
# =========================
def main():
    print("=" * 60)
    print("KIỂM TRA HỆ THỐNG TRƯỚC KHI CHẠY MÔ HÌNH")
    print("=" * 60)
    print("\n1. Kiểm tra dữ liệu:")
    if not check_data_structure():
        return
    print("\n2. Bắt đầu test tạo mô hình...")
    print("=" * 60)
    try:
        print("=== Bắt đầu quá trình chuẩn bị dữ liệu và tạo mô hình ===")
        animals_dir = './animals'
        data_path = './data'
        extract_data(animals_dir, data_path)
        organize_data(data_path)
        print("\nTạo data generators...")
        train_gen, val_gen = create_data_generators(data_path)
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Classes: {list(train_gen.class_indices.keys())}")
        print(f"Số lớp: {len(train_gen.class_indices)}")
        print("\nTạo mô hình CNN nâng cao...")
        model = create_advanced_cnn_model(len(train_gen.class_indices))
        print(model.summary())
        print("\nHuấn luyện mô hình (5 epoch)...")
        history = train_model(model, train_gen, val_gen)
        print("\nBiểu đồ training/loss:")
        plot_training_history(history)
        print("\nTest hoàn tất!")
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình test: {e}")
        print("\nGợi ý khắc phục:")
        print("1. Kiểm tra lại các file zip trong thư mục animals/")
        print("2. Đảm bảo đủ dung lượng ổ cứng (ít nhất 2GB)")
        print("3. Kiểm tra RAM (khuyến nghị ít nhất 8GB)")
        print("4. Nếu dùng GPU, kiểm tra CUDA và cuDNN")

if __name__ == "__main__":
    main()
