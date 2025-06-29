# HƯỚNG DẪN SỬ DỤNG MÔ HÌNH CNN PHÂN LOẠI ĐỘNG VẬT

📋 MÔ TẢ DỰ ÁN:

- Xây dựng mô hình CNN từ đầu (from scratch) để phân loại 10 loại động vật
- Không sử dụng pre-trained models (ResNet, VGG, EfficientNet,...)
- Framework: TensorFlow/Keras
- Dữ liệu: 10 loại động vật (butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel)

🚀 CÁCH CHẠY:

1. Đảm bảo tất cả file .zip có trong thư mục animals/
2. Cài đặt dependencies: pip install tensorflow numpy matplotlib scikit-learn seaborn
3. Chạy: python run_training.py

🏗️ KIẾN TRÚC MÔ HÌNH:

- Mô hình CNN tự thiết kế với 5 khối Convolution
- Sử dụng Batch Normalization và Dropout để tránh overfitting
- Global Average Pooling thay vì Flatten
- Dense layers với regularization

📊 ĐẶC ĐIỂM:

- Input size: 224x224x3
- Data augmentation: rotation, shift, flip, zoom, brightness
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

📁 CẤU TRÚC FILE:

- imageClassification.py: Mã nguồn chính
- run_training.py: Script kiểm tra và chạy training
- animals/: Thư mục chứa dữ liệu (file .zip)

🎯 KẾT QUẢ:

- Model được lưu trong 'animal_classifier_final.h5'
- Biểu đồ training history
- Classification report và confusion matrix

💡 MÔ HÌNH CNN TỰ THIẾT KẾ:

1. Khối 1: Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout
2. Khối 2: Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout
3. Khối 3: Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout
4. Khối 4: Conv2D(256) + BatchNorm + Conv2D(256) + MaxPool + Dropout
5. Khối 5: Conv2D(512) + BatchNorm + Conv2D(512) + GlobalAvgPool
6. Classifier: Dense(1024) + Dense(512) + Dense(256) + Dense(num_classes)

🔧 TÍNH NĂNG NÂNG CAO:

- Hỗ trợ 2 mô hình: CNN cơ bản và CNN với Residual connections
- Tự động giải nén và tổ chức dữ liệu
- Data augmentation mạnh mẽ
- Visualization và evaluation chi tiết
