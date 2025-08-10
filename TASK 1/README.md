# 🤖 K-Nearest Neighbors (KNN) Classifier

Một mini project về thuật toán **K-Nearest Neighbors** được implement từ scratch bằng Python.

## 📖 Giới thiệu

K-Nearest Neighbors (KNN) là một thuật toán machine learning đơn giản và hiệu quả thuộc nhóm **lazy learning**. Thuật toán này dựa trên nguyên lý: "Những điểm gần nhau thường có cùng nhãn".

### 🔍 Cách hoạt động:
1. **Training**: Lưu trữ tất cả dữ liệu training
2. **Prediction**: 
   - Tính khoảng cách từ điểm cần dự đoán đến tất cả điểm training
   - Chọn k điểm gần nhất
   - Voting để quyết định nhãn (class xuất hiện nhiều nhất)

## 🚀 Tính năng

### ✨ KNNClassifier Class
- ✅ Implementation KNN từ scratch
- ✅ Tính khoảng cách Euclidean
- ✅ Thuật toán voting cho classification
- ✅ Predict probability cho từng class
- ✅ Dễ dàng tùy chỉnh giá trị k

### 📊 KNNVisualizer Class
- ✅ Visualization dataset 2D
- ✅ Vẽ decision boundary
- ✅ So sánh hiệu suất với các giá trị k khác nhau
- ✅ Tìm k tối ưu tự động

### 🎯 Demo Scripts
- ✅ Demo với Iris dataset (classic ML dataset)
- ✅ Demo với synthetic dataset
- ✅ Đánh giá và so sánh model

## 🛠️ Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- pip

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Hoặc cài đặt manual
```bash
pip install numpy matplotlib scikit-learn seaborn
```

## 🎮 Cách sử dụng

### Chạy demo đầy đủ
```bash
python knn_classifier.py
```

### Sử dụng trong code

#### 1. Basic Usage
```python
from knn_classifier import KNNClassifier

# Khởi tạo model
knn = KNNClassifier(k=5)

# Training
knn.fit(X_train, y_train)

# Prediction
predictions = knn.predict(X_test)

# Probability prediction
probabilities = knn.predict_proba(X_test)
```

#### 2. Visualization
```python
from knn_classifier import KNNVisualizer

visualizer = KNNVisualizer()

# Vẽ dataset
visualizer.plot_dataset(X, y, "My Dataset")

# Vẽ decision boundary
visualizer.plot_decision_boundary(knn, X, y)

# Tìm k tối ưu
best_k, best_accuracy = visualizer.compare_k_values(
    X_train, y_train, X_test, y_test
)
```

## 📊 Kết quả Demo

### 🌸 Iris Dataset
- **Dataset**: 150 samples, 3 classes (Setosa, Versicolor, Virginica)
- **Features**: Sử dụng 2 features đầu để visualization
- **Accuracy**: ~95-97% với k=5

### 🔬 Synthetic Dataset  
- **Dataset**: 300 samples, 2 classes
- **Features**: 2D synthetic data
- **Accuracy**: ~85-90% với k tối ưu

## 🧮 Thuật toán Chi tiết

### Distance Calculation
```python
def _euclidean_distance(self, point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))
```

### K-Neighbors Selection
```python
def _get_neighbors(self, test_point):
    distances = []
    for i, train_point in enumerate(self.X_train):
        distance = self._euclidean_distance(test_point, train_point)
        distances.append((distance, self.y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = [distances[i][1] for i in range(self.k)]
    return neighbors
```

### Voting Mechanism
```python
def predict(self, X):
    predictions = []
    for test_point in X:
        neighbors = self._get_neighbors(test_point)
        prediction = Counter(neighbors).most_common(1)[0][0]
        predictions.append(prediction)
    return np.array(predictions)
```

## 📈 Ưu điểm & Nhược điểm

### ✅ Ưu điểm
- **Đơn giản**: Dễ hiểu và implement
- **Không cần training**: Lazy learning algorithm
- **Versatile**: Có thể dùng cho cả classification và regression
- **Non-parametric**: Không giả định về phân phối dữ liệu

### ❌ Nhược điểm
- **Chậm**: O(n) cho mỗi prediction
- **Memory intensive**: Phải lưu tất cả training data
- **Sensitive to noise**: Bị ảnh hưởng bởi outliers
- **Curse of dimensionality**: Hiệu quả giảm với nhiều features

## 🔧 Tối ưu hóa

### Chọn K
- **K nhỏ**: Model phức tạp, có thể overfitting
- **K lớn**: Model đơn giản, có thể underfitting
- **Rule of thumb**: K = √n (n là số training samples)
- **Best practice**: Sử dụng cross-validation

### Distance Metrics
Hiện tại sử dụng Euclidean distance, có thể mở rộng:
- Manhattan distance
- Minkowski distance  
- Cosine similarity

## 🎯 Ứng dụng thực tế

1. **Recommendation Systems**: Netflix, Amazon
2. **Pattern Recognition**: Nhận dạng chữ viết tay
3. **Computer Vision**: Classification ảnh
4. **Text Mining**: Phân loại văn bản
5. **Medical Diagnosis**: Chẩn đoán y tế

## 🔮 Mở rộng

### Tính năng có thể thêm:
- [ ] Weighted KNN (khoảng cách gần có trọng số cao hơn)
- [ ] KD-Tree để tối ưu tìm kiếm
- [ ] Cross-validation tự động
- [ ] Nhiều distance metrics
- [ ] KNN Regression
- [ ] Parallel processing

### Datasets khác để test:
- [ ] Wine dataset
- [ ] Breast Cancer dataset
- [ ] Digits dataset
- [ ] Custom datasets

## 👨‍💻 Tác giả

**PAYT CLUB - FRESHER**  
Mini Project: K-Nearest Neighbors Implementation

## 📜 License

Dự án này được tạo ra cho mục đích học tập và nghiên cứu.

---

### 🎉 Happy Learning with KNN! 

*"In machine learning, the nearest neighbors are your best friends!"*
