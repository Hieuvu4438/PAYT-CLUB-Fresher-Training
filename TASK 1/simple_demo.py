import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import KNNClassifier, KNNVisualizer

def create_simple_dataset():
    class1_x = np.random.normal(2, 0.5, 20)
    class1_y = np.random.normal(2, 0.5, 20)
    class1 = np.column_stack((class1_x, class1_y))
    labels1 = np.zeros(20)  # Label 0
    
    class2_x = np.random.normal(6, 0.5, 20)
    class2_y = np.random.normal(6, 0.5, 20)
    class2 = np.column_stack((class2_x, class2_y))
    labels2 = np.ones(20)   # Label 1
   
    class3_x = np.random.normal(2, 0.5, 20)
    class3_y = np.random.normal(6, 0.5, 20)
    class3 = np.column_stack((class3_x, class3_y))
    labels3 = np.full(20, 2)  # Label 2
    
    X = np.vstack((class1, class2, class3))
    y = np.hstack((labels1, labels2, labels3))
    
    return X, y

def simple_knn_demo():
    print("🤖 SIMPLE KNN DEMO")
    print("=" * 50)
    
    X, y = create_simple_dataset()
    print(f"📊 Dataset: {len(X)} điểm, {len(np.unique(y))} classes")
    

    n_train = 45 
    indices = np.random.permutation(len(X))
    X_train, X_test = X[indices[:n_train]], X[indices[n_train:]]
    y_train, y_test = y[indices[:n_train]], y[indices[n_train:]]
    
    print(f"🎯 Training: {len(X_train)} điểm")
    print(f"🔍 Testing: {len(X_test)} điểm")

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print(f"✅ Accuracy: {accuracy:.2%}")

    visualizer = KNNVisualizer()
    
    plt.figure(figsize=(15, 5))
    

    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   label=f'Class {i}', alpha=0.7, s=50)
    plt.title('Original Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Training data
    plt.subplot(1, 3, 2)
    for i in range(3):
        mask = y_train == i
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[i], 
                   label=f'Class {i}', alpha=0.7, s=50, marker='o')
    
    # Vẽ test points
    for i in range(len(X_test)):
        predicted_class = int(y_pred[i])
        actual_class = int(y_test[i])
        marker = 's' if predicted_class == actual_class else 'x'
        size = 100 if predicted_class == actual_class else 150
        plt.scatter(X_test[i, 0], X_test[i, 1], c=colors[predicted_class], 
                   marker=marker, s=size, edgecolors='black', linewidth=2)
    
    plt.title('Training Data + Predictions\n(□ = Correct, ✗ = Wrong)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Decision boundary
    plt.subplot(1, 3, 3)
    
    # Tạo grid để vẽ decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # Vẽ training data
    for i in range(3):
        mask = y_train == i
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[i], 
                   label=f'Class {i}', alpha=0.8, s=50, edgecolors='black')
    
    plt.title('Decision Boundary (k=3)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Demo dự đoán cho một điểm cụ thể
    print("\n🎯 DEMO DỰ ĐOÁN CHO ĐIỂM MỚI")
    print("-" * 30)
    
    test_point = np.array([[4, 4]])  # Điểm ở giữa
    prediction = knn.predict(test_point)
    probabilities = knn.predict_proba(test_point)
    
    print(f"Điểm test: {test_point[0]}")
    print(f"Dự đoán: Class {prediction[0]}")
    print("Xác suất cho từng class:")
    for cls, prob in probabilities[0].items():
        print(f"  Class {cls}: {prob:.2%}")

def compare_k_values_demo():
    """
    Demo so sánh các giá trị k khác nhau
    """
    print("\n📊 SO SÁNH CÁC GIÁ TRỊ K")
    print("=" * 30)
    
    # Tạo dataset
    X, y = create_simple_dataset()
    
    # Chia dữ liệu
    n_train = 45
    indices = np.random.permutation(len(X))
    X_train, X_test = X[indices[:n_train]], X[indices[n_train:]]
    y_train, y_test = y[indices[:n_train]], y[indices[n_train:]]
    
    # Test các giá trị k
    k_values = [1, 3, 5, 7, 9, 11]
    accuracies = []
    
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        print(f"k={k}: Accuracy = {accuracy:.2%}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, markersize=8)
    plt.title('KNN Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Đánh dấu k tốt nhất
    best_k = k_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
    plt.text(best_k, best_accuracy, f'Best K={best_k}\nAccuracy={best_accuracy:.1%}', 
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.ylim(0, 1.05)
    plt.show()
    
    return best_k, best_accuracy

if __name__ == "__main__":
    print("🎯 WELCOME TO SIMPLE KNN DEMO!")
    print("Đây là demo đơn giản để hiểu KNN hoạt động như thế nào\n")
    
    # Set random seed để có kết quả reproducible
    np.random.seed(42)
    
    # Demo cơ bản
    simple_knn_demo()
    
    # Demo so sánh k values
    best_k, best_acc = compare_k_values_demo()
    
    print(f"\n🏆 KẾT LUẬN:")
    print(f"K tốt nhất: {best_k}")
    print(f"Accuracy cao nhất: {best_acc:.1%}")
    
    print("\n✨ Hãy thử chạy knn_classifier.py để xem demo đầy đủ!")
    print("🚀 Happy Learning!")
