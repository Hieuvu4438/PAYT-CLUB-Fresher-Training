import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _get_neighbors(self, test_point):
        distances = []
        for i, train_point in enumerate(self.X_train):
            distance = self._euclidean_distance(test_point, train_point)
            distances.append((distance, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(self.k)]
        
        return neighbors
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for test_point in X:
            neighbors = self._get_neighbors(test_point)  
            prediction = Counter(neighbors).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []
        
        for test_point in X:
            neighbors = self._get_neighbors(test_point)
            counter = Counter(neighbors)
            total_neighbors = len(neighbors)
            prob_dict = {cls: count/total_neighbors for cls, count in counter.items()}
            probabilities.append(prob_dict)
        
        return probabilities


class KNNVisualizer:
    @staticmethod
    def plot_dataset(X, y, title="Dataset"):

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(classifier, X, y, resolution=0.01):

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = classifier.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.colorbar(scatter)
        plt.title(f'KNN Decision Boundary (k={classifier.k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    @staticmethod
    def compare_k_values(X_train, y_train, X_test, y_test, k_range=range(1, 21)):

        accuracies = []
        
        for k in k_range:
            knn = KNNClassifier(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        plt.figure(figsize=(10, 6))
        plt.plot(k_range, accuracies, marker='o', linewidth=2, markersize=8)
        plt.title('KNN Accuracy vs K Value')
        plt.xlabel('K Value')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)

        best_k = k_range[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7)
        plt.text(best_k, best_accuracy, f'Best K={best_k}\nAccuracy={best_accuracy:.3f}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        
        return best_k, best_accuracy


def demo_iris_dataset():

    print("=" * 60)
    print("DEMO: KNN với Iris Dataset")
    print("=" * 60)
 
    iris = load_iris()
    X, y = iris.data[:, :2], iris.target 
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
   
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    visualizer = KNNVisualizer()
  
    visualizer.plot_dataset(X, y, "Iris Dataset (2 features)")

    visualizer.plot_decision_boundary(knn, X, y)

    best_k, best_accuracy = visualizer.compare_k_values(X_train, y_train, X_test, y_test)
    print(f"\nBest K value: {best_k} with accuracy: {best_accuracy:.3f}")


def demo_synthetic_dataset():

    print("=" * 60)
    print("DEMO: KNN với Synthetic Dataset")
    print("=" * 60)
 
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    visualizer = KNNVisualizer()
   
    visualizer.plot_dataset(X, y, "Synthetic Dataset")
   
    best_k, best_accuracy = visualizer.compare_k_values(X_train, y_train, X_test, y_test)
   
    knn_best = KNNClassifier(k=best_k)
    knn_best.fit(X_train, y_train)
   
    visualizer.plot_decision_boundary(knn_best, X, y)
  
    y_pred = knn_best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Best K: {best_k}")
    print(f"Best Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    print("K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
    print("Chào mừng đến với mini project KNN!")
    print()

    demo_iris_dataset()
    
    print("\n" + "="*80 + "\n")
   
    demo_synthetic_dataset()
    
    print("\n" + "="*60)
    print("DEMO hoàn thành! Cảm ơn bạn đã sử dụng KNN Classifier!")
    print("="*60)
