import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load and preprocess the digits dataset
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN classifier without RBM
knn = KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree')
knn.fit(X_train, Y_train)
print("KNN without using RBM:\n", classification_report(Y_test, knn.predict(X_test)))

# RBM + KNN pipeline
rbm = BernoulliRBM(n_components=625, learning_rate=0.00001, n_iter=10, random_state=42)
rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("KNN", knn)])
rbm_features_classifier.fit(X_train, Y_train)
print("KNN using RBM features:\n", classification_report(Y_test, rbm_features_classifier.predict(X_test)))

# Visualize the original and transformed data
X_transformed = rbm.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(X[0].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
axes[0].set_title('Original Image')
axes[1].imshow(X_transformed[0].reshape(25, 25), cmap=plt.cm.gray_r, interpolation='nearest')
axes[1].set_title('Transformed Image')
plt.show()
