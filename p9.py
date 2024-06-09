import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Adversarial Training
def adversarial_training(model, X_train, y_train, epsilon=0.01, epochs=10):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        epoch_loss, correct_predictions = 0, 0
        for i in range(len(X_train)):
            x, y = X_train[i:i+1], y_train[i:i+1]
            with tf.GradientTape() as tape:
                preds = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            perturbations = [epsilon * tf.sign(grad) for grad in gradients]
            for var, pert in zip(model.trainable_variables, perturbations):
                var.assign_add(pert)
            epoch_loss += loss.numpy().mean()
            correct_predictions += np.argmax(preds) == np.argmax(y)
        history['loss'].append(epoch_loss / len(X_train))
        history['accuracy'].append(correct_predictions / len(X_train))
    return history

# Tangent Distance
def tangent_distance(x1, x2):
    return euclidean_distances([x1.ravel()], [x2.ravel()])[0][0]

# Tangent Propagation
def tangent_propagation(model, X_train, y_train, epochs=10):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        epoch_loss, correct_predictions = 0, 0
        for i in range(len(X_train)):
            x, y = X_train[i:i+1], y_train[i:i+1]
            with tf.GradientTape() as tape:
                preds = model(x)
                loss = tf.keras.losses.categorical_crossentropy(y, preds)
            gradients = tape.gradient(loss, model.trainable_variables)
            for var, grad in zip(model.trainable_variables, gradients):
                var.assign_add(-grad)
            epoch_loss += loss.numpy().mean()
            correct_predictions += np.argmax(preds) == np.argmax(y)
        history['loss'].append(epoch_loss / len(X_train))
        history['accuracy'].append(correct_predictions / len(X_train))
    return history

# Tangent Classifier
def tangent_classifier(X_train, y_train, X_test):
    predictions = [y_train[np.argmin([tangent_distance(test_sample, train_sample) for train_sample in X_train])]
                   for test_sample in X_test]
    return predictions

# Plot history
def plot_history(history, title):
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss')
    plt.legend()
    plt.show()

# Sample Data
X_train = np.random.rand(100, 784)
y_train = np.eye(10)[np.random.randint(0, 10, size=100)]
X_test = np.random.rand(20, 784)

# Model
model = Sequential([
    Dense(10, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax')
])

# Adversarial Training
adv_history = adversarial_training(model, X_train, y_train)
plot_history(adv_history, 'Adversarial Training')

# Tangent Propagation
tangent_history = tangent_propagation(model, X_train, y_train)
plot_history(tangent_history, 'Tangent Propagation')

# Tangent Classifier Predictions
predictions = tangent_classifier(X_train, y_train, X_test)
print(predictions)
