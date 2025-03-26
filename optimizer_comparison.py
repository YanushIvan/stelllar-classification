from data_preprocessing import load_and_preprocess_data
from model import build_model, train_model
from tabulate import tabulate
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from evaluation import plot_class_distribution
from evaluation import plot_plate_distribution
# Установка seed для воспроизводимости
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)    
random.seed(seed)

# Настройка оптимизаторов
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Nadam

optimizers = {
    'adam': Adam(learning_rate=0.001),
    'sgd': SGD(learning_rate=0.01),
    'rmsprop': RMSprop(learning_rate=0.001),
    'adadelta': Adadelta(learning_rate=1.0),
    'adagrad': Adagrad(learning_rate=0.01),
    'nadam': Nadam(learning_rate=0.001)
}

def compare_optimizers(X_train, y_train, X_test, y_test, optimizers, load_weights=True):
    results = []

    for optimizer_name, optimizer in optimizers.items():
        print(f"Training with optimizer: {optimizer_name}")
        model = build_model(input_dim=X_train.shape[1])

        # Обучение модели с возможностью загрузки весов
        history = train_model(
            model, X_train, y_train,
            optimizer=optimizer,
            epochs=1,
            batch_size=64,
            validation_split=0.2,
            load_weights=load_weights,
            optimizer_name=optimizer_name
        )
        
        # Сохранение метрик
        training_loss = history.history['loss'][-1]
        training_accuracy = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_accuracy = history.history['val_accuracy'][-1]

        # Добавление результатов в список
        results.append([optimizer_name, training_loss, training_accuracy, val_loss, val_accuracy])

    # Вывод таблицы
    headers = ["Optimizers", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]
    print(tabulate(results, headers=headers, tablefmt="grid"))

# Загрузка и подготовка данных
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Сравнение оптимизаторов
compare_optimizers(X_train, y_train, X_test, y_test, optimizers, load_weights=True)  # Установите load_weights=True для загрузки весов

plot_class_distribution(y_train)
plot_class_distribution(y_test)   
data = pd.read_csv('data/star_classification.csv')
plot_plate_distribution(data)