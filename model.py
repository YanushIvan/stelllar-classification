from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_model(input_dim):
    # Создание модели
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 класса: GALAXY, QSO, STAR

    return model

def train_model(model, X_train, y_train, optimizer='adam', epochs=1, batch_size=64, validation_split=0.2, load_weights=True, optimizer_name=None):
    if load_weights and optimizer_name:
        # Загрузка весов, если указано
        model.load_weights(f'{optimizer_name}.weights.h5')

    # Компиляция модели
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    if optimizer_name:
        # Сохранение весов после обучения
        model.save_weights(f'{optimizer_name}.weights.h5')

    return history