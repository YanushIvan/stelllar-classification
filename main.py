from data_preprocessing import load_and_preprocess_data
from model import build_model, train_model

# Загрузка и подготовка данных
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Создание модели
model = build_model(input_dim=X_train.shape[1])

# Обучение модели
history = train_model(model, X_train, y_train, optimizer='adam', epochs=10, batch_size=64, validation_split=0.2)