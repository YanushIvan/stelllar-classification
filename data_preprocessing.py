import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_and_preprocess_data():
    dataset_path = os.path.join('data', 'star_classification.csv')

    # Загрузка датасета
    data = pd.read_csv(dataset_path)

    X = data.drop('class', axis=1)  # Признаки
    y = data['class'] 

    # Преобразование строковых меток в числовые
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test