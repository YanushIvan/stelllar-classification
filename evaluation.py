# evaluation.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_class_distribution(y):
    # Подсчет количества элементов каждого класса
    class_distribution = pd.Series(y).value_counts().sort_index()

    # Данные для графика
    classes = ['GALAXY', 'QSO', 'STAR']  # Названия классов
    counts = class_distribution.values  # Количество элементов каждого класса

    # Цвета для каждого класса
    colors = ['skyblue', 'lightgreen', 'salmon']

    # Создание графика
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=colors)
    plt.title('Distribution of the Target Class', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 500, f'{int(height)}', ha='center', fontsize=12)

    # Добавление легенды
    legend_labels = ['GALAXY (skyblue)', 'QSO (lightgreen)', 'STAR (salmon)']
    plt.legend(bars, legend_labels, title='Class Colors', fontsize=12, title_fontsize=12)

    plt.show()
import matplotlib.pyplot as plt

def plot_plate_distribution(data):
    # Создание гистограммы
    plt.figure(figsize=(12, 6))
    plt.hist(data['plate'], bins=10, color='dodgerblue', edgecolor='black')

    # Добавление подписей
    plt.title('Distribution of the Plate Feature', fontsize=16)
    plt.xlabel('Plate', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Отображение графика
    plt.show()
