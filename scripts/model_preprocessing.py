import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def preprocess_data():
    # Загружаем тренировочные данные
    train_data = pd.read_csv('data/train/train_data.csv')

    # Инициализируем StandardScaler
    scaler = StandardScaler()

    # Предполагаем, что 'Эффективная температура' - это целевая переменная
    X = train_data[['Средняя температура']]

    # Применяем масштабирование
    X_scaled = scaler.fit_transform(X)

    # Создаем новый DataFrame
    train_data_scaled = pd.DataFrame(data=X_scaled, columns=['Средняя температура'])
    train_data_scaled['Эффективная температура'] = train_data['Эффективная температура']

    # Сохраняем предобработанные данные
    train_data_scaled.to_csv('data/train/train_data_scaled.csv', index=False)
    print("Данные предобработаны и сохранены.")


if __name__ == "__main__":
    preprocess_data()
