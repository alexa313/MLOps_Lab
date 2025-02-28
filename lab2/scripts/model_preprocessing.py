import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def preprocess_data():
    # Загружаем данные
    data = pd.read_csv('data/train/train_data.csv')

    # Инициализируем StandardScaler
    scaler = StandardScaler()

    # Предполагаем, что 'Эффективная температура' - это целевая переменная
    X = data[['Средняя температура']]
    y = data['Эффективная температура']

    # Применяем масштабирование
    X_scaled = scaler.fit_transform(X)

    # Создаем новый DataFrame для предобработанных данных
    scaled_data = pd.DataFrame(data=X_scaled, columns=['Средняя температура'])
    scaled_data['Эффективная температура'] = y.reset_index(drop=True)

    # Разделяем данные на обучающую и тестовую выборки
    train_df, test_df = train_test_split(scaled_data, test_size=0.2, random_state=42)

    # Создаем необходимые директории, если их нет
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Сохраняем предобработанные и разделенные данные
    train_df.to_csv('data/train/train_data_scaled.csv', index=False)
    test_df.to_csv('data/test/test_data_scaled.csv', index=False)
    
    print("Данные предобработаны и сохранены.")
    
if __name__ == "__main__":
    preprocess_data()

